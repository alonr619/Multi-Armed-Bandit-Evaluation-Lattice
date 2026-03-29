import os
import random
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterator, TypeVar

T = TypeVar("T")

_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_TEXT_MARKERS = (
    "rate limit",
    "too many requests",
    "resource exhausted",
    "overloaded",
    "temporarily unavailable",
    "service unavailable",
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection error",
    "try again",
)

_DEFAULT_MAX_IN_FLIGHT = 2
_DEFAULT_MIN_INTERVAL_SECONDS = 0.0
_DEFAULT_MAX_RETRIES = 6
_DEFAULT_BACKOFF_BASE_SECONDS = 1.0
_DEFAULT_BACKOFF_MAX_SECONDS = 30.0
_DEFAULT_BACKOFF_JITTER = 0.2

_THROTTLES: dict[str, "_ProviderThrottle"] = {}
_THROTTLES_LOCK = threading.Lock()


def _provider_env_prefix(provider_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", provider_name).strip("_").upper()
    return normalized or "LLM"


def _read_int_env(name: str, default: int, min_value: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, value)


def _read_float_env(name: str, default: float, min_value: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(min_value, value)


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    if response is None:
        return None

    for attr in ("status_code", "status"):
        value = getattr(response, attr, None)
        if isinstance(value, int):
            return value
    return None


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None)
    if headers is None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        return None

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None
    retry_after = str(retry_after).strip()
    if not retry_after:
        return None

    try:
        return max(0.0, float(retry_after))
    except ValueError:
        pass

    try:
        retry_after_dt = parsedate_to_datetime(retry_after)
    except (TypeError, ValueError):
        return None

    if retry_after_dt.tzinfo is None:
        retry_after_dt = retry_after_dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max(0.0, (retry_after_dt - now).total_seconds())


def is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    status_code = _extract_status_code(exc)
    if status_code in _RETRYABLE_STATUS_CODES:
        return True

    error_text = str(exc).lower()
    return any(marker in error_text for marker in _RETRYABLE_TEXT_MARKERS)


@dataclass
class _ProviderThrottle:
    max_in_flight: int
    min_interval_seconds: float
    semaphore: threading.BoundedSemaphore = field(init=False)
    interval_lock: threading.Lock = field(default_factory=threading.Lock)
    next_allowed_time: float = 0.0

    def __post_init__(self) -> None:
        self.semaphore = threading.BoundedSemaphore(self.max_in_flight)

    def acquire(self) -> None:
        self.semaphore.acquire()
        self._wait_for_turn()

    def release(self) -> None:
        self.semaphore.release()

    def _wait_for_turn(self) -> None:
        if self.min_interval_seconds <= 0:
            return

        while True:
            with self.interval_lock:
                now = time.monotonic()
                if now >= self.next_allowed_time:
                    self.next_allowed_time = now + self.min_interval_seconds
                    return
                sleep_for = self.next_allowed_time - now
            time.sleep(sleep_for)


def _get_provider_throttle(provider_name: str) -> _ProviderThrottle:
    provider_key = provider_name.lower()
    with _THROTTLES_LOCK:
        existing = _THROTTLES.get(provider_key)
        if existing is not None:
            return existing

        prefix = _provider_env_prefix(provider_name)
        max_in_flight = _read_int_env(
            f"{prefix}_MAX_IN_FLIGHT",
            _read_int_env("LLM_MAX_IN_FLIGHT", _DEFAULT_MAX_IN_FLIGHT, min_value=1),
            min_value=1,
        )
        min_interval_seconds = _read_float_env(
            f"{prefix}_MIN_INTERVAL_SECONDS",
            _read_float_env(
                "LLM_MIN_INTERVAL_SECONDS",
                _DEFAULT_MIN_INTERVAL_SECONDS,
                min_value=0.0,
            ),
            min_value=0.0,
        )
        throttle = _ProviderThrottle(
            max_in_flight=max_in_flight,
            min_interval_seconds=min_interval_seconds,
        )
        _THROTTLES[provider_key] = throttle
        return throttle


@contextmanager
def provider_request_slot(provider_name: str) -> Iterator[None]:
    throttle = _get_provider_throttle(provider_name)
    throttle.acquire()
    try:
        yield
    finally:
        throttle.release()


def call_with_retry(
    fn: Callable[[], T],
    *,
    provider_name: str,
    model: str,
) -> T:
    prefix = _provider_env_prefix(provider_name)
    max_retries = _read_int_env(
        f"{prefix}_MAX_RETRIES",
        _read_int_env("LLM_MAX_RETRIES", _DEFAULT_MAX_RETRIES, min_value=0),
        min_value=0,
    )
    backoff_base = _read_float_env(
        f"{prefix}_BACKOFF_BASE_SECONDS",
        _read_float_env(
            "LLM_BACKOFF_BASE_SECONDS",
            _DEFAULT_BACKOFF_BASE_SECONDS,
            min_value=0.0,
        ),
        min_value=0.0,
    )
    backoff_max = _read_float_env(
        f"{prefix}_BACKOFF_MAX_SECONDS",
        _read_float_env(
            "LLM_BACKOFF_MAX_SECONDS",
            _DEFAULT_BACKOFF_MAX_SECONDS,
            min_value=0.0,
        ),
        min_value=0.0,
    )
    backoff_jitter = _read_float_env(
        f"{prefix}_BACKOFF_JITTER",
        _read_float_env("LLM_BACKOFF_JITTER", _DEFAULT_BACKOFF_JITTER, min_value=0.0),
        min_value=0.0,
    )

    attempt = 0
    while True:
        try:
            with provider_request_slot(provider_name):
                return fn()
        except Exception as exc:
            if attempt >= max_retries or not is_retryable_exception(exc):
                raise

            backoff_delay = min(backoff_max, backoff_base * (2 ** attempt))
            retry_after_delay = _extract_retry_after_seconds(exc) or 0.0
            sleep_seconds = max(backoff_delay, retry_after_delay)
            if backoff_jitter > 0:
                jitter_multiplier = random.uniform(
                    max(0.0, 1.0 - backoff_jitter),
                    1.0 + backoff_jitter,
                )
                sleep_seconds *= jitter_multiplier

            attempt += 1
            print(
                f"[retry][{provider_name}][{model}] attempt {attempt}/{max_retries} "
                f"in {sleep_seconds:.2f}s after {type(exc).__name__}: {exc}"
            )
            time.sleep(sleep_seconds)

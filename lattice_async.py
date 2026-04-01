import argparse
import asyncio
import concurrent.futures
import contextlib
import csv
import os
import re
import shlex
import statistics
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

import agents.anthropic as anthropic_module
from agents.gemini import Gemini
from agents.grok import Grok
from agents.openai import OpenAI
from agents.resilience import retry_log_sink
from config import NUM_PULLS
from conversation import conversation
from dotenv import dotenv_values
from util import get_summary_rows, total_expected_score, total_score


@dataclass(frozen=True)
class ReasoningProfile:
    openai_effort: str | None
    anthropic_effort: str | None
    anthropic_thinking_type: str
    gemini_effort: str | None


@dataclass(frozen=True)
class LatticeSpec:
    name: str
    models: tuple[str, ...]
    reasoning: ReasoningProfile
    pairs: tuple[tuple[str, str], ...] | None = None


@dataclass(frozen=True)
class MatchTask:
    good_model: str
    bad_model: str
    repeat_index: int


@dataclass(frozen=True)
class MatchResult:
    task: MatchTask
    pulls: list[tuple[int, float]]
    total_score: float
    expected_score: float
    elapsed_seconds: float
    match_log_path: Path


@dataclass(frozen=True)
class SettingsLoadReport:
    path: Path
    exists: bool
    file_values: dict[str, str]
    applied_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]


LATTICES: dict[str, LatticeSpec] = {
    "openai-none": LatticeSpec(
        name="openai-none",
        models=("gpt-4o-mini", "gpt-4.1", "gpt-5.1", "gpt-5.4"),
        reasoning=ReasoningProfile(
            openai_effort="none",
            anthropic_effort=None,
            anthropic_thinking_type="disabled",
            gemini_effort="none",
        ),
    ),
    "mixed-low": LatticeSpec(
        name="mixed-low",
        models=("gpt-5.4", "claude-opus-4.6", "gemini-3.1-pro-preview"),
        reasoning=ReasoningProfile(
            openai_effort="low",
            anthropic_effort="low",
            anthropic_thinking_type="adaptive",
            gemini_effort="low",
        ),
    ),
    "gpt-5.4-reasoning-cross": LatticeSpec(
        name="gpt-5.4-reasoning-cross",
        models=(
            "gpt-5.4-reasoning-none",
            "gpt-5.4-reasoning-low",
            "gpt-5.4-reasoning-medium",
            "gpt-5.4-reasoning-high",
        ),
        reasoning=ReasoningProfile(
            openai_effort="none",
            anthropic_effort=None,
            anthropic_thinking_type="disabled",
            gemini_effort="none",
        ),
        pairs=(
            ("gpt-5.4-reasoning-none", "gpt-5.4-reasoning-none"),
            ("gpt-5.4-reasoning-low", "gpt-5.4-reasoning-none"),
            ("gpt-5.4-reasoning-medium", "gpt-5.4-reasoning-none"),
            ("gpt-5.4-reasoning-high", "gpt-5.4-reasoning-none"),
            ("gpt-5.4-reasoning-none", "gpt-5.4-reasoning-low"),
            ("gpt-5.4-reasoning-none", "gpt-5.4-reasoning-medium"),
            ("gpt-5.4-reasoning-none", "gpt-5.4-reasoning-high"),
        ),
    ),
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(rows)


def _build_matrix_rows(
    title: str,
    models: tuple[str, ...],
    values: dict[tuple[str, str], float],
) -> list[list[object]]:
    rows = [[f"{title} (Good \\ Evil)"] + [f"Evil: {model}" for model in models]]
    for good_model in models:
        row_values: list[object] = []
        for bad_model in models:
            value = values.get((good_model, bad_model))
            row_values.append("" if value is None else value)
        rows.append(
            [f"Good: {good_model}"] + row_values
        )
    return rows


def _apply_reasoning_profile(profile: ReasoningProfile) -> None:
    OpenAI.reasoning_effort_override = profile.openai_effort
    Grok.reasoning_effort_override = profile.openai_effort
    Gemini.reasoning_effort_override = profile.gemini_effort
    anthropic_module.ANTHROPIC_REASONING_EFFORT = profile.anthropic_effort
    anthropic_module.ANTHROPIC_THINKING = {"type": profile.anthropic_thinking_type}


def _build_match_tasks(lattice: LatticeSpec, repeats: int) -> list[MatchTask]:
    if lattice.pairs is not None:
        model_pairs = lattice.pairs
    else:
        model_pairs = tuple(
            (good_model, bad_model)
            for good_model in lattice.models
            for bad_model in lattice.models
        )

    return [
        MatchTask(good_model=good_model, bad_model=bad_model, repeat_index=repeat_idx)
        for repeat_idx in range(1, repeats + 1)
        for good_model, bad_model in model_pairs
    ]


def _build_match_log_path(match_logs_dir: Path, task: MatchTask) -> Path:
    return match_logs_dir / (
        f"{_safe_name(task.good_model)}_vs_{_safe_name(task.bad_model)}"
        f"_r{task.repeat_index:03d}.log"
    )


def _load_settings_file(path: Path) -> SettingsLoadReport:
    if not path.exists():
        return SettingsLoadReport(
            path=path,
            exists=False,
            file_values={},
            applied_keys=(),
            skipped_keys=(),
        )

    values = dotenv_values(path)
    file_values: dict[str, str] = {}
    applied_keys: list[str] = []
    skipped_keys: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        file_values[key] = value
        if key in os.environ:
            skipped_keys.append(key)
            continue
        os.environ[key] = value
        applied_keys.append(key)

    return SettingsLoadReport(
        path=path,
        exists=True,
        file_values=file_values,
        applied_keys=tuple(applied_keys),
        skipped_keys=tuple(skipped_keys),
    )


class _TeeStream:
    def __init__(self, *streams: TextIO, lock: threading.Lock):
        self._streams = streams
        self._lock = lock

    def write(self, data: str) -> int:
        with self._lock:
            for stream in self._streams:
                stream.write(data)
                if data.endswith("\n"):
                    stream.flush()
        return len(data)

    def flush(self) -> None:
        with self._lock:
            for stream in self._streams:
                stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


@contextlib.contextmanager
def _tee_terminal_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        shared_lock = threading.Lock()
        sys.stdout = _TeeStream(original_stdout, log_file, lock=shared_lock)
        sys.stderr = _TeeStream(original_stderr, log_file, lock=shared_lock)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _timestamp_for_filename(now: datetime) -> str:
    return now.strftime("%Y%m%d_%H%M%S_%f")


def _build_log_path(output_root: Path, lattice_name: str, now: datetime) -> Path:
    return output_root / lattice_name / "logs" / f"{_timestamp_for_filename(now)}.log"


def _default_threadpool_workers() -> int:
    # Matches Python's default ThreadPoolExecutor sizing formula.
    return min(32, (os.cpu_count() or 1) + 4)


def _resolve_threadpool_workers(
    requested_workers: int | None,
    *,
    max_concurrent_games: int,
) -> int:
    if requested_workers is None:
        return max(max_concurrent_games, _default_threadpool_workers())
    return max(1, requested_workers)


def _print_run_header(
    *,
    run_started_at: datetime,
    log_path: Path,
    lattice: LatticeSpec,
    args: argparse.Namespace,
    settings_report: SettingsLoadReport,
    threadpool_workers: int,
) -> None:
    cli_command = shlex.join([Path(sys.executable).name, *sys.argv])

    print("=" * 100)
    print(f"Run timestamp: {run_started_at.isoformat()}")
    print(f"Lattice: {lattice.name}")
    print(f"CLI command: {cli_command}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Log file: {log_path.resolve()}")
    print("Arguments:")
    print(f"  lattice={args.lattice}")
    print(f"  num_pulls={args.num_pulls}")
    print(f"  repeats={args.repeats}")
    print(f"  max_concurrent_games={args.max_concurrent_games}")
    print(f"  threadpool_workers={threadpool_workers}")
    print(f"  output_dir={args.output_dir.resolve()}")
    print(f"  debug={args.debug}")
    print(f"  settings_file={args.settings_file}")
    print("Reasoning profile:")
    print(f"  openai_effort={lattice.reasoning.openai_effort}")
    print(f"  anthropic_effort={lattice.reasoning.anthropic_effort}")
    print(f"  anthropic_thinking_type={lattice.reasoning.anthropic_thinking_type}")
    print(f"  gemini_effort={lattice.reasoning.gemini_effort}")
    if lattice.pairs is not None:
        print(f"  custom_pairs={len(lattice.pairs)}")

    if not settings_report.exists:
        print(f"[settings] no settings file at {settings_report.path}; using environment/default values.")
    else:
        print(
            f"[settings] loaded {len(settings_report.applied_keys)} var(s) from {settings_report.path} "
            f"(skipped {len(settings_report.skipped_keys)} already-set env var(s))."
        )
        print("[settings] effective values:")
        for key in sorted(settings_report.file_values):
            value = os.environ.get(key, settings_report.file_values[key])
            source = "env" if key in settings_report.skipped_keys else "settings-file"
            print(f"  {key}={value}  (source={source})")

    print("=" * 100)


async def _run_single_match(
    task: MatchTask,
    *,
    game_slot: asyncio.Semaphore,
    match_logs_dir: Path,
    num_pulls: int,
    debug: bool,
) -> MatchResult:
    match_log_path = _build_match_log_path(match_logs_dir, task)

    def _run_conversation_with_match_log() -> list[tuple[int, float]]:
        match_log_path.parent.mkdir(parents=True, exist_ok=True)
        with match_log_path.open("w", encoding="utf-8", buffering=1) as match_log:
            def emit(message: str) -> None:
                text = message if message.endswith("\n") else f"{message}\n"
                match_log.write(text)

            emit(f"Match: {task.good_model} (good) vs {task.bad_model} (bad)")
            emit(f"Repeat index: {task.repeat_index}")
            emit(f"Debug mode: {debug}")
            emit("-" * 80)

            with retry_log_sink(emit):
                return conversation(
                    num_pulls,
                    task.good_model,
                    task.bad_model,
                    debug,
                    emit=emit,
                )

    async with game_slot:
        start = time.perf_counter()
        pulls = await asyncio.to_thread(_run_conversation_with_match_log)
        elapsed_seconds = time.perf_counter() - start
        return MatchResult(
            task=task,
            pulls=pulls,
            total_score=total_score(pulls),
            expected_score=total_expected_score(pulls),
            elapsed_seconds=elapsed_seconds,
            match_log_path=match_log_path,
        )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _stdev(values: list[float]) -> float:
    return round(statistics.pstdev(values), 3) if len(values) >= 2 else 0.0


async def _run_lattice(
    lattice: LatticeSpec,
    *,
    num_pulls: int,
    repeats: int,
    max_concurrent_games: int,
    output_root: Path,
    debug: bool,
) -> None:
    _apply_reasoning_profile(lattice.reasoning)
    tasks = _build_match_tasks(lattice, repeats)
    lattice_dir = output_root / lattice.name
    matches_dir = lattice_dir / "matches"
    match_logs_dir = lattice_dir / "match_logs"

    game_slot = asyncio.Semaphore(max_concurrent_games)
    async def _run_and_capture(task: MatchTask) -> tuple[MatchTask, MatchResult | None, Exception | None]:
        try:
            result = await _run_single_match(
                task,
                game_slot=game_slot,
                match_logs_dir=match_logs_dir,
                num_pulls=num_pulls,
                debug=debug,
            )
            return task, result, None
        except Exception as exc:
            return task, None, exc

    running = [asyncio.create_task(_run_and_capture(task)) for task in tasks]

    successful_results: list[MatchResult] = []
    failures: list[tuple[MatchTask, Exception]] = []
    total = len(running)
    completed = 0

    for job in asyncio.as_completed(running):
        task, result, exc = await job
        if exc is None and result is not None:
            successful_results.append(result)

            summary_rows = [["Metric", "Value"]] + [list(row) for row in get_summary_rows(result.pulls)]
            match_path = matches_dir / (
                f"{_safe_name(result.task.good_model)}_vs_{_safe_name(result.task.bad_model)}"
                f"_r{result.task.repeat_index:03d}.csv"
            )
            _write_csv(match_path, summary_rows)

            completed += 1
            print(
                f"[{lattice.name}] {completed}/{total} "
                f"{result.task.good_model} vs {result.task.bad_model} "
                f"(run {result.task.repeat_index}) "
                f"total={result.total_score:.3f} expected={result.expected_score:.3f} "
                f"time={result.elapsed_seconds:.2f}s "
                f"log={result.match_log_path}"
            )
        else:
            completed += 1
            failures.append((task, exc if exc is not None else RuntimeError("unknown error")))
            print(
                f"[{lattice.name}] {completed}/{total} FAILED "
                f"{task.good_model} vs {task.bad_model} "
                f"(run {task.repeat_index}): {exc}"
            )

    if not successful_results:
        raise RuntimeError(f"No successful matches were produced for lattice '{lattice.name}'.")

    actual_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    expected_values: dict[tuple[str, str], list[float]] = defaultdict(list)

    run_rows: list[list[object]] = [
        [
            "good_model",
            "bad_model",
            "repeat_index",
            "actual_score",
            "expected_score",
            "elapsed_seconds",
            "match_log_path",
        ]
    ]
    for result in successful_results:
        pair = (result.task.good_model, result.task.bad_model)
        actual_values[pair].append(result.total_score)
        expected_values[pair].append(result.expected_score)
        run_rows.append(
            [
                result.task.good_model,
                result.task.bad_model,
                result.task.repeat_index,
                result.total_score,
                result.expected_score,
                round(result.elapsed_seconds, 3),
                str(result.match_log_path.resolve()),
            ]
        )

    _write_csv(lattice_dir / "runs.csv", run_rows)

    if failures:
        failure_rows = [["good_model", "bad_model", "repeat_index", "match_log_path", "error"]]
        for task, exc in failures:
            failure_rows.append(
                [
                    task.good_model,
                    task.bad_model,
                    task.repeat_index,
                    str(_build_match_log_path(match_logs_dir, task).resolve()),
                    str(exc),
                ]
            )
        _write_csv(lattice_dir / "failures.csv", failure_rows)

    actual_mean = {pair: _mean(values) for pair, values in actual_values.items()}
    expected_mean = {pair: _mean(values) for pair, values in expected_values.items()}
    actual_stdev = {pair: _stdev(values) for pair, values in actual_values.items()}
    expected_stdev = {pair: _stdev(values) for pair, values in expected_values.items()}

    _write_csv(
        lattice_dir / "actual_scores_mean.csv",
        _build_matrix_rows("Actual score mean", lattice.models, actual_mean),
    )
    _write_csv(
        lattice_dir / "expected_scores_mean.csv",
        _build_matrix_rows("Expected score mean", lattice.models, expected_mean),
    )
    _write_csv(
        lattice_dir / "actual_scores_stdev.csv",
        _build_matrix_rows("Actual score stdev", lattice.models, actual_stdev),
    )
    _write_csv(
        lattice_dir / "expected_scores_stdev.csv",
        _build_matrix_rows("Expected score stdev", lattice.models, expected_stdev),
    )

    print(
        f"[{lattice.name}] completed with {len(successful_results)} successes and "
        f"{len(failures)} failures. Outputs: {lattice_dir}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run lattice evaluations concurrently with retry/backoff at the provider request layer. "
            "Tune provider limits with env vars like OPENAI_MAX_IN_FLIGHT, ANTHROPIC_MAX_IN_FLIGHT, "
            "GEMINI_MAX_IN_FLIGHT, and corresponding *_MIN_INTERVAL_SECONDS / *_MAX_RETRIES."
        )
    )
    parser.add_argument(
        "--lattice",
        type=str,
        choices=["openai-none", "mixed-low", "gpt-5.4-reasoning-cross", "both"],
        default="both",
        help="Which preset lattice to run.",
    )
    parser.add_argument("--num-pulls", type=int, default=NUM_PULLS, help="Pulls per game.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many repeated games to run per (good, evil) pair.",
    )
    parser.add_argument(
        "--max-concurrent-games",
        type=int,
        default=4,
        help="Maximum games run in parallel. Games remain turn-sequential internally.",
    )
    parser.add_argument(
        "--threadpool-workers",
        type=int,
        default=None,
        help=(
            "Worker threads used by asyncio.to_thread for game execution. "
            "Default: max(max_concurrent_games, Python default threadpool size)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/async_lattices"),
        help="Where to write lattice outputs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable verbose per-turn game logging (can be very noisy with concurrency).",
    )
    parser.add_argument(
        "--no-debug",
        action="store_false",
        dest="debug",
        help=(
            "Disable per-turn conversation logging. "
            "By default, debug logging is enabled and saved into run log files."
        ),
    )
    parser.add_argument(
        "--settings-file",
        type=Path,
        default=Path("lattice_async.settings.txt"),
        help=(
            "Path to .env-style settings file. If present, values are loaded before running. "
            "Existing environment variables take precedence over file values."
        ),
    )
    return parser.parse_args()


async def _async_main() -> None:
    args = _parse_args()
    settings_report = _load_settings_file(args.settings_file)
    threadpool_workers = _resolve_threadpool_workers(
        args.threadpool_workers,
        max_concurrent_games=args.max_concurrent_games,
    )

    selected = (
        [LATTICES["openai-none"], LATTICES["mixed-low"]]
        if args.lattice == "both"
        else [LATTICES[args.lattice]]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=threadpool_workers) as pool:
        loop.set_default_executor(pool)
        for lattice in selected:
            run_started_at = datetime.now()
            log_path = _build_log_path(args.output_dir, lattice.name, run_started_at)
            with _tee_terminal_output(log_path):
                try:
                    _print_run_header(
                        run_started_at=run_started_at,
                        log_path=log_path,
                        lattice=lattice,
                        args=args,
                        settings_report=settings_report,
                        threadpool_workers=threadpool_workers,
                    )
                    print(
                        f"[{lattice.name}] starting: {len(lattice.models)} models, "
                        f"{args.repeats} repeat(s), {args.num_pulls} pulls, "
                        f"max_concurrent_games={args.max_concurrent_games}, "
                        f"threadpool_workers={threadpool_workers}"
                    )
                    start = time.perf_counter()
                    await _run_lattice(
                        lattice,
                        num_pulls=args.num_pulls,
                        repeats=args.repeats,
                        max_concurrent_games=args.max_concurrent_games,
                        output_root=args.output_dir,
                        debug=args.debug,
                    )
                    elapsed = time.perf_counter() - start
                    print(f"[{lattice.name}] elapsed {elapsed:.2f}s")
                except Exception:
                    print(f"[{lattice.name}] unhandled exception:")
                    traceback.print_exc()
                    raise
                finally:
                    print(f"[{lattice.name}] log saved: {log_path.resolve()}")


if __name__ == "__main__":
    try:
        asyncio.run(_async_main())
    except Exception:
        raise SystemExit(1)

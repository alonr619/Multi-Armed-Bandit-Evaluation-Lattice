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
from agents.main import call_good_agent
from agents.openai import OpenAI
from agents.resilience import retry_log_sink
from bandit import n_armed_bandit
from config import NUM_PULLS
from dotenv import dotenv_values
from prompts import get_good_solo_prompt
from util import get_summary, get_summary_rows, total_expected_score, total_score


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


@dataclass(frozen=True)
class SoloTask:
    model: str
    repeat_index: int


@dataclass(frozen=True)
class SoloResult:
    task: SoloTask
    pulls: list[tuple[int, float]]
    total_score: float
    expected_score: float
    elapsed_seconds: float
    game_log_path: Path


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
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(rows)


def _apply_reasoning_profile(profile: ReasoningProfile) -> None:
    OpenAI.reasoning_effort_override = profile.openai_effort
    Grok.reasoning_effort_override = profile.openai_effort
    Gemini.reasoning_effort_override = profile.gemini_effort
    anthropic_module.ANTHROPIC_REASONING_EFFORT = profile.anthropic_effort
    anthropic_module.ANTHROPIC_THINKING = {"type": profile.anthropic_thinking_type}


def _build_tasks(models: tuple[str, ...], repeats: int) -> list[SoloTask]:
    return [
        SoloTask(model=model, repeat_index=repeat_idx)
        for repeat_idx in range(1, repeats + 1)
        for model in models
    ]


def _build_game_log_path(game_logs_dir: Path, task: SoloTask) -> Path:
    return game_logs_dir / f"{_safe_name(task.model)}_r{task.repeat_index:03d}.log"


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


def solo_conversation(
    num_pulls: int,
    model_id: str,
    debug: bool = False,
    emit=None,
) -> list[tuple[int, float]]:
    all_results: list[tuple[int, float]] = []
    good_history_turns: list[dict] = []
    bad_messages: list[str] = []
    cache_discount_warnings_shown: set[str] = set()
    log = emit if emit is not None else print
    solo_prompt = get_good_solo_prompt(num_pulls)

    for current_pull in range(num_pulls):
        response = call_good_agent(
            model=model_id,
            current_turn=current_pull + 1,
            past_results=all_results,
            bad_messages=bad_messages,
            good_history_turns=good_history_turns,
            num_pulls=num_pulls,
            prompt_override=solo_prompt,
            include_bad_message=False,
        )

        cache_note = response.get("cache_discount_note")
        warning_key = f"{model_id}:{cache_note}"
        if cache_note and warning_key not in cache_discount_warnings_shown:
            log(f"[cache-warning][{model_id}] {cache_note}")
            cache_discount_warnings_shown.add(warning_key)

        if debug:
            log(f"Model ({model_id}): {response['llm_response']}\n")
            if response.get("usage"):
                log(f"Usage ({model_id}): {response['usage']}")

        if response.get("history_turn"):
            good_history_turns.append(response["history_turn"])

        if response["arm_pulled"] is not None:
            arm = int(response["arm_pulled"])
            result = n_armed_bandit(arm)
            all_results.append((arm, result))
            if debug:
                log(f"Pull {current_pull + 1}: arm {arm} gave {result} points")

    if debug:
        log(get_summary(all_results, num_pulls))
    return all_results


async def _run_single_game(
    task: SoloTask,
    *,
    game_slot: asyncio.Semaphore,
    game_logs_dir: Path,
    num_pulls: int,
    debug: bool,
) -> SoloResult:
    game_log_path = _build_game_log_path(game_logs_dir, task)

    def _run_with_game_log() -> list[tuple[int, float]]:
        game_log_path.parent.mkdir(parents=True, exist_ok=True)
        with game_log_path.open("w", encoding="utf-8", buffering=1) as game_log:
            def emit(message: str) -> None:
                text = message if message.endswith("\n") else f"{message}\n"
                game_log.write(text)

            emit(f"Model: {task.model}")
            emit(f"Repeat index: {task.repeat_index}")
            emit(f"Debug mode: {debug}")
            emit("-" * 80)

            with retry_log_sink(emit):
                return solo_conversation(
                    num_pulls=num_pulls,
                    model_id=task.model,
                    debug=debug,
                    emit=emit,
                )

    async with game_slot:
        start = time.perf_counter()
        pulls = await asyncio.to_thread(_run_with_game_log)
        elapsed_seconds = time.perf_counter() - start
        return SoloResult(
            task=task,
            pulls=pulls,
            total_score=total_score(pulls),
            expected_score=total_expected_score(pulls),
            elapsed_seconds=elapsed_seconds,
            game_log_path=game_log_path,
        )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _stdev(values: list[float]) -> float:
    return round(statistics.pstdev(values), 3) if len(values) >= 2 else 0.0


async def _run_solo_lattice(
    lattice: LatticeSpec,
    *,
    num_pulls: int,
    repeats: int,
    max_concurrent_games: int,
    output_root: Path,
    debug: bool,
) -> None:
    _apply_reasoning_profile(lattice.reasoning)
    tasks = _build_tasks(lattice.models, repeats)
    lattice_dir = output_root / lattice.name
    games_dir = lattice_dir / "games"
    game_logs_dir = lattice_dir / "game_logs"

    game_slot = asyncio.Semaphore(max_concurrent_games)

    async def _run_and_capture(task: SoloTask) -> tuple[SoloTask, SoloResult | None, Exception | None]:
        try:
            result = await _run_single_game(
                task,
                game_slot=game_slot,
                game_logs_dir=game_logs_dir,
                num_pulls=num_pulls,
                debug=debug,
            )
            return task, result, None
        except Exception as exc:
            return task, None, exc

    running = [asyncio.create_task(_run_and_capture(task)) for task in tasks]

    successful_results: list[SoloResult] = []
    failures: list[tuple[SoloTask, Exception]] = []
    total = len(running)
    completed = 0

    for job in asyncio.as_completed(running):
        task, result, exc = await job
        if exc is None and result is not None:
            successful_results.append(result)
            summary_rows = [["Metric", "Value"]] + [list(row) for row in get_summary_rows(result.pulls)]
            game_path = games_dir / f"{_safe_name(result.task.model)}_r{result.task.repeat_index:03d}.csv"
            _write_csv(game_path, summary_rows)

            completed += 1
            print(
                f"[{lattice.name}] {completed}/{total} "
                f"{result.task.model} (run {result.task.repeat_index}) "
                f"total={result.total_score:.3f} expected={result.expected_score:.3f} "
                f"time={result.elapsed_seconds:.2f}s "
                f"log={result.game_log_path}"
            )
        else:
            completed += 1
            failures.append((task, exc if exc is not None else RuntimeError("unknown error")))
            print(
                f"[{lattice.name}] {completed}/{total} FAILED "
                f"{task.model} (run {task.repeat_index}): {exc}"
            )

    if not successful_results:
        raise RuntimeError(f"No successful games were produced for lattice '{lattice.name}'.")

    run_rows: list[list[object]] = [
        [
            "model",
            "repeat_index",
            "actual_score",
            "expected_score",
            "elapsed_seconds",
            "game_log_path",
        ]
    ]
    actual_values: dict[str, list[float]] = defaultdict(list)
    expected_values: dict[str, list[float]] = defaultdict(list)
    for result in successful_results:
        actual_values[result.task.model].append(result.total_score)
        expected_values[result.task.model].append(result.expected_score)
        run_rows.append(
            [
                result.task.model,
                result.task.repeat_index,
                result.total_score,
                result.expected_score,
                round(result.elapsed_seconds, 3),
                str(result.game_log_path.resolve()),
            ]
        )

    _write_csv(lattice_dir / "runs.csv", run_rows)

    if failures:
        failure_rows = [["model", "repeat_index", "game_log_path", "error"]]
        for task, exc in failures:
            failure_rows.append(
                [
                    task.model,
                    task.repeat_index,
                    str(_build_game_log_path(game_logs_dir, task).resolve()),
                    str(exc),
                ]
            )
        _write_csv(lattice_dir / "failures.csv", failure_rows)

    mean_rows = [["model", "actual_score_mean", "expected_score_mean", "n"]]
    stdev_rows = [["model", "actual_score_stdev", "expected_score_stdev", "n"]]
    for model in lattice.models:
        model_actual = actual_values[model]
        model_expected = expected_values[model]
        mean_rows.append([model, _mean(model_actual), _mean(model_expected), len(model_actual)])
        stdev_rows.append([model, _stdev(model_actual), _stdev(model_expected), len(model_actual)])

    _write_csv(lattice_dir / "model_scores_mean.csv", mean_rows)
    _write_csv(lattice_dir / "model_scores_stdev.csv", stdev_rows)

    print(
        f"[{lattice.name}] completed with {len(successful_results)} successes and "
        f"{len(failures)} failures. Outputs: {lattice_dir}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run solo lattice evaluations concurrently with retry/backoff at the provider request layer. "
            "Tune provider limits with env vars like OPENAI_MAX_IN_FLIGHT, ANTHROPIC_MAX_IN_FLIGHT, "
            "GEMINI_MAX_IN_FLIGHT, and corresponding *_MIN_INTERVAL_SECONDS / *_MAX_RETRIES."
        )
    )
    parser.add_argument(
        "--lattice",
        type=str,
        choices=["openai-none", "mixed-low", "both"],
        default="both",
        help="Which preset model set to run.",
    )
    parser.add_argument("--num-pulls", type=int, default=NUM_PULLS, help="Pulls per game.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many solo games to run per model.",
    )
    parser.add_argument(
        "--max-concurrent-games",
        type=int,
        default=4,
        help="Maximum solo games run in parallel.",
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
        default=Path("results/async_solo"),
        help="Where to write outputs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable verbose per-turn logging.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_false",
        dest="debug",
        help="Disable per-turn logging.",
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
                    await _run_solo_lattice(
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

import argparse
import asyncio
import csv
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import agents.anthropic as anthropic_module
from agents.gemini import Gemini
from agents.grok import Grok
from agents.openai import OpenAI
from config import NUM_PULLS
from conversation import conversation
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


def _build_matrix_rows(
    title: str,
    models: tuple[str, ...],
    values: dict[tuple[str, str], float],
) -> list[list[object]]:
    rows = [[f"{title} (Good \\ Evil)"] + [f"Evil: {model}" for model in models]]
    for good_model in models:
        rows.append(
            [f"Good: {good_model}"] + [values[(good_model, bad_model)] for bad_model in models]
        )
    return rows


def _apply_reasoning_profile(profile: ReasoningProfile) -> None:
    OpenAI.reasoning_effort_override = profile.openai_effort
    Grok.reasoning_effort_override = profile.openai_effort
    Gemini.reasoning_effort_override = profile.gemini_effort
    anthropic_module.ANTHROPIC_REASONING_EFFORT = profile.anthropic_effort
    anthropic_module.ANTHROPIC_THINKING = {"type": profile.anthropic_thinking_type}


def _build_match_tasks(models: tuple[str, ...], repeats: int) -> list[MatchTask]:
    return [
        MatchTask(good_model=good_model, bad_model=bad_model, repeat_index=repeat_idx)
        for repeat_idx in range(1, repeats + 1)
        for good_model in models
        for bad_model in models
    ]


async def _run_single_match(
    task: MatchTask,
    *,
    game_slot: asyncio.Semaphore,
    num_pulls: int,
    debug: bool,
) -> MatchResult:
    async with game_slot:
        start = time.perf_counter()
        pulls = await asyncio.to_thread(
            conversation,
            num_pulls,
            task.good_model,
            task.bad_model,
            debug,
        )
        elapsed_seconds = time.perf_counter() - start
        return MatchResult(
            task=task,
            pulls=pulls,
            total_score=total_score(pulls),
            expected_score=total_expected_score(pulls),
            elapsed_seconds=elapsed_seconds,
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
    tasks = _build_match_tasks(lattice.models, repeats)
    lattice_dir = output_root / lattice.name
    matches_dir = lattice_dir / "matches"

    game_slot = asyncio.Semaphore(max_concurrent_games)
    async def _run_and_capture(task: MatchTask) -> tuple[MatchTask, MatchResult | None, Exception | None]:
        try:
            result = await _run_single_match(
                task,
                game_slot=game_slot,
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
                f"time={result.elapsed_seconds:.2f}s"
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
        ["good_model", "bad_model", "repeat_index", "actual_score", "expected_score", "elapsed_seconds"]
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
            ]
        )

    _write_csv(lattice_dir / "runs.csv", run_rows)

    if failures:
        failure_rows = [["good_model", "bad_model", "repeat_index", "error"]]
        for task, exc in failures:
            failure_rows.append([task.good_model, task.bad_model, task.repeat_index, str(exc)])
        _write_csv(lattice_dir / "failures.csv", failure_rows)

    actual_mean = {(g, b): _mean(actual_values[(g, b)]) for g in lattice.models for b in lattice.models}
    expected_mean = {
        (g, b): _mean(expected_values[(g, b)]) for g in lattice.models for b in lattice.models
    }
    actual_stdev = {
        (g, b): _stdev(actual_values[(g, b)]) for g in lattice.models for b in lattice.models
    }
    expected_stdev = {
        (g, b): _stdev(expected_values[(g, b)]) for g in lattice.models for b in lattice.models
    }

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
        choices=["openai-none", "mixed-low", "both"],
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
        "--output-dir",
        type=Path,
        default=Path("results/async_lattices"),
        help="Where to write lattice outputs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose per-turn game logging (can be very noisy with concurrency).",
    )
    return parser.parse_args()


async def _async_main() -> None:
    args = _parse_args()

    selected = (
        [LATTICES["openai-none"], LATTICES["mixed-low"]]
        if args.lattice == "both"
        else [LATTICES[args.lattice]]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for lattice in selected:
        print(
            f"[{lattice.name}] starting: {len(lattice.models)} models, "
            f"{args.repeats} repeat(s), {args.num_pulls} pulls, "
            f"max_concurrent_games={args.max_concurrent_games}"
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


if __name__ == "__main__":
    asyncio.run(_async_main())

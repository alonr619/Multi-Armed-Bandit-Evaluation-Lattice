import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PULL_RE = re.compile(r"Pull (\d+): arm (\d+) gave ([0-9]+(?:\.[0-9]+)?) points")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ARMS
from util import expected_score

OUTPUT_DIR = ROOT / "results" / "analysis_quick"

SOLO_LOG_DIRS = [
    ROOT / "results" / "async_solo_50x30" / "openai-none" / "game_logs",
    ROOT / "results" / "async_solo_50x30_extra" / "openai-none" / "game_logs",
    ROOT / "results" / "async_solo_topup_20260331_014640" / "openai-none" / "game_logs",
]

MATCH_LOG_DIRS = sorted(
    [
        path
        for path in (ROOT / "results").glob("async_lattices_50x30_*/openai-none/match_logs")
        if path.is_dir()
    ]
)

ABANDONMENT_WINDOW = 3
FINAL_WINDOW = 10


@dataclass(frozen=True)
class RunRecord:
    good_model: str
    bad_model: str
    source: str
    run_id: str
    rounds: tuple[int, ...]
    arms: tuple[int, ...]
    rewards: tuple[float, ...]


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _optimal_arm() -> int:
    return max(range(len(ARMS)), key=expected_score)


OPTIMAL_ARM = _optimal_arm()
SAFE_ARM = min(range(len(ARMS)), key=expected_score)


def parse_log(path: Path, good_model: str, bad_model: str, source: str) -> RunRecord | None:
    rounds: list[int] = []
    arms: list[int] = []
    rewards: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = _strip_ansi(raw_line.strip())
            match = PULL_RE.search(line)
            if not match:
                continue
            rounds.append(int(match.group(1)))
            arms.append(int(match.group(2)))
            rewards.append(float(match.group(3)))

    if not rounds:
        return None

    return RunRecord(
        good_model=good_model,
        bad_model=bad_model,
        source=source,
        run_id=f"{source}:{path.name}",
        rounds=tuple(rounds),
        arms=tuple(arms),
        rewards=tuple(rewards),
    )


def iter_solo_runs() -> list[RunRecord]:
    runs: list[RunRecord] = []
    for log_dir in SOLO_LOG_DIRS:
        for path in sorted(log_dir.glob("*.log")):
            model = path.stem.rsplit("_r", 1)[0]
            parsed = parse_log(path, good_model=model, bad_model="no_evil_model", source=log_dir.parent.parent.name)
            if parsed is not None:
                runs.append(parsed)
    return runs


def iter_match_runs() -> list[RunRecord]:
    runs: list[RunRecord] = []
    for log_dir in MATCH_LOG_DIRS:
        source = log_dir.parent.parent.name
        for path in sorted(log_dir.glob("*.log")):
            stem = path.stem
            good_model, bad_with_repeat = stem.split("_vs_", 1)
            bad_model = bad_with_repeat.rsplit("_r", 1)[0]
            parsed = parse_log(path, good_model=good_model, bad_model=bad_model, source=source)
            if parsed is not None:
                runs.append(parsed)
    return runs


def build_run_frame(runs: list[RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in runs:
        n_rounds = len(run.rounds)
        discovered = OPTIMAL_ARM in run.arms
        discovery_round = (run.arms.index(OPTIMAL_ARM) + 1) if discovered else None
        last_optimal_round = (
            max(idx + 1 for idx, arm in enumerate(run.arms) if arm == OPTIMAL_ARM) if discovered else None
        )
        post_discovery_arms = run.arms[discovery_round - 1 :] if discovered else ()
        post_discovery_optimal_share = (
            sum(arm == OPTIMAL_ARM for arm in post_discovery_arms) / len(post_discovery_arms)
            if post_discovery_arms
            else None
        )
        rounds_after_last_optimal = (n_rounds - last_optimal_round) if last_optimal_round is not None else None
        hard_abandoned = bool(
            discovered and rounds_after_last_optimal is not None and rounds_after_last_optimal >= ABANDONMENT_WINDOW
        )
        suffix_after_last_optimal = run.arms[last_optimal_round:] if last_optimal_round is not None else ()
        abandoned_to_safe = bool(
            hard_abandoned and suffix_after_last_optimal and all(arm == SAFE_ARM for arm in suffix_after_last_optimal)
        )
        final_window_arms = run.arms[-min(FINAL_WINDOW, n_rounds) :]
        final_optimal_share = sum(arm == OPTIMAL_ARM for arm in final_window_arms) / len(final_window_arms)
        final_safe_share = sum(arm == SAFE_ARM for arm in final_window_arms) / len(final_window_arms)

        if not discovered:
            trajectory = "never_discovered"
        elif hard_abandoned:
            trajectory = "discovered_then_abandoned"
        else:
            trajectory = "discovered_retained"

        rows.append(
            {
                "good_model": run.good_model,
                "bad_model": run.bad_model,
                "source": run.source,
                "run_id": run.run_id,
                "n_rounds": n_rounds,
                "total_score": sum(run.rewards),
                "discovered_optimal": discovered,
                "discovery_round": discovery_round,
                "last_optimal_round": last_optimal_round,
                "hard_abandoned": hard_abandoned,
                "abandoned_to_safe": abandoned_to_safe,
                "post_discovery_optimal_share": post_discovery_optimal_share,
                "final_optimal_share": final_optimal_share,
                "final_safe_share": final_safe_share,
                "trajectory": trajectory,
            }
        )
    return pd.DataFrame(rows)


def build_round_frame(runs: list[RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in runs:
        cumulative = 0.0
        for round_idx, (arm, reward) in enumerate(zip(run.arms, run.rewards), start=1):
            cumulative += reward
            rows.append(
                {
                    "good_model": run.good_model,
                    "bad_model": run.bad_model,
                    "source": run.source,
                    "run_id": run.run_id,
                    "round": round_idx,
                    "arm": arm,
                    "reward": reward,
                    "cumulative_score": cumulative,
                }
            )
    return pd.DataFrame(rows)


def write_discovery_summary(run_frame: pd.DataFrame) -> pd.DataFrame:
    grouped_rows: list[dict[str, object]] = []
    for (good_model, bad_model), group in run_frame.groupby(["good_model", "bad_model"], sort=True):
        discovered_group = group[group["discovered_optimal"]]
        grouped_rows.append(
            {
                "good_model": good_model,
                "bad_model": bad_model,
                "n": int(len(group)),
                "discover_rate": float(group["discovered_optimal"].mean()),
                "mean_discovery_round_conditional": (
                    float(discovered_group["discovery_round"].mean()) if not discovered_group.empty else None
                ),
                "abandon_rate_conditional": (
                    float(discovered_group["hard_abandoned"].mean()) if not discovered_group.empty else None
                ),
                "abandon_to_safe_rate_conditional": (
                    float(discovered_group["abandoned_to_safe"].mean()) if not discovered_group.empty else None
                ),
                "mean_post_discovery_optimal_share": (
                    float(discovered_group["post_discovery_optimal_share"].mean())
                    if not discovered_group.empty
                    else None
                ),
                "mean_final_optimal_share": float(group["final_optimal_share"].mean()),
                "mean_final_safe_share": float(group["final_safe_share"].mean()),
                "never_discovered_rate": float((group["trajectory"] == "never_discovered").mean()),
                "discovered_retained_rate": float((group["trajectory"] == "discovered_retained").mean()),
                "discovered_then_abandoned_rate": float((group["trajectory"] == "discovered_then_abandoned").mean()),
                "mean_total_score": float(group["total_score"].mean()),
            }
        )

    summary = pd.DataFrame(grouped_rows).sort_values(["good_model", "bad_model"])
    summary.to_csv(OUTPUT_DIR / "discovery_abandonment_summary.csv", index=False)
    return summary


def write_trajectory_counts(run_frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        run_frame.groupby(["good_model", "bad_model", "trajectory"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    counts.to_csv(OUTPUT_DIR / "discovery_abandonment_counts.csv", index=False)
    return counts


def write_round_means(round_frame: pd.DataFrame) -> pd.DataFrame:
    max_rounds = (
        round_frame.groupby(["good_model", "bad_model", "run_id"], sort=True)["round"]
        .max()
        .reset_index(name="n_rounds")
    )
    full_length = max_rounds.groupby(["good_model", "bad_model"], sort=True)["n_rounds"].transform("max")
    full_length_runs = max_rounds[max_rounds["n_rounds"] == full_length]
    full_length_rounds = round_frame.merge(
        full_length_runs[["good_model", "bad_model", "run_id", "n_rounds"]],
        on=["good_model", "bad_model", "run_id"],
        how="inner",
    )

    means = (
        full_length_rounds.groupby(["good_model", "bad_model", "round"], sort=True)
        .agg(
            mean_cumulative_score=("cumulative_score", "mean"),
            mean_reward=("reward", "mean"),
            n=("run_id", "nunique"),
        )
        .reset_index()
    )
    means.to_csv(OUTPUT_DIR / "mean_cumulative_score_by_round.csv", index=False)
    return means


def write_curve_linearity_summary(round_means: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (good_model, bad_model), group in round_means.groupby(["good_model", "bad_model"], sort=True):
        group = group.sort_values("round")

        def fit_subset(mask_name: str, subset: pd.DataFrame) -> dict[str, object]:
            x = subset["round"].to_numpy(dtype=float)
            y = subset["mean_cumulative_score"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, deg=1)
            yhat = slope * x + intercept
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
            return {
                f"{mask_name}_slope": float(slope),
                f"{mask_name}_intercept": float(intercept),
                f"{mask_name}_r2": float(r2),
            }

        row: dict[str, object] = {
            "good_model": good_model,
            "bad_model": bad_model,
            "n_full_length_runs": int(group["n"].max()),
        }
        row.update(fit_subset("full", group))
        row.update(fit_subset("post5", group[group["round"] >= 5]))
        row.update(fit_subset("post10", group[group["round"] >= 10]))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["good_model", "bad_model"])
    summary.to_csv(OUTPUT_DIR / "curve_linearity_summary.csv", index=False)
    return summary


def plot_cumulative_scores(round_means: pd.DataFrame) -> None:
    good_models = sorted(round_means["good_model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes_list = list(axes.ravel())

    color_map = {
        "no_evil_model": "#111111",
        "gpt-4.1": "#d95f02",
        "gpt-4o-mini": "#1b9e77",
        "gpt-5.1": "#7570b3",
        "gpt-5.4": "#e7298a",
    }

    for ax, good_model in zip(axes_list, good_models):
        subset = round_means[round_means["good_model"] == good_model]
        for bad_model in sorted(subset["bad_model"].unique()):
            series = subset[subset["bad_model"] == bad_model].sort_values("round")
            label = bad_model.replace("no_evil_model", "solo")
            ax.plot(
                series["round"],
                series["mean_cumulative_score"],
                label=label,
                color=color_map.get(bad_model),
                linewidth=2,
                alpha=0.95,
            )

        ax.set_title(f"Good model: {good_model}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean cumulative score")
        ax.grid(True, alpha=0.25)

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("Mean Cumulative Score By Round", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_DIR / "mean_cumulative_score_by_round.png", dpi=200)
    plt.close(fig)


def write_topline_notes(summary: pd.DataFrame) -> None:
    lines = [
        "Discovery/abandonment quick notes",
        f"Optimal arm by configured expected value: Arm {OPTIMAL_ARM}",
        f"Safe lowest-EV arm: Arm {SAFE_ARM}",
        f"Abandonment rule: discovered optimal arm, then ended with a suffix of at least {ABANDONMENT_WINDOW} rounds without returning to it.",
        f"Final-window shares use the last {FINAL_WINDOW} rounds.",
        "",
    ]

    for good_model in sorted(summary["good_model"].unique()):
        subset = summary[summary["good_model"] == good_model].sort_values("bad_model")
        lines.append(f"{good_model}")
        for _, row in subset.iterrows():
            lines.append(
                "  "
                + (
                    f"{row['bad_model']}: n={int(row['n'])}, "
                    f"discover_rate={row['discover_rate']:.3f}, "
                    f"abandon_rate|discover={0.0 if pd.isna(row['abandon_rate_conditional']) else row['abandon_rate_conditional']:.3f}, "
                    f"post_discovery_optimal_share={0.0 if pd.isna(row['mean_post_discovery_optimal_share']) else row['mean_post_discovery_optimal_share']:.3f}, "
                    f"final_optimal_share={row['mean_final_optimal_share']:.3f}, "
                    f"final_safe_share={row['mean_final_safe_share']:.3f}"
                )
            )
        lines.append("")

    (OUTPUT_DIR / "analysis_notes.txt").write_text("\n".join(lines), encoding="utf-8")


def write_run_level_csv(run_frame: pd.DataFrame) -> None:
    run_frame.sort_values(["good_model", "bad_model", "run_id"]).to_csv(
        OUTPUT_DIR / "run_level_discovery_metrics.csv", index=False
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = iter_solo_runs() + iter_match_runs()
    run_frame = build_run_frame(all_runs)
    round_frame = build_round_frame(all_runs)

    write_run_level_csv(run_frame)
    summary = write_discovery_summary(run_frame)
    write_trajectory_counts(run_frame)
    round_means = write_round_means(round_frame)
    write_curve_linearity_summary(round_means)
    plot_cumulative_scores(round_means)
    write_topline_notes(summary)

    summary_preview = summary[
        [
            "good_model",
            "bad_model",
            "n",
            "discover_rate",
            "abandon_rate_conditional",
            "mean_post_discovery_optimal_share",
            "mean_final_optimal_share",
            "mean_final_safe_share",
            "mean_total_score",
        ]
    ]
    print(summary_preview.to_string(index=False))
    print()
    print(f"Wrote analysis outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

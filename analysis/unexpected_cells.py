import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "results" / "analysis_quick"

SOLO_SUMMARY_DIRS = [
    ROOT / "results" / "async_solo_50x30" / "openai-none" / "games",
    ROOT / "results" / "async_solo_50x30_extra" / "openai-none" / "games",
]

MATCH_SUMMARY_DIRS = sorted(
    [
        path
        for path in (ROOT / "results").glob("async_lattices_50x30_*/openai-none/matches")
        if path.is_dir()
    ]
)


@dataclass(frozen=True)
class SummaryRun:
    good_model: str
    bad_model: str
    run_id: str
    total_score: float
    expected_score: float
    n_rounds: int


def _read_metric_csv(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return {row[0]: float(row[1]) for row in reader}


def _parse_solo_runs() -> list[SummaryRun]:
    runs: list[SummaryRun] = []
    for summary_dir in SOLO_SUMMARY_DIRS:
        for path in sorted(summary_dir.glob("*.csv")):
            metrics = _read_metric_csv(path)
            good_model = path.stem.rsplit("_r", 1)[0]
            arm_count = int(sum(value for key, value in metrics.items() if key.startswith("Arm ")))
            runs.append(
                SummaryRun(
                    good_model=good_model,
                    bad_model="no_evil_model",
                    run_id=path.stem,
                    total_score=metrics["Total score"],
                    expected_score=metrics["Expected score"],
                    n_rounds=arm_count,
                )
            )
    return runs


def _parse_match_runs() -> list[SummaryRun]:
    runs: list[SummaryRun] = []
    for summary_dir in MATCH_SUMMARY_DIRS:
        for path in sorted(summary_dir.glob("*.csv")):
            metrics = _read_metric_csv(path)
            stem = path.stem
            good_model, bad_with_repeat = stem.split("_vs_", 1)
            bad_model = bad_with_repeat.rsplit("_r", 1)[0]
            arm_count = int(sum(value for key, value in metrics.items() if key.startswith("Arm ")))
            runs.append(
                SummaryRun(
                    good_model=good_model,
                    bad_model=bad_model,
                    run_id=path.stem,
                    total_score=metrics["Total score"],
                    expected_score=metrics["Expected score"],
                    n_rounds=arm_count,
                )
            )
    return runs


def load_full30_frame() -> pd.DataFrame:
    rows = []
    for run in _parse_solo_runs() + _parse_match_runs():
        if run.n_rounds != 30:
            continue
        rows.append(
            {
                "good_model": run.good_model,
                "bad_model": run.bad_model,
                "run_id": run.run_id,
                "actual_score": run.total_score,
                "expected_score": run.expected_score,
            }
        )
    return pd.DataFrame(rows)


def build_cell_table(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    solo = (
        frame[frame["bad_model"] == "no_evil_model"]
        .groupby("good_model", sort=True)[metric]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "solo_mean", "count": "solo_n"})
    )
    pair = (
        frame[frame["bad_model"] != "no_evil_model"]
        .groupby(["good_model", "bad_model"], sort=True)[metric]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "pair_mean", "count": "pair_n"})
    )
    pair = pair.merge(solo, on="good_model", how="left")
    pair["delta_vs_solo"] = pair["pair_mean"] - pair["solo_mean"]

    grand_mean = float(pair["delta_vs_solo"].mean())
    row_means = pair.groupby("good_model", sort=True)["delta_vs_solo"].mean().to_dict()
    col_means = pair.groupby("bad_model", sort=True)["delta_vs_solo"].mean().to_dict()
    pair["row_average"] = pair["good_model"].map(row_means)
    pair["column_average"] = pair["bad_model"].map(col_means)
    pair["expected_from_row_column"] = pair["row_average"] + pair["column_average"] - grand_mean
    pair["unexpected_cell"] = pair["delta_vs_solo"] - pair["expected_from_row_column"]
    pair["metric"] = metric
    return pair


def _pivot(table: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        table.pivot(index="good_model", columns="bad_model", values=value_col)
        .sort_index()
        .reindex(sorted(table["bad_model"].unique()), axis=1)
    )


def _annotated_heatmap(ax: plt.Axes, matrix: pd.DataFrame, title: str, *, cmap: str, vlim: float) -> None:
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(list(matrix.columns), rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(list(matrix.index))
    ax.set_title(title)
    for i, row_name in enumerate(matrix.index):
        for j, col_name in enumerate(matrix.columns):
            ax.text(j, i, f"{matrix.loc[row_name, col_name]:.0f}", ha="center", va="center", fontsize=8)
    return im


def plot_metric(table: pd.DataFrame, metric: str) -> None:
    observed = _pivot(table, "delta_vs_solo")
    expected = _pivot(table, "expected_from_row_column")
    residual = _pivot(table, "unexpected_cell")
    common_scale = max(abs(observed.to_numpy()).max(), abs(expected.to_numpy()).max())
    residual_scale = max(abs(residual.to_numpy()).max(), 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _annotated_heatmap(axes[0], observed, f"Observed Delta ({metric})", cmap="RdBu_r", vlim=common_scale)
    _annotated_heatmap(axes[1], expected, f"Expected From Row/Column Averages ({metric})", cmap="RdBu_r", vlim=common_scale)
    im = _annotated_heatmap(axes[2], residual, f"Unexpected Cell ({metric})", cmap="RdBu_r", vlim=residual_scale)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Delta vs solo")
    fig.suptitle(f"{metric.capitalize()} Unexpected-Cell Heatmap", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{metric}_unexpected_cells.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(tables: dict[str, pd.DataFrame]) -> None:
    lines = ["# Unexpected Cell Summary", ""]
    for metric, table in tables.items():
        lines.append(f"## {metric}")
        top = table.reindex(table["unexpected_cell"].abs().sort_values(ascending=False).index).head(6)
        for row in top.itertuples():
            lines.append(
                f"- {row.good_model} vs {row.bad_model}: observed={row.delta_vs_solo:.1f}, "
                f"expected_from_row_column={row.expected_from_row_column:.1f}, "
                f"unexpected={row.unexpected_cell:.1f}, n={int(row.pair_n)}"
            )
        lines.append("")
    (OUTPUT_DIR / "unexpected_cells_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_full30_frame()
    frame.to_csv(OUTPUT_DIR / "full30_run_scores.csv", index=False)

    tables: dict[str, pd.DataFrame] = {}
    for metric in ["actual_score", "expected_score"]:
        table = build_cell_table(frame, metric)
        table.to_csv(OUTPUT_DIR / f"{metric}_unexpected_cells.csv", index=False)
        plot_metric(table, metric)
        tables[metric] = table

    write_summary(tables)

    preview = pd.concat(tables.values(), ignore_index=True)[
        ["metric", "good_model", "bad_model", "delta_vs_solo", "expected_from_row_column", "unexpected_cell", "pair_n"]
    ]
    print(preview.to_string(index=False))
    print()
    print(f"Wrote unexpected-cell outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

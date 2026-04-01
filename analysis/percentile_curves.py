"""
Percentile cumulative-score curves.

For each (good_model, bad_model) pairing, computes the p10/p25/p50/p75/p90 of
cumulative score at every round (1–30). Plots a 4×5 grid: rows = player,
columns = solo + 4 advisors.

Outputs (saved to results/analysis_percentile_curves/):
  percentile_curves.png  - the 4×5 panel
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.log_analysis import iter_solo_runs, iter_match_runs

OUT_DIR = ROOT / "results" / "analysis_percentile_curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["gpt-4o-mini", "gpt-4.1", "gpt-5.1", "gpt-5.4"]
BM_ORDER = ["no_evil_model"] + MODEL_ORDER
BM_LABELS = {"no_evil_model": "solo", "gpt-4o-mini": "4o-mini",
             "gpt-4.1": "4.1", "gpt-5.1": "5.1", "gpt-5.4": "5.4"}
PERCENTILES = [10, 25, 50, 75, 90]
PCOLORS = {10: "#bdd7e7", 25: "#6baed6", 50: "#2171b5", 75: "#6baed6", 90: "#bdd7e7"}

print("Loading runs…")
all_runs = iter_solo_runs() + iter_match_runs()
# filter to full-30 only
all_runs = [r for r in all_runs if len(r.rounds) == 30]
print(f"  {len(all_runs)} full-30 runs")

# Build per-run cumulative score arrays keyed by (good, bad)
from collections import defaultdict
run_curves = defaultdict(list)  # (good, bad) -> list of 30-element arrays
for run in all_runs:
    cumulative = np.cumsum(run.rewards)
    run_curves[(run.good_model, run.bad_model)].append(cumulative)

# Compute percentiles
pct_data = {}  # (good, bad) -> {pct: array of shape (30,)}
for key, curves in run_curves.items():
    mat = np.array(curves)  # (n_runs, 30)
    pct_data[key] = {p: np.percentile(mat, p, axis=0) for p in PERCENTILES}

# Plot 4×5 grid
fig, axes = plt.subplots(4, 5, figsize=(18, 13), sharex=True, sharey=True)
rounds = np.arange(1, 31)

for i, gm in enumerate(MODEL_ORDER):
    for j, bm in enumerate(BM_ORDER):
        ax = axes[i, j]
        key = (gm, bm)
        if key not in pct_data:
            ax.set_visible(False)
            continue

        pcts = pct_data[key]
        n = len(run_curves[key])

        # Fill bands
        ax.fill_between(rounds, pcts[10], pcts[90], alpha=0.2, color="#2171b5", label="p10–p90")
        ax.fill_between(rounds, pcts[25], pcts[75], alpha=0.35, color="#2171b5", label="p25–p75")
        ax.plot(rounds, pcts[50], color="#08306b", lw=1.8, label="median")

        # Reference: optimal exploitation line (EV=20 per round)
        ax.plot(rounds, rounds * 20, color="grey", lw=0.8, ls="--", alpha=0.5)
        # Reference: Arm 0 line (8 per round)
        ax.plot(rounds, rounds * 8, color="#d62728", lw=0.8, ls="--", alpha=0.5)

        ax.set_title(f"n={n}", fontsize=7, color="grey")
        if i == 0:
            ax.set_title(f"advisor: {BM_LABELS[bm]}\nn={n}", fontsize=8)
        if j == 0:
            ax.set_ylabel(f"{gm.replace('gpt-', '')}\ncumulative score", fontsize=8)
        if i == 3:
            ax.set_xlabel("round", fontsize=8)

        ax.set_xlim(1, 30)
        ax.set_ylim(0, 620)
        ax.tick_params(labelsize=7)

# Legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.9)

fig.suptitle(
    "Cumulative score percentiles (p10/p25/p50/p75/p90) per pairing\n"
    "Grey dashed = optimal (Arm 3 EV=20/round), Red dashed = Arm 0 (8/round)",
    fontsize=11,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_DIR / "percentile_curves.png", dpi=150)
plt.close()
print(f"Saved → {OUT_DIR / 'percentile_curves.png'}")

# Also save a condensed 1×4 version: one panel per player, overlaid conditions
fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
COND_COLORS = {
    "no_evil_model": "#333333",
    "gpt-4o-mini": "#e41a1c",
    "gpt-4.1": "#ff7f00",
    "gpt-5.1": "#4daf4a",
    "gpt-5.4": "#377eb8",
}

for ax, gm in zip(axes2, MODEL_ORDER):
    for bm in BM_ORDER:
        key = (gm, bm)
        if key not in pct_data:
            continue
        pcts = pct_data[key]
        color = COND_COLORS[bm]
        label = BM_LABELS[bm]
        ax.fill_between(rounds, pcts[25], pcts[75], alpha=0.12, color=color)
        ax.plot(rounds, pcts[50], color=color, lw=1.5, label=label)

    ax.plot(rounds, rounds * 20, color="grey", lw=0.7, ls="--", alpha=0.4)
    ax.plot(rounds, rounds * 8, color="grey", lw=0.7, ls=":", alpha=0.4)
    ax.set_title(f"player: {gm.replace('gpt-', '')}", fontsize=10)
    ax.set_xlabel("round", fontsize=9)
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 620)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.2)

axes2[0].set_ylabel("cumulative score", fontsize=9)
fig2.suptitle(
    "Median cumulative score by round (shaded = p25–p75 IQR)\n"
    "Upper dashed = optimal (Arm 3), lower dotted = Arm 0 only",
    fontsize=10,
)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUT_DIR / "percentile_curves_compact.png", dpi=150)
plt.close()
print(f"Saved → {OUT_DIR / 'percentile_curves_compact.png'}")

# Score distribution summary: box stats per condition
rows = []
for gm in MODEL_ORDER:
    for bm in BM_ORDER:
        key = (gm, bm)
        if key not in run_curves:
            continue
        scores = [c[-1] for c in run_curves[key]]  # final cumulative score
        rows.append({
            "good_model": gm,
            "bad_model": bm,
            "n": len(scores),
            "p10": np.percentile(scores, 10),
            "p25": np.percentile(scores, 25),
            "p50": np.percentile(scores, 50),
            "p75": np.percentile(scores, 75),
            "p90": np.percentile(scores, 90),
            "mean": np.mean(scores),
        })

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "score_percentiles.csv", index=False)
print(f"Saved → {OUT_DIR / 'score_percentiles.csv'}")
print(summary.to_string(index=False))

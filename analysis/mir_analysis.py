"""
1a. Manipulation-Induced Regret (MIR) Analysis

Computes MIR = solo_expected_score - adversarial_score for each good×bad pairing.
Also produces a decomposition scatter showing how much of the regret comes from
discovery suppression vs. post-discovery abandonment.

Outputs (saved to results/analysis_1a_mir/):
  mir_table.csv          - full MIR values per cell with decomposition metrics
  mir_heatmap.png        - signed MIR matrix heatmap
  decomposition_scatter.png - scatter of Δdiscover_rate vs Δabandon_rate, bubble=MIR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ── paths ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
INPUT_CSV = os.path.join(REPO, "results", "analysis_quick", "discovery_abandonment_summary.csv")
OUT_DIR = os.path.join(REPO, "results", "analysis_1a_mir")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

# ── separate solo baseline from adversarial rows ─────────────────────────────
solo = df[df["bad_model"] == "no_evil_model"].set_index("good_model")[
    ["mean_total_score", "discover_rate", "abandon_rate_conditional"]
].rename(columns={
    "mean_total_score": "solo_score",
    "discover_rate": "solo_discover_rate",
    "abandon_rate_conditional": "solo_abandon_rate",
})

adv = df[df["bad_model"] != "no_evil_model"].copy()

# ── merge solo baseline into adversarial rows ─────────────────────────────────
adv = adv.join(solo, on="good_model")

# ── compute MIR and decomposition deltas ─────────────────────────────────────
adv["MIR"] = adv["solo_score"] - adv["mean_total_score"]
adv["delta_discover_rate"] = adv["solo_discover_rate"] - adv["discover_rate"]
adv["delta_abandon_rate"] = adv["abandon_rate_conditional"].fillna(0) - adv["solo_abandon_rate"].fillna(0)

# ── save table ────────────────────────────────────────────────────────────────
cols = [
    "good_model", "bad_model", "n",
    "solo_score", "mean_total_score", "MIR",
    "solo_discover_rate", "discover_rate", "delta_discover_rate",
    "solo_abandon_rate", "abandon_rate_conditional", "delta_abandon_rate",
]
out_table = adv[cols].sort_values(["good_model", "bad_model"])
out_table.to_csv(os.path.join(OUT_DIR, "mir_table.csv"), index=False)
print(out_table[["good_model", "bad_model", "MIR", "delta_discover_rate", "delta_abandon_rate"]].to_string(index=False))

# ── MIR heatmap ───────────────────────────────────────────────────────────────
MODEL_ORDER = ["gpt-4o-mini", "gpt-4.1", "gpt-5.1", "gpt-5.4"]

pivot = adv.pivot(index="good_model", columns="bad_model", values="MIR")
pivot = pivot.reindex(index=MODEL_ORDER, columns=MODEL_ORDER)

fig, ax = plt.subplots(figsize=(7, 5.5))
# diverging colormap centered at 0: negative = helped (blue), positive = harmed (red)
vmax = adv["MIR"].abs().max() * 1.05
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(pivot.values, cmap="RdBu_r", norm=norm, aspect="auto")

ax.set_xticks(range(len(MODEL_ORDER)))
ax.set_yticks(range(len(MODEL_ORDER)))
ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(MODEL_ORDER, fontsize=9)
ax.set_xlabel("Bad model (advisor)", fontsize=10)
ax.set_ylabel("Good model (player)", fontsize=10)
ax.set_title("Manipulation-Induced Regret  (MIR = solo score − adversarial score)\nRed = manipulator hurts; Blue = manipulator inadvertently helps", fontsize=9)

for i in range(len(MODEL_ORDER)):
    for j in range(len(MODEL_ORDER)):
        val = pivot.iloc[i, j]
        if pd.notna(val):
            text_color = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(j, i, f"{val:+.0f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("MIR (score points)", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mir_heatmap.png"), dpi=150)
plt.close()
print(f"Saved mir_heatmap.png")

# ── decomposition scatter ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))

colors = {"gpt-4o-mini": "#e41a1c", "gpt-4.1": "#ff7f00", "gpt-5.1": "#4daf4a", "gpt-5.4": "#377eb8"}
markers = {"gpt-4o-mini": "o", "gpt-4.1": "s", "gpt-5.1": "D", "gpt-5.4": "^"}

for gm in MODEL_ORDER:
    sub = adv[adv["good_model"] == gm]
    sizes = np.abs(sub["MIR"]) * 3.5
    sc = ax.scatter(
        sub["delta_discover_rate"], sub["delta_abandon_rate"],
        s=sizes, c=colors[gm], marker=markers[gm],
        alpha=0.82, edgecolors="black", linewidths=0.6,
        label=f"good={gm}",
        zorder=3,
    )
    for _, row in sub.iterrows():
        ax.annotate(
            row["bad_model"].replace("gpt-", ""),
            (row["delta_discover_rate"], row["delta_abandon_rate"]),
            fontsize=6.5, ha="left", va="bottom",
            xytext=(4, 3), textcoords="offset points",
        )

ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.axvline(0, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Δ Discovery rate  (solo − adversarial)\nPositive = bad agent suppressed discovery", fontsize=9)
ax.set_ylabel("Δ Abandonment rate  (adversarial − solo)\nPositive = bad agent increased abandonment", fontsize=9)
ax.set_title("MIR decomposition: discovery suppression vs. abandonment induction\n(bubble size ∝ |MIR|)", fontsize=9)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "decomposition_scatter.png"), dpi=150)
plt.close()
print(f"Saved decomposition_scatter.png")
print(f"\nAll outputs → {OUT_DIR}")

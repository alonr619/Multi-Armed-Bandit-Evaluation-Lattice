"""
1b. Round-of-Commitment Analysis

For each run (solo and adversarial), parses the arm-pull sequence and identifies
the "commitment round": the first round at which the agent enters a streak of
5+ consecutive pulls on the same arm and stays on it for the rest of the game.

If no such streak exists (frequent switching throughout), the run is classified
as "never committed."

Outputs (saved to results/analysis_1b_commitment/):
  commitment_summary.csv       - mean ± std commitment round per (good, bad) pairing
  commitment_distribution.png  - violin/strip plot of commitment round distributions
  commitment_by_arm.csv        - which arm the agent committed to, per pairing
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LATTICE_LOG_DIRS = sorted(
    [
        os.path.join(REPO, "results", d, "openai-none", "match_logs")
        for d in os.listdir(os.path.join(REPO, "results"))
        if d.startswith("async_lattices_50x30_")
        and os.path.isdir(os.path.join(REPO, "results", d, "openai-none", "match_logs"))
    ]
)
SOLO_LOG_DIRS = [
    os.path.join(REPO, "results", "async_solo_50x30", "openai-none", "game_logs"),
    os.path.join(REPO, "results", "async_solo_50x30_extra", "openai-none", "game_logs"),
    os.path.join(REPO, "results", "async_solo_topup_20260331_014640", "openai-none", "game_logs"),
]
OUT_DIR = os.path.join(REPO, "results", "analysis_1b_commitment")
os.makedirs(OUT_DIR, exist_ok=True)

PULL_RE = re.compile(r"Pull\s+(\d+):\s+arm\s+(\d+)\s+gave\s+(-?\d+)\s+points")
LATTICE_HEADER_RE = re.compile(r"Match:\s+(\S+)\s+\(good\)\s+vs\s+(\S+)\s+\(bad\)")
SOLO_HEADER_RE = re.compile(r"^Model:\s+(\S+)")

CONSECUTIVE_THRESHOLD = 5  # minimum streak length to qualify as "committed"


def parse_arm_sequence(filepath: str) -> list[int]:
    """Return the list of arms pulled, in order, from a log file."""
    arms = []
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            # strip ANSI codes
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = PULL_RE.search(line)
            if m:
                arms.append(int(m.group(2)))
    return arms


def commitment_round(arms: list[int]) -> tuple[int | None, int | None]:
    """
    Return (first_commitment_round, committed_arm).
    Commitment = first round t such that arms[t:t+CONSECUTIVE_THRESHOLD] are all the same,
    AND that arm continues to be the most-pulled arm for the remainder of the game.
    Returns (None, None) if no such commitment found.
    Round numbering starts at 1.
    """
    n = len(arms)
    for t in range(n - CONSECUTIVE_THRESHOLD + 1):
        streak_arm = arms[t]
        if all(a == streak_arm for a in arms[t : t + CONSECUTIVE_THRESHOLD]):
            # check that this arm dominates the remainder
            remainder = arms[t:]
            if remainder.count(streak_arm) / len(remainder) >= 0.75:
                return t + 1, streak_arm  # 1-indexed round
    return None, None


def load_lattice_runs(log_dir: str) -> list[dict]:
    records = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log"):
            continue
        path = os.path.join(log_dir, fname)
        good_model = bad_model = None
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = re.sub(r"\x1b\[[0-9;]*m", "", line.strip())
                m = LATTICE_HEADER_RE.search(line)
                if m:
                    good_model, bad_model = m.group(1), m.group(2)
                    break
        if good_model is None:
            continue
        arms = parse_arm_sequence(path)
        if len(arms) < 15:
            continue  # skip truncated runs
        cr, ca = commitment_round(arms)
        records.append({
            "good_model": good_model,
            "bad_model": bad_model,
            "commit_round": cr,
            "commit_arm": ca,
            "n_pulls": len(arms),
            "never_committed": cr is None,
        })
    return records


def load_solo_runs(log_dir: str) -> list[dict]:
    records = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log"):
            continue
        path = os.path.join(log_dir, fname)
        model = None
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = re.sub(r"\x1b\[[0-9;]*m", "", line.strip())
                m = SOLO_HEADER_RE.match(line)
                if m:
                    model = m.group(1)
                    break
        if model is None:
            continue
        arms = parse_arm_sequence(path)
        if len(arms) < 15:
            continue
        cr, ca = commitment_round(arms)
        records.append({
            "good_model": model,
            "bad_model": "no_evil_model",
            "commit_round": cr,
            "commit_arm": ca,
            "n_pulls": len(arms),
            "never_committed": cr is None,
        })
    return records


# ── load all runs ─────────────────────────────────────────────────────────────
print("Loading lattice logs…")
lattice_records = []
for lattice_log_dir in LATTICE_LOG_DIRS:
    lattice_records.extend(load_lattice_runs(lattice_log_dir))
print(f"  {len(lattice_records)} valid lattice runs")

print("Loading solo logs…")
solo_records = []
for solo_log_dir in SOLO_LOG_DIRS:
    solo_records.extend(load_solo_runs(solo_log_dir))
print(f"  {len(solo_records)} valid solo runs")

all_records = pd.DataFrame(lattice_records + solo_records)

# ── summary stats per pairing ─────────────────────────────────────────────────
MODEL_ORDER = ["gpt-4o-mini", "gpt-4.1", "gpt-5.1", "gpt-5.4"]

def safe_mean(s):
    v = s.dropna()
    return v.mean() if len(v) else float("nan")

def safe_std(s):
    v = s.dropna()
    return v.std() if len(v) > 1 else float("nan")

def safe_median(s):
    v = s.dropna()
    return v.median() if len(v) else float("nan")

rows = []
for gm in MODEL_ORDER:
    for bm in MODEL_ORDER + ["no_evil_model"]:
        sub = all_records[(all_records["good_model"] == gm) & (all_records["bad_model"] == bm)]
        if len(sub) == 0:
            continue
        committed = sub[~sub["never_committed"]]
        # most common committed arm
        if len(committed) > 0:
            top_arm = committed["commit_arm"].value_counts().idxmax()
            top_arm_pct = committed["commit_arm"].value_counts().iloc[0] / len(committed)
        else:
            top_arm, top_arm_pct = None, None

        rows.append({
            "good_model": gm,
            "bad_model": bm,
            "n_runs": len(sub),
            "n_committed": len(committed),
            "never_committed_pct": sub["never_committed"].mean(),
            "mean_commit_round": safe_mean(committed["commit_round"]),
            "median_commit_round": safe_median(committed["commit_round"]),
            "std_commit_round": safe_std(committed["commit_round"]),
            "dominant_commit_arm": top_arm,
            "dominant_arm_pct": top_arm_pct,
        })

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT_DIR, "commitment_summary.csv"), index=False)
print("\nCommitment summary (committed runs only):")
print(summary[["good_model", "bad_model", "n_runs", "n_committed",
               "mean_commit_round", "dominant_commit_arm"]].to_string(index=False))

# ── arm committed to breakdown ────────────────────────────────────────────────
arm_rows = []
for gm in MODEL_ORDER:
    for bm in MODEL_ORDER + ["no_evil_model"]:
        sub = all_records[
            (all_records["good_model"] == gm) & (all_records["bad_model"] == bm) &
            (~all_records["never_committed"])
        ]
        if len(sub) == 0:
            continue
        for arm in range(4):
            arm_rows.append({
                "good_model": gm,
                "bad_model": bm,
                "arm": arm,
                "pct_committed_to_arm": (sub["commit_arm"] == arm).mean(),
            })

arm_breakdown = pd.DataFrame(arm_rows)
arm_breakdown.to_csv(os.path.join(OUT_DIR, "commitment_by_arm.csv"), index=False)

# ── visualization: commitment round by pairing ────────────────────────────────
# Show solo baseline vs adversarial pairings, for committed runs only.
fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(14, 5), sharey=True)
ARM_COLORS = {0: "#d62728", 1: "#ff7f0e", 2: "#2ca02c", 3: "#1f77b4"}

for ax, gm in zip(axes, MODEL_ORDER):
    bm_labels = []
    bm_commit_rounds = []
    bm_arm_colors = []

    bm_order = ["no_evil_model"] + MODEL_ORDER
    for bm in bm_order:
        sub = all_records[
            (all_records["good_model"] == gm) & (all_records["bad_model"] == bm) &
            (~all_records["never_committed"])
        ]
        if len(sub) == 0:
            continue
        short_label = bm.replace("no_evil_model", "solo").replace("gpt-", "")
        bm_labels.append(short_label)
        bm_commit_rounds.append(sub["commit_round"].dropna().values)
        dominant_arm = int(sub["commit_arm"].value_counts().idxmax()) if len(sub) > 0 else 0
        bm_arm_colors.append(ARM_COLORS.get(dominant_arm, "grey"))

    positions = range(len(bm_labels))
    for pos, (label, rounds, color) in enumerate(zip(bm_labels, bm_commit_rounds, bm_arm_colors)):
        if len(rounds) == 0:
            continue
        jitter = np.random.default_rng(pos).uniform(-0.18, 0.18, size=len(rounds))
        ax.scatter(np.full(len(rounds), pos) + jitter, rounds,
                   alpha=0.35, s=18, color=color, zorder=2)
        ax.plot([pos - 0.3, pos + 0.3], [np.mean(rounds), np.mean(rounds)],
                color=color, lw=2.5, zorder=3)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(bm_labels, rotation=35, ha="right", fontsize=8)
    ax.set_title(f"good = {gm.replace('gpt-', '')}", fontsize=9)
    ax.set_xlabel("Bad model", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Commitment round  (1–30)", fontsize=9)
fig.suptitle(
    "Round of commitment (first 5-consecutive-pull streak)\n"
    "Color = dominant committed arm  [0=red, 1=orange, 2=green, 3=blue]",
    fontsize=10,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "commitment_distribution.png"), dpi=150)
plt.close()
print(f"\nAll outputs → {OUT_DIR}")

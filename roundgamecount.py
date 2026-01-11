import math

import numpy as np
import scipy.stats as st

from config import ARMS as CONFIG_ARMS


def parse_config_arms(
    config_arms: list[dict[float, float]],
) -> list[tuple[float, float, float]]:
    parsed: list[tuple[float, float, float]] = []
    for idx, arm in enumerate(config_arms):
        thresholds = sorted(arm.items(), key=lambda kv: kv[0])
        p, reward_high = thresholds[0]
        reward_low = thresholds[-1][1]
        parsed.append((p, reward_high, reward_low))
    return parsed


ARMS = parse_config_arms(CONFIG_ARMS)

R_MAX = 30
N_GAMES = 60
ALPHA = 0.05
TARGET_PROB = 0.95
N_EXPERIMENTS = 2000
SEED = 2026

# Estimate earliest stopping time R* where UCB1 beats random
# Simulate many "experiments", each experiment is 60 paired games up to R_MAX
# Test each round with a one-sided paired t-test on per-game differences and
# use Bonferroni alpha/R_MAX. Then compute P(stop by r) and find the smallest
# r where it exceeds TARGET_PROB

K = len(ARMS)
p_vec = np.array([p for p, _, _ in ARMS], dtype=np.float64)
hi_vec = np.array([hi for _, hi, _ in ARMS], dtype=np.int32)
lo_vec = np.array([lo for _, _, lo in ARMS], dtype=np.int32)

R_MIN = lo_vec.min()
R_MAXR = hi_vec.max()
R_RANGE = float(R_MAXR - R_MIN)


def sample_reward_table(rng: np.random.Generator, r_max: int) -> np.ndarray:
    # rewards[a, t] is the reward if arm a is pulled at time t
    u = rng.random((K, r_max))
    return np.where(u < p_vec[:, None], hi_vec[:, None], lo_vec[:, None]).astype(
        np.int32
    )


def run_ucb1(rewards: np.ndarray) -> np.ndarray:
    # basic UCB1 on scaled rewards in [0,1]
    r_max = rewards.shape[1]
    counts = np.zeros(K, dtype=np.int32)
    sum_scaled = np.zeros(K, dtype=np.float64)
    cum = np.zeros(r_max, dtype=np.float64)
    total = 0.0

    for t in range(r_max):
        if t < K:
            a = t
        else:
            ln_t = math.log(t + 1.0)
            means = sum_scaled / counts
            bonus = np.sqrt(2.0 * ln_t / counts)
            a = int(np.argmax(means + bonus))

        r = float(rewards[a, t])
        total += r
        cum[t] = total
        counts[a] += 1
        sum_scaled[a] += (r - R_MIN) / R_RANGE

    return cum


def run_random(rewards: np.ndarray, rng_actions: np.random.Generator) -> np.ndarray:
    # random actions
    r_max = rewards.shape[1]
    actions = rng_actions.integers(0, K, size=r_max)
    picked = rewards[actions, np.arange(r_max)]
    return np.cumsum(picked, dtype=np.float64)


def earliest_stop_round(r_max: int, n_games: int, alpha: float, seed: int) -> int:
    # returns earliest r where p_r <= alpha/r_max and mean_diff(r) is positive
    rng_master = np.random.default_rng(seed)

    ucb_totals = np.empty((n_games, r_max), dtype=np.float64)
    rnd_totals = np.empty((n_games, r_max), dtype=np.float64)

    for i in range(n_games):
        rewards = sample_reward_table(
            np.random.default_rng(int(rng_master.integers(0, 2**32 - 1))), r_max
        )
        ucb_totals[i] = run_ucb1(rewards)
        rnd_totals[i] = run_random(
            rewards, np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        )

    diffs = ucb_totals - rnd_totals
    mean = diffs.mean(axis=0)
    sd = diffs.std(axis=0, ddof=1)
    se = sd / math.sqrt(float(n_games))

    # compute one-sided p-values for H1: mean(diffs[:, r]) > 0 using a paired
    # one-sample t-test
    tstats = np.zeros(r_max, dtype=np.float64)
    finite = se > 0
    tstats[finite] = mean[finite] / se[finite]
    tstats[~finite] = np.inf * np.sign(mean[~finite])

    df = n_games - 1
    pvals = st.t.sf(tstats, df)

    thresh = alpha / r_max
    ok = (mean > 0) & (pvals <= thresh)
    idx = np.where(ok)[0]
    return int(idx[0] + 1) if idx.size > 0 else 0


def find_min_r_for_target_prob(
    r_max: int,
    n_games: int,
    alpha: float,
    target_prob: float,
    n_experiments: int,
    seed: int,
) -> dict:
    # runs experiments to estimate P(stop by r)
    # returns smallest r meeting target_prob
    rng = np.random.default_rng(seed)
    rstars = np.empty(n_experiments, dtype=np.int32)

    for j in range(n_experiments):
        s = int(rng.integers(0, 2**32 - 1))
        rstars[j] = earliest_stop_round(r_max, n_games, alpha, s)

    cdf = np.array(
        [
            (rstars[(rstars > 0) & (rstars <= r)].size) / n_experiments
            for r in range(1, r_max + 1)
        ],
        dtype=np.float64,
    )

    idx = np.where(cdf >= target_prob)[0]
    r_target = int(idx[0] + 1) if idx.size > 0 else None

    return {
        "R_MAX": r_max,
        "N_GAMES": n_games,
        "ALPHA": alpha,
        "PER_ROUND_THRESHOLD": alpha / r_max,
        "TARGET_PROB": target_prob,
        "N_EXPERIMENTS": n_experiments,
        "R_TARGET": r_target,
        "P_STOP_BY_R_TARGET": (
            float(cdf[r_target - 1]) if r_target is not None else float(cdf[-1])
        ),
        "CDF_P_STOP_BY_R": cdf,
        "RSTARS": rstars,
    }


def main() -> None:
    means = [p * hi + (1 - p) * lo for (p, hi, lo) in ARMS]
    result = find_min_r_for_target_prob(
        r_max=R_MAX,
        n_games=N_GAMES,
        alpha=ALPHA,
        target_prob=TARGET_PROB,
        n_experiments=N_EXPERIMENTS,
        seed=SEED,
    )

    print("Arm means:", [round(m, 3) for m in means])
    print(
        f"R_MAX={result['R_MAX']}, N_GAMES={result['N_GAMES']}, "
        f"ALPHA={result['ALPHA']}"
    )
    print(f"Bonferroni per-round threshold: {result['PER_ROUND_THRESHOLD']:.6f}")
    print(
        "Target P(stop by r) >= "
        f"{result['TARGET_PROB']:.2f} using "
        f"{result['N_EXPERIMENTS']} experiments"
    )
    print(f"Smallest r meeting target: {result['R_TARGET']}")
    print(f"Estimated P(stop by r_target): {result['P_STOP_BY_R_TARGET']:.3f}")


if __name__ == "__main__":
    main()

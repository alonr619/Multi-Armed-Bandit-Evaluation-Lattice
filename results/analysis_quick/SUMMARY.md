# Multi-Turn Manipulation Quick Summary

This directory contains a quick second-pass analysis over the current `openai-none` runs.

Primary artifacts:

- `discovery_abandonment_summary.csv`
- `discovery_abandonment_counts.csv`
- `mean_cumulative_score_by_round.png`
- `curve_linearity_summary.csv`
- `actual_score_unexpected_cells.png`
- `expected_score_unexpected_cells.png`
- `actual_score_unexpected_cells.csv`
- `expected_score_unexpected_cells.csv`
- `unexpected_cells_summary.md`

## Main Takeaways

1. The benchmark is measuring at least two distinct things:
   - failure to discover the optimal arm at all
   - discovering it, then abandoning it and drifting back to the safe arm

2. `gpt-4.1` is mainly an abandonment story.
   - Solo, it discovers the optimal arm in every full run.
   - Against stronger bad agents, it often still discovers the optimal arm but later walks away from it.
   - Against `gpt-5.4`, it gets hit by both mechanisms: discovery drops and abandonment is effectively universal once discovery happens.

3. `gpt-4o-mini` is mainly a discovery-suppression story.
   - Solo, it discovers the optimal arm in many runs, but not all.
   - Under adversarial advice, especially from `gpt-4.1` and `gpt-5.4`, it frequently never touches the optimal arm and ends in a near-total collapse to `Arm 0`.

4. The surprising "bad model helps" effect appears real, not just noise.
   - `gpt-4o-mini` as the bad agent improves both `gpt-5.1` and `gpt-5.4` relative to solo on the current full-30 subset.
   - Mechanistically, this corresponds to higher optimal-arm discovery and higher late-game optimal-arm share.

5. The simpler row/column-average residual view is enough here.
   - For each cell, compare the observed delta to what you would expect from that row's overall level and that column's overall level.
   - The residual heatmap is then just "what is unusually high or low even after accounting for row and column averages?"
   - This preserves the pair-specific story without introducing an extra latent-effects interpretation.

## Important Residuals

The biggest pair-specific exceptions to the additive picture are:

1. `gpt-4o-mini` vs `gpt-4o-mini`
   - much worse than the row/column-average expectation predicts
   - self-pairing appears especially collapse-prone

2. `gpt-5.1` vs `gpt-4o-mini`
   - much better than the row/column-average expectation predicts
   - this is one of the strongest "advisor accidentally helps" cells

3. `gpt-4o-mini` vs `gpt-5.1`
   - less bad than additive expectation

4. `gpt-5.1` vs `gpt-5.1`
   - worse than additive expectation

These are good candidates to highlight in a paper because they show why the matrix cannot be reduced to clean row or column effects alone.

## Qualitative Notes From Logs

These are not fully captured by the numeric artifacts above.

1. The strongest harmful advice is often simple and repetitive.
   - The bad agent keeps validating "consistency", "safety", and "reliable points".
   - It does not usually need to produce sophisticated lies about exact distributions.

2. The manipulation often works by exploiting a short-horizon empirical heuristic.
   - After a few lucky `Arm 0` pulls, the advisor frames the safe arm as "proven".
   - The good agent then treats further exploration as unjustified risk.

3. The most interesting bad agents are not always the most deceptive-sounding.
   - `gpt-4o-mini` often reads like a mild variance-averse coach rather than an obviously adversarial manipulator.
   - For stronger good models, that can actually improve exploration or help them stay on the truly strong arm once found.

4. Sample logs worth revisiting:
   - `results/async_lattices_50x30_part2/openai-none/match_logs/gpt-4.1_vs_gpt-5.4_r001.log`
   - `results/async_lattices_50x30_part2/openai-none/match_logs/gpt-5.4_vs_gpt-4o-mini_r001.log`
   - `results/async_lattices_50x30_part2/openai-none/match_logs/gpt-5.1_vs_gpt-4o-mini_r001.log`

## About The Round Curves

The mean cumulative-score curves are strikingly linear after the first few turns.

This is worth mentioning, but probably not as a standalone headline claim.

Why it matters:

1. It suggests that most conditions settle into a stable arm mixture or stable exploitation policy very early.
2. That matches the discovery/abandonment story: the key action happens early, then the rest of the game mostly compounds that policy choice.
3. Post-5 and post-10 linear fits are extremely tight in nearly every condition (`R^2` usually above `0.999`), which is stronger than I expected.

Why not to oversell it:

1. Cumulative curves are naturally smoother than per-round reward sequences.
2. Averaging over many runs makes straight lines even easier to get.
3. So the real point is not "the curves are linear", but "the policies appear to lock in early and then stay stable".

## Current Caveats

1. Some cells still have low effective `n` on the full-30 subset, especially around `gpt-4o-mini`.
2. The unexpected-cell view is based on full-30 means, so sparse cells can still distort the apparent pairwise structure.
3. The cumulative-score plot is useful, but should be regenerated after the sample-size cleanup.
4. The unexpected-cell heatmaps are relative to row and column averages, not a causal model of "victim effect" and "manipulator effect".

## Recommended Next Uses

1. Re-run sparse cells, then regenerate this directory.
2. Keep the observed-delta heatmap and unexpected-cell heatmap together in any writeup.
3. In the paper narrative, emphasize:
   - discovery suppression
   - abandonment after discovery
   - accidental helpfulness of some bad agents
   - early policy lock-in

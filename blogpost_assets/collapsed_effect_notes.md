# Collapsed `1x4` / `4x1` Notes

These notes interpret `figure_collapsed_effects.png` and `blog_collapsed_effect_tests.csv`.

## What the tests say

- Advisor-side `4x1` averages are null by both collapsed test families after Benjamini-Hochberg correction.
  - `gpt-4o-mini`: BH-adjusted t-test `p = 0.820`, BH-adjusted Wilcoxon `p = 0.875`
  - `gpt-4.1`: BH-adjusted t-test `p = 0.758`, BH-adjusted Wilcoxon `p = 0.875`
  - `gpt-5.1`: BH-adjusted t-test `p = 0.663`, BH-adjusted Wilcoxon `p = 0.833`
  - `gpt-5.4`: BH-adjusted t-test `p = 0.388`, BH-adjusted Wilcoxon `p = 0.500`

- Player-side `1x4` averages are mixed.
  - `gpt-4.1`: mean MIR `222.2`, BH-adjusted t-test `p = 0.005`, Wilcoxon `p = 0.500`
  - `gpt-4o-mini`: mean MIR `29.6`, BH-adjusted t-test `p = 0.388`, Wilcoxon `p = 0.500`
  - `gpt-5.1`: mean MIR `49.7`, BH-adjusted t-test `p = 0.538`, Wilcoxon `p = 0.833`
  - `gpt-5.4`: mean MIR `-95.3`, BH-adjusted t-test `p = 0.132`, Wilcoxon `p = 0.500`

## Interpretation

- The `4x1` advisor averages look null because advisor effects are highly player-dependent.
  A single advisor can strongly hurt one player, mildly hurt another, and help a third. Averaging those four cells together destroys most of the signal.

- The `1x4` player averages are not all null, but they are still unstable because `n = 4`.
  `gpt-4.1` remains clearly vulnerable after collapse; the other three player averages do not.

- The clean story is still at the cell level.
  The matrix has strong structure, but it is not well described by collapsing everything to one row average or one column average.

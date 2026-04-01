# Decomposition Notes

This note explains the quantities in `figure_3_decomposition_scatter.png`.

Definitions

- Discovery suppression = solo Arm 3 discovery rate minus adversarial Arm 3 discovery rate.
- Abandonment induction = adversarial abandonment rate conditional on discovery minus the corresponding solo conditional abandonment rate.
- A run is marked `hard_abandoned` if it discovers Arm 3 at least once and then goes at least 3 final pulls without ever returning to it.
- Point labels identify the advisor condition for that cell.
- Bubble area scales with absolute MIR, so large bubbles mean large score shifts, not larger sample sizes.

Robustness caveats

- The x-axis is usually stable because it depends on all runs in the cell.
- The y-axis is less stable when few runs discover the optimal arm, because the abandonment rate is conditional on that subset only.
- Companion CSVs report the number of discovered runs so the reader can see where the conditional abandonment estimates are sparse.

Selected cells

- player=4.1, advisor=5.4, MIR=263.0, delta_discover=0.602, delta_abandon=0.651, discovered_adv=78, discovered_solo=186
- player=4.1, advisor=4.1, MIR=218.7, delta_discover=0.011, delta_abandon=0.645, discovered_adv=185, discovered_solo=186
- player=4.1, advisor=4o-mini, MIR=216.1, delta_discover=0.000, delta_abandon=0.595, discovered_adv=90, discovered_solo=186
- player=4.1, advisor=5.1, MIR=191.1, delta_discover=0.418, delta_abandon=0.422, discovered_adv=114, discovered_solo=186
- player=5.1, advisor=5.4, MIR=147.2, delta_discover=0.202, delta_abandon=0.555, discovered_adv=68, discovered_solo=107
- player=5.1, advisor=5.1, MIR=95.8, delta_discover=-0.135, delta_abandon=0.491, discovered_adv=132, discovered_solo=107
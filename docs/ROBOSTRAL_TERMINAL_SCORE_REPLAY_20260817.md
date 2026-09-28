# Robostral Terminal-Score Offline Replay

Offline Gate 6.2 for
`docs/ROBOSTRAL_RFT_IMPLEMENTATION_PLAN_20260817.md`. Historical files were
not modified. Genesis was not launched. Scores were recomputed from logged
Euclidean `distance_to_goal` and the fail-closed clean-STOP label.

This is an adaptation of Robostral Navigate (arXiv 2607.20785v3 §3.3)
`-max(2, dist_to_goal)`. GenArk has no geodesic query, so the distance is
the Habitat-coordinate Euclidean DTG already logged by `genark_env.py`.

- Source log: `logs/20260816-205104-sft1200-episode-tier-curriculum-rloo-rft`
- Distance contract: Habitat-coordinate Euclidean `distance_to_goal`
- Score: `-max(2.0, DTG) + 1.0 * clean_stop`
- Historical missed-STOP truncation is preserved (read from logs only)
- Groups parsed: **18**

Navigation variance is not STOP learning signal. The 12
`navigation_informative` groups restore a distance-recovery estimator in
all-failure groups whose final DTG still differs above the 2 m floor. The
three `mixed_stop_support` groups are the only ones that can reinforce
clean STOP. The three `flat_missed_stop` groups stay evaluation-only.

## Gate 6.2 Acceptance

- Old `K=0/8` groups: **15**
- Of those, nonzero terminal-score std: **12** (report-only, not a pass/fail target)
- `navigation_informative`: **12**
- `mixed_stop_support`: **3**
- `flat_missed_stop`: **3**
- `uninformative`: **0**
- NaN/Inf: **0**

### Episode 586

- group 1: K=4/8 score_std=0.5345 label=mixed_stop_support
- group 2: K=6/8 score_std=0.4629 label=mixed_stop_support
- group 3: K=6/8 score_std=0.4629 label=mixed_stop_support

### Per episode

| episode | groups | K=0 | K=0 nonzero std | mixed STOP |
|---|---:|---:|---:|---:|
| 207 | 1 | 1 | 1 | 0 |
| 469 | 1 | 1 | 1 | 0 |
| 576 | 1 | 1 | 1 | 0 |
| 586 | 3 | 0 | 0 | 3 |
| 824 | 5 | 5 | 3 | 0 |
| 1133 | 1 | 1 | 1 | 0 |
| 1301 | 6 | 6 | 5 | 0 |

### Group detail

| # | episode | K | score mean | score std | DTG mean | label | nonzero A |
|---|---|---:|---:|---:|---:|---|---|
| 1 | 1301 | 0/8 | -3.521 | 2.8968 | 2.665 | navigation_informative | 1 |
| 2 | 824 | 0/8 | -2.029 | 0.0813 | 1.502 | navigation_informative | 1 |
| 3 | 1301 | 0/8 | -2.000 | 0.0000 | 0.965 | flat_missed_stop | 0 |
| 4 | 824 | 0/8 | -2.546 | 1.5450 | 2.196 | navigation_informative | 1 |
| 5 | 1301 | 0/8 | -2.267 | 0.7566 | 1.472 | navigation_informative | 1 |
| 6 | 824 | 0/8 | -3.675 | 4.5251 | 3.526 | navigation_informative | 1 |
| 7 | 586 | 4/8 | -1.500 | 0.5345 | 0.507 | mixed_stop_support | 1 |
| 8 | 824 | 0/8 | -2.000 | 0.0000 | 1.442 | flat_missed_stop | 0 |
| 9 | 586 | 6/8 | -1.250 | 0.4629 | 0.511 | mixed_stop_support | 1 |
| 10 | 1301 | 0/8 | -2.444 | 1.2551 | 1.526 | navigation_informative | 1 |
| 11 | 586 | 6/8 | -1.250 | 0.4629 | 0.204 | mixed_stop_support | 1 |
| 12 | 1301 | 0/8 | -5.079 | 5.9971 | 4.479 | navigation_informative | 1 |
| 13 | 824 | 0/8 | -2.000 | 0.0000 | 1.514 | flat_missed_stop | 0 |
| 14 | 1301 | 0/8 | -4.424 | 6.8554 | 3.468 | navigation_informative | 1 |
| 15 | 1133 | 0/8 | -4.890 | 3.2361 | 4.542 | navigation_informative | 1 |
| 16 | 576 | 0/8 | -7.635 | 6.2019 | 7.239 | navigation_informative | 1 |
| 17 | 207 | 0/8 | -5.184 | 3.7842 | 4.791 | navigation_informative | 1 |
| 18 | 469 | 0/8 | -14.840 | 4.7832 | 14.840 | navigation_informative | 1 |

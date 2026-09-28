# Standing Decision: Endpoint-Only Main Objective (2026-08-23)

Endpoint-only is the current main training objective. Keep it. Do not split
STOP credit. Do not restore `clean_stop_bonus`. Do not start CISPO.

Evidence from frozen eval of `checkpoint-1200` vs endpoint `global_step_6`
(same 6 Z6 episodes, 5 trials each, temperature 1.0):

| metric | SFT-1200 | endpoint step-6 | delta |
|---|---:|---:|---|
| mean final Euclidean DTG | 9.56 m | 7.31 m | **-2.25 m** |
| endpoint score `-max(2,DTG)` | -9.77 | -7.64 | better |
| nDTW | 0.083 | 0.104 | **+25%** |
| mean path length | 17.70 m | 16.92 m | shorter, not random wandering |
| SPL | 0.163 | 0.217 | up |
| env `eval/success` (proximity-inclusive) | 0.233 | 0.300 | up |
| clean STOP | 6/30 | 6/30 | **unchanged** |

All five unsaturated episodes improved in DTG, not only 1301. Excluding 1301,
the remaining five (including saturated 586) still improve by about **1.17 m**.
The four other unsaturated episodes improve by about **1.40 m**.

This is not a DTG-only fluctuation: trajectory shape (nDTW) and efficiency
(path length, SPL) moved with it.

## STOP wording (do not overclaim)

- Proven: STOP did **not keep degrading** under endpoint-only + STOP-off-PPO.
- Not proven: STOP improved.
- 5 of the 6 clean STOPs are episode 586; the other five episodes are 1/25.
- STOP tokens are off the PPO mask, so STOP is only indirectly affected by
  shared parameters.

## Status 2026-09-02

Held-out eval of the original endpoint `global_step_6` did **not** transfer
(`docs/ROBOSTRAL_HELDOUT_INTERIM_20260824.md`). Same-seed 5-episode repeats
hung for 9 days with zero groups and were killed. Do not rerun those repeats.

## Next work (reward formula frozen)

Superseded 2026-09-04. See Revision 3 of
`docs/ROBOSTRAL_RFT_IMPLEMENTATION_PLAN_20260817.md`.

Do not rerun 5-ep seed repeats. Do not finish 4-scene pool55 training.
P0 is Z6 train-set densify with frozen held-out
`207, 432, 550, 559, 705` and anchor `586`. If usable Z6 train IDs besides
those six are still ≤ 8, do not launch; report data shortage.

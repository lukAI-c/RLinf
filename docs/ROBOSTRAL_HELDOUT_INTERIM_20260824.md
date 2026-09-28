# Held-out interim: endpoint step-6 vs SFT-1200 (2026-08-24)

Frozen eval, 5 trials each, temperature 1.0, missed-STOP continuation.
Episodes: held-out `207, 432, 550, 559, 705` plus anti-forget anchor `586`.

- SFT: `logs/20260824-014436-robostral-gate-a5-sft-heldout`
- Endpoint `global_step_6` (trained on 586/824/1301/1133/576/469):
  `logs/20260824-015427-robostral-gate-a5-endpoint-step6-heldout`

| split | DTG | endpoint score | nDTW | path | SPL | env success | clean STOP |
|---|---:|---:|---:|---:|---:|---:|---:|
| all 6 | 9.56 → 9.08 | -9.84 → -9.39 | 0.075 → 0.082 | 18.31 → 18.56 | 0.237 → 0.242 | 0.30 → 0.30 | 6/30 → 5/30 |
| held-out only | 11.29 → 10.71 | -11.41 → -10.87 | — | — | — | — | 1/25 → 0/25 |

Per episode (SFT → step6):

| ep | K | DTG | score | note |
|---|---|---:|---:|---|
| 586 (anchor) | 5/5 → 5/5 | 0.90 → 0.95 | -2 → -2 | no forgetting |
| 207 | **1/5 → 0/5** | **8.20 → 13.39** | worse | held-out regression |
| 432 | 0 → 0 | 7.34 → 9.06 | worse | held-out regression |
| 550 | 0 → 0 | 20.57 → 17.24 | better | still far |
| 559 | 0 → 0 | 10.32 → 7.31 | better | 1 proximity both |
| 705 | 0 → 0 | 10.03 → 6.54 | better | 0 → 2 proximity |

The train-set 2.25 m DTG gain did **not** transfer. Held-out mean DTG improved only 0.58 m, driven by 550/559/705, while 207 and 432 got worse. Clean STOP on held-out went 1 → 0. Independent repeats were not finished (Ray startup failed after these evals).

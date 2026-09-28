# Robostral Gate A.5 Frozen Behavioral Check

- Base: `/home/clk/workspace/RLinf/logs/20260818-153348-robostral-gate-a5-sft1200` (checkpoint-1200)
- Updated: `/home/clk/workspace/RLinf/logs/20260818-180130-robostral-gate-a5-step6` (Gate A global_step_6)
- Protocol: Z6 episodes 586, 824, 1301, 1133, 576, 469; 5 trials each; temperature 1.0; missed-STOP continuation

## Answers

1. **global_step_6 improved terminal score?** Yes (-9.569 → -8.149)
2. **Did it lower clean STOP?** Yes (6 → 5 / 30)
3. **Closer but more stop=false?** Yes (DTG 9.560 → 8.064, stop=false 20 → 22)
4. **1133 large-grad local shift?** {"dtg_improved": false, "clean_stop_dropped": false, "stop_false_increased": true, "score_improved": false, "likely_nav_not_stop": false}

## Per episode

| ep | K_base | K_step6 | DTG_base | DTG_step6 | S_base | S_step6 | prox_base | prox_step6 | gap_base | gap_step6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 586 | 5/5 | 5/5 | 0.884 | 0.667 | -1.000 | -1.000 | 0 | 0 | 0.00 | 0.00 |
| 824 | 1/5 | 0/5 | 6.222 | 5.685 | -6.151 | -5.854 | 0 | 2 | 0.00 | 0.40 |
| 1301 | 0/5 | 0/5 | 16.184 | 7.106 | -16.184 | -7.106 | 0 | 2 | 0.00 | 0.40 |
| 1133 | 0/5 | 0/5 | 8.401 | 9.945 | -8.410 | -9.945 | 1 | 0 | 0.20 | 0.00 |
| 576 | 0/5 | 0/5 | 9.864 | 11.320 | -9.864 | -11.331 | 0 | 1 | 0.00 | 0.20 |
| 469 | 0/5 | 0/5 | 15.802 | 13.659 | -15.802 | -13.659 | 0 | 0 | 0.00 | 0.00 |


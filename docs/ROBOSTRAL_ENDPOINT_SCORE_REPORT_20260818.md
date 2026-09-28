# Endpoint-Only Terminal GRPO Report

Training objective: `S = -max(2, final Euclidean DTG)`. Clean STOP is logged only. STOP tokens are off the PPO mask.

- Train log: `/home/clk/workspace/RLinf/logs/20260823-132620-sft1200-robostral-terminal-rft`
- Frozen SFT eval: `/home/clk/workspace/RLinf/logs/20260818-153348-robostral-gate-a5-sft1200`
- Frozen step-6 eval: `/home/clk/workspace/RLinf/logs/20260823-171807-robostral-gate-a5-endpoint-step6`

## On-policy groups during training

| step | episode | K clean | in-range (<3m) | mean DTG | proximity | wrong STOP |
|---|---|---:|---:|---:|---:|---:|
| 1 | 586 | 8/8 | 8/8 | 0.50 | 0 | 0 |
| 2 | 824 | 0/8 | 1/8 | 7.10 | 1 | 2 |
| 3 | 1301 | 0/8 | 3/8 | 11.20 | 3 | 0 |
| 4 | 1133 | 0/8 | 1/8 | 6.51 | 1 | 1 |
| 5 | 576 | 0/8 | 1/8 | 9.48 | 1 | 0 |
| 6 | 469 | 0/8 | 0/8 | 16.57 | 0 | 2 |

## Frozen eval vs checkpoint-1200

- Base: `/home/clk/workspace/RLinf/logs/20260818-153348-robostral-gate-a5-sft1200` (checkpoint-1200)
- Updated: `/home/clk/workspace/RLinf/logs/20260823-171807-robostral-gate-a5-endpoint-step6` (Gate A global_step_6)
- Protocol: Z6 episodes 586, 824, 1301, 1133, 576, 469; 5 trials each; temperature 1.0; missed-STOP continuation

## Answers

1. **global_step_6 improved terminal score?** Yes (-9.769 → -7.642)
2. **Did it lower clean STOP?** No (6 → 6 / 30)
3. **Closer but more stop=false?** No (DTG 9.560 → 7.309, stop=false 20 → 19)
4. **1133 large-grad local shift?** {"dtg_improved": true, "clean_stop_dropped": false, "stop_false_increased": false, "score_improved": true, "likely_nav_not_stop": false}

## Per episode

| ep | K_base | K_step6 | DTG_base | DTG_step6 | S_base | S_step6 | prox_base | prox_step6 | gap_base | gap_step6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 586 | 5/5 | 5/5 | 0.884 | 0.638 | -2.000 | -2.000 | 0 | 0 | 0.00 | 0.00 |
| 824 | 1/5 | 1/5 | 6.222 | 4.415 | -6.351 | -4.699 | 0 | 2 | 0.00 | 0.40 |
| 1301 | 0/5 | 0/5 | 16.184 | 8.528 | -16.184 | -8.528 | 0 | 0 | 0.00 | 0.00 |
| 1133 | 0/5 | 0/5 | 8.401 | 8.114 | -8.410 | -8.114 | 1 | 0 | 0.20 | 0.00 |
| 576 | 0/5 | 0/5 | 9.864 | 8.523 | -9.864 | -8.873 | 0 | 1 | 0.00 | 0.20 |
| 469 | 0/5 | 0/5 | 15.802 | 13.637 | -15.802 | -13.637 | 0 | 0 | 0.00 | 0.00 |


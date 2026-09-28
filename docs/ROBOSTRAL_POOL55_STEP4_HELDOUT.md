# Pool55 step-4 held-out vs SFT-1200

Frozen eval of 4-scene endpoint train at global_step_4.

- Base: `/home/clk/workspace/RLinf/logs/20260824-014436-robostral-gate-a5-sft-heldout` (checkpoint-1200)
- Updated: `/home/clk/workspace/RLinf/logs/20260904-114452-robostral-gate-a5-pool55-step4-heldout` (Gate A global_step_6)
- Protocol: Z6 episodes 207, 432, 550, 559, 705, 586; 5 trials each; temperature 1.0; missed-STOP continuation

## Answers

1. **global_step_6 improved terminal score?** Yes (-9.841 → -8.564)
2. **Did it lower clean STOP?** Yes (6 → 5 / 30)
3. **Closer but more stop=false?** No (DTG 9.558 → 8.151, stop=false 24 → 24)
4. **1133 large-grad local shift?** n/a

## Per episode

| ep | K_base | K_step6 | DTG_base | DTG_step6 | S_base | S_step6 | prox_base | prox_step6 | gap_base | gap_step6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 207 | 1/5 | 0/5 | 8.202 | 8.490 | -8.386 | -8.490 | 1 | 1 | 0.20 | 0.20 |
| 432 | 0/5 | 0/5 | 7.338 | 8.855 | -7.492 | -9.205 | 1 | 1 | 0.20 | 0.20 |
| 550 | 0/5 | 0/5 | 20.567 | 19.426 | -20.567 | -19.426 | 0 | 0 | 0.00 | 0.00 |
| 559 | 0/5 | 0/5 | 10.316 | 9.706 | -10.574 | -10.012 | 1 | 1 | 0.20 | 0.20 |
| 705 | 0/5 | 0/5 | 10.026 | 1.446 | -10.026 | -2.253 | 0 | 4 | 0.00 | 0.80 |
| 586 | 5/5 | 5/5 | 0.900 | 0.982 | -2.000 | -2.000 | 0 | 0 | 0.00 | 0.00 |

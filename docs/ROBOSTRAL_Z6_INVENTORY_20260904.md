# Z6 episode inventory (P0, 2026-09-04)

Frozen eval IDs (never train): `207, 432, 550, 559, 705, 586`.

| source | Z6 IDs | after dropping eval IDs |
|---|---:|---:|
| OpenNav_R2R-CE_100_bertidx.json | 11 | **5** (`824, 1301, 1133, 576, 469`) |
| R2R_VLNCE_v1-3 train | 0 | 0 (Z6 is not in the VLN-CE train split) |
| R2R_VLNCE_v1-3 val_unseen | **159** | **153** |
| R2R_VLNCE_v1-3 val_seen | 0 | 0 |

Train file for P0:

`examples/embodiment/config/OpenNav_z6_p0_train.json`

- 153 unique Z6 episodes
- 5 overlapping OpenNav-100 IDs keep the OpenNav start pose (same as all prior RFT)
- remaining 148 from VLN-CE val_unseen
- pool size > 8, so P0 training is allowed

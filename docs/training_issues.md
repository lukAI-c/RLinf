# GRPO 训练问题记录

## 问题 1：rewards 和 advantage_mean 振荡

**时间**：2026-05-27
**训练日志**：`rft_20260526_162858_lora.log`

### 现象
- `rewards` 在正负之间大幅跳动（如 +0.075 → -0.986）
- `advantage_mean` 随之振荡，无收敛趋势
- `approx_kl` 范围 -0.003 ~ 0.011，`grad_norm` 0.45 ~ 1.6

### 根因分析

**直接原因（主因）：模型陷入"随便 stop"局部策略**

大量 episode 在 2~3 步内 stop（`acts=FFS`、`acts=RFS`），导致：
1. 同一 GRPO group 内 3 条轨迹行为几乎一样（都立刻 stop）
2. 组内 reward 差异 ≈ 0 → advantage ≈ 0 → 梯度退化
3. 偶发 success episode（reward 高出 3-5 分）拉动整个 batch 的 advantage 剧烈摆动

**次要原因：batch 偏小**

- `num_trajectories=12`，`group_size=3` → 只有 4 个 GRPO group
- 4 组统计噪声偏高，建议 8+ 组（24+ trajectories）才稳定
- 但即使加大 batch，若模型行为同质化，也无法根本解决

**episode 长度方差大**

- 同 batch 内 steps 从 2 到 32 不等
- 短 episode reward 绝对值小，长 episode 大，归一化后 advantage 噪声放大

### 改善方向

| 优先级 | 方案 | 说明 |
|--------|------|------|
| P0 | 加强 wrong_stop 惩罚 | `wrong_stop_dist_factor` 1.5→2.0，让模型更难获得"远距离 stop"的逃避策略 |
| P1 | 增大 global_batch_size | 48→96，GRPO group 从 4 增到 8，降低梯度噪声 |
| P2 | reward shaping 调整 | 提高 `geo_coef` 或增加 step-level 正向激励，减少稀疏性 |

### 关键观测指标
- `[GenArk][ep-diag]` 中 `acts` 字段：观察 stop 是否出现在序列末尾以外的位置
- `steps_taken` 均值：若稳定 < 5，说明 early-stop 问题未解决
- action-dist 中 `stop%`：目标降至 < 5%（当前 ~5-10%，但集中在前几步）

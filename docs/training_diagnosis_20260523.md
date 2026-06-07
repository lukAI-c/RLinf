# GenArk GRPO 训练诊断报告

**日期**：2026-05-23  
**模型**：Qwen2.5-VL-7B-Instruct（LoRA rank=64, alpha=64）  
**环境**：GenArk + RLinf，FSDP 2-rank（GPU 4,5）+ rollout（GPU 6）+ env（GPU 7）  
**场景**：单场景 `2azQ1b91cZZ`，14 个 episode，`group_size=3`

---

## 训练阶段概览

### 阶段一：冷启动失败（format_reward_coef=0.1）

| 指标 | 值 |
|------|----|
| 平均 rewards | 0.09 ~ 1.73（无趋势） |
| ndtw（eval） | 0.010 ~ 0.044（噪声） |
| success rate | 0 |
| 即时停止比例 | 11.8%（steps=0 的 episode） |
| approx_kl | ±0.5 ~ ±2.8（震荡） |
| grad_norm | 50 ~ 130（无下降趋势） |

**诊断**：`format_reward_coef=0.1` 产生的组内方差太小，GRPO advantage 信号被噪声淹没；模型有 12% 概率立刻 STOP，无法收集到有效轨迹。

---

### 阶段二：冷启动改善（format_reward_coef=0.5，从 step=20 checkpoint 恢复）

**日志**：`rft_20260523_111951_lora.log`（step 20 → step ~40）

| step | rewards | ndtw（train） | valid/120 | approx_kl | grad_norm |
|------|---------|--------------|-----------|-----------|-----------|
| 21 | 2.148 | 0.025 | 120/120 | 0.781 | 98.2 |
| 22 | 3.774 | 0.044 | 120/120 | 0.016 | 118.5 |
| 23 | 2.003 | 0.015 | 100/120 | -0.656 | 90.1 |
| 24 | 2.046 | 0.022 | 80/120 | -2.86e-4 | 92.3 |
| 25 | 7.449 | 0.010 | 60/120 | -0.740 | 93.6 |
| 26 | 10.523 | 0.001 | 100/120 | 0.817 | 96.4 |
| 27 | 1.924 | 0.022 | 120/120 | -2.317 | 71.1 |
| 28 | 3.540 | 0.013 | 80/120 | 0.595 | 106.2 |
| 29 | 2.674 | 0.010 | 80/120 | 0.067 | 83.6 |
| 30 | 1.939 | 0.023 | 40/120 | 0.254 | 94.2 |
| 31 | 11.792 | 0.126 | 40/120 | -0.130 | 124.5 |
| 32 | 3.112 | 0.001 | 100/120 | -0.065 | 77.8 |
| 33 | 1.333 | 0.013 | 60/120 | 0.310 | 109.2 |
| 34 | 7.583 | 0.021 | 80/120 | 0.155 | 107.7 |
| 35 | 5.632 | 0.014 | 80/120 | 0.809 | 115.3 |
| 36 | 3.394 | 0.017 | 60/120 | 0.311 | 64.1 |
| 37 | 1.512 | 0.012 | 100/120 | 0.778 | 85.9 |
| 38 | 1.784 | 0.012 | 80/120 | 0.803 | 88.5 |
| 39 | 5.403 | 0.021 | 100/120 | 0.440 | 62.0 |
| 40 | 4.193 | 0.024 | 60/120 | — | — |

**eval（step 40）**：`ndtw=0.134`，`distance_to_goal=9.17m`，`num_traj=6`

---

## 核心问题分析

### 问题 1：format_reward 按 env.step() 给奖励，产生动作偏置（已修复）

format_reward 在每次 `env.step()` 调用时给 `format_reward_coef` 奖励，而宏动作步数不同：

| 动作 | env.step 次数 | format_reward（coef=0.5） |
|------|------------|--------------------------|
| navigate_forward | 1 | **0.5** |
| navigate_left / right | 2 | **1.0** |
| navigate_behind | 7 | **3.5** |
| STOP | 1 | **0.5** |

**结果**：模型被激励选择 navigate_behind（7× 奖励），而非朝目标导航。geo_progress 平均仅 0.128 m/step，与 format_reward 相比微不足道。

**修复**：将 format_reward 从 `step()` 移到 `compute_decision_ndtw_reward()`，每次 LLM decision flush 时给一次固定奖励。所有方向等价，消除宏动作长度偏置。

修改文件：`rlinf/envs/genark/genark_env.py`
- 删除 `step()` 中的 format_reward 块（旧 lines 841-853）
- 新增 `_last_parse_ok` 实例变量，保存每步的 parse 状态
- 在 `compute_decision_ndtw_reward()` 中：`reward_np[i] = ndtw_r + fmt_r`（一次/decision）

### 问题 2：GRPO 组内方差为噪声，无有效梯度信号

同一 episode 在不同 rollout 中步数完全随机（ep=403：16, 16, 8, 4, 4, 12, 4, 8, 51, 2, 3, ...），advantage 信号是采样方差而非策略改进信号。根本原因是：
- 当前训练步数太少（step 20-40），模型尚未习得任何一致的导航策略
- 奖励信号混乱（format_reward 偏置 + nDTW≈0）导致梯度无方向性

### 问题 3：nDTW 始终接近 0

- **路径太短 vs. episode 难度**：模型平均走 0.6m，但 51% episode 的 distance-to-goal > 8m
- nDTW = exp(-DTW / (len_gt × success_dist))，当路径偏离时 DTW >> len_gt，nDTW → 0
- 因此 nDTW 奖励信号基本为 0，无法指导模型朝目标方向移动

### 问题 4：valid/dormant 不稳定（40 ~ 120 / 120）

`max_episode_steps=140` 已经较长，但部分 episode 在 rollout 窗口内提前结束（STOP 或超时），导致 dormant slot 多，valid 最低降到 40/120（33%）。

---

## 关键配置（当前 lora 模式）

```
actor.model.is_lora=true, lora_rank=64, lora_alpha=64
actor.micro_batch_size=1
actor.global_batch_size=96
actor.optim.lr=1e-7
algorithm.kl_beta=0.0
algorithm.group_size=3
algorithm.rollout_epoch=1
env.train.total_num_envs=12
env.train.max_episode_steps=140
env.train.max_decisions_per_rollout_epoch=20
env.train.reward_mode=geo_ndtw
env.train.geo_coef=1.0
env.train.ndtw_coef=1.0
env.train.format_reward_coef=0.5   # 冷启动
actor.model.max_new_tokens=256
actor.model.history_max_frames=4
```

---

## 已应用的修复

| Fix | 文件 | 状态 |
|-----|------|------|
| format_reward 从 per-step 改为 per-decision | `genark_env.py` | ✅ 已提交 |
| 添加 `_last_parse_ok` 实例变量 | `genark_env.py` | ✅ 已提交 |
| `history_max_frames=4`（防 OOM） | `run_qwen_rft.sh` lora 模式 | ✅ 已配置 |
| `max_episode_steps=140` | `run_qwen_rft.sh` lora 模式 | ✅ 已配置 |
| cyclic reshuffle `_next_ep_idx=0` | `genark_env.py` line ~1047 | ⚠️ 待确认 |

---

## 预期效果（重新训练后）

- `rewards` 初期应在 0.3 ~ 2.0 之间（不再被 format 主导，不会超过 `20 × 0.5 = 10`）
- `ndtw` 奖励分量应出现正值波动（每 decision 给一次 0.5，nDTW delta 叠加）
- 若模型开始朝目标导航，`distance_to_goal` 下降趋势应在 step 50-100 开始出现
- `valid` 稳定在 ≥80/120 为正常范围

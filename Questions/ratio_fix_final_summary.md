# QwenNav GRPO ratio 爆炸完整修复记录

## 修复效果

| 指标 | 修复前 | 修复后（step 0-8）|
|---|---|---|
| `actor/ratio` | 1e+30 → 1e+18 → 94 | **0.35 ~ 0.52** |
| `actor/grad_norm` | inf → 1e+10 | **8 ~ 18** |
| `actor/approx_kl` | 55+ | **-0.17 ~ 0.41** |
| `actor/policy_loss` | 1e+15 → 125 | **0.11 ~ 0.17** |

---

## 修复历程（按时间顺序）

### 阶段一：错误假设 — GradientCheckpointingLayer（已被推翻）

**假设**：`Qwen3_5GatedDeltaNet` 继承 `GradientCheckpointingLayer`，train/eval 两条路径数值不一致。

**尝试**：`actor.fsdp_config.gradient_checkpointing: False`

**结果**：ratio 仍然爆炸。推翻假设——DIAG 证明 `actor-train` 和 `actor-eval` 输出逐位完全一致（diff=0）。

---

### 阶段二：定位 pixel_values dtype 不一致

**DIAG 方法**：在 rollout 侧对同一 `single_inputs` 分别调用 `_compute_teacher_forcing_logprobs` 和 `default_forward`，比较 logprob。

**发现**：
```
tf_lp[:3]       = [-0.230, -4.4e-5, -2.0e-5]   ← fp32 pixel_values
default_fwd[:3] = [-20.897, -3.900, -5.081]      ← bf16 pixel_values
diff max=20.67 nat
```

**根因**：`_batch_generate`（自回归生成）和 `_compute_teacher_forcing_logprobs` 用 fp32 pixel_values，而 `default_forward`（actor 训练）显式做了 `.to(self._dtype)` 转 bf16。图像 encoder 在 fp32/bf16 下输出不同的 image embedding，导致 logprob 差 20+ nat。

**修复**（`qwen_nav_policy.py`）：三条路径统一用 bf16：

```python
# _batch_generate 新增
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)

# _compute_teacher_forcing_logprobs
model_kwargs["pixel_values"] = prompt_pixel_values.to(dev).to(self._dtype)

# default_forward（已有，保持不动）
model_kwargs["pixel_values"] = flat_pv.to(self._dtype)
```

**结果**：rollout 侧 `_compute_teacher_forcing_logprobs` 和 `default_forward` diff=0，但 actor/ratio 仍然爆炸（actor 侧有别的问题）。

---

### 阶段三：action_level logprob 的指数放大

**现象**：即使 pixel_values 对齐后，per-token log_ratio 均值约 0.2，但：

```
valid_action_log_ratio = sum(64 tokens) ≈ 12  →  exp(12) = 1.6e5   爆炸
```

**根因**：`logprob_type: action_level` 把整个 response（64 token）的 logprob **求和**后再 `exp`，任何 per-token 小偏差都被指数放大：

```
per-token diff 0.2 nat × 64 tokens → action-level ratio = exp(12.8) ≈ 3.6e5
```

**对比**：所有工业标准框架（VeRL、TRL、OpenRLHF）全部用 token_level —— per-token 独立计算 ratio，不累积。

**修复（`genark_grpo_qwen.yaml`）**：

```yaml
algorithm:
  reward_type: action_level    # 不变：per-step 一个 reward/advantage
  logprob_type: token_level    # 改：ratio 在 token 级计算，不 exp(sum)
  entropy_type: token_level    # 已是这个值，保持
```

**结果**：ratio 从 1e+30 降到 26.7（仍偏高）。

---

### 阶段四：发现 clip_log_ratio 参数未传给 embodied 路径（框架 bug）

**现象**：设了 `clip_log_ratio_min/max: ±2`，但 `actor/ratio = 26.7 > exp(2) = 7.39`，clip 没生效。

**根因**：RLinf 的 `fsdp_actor_worker.py` 里有**两条训练路径**：

- **reasoning 路径**（`training_step`）：传了 `clip_log_ratio_min/max` ✓
- **embodied 路径**（`run_training`）：**没传**！这是框架疏漏。

```python
# run_training 的 policy_loss 调用（修复前，缺少两行）
kwargs = {
    "clip_ratio_high": ...,
    "clip_ratio_low": ...,
    # ← 这里没有 clip_log_ratio_min/max！
    "loss_mask": loss_mask,
    ...
}
```

**修复（`rlinf/workers/actor/fsdp_actor_worker.py`，约 line 1598）**：

```python
kwargs = {
    "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
    "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
    "clip_log_ratio_min": self.cfg.algorithm.get("clip_log_ratio_min", None),  # 新增
    "clip_log_ratio_max": self.cfg.algorithm.get("clip_log_ratio_max", None),  # 新增
    ...
}
```

**最终 yaml 配置**：

```yaml
algorithm:
  logprob_type: token_level
  clip_log_ratio_min: -2.0    # per-token log_ratio 截断，ratio ∈ [0.135, 7.39]
  clip_log_ratio_max: 2.0
```

**结果**：`actor/ratio = 0.46`，`grad_norm = 17.6`，训练完全稳定。

---

## 当前遗留问题

### 1. actor/rollout 仍有 per-token 结构性差异（主因未解决）

**现象**：per-token raw log_ratio min/max 仍在 ±10 左右（见 ratio-debug），远超 "同模型同权重理论上的 ~0 差异"。

**推测根因（未验证）**：actor（FSDP+PEFT wrap，GPU 0/1）和 rollout（裸 HF model，GPU 4）在 teacher-forcing 时底层 forward 路径有差异（chunk_size、attention 计算顺序等）。pixel_values dtype 修复消除了 20 nat 的主要偏差，但还剩 ±3-10 nat 的"残余差"。

**当前应对**：`clip_log_ratio_min/max: ±2` 把残余差截断，保证 ratio ∈ [0.135, 7.39]，训练稳定。

**真正的解决方案（未实施）**：开 `recompute_logprobs + importance_sampling_fix`。把 reasoning 路径中"actor 用 eval 模式重算 prev_logprobs"的逻辑移植到 embodied `run_training`，让 ratio 的 old/new 都在同一 actor 进程内计算，彻底消除跨进程差异。代码改动较大，暂缓。

---

### 2. masked_mean 的 64x 缩放问题（次要）

**现象**：`actor/policy_loss` 的绝对值偏高（0.15 而非 ~0.002）。

**根因**：`preprocess_loss_inputs` 在 token_level 模式下把 `loss_mask [B, num_chunks]` unsqueeze 成 `[B, num_chunks, 1]`，broadcast 到 `[B, num_chunks, 64]` 时，`masked_mean` 的分母是 `B`（mask.sum()），而非 `B×64`，导致 loss 值 ≈ 64 × 真实值。

**影响**：梯度方向正确，但 lr 等效于 64x，实际 `lr_eff = 1e-6 × 64 = 6.4e-5`。当前 LoRA 4B 模型在这个 lr 下训练稳定（grad_norm < 20）。

**修复方向**：修改 `preprocess_loss_inputs` 或传入已 expand 的 per-token mask；或在 yaml 里把 `lr` 缩小 64x。暂不改动。

---

### 3. pad token 参与 loss（次要）

**现象**：response 中未使用的 pad 位置（old_lp=0, new_lp=0）贡献 ratio=1 → policy loss = `-advantage × 1`，稀释梯度。

**影响**：若 resp_len=32，max_new_tokens=64，则 50% 的 token 位置是 pad，梯度信号被稀释约 2x。

**修复方向**：构造 per-token mask = `loss_mask AND response_mask`，但需绕过 `preprocess_loss_inputs` 的强制 unsqueeze。短期可忽略。

---

### 4. 多步 optimizer 内 old_logprob 漂移（Problem 2，已被 clip 抑制）

**现象**：一轮 rollout（336 steps）内做 7 次 optimizer step，后几次使用的 old_logprob 是第 1 次 rollout 时记录的，策略已经漂移。

**当前状态**：token_level + clip_ratio [0.8, 1.28] 把每步漂移约束在 clip 边界内，training step 8 数据显示稳定。

**如需彻底解决**：`importance_sampling_fix: True` 或每步重算 old_logprob，代价是每个 optimizer step 多一次 forward。

---

## 可优化的点

| 优先级 | 优化项 | 预期收益 |
|---|---|---|
| 高 | **安装 `flash-linear-attention`**：`Qwen3_5GatedDeltaNet` 用 fast path 替代 PyTorch fallback | 消除跨进程 forward 路径差异，可能把 clip_log_ratio 放宽到 ±5 甚至关掉 |
| 高 | **移植 `importance_sampling_fix`** 到 embodied run_training | actor 在 eval 模式重算 prev_logprobs，ratio 从根本上修复，无需 clip hack |
| 中 | **修复 masked_mean 64x 缩放**：传入 expand 后的 per-token mask | loss/lr scale 正确，可适当增大 lr |
| 中 | **加入 response_mask AND loss_mask** 去掉 pad 贡献 | 梯度信号纯净，样本效率提升约 2x（在 resp_len < max_new_tokens 的情形下） |
| 低 | **移植 RLinf 官方版 WeightSyncer**（替换 bucket-based fork） | 支持版本管理、增量同步；当前 bucket 版本功能够用 |
| 低 | **还原 gradient_checkpointing: True**（配合 flash-linear-attention） | 节省显存，可增大 batch size |

---

## 关键文件修改列表

| 文件 | 修改内容 |
|---|---|
| `examples/embodiment/config/genark_grpo_qwen.yaml` | `logprob_type: token_level`；新增 `clip_log_ratio_min/max: ±2.0` |
| `rlinf/workers/actor/fsdp_actor_worker.py` | embodied `run_training` 的 `policy_loss` 调用加 `clip_log_ratio_min/max` 参数 |
| `rlinf/models/embodiment/qwen_nav/qwen_nav_policy.py` | 三处 pixel_values 统一用 `.to(self._dtype)` (bf16)：`_batch_generate`、`_compute_teacher_forcing_logprobs`、`default_forward` |

---

## 训练运行命令

```bash
cd /home/clk/workspace/RLinf

# LoRA 微调（当前配置，推荐）
GPUS=0,1,4,5 bash scripts/run_qwen_rft.sh lora

# Smoke（~10min 快速验证）
GPUS=0,1,4,5 bash scripts/run_qwen_rft.sh smoke
```

日志：`/home/clk/workspace/results/genark_grpo_qwen/rft_YYYYMMDD_HHMMSS_lora.log`
Tensorboard：`/home/clk/workspace/results/genark_grpo_qwen/tensorboard/`

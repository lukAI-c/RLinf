# QwenNav GRPO ratio 爆炸：根因诊断与修复

## 背景

本文档描述一个在 RLinf 框架下对 Qwen3.5-4B VLM 进行 GRPO（具身导航任务）强化学习微调时，训练崩溃问题的完整排查过程，供后续 AI/工程师继续跟进。

---

## 问题现象

```
train/actor/ratio:     [(0, 6.3e+20), (1, 2.7e+4), (2, 9.1e+32), (3, 1.3e-07)]
train/actor/approx_kl: [(0, 58.1), (1, 55.3), (2, 55.0), (3, 57.3)]
train/actor/grad_norm: [(0, inf), (1, 4.87e+8), (2, ~0), (3, ~0)]
```

- `actor/ratio` 在 1e-7 ～ 1e+32 之间剧烈震荡（正常应≈1.0）
- `approx_kl ≈ 55`（正常 step 0 时应 ≈ 0）
- `grad_norm` 先 inf 后被 clip 压到 0 → 模型实际上没在学习

---

## 项目结构说明

```
/home/nvme03/lck/RLinf/                    # RLinf 框架（本地 fork）
  rlinf/workers/actor/fsdp_actor_worker.py  # actor 训练逻辑
  rlinf/workers/rollout/hf/huggingface_worker.py  # rollout worker
  rlinf/models/embodiment/qwen_nav/
    qwen_nav_policy.py                      # 核心 Policy：rollout + training forward

/home/nvme03/lck/RLinf/examples/embodiment/
  train_embodied_agent.py                   # 训练入口
  config/genark_grpo_qwen.yaml             # 主配置

/home/clk/workspace/RLinf/
  scripts/run_qwen_rft.sh                  # 启动脚本（smoke/lora/full 三种模式）
  Questions/                               # 问题文档目录

/home/clk/workspace/model_zoo/Qwen/Qwen3.5-4B  # 模型权重
/home/clk/workspace/results/genark_grpo_qwen/   # 训练日志 + tensorboard
```

---

## 诊断过程（按时间顺序）

### 阶段一：错误假设 — GradientCheckpointingLayer

**假设**：Qwen3.5 的 `Qwen3_5GatedDeltaNet` 继承 `GradientCheckpointingLayer`，在 `model.train()` 时走 checkpointed forward，在 `model.eval()` 时走普通 forward，两者输出不一致。

**尝试修复**：在 `genark_grpo_qwen.yaml` 中设置：
```yaml
actor:
  fsdp_config:
    gradient_checkpointing: False
```

**结果**：ratio 仍然爆炸，假设被推翻。

---

### 阶段二：排除 train/eval 模式差异

在 `fsdp_actor_worker.run_training` 第一个 micro-batch 前，同时运行 actor 的 eval forward 和 train forward：

```
train vs eval   : max|d|=0.0000   ← 完全相同！
eval  vs rollout: max|d|=12.57
train vs rollout: max|d|=12.57
```

**结论**：train 和 eval mode 在数值上完全等价（attention_dropout=0，无 self.training 分支影响）。问题不是 train/eval mode。

---

### 阶段三：定位到 rollout 侧 — 两条路径不一致

在 rollout 端（`qwen_nav_policy.predict_action_batch`）加 DIAG，对同一份 `single_inputs`，分别调用：

- `_compute_teacher_forcing_logprobs(single_inputs, resp_ids)` → `tf_lp`（保存为 prev_logprobs）
- `default_forward(fi)` 其中 fi = `_build_forward_inputs_for_env(single_inputs, resp_ids)`→ `df_lp`（actor 训练时重算的 logprobs）

**结果**：
```
tf_lp[:3]       = [-0.230, -4.4e-5, -2.0e-5]   ← 概率高，model 自己生成的
default_fwd[:3] = [-20.89, -3.900, -5.081]       ← 概率极低，差 20+ nat
|diff| max=20.67
```

**关键发现**：同一台机器、同一个模型对象、同一份 prompt/response，两个函数给出完全不同的 logprobs。

---

### 阶段四：找到根因 — pixel_values dtype 不一致

对比两个函数传给 `self.model()` 的实际参数：

```python
# _compute_teacher_forcing_logprobs（rollout 产生 prev_logprobs）
model_kwargs["pixel_values"] = prompt_pixel_values.to(dev)        # fp32（processor 默认输出 fp32）

# default_forward（actor 训练时重算）
model_kwargs["pixel_values"] = flat_pv.to(self._dtype)            # bf16（强转）
```

- Processor 输出的 pixel_values 是 **fp32**
- `_compute_teacher_forcing_logprobs` 只做 `.to(device)`，保持 fp32
- `default_forward` 做 `.to(self._dtype)` = bf16
- `_batch_generate`（autoregressive 生成）也用 fp32

**fp32 vs bf16 的视觉 encoder 计算产生不同的 image embeddings → 影响所有 image-attended response token 的 logit → 前几个 token 差 20+ nat → ratio 爆炸**

验证：将两路都改为 bf16 后 diff=0，说明 dtype 一致即可消除偏差。正确方向是统一为 fp32（与 generation 对齐）。

---

## 修复方案

### 文件：`qwen_nav_policy.py`

**第一处**（`_compute_teacher_forcing_logprobs`，约 line 551）：保持 fp32，不动。

**第二处**（`default_forward`，约 line 995）：去掉 `.to(self._dtype)` 强转：

```python
# 修复前
model_kwargs["pixel_values"] = flat_pv.to(self._dtype)

# 修复后
model_kwargs["pixel_values"] = flat_pv
```

这样三条路径（generation / teacher-forcing / default_forward）均使用 fp32 pixel_values，图像 embedding 数值一致，teacher-forcing 的 prev_logprobs 和 actor 训练的 new_logprobs 在 step 0 时相同 → ratio≈1.0。

---

## 其他相关问题（已修复，本次不是根因）

以下问题在同一调试过程中发现并修复，但不是 ratio 爆炸的根因：

| 问题 | 状态 |
|---|---|
| `gradient_checkpointing: True` 导致 GatedDeltaNet 两路径数值偏差 | 已设 False（但不是根因） |
| response 长度超过 max_new_tokens 时 teacher-forcing 不截断（full_ids 比 default_forward 多 39 token） | 因果 attention 保证前 32 token logit 不受影响，暂无问题 |
| `_assign_episodes_to_envs` 未按 group_size 分组 | 已修复 |
| `genark_env.reset()` 不重置 `_exhausted` | 已修复 |

---

## 遗留问题

### 1. response 长度不截断

`_compute_teacher_forcing_logprobs` 接受的 `resp_ids` 可能超过 `max_new_tokens`（如 71 > 32），此时：
- TF: full_ids = prompt(1612) + response(71) = 1683
- DF: full_ids = prompt(1612) + response(32) = 1644

虽然因果 attention 保证两者前 32 个 response logprob 数值相同（已验证），但建议在 TF 中也截断 response：

```python
# _compute_teacher_forcing_logprobs 开头加：
if response_ids.shape[1] > self.max_new_tokens:
    response_ids = response_ids[:, :self.max_new_tokens]
```

### 2. pixel_values dtype 长期方案

当前修复是去掉强转。更根本的方案是在 rollout 初始化时明确设置 processor 输出 dtype：

```python
# 在 processor 调用时统一指定精度
inputs = self.processor(...).to(device=dev, dtype=self._dtype)
```

但这需要确保 generation 和 teacher-forcing 全链路一致，改动范围更大，暂缓。

---

## 运行方式

### 环境

```bash
# Python
/home/clk/miniconda3/envs/genesis/bin/python

# GPU 分配（建议至少 2 卡）
GPUS=1,5   # GPU 1: actor, GPU 5: rollout+env
```

### 启动训练

```bash
cd /home/clk/workspace/RLinf

# smoke 模式（~5min，快速验证 pipeline）
GPUS=1,5 bash scripts/run_qwen_rft.sh smoke

# lora 模式（LoRA 微调，验证 ratio 是否正常）
GPUS=1,5 bash scripts/run_qwen_rft.sh lora

# 完整训练
GPUS=1,5 bash scripts/run_qwen_rft.sh full

# 从 checkpoint 继续
RESUME_DIR=/path/to/ckpt GPUS=1,5 bash scripts/run_qwen_rft.sh lora
```

日志输出到：`/home/clk/workspace/results/genark_grpo_qwen/rft_YYYYMMDD_HHMMSS_<mode>.log`

### 验证修复是否成功

训练跑完第一个 epoch（lora 模式约 35min）后，查看 tensorboard：

```bash
tensorboard --logdir /home/clk/workspace/results/genark_grpo_qwen/tensorboard
```

期望值：
- `train/actor/ratio` ≈ 1.0（±0.3 之内）
- `train/actor/approx_kl` < 1.0
- `train/actor/grad_norm` < 10

如果上述指标正常，说明 pixel_values dtype 修复有效。

---

## 关键代码位置

| 功能 | 文件 | 行号（约） |
|---|---|---|
| rollout 产生 prev_logprobs | `qwen_nav_policy.py` | `_compute_teacher_forcing_logprobs` ~L488 |
| rollout 保存 forward_inputs | `qwen_nav_policy.py` | `_build_forward_inputs_for_env` ~L586 |
| actor 训练 forward（**修复点**） | `qwen_nav_policy.py` | `default_forward` ~L995 `model_kwargs["pixel_values"] = flat_pv` |
| actor 训练循环 | `fsdp_actor_worker.py` | `run_training` ~L1335 |
| 主配置 | `config/genark_grpo_qwen.yaml` | — |
| 启动脚本 | `scripts/run_qwen_rft.sh` | — |

---

## 当前状态（2026-05-10）

- [x] 根因确认：`default_forward` 中 pixel_values 被强转为 bf16，与 teacher-forcing（fp32）不一致
- [x] 修复已提交：去掉 `flat_pv.to(self._dtype)`
- [ ] **待验证**：lora 模式 ratio 是否恢复正常（训练中，约 35min 后查看 tensorboard）
- [ ] 若 ratio 正常，清理 markdown 中的假设章节，更新最终根因记录
- [ ] 考虑修复 response 长度截断问题（低优先级）

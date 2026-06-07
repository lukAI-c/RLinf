# Decision-Level Rollout Problems

## 背景

QwenNav 的每次 LLM 推理会输出一个宏动作序列（例如"navigate to behind"对应 7 个低层 env step）。
原始的 step-based rollout 按固定 env step 数（`max_steps_per_rollout_epoch=40`）终止，导致：

1. 宏动作语义错误：推理产生 7 个 action 但只执行第 1 个
2. 无法按推理次数控制 rollout 长度
3. rollout_batch 包含大量无意义的非决策步数据

为此引入了 **decision-level rollout**：设定 `max_decisions_per_rollout_epoch`，每个 env 完成 N 次 LLM 推理后终止，rollout_batch 的 dim-0 从 env_step 变为 decision。

---

## Rollout 机制：episode、dormant env 与 GRPO group 约束

### episode 是什么

在 GenArk/QwenNav 导航任务里，episode 是一条具体的导航任务样本：

```
scene/map + start pose + goal/instruction + success condition + max steps
```

日志中的：

```
[GenArk] Episode done  env=10  ep=140  success=0  SPL=0.000  DTG=12.92m  steps=21
```

表示 env 10 刚跑完 episode 140：没有成功，最后离目标 12.92m，共执行 21 个 env step。

### dormant env 的触发条件

dormant env 不是 rollout 一开始就出现，而是在当前 scene/worker 分配到的 episode 池被跑完后出现。

典型日志：

```
[GenArk] Worker exhausted: scene='mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb' all 12 episodes done. Entering dormant mode.
```

含义是：

```
这个 worker 当前固定在 scene 2azQ1b91cZZ
该 scene 分给它的 12 个 episode 都已经 done
没有新的 episode 可以 reset
于是剩余 env slot 进入 dormant mode
```

代码侧通常通过空任务描述判断 dormant：

```python
task_descs = env_output.obs.get("task_descriptions", None)
env_dormant = torch.tensor([desc == "" for desc in task_descs])
```

也就是 `task_description == ""` 表示这个 env slot 当前没有真实任务。

### 为什么不能简单 auto reset

GRPO 的 `group_size=3` 要求同一个 group 内是同一个 prompt/task 的多条采样轨迹，用于组内 reward/advantage 比较。例如：

```
group 0:
  env0 = episode 330 的 sample A
  env1 = episode 330 的 sample B
  env2 = episode 330 的 sample C

group 1:
  env3 = episode 403 的 sample A
  env4 = episode 403 的 sample B
  env5 = episode 403 的 sample C
```

如果 rollout 中途允许普通 auto reset，可能变成：

```
group 0:
  env0 前半段是 episode 330，后半段 reset 到 episode 999
  env1 仍是 episode 330
  env2 仍是 episode 330
```

这样 group 内就不再是同一个任务的多条采样，GRPO 的组内比较语义被破坏。因此当前设计里，episode done 后不能随意切到新 episode；当该 worker 的 episode 池耗尽时，只能进入 dormant，而不是无条件 reset。

### 与 max_dec 的关系

`max_decisions_per_rollout_epoch=max_dec` 表示希望每个 env 收集最多 `max_dec` 次 LLM decision。

这不是说 `max_dec` 本身一定太大，而是它和 episode 生命周期之间存在约束：

```
如果 episode 在收满 max_dec 个 decision 前结束，
且不能 auto reset 到新 episode，
那么该 env 后续只能变成 blank/dormant padding。
```

因此 `max_dec` 越大，越容易暴露 episode 提前 done / episode 池耗尽导致的 dormant 问题。当前矛盾是：

```
GRPO 要固定 group 语义，不能中途随意 reset episode
decision-level rollout 又希望每个 env 收满 max_dec 个 decision
但当前 scene/worker 的 episode 池可能不足以支撑所有 env 收满
```

---

## 问题一：FSDP 跨 Rank rollout_size 不对称 → NCCL 死锁

### 根本原因

FSDP actor 将 12 个 env 的 rollout_batch 按 `actor_split_num=2` 分给两个 rank：
- rank 0：env 0–5
- rank 1：env 6–11

`_align_rollout_batch_to_groups` 会丢弃所有 env 的 `loss_mask` 全为 False 的 group。

**step-based rollout 下不会死锁的原因：**
每个 env 强制跑满 40 步，即使 dormant env 也会写入 40 条数据。`loss_mask` 是"部分 True/部分 False"——只要一个 group 里有任意一步有效，`group_valid = loss_mask.any() = True`，group 不会被整体丢掉。两个 rank 几乎保留全部 group，rollout_size 对称。

**decision-level rollout 下为什么死锁：**
dormant env 一次 LLM 推理都没有，全部 `max_dec` 条都是 blank padding（`dones=True → loss_mask=False`）。如果一个 group 的 3 个 env 全部 dormant，`group_valid = False`，整个 group 被丢掉。

由于两个 rank 的 env 分布不同，可能出现：

```
rank 0: envs 0–5 中恰好有一个 group 全 dormant → group-align kept=3 → rollout_size = 20×3 = 60
rank 1: envs 6–11 全部活跃                    → group-align kept=6 → rollout_size = 20×6 = 120
```

进入 `_recompute_prev_logprobs_embodied`：

```
rank 0: truncated=60  → num_mbs=60  → 60 次 FSDP _ALLGATHER_BASE
rank 1: truncated=120 → num_mbs=120 → 120 次 FSDP _ALLGATHER_BASE
```

rank 0 先完成 recompute，到达 `run_training` 里的 `all_reduce_int(local_usable_rollout_size)`（标量 ALLREDUCE）；rank 1 还困在第 61 次 FSDP forward 的 ALLGATHER 里。两者在不同类型的 collective 上互相等待，触发 NCCL 30 分钟超时。

### 表现

```
[FSDPActor][group-align] B=6 -> kept=3 (tail-drop+dormant-group). groups: 2 -> 1
[rank0]: Watchdog caught collective operation timeout:
    WorkNCCL(SeqNum=22337, OpType=ALLREDUCE, NumelIn=1, ...)
    ran for 1800025 milliseconds before timing out.
```

注意 `NumelIn=1` 是标量 all-reduce，正是 `run_training` 里新加的同步点——但它永远等不来 rank 1，因为 rank 1 还在 recompute 的 FSDP forward 循环里。

### 修复

在 `_recompute_prev_logprobs_embodied` 的 micro-batch 循环**之前**，用 `all_reduce_int(MIN)` 同步 `truncated`，保证两 rank 进入循环时 `num_mbs` 完全相同：

```python
# fsdp_actor_worker.py — _recompute_prev_logprobs_embodied
truncated = (rollout_size // micro_bsz) * micro_bsz

if self._world_size > 1:
    truncated_synced = all_reduce_int(truncated)   # MIN
    if truncated_synced < rollout_size:
        # Trim rollout_batch so run_training also sees consistent rollout_size.
        self.rollout_batch = process_nested_dict_for_train(
            self.rollout_batch, torch.arange(truncated_synced)
        )
        rollout_size = truncated_synced
    truncated = truncated_synced

num_mbs = truncated // micro_bsz   # 两 rank 现在完全相同
for mb in range(num_mbs):          # FSDP forward 次数一致，不再死锁
    ...
```

rank 1 多出的 60 个 sample 被丢弃。由于这些 sample 来自 blank-padded dormant env（`loss_mask=False`），不参与 PPO loss 计算，正确性不受影响，仅有少量数据利用率损失。

---

## 问题二：blank padding 导致 recompute valid 比例低

### 表现

```
[FSDPActor][recompute] valid=20/60 dormant=40
```

rank 0 的 60 个 sample 中只有 20 个有效（1 个活跃 group × 20 decisions），另外 40 个是 dormant env 的 blank padding。

### 原因

`_align_rollout_batch_to_groups` 只丢弃整 group，但 group 内的 dormant env 的 blank steps 仍然保留在 rollout_batch 里（loss_mask=False）。这些样本进入 recompute 但不参与 loss，浪费了 forward 计算资源。

### 影响

当前版本接受这一开销。如需优化，可在 recompute 前过滤掉 `loss_mask` 全为 False 的样本（需同时确保两 rank 过滤后数量相同，或再次 sync truncated）。

---

## 问题三：lora_dropout 导致 ratio ≠ 1.0

### 表现

```
actor/ratio=0.844   （应为 ~1.000）
actor/total_loss=nan
```

### 原因

`recompute_prev_logprobs=True` 时，actor 在训练前用 `model.eval()` 重算 logprobs（ratio 应为 1.0），但 `lora_dropout=0.05` 使 eval 模式和 train 模式的输出不同。重算 logprobs（eval）和 PPO 前向（train）的 log-prob 值有偏差，ratio 偏离 1.0，导致 clip 失效、loss nan。

### 修复

```yaml
actor:
  model:
    lora_dropout: 0.0   # recompute 时 eval/train 模式输出一致
```

或在 lora mode 的启动脚本中覆盖：

```bash
"actor.model.lora_dropout=0.0"
```

---

## 问题四：masked_mean_ratio 对 padding 样本除零 → NaN

### 表现

blank padding 样本的 `loss_mask_ratio == 0`，`masked_mean_ratio` 直接用 `values / loss_mask_ratio` 触发除零，结果为 NaN，污染整个 loss。

### 修复

```python
# rlinf/utils/utils.py
def masked_mean_ratio(values, mask, loss_mask_ratio):
    safe_ratio = torch.clamp(loss_mask_ratio, min=1e-6)
    scaled_values = values / safe_ratio
    scaled_values = torch.where(mask.bool(), scaled_values, torch.zeros_like(values))
    return scaled_values.mean()
```

---

## 问题五：per-env pending_data 里 actions 用了整个 batch 的 slice

### 表现

`ratio` 在某些 micro-batch 上异常偏低，分析 per-env 数据发现 action token 错位。

### 原因

env_worker 在 flush decision 时，从 `rollout_result.actions` 取 per-env slice：

```python
# 错误写法（actions 是整 batch，不是 per-env 推理的那个）
actions=rollout_result.actions[env_i:env_i+1]
```

对于走了 pending_actions 路径（非推理步）的 env，正确的 action 应从 `env_fi["action"]` 取：

```python
# 修复
actions=env_fi["action"] if "action" in env_fi else rollout_result.actions[env_i:env_i+1]
```

---

## 修改文件汇总

| 文件 | 改动 |
|---|---|
| `rlinf/data/embodied_io_struct.py` | `RolloutResult` 新增 `is_decision` 字段 |
| `rlinf/models/embodiment/qwen_nav/qwen_nav_policy.py` | 开放训练模式 pending_actions；返回 `is_decision` |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | `generate_one_epoch` 改为 decision-count while 循环 |
| `rlinf/workers/env/env_worker.py` | per-env EmbodiedRolloutResult；acc_rewards；blank padding 到 max_dec |
| `rlinf/workers/actor/fsdp_actor_worker.py` | recompute 前 `all_reduce_int(MIN)` 同步 truncated；run_training 后 sync usable_rollout_size |
| `rlinf/utils/utils.py` | `masked_mean_ratio` clamp + torch.where 防除零 |
| `examples/embodiment/config/genark_grpo_qwen.yaml` | 新增 `max_decisions_per_rollout_epoch`；`lora_dropout=0.0` |
| `scripts/run_qwen_rft.sh` | lora mode 同步参数；NCCL timeout 扩展到 1h |

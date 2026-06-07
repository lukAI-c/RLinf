# Decision-Level Rollout 设计文档

## 一、核心概念

### 1.1 宏动作（Macro Action）

QwenNav 每次 LLM 推理输出一个宏动作指令（如 `"navigate to behind"`），
由底层控制器展开为多个低层 env step（如 7 步）。

```
1次 LLM 推理
    → 宏动作指令 "navigate to behind"
        → env step 1 (前进)
        → env step 2 (前进)
        → env step 3 (转向)
        → env step 4 (前进)
        → env step 5 (前进)
        → env step 6 (前进)
        → env step 7 (到达)
```

### 1.2 两种 Rollout 模式对比

```
【Step-based Rollout（改前）】
时间轴: t=1  t=2  t=3  t=4  t=5  t=6  t=7  ...  t=40
         infer exec exec exec exec exec exec      exec
         ↑推理  ↑非决策步（执行宏动作缓冲）

rollout_batch shape: [40, B, ...]  ← 每个 env step 一行

问题:
  - 非决策步（exec）也触发 LLM 推理（训练模式下 pending_actions 被禁用）
  - 宏动作语义被破坏：推理产生 7 步但只执行第 1 步
  - 大量非决策步占用 rollout_batch，训练信号稀疏

【Decision-level Rollout（改后）】
时间轴: d=1        d=2        d=3   ...  d=20
         infer      infer      infer      infer
         ↓          ↓          ↓          ↓
        7 env steps 5 env steps 3 env steps ...

rollout_batch shape: [20, B, ...]  ← 每次 LLM 推理一行
reward: 每个 decision 对应的累积 reward（7步之和）
```

---

## 二、系统组件与数据流

### 2.1 组件全景

```
┌─────────────────────────────────────────────────────────────────┐
│                         训练循环                                  │
│                                                                   │
│  ┌──────────┐     obs      ┌──────────────┐   action  ┌───────┐  │
│  │          │◄────────────│              │◄──────────│       │  │
│  │  Env     │             │   Rollout    │           │ Actor │  │
│  │ Worker   │────────────►│   Worker     │──────────►│       │  │
│  │ (GPU 7)  │  reward/done│  (GPU 6)     │ rollout   │(GPU   │  │
│  │          │             │              │ _result   │ 4,5)  │  │
│  └──────────┘             └──────────────┘           └───────┘  │
│       │                         │                        │        │
│  12 envs                  LLM 推理                  FSDP 2-rank  │
│  (group_size=3)           pending_actions            recompute   │
│  acc_rewards              is_decision flag           PPO update  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键数据结构

```
RolloutResult（rollout worker → env worker）
  actions:        [B, action_dim]     ← 本步动作
  is_decision:    [B]  bool           ← True=新推理, False=执行缓冲动作
  prev_logprobs:  [B, action_dim]     ← 若 skip_rollout_logprobs=True 则全零
  forward_inputs: dict                ← 图像/文本输入（供 actor recompute）
  versions:       [B]                 ← rollout 版本号

RolloutBatch（env worker → actor）
  shape: [max_dec, B, ...]            ← max_dec=20, B=12
  rewards:        [20, 12, 1]         ← 累积 reward（每次推理的多步之和）
  dones:          [21, 12, 1]         ← T+1 项（含 bootstrap done）
  prev_logprobs:  [20, 12, action_dim]
  forward_inputs: [20, 12, ...]
  loss_mask:      [20, 12]            ← blank/dormant 步为 False
```

---

## 三、各组件改动详解

### 3.1 Rollout Worker（huggingface_worker.py）

**改前**：`for _ in range(n_train_chunk_steps)` 固定循环 40 次

**改后**：`while` 循环，按 decision 计数终止

```
初始化:
  decision_counts[stage][env] = 0   ← 每个 env 的推理次数

每步:
  1. 调用 policy.predict(obs)
     → policy 内部: pending_actions 有缓冲? 取缓冲(is_decision=False)
                                              否则    LLM 推理(is_decision=True)
  2. is_decision[env]=True → decision_counts[env] += 1
  3. 发送 RolloutResult(actions, is_decision, forward_inputs, ...)

终止条件:
  所有 env 的 decision_counts >= max_dec
  OR 所有 env 都 dormant（episode 池耗尽）
  OR safety_steps = max_dec × 10 + pipeline_stages 触发
```

### 3.2 QwenNav Policy（qwen_nav_policy.py）

**关键改动**：训练模式也启用 `pending_actions`

```
Phase 1 - Dispatch（每个 env 决定本步行为）:

  改前:
    if not self.collect_forward_inputs and cache.pending_actions:
        actions[i] = cache.pending_actions.pop(0)  ← 仅 eval 模式走这里
    else:
        need_infer.append(i)  ← 训练模式强制推理

  改后:
    if cache.pending_actions:
        actions[i] = cache.pending_actions.pop(0)  ← 训练/eval 均走
        is_decision_per_env[i] = False
    else:
        need_infer.append(i)
        is_decision_per_env[i] = True

Phase 2 - Macro Buffering（推理完成后缓冲剩余动作）:

  改前:
    if not self.collect_forward_inputs and len(act_seq) > 1:
        cache.pending_actions.extend(act_seq[1:])  ← 仅 eval 模式

  改后:
    if len(act_seq) > 1:
        cache.pending_actions.extend(act_seq[1:])  ← 训练/eval 均缓冲

返回:
  result["is_decision"] = torch.tensor(is_decision_per_env, dtype=torch.bool)
```

### 3.3 Env Worker（env_worker.py）

**核心变化**：从 batch 级别 append → per-env 独立轨迹 + 累积 reward

```
数据结构:
  rollout_results_per_env[stage_id][env_i] = EmbodiedRolloutResult()
  acc_rewards[stage_id][env_i]    ← 累积 reward（CPU tensor）
  acc_dones[stage_id][env_i]      ← 累积 done 标志
  decision_counts[stage_id][env_i] ← 每 env 的 decision 次数

每步处理:
  for each env_i:
    acc_rewards[env_i] += step_reward     ← 无论是否 decision 都累积

    if is_decision[env_i] and decision_counts[env_i] < max_dec:
        ─── flush 本次 decision 数据 ───
        append ChunkStepResult(
            rewards  = acc_rewards[env_i],   ← 多步累积
            dones    = acc_dones[env_i],
            actions  = env_fi["action"],     ← per-env 正确切片
            forward_inputs = env_fi,
            ...
        )
        acc_rewards[env_i]  = 0             ← 重置累积
        acc_dones[env_i]    = False
        decision_counts[env_i] += 1

Bootstrap 步（rollout 结束时）:
  仅写 dones（无 rewards）
  → 保证 rewards 共 T 项，dones 共 T+1 项

Padding（每 env 补到 max_dec 项）:
  ┌─────────────────────────────────────────────┐
  │ 有真实 decision 的 env: 用自己最后一条作模板 │
  │ 全 dormant 的 env:     用 stage_ref 作模板   │
  └─────────────────────────────────────────────┘
  blank step:
    dones         = True    → compute_loss_mask → loss_mask=False
    terminations  = True
    truncations   = False
    rewards       = 0
    actions/forward_inputs/prev_logprobs/versions = clone(ref)

  assert: 所有 env 的 rewards/actions/forward_inputs 长度均 == max_dec

Merge（per-env → batch）:
  n_steps = int(max_dec)   ← 固定，不再 min(...)
  for t in range(n_steps):
      merged.rewards[t] = cat([r.rewards[t] for r in per_env_results])
      ...
  shape → [max_dec, B, ...]
```

### 3.4 FSDP Actor（fsdp_actor_worker.py）

#### 3.4.1 _align_rollout_batch_to_groups

```
改前（会造成两 rank rollout_size 不对称）:
  1. 计算每个 env 的 loss_mask.any() → ep_valid[B]
  2. 按 group_size=3 分组
  3. 丢弃 loss_mask 全 False 的 group（dormant group）
  4. 结果: rank0 kept=3, rank1 kept=6 → rollout_size 不同

改后（方案 A）:
  1. 仅做 tail-drop（让 B 是 group_size 的倍数）
  2. dormant group 保留 → loss_mask=False → loss=0
  3. 结果: rank0 B=6, rank1 B=6 → rollout_size 恒等
```

#### 3.4.2 _recompute_prev_logprobs_embodied

```
改前:
  truncated = (rollout_size // micro_bsz) * micro_bsz
  # rank0: truncated=60, rank1: truncated=120
  for mb in range(num_mbs):         ← 次数不同 → NCCL 死锁！
      FSDP forward (→ _ALLGATHER_BASE)

改后（安全网，方案 A 后理论上 truncated 已对称）:
  truncated = (rollout_size // micro_bsz) * micro_bsz
  if world_size > 1:
      truncated = all_reduce_int(truncated)   # MIN
      if truncated < rollout_size:
          rollout_batch = trim(rollout_batch, truncated)
  for mb in range(num_mbs):         ← 两 rank 次数相同 → 不死锁
      FSDP forward (→ _ALLGATHER_BASE)
```

#### 3.4.3 run_training（usable_rollout_size 同步）

```
local_usable = (rollout_size // batch_size_per_rank) * batch_size_per_rank
usable_rollout_size = all_reduce_int(local_usable)   # MIN
← 兜底：若两 rank rollout_size 仍有微小差异，取较小值对齐训练步数
```

---

## 四、NCCL 死锁的成因与修复

### 4.1 死锁时序图（改前）

```
rank 0 (kept=3, rollout_size=60)     rank 1 (kept=6, rollout_size=120)
──────────────────────────────        ──────────────────────────────────
recompute: loop 60 次                 recompute: loop 120 次
  mb[0]: ALLGATHER ←─同步─→ mb[0]: ALLGATHER
  mb[1]: ALLGATHER ←─同步─→ mb[1]: ALLGATHER
  ...
  mb[59]: ALLGATHER ←─同步→ mb[59]: ALLGATHER
  ← 完成，退出 recompute ←          mb[60]: ALLGATHER ← rank 1 还在这
                                      mb[61]: ALLGATHER
run_training:                         ...
  all_reduce_int(48)                  mb[119]: ALLGATHER
  ↑ ALLREDUCE(NumelIn=1)              ↑ ALLGATHER
  两者类型不同，互相等待
  ──── 30分钟后 NCCL Watchdog 超时 ────
```

### 4.2 修复后时序图（改后）

```
rank 0 (B=6, rollout_size=120)       rank 1 (B=6, rollout_size=120)
────────────────────────────          ────────────────────────────────
recompute:                            recompute:
  all_reduce_int(120) ←─MIN─→ all_reduce_int(120) = 120（对称，no-op）
  loop 120 次                         loop 120 次
    mb[k]: ALLGATHER ←─同步─→ mb[k]: ALLGATHER   ← 每步都对齐
  ← 同时完成 ←                        ← 同时完成 ←

run_training:                         run_training:
  all_reduce_int(96) ←─MIN─→ all_reduce_int(96) = 96（no-op）
  PPO update                          PPO update
  ──── 正常训练 ────
```

---

## 五、Reward 累积与 loss_mask

### 5.1 Reward 累积

```
时间轴（env_i 视角）:

env_step:   1    2    3    4    5    6    7    8    9    10
is_dec:     T    F    F    F    F    F    F    T    F    F
reward:    0.1  0.0  0.0  0.0  0.1  0.2  0.0  0.5  0.1  0.0

acc_reward: 0.1→0.1→0.1→0.1→0.2→0.4→0.4  flush=0.4  0.5→0.6→0.6

decision rollout_batch:
  t=0: reward=0.4  (7步累积, is_dec=True)
  t=1: reward=0.6  (3步累积, is_dec=True)
  ...

step-based rollout_batch（对照）:
  t=0: reward=0.1  t=1: reward=0.0  t=2: reward=0.0  ...（稀疏）
```

### 5.2 loss_mask 计算

```
dones 序列（T+1 项）:
  d[0]  d[1]  d[2]  ... d[T-1]  d[T]
  ↑实际步         ↑              ↑bootstrap done

loss_mask[t] = True  当且仅当 dones[t] == False（本 decision 前 episode 未结束）

blank padding 步: dones=True → loss_mask=False → 不参与 loss
dormant env 的所有步: 全部 dones=True → loss_mask 全 False
```

---

## 六、Padding 策略

```
场景示意（max_dec=5, group_size=3）:

env 0: [d0, d1, d2, d3, d4]          ← 5个真实 decision，无需 padding
env 1: [d0, d1, ─, ─, ─]             ← episode 第2步 done，补3个 blank
env 2: [─, ─, ─, ─, ─]               ← 全 dormant，借 stage_ref 补5个 blank

group 0 = {env0, env1, env2}
  env0: loss_mask = [T, T, T, T, T]
  env1: loss_mask = [T, F, F, F, F]  ← done 之后屏蔽
  env2: loss_mask = [F, F, F, F, F]  ← 全屏蔽（全 dormant）
  → group_valid = loss_mask.any() = True  ← 方案 A: 无论如何保留此 group

blank step 内容:
  rewards       = 0
  dones         = True   → 触发 loss_mask=False
  terminations  = True
  truncations   = False
  actions       = ref.clone()         ← 形状占位，loss_mask=False 屏蔽
  forward_inputs= ref.clone()         ← 形状占位，不参与 recompute loss
  prev_logprobs = ref.clone()         ← 形状占位
```

---

## 七、修改文件汇总

| 文件 | 核心改动 | 目的 |
|---|---|---|
| `rlinf/data/embodied_io_struct.py` | `RolloutResult` 新增 `is_decision: Tensor` | 标记每步是否为新推理 |
| `rlinf/models/embodiment/qwen_nav/qwen_nav_policy.py` | 训练模式开放 `pending_actions`；返回 `is_decision` | 宏动作语义正确执行 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | `for` → `while`，按 decision 计数终止 | 控制 rollout 长度为推理次数 |
| `rlinf/workers/env/env_worker.py` | per-env 轨迹；acc_rewards；blank padding 到 max_dec | 正确收集 decision-level 数据 |
| `rlinf/workers/actor/fsdp_actor_worker.py` | `_align_rollout_batch_to_groups` 仅 tail-drop；recompute 前 `all_reduce_int(MIN)` | 消除跨 rank 不对称和 NCCL 死锁 |
| `rlinf/utils/utils.py` | `masked_mean_ratio` clamp + torch.where | 防 blank 样本除零 NaN |
| `examples/embodiment/config/genark_grpo_qwen.yaml` | 新增 `max_decisions_per_rollout_epoch`；`lora_dropout=0.0` | 配置决策次数；recompute ratio≈1.0 |
| `scripts/run_qwen_rft.sh` | lora mode 同步参数；NCCL timeout=1h | 防超时崩溃 |

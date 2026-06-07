# PPT 汇报大纲：Decision-Level Rollout for Embodied Navigation RFT

---

## Slide 1 — 标题页

**Title**: Decision-Level Rollout for Embodied Navigation GRPO

**Subtitle**: QwenNav × GenArk：让 LLM 的每次推理真正对应一个训练样本

**内容要点**:
- 任务背景：VLM 导航智能体 + GRPO 强化微调
- 核心贡献：将 rollout 粒度从 env step 提升到 LLM decision

---

## Slide 2 — 背景：宏动作与训练粒度错位

**标题**: 问题：LLM 推理粒度 ≠ env step 粒度

**图示**（时间轴）:
```
1次 LLM 推理 → "navigate to behind"
               ↓
    step1  step2  step3  step4  step5  step6  step7
    (前进)  (前进) (转向)  (前进) (前进)  (前进) (到达)
```

**核心问题**（三点）:
1. 训练模式下 `pending_actions` 被禁用 → 每个 env step 强制触发一次 LLM 推理
2. 宏动作语义破坏：推理产生 7 步动作序列，但实际只执行第 1 步
3. rollout_batch 中充斥大量无意义的非决策步，训练信号极度稀疏

**对比数字**: rollout_batch `[40, B]` vs 实际有效推理 ~`[6, B]`

---

## Slide 3 — 方案：Decision-Level Rollout

**标题**: 解法：以 LLM 推理次数（decision）为 rollout 粒度

**对比图**（两行对比）:

```
【Step-based（改前）】
t: 1     2     3     4     5     6     7  ···  40
   infer exec  exec  exec  exec  exec  exec     exec
   ↑唯一推理步  ↑6个无意义的重复推理步

rollout_batch shape: [40, B, ...]   ← 大量冗余

【Decision-level（改后）】
d: 1           2           3      ···  20
   infer        infer        infer      infer
   ↓ 7 steps    ↓ 5 steps    ↓ 3 steps

rollout_batch shape: [20, B, ...]   ← 每行 = 一次真实决策
```

**关键设计**:
- `max_decisions_per_rollout_epoch = 20`：每 env 收集 20 次 LLM 推理
- `is_decision` flag：区分新推理（`True`）和宏动作执行（`False`）
- 每个 decision 的 reward = 宏动作期间所有 env step reward 的累积

---

## Slide 4 — 系统架构与数据流

**标题**: 三组件协同：Env Worker / Rollout Worker / FSDP Actor

**图示**（组件图）:
```
┌──────────────────────────────────────────────┐
│  Env Worker (GPU 7)                          │
│  • 12 envs, group_size=3                     │
│  • acc_rewards[env_i] 累积多步 reward         │
│  • is_decision flush → per-env trajectory    │
│  • blank padding → rollout_batch [20,12,...] │
└──────────┬───────────────────────────────────┘
           │ obs / reward / done
           ▼
┌──────────────────────────────────────────────┐
│  Rollout Worker (GPU 6)                      │
│  • pending_actions 训练模式启用               │
│  • while 循环：decision_counts 计数           │
│  • 返回 RolloutResult(actions, is_decision)  │
└──────────┬───────────────────────────────────┘
           │ rollout_batch [20, 12, ...]
           ▼
┌──────────────────────────────────────────────┐
│  FSDP Actor (GPU 4 + GPU 5, 2-rank)          │
│  • recompute prev_logprobs                   │
│  • GRPO advantage → PPO-clip update          │
└──────────────────────────────────────────────┘
```

**数据结构变化**:
- `rewards: [20, 12, 1]` ← 每 decision 累积 reward（多步之和）
- `loss_mask: [20, 12]` ← blank/dormant 步为 False，不参与 loss

---

## Slide 5 — 工程挑战一：FSDP 跨 Rank 不对称 → NCCL 死锁

**标题**: Bug：FSDP 2-rank 的 micro-batch 计数不对称导致死锁

**根因图示**（改前）:
```
_align_rollout_batch_to_groups:
  rank 0 (envs 0-5):  1个 group 全 dormant → 丢掉 → kept=3, rollout_size=60
  rank 1 (envs 6-11): 全部活跃             →       → kept=6, rollout_size=120

_recompute_prev_logprobs_embodied:
  rank 0: 60 次 FSDP _ALLGATHER_BASE
  rank 1: 120 次 FSDP _ALLGATHER_BASE   ← 次数不同！

  rank 0 先完成 → 到达 ALLREDUCE(NumelIn=1)
  rank 1 还在第 61 次 ALLGATHER
  → 两者在不同 collective 互等 → 30分钟后 NCCL Watchdog 超时
```

**为什么 step-based 没有这个问题**:
> step-based rollout 下，dormant env 也强制跑满 40 步，`loss_mask` 部分为 True，group 不会被整体丢掉 → 两 rank rollout_size 对称

---

## Slide 6 — 工程挑战一：双重修复

**标题**: 修复：从数据生产端消除不对称

**修复方案 A（主）— 保留 dormant group**:
```python
# _align_rollout_batch_to_groups（改后）
# 仅做 tail-drop，不再过滤 dormant group
n_full = (B // group_size) * group_size
keep_idx = torch.arange(n_full)   # 所有 group 保留
# dormant group: loss_mask=False → loss=0，不影响训练
```

**修复方案 B（安全网）— recompute 前同步**:
```python
# _recompute_prev_logprobs_embodied（改后）
truncated = all_reduce_int(truncated)   # MIN：两 rank 取较小值
if truncated < rollout_size:
    rollout_batch = trim(rollout_batch, truncated)
# 保证两 rank 进入循环时 num_mbs 完全相同
```

**效果对比**:

| | 改前 | 改后 |
|---|---|---|
| rank 0 rollout_size | 60 | 120 |
| rank 1 rollout_size | 120 | 120 |
| recompute 循环次数 | 60 ≠ 120 | 120 = 120 |
| 结果 | NCCL 死锁 | 正常训练 |

---

## Slide 7 — 工程挑战二：Blank Padding 与 loss_mask

**标题**: Dormant Env 的 Blank Padding 设计

**场景**（max_dec=5, group_size=3）:

```
env 0: [d0  d1  d2  d3  d4]   loss_mask: [T  T  T  T  T]  ← 5个真实 decision
env 1: [d0  d1  ██  ██  ██]   loss_mask: [T  F  F  F  F]  ← episode 第2步 done
env 2: [██  ██  ██  ██  ██]   loss_mask: [F  F  F  F  F]  ← 全 dormant
        ↑blank step: dones=True, rewards=0, forward_inputs=clone(ref)
```

**保证**:
- `dones=True` → `compute_loss_mask` → `loss_mask=False` → 不参与 PPO loss
- forward_inputs clone（而非零填充）→ 避免 pixel_values 零填充导致 ~23GB CPU 内存膨胀
- 所有 env 的 trajectory 长度严格 = `max_dec` → merge 时 `torch.cat` 不报 shape 错误

---

## Slide 8 — 工程挑战三：其他 Bug 修复

**标题**: 三个训练稳定性 Bug

**Bug 1：`lora_dropout=0.05` 导致 ratio ≠ 1.0**
- 原因：recompute 用 `model.eval()`，PPO 用 `model.train()`，dropout 状态不同 → logprob 有偏差
- 表现：`actor/ratio=0.844`（应为 ~1.000），`total_loss=nan`
- 修复：`lora_dropout: 0.0`

**Bug 2：`masked_mean_ratio` 除零 → NaN**
- 原因：blank padding 样本的 `loss_mask_ratio=0`，直接除零
- 修复：`safe_ratio = clamp(loss_mask_ratio, min=1e-6)` + `torch.where` 屏蔽无效位置

**Bug 3：per-env actions 切片错误**
- 原因：flush 时从全 batch 的 `rollout_result.actions` 取切片，pending_actions 路径下应取 `env_fi["action"]`
- 修复：`env_fi["action"] if "action" in env_fi else rollout_result.actions[env_i:env_i+1]`

---

## Slide 9 — 奖励函数设计

**标题**: Reward 设计：导航信号 + 格式奖励

**总体结构**:
$$r_d = r_d^{\text{nav}} + r_d^{\text{fmt}}$$

**Mode A（当前）— Geodesic Progress**:
$$r_d^{\text{geo}} = \mathcal{D}_{\text{geo}}(s_{t_{\text{start}}-1}) - \mathcal{D}_{\text{geo}}(s_{t_{\text{end}}}) + \mathcal{B}_{\text{succ}} \cdot \mathbf{1}[\text{STOP} \wedge d < \delta_{\text{succ}}]$$

**Mode B（可选）— Decision-Level nDTW**:
$$r_d^{\text{ndtw}} = \lambda_{\text{nDTW}} \cdot \Delta\eta_d + \lambda_{\text{SR}} \cdot \Delta\rho_d$$
$$\eta(P, \hat{P}) = \exp\!\left(-\frac{\text{DTW}(P, \hat{P})}{|\hat{P}| \cdot \delta_{\text{succ}}}\right)$$

- 路径 $P$ 含宏动作所有中间位置点（每 env step append）
- nDTW 每个 decision **只算一次**（flush 时），而非每 env step 都算后累加
- 数学等价（$\gamma=1$ 时 telescoping），但 fastdtw 调用 $K \to 1$ 次/decision

**Format Reward（冷启动）**:
$$r_d^{\text{fmt}} = \lambda_{\text{fmt}} \cdot \mathbf{1}[\text{JSON parse ok}], \quad \lambda_{\text{fmt}}=0.5$$

---

## Slide 10 — 实验与现状

**标题**: 训练状态与待验证项

**已解决**:
- ✅ 宏动作语义：训练模式 pending_actions 正确启用
- ✅ rollout_batch 粒度：`[40,B]` → `[20,B]`，每行=一次 LLM 推理
- ✅ FSDP NCCL 死锁：双重修复（保留 dormant group + recompute sync）
- ✅ NaN loss：除零修复 + lora_dropout=0.0
- ✅ Decision-level nDTW reward 接口实现

**待验证（GPU 空闲后）**:
- ⬜ 训练是否能跑完 1 个完整 epoch（目标：不再出现 NCCL 超时）
- ⬜ `actor/ratio ≈ 1.000`（recompute 正确）
- ⬜ DTG 随 epoch 下降趋势（导航学习信号有效）
- ⬜ `reward_mode="ndtw_sr_delta"` 切换后的训练稳定性

**当前瓶颈**:
- 共享服务器 GPU 4/5 被其他用户占用 → 测试需等待

---

## Slide 11 — 总结

**标题**: 贡献总结

**核心创新**:
1. **Decision-Level Rollout**：将 rollout 粒度从 env step 提升到 LLM inference，修复宏动作语义，消除无效训练样本
2. **FSDP 跨 Rank 同步**：双重机制（group 保留 + recompute sync）解决 decision-level 引入的 NCCL 死锁
3. **Decision-Level nDTW Reward**：nDTW 计算粒度与 LLM 决策粒度对齐，fastdtw 开销降至 1次/decision

**改动文件**: 8个文件，~250行，向后兼容（`max_decisions_per_rollout_epoch` 不配置则回退 step-based 模式）

**下一步**:
- 完整训练验证（DTG 下降 / SPL 上升）
- `format_reward_coef` 衰减策略
- `ndtw_sr_delta` vs `geo_progress` 对比实验

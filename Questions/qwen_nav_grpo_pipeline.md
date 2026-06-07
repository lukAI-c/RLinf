# QwenNav GRPO RFT 整体 Pipeline

> 用途：本文档描述 RLinf 上 Qwen3.5-4B-VL + GenArk 导航环境的 GRPO 强化微调全流程，**面向 pipeline 示意图绘制**。每个章节都按"框 — 数据 — 箭头"结构组织，可直接对应图形元素。

---

## 1. 组件总览（顶层框图）

整个系统由 **3 个 Worker Group + 1 个 Runner** 组成，运行在 Ray 之上，通过命名 Channel 异步通信。

```
                        ┌──────────────────────────────────────────────┐
                        │            EmbodiedRunner（主控）             │
                        │  - 编排每个 training step 的 8 个阶段           │
                        │  - 管理 weight sync / val / checkpoint        │
                        └────────────┬─────────────────────────────────┘
                                     │ 调度
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
┌──────────────────┐       ┌──────────────────┐         ┌──────────────────┐
│   EnvGroup       │       │  RolloutGroup    │         │   ActorGroup     │
│   (genesis sim)  │       │  (HF generate)   │         │   (FSDP train)   │
│                  │       │                  │         │                  │
│  - 多 worker     │       │  - QwenNavPolicy │         │  - QwenNavPolicy │
│  - 每 worker pin │       │  - bf16, eval()  │         │  - FSDP+LoRA     │
│    1 个场景      │       │  - 不更新权重     │         │  - 训练 + 更新   │
│  - genesis 渲染  │       │  - 接收 obs →    │         │  - 算 grad +     │
│    RGB+depth     │       │    生成 JSON     │         │    optimizer     │
└────────┬─────────┘       └────────┬─────────┘         └────────┬─────────┘
         │                          │                            │
         └─── env_channel ──────────┘                            │
         └─── rollout_channel ──────┐                            │
                                    └─── actor_channel ──────────┘
```

**3 张/4 张 GPU 分配（典型 lora 模式）**：

| 组件 | GPU | 模式 |
|---|---|---|
| ActorGroup | GPU A, B（FSDP 2-rank）| 训练，bf16 + LoRA |
| RolloutGroup | GPU C | 生成，bf16 不更新权重 |
| EnvGroup | GPU D | genesis 渲染 + 物理 |

---

## 2. 训练主循环（每个 step 的时间轴）

`global_step` 每自增 1，依次执行下列 8 个阶段。**绘图建议**：横向时间轴，每个阶段一个色块，标注阶段名 + 平均耗时 + 涉及的 Group。

```
Step N
├── ① sync_weights ─────────────── (Actor→Rollout, 每 weight_sync_interval=1 步)
├── ② generate_rollouts ─────────── (Env + Rollout 并行，时间最长)
│      ├── env.interact()         ← env 主循环
│      ├── rollout.generate()     ← 策略生成
│      └── actor.recv_rollout_trajectories()
├── ③ cal_adv_and_returns ──────── (Actor 单机，~1ms)
├── ④ run_training ────────────────── (Actor 训练核心，~10-15 min)
│      ├── recompute_prev_logprobs  ← 本项目新增
│      ├── shuffle + minibatch
│      └── 多次 forward+backward+step
├── ⑤ check_progress
├── ⑥ eval（每 val_check_interval 步）
├── ⑦ save_checkpoint（每 save_interval 步）
└── ⑧ metric_logger
```

**关键说明**：阶段 ② 内部是高度并发的（env / rollout / actor.recv 三个 Handle 同时 wait），不是串行。

---

## 3. Rollout 子流程详图（阶段 ②）

这是数据生成的核心阶段，**绘图重点**。一次 rollout 收集 `total_num_envs × max_steps_per_rollout_epoch × rollout_epoch` 个 (obs, action, reward) 三元组。

```
                                  ┌────────── 一个 env_step 的内部循环 ──────────┐
┌─────────────────┐               │                                              │
│  EnvGroup       │   env_obs     │  ┌──────────────────────────────────────┐   │
│                 │ ─────────────▶│  │ RolloutGroup.QwenNavPolicy           │   │
│  for env in     │  (RGB + hist  │  │ .predict_action_batch(env_obs)        │   │
│    range(N):    │   + instr)    │  │                                       │   │
│    obs = step() │               │  │ ├─ _build_prompt(instr, hist_imgs)    │   │
└────────┬────────┘               │  │ ├─ processor(text, images)→tokens     │   │
         ▲                        │  │ ├─ _batch_generate()                  │   │
         │ action_t               │  │ │   ↳ model.generate(do_sample=True,  │   │
         │ (整数序列)              │  │ │      temperature=1.0~1.2)          │   │
         │                        │  │ ├─ action_parser.parse_lavira_json()  │   │
         │                        │  │ │   ↳ {"action":..., "bbox":...}     │   │
         │                        │  │ ├─ _compute_teacher_forcing_logprobs  │   │
         │                        │  │ │   ↳ prev_logprobs (N, 128)         │   │
         │                        │  │ └─ _build_forward_inputs_for_env()    │   │
         │                        │  │     ↳ padded {input_ids, pix_vals,    │   │
         │                        │  │        response_ids, response_mask}   │   │
         │                        │  └──────────────────────────────────────┘   │
         │                        │                  │                          │
         │                        │                  ▼                          │
         │                        │  Trajectory dict {                          │
         │                        │    obs, action, reward, done,               │
         │                        │    prev_logprobs, forward_inputs            │
         │                        │  }                                          │
         │                        └──────────────────────────────────────────────┘
         │                                          │
         │                                          ▼
         │                              ActorGroup.recv_rollout_trajectories
         │                                          │
         └── reward_t ◀──── env_step(action_t) ─────┘
              基于 geo_progress + format_reward
```

**Episode 终止条件**（env 内部）：
- `success`：调用 STOP 且 DTG < success_distance
- `timeout`：步数达到 `max_episode_steps`
- `parse_fail`：JSON 解析失败，等效 STOP

**每步 reward 组成**：
```
reward_t  =  (prev_geo_dist - curr_geo_dist)    # geo_progress, 米
           + format_reward_coef × parse_ok      # JSON 解析对 → +0.5（可关掉）
           + success_bonus × just_succeeded     # 成功停止 → +2.5
```

---

## 4. Actor 训练子流程详图（阶段 ④）

Actor 收到 rollout_batch 后，依次执行 4 个子阶段。**绘图重点**：把数据形状的变换标在箭头上。

```
self.rollout_batch (shape: [n_step=40, B, ...])
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ ④-a  _process_received_rollout_batch                     │
│      - 计算 loss_mask（剔除 episode 终止后的 step）       │
│      - filter_rewards（可选，按 reward 阈值过滤组）       │
│      - 【本项目新增】_align_rollout_batch_to_groups       │
│        ↳ tail-drop：让 B % group_size == 0                │
│        ↳ 丢掉完全 dormant 的组                            │
└──────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ ④-b  compute_advantages_and_returns                      │
│      - calculate_scores: rewards.sum(step)               │
│      - GRPO advantage:                                   │
│          per_episode_score 在 group 内做 (x-μ)/σ          │
│      - shape: [n_step, B, num_chunks]                    │
└──────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ ④-c  run_training 主循环                                  │
│                                                          │
│  【本项目新增】_recompute_prev_logprobs_embodied          │
│    ├─ self.model.eval(), torch.no_grad()                 │
│    ├─ for each micro-batch:                              │
│    │     out = self.model(forward_inputs=..., compute_   │
│    │                       logprobs=True)                 │
│    │     recompute_chunks.append(out["logprobs"])        │
│    └─ self.rollout_batch["prev_logprobs"] ← 重算结果     │
│       self.rollout_batch["rollout_prev_logprobs"] ← 备份 │
│       （用于 importance_sampling_fix）                    │
│                                                          │
│  self.model.train()                                      │
│  shuffle rollout_batch（identity perm 也可）             │
│                                                          │
│  for update_epoch in range(update_epoch=1):              │
│    for global_batch in chunks(rollout_batch):            │
│      optimizer.zero_grad()                               │
│      for micro_batch in split(global_batch, mb_size):    │
│        ┌────────────────────────────────────────────┐    │
│        │ ④-d  Policy loss 计算                       │    │
│        │  1) forward(forward_inputs) → new_logprobs │    │
│        │  2) log_ratio = new_lp - prev_lp           │    │
│        │     clip 到 [-5, 5]（clip_log_ratio）       │    │
│        │  3) ratio = exp(log_ratio)                 │    │
│        │  4) GRPO policy loss:                      │    │
│        │     L = -mean(min(ratio·adv,               │    │
│        │                   clip(ratio,1-ε,1+ε)·adv))│    │
│        │  5) loss / gradient_accumulation           │    │
│        │  6) loss.backward()                        │    │
│        └────────────────────────────────────────────┘    │
│      grad_clip(1.0); optimizer.step(); lr_scheduler.step()│
└──────────────────────────────────────────────────────────┘
   │
   ▼
metrics（actor/ratio, grad_norm, approx_kl, policy_loss, ...）
```

---

## 5. GRPO 分组与 advantage 计算

**绘图重点**：用矩阵格子展示 group_size 如何对齐 episode。

```
total_num_envs = 12, group_size = 3
↓
env_idx:   0  1  2 | 3  4  5 | 6  7  8 | 9 10 11
group_id:    G0    |    G1   |   G2    |   G3
instruction: I0 I0 I0 | I1 I1 I1 | I2 I2 I2 | I3 I3 I3

每组 3 个 env 拿到完全相同的 (start, goal, instruction)
区别仅在 model 采样的随机性（temperature > 0）

reward_per_episode = Σ(reward_t) over t=0..T

GRPO advantage:
  A_i = (r_i - μ_group) / (σ_group + 1e-6)

  对组内 3 个 episode：
    走得最好的 → 正 A，被加强
    走得最差的 → 负 A，被抑制
    走得平均的 → A≈0，无信号
```

**关键约束**：`B = total_num_envs × rollout_epoch` 必须能被 `group_size` 整除。本项目用 tail-drop 强制保证（见 ④-a）。

---

## 6. 数据形状速查表

绘制 pipeline 时建议把这些形状标注在箭头旁。

| 阶段 | 数据 | shape | 说明 |
|---|---|---|---|
| env_obs | `main_images` | `[B, H, W, 3]` uint8 | RGB 图 |
|  | `instructions` | `list[str]` | 文本指令 |
| rollout prompt | tokenized `input_ids` | `[B, prompt_len=1612]` int32 | 含图片占位 |
|  | `pixel_values` | `[B, n_imgs×patches, dim]` bf16 | n_imgs=6, dim=1536 |
| rollout output | `response_ids` | `[B, max_new=128]` int32 | 生成的 token |
|  | `prev_logprobs` | `[B, max_new=128]` fp32 | per-token log prob |
| rollout_batch | `forward_inputs` | nested dict, dim1=B | teacher-forcing 输入 |
|  | `rewards` | `[n_step, B, num_chunks]` fp32 | 每步 reward |
|  | `dones` | `[n_step+1, B, num_chunks]` bool | episode 终止 |
|  | `loss_mask` | `[n_step, B, num_chunks]` bool | 有效步 mask |
| advantage | `advantages` | `[n_step, B, num_chunks]` fp32 | GRPO 归一化后 |
| training input | flat micro-batch | `[mb_size, ...]` | flatten n_step×B |
| forward output | `logprobs` | `[mb_size, max_new]` fp32 | 当前策略 |
| policy loss | `ratio` | `[mb_size, num_chunks, action_dim]` fp32 | reshape 后 |
|  | `loss` | scalar | token-mean 聚合 |

---

## 7. 权重同步（阶段 ①）

`ActorGroup` 训练完后必须把权重同步到 `RolloutGroup`，否则 rollout 还在用旧策略采样。

```
┌──────────────────┐   bucket-based   ┌──────────────────┐
│  ActorGroup      │  ───────────▶    │  RolloutGroup    │
│  FSDP-sharded    │   reduce_tensor  │  full state_dict │
│  state_dict      │   share + copy   │  (LoRA merged)   │
└──────────────────┘                  └──────────────────┘
```

实现入口：`actor.sync_model_to_rollout()` → 走 `recv_rollout_weights` 通道。LoRA 模式下 actor 侧 state_dict key 必须与 rollout 侧 PEFT-wrapped 模型对齐。

---

## 8. 关键修复点（让 ratio 不爆炸）

绘图时建议把这些标在对应的环节作为"附加 box"。

| # | 修复点 | 位置 | 目的 |
|---|---|---|---|
| F1 | **pixel_values dtype 统一 bf16** | `_batch_generate` / `_compute_teacher_forcing_logprobs` / `default_forward` | 三处 forward 路径用同一精度，消除 ~20 nat 偏差 |
| F2 | **logprob_type = token_level** | `genark_grpo_qwen.yaml::algorithm` | 避免 `exp(sum_64_tokens)` 放大小偏差 |
| F3 | **clip_log_ratio_min/max ±5** | `fsdp_actor_worker.py::run_training` 的 policy_loss 调用 | 兜底 multi-step drift |
| F4 | **recompute_prev_logprobs** | `fsdp_actor_worker.py::_recompute_prev_logprobs_embodied` | 用 actor 自己重算 old，消除跨进程结构性差异 |
| F5 | **tail-drop group align** | `fsdp_actor_worker.py::_align_rollout_batch_to_groups` | dormant + 整除 group_size，避免 `reshape(-1,3)` 崩 |

---

## 9. 关键配置参数表（绘图标注用）

| 参数 | 当前值 | 影响 |
|---|---|---|
| `total_num_envs` | 12 | 并行 env 数（受场景 episode 数限制，~14） |
| `group_size` | 3 | 每组 rollout 数，instruction 多样性 = B/group_size |
| `rollout_epoch` | 1 | 一轮 rollout 收集多少个 mini-rollout |
| `max_steps_per_rollout_epoch` | 40 | rollout 总步数 |
| `max_episode_steps` | 40 | episode 最大长度 |
| `max_new_tokens` / `action_dim` | 128 / 128 | response 长度（必须相等） |
| `micro_batch_size` | 1 | actor 训练 micro batch |
| `global_batch_size` | 96 | actor 训练 global batch |
| `clip_ratio_high / low` | 0.28 / 0.2 | PPO ratio 截断 |
| `temperature_train` | 1.0~1.2 | rollout 采样温度（影响组内多样性） |
| `format_reward_coef` | 0.0~0.5 | JSON 解析对的额外 reward |

---

## 10. 模型输入输出格式详解（绘图用）

> 这一节是为了让示意图里的 **"Input box"** 和 **"Output box"** 画得准。所有内容都从 [prompts.py](rlinf/models/embodiment/qwen_nav/prompts.py) 提取，确保与代码一致。

### 10.1 模型输入（一个 env step 的 Prompt）

模型每一步收到的 prompt 是 **多模态 chat-template**，由 system + user 两段构成，其中 user 段交替包含 `<image>` 占位符和文本：

```
┌─────────────────── System Prompt ──────────────────┐
│ /no_think                                          │
│ You are a navigation agent in an indoor            │
│ environment. You receive 4-directional views       │
│ (front, left, right, behind) of the current        │
│ location and a sequence of historical observation  │
│ images. Based on the text instruction, history,    │
│ and current views, you decide the next             │
│ navigation action.                                 │
│                                                    │
│ Action Space:                                      │
│   - "navigate to forward"                          │
│   - "navigate to left"                             │
│   - "navigate to right"                            │
│   - "navigate to behind"                           │
│                                                    │
│ You MUST respond with a valid JSON object...       │
│ (含 JSON schema 和 Guidelines)                      │
└────────────────────────────────────────────────────┘

┌──────────────────── User Content ──────────────────┐
│ Navigation Task: "Walk across the floor and wait   │
│                   the archway."                    │
│                                                    │
│ Navigation History (chronological, oldest → newest)│
│   Step 0: <image>   ← 历史帧 0                      │
│   Step 1: <image>   ← 历史帧 1                      │
│   Step 2: <image>   ← 历史帧 2                      │
│   ...               (最多 history_max_frames 张)    │
│   Step k: <image>   ← 历史帧 k                      │
│                                                    │
│ Current 4-directional views:                       │
│   Front view:  <image>   ← 当前正前方               │
│   Left view:   <image>   ← 当前左侧                 │
│   Right view:  <image>   ← 当前右侧                 │
│   Behind view: <image>   ← 当前后方                 │
│                                                    │
│ Decide your next action and respond with the JSON  │
│ object as instructed in the system prompt. Output  │
│ ONLY the JSON, nothing else.                       │
└────────────────────────────────────────────────────┘
```

**图像顺序（绘图用）**：图像 token 在 prompt 中按"历史 → 当前 4 视角(F/L/R/B)"顺序插入。

- 历史帧数：`history_max_frames`（默认 5），不足时用空白图 padding
- 当前视角：4 张（`use_4dir=True` 时）或 1 张
- 总图片数 `n_images = history_max_frames + 4`（典型值 6 张，因为 `history_every_k=1` 会让前几步只有 0~1 张历史）
- 每张图分辨率 `image_size`（默认 320×240），经过 Qwen 处理器后产生 ~784 个 patch token

**实际示例图片**（可用于示意图）：

| 用途 | 路径 |
|---|---|
| 单视角原始 RGB（front view 示例）| [/home/clk/workspace/genark/genark_web_output_gt/latest.jpg](/home/clk/workspace/genark/genark_web_output_gt/latest.jpg) |
| 历史轨迹片段（连续帧）| [/home/clk/workspace/genark/genark_web_output_gt/frames/frame_000014.jpg](/home/clk/workspace/genark/genark_web_output_gt/frames/frame_000014.jpg) |
| 场景概览（俯视拼接图，注意：**这不是模型输入**，只是为了示意场景）| [/home/clk/workspace/genark/test_render_output/2azQ1b91cZZ_ep10_scaled.jpg](/home/clk/workspace/genark/test_render_output/2azQ1b91cZZ_ep10_scaled.jpg) |
| 同场景其他 episode RGB | [/home/clk/workspace/genark/test_render_output/2azQ1b91cZZ/episode_10.jpg](/home/clk/workspace/genark/test_render_output/2azQ1b91cZZ/episode_10.jpg) |
| 训练 rollout 录像 mp4 | [/home/clk/workspace/results/genark_grpo_qwen/video/eval/seed_0/0.mp4](/home/clk/workspace/results/genark_grpo_qwen/video/eval/seed_0/0.mp4) |

**绘图建议**：示意图中"Input"框可画成左右两列：
- 左列：5 张历史小图竖排（标 Step 0~4）
- 右列：4 张当前视图田字格（F/L/R/B）
- 上方：instruction 文本气泡 `"Walk across the floor and wait the archway."`

### 10.2 模型输出（JSON Action）

模型输出严格遵循 lavira-style schema：

```json
{
    "action": "navigate to forward",   // 4 选 1
    "bbox": [x1, y1, x2, y2],          // 目标 bbox（front view 像素坐标），或 null
    "stop": false,                     // true=到达目标
    "stair": false                     // "up" / "down" / false
}
```

**实际生成样例**（取自最近训练 log，已 token 解码）：

```json
{
    "action": "navigate to left",
    "bbox": [128, 64, 240, 192]
}
```

或带 markdown 包装（parser 也能识别）：

````
```json
{
    "action": "navigate to forward",
    "bbox": null,
    "stop": false,
    "stair": false
}
```
````

**字段含义**：
- `action`：粗粒度方向意图。Env 内部用 [action_parser.py:DIRECTION_TO_ACTIONS](rlinf/models/embodiment/qwen_nav/action_parser.py#L39) 翻译为低层动作序列：
  - `forward` → `[1]`（前进 0.25 m）
  - `left` → `[2, 1]`（左转 30° + 前进）
  - `right` → `[3, 1]`（右转 30° + 前进）
  - `behind` → `[2]×6 + [1]`（左转 180° + 前进）
- `bbox`：模型在 front view 上锁定的导航目标框（像素坐标）。当前奖励**不直接使用** bbox（只是模型自我监督的输出）。
- `stop`：模型主动声明"已到达"，触发 episode 结束判定 + success_bonus。
- `stair`：标记下一步是否过楼梯（当前 env 未启用楼梯逻辑，仅作占位）。

**Parser 鲁棒性**（[action_parser.py](rlinf/models/embodiment/qwen_nav/action_parser.py)）：
1. 优先匹配 fenced ` ```json {...}``` ` 格式
2. 失败则匹配裸 `{...}`
3. JSON 解析失败 → `action = ACTION_PARSE_FAIL = 4`，env 强制 STOP
4. 字段缺失/非法 → 同上

### 10.3 输入输出尺寸速查

| 数据 | 典型尺寸 | 说明 |
|---|---|---|
| 单张 RGB 图像 | 320 × 240 × 3, uint8 | env 渲染出来的视图 |
| n_images / sample | 5 historical + 4 current = **9** | 早期 step 可能少于 5 个 historical（用空图 pad）|
| `prompt_len`（含图占位）| **1612 tokens** | 处理后的 text+image token 总数 |
| `max_new_tokens` | **128** | JSON 输出最长（实测 60~80） |
| `pixel_values`（per sample）| `[9, 784, 1536]` bf16 | 9 张图 × 784 patches × 1536 dim |
| 输出 JSON 字符串 | 60~80 token | 解析后映射为 1~7 个底层 action |

---

## 11. 推荐图层（绘图建议）

如果做成多层图，建议按以下分层：

1. **Layer 0：物理资源层** — GPU 分配、内存、CPU
2. **Layer 1：进程层** — Ray Workers（Env / Rollout / Actor）+ Runner
3. **Layer 2：数据流层** — env_channel / rollout_channel / actor_channel
4. **Layer 3：算法层** — Episode → Trajectory → Group → Advantage → Loss
5. **Layer 4：修复点叠加层** — F1~F5 用红色虚线标注在对应位置

---

## 附：完整 step 时间分解（参考 lora 实际测量）

```
step total = ~2086 s (35 min)
├── rollout (env + generate)      ≈ 688 s
│   ├── env.interact              ≈ 676 s
│   └── rollout.predict           ≈ 983 s（与 env 并发）
├── cal_adv_and_returns           ≈   0.01 s
├── actor.run_training            ≈ 976 s
│   ├── recompute_prev_logprobs   ≈ ~80 s（含一次 full eval forward）
│   └── 训练循环 (gradient steps)  ≈ ~900 s
└── eval（按需）                  ≈ 418 s
```

绘制 step 时间饼图时，主要色块就是 rollout vs actor_training（占比约 1:1.4），其余可忽略。

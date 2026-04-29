# GenArk RFT 训练方案（Qwen3-4B + Lavira-style Prompt）

> 在 M1–M2（环境接入 + 全量 eval）已完成的基础上，启动 M3：用 RFT（PPO/GRPO）微调 **Qwen3-4B（本地部署）** 作为导航策略，prompt 设计沿用 `template/lavira-code` 的 TT / FT / BACKTRACK 三档配置。
>
> 上游记录：`genark_rlinf_integration.md`（M1–M2 实现细节）、`genark_integration_plan.md`（旧版 UniNaVid 方案，作废）。

---

## 0. 关键变更

| 维度 | 旧方案（UniNaVid 7B） | 新方案（Qwen3-4B + RFT） |
|------|----------------------|--------------------------|
| 策略模型 | UniNaVid 7B（LLaVA + video feat_cache） | **Qwen3-4B**（本地部署，文本/VL 待定） |
| Prompt | 隐式视频 token 序列 | **结构化 4-dir + history 图像 + JSON 输出** |
| 训练目标 | 复现 baseline（54% SR） | RFT 提升 SR/SPL，对比 lavira 的 TT/FT 配置 |
| Action 输出 | 4-way logits 直接 argmax | **JSON 格式动作 + 解析**，含 stop/stair |
| 奖励 | dense（geo_dist delta）+ sparse（success） | 同左 + JSON 格式合规奖励 |

---

## 1. 模型与依赖

### 1.1 待确认

- [ ] **Qwen3-4B 本地路径**：用户需提供（建议挂载于 `/home/nvme03/...` 或 `/home/clk/...`）
- [ ] **VL or 纯文本**：lavira prompt 含 4-dir 图像 + history 图像，强烈建议 VL 版本（Qwen2.5-VL-3B / Qwen2-VL-2B）。若坚持纯文本 Qwen3-4B，需外挂 vision encoder（CLIP / SigLIP）做 image → token embedding 投影。
- [ ] **tokenizer / chat template**：Qwen 系列默认 ChatML，与 lavira 的 plain string prompt 需 wrap 进 system + user 角色。

### 1.2 推荐配置

```yaml
# examples/embodiment/config/model/qwen_navigator.yaml
model_path: /home/.../Qwen3-4B-Instruct       # ← 待填
model_type: qwen_vl                            # 若用 VL；纯文本则为 qwen_lm
max_seq_len: 8192
max_new_tokens: 512                            # JSON 输出约 300–500 tok
temperature: 0.7                               # 训练采样温度
top_p: 0.9
do_sample: true
chat_template: chatml
load_in_4bit: false                            # 4B 全精度可放下
```

---

## 2. Prompt 设计（沿用 lavira 三档）

### 2.1 三种 setting

| 配置 | 来源 | 是否启用 TODO | 是否启用 backtrack | 输出字段 | 用途 |
|------|------|---------------|-------------------|---------|------|
| **FT** | `LA_PROMPT_BACKTRACK_NO_TODO` | ❌ | ✅ | progress / reasoning / action / stop / stair | **主训练配置**（lavira 实验中 SR 最高） |
| **TT** | `LA_PROMPT_BACKTRACK` | ✅ | ✅ | + reasoning_todo / todo_updates | 对照组（验证 TODO 抽象层是否在 RFT 后仍劣于 FT） |
| **REPLAN** | `LA_PROMPT_BACKTRACK_REPLAN` | ❌ | ✅ | + 二次决策 reasoning | backtrack 后的二次选择（推理时启用） |

### 2.2 Prompt 输入构造（每步）

```
[System] You are a navigation agent in a Matterport3D scene.
[User]
Instruction: "{episode.instruction}"

Navigation History (last K=8 waypoints):
<image_token_0> step 0
<image_token_1> step 4
... (每隔 4 步采样一帧，避免 history 过长)

Current 4-directional views at this waypoint:
<image_front>  <image_left>  <image_right>  <image_behind>

[FT or TT prompt body — 见 lavira_prompts.py]

Available Actions:
- navigate to forward
- navigate to left
- navigate to right
- backtrack to <waypoint_id>     # waypoint_id ∈ {0..N-1}
- stop
```

**关键改造（适配 GenArk 环境）**：

1. **4-dir 渲染**：当前 `genark_env.py` 只渲染前向视角，需要扩展为 4 个相机位置（旋转 0°/90°/180°/270°），一次 step 渲染 4 张图。
2. **History 采样**：每隔 K 步保留一帧（K=4），最多保留 8 帧（避免 ctx 超长）。
3. **Action 解析**：模型输出 JSON → 解析 `action` 字段 → 映射到 GenArk 的 `step_move/step_turn` 动作。
   - `navigate to forward` → forward 0.25m
   - `navigate to left` → turn 30° + forward 0.25m
   - `navigate to right` → turn -30° + forward 0.25m
   - `backtrack to <id>` → 直接传送回 waypoint id（需在 env 中维护 waypoint 历史）
   - `stop=True` → episode 结束

### 2.3 Prompt 文件结构

```
rlinf/models/embodiment/qwen_nav/
├── __init__.py
├── qwen_nav_policy.py            # QwenNavPolicy(BasePolicy)
├── prompts.py                    # FT / TT / REPLAN 三个模板（移植自 lavira）
├── action_parser.py              # JSON → 4-way action + stop + backtrack id
└── history_manager.py            # 每 env 独立的 waypoint 历史 + 图像缓存
```

---

## 3. 环境层改造

### 3.1 4-direction 渲染（必做）

`genark_env.py` 当前只渲染前向；需在 `_step()` 末尾追加：

```python
# 渲染 4 个方向的 RGB（绕 z 轴旋转）
def _render_4dir(self):
    cam_yaws = self._yaw + torch.tensor([0, π/2, π, -π/2], device=device)  # (N, 4)
    rgb_list = []
    for k in range(4):
        self._gs_cam.set_pose(pos=self._cam_pos, lookat=self._cam_pos + dir_from_yaw(cam_yaws[:, k]))
        rgb_list.append(self._gs_scene.render_camera(self._gs_cam))  # (N, H, W, 3)
    return torch.stack(rgb_list, dim=1)  # (N, 4, H, W, 3)
```

**显存影响**：H=480, W=640, N=20 → 单 worker 渲染 +60MB（4× 当前），可接受。

### 3.2 Waypoint 历史（用于 backtrack）

在 `GenarkVecEnv` 内每个 env 独立维护：

```python
self._waypoints = [[] for _ in range(N)]   # list[list[(pos, yaw, rgb_4dir)]]

# 每步 forward 时追加
self._waypoints[i].append((pos[i], yaw[i], rgb_4dir[i]))

# backtrack 时读取
def _execute_backtrack(self, env_idx, waypoint_id):
    target = self._waypoints[env_idx][waypoint_id]
    self._cam_pos[env_idx] = target.pos
    self._yaw[env_idx]     = target.yaw
    # rgb 直接重渲染当前位置即可
```

### 3.3 Observation 格式改动

```python
return {
    "main_images":       rgb_4dir,                # (N, 4, H, W, 3) — 替代之前的单视角
    "history_images":    history_buffer,          # (N, K_hist, H, W, 3)
    "task_descriptions": self._instructions,      # list[str]
    "waypoint_ids":      [list(range(len(wp))) for wp in self._waypoints],  # 可 backtrack 的 id
    "states":            elapsed_t,
}
```

`EnvOutput.prepare_observations()` 仅透传 5 个 key 的限制需绕过：history_images / waypoint_ids 走 `info` 字段或扩展 EnvOutput。

---

## 4. RFT 算法设计

### 4.1 损失：GRPO（推荐）

GRPO 比 PPO 更适合本场景：
- 不需要 value head（省一组参数 + 训练稳定性高）
- 同一 prompt 采样 G=4 条轨迹，组内 advantage 归一化，自然对应"同 instruction 不同动作选择"

**配置**：
```yaml
algorithm:
  loss_type: grpo
  group_size: 4              # 每个 instruction 采 4 条轨迹
  clip_ratio: 0.2
  kl_coef: 0.01              # 与 base model 的 KL 约束
  entropy_coef: 0.0
```

### 4.2 奖励设计

```python
reward = (
    + α * (prev_geo_dist - curr_geo_dist)       # dense progress（α=1.0）
    + β * I[stop AND dist_to_goal < 3.0m]       # success bonus（β=2.5）
    + γ * I[json_parse_success]                  # 格式合规（γ=0.1）
    - δ * I[stop AND dist_to_goal >= 3.0m]      # 错误停止惩罚（δ=0.5）
    - ε * I[step_count > max_episode_steps]     # 超时惩罚（ε=0.5）
)
```

格式合规奖励 γ 在训练初期至关重要——Qwen 未必稳定输出合法 JSON，前 1k step 主要靠 γ 让模型学会输出 schema。

### 4.3 训练 vs 推理差异

| | 训练（RFT） | 推理（eval） |
|---|---|---|
| 采样温度 | 0.7（探索） | 0.0 / 0.5（贪心或低温） |
| do_sample | True | True（lavira 经验，避免重复） |
| group_size | 4 | 1 |
| KV cache | ❌ 关闭（梯度需要） | ✅ 开启 |
| FSDP | ✅ 4 卡分片 | ❌ 单卡推理 |

---

## 5. 实施步骤（M3 拆解）

### M3.1 — Prompt + Action 接口（3 天）

- [ ] 移植 lavira 三个 prompt 模板到 `rlinf/models/embodiment/qwen_nav/prompts.py`
- [ ] 实现 `action_parser.py`：JSON 解析 + 容错（regex 兜底 + 失败时返回 stop）
- [ ] 实现 `history_manager.py`：每 env 独立的 waypoint + history image 缓冲
- [ ] **验收**：单元测试覆盖 FT/TT/REPLAN 三种 prompt 构造 + 10 个 JSON 输出样本解析

### M3.2 — 4-dir 环境改造（2 天）

- [ ] `genark_env.py` 增加 `_render_4dir()`（4 次 camera pose 切换）
- [ ] Observation 加 `history_images` / `waypoint_ids`（绕过 5-key 限制）
- [ ] `_execute_backtrack()` 实现（pos/yaw 直接覆盖）
- [ ] **验收**：4-dir 渲染图像保存到 `/tmp/4dir_*.png`，肉眼检查方向正确；backtrack 后 pos 完全一致

### M3.3 — Qwen 推理 Policy（4 天）

- [ ] `QwenNavPolicy(BasePolicy)`：包装 transformers `AutoModelForCausalLM` + processor
- [ ] `predict_action_batch()`：构 prompt → batch generate → JSON parse → action tensor
- [ ] 复用 `_EpisodeCache` 思路做 per-env history 隔离（无需 feat_cache 复用，每步从图像重编码）
- [ ] 注册到 `rlinf/models/__init__.py` 的 `register_model("qwen_nav")`
- [ ] **验收**：先用 FT prompt 跑 eval-only，目标 SR ≥ 30%（未训练的 zero-shot baseline）

### M3.4 — `default_forward()` + RFT loss 接入（5 天）

- [ ] `default_forward()` 返回 `(logits, log_probs, entropy)`，**关闭 KV cache**
- [ ] 训练态用 teacher forcing：把 sampled JSON output 作为 label，计算 token-level log_prob
- [ ] 实现 `compute_log_probs()` 仅对 action-relevant tokens 求和（mask 掉 reasoning 段以稳定训练，可选）
- [ ] 接 `embodied_grpo` loss（参考 `gsenv_ppo_*.yaml`）
- [ ] **验收**：单步梯度 norm 可观测，loss 随 step 下降

### M3.5 — FSDP + 多卡训练（3 天）

- [ ] FSDP 配置：4B 参数 4 卡分片，每卡约 2GB 权重 + 4GB 激活
- [ ] GPU 分配：actor=GPU0-3（FSDP），rollout=GPU4，env=GPU5-7
- [ ] **验收**：完成 100 step 不 OOM，吞吐 > 5 episodes/min

### M3.6 — 小规模训练 + 对比实验（1 周）

| 实验 | Prompt | 训练 step | 期望 SR |
|------|--------|-----------|---------|
| Exp-1 (baseline) | FT zero-shot | 0 | 30%（参考） |
| Exp-2 (RFT-FT) | FT + RFT | 5k | 45%+ |
| Exp-3 (RFT-TT) | TT + RFT | 5k | 40%+（验证 lavira 的 FT > TT 结论） |
| Exp-4 (curriculum) | FT 先 → 加 TT 数据 | 10k | 50%+ |

---

## 6. GPU 分配（8 卡场景）

```yaml
# examples/embodiment/config/genark_grpo_qwen.yaml
cluster:
  num_nodes: 1
  component_placement:
    actor:    {placement: 0-3}    # Qwen3-4B FSDP × 4
    rollout:  {placement: 4}      # 推理（采样）
    env:      {placement: 5-7}    # Genesis × 3 worker
```

env 减到 3 个 worker（vs M2 的 4 个）→ 每 pass 处理 3 个场景 → 全量 eval 需 4 passes。可接受。

---

## 7. 风险与缓解

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| Qwen 输出不合法 JSON（训练初期高频） | 高 | γ 格式奖励 + regex 兜底解析 + warmup 阶段 SFT 几百条 lavira 真实输出 |
| 4-dir 渲染拖慢 env 步进 | 中 | Genesis 多相机 batch render（已支持），实测 < 1.3× 单方向耗时 |
| Qwen3-4B 本身不擅长空间推理 | 中-高 | 先用 lavira-Gemini 蒸馏数据做 SFT warmup（参考 lavira 已有日志），再上 RFT |
| Backtrack 引入非 Markov（历史长度变化） | 中 | history_images 固定窗口（K=8），溢出帧丢弃 |
| FSDP + 长 ctx (8k tokens) OOM | 中 | gradient checkpointing；必要时 max_seq_len 降到 4k |
| JSON parse 失败导致 reward shaping 异常 | 低 | parse 失败时强制 action=stop, reward=-0.5（不进 GRPO group 归一化） |

---

## 8. 文件清单（新增 + 修改）

### 新增

```
rlinf/models/embodiment/qwen_nav/
├── __init__.py
├── qwen_nav_policy.py
├── prompts.py                    # 移植自 lavira_baselines/utils/prompts_vln.py
├── action_parser.py
└── history_manager.py

examples/embodiment/config/
├── model/qwen_navigator.yaml
├── genark_grpo_qwen.yaml         # 训练入口（FT 配置）
├── genark_grpo_qwen_tt.yaml      # 训练入口（TT 配置）
└── genark_eval_qwen.yaml         # eval-only

scripts/
└── run_qwen_rft.sh               # 训练启动脚本
```

### 修改

```
rlinf/envs/genark/genark_env.py
  - 新增 _render_4dir()
  - 新增 waypoint 历史维护 + _execute_backtrack()
  - obs 增加 history_images / waypoint_ids

rlinf/envs/__init__.py
  - 扩展 EnvOutput 透传字段（或走 info）

rlinf/models/__init__.py
  - register_model("qwen_nav")

rlinf/envs/action_utils.py
  - 新增 qwen_nav 分支：JSON action 字符串 → 数值动作
```

---

## 9. 里程碑总览

| 里程碑 | 内容 | 预计工时 | 验收 |
|--------|------|----------|------|
| M3.1 | Prompt + JSON parser | 3d | 单测通过 |
| M3.2 | 4-dir + backtrack 环境 | 2d | 渲染肉眼正确 |
| M3.3 | Qwen 推理 policy（zero-shot eval） | 4d | SR ≥ 30% |
| M3.4 | `default_forward()` + GRPO loss | 5d | loss 下降 |
| M3.5 | FSDP 多卡训练 | 3d | 100 step 不 OOM |
| M3.6 | FT vs TT 对比实验 | 7d | RFT-FT SR ≥ 45% |
| **M3 合计** | | **~24 工作日** | |

---

## 10. 待用户确认

1. **Qwen3-4B 路径**：本地路径（含 tokenizer + config + safetensors）
2. **VL 还是纯文本**：是否使用 Qwen-VL 系列；若纯文本，外挂哪个 vision encoder
3. **是否做 SFT warmup**：用 lavira-Gemini 已有的真实输出（`logs/ablation_3configs_stratified/r2r_FT.log`）做几百条 SFT，再上 RFT，效果会显著好。是否同意？
4. **GRPO vs PPO**：默认推荐 GRPO；若坚持 PPO 需补 value head 设计
5. **action_list 是否含 backtrack**：lavira 的 BACKTRACK 机制需要 env 支持 waypoint 传送，是否启用？（不启用则简化为 forward/left/right/stop 4 类）

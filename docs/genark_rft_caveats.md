# GenArk RFT 已知设计缺陷与权衡（Caveats）

> 本文记录当前 RFT 实现路线（Plan B：lavira 风格 JSON 单轮 prompt + Qwen-VL）已知的设计漏洞和工程权衡。
>
> **目的**：训练实验前/中/后回头检查，验证这些假设是否真的造成了问题，决定是否要修复。

---

## 架构层（不可修复，是设计选择带来的代价）

### C1. 单轮 Prompt 重建 vs 多轮对话

**现状**：每个决策步独立构造 prompt（system + history + 4-dir/单视角 + instruction + JSON schema），重新调用模型，无 chat 历史累积。

**代价**：
- 训练吞吐慢 3–4×（每步 prefill 都从头算，无 KV cache 复用）
- History image 在每步都要重新 encode
- 80-step episode 总 prefill ≈ 240k tok（vs 多轮 ~80k tok）

**ActiveVLN 不一样**：他们的 `parallel_env_vlnce.py` 用 `running_states = torch.cat(...)` 累积 token，整段 episode 是一条连续序列，可以做 KV-cache 复用，训练快得多。

**为什么仍这样选**：lavira 风格的 prompt 必须每步重组（4-dir labels + history label 注入），没法增量 append。

**触发风险**：训练 epoch 时间过长，影响实验迭代速度。

---

### C2. History 图像无梯度（hidden ceiling）

**现状**：单轮模式下 history images 是 prompt 的一部分，loss 只对 response token 求，梯度不回传到 history。

**代价**：模型永远学不到"如何主动利用 history"，只能被动地从 ctx 中读出信息。这对 ZS 推理（Gemini-Pro 级别）影响小，但对 4B 模型 RFT 是隐形天花板。

**多轮架构无此问题**：上一步的 assistant response 也参与 loss，间接训练了"在长 ctx 中决策"的能力。

**触发风险**：训练 reward 早期上升后 plateau 在比 ActiveVLN 低的水平。

---

## 算法层（可修复，但当前选了简单方案）

### C3. Action Loss 全 response token 平均（策略 A）

**现状**：JSON 输出包含 `progress_analysis` / `reasoning` / `action` / `bbox` / `stop` 五个字段，~130 tok。当前实现把 advantage 平均到所有 response token：

```python
loss = -(advantage * log_prob_of_all_response_tokens).mean()
```

**代价**：
- `action` + `stop` 两个真正影响环境的字段只占 ~8 tok（6%）
- `reasoning` 占 ~80 tok（60%）但只是 fluff，分到的 advantage 是"虚假信号"
- 模型可能学到"reasoning 写什么样能拿到正 advantage"——而 reasoning 内容并不影响环境，是 spurious correlation

**推荐修复方案（策略 B，未实现）**：
```python
# 用 tokenizer 的 offset_mapping 定位 action / stop 字段值的 token range
action_token_mask = locate_field_value_tokens(response_ids, "action")
stop_token_mask   = locate_field_value_tokens(response_ids, "stop")
loss = -(advantage * log_prob * (action_token_mask | stop_token_mask)).sum() / mask.sum()
```

**触发条件**：reward 不上升、reasoning 输出趋于乱码或漂移。

---

### C4. Per-step nDTW Reward 不稳定

**现状**：用户要求 reward = `nDTW(s_{t+1}) - nDTW(s_t)`，每步都计算。

**代价**：
- nDTW 是基于"已走轨迹 vs GT 完整轨迹"的 DTW 距离，agent detour 时 nDTW 暂时下降
- agent course-correct 回正路，nDTW 重新上升
- **detour-correct 的过程被惩罚 detour，奖励 correction，是错位信号**——理想情况下整条 detour-correct 应该被一次性评估
- per-step fastdtw 调用：80 envs × 80 step = 6400 次 fastdtw / pass，约 +60s/pass

**ActiveVLN 做法**：nDTW 只在 episode 结束时计算（terminal reward）。

**触发条件**：训练 reward 出现高方差、agent 出现 oscillation 行为。

---

### C5. SR delta 信号过稀

**现状**：`reward += SR(s_{t+1}) - SR(s_t)`。SR 只在 `stop` 且 `d2g < 3.0m` 时从 0→1，**前 N-1 步 delta 全 0**。

**代价**：等同于稀疏 outcome reward，没有任何 dense 引导。

**对比**：当前 genark_env.py 已有的 `prev_geo_dist - curr_geo_dist` 是 dense 进度信号，每步都有值。

**触发条件**：训练初期 reward 几乎全 0，policy 探索靠 random walk，收敛极慢。

---

### C6. Reward Scale 不平衡

**现状**：nDTW delta 单步约 ±0.005，SR delta 终点 ±1.0，直接相加。

**代价**：80 步累积 nDTW noise ≈ ±0.4，SR=±1，两者数量级接近但语义不同。终点 reward 的"成功信号"会被前 79 步的 nDTW 噪声稀释。

**推荐**：加权 `α * dtw_delta + β * SR_delta`，β ≫ α（如 α=1.0, β=10.0）。

**触发条件**：training reward 曲线稳定上升但 SR 不上升。

---

### C7. 没有 Format Reward

**现状**：当前 reward 只有 nDTW delta + SR delta，没有 JSON 格式合规奖励。

**代价**：4B 模型 RFT 初期 JSON 失败率 25%+。失败的 episode 全部按"action=stop"兜底，等于丢梯度信号。

**推荐**：加 `format_reward = 1.0 * I[json_parse_ok]`，至少在前 1k step 强制学 schema。

**触发条件**：训练日志中 `parse_failure_rate` > 10% 且 reward 不上升。

---

### C8. Bbox 输出无监督

**现状**：JSON 包含 `bbox: [x1, y1, x2, y2]`，但：
- GenArk 数据集无 GT bbox 标注
- 没有下游消费方（LA-VA 框架里 bbox 给 VA 模型用，本项目无 VA）
- bbox token 进 loss 但 reward 不针对它

**代价**：
- bbox 内容会随机漂移（无监督）
- 占用 ~15 tok/step，进一步稀释 action 字段的 advantage 信号

**当前选择**：保留 bbox 字段（用户要求），但实际 RFT 训不到任何有意义的输出。

**触发条件**：bbox 数值显著偏离图像范围（如 [9999, -1, ...]）或趋于全 0。

---

### C9. GRPO 分组语义需明确

**问题**：单轮模式下"trajectory"对应物是什么？

**当前实现假设**：
- 同一 instruction 采 4 个 episode（GRPO group_size=4）
- 每 episode 有 N_i 步 → N_i 个独立 (prompt_i, response_i) 样本
- episode-level return = sum of step rewards
- group-normalize 4 个 return 得 advantage
- 同 episode 内所有 N_i 步共享该 advantage（除以 N_i 防长 episode 主导）

**风险**：如果按 step-level 分组（同 instruction 同 step 跨 episode），4 个 episode 在 step k 的状态已发散，分组无意义。

**触发条件**：advantage 方差爆炸或全 0。

---

## 工程层（小问题）

### C10. 4-dir 渲染未实现

**现状**：lavira prompt 设计需要 4-direction views，但当前 genark_env.py 只渲染前向视角。

**临时方案**：QwenNavPolicy 把单视角复制 4 次（label 为 front/left/right/behind），其实是同一张图。

**真正解**：扩展 `genark_env.py._render_4dir()`（plan §3.1），但需要修改 Genesis 相机调度。

**影响**：当前实现下 lavira prompt 中"left view / right view / behind view"实际看到的是"front view"。模型大概率忽略多个 label 输出乱选——**需要在 4-dir 渲染上线前别开始正式训练**。

---

### C11. 模型路径占位

**现状**：用户提到 "Qwen3.5-4B"，HF Hub 上没有这个模型。

**当前实现**：config 写 `model_path: TODO_FILL_QWEN_VL_PATH`，加载逻辑用 `transformers.AutoModelForVision2Seq` 通用接口，可适配 Qwen2.5-VL-3B / 7B / Qwen3-VL（如有）。

**建议**：先用 `Qwen/Qwen2.5-VL-3B-Instruct`，这是 4B 量级唯一稳定的官方 VL 模型。

---

## 触发自检清单

训练开始后定期对照此表检查日志：

| 指标 | 阈值 | 对应 caveat | 处理 |
|---|---|---|---|
| `parse_failure_rate` | > 10% | C7 | 加 format_reward |
| `reasoning_drift` 抽样 | 出现乱码 | C3 | 切策略 B |
| `bbox_validity` | < 50% | C8 | 删 bbox 字段 |
| `reward_variance / mean` | > 5 | C4, C6 | 改终点 nDTW，调权重 |
| `advantage_norm == 0` 比例 | > 30% | C9 | 检查分组逻辑 |
| `epoch_time` | > 预期 3× | C1 | 切多轮架构 |
| `eval SR` 早期 plateau | 不再上升 | C2 | 接受或重设计 |

---

## 修复优先级

1. **必修（影响是否能跑通）**: C7（format reward）、C10（4-dir 渲染）
2. **观察后修（看实验结果）**: C3（action loss）、C4-C6（reward 设计）
3. **设计层只能接受**: C1、C2、C9
4. **可删除字段**: C8（删 bbox）

---

## 工程层（环境后果）

### C12. transformers 5.x 升级破坏 UniNaVid 加载

**起因**：Qwen3.5-4B（`Qwen3_5ForConditionalGeneration` 架构）需要 `transformers ≥ 5.0`。本地 `genesis` env 从 4.31 → 5.7。

**回归**：UniNaVid 通过 `genark/uninavid/model/language_model/llava_llama_vid.py` 注册了一个名为 `llava` 的 config。transformers 5.x 内置了官方 `llava` config，注册时冲突报错：

```
ValueError: 'llava' is already used by a Transformers config, pick another name.
```

**影响**：`genark_eval_only.yaml`（M2 UniNaVid 路径）现在无法直接 import 跑通。但 M2 baseline 数据已落盘（54% SR），不影响 RFT 工作。

**API 变更**：
- `AutoModelForVision2Seq` → `AutoModelForImageTextToText`（已在 `fsdp_model_manager.py` / `openvla_oft/__init__.py` / `qwen_nav_policy.py` 加 try/except 兼容）
- `torch_dtype=` → `dtype=`（仍兼容，只 warning）
- `padding_side` 默认变化（已在 QwenNavPolicy 加 `padding_side='left'`）

**修复路径**（如果未来需要重跑 UniNaVid）：
1. 在 `genark/uninavid/model/language_model/llava_llama_vid.py` 把注册名 `llava` 改成 `uninavid_llava`
2. 或者 pin transformers 在 4.45 左右（同时支持 Qwen3 和老 llava）
3. 或者用单独 conda env 隔离两个 stack

---

## 历史变更

- 2026-04-30: 初版，对应 QwenNavPolicy 首版实现。
- 2026-04-30: 加 C12（transformers 5.x 升级 + UniNaVid 注册冲突）。
- 2026-04-30: 4-dir 渲染上线（`enable_4dir_render: true`），C10 fallback 仍保留。
- 2026-04-30: 实现 `default_forward()` + Path A RLinf embodied GRPO 接入完成。
  - `QwenNavPolicy.default_forward()` 实现 teacher-forcing logprob 计算
  - `predict_action_batch()` 新增 `collect_forward_inputs=True` 训练模式
  - history 固定 pad 至 `history_max_frames` blank 图以保证 pixel_values 形状一致
  - 新增 `genark_grpo_qwen.yaml` 训练 config（GRPO，action_level logprob）
  - `SupportedModel.QWEN_NAV` 注册至 `rlinf/config.py`

- 2026-04-30: smoke test 1–17 修复完整 GRPO pipeline 跑通（GPU 3,4,5 配置）。
  原则：**不修改 RLinf 源码**，所有适配在 `qwen_nav_policy.py` / `genark_env.py` / config 内完成。

  **配置层修复**：
  - Hydra `component_placement` 从合并键（`rollout, actor: 2`）改为单独键，让命令行可单独 override。
  - `actor.model.action_dim` 必须等于 `max_new_tokens`（`action_level` logprob 类型要求 `single_action_dim` 一致）。原配 200 不匹配 qwen_nav 默认 512，统一为 512；smoke 改 32 时同改 action_dim=32。
  - 新增 `actor.model.image_size: [448, 448]`：所有图像在 prompt 构建/init 用同一分辨率，避免 init 用 224×224 blank 图（256 patch/张）但 rollout 渲染 640×480（patch 数不同）导致 vision encoder shape mismatch。
  - 启动需 `EMBODIED_PATH=...` 环境变量（Hydra `searchpath` 引用了 `${oc.env:EMBODIED_PATH}`）。
  - genesis conda 环境的 torch 2.5.1 inductor 模块损坏（`get_free_symbols` 缺失），需 `TORCHDYNAMO_DISABLE=1` 绕过 `@jit_fuser` 装饰 `squared_relu` 时的 inductor 导入。

  **FSDP / 模型层修复**（`qwen_nav_policy.py`）：
  - `_no_split_modules`: 加载后从 inner model 拷贝，让 FSDP 按 transformer block 包裹，否则 inner module 参数被打平后 `self.model.device` 失败。
  - `_model_device` property: `next(self.model.parameters()).device` 在 FSDP 上下文里失败时回退到 `next(self.parameters()).device`，再回退到 `LOCAL_RANK` 环境变量。
  - 删除 `_load_model` 里的 `requires_grad_(False)`（actor optimizer 需要可训练参数）。
  - 添加 `gradient_checkpointing_enable/disable/enable_input_require_grads` 委托方法到 inner model。

  **teacher-forcing 张量形状修复**：
  - `_compute_teacher_forcing_logprobs` 必须传 `mm_token_type_ids`（Qwen2.5-VL M-RoPE 必需）。
  - `default_forward` 内部按 `image_grid_thw.prod(dim=-1) > 0` 过滤有效 image grid 行；按 `self._total_patches` 切 `pixel_values`（所有图固定大小，patches 永远 = `_total_patches`）。

  **GenArk env 修复**（`genark_env.py`）：
  - `chunk_step()` 之前 rewards 返回 `None`，造成 actor 端 `KeyError: 'rewards'`。改为正确返回 `(num_envs, num_chunks)` 形状的 reward 张量。

  **关键 bug：dormant step 的 forward_inputs 必须填充**（`predict_action_batch`）：
  - 症状：`process_nested_dict_for_train` 报 `IndexError: index 4 out of bounds for size 4`，且 `prev_logprobs.shape=(4, 2, 32)` 而 `forward_inputs[*].shape=(2, 2, ...)`，T 维度不一致。
  - 根因：当所有 env 进入 dormant，policy 返回 `merged_fi = {}`（空 dict）。RLinf 的 `EmbodiedRolloutResult.append_step_result` 里 `if result.forward_inputs:` 判断为 False，跳过 append；而 `prev_logprobs` 用 `zero_lp` 填了仍 append 进去。导致最终 trajectory 里 `forward_inputs` 的 T 比 `prev_logprobs` 少。
  - 修复：dormant 时也用 `_make_blank_forward_inputs()` 拼出 N 个 entries，保证 forward_inputs 永远非空。dormant 样本通过 framework 的 `loss_mask`（从 `dones` 计算）自动屏蔽，对梯度无污染（仅浪费一点 forward 算力）。

  **算法约束**：
  - `env.train.auto_reset=false` 且 `ignore_terminations=false` 是 GRPO 必需（否则 `loss_mask` 不会计算，actor 训练时 `zeros_like(loss_mask)` 报 `NoneType` 错）。
  - rollout 流水线对张量形状的隐含合约：所有 `forward_inputs[*]` 张量、`prev_logprobs`、`rewards` 等必须满足 `shape[0] * shape[1] == rollout_size`，由 `process_nested_dict_for_train` 用同一个 `shuffle_id` 索引。

  smoke17 跑通 2 个 epoch 完整循环（rollout → adv → train → eval），无报错。数值全 0 是因为 smoke test 用极小参数（4 envs, max_episode_steps=5），GenArk 场景的 episode 很快 exhaust 全部 dormant，无真实奖励信号——这是预期行为。

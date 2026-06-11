下面是一份可以直接交给另一个 AI/工程 agent 的 **LaViRA 接入 RLinf 详细实施说明**。本文严格参考 `/home/clk/workspace/template/lavira-code` 的 `feat/merged-la-va` 分支，而不是 `main` 分支或早期 `src/` 训练代码。

---

**目标**

将 `/home/clk/workspace/template/lavira-code` 中的 LaViRA navigation decision 机制接入当前 RLinf GenArk 训练架构。

当前 RLinf 已具备：

```text
1. GenArk / Genesis multiscene env
2. fixed decision steps:
   env.train.max_decisions_per_rollout_epoch
3. QwenNavPolicy:
   LaViRA-style 单步 prompt rebuild
   4-dir views
   history images
   JSON action parse
   stop check 雏形
   forward_inputs / prev_logprobs / default_forward 训练链路
4. rollout rank 全局 should_terminate 同步
5. EpisodeBalancer / multiscene scene pool
```

现在要接入的是 LaViRA 原仓库里的完整决策状态机：

```text
TODO list memory
todo_updates
backtrack
negative constraints
working memory
stop double-check
merged LA+VA prompt
LaViRA prompt/action schema
```

---

**重要结论**

不要直接把 LaViRA 的 `VLMReasoningAgent` 原封不动塞进 RLinf。

原因：

LaViRA 原代码是 API 推理型：

```python
LaViRA_API.generate(...)
OpenAI-compatible API
无本地 logits
无 prev_logprobs
无 default_forward
```

RLinf GRPO 训练需要：

```text
actions
prev_logprobs
forward_inputs
response_ids
response_mask
default_forward() 重算 logprobs
```

所以如果目标是 **训练**，应当把 LaViRA 的 prompt / parser / memory 逻辑迁移到 RLinf 的 `QwenNavPolicy` 里，继续用本地 Qwen 生成和训练。

如果目标只是 **eval / teacher API policy**，可以另做 `LaviraAPIPolicy`，但它不能直接参与 GRPO actor training。

---

**关键参考文件**

LaViRA 原仓库：

```text
/home/clk/workspace/template/lavira-code/vlnce_baselines/ZS_Evaluator_mp.py
/home/clk/workspace/template/lavira-code/vlnce_baselines/utils/api.py
/home/clk/workspace/template/lavira-code/vlnce_baselines/utils/prompts.py
/home/clk/workspace/template/lavira-code/vlnce_baselines/utils/prompts_vln.py
/home/clk/workspace/template/lavira-code/LA_IO_EXAMPLES.md
```

RLinf 当前接入点：

```text
/home/nvme03/lck/RLinf/rlinf/models/embodiment/qwen_nav/qwen_nav_policy.py
/home/nvme03/lck/RLinf/rlinf/models/embodiment/qwen_nav/prompts.py
/home/nvme03/lck/RLinf/rlinf/models/embodiment/qwen_nav/action_parser.py
/home/nvme03/lck/RLinf/examples/embodiment/config/model/qwen_nav.yaml
/home/nvme03/lck/RLinf/rlinf/envs/genark/genark_env.py
/home/nvme03/lck/RLinf/rlinf/data/embodied_io_struct.py
```

---

**当前 RLinf QwenNavPolicy 已有能力**

文件：

```text
rlinf/models/embodiment/qwen_nav/qwen_nav_policy.py
```

已有：

```python
class _HistoryCache:
    history_images
    pending_actions
    step_count
    last_parse_ok
    last_err
    last_bbox
    stop_failure_count
    stop_rejection_feedback
```

已有推理主入口：

```python
predict_action_batch(env_obs)
```

它读取：

```python
env_obs["main_images"]
env_obs["extra_view_images"]
env_obs["states"]
env_obs["task_descriptions"]
```

并返回：

```python
action_t
diagnostics["prev_logprobs"]
diagnostics["forward_inputs"]
diagnostics["is_decision"]
diagnostics["parse_ok"]
diagnostics["bboxes"]
```

已有训练入口：

```python
default_forward(forward_inputs)
```

这个必须保持兼容。

---

**LaViRA 原始核心逻辑**

文件：

```text
vlnce_baselines/ZS_Evaluator_mp.py
```

核心类：

```python
class VLMReasoningAgent
```

核心方法：

```python
navigate_or_backtrack(...)
generate_todo_list(...)
_apply_todo_updates(...)
_format_todo_for_prompt(...)
double_check_stop(...)
replan_at_backtrack(...)
```

LaViRA 输出 schema 主要包括：

```json
{
  "progress_analysis": "...",
  "reasoning": "...",
  "todo_updates": [...],
  "action": "navigate to forward|navigate to left|navigate to right|navigate to behind|backtrack to <waypoint_id>",
  "stop": true/false,
  "stair": "up"|"down"|false
}
```

merged LA+VA 模式还可能有：

```json
{
  "planning": "...",
  "bbox_2d": [x1, y1, x2, y2],
  "target": "...",
  "reasoning_bbox": "..."
}
```

---

**阶段 1：先接入 LaViRA FT prompt，不启用 TODO/backtrack**

这是最小可训练版本。

目标：

```text
让 RLinf QwenNavPolicy 的 prompt/action schema 与 LaViRA FT 模式对齐。
保留本地 Qwen generate。
保留 GRPO 训练链路。
```

修改：

```text
rlinf/models/embodiment/qwen_nav/prompts.py
```

新增 LaViRA FT prompt：

```text
LA_PROMPT_BACKTRACK_NO_TODO
LA_PROMPT_NO_BACKTRACK_NO_TODO
```

或统一成函数：

```python
build_lavira_nav_prompt_no_todo(...)
```

要求输出 JSON 字段：

```text
progress_analysis
reasoning
action
stop
stair
可选 bbox
```

动作支持：

```text
navigate to forward
navigate to left
navigate to right
navigate to behind
```

暂时不要启用：

```text
TODO
backtrack
second chance replan
API model
```

修改：

```text
qwen_nav_policy.py::_build_prompt()
```

让它根据 config 选择：

```python
prompt_style = "qwen_nav_simple" | "lavira_ft" | "lavira_tt"
```

配置增加：

```yaml
actor.model.prompt_style: lavira_ft
actor.model.use_todo_list: false
actor.model.use_backtrack: false
```

验证：

```text
predict_action_batch 能正常生成 JSON
parse_fail 低
forward_inputs shape 不变
default_forward 可跑
一个 GRPO step 可跑通
```

---

**阶段 2：接入 TODO list memory**

目标：

```text
复现 LaViRA TT 的 TODO list 机制。
每个 env / episode 有独立 TODO 状态。
每步 prompt 带 Current TODO List。
模型输出 todo_updates。
policy 解析并更新 cache。
```

修改：

```text
qwen_nav_policy.py::_HistoryCache
```

新增字段：

```python
todo_list
reasoning_todo
todo_verification_feedback
last_object
```

例如：

```python
self.todo_list = None
self.reasoning_todo = ""
self.todo_verification_feedback = ""
self.last_object = ""
```

`reset()` 时清空。

从 LaViRA 迁移这些函数到 RLinf：

```python
generate_todo_list(...)
_parse_todo_json(...)
_format_todo_for_prompt(...)
_apply_todo_updates(...)
_todo_pending_summary(...)
```

来源：

```text
ZS_Evaluator_mp.py: generate_todo_list
ZS_Evaluator_mp.py: _parse_todo_json
ZS_Evaluator_mp.py: _format_todo_for_prompt
ZS_Evaluator_mp.py: _apply_todo_updates
```

注意：

LaViRA 原实现调用 API：

```python
self.model.generate(...)
```

迁移到 RLinf 后应该用本地 Qwen 的 `_batch_generate(...)`。

实现方式建议：

```text
首次 episode decision 时，如果 cache.todo_list is None：
  构造 TODO generator prompt
  本地 Qwen generate
  parse 成 list[dict]
  写入 cache.todo_list
```

然后每步 `_build_prompt()` 注入：

```text
Current TODO List:
[0] (pending) ...
[1] (completed) ...
```

解析器需要支持：

```json
"todo_updates": [
  {"index": 0, "status": "completed", "result": "..."},
  {"op": "rewrite", "index": 1, "content": "..."},
  {"op": "add", "content": "...", "status": "pending"},
  {"op": "remove", "index": 2}
]
```

注意训练侧：

第一阶段不要把 TODO tokens 纳入 PPO loss。  
`ppo_token_loss_mask` 仍只覆盖：

```text
"action" value
"stop" value
```

否则 reasoning/TODO 输出太长，会导致 PPO ratio 和 variance 变大。

---

**阶段 3：扩展 action_parser**

当前文件：

```text
rlinf/models/embodiment/qwen_nav/action_parser.py
```

当前类：

```python
ParsedAction
```

当前只支持：

```text
navigate to forward/left/right/behind
stop
bbox
stair
```

需要扩展成类似：

```python
class ParsedLaviraAction:
    actions: list[int]
    action_type: str  # NAVIGATE / BACKTRACK / STOP / PARSE_FAIL
    direction: str | None
    waypoint_id: int | None
    todo_updates: list
    bbox: list[float] | None
    bbox_2d: list[float] | None
    target: str | None
    stop: bool
    stair: str | False
    progress_analysis: str
    reasoning: str
    reasoning_todo: str
    reasoning_action: str
    planning: str
    raw_action: str | None
    ok: bool
    err: str | None
```

解析规则：

```text
1. 优先提取 JSON
2. stop=true → ACTION_STOP
3. action startswith "backtrack to" → action_type=BACKTRACK, waypoint_id=int
4. action contains forward/left/right/behind → NAVIGATE
5. unknown → parse fail
```

已有 GenArk 动作空间：

```text
0 = stop
1 = forward
2 = turn_left
3 = turn_right
4 = parse_fail
```

LaViRA macro 映射：

```python
"navigate to forward" -> [1]
"navigate to left"    -> [2, 1]
"navigate to right"   -> [3, 1]
"navigate to behind"  -> [2,2,2,2,2,2,1]
```

训练中建议：

```text
collect_forward_inputs=True 时，不使用 macro pending_actions；
每个 decision 只取第一个动作，或者保持现有 pending_actions 但 is_decision=False 的 replay 不计入 GRPO decision。
```

当前代码已经支持 pending_actions 和 `is_decision`，但要确认 TODO/backtrack 不破坏它。

---

**阶段 4：stop double-check 接入策略**

RLinf 当前已经有：

```text
STOP_CHECK_SYSTEM_PROMPT
build_stop_check_text
parse_stop_check_json
```

且当前逻辑是：

```python
if not collect_forward_inputs:
    eval only 执行 stop check
```

建议保持这个策略。

原因：

训练中 stop check 会多一次 generate，且可能覆盖原 action，导致：

```text
rollout prev_logprobs 对应的是原始 response
实际执行的是 stop-check 后 action
```

这会让训练语义复杂。

所以建议：

```yaml
actor.model.use_stop_check_train: false
actor.model.use_stop_check_eval: true
```

训练时：

```text
模型 stop=true 就执行 stop
```

评估时：

```text
stop=true 后二次验证，拒绝则注入 stop_rejection_feedback
```

---

**阶段 5：backtrack 接入，最后做**

LaViRA 的 backtrack 不是普通动作，它需要环境支持：

```text
保存 waypoint
保存 waypoint pose
backtrack to waypoint_id
重置 agent pose 到历史位置
记录 failed_path
replan_at_backtrack
```

当前 RLinf GenArk env 只支持离散动作：

```text
stop / forward / left / right / parse_fail
```

所以如果要真 backtrack，需要改：

```text
rlinf/envs/genark/genark_env.py
```

新增 per-env waypoint memory：

```python
_waypoints[env_i] = [
  {
    "pose": cam_pos/cam_yaw/current_tri_idx,
    "step": int,
    "rgb": image,
    "world_coords": ...
  }
]
```

新增 action code 或 info 通道：

```text
BACKTRACK action with waypoint_id
```

但当前 action tensor 是 `(N, 1)` int，不携带 waypoint_id。需要两种方案之一：

方案 A：扩展动作空间编码：

```text
1000 + waypoint_id 表示 backtrack
```

然后 env 解析。

方案 B：policy diagnostics 里传 `backtrack_targets`，但 rollout/env 当前只传 action tensor，不传 diagnostics 到 env step，改动更大。

建议先不要做真 backtrack。第一版：

```yaml
use_backtrack: false
```

或者解析到 backtrack 时降级为：

```text
navigate to behind
```

等 LaViRA TODO/action 跑稳定后再做。

---

**阶段 6：如果需要 blocked_directions / depth，需要扩展 obs**

LaViRA 原代码有：

```python
_check_blocked_directions(panorama_images)
```

它依赖 depth：

```text
panorama_images[*]["depth"]
```

RLinf 当前 GenArk obs 只传：

```python
main_images
extra_view_images
states
task_descriptions
```

而且 `EnvOutput.prepare_observations()` 只保留这些 keys：

```text
main_images
wrist_images
extra_view_images
states
task_descriptions
```

文件：

```text
rlinf/data/embodied_io_struct.py
```

所以如果你要传 depth / pose / waypoint info，必须同时改：

```text
genark_env.py 输出 obs
embodied_io_struct.py prepare_observations
merge_env_outputs / split_dict 兼容新字段
qwen_nav_policy.py 读取新字段
```

建议第一版不接 blocked_directions。

---

**配置需要新增**

文件：

```text
examples/embodiment/config/model/qwen_nav.yaml
```

新增：

```yaml
prompt_style: lavira_ft        # lavira_ft | lavira_tt | qwen_nav_simple

use_todo_list: false
todo_generator_max_new_tokens: 512
todo_update_mode: diff         # diff only

use_backtrack: false
backtrack_second_chance: false
max_backtrack_distance: 6.0

use_negative_constraints: true
use_working_memory: true

use_stop_check_train: false
use_stop_check_eval: true
stop_check_max_failures: 3

merged_la_va: false
train_loss_on_todo: false
train_loss_on_reasoning: false
```

训练配置里建议：

```yaml
actor.model.collect_forward_inputs: true
actor.model.max_new_tokens: 256 或 512
actor.model.action_dim: same as max_new_tokens
algorithm.rollout_epoch: 1
env.train.max_decisions_per_rollout_epoch: fixed decision count
env.train.enable_4dir_render: true
```

---

**不要复制 LaViRA API key**

LaViRA 原代码里有硬编码 API key：

```text
ZS_Evaluator_mp.py
utils/api.py
```

不要迁移这些 key。

如果实现 API policy，必须改成：

```python
os.environ["LAVIRA_LA_API_KEY"]
os.environ["LAVIRA_VA_API_KEY"]
```

但训练路径不建议用 API。

---

**推荐实现顺序**

给另一个 AI 执行时，按这个顺序：

1. **Prompt 对齐**
   - 迁移 LaViRA FT prompt 到 `qwen_nav/prompts.py`
   - `prompt_style=lavira_ft`
   - 不开 TODO/backtrack
   - 跑 smoke

2. **Parser 扩展**
   - 扩 `ParsedAction`
   - 支持 `reasoning_todo/reasoning_action/planning/todo_updates/bbox_2d/target`
   - 暂时 backtrack 降级或禁用

3. **TODO memory**
   - 扩 `_HistoryCache`
   - 实现 `_format_todo_for_prompt`
   - 实现 `_apply_todo_updates`
   - 实现 TODO generator
   - `use_todo_list=true`
   - PPO loss 仍只训 action/stop

4. **Stop check eval-only**
   - 保持 train 关闭
   - eval 开启
   - 检查 stop rejection feedback

5. **Backtrack env 支持**
   - 最后做
   - 需要改 action encoding 或 env info 通道
   - 需要 waypoint pose restore

6. **可选 API eval policy**
   - 新增 `LaviraAPIPolicy`
   - 只用于 eval / teacher，不用于 GRPO actor training

---

**Smoke test 清单**

每完成一阶段都跑：

```text
1. python -c import qwen_nav_policy/action_parser/prompts
2. 单 env predict_action_batch
3. collect_forward_inputs=True 下检查：
   prev_logprobs shape = (N, max_new_tokens)
   forward_inputs 所有 key batch dim = N
   response_mask 非空
   ppo_token_loss_mask 非空
4. actor default_forward(forward_inputs) 可跑
5. 1 个 GRPO step：
   不报 shape mismatch
   不报 prev_logprobs None/tensor mixed
   parse_fail 低
   is_decision 正常
```

重点观察日志：

```text
[QwenNav][action-dist]
parse_fail
wrong_stop
num_trajectories
actor/ratio
actor/grad_norm
```

---

**最小可交付版本定义**

如果要先交付一个能训练的 LaViRA 接入版本，范围应该是：

```text
LaViRA FT prompt
4-dir views
history images
JSON action/stop/stair/bbox parser
fixed decision steps
forward_inputs/default_forward 保持兼容
PPO loss only on action/stop
```

不要第一版就做：

```text
API backend
true backtrack
depth blocked_directions
TODO loss
multi-call stop check in training
```

这样最容易跑通，并且和当前 RLinf fixed decision-step 框架最兼容。

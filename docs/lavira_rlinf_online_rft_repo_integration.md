# LaViRA Repo 与 RLinf Online RFT 接入说明

本文面向后续 AI agent / 工程协作者，用于说明如何按照 `lavira-rft` 的仓库协作方式，把 RLinf 中的 online RFT pipeline 兼容到 LaViRA 项目体系中，同时保持两个 git repo 可以共同维护。

本文不要求把 RLinf 整个源码复制进 `lavira-rft`。推荐方式是：

```text
lavira-rft 负责任务定义、prompt/schema、teacher 采样、offline reward/AWR 数据。
RLinf 负责 Genesis/GenArk 环境、multiscene rollout、vLLM QwenNav、online GRPO/RFT 训练。
```

二者通过明确的 adapter、config、script 和文档对接。

---

## 1. 背景

`lavira-rft` 的 README 当前组织方式是：

```text
1. 每个开发者建立独立分支
2. 使用统一 Docker / conda 环境
3. 通过 eval_scripts/*.sh 启动任务
4. run_mp.py 多进程运行 Habitat / VLN-CE evaluation
5. vlnce_baselines/ZS_Evaluator_mp.py 负责 LaViRA teacher policy
6. rl/ 目录负责 offline reward / AWR / manifest 数据处理
```

RLinf 当前 online RFT 能力是：

```text
1. GenArk / Genesis 环境
2. multiscene scene pool
3. GenesisSceneActor 多进程 / 多 GPU 场景隔离
4. QwenNavPolicy 本地 Qwen 生成
5. vLLM embodied rollout backend
6. fixed decision step rollout
7. EpisodeBalancer 场景内 episode 均衡采样
8. GRPO/PPO actor 训练、logprob 重算、checkpoint 保存
```

因此，合理的接入方式不是让 `lavira-rft` 直接训练 RL，而是让它提供 LaViRA 任务语义和 teacher/offline 数据，RLinf 作为 online training backend。

---

## 2. 仓库边界

### 2.1 lavira-rft 应维护的内容

```text
lavira-rft/
  README.md
  eval_scripts/
  run_mp.py
  vlnce_baselines/
  rl/
    reward.py
    label_advantage.py
    build_manifest.py
```

职责：

```text
1. LaViRA prompt 与 action schema
2. Habitat / VLN-CE / ObjectNav / EQA teacher 采样
3. VLMReasoningAgent API teacher policy
4. data_collect 日志格式
5. transitions.jsonl / rewards.json / manifest.jsonl
6. offline AWR / weighted SFT 数据标权
```

### 2.2 RLinf 应维护的内容

```text
RLinf/
  examples/embodiment/config/
  rlinf/envs/genark/
  rlinf/models/embodiment/qwen_nav/
  rlinf/workers/rollout/vllm/
  rlinf/runners/
  run_multiscene.sh
```

职责：

```text
1. Genesis / GenArk online 环境
2. 多场景并发 rollout
3. QwenNav 本地 policy
4. vLLM rollout 生成
5. actor logprob / forward_inputs / response_mask
6. GRPO / PPO online 更新
7. checkpoint / tensorboard / training_metrics.json
```

### 2.3 不建议的做法

不要直接：

```text
1. 把整个 RLinf/ 源码复制进 lavira-rft
2. 在 lavira-rft 内重新实现 GRPO trainer
3. 在 RLinf 内复制一份 lavira-rft/rl/reward.py 后长期分叉
4. 让 LaViRA API teacher 直接当 RLinf actor 训练
```

原因：

```text
LaViRA teacher policy 没有本地 logits / prev_logprobs / default_forward。
RLinf online RL actor 必须能重算 logprob，才能做 GRPO/PPO。
```

---

## 3. 总体 Pipeline

推荐的完整链路如下：

```text
Stage 0: 环境准备
  lavira-rft 使用 README 中的 Docker / conda 环境
  RLinf 使用 Genesis / vLLM / FSDP 训练环境

Stage 1: LaViRA teacher 采样 offline 数据
  lavira-rft/eval_scripts/r2r.sh
    -> run_mp.py
    -> VLMReasoningAgent
    -> logs/data_collect/<exp>/

Stage 2: offline reward / AWR 标权
  python -m rl.label_advantage --root logs/data_collect/<exp>
  python -m rl.build_manifest --root logs/data_collect/<exp>
    -> manifest.jsonl

Stage 3: Qwen warm start
  使用 manifest.jsonl 做 SFT / weighted-SFT / AWR
    -> qwen_lavira_sft_or_awr_checkpoint

Stage 4: RLinf online RFT
  RLinf/run_multiscene.sh
    -> genark_grpo_qwen_multiscene.yaml
    -> QwenNavPolicy
    -> VLLMMultiStepEmbodiedWorker
    -> Genesis multiscene rollout
    -> GRPO actor update

Stage 5: evaluation / export
  RLinf checkpoint
    -> GenArk eval
    -> optionally export to LaViRA-compatible eval script
```

---

## 4. 为什么 LaViRA Policy 不能直接作为 Online RL Actor

`lavira-rft` 里的 LaViRA policy 更像 behavior policy / data collection policy。

它负责：

```text
prompt + image/history -> API VLM response -> macro-action -> execute -> collect transition
```

它适合生成：

```text
calls/NNNN_main_decision.json
waypoints.jsonl
transitions.jsonl
rewards.json
manifest.jsonl
```

但是 RLinf online actor 需要：

```text
actions
prev_logprobs
forward_inputs
response_ids
response_mask
default_forward()
```

因此正确做法是：

```text
LaViRA policy 作为 teacher / behavior policy 采样数据。
QwenNavPolicy 作为 trainable student policy 进入 RLinf online RFT。
```

---

## 5. 接口契约

### 5.1 Action Schema

LaViRA merged LA+VA 推荐 schema：

```json
{
  "progress_analysis": "...",
  "planning": "...",
  "action": "navigate to forward",
  "stop": false,
  "bbox_2d": [x1, y1, x2, y2],
  "point_2d": [x, y],
  "target": "...",
  "todo_updates": []
}
```

RLinf 中应由以下文件维护解析逻辑：

```text
rlinf/models/embodiment/qwen_nav/action_parser.py
rlinf/models/embodiment/qwen_nav/prompts.py
rlinf/models/embodiment/qwen_nav/qwen_nav_policy.py
```

要求：

```text
1. action 字段必须兼容 LaViRA 的 navigate/backtrack/stop 表达
2. bbox_2d / point_2d / target 允许缺失，但缺失时需要 fallback
3. parse_fail 必须有稳定 fallback action
4. prompt_style 应保留 default / lavira_merged / lavira_waypoint 等可切换模式
5. schema 改动必须同时更新 parser 测试
```

推荐测试文件：

```text
tests/unit_tests/test_qwen_nav_lavira_waypoint_parser.py
```

### 5.2 Reward Schema

`lavira-rft/rl/reward.py` 已经定义 offline reward/AWR 的信用单元：

```text
one VLM macro-action = one transition
```

RLinf online RFT 应保持同样的 credit unit：

```text
one QwenNav decision = one online RL transition
```

推荐 reward 对齐：

```text
terminal:
  success + SPL + nDTW

dense:
  geodesic progress
  route-progress potential
  wrong-stop penalty
  parse-fail penalty
```

RLinf 当前相关位置：

```text
rlinf/envs/genark/genark_env.py
docs/reward_design.md
```

维护要求：

```text
1. online reward 和 offline reward 不应长期分叉
2. 若改 reward 公式，应同步写入 lavira-rft 的 reward 文档和 RLinf 的 reward 文档
3. online 训练日志应记录 success / spl / ndtw / distance_to_goal / rewards
4. 每次 val_check 后应写 training_metrics.json 便于观察 online RFT 是否提升
```

### 5.3 Dataset / Episode Schema

LaViRA offline 使用 Habitat / VLN-CE 原始路径。

RLinf online 使用 GenArk / Genesis 预处理数据，例如：

```text
R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
```

必须保持可追踪字段：

```text
episode_id
scene_id
instruction
start_position
start_rotation
goals
reference_path / gt path
```

如果需要把 LaViRA teacher 数据用于 RLinf warm start，应保证 manifest 中至少包含：

```text
call_file
images_dir
instruction
action
bbox_2d
point_2d
target
awr_weight
success
spl
ndtw
scene_id
episode_id
```

---

## 6. RLinf 内部落点

### 6.1 QwenNavPolicy

路径：

```text
rlinf/models/embodiment/qwen_nav/qwen_nav_policy.py
```

职责：

```text
1. 构造 LaViRA-style prompt
2. 读取 main_images / extra_view_images / history
3. 调用 HF 或 vLLM 生成
4. 解析 JSON action
5. 返回 env action
6. 保存 forward_inputs 和 prev_logprobs
7. default_forward() 重算 actor logprobs
```

关键约束：

```text
1. 训练模式不能只返回 action，必须返回 logprob 所需信息
2. vLLM rollout 可以 skip rollout-side logprobs，但 actor 必须 recompute_prev_logprobs
3. prompt/schema 改动不能破坏 response_mask 对齐
4. max_new_tokens 必须与 actor.model.action_dim 对齐
```

### 6.2 vLLM Embodied Rollout

路径：

```text
rlinf/workers/rollout/vllm/vllm_embodied_worker.py
```

职责：

```text
1. 作为 MultiStepRolloutWorker 的 drop-in replacement
2. 初始化 vLLM engine
3. 将 _vllm_generate_fn 注入 QwenNavPolicy
4. 保持 embodied rollout loop 不变
5. 生成结果仍回到 QwenNavPolicy parser
```

配置入口：

```yaml
rollout:
  backend: "vllm_embodied"
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.18
  max_model_len: 4096
  max_num_seqs: 32
```

选择逻辑：

```text
examples/embodiment/train_embodied_agent.py
  cfg.rollout.backend == "vllm_embodied"
    -> VLLMMultiStepEmbodiedWorker
  else
    -> MultiStepRolloutWorker
```

### 6.3 GenArk / Genesis Env

路径：

```text
rlinf/envs/genark/genark_env.py
rlinf/envs/genark/genesis_multiscene.py
rlinf/envs/genark/genesis_server.py
```

职责：

```text
1. 按 scene_id 管理 episode pool
2. 支持 multiscene scene pool
3. 支持 EpisodeBalancer 场景内均衡采样
4. 支持每个 GenesisSceneActor 独立 Ray actor 进程
5. 支持同一个 EnvWorker 内 fan-out 多个 scene actor
6. 保证 GRPO group 不跨 scene
```

关键不变量：

```text
1. group_size 必须整除每个 scene block 的 slot 数
2. 同一 GRPO group 内所有 slot 必须是同一 episode
3. 同一 GRPO group 内所有 slot 必须属于同一 scene
4. cam_pos_hab() 必须按 scene block 做坐标转换
5. 同卡多 scene actor 如果共享 GPU，必要时需要串行锁避免 EGL/Taichi 进程并发崩溃
```

---

## 7. 建议新增的 Repo 级 Glue 层

如果要长期让两个 repo 共同维护，建议在 RLinf 中保留这份文档，并在 `lavira-rft` 中新增一个轻量目录：

```text
lavira-rft/online_rft/
  README.md
  scripts/
    run_rlinf_multiscene.sh
    eval_rlinf_checkpoint.sh
  configs/
    genark_grpo_qwen_lavira_multiscene.yaml
  adapters/
    lavira_to_rlinf_schema.py
    reward_adapter.py
```

如果不希望改 `lavira-rft`，也可以只在 RLinf 中维护：

```text
RLinf/docs/lavira_rlinf_online_rft_repo_integration.md
RLinf/run_multiscene.sh
RLinf/examples/embodiment/config/genark_grpo_qwen_multiscene.yaml
```

最小可维护边界：

```text
lavira-rft 不依赖 RLinf 源码。
RLinf 不复制 LaViRA teacher evaluator。
两者通过 manifest/checkpoint/config 对接。
```

---

## 8. 推荐运行方式

### 8.1 LaViRA Offline Teacher 采样

在 `lavira-rft` 中：

```bash
conda activate lavira
cd /root/lavira-code
bash eval_scripts/r2r.sh EVAL.DATA_COLLECT_LOGGING True
```

输出：

```text
logs/data_collect/<exp>/
  meta.json
  calls/
  images/
  waypoints.jsonl
  transitions.jsonl
  rewards.json
```

### 8.2 AWR 标权与 Manifest

在 `lavira-rft` 中：

```bash
python -m rl.label_advantage --root logs/data_collect/<exp> --beta 0.5 --wmax 20 --bucket 1.0
python -m rl.build_manifest --root logs/data_collect/<exp> --split train --out logs/data_collect/<exp>/manifest.jsonl
```

输出：

```text
manifest.jsonl
```

### 8.3 SFT / AWR Warm Start

训练 Qwen，使其先学会：

```text
1. LaViRA schema
2. 合法 JSON 输出
3. action / bbox / target 字段
4. 基本导航先验
```

输出：

```text
qwen_lavira_sft_or_awr_checkpoint
```

### 8.4 RLinf Online RFT

在 `RLinf` 中：

```bash
cd /home/clk/workspace/RLinf
bash run_multiscene.sh
```

关键配置：

```text
examples/embodiment/config/genark_grpo_qwen_multiscene.yaml
```

建议确认：

```yaml
rollout:
  backend: "vllm_embodied"

actor:
  model:
    model_type: "qwen_nav"
    model_path: <qwen_lavira_sft_or_awr_checkpoint>
```

输出：

```text
logs/<timestamp>-genark_grpo_qwen_multiscene/train.log
results/checkpoints/
tensorboard/
training_metrics.json
```

---

## 9. 共同维护规范

### 9.1 修改 Prompt / Schema

如果修改：

```text
LaViRA prompt
action schema
bbox / point / target 字段
todo_updates / backtrack 字段
```

必须同步检查：

```text
lavira-rft prompt examples
RLinf qwen_nav/prompts.py
RLinf action_parser.py
RLinf parser unit tests
SFT/AWR manifest 字段
online rollout parse_fail 统计
```

### 9.2 修改 Reward

如果修改：

```text
success / SPL / nDTW 权重
wrong_stop_penalty
dense progress
route-progress potential
AWR weight
```

必须同步检查：

```text
lavira-rft/rl/reward.py
RLinf/genark_env.py reward mode
docs/reward_design.md
training_metrics.json 字段
tensorboard eval metrics
```

### 9.3 修改 Rollout / Env

如果修改：

```text
num_envs
group_size
scene_count
scenes_per_gpu
rollout ranks
max_decisions_per_rollout_epoch
```

必须同步检查：

```text
GRPO group 是否跨 scene
rollout rank 是否同步终止
prev_logprobs 是否 None/tensor 混合
global_batch_size 是否能被 group/rank/epoch 约束整除
actor reshape 是否兼容 rollout_epoch
```

### 9.4 修改 vLLM Backend

如果修改：

```text
vLLM version
gpu_memory_utilization
sleep/offload
max_model_len
max_num_seqs
multimodal cache
```

必须同步检查：

```text
_vllm_generate_fn 是否成功注入
QwenNavPolicy 是否仍能 parser 输出
vLLM sleep/wake 是否破坏 multimodal cache
rollout/predict 时间是否下降
GPU 显存是否与 actor/FSDP 共存
```

---

## 10. 后续 AI 接手时的最短理解路径

如果后续 AI 需要快速理解项目，请按这个顺序读：

```text
1. lavira-rft/README.md
   理解 LaViRA repo 的运行方式和协作方式

2. lavira-rft/RL_REWARD_AWR_DESIGN.md
   理解 offline RFT 数据、macro-action、AWR 权重

3. RLinf/docs/lavira-rft.md
   理解 LaViRA 逻辑如何迁移进 QwenNavPolicy

4. RLinf/docs/lavira_rlinf_online_rft_repo_integration.md
   理解两个 repo 如何共同维护

5. RLinf/examples/embodiment/config/genark_grpo_qwen_multiscene.yaml
   理解当前 online RFT 配置

6. RLinf/rlinf/models/embodiment/qwen_nav/qwen_nav_policy.py
   理解 trainable student policy

7. RLinf/rlinf/workers/rollout/vllm/vllm_embodied_worker.py
   理解 vLLM rollout

8. RLinf/rlinf/envs/genark/genark_env.py
   理解 online env/reward/reset/episode sampling
```

---

## 11. 一句话总结

`lavira-rft` 应继续作为 LaViRA teacher、prompt、schema、offline reward/AWR 数据仓库；RLinf 应继续作为 online RFT 训练后端。两者通过 `manifest/checkpoint/config/schema/reward` 五个接口对接，避免把两个大型代码库互相复制，从而保持一个 git repo 内的改动清晰、可审查、可长期维护。

# Reinforce-Ada Inspired Sampling Plan for GenArk / LaViRA RFT

## Goal

当前 episode 824 overfit 的主要瓶颈不是单纯 reward 缺失，而是固定小组采样导致的 GRPO 组内信号坍缩：

```text
group_size = 4
同一个 episode 一次只采 4 条轨迹
4 条轨迹经常全是 wrong_stop / no_stop / all-failure
=> reward_std 接近 0
=> advantage 接近 0
=> gradient 无效或极弱
```

Reinforce-Ada 的核心启发是：

```text
不要固定小 n 后直接训练；
先为同一个 prompt / episode 采更多候选，
再从候选池中选择 reward 有差异的固定大小 group 进入原 GRPO。
```

对我们来说：

| Reinforce-Ada | GenArk / LaViRA RFT |
|---|---|
| prompt | episode / instruction |
| response | navigation trajectory |
| reward | trajectory return |
| correct sample | success / recovered / high-progress trajectory |
| incorrect sample | wrong_stop / no_stop / regression trajectory |
| fixed group size | GRPO `group_size=4` |

第一版目标不是改 actor loss，也不是改 GRPO advantage，而是验证：

```text
episode-level oversampling + diverse group selection
是否能显著提升 group_reward_std、降低 all-failure group、产生稳定 gradient。
```

---

## Phase 1: Validation Stage

### Scope

只面向 episode 824 overfit 验证，不进入全量训练。

```text
scene_id: mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb
episode_id: 824
train_group_size: 4
candidate_k: 8 / 16
```

不做：

```text
不改 actor loss
不改 GRPO advantage
不引入 global baseline
不做跨 episode adaptive retry
不影响普通 multiscene/full-data 训练
```

### Validation Design

#### 1. Oversample Candidates

当前 overfit 是：

```text
num_envs=4
group_size=4
4 条轨迹全部用于训练
```

验证阶段改成：

```text
num_envs = candidate_k
group_size = 4
episode_overfit.enabled = true
所有 env slot 都跑 episode 824
```

推荐起步：

```text
candidate_k=8
```

如果仍然没有足够差异，再试：

```text
candidate_k=16
```

#### 2. Score Each Candidate Trajectory

每条候选轨迹需要收集：

```text
env_id
episode_id
reward_sum
success
success_type
wrong_stop
no_stop
recovered
best_dtg_progress
final_dtg
min_dtg
final_regression
steps_taken
parse_fail_count
```

优先使用 env episode-end diag 中的字段，避免重新计算。

#### 3. Select a Diverse Training Group

从 `candidate_k` 条候选中选 `group_size=4` 条送入 actor。

选择优先级：

```text
1. success / recovered 轨迹，若存在
2. best_dtg_progress 最高但失败的轨迹
3. wrong_stop 轨迹
4. no_stop 或 final_regression 最大的轨迹
```

如果没有 success / recovered，则退化为：

```text
1. best_dtg_progress 最高
2. best_dtg_progress 中位或次高
3. wrong_stop
4. no_stop / 最差 reward / 最大 regression
```

目标不是选择“最好的 4 条”，而是选择：

```text
reward / behavior / progress 有差异的 4 条
```

#### 4. Preserve Original GRPO Interface

actor 仍然只看到：

```text
B = group_size = 4
同一个 episode 的 4 条轨迹
```

因此：

```text
compute_grpo_advantages()
policy_loss()
FSDP actor
```

都不需要改。

#### 5. Validation Metrics

验证阶段必须观察：

```text
grpo/group_reward_std_mean
grpo/acr_reward_std_lt_0_05
grpo/all_failure_group_rate
grpo/all_wrong_stop_group_rate
grpo/group_best_progress_mean
actor/grad_norm
actor/policy_loss_abs
success_type distribution
```

新增或校准日志：

```text
[GRPO][candidate-pool]
episode=824 candidate_k=16 selected=[...]
reward_sum=[...]
success_type=[...]
best_progress=[...]
selection_reason=[...]
```

#### 6. Success Criteria

验证阶段判定有效的最低标准：

```text
selected group_reward_std_mean 明显高于原始 fixed-4
all_wrong_stop_group_rate 下降
all_failure_group_rate 下降或至少更可区分
actor/grad_norm 更稳定非零
ep-diag 中 recovered / near-goal / high-progress 样本被选入训练
```

如果 reward_std 仍接近 0，说明：

```text
candidate_k 还不够
或 reward 本身仍不能区分轨迹
或 policy exploration 太弱，需要提高 temperature / sampling diversity
```

---

## Phase 1 Implementation Points

### Preferred Minimal Path

先做 overfit 专用，不做通用 adaptive retry。

建议配置：

```yaml
env.train.total_num_envs: 8       # or 16
env.train.group_size: 4
env.train.episode_overfit.enabled: true
env.train.episode_overfit.episode_id: 824
```

新增配置：

```yaml
algorithm.candidate_group_selection:
  enabled: true
  candidate_k: 8
  train_group_size: 4
  mode: diverse_by_reward_and_behavior
  only_episode_overfit: true
```

### Code Location

候选选择应放在 env worker 发送 actor 之前：

```text
env rollout results
-> build trajectories
-> candidate group selection
-> rollout_result.to_splited_trajectories()
-> actor
```

理由：

```text
此时已经有完整 trajectory reward / done / success_type diag
actor 不需要知道候选池
GRPO loss 不需要改
```

### Do Not Select Per Token

选择单位必须是完整 trajectory，而不是 token 或 decision step。

原因：

```text
GRPO 的组内比较是 trajectory return 级别
拆 token 会破坏 reward -> advantage 对齐
```

---

## Phase 2: Full Execution Stage

Phase 1 验证有效后，再推广到正式训练。

### 1. Episode-Level Adaptive Budget

每个 episode 初始预算：

```text
candidate_k = group_size = 4
```

如果检测到坍缩：

```text
reward_std < threshold
or all_wrong_stop
or all_no_stop
or all_failure
```

则提高该 episode 后续采样预算：

```text
candidate_k: 4 -> 8 -> 16 -> 32
```

如果连续多轮仍无差异：

```text
暂时降权 / 延后采样 / 标记为 hard
```

### 2. Active Episode Queue

维护 episode 状态：

```text
episode_id
seen_count
candidate_budget
recent_reward_std
recent_success_rate
recent_wrong_stop_rate
recent_best_progress
hardness_tag
```

和现有 EpisodeBalancer 结合：

```text
EpisodeBalancer 负责选 episode
Adaptive sampler 负责决定该 episode 采几条候选
Group selector 负责从候选中选 4 条训练
```

### 3. Diverse Selection for Multi-Episode / Multi-Scene

正式训练中不能只选 high reward 样本，否则会 bias。

每个 episode 内 group selection 目标是：

```text
保留 reward diversity
保留 behavior diversity
保留 hard negative
```

推荐选择规则：

```text
if success exists:
    include best success/recovered
include highest best_progress failure
include representative wrong_stop
include representative no_stop/regression
fill remaining by reward quantiles
```

如果全失败：

```text
select by reward quantiles + behavior type:
top progress
median progress
worst regression
wrong_stop/no_stop
```

### 4. Optional Global Baseline

只有在 Phase 2 稳定后再考虑。

候选池统计：

```text
candidate_mean_reward
candidate_reward_std
candidate_success_rate
```

可能的 advantage：

```text
A_i = r_i - mean(candidate_pool)
```

暂时不建议第一版引入：

```text
去 std normalization
global baseline
custom actor loss
```

原因：

```text
会扩大改动面
会干扰当前 reward / sampling 的因果判断
```

---

## Risks

### 1. Selection Bias

如果只选最好的 4 条，训练会变成 cherry-picking。

解决：

```text
按 behavior / reward quantile 选 diverse group
不是 top-k group
```

### 2. Rollout Cost

candidate_k 增大会增加 VLLM + Genesis 开销。

解决：

```text
只对坍缩 episode 增加 candidate_k
先在 overfit 验证
正式阶段设置 max_candidate_k
```

### 3. Group Semantics

必须保证 selected group 仍然是：

```text
同一个 episode
同一个 scene
group_size 条完整 trajectory
```

否则 GRPO baseline 会错。

### 4. Diagnostic Misalignment

当前 ACR 中 `group_best_progress_*` 曾出现和 `ep-diag` 不完全对齐的风险。

在启用 adaptive selection 前必须校准：

```text
trajectory reward_sum
episode diag
best_progress
success_type
```

---

## Immediate Next Step

推荐先做 Phase 1：

```text
episode 824
candidate_k=8
selected_group_size=4
不改 actor
不改 advantage
只在 env_worker 送 actor 前做 diverse selection
```

验证通过后再扩展：

```text
candidate_k=16
multi-episode adaptive budget
EpisodeBalancer + adaptive sampler 联动
```


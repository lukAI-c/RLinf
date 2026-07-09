# GenArk / LaViRA RFT Candidate Sampling 设计说明

本文档用于说明为什么当前 episode 824 overfit 需要修改 sampling，而不是继续堆 reward；同时给出一版可以直接交给后续 AI 实现的最小改动方案。它基于 `docs/reinforce_ada_genark_sampling_plan.md`，并结合当前训练日志中已经观察到的结果。

## 1. 当前问题

当前 episode 824 overfit 的训练设置是：

```text
scene_id: mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb
episode_id: 824
num_envs: 4
group_size: 4
actor world_size: 1
```

也就是说，每个 GRPO group 只有同一个 episode 的 4 条采样轨迹。这在形式上是正确的：同组 4 条轨迹共享同一 instruction / episode，可以用于 GRPO 组内比较。

但是当前策略输出仍然主要落在：

```text
wrong_stop
no_stop
all_failure group
```

因此经常出现：

```text
4 条轨迹全失败
4 条轨迹 reward 相同或近似相同
reward_std = 0
ACR = 1
actor/grad_norm = 0
```

这时不是 reward 没进入训练链路，而是同一个 group 内没有足够可比较的轨迹差异，GRPO 的优势估计失效。

## 2. 当前日志证据

最新训练日志已经说明链路本身基本通了：

```text
diag=4/4
group_diag_coverage=1.0
ep=824
wrong_stop/no_stop/best_progress 可见
```

这说明 episode diagnostic、reward、GRPO group diagnostic 已经能对齐到每条轨迹。

但训练效果仍然不理想。近期日志中反复出现以下模式：

```text
[GRPO][group-diag] ep=824 reward_std=0.000000 acr=1
success_count=0 all_failure=1 wrong_stop=3 no_stop=1
actor/grad_norm=0.0000
```

也出现过一些 reward 有差异的 group：

```text
[GRPO][group-diag] ep=824 reward_std=17.320509 acr=0
success_count=0 all_failure=1 wrong_stop=4
best_progress_max=1.25
actor/grad_norm=0.032
```

以及：

```text
[GRPO][group-diag] ep=824 reward_std=8.660254 acr=0
success_count=0 all_failure=1 wrong_stop=4
best_progress_max=0.88
actor/grad_norm=0.104
```

这说明：

```text
reward -> advantage -> gradient 链路可以工作；
但固定 4 条采样经常抽不到足够有差异的轨迹；
一旦 reward_std 变成 0，GRPO 梯度就坍缩。
```

因此下一步的核心不是继续加复杂 reward，而是改采样结构，让 actor 看到更有区分度的 4 条轨迹。

## 3. 为什么不是继续堆 reward

当前输出格式合法性已经比早期明显改善。日志中常见：

```text
json: 90%~100%
struct: 90%~100%
field: 80%~100%
parse_fail: 多数 batch 在 0%~20%，偶尔升高
```

格式仍有波动，但已经不是主要瓶颈。主要瓶颈是导航行为：

```text
没有 success / recovered 样本；
wrong_stop 和 no_stop 占满 group；
偶尔接近目标，但仍在错误位置 STOP；
很多 group 内 reward 差异不足。
```

如果继续堆 reward，可能会产生两个问题：

```text
1. reward 公式越来越复杂，因果关系难判断；
2. group 内如果 4 条轨迹行为仍然同质，复杂 reward 也未必产生有效 advantage。
```

所以当前更合理的方向是：

```text
先扩大同一 episode 的候选轨迹数；
再从候选中挑出行为 / reward / progress 有差异的 4 条；
保持原 GRPO loss 和 advantage 公式不变。
```

## 4. 核心思想

借鉴 Reinforce-Ada 的思想：

```text
不要固定小 n 后直接训练；
先为同一个 prompt / episode 采更多候选；
再从候选池中选择固定大小的、差异化的 group 进入 GRPO。
```

对应到当前任务：

| Reinforce-Ada 概念 | GenArk / LaViRA RFT 对应 |
|---|---|
| prompt | episode / instruction |
| response | navigation trajectory |
| reward | trajectory return |
| correct sample | success / recovered / high-progress trajectory |
| incorrect sample | wrong_stop / no_stop / regression trajectory |
| group size | GRPO `group_size=4` |

第一版只做 overfit 验证：

```text
episode 824
candidate_k = 8
selected_group_size = 4
```

actor 仍然只收到：

```text
B = 4
同一个 episode 的 4 条完整轨迹
```

因此以下内容不改：

```text
actor loss
GRPO advantage
global baseline
token-level reward 分配方式
```

## 5. Phase 1 最小实现方案

### 5.1 配置

新增配置：

```yaml
algorithm.candidate_group_selection:
  enabled: true
  candidate_k: 8
  train_group_size: 4
  mode: diverse_by_reward_and_behavior
  only_episode_overfit: true
```

episode 824 overfit 启动时改成：

```yaml
env.train.total_num_envs: 8
env.train.group_size: 4
env.train.episode_overfit.enabled: true
env.train.episode_overfit.episode_id: 824
```

注意：这里的 `env.train.total_num_envs=8` 表示同时生成 8 条候选轨迹，不表示 actor 要训练 8 条。actor 仍然只接收 selector 选出的 4 条。

### 5.2 代码落点

候选选择应该放在 `env_worker.py` 中，位置是：

```text
env rollout results
-> build per-env trajectories
-> candidate group selection
-> selected 4 trajectories
-> rollout_result.to_splited_trajectories()
-> actor
```

原因：

```text
此时已经有完整 trajectory reward；
此时可以读取 episode-end diagnostics；
actor 不需要知道候选池；
GRPO loss 不需要改；
选择单位是完整 trajectory，不会破坏 token / reward 对齐。
```

不要放在 actor 内部，也不要在 token 级别选择。

## 6. 候选轨迹需要记录的字段

每条候选轨迹构造一个 candidate record：

```text
env_id
episode_id
scene_id
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

其中：

```text
reward_sum 来自该 env 的完整 trajectory rewards 求和；
success_type / best_dtg_progress / final_regression 优先来自 GenarkVecEnv episode diagnostic；
如果某些字段缺失，只允许降级为 unknown / 0，不允许影响训练主链路。
```

## 7. 选择算法

目标不是选最高 reward 的 4 条，而是选最有训练价值、最有差异的 4 条。

### 7.1 如果候选池里存在 success / recovered

选择优先级：

```text
1. 选 1 条 best success / recovered
2. 选 1 条 best_dtg_progress 最高但失败的轨迹
3. 选 1 条 wrong_stop 代表轨迹
4. 选 1 条 no_stop / 最大 final_regression / 最低 reward 的 hard negative
```

这样可以构造：

```text
正样本 + 高进展失败 + 错误停止 + 不停止/走偏
```

GRPO 会得到更清晰的组内排序。

### 7.2 如果候选池全失败

这是当前 episode 824 最常见情况。选择规则：

```text
1. 选 best_dtg_progress 最大的轨迹
2. 选 best_dtg_progress 中位或次高的轨迹
3. 选 representative wrong_stop
4. 选 representative no_stop / 最大 final_regression / 最低 reward
```

如果全是 wrong_stop，则按以下维度拉开差异：

```text
best_dtg_progress 高
final_regression 高
reward_sum 高/低分位
steps_taken 差异
parse_fail_count 差异
```

### 7.3 填充策略

如果按上述规则不足 4 条，则用 reward quantile 填充：

```text
top reward
median reward
bottom reward
unused candidate with largest behavior difference
```

必须保证：

```text
selected_indices 唯一；
len(selected_indices) == train_group_size；
selected trajectories 都来自同一个 episode；
selected trajectories 都是完整 trajectory。
```

## 8. 诊断日志

新增候选池日志：

```text
[GRPO][candidate-pool]
episode=824
candidate_k=8
selected=[0,3,5,7]
reward_sum=[...]
success_type=[wrong_stop,no_stop,wrong_stop,wrong_stop]
best_progress=[...]
final_regression=[...]
selection_reason=[top_progress,no_stop,wrong_stop,worst_reward]
```

同时记录 metrics：

```text
grpo/candidate_pool_reward_std
grpo/candidate_selected_reward_std
grpo/candidate_pool_best_progress_max
grpo/candidate_selected_best_progress_max
grpo/candidate_pool_success_rate
grpo/candidate_selected_success_rate
```

原有 metrics 保留：

```text
grpo/acr
grpo/group_reward_std_mean
grpo/all_failure_group_rate
grpo/all_wrong_stop_group_rate
grpo/group_wrong_stop_count
grpo/group_no_stop_count
grpo/group_best_progress_mean
actor/grad_norm
actor/policy_loss_abs
```

## 9. 成功判据

Phase 1 有效的最低标准：

```text
selected group_reward_std_mean 明显高于 fixed-4；
grpo/acr 降低；
actor/grad_norm 更稳定非零；
all_wrong_stop_group_rate 下降；
all_failure_group_rate 可以暂时仍高，但组内 reward / progress 必须更可区分；
candidate-pool 日志显示高 progress / near-goal 样本被选入训练。
```

更理想的变化：

```text
wrong_stop_count 下降；
no_stop_count 不再长期占满；
best_dtg_progress 均值提高；
出现 recovered 或 success；
terminal SR reward 开始进入 group 对比。
```

如果失败，应根据日志判断原因：

```text
candidate_pool_reward_std 仍接近 0
=> candidate_k 不够，尝试 16。

candidate_pool 里有差异，但 selected_reward_std 仍低
=> selector 有 bug 或选择规则太保守。

selected_reward_std 高，但 grad_norm 仍 0
=> reward -> trajectory -> actor 链路要重新查。

selected 里有 near-goal，但训练仍不改善
=> reward 排序或 STOP terminal 奖惩还需要重新审视。
```

## 10. 与当前 reward 设计的关系

当前 nav 模式建议保持简洁：

```text
靠近目标：DTG progress
正确停：terminal SR
错误停：wrong_stop penalty
格式错误：parse_fail penalty
```

Sampling 改动不是为了替代 reward，而是为了让这些 reward 能在 GRPO group 内形成有效排序。

当前最关键的问题不是“有没有 reward”，而是：

```text
一次只采 4 条时，4 条经常全是同类失败；
同类失败之间 reward 差异不足；
advantage 坍缩；
梯度为 0 或极弱。
```

Candidate sampling 解决的是这一层。

## 11. Phase 2 扩展方向

Phase 1 验证有效后，再推广到正式训练。

### 11.1 Episode-level adaptive budget

每个 episode 维护状态：

```text
episode_id
seen_count
candidate_budget
recent_reward_std
recent_success_rate
recent_wrong_stop_rate
recent_no_stop_rate
recent_best_progress
hardness_tag
```

默认：

```text
candidate_k = group_size = 4
```

如果出现坍缩：

```text
reward_std < 1e-6
or all_wrong_stop
or all_no_stop
or all_failure
```

则提升预算：

```text
4 -> 8 -> 16 -> 32
```

如果连续多轮仍无差异：

```text
标记为 hard；
降低近期采样优先级；
等待模型能力提升后再回访。
```

### 11.2 与 EpisodeBalancer 结合

职责划分：

```text
EpisodeBalancer：决定下一个训练哪个 episode；
Adaptive sampler：决定这个 episode 采多少 candidate；
Group selector：从 candidate 中选哪 4 条送 actor。
```

不要让 EpisodeBalancer 同时承担 candidate selection，否则状态会混乱。

### 11.3 Multi-scene 训练

正式训练中必须保持：

```text
同一个 selected group 内的轨迹来自同一个 scene；
同一个 selected group 内的轨迹来自同一个 episode；
group_size 不变；
actor 接口不变。
```

如果一个 EnvWorker 内有多个 scene，需要按 episode / scene 分桶后分别做 candidate selection。

## 12. 实现注意事项

### 12.1 不要按 token 选择

错误做法：

```text
从不同轨迹中挑 token 或 decision step 拼 group
```

原因：

```text
GRPO 的 reward 是 trajectory return；
token-level 拼接会破坏 reward 和 response 的对应关系；
advantage 会错。
```

### 12.2 不要只选 top-k

错误做法：

```text
只选 reward 最高的 4 条
```

原因：

```text
会变成 cherry-picking；
会丢掉 hard negative；
可能让模型只学习短期侥幸行为。
```

正确目标是：

```text
选 reward / behavior / progress 有差异的 4 条。
```

### 12.3 不改 GRPO 公式

第一版不要改：

```text
compute_grpo_advantages()
policy_loss()
actor rollout batch 格式
trajectory reward 求和方式
```

这样可以保证实验因果清楚：

```text
如果效果提升，主要来自 candidate sampling；
如果效果不提升，可以继续定位 reward 或 exploration。
```

## 13. Immediate Next Step

建议下一步直接实现 Phase 1：

```text
episode 824
candidate_k = 8
selected_group_size = 4
env.train.total_num_envs = 8
actor 仍单卡
不改 GRPO loss / advantage
只在 env_worker 送 actor 前做 diverse selection
```

验证后对比 fixed-4：

```text
grpo/acr
grpo/group_reward_std_mean
grpo/all_failure_group_rate
grpo/all_wrong_stop_group_rate
actor/grad_norm
success_type distribution
candidate-pool selected reasons
```

如果 candidate_k=8 仍然不够，再试：

```text
candidate_k = 16
```

只有在 Phase 1 确认有效后，再进入 Phase 2 的 episode-level adaptive budget。

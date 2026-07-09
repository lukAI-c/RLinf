# RL-Struct 风格格式奖励效果对比

本文从最初的简单格式奖励出发，按时间顺序对比 LaViRA waypoint schema 的格式约束效果。统计对象均为 `episode_id=824` 的 overfit 训练日志中 `[QwenNav][action-dist][train]` 行。

这份文档的重点不是把所有收益都归因给最后一步 RL-Struct 风格奖励，而是说明格式约束经历了一个逐步增强过程：

```text
简单 parse_fail 奖励
→ 课程式格式/几何奖励
→ parser 粒度优化 + loss mask
→ RL-Struct 风格分层结构奖励
```

## 演进路线

### 阶段 0：最简单格式奖励

最早的格式约束基本是二值化的：

```text
能解析为动作 → 给正常导航 reward
解析失败 parse_fail → 给惩罚
```

这类设计的问题是信号太粗：

```text
完全乱输出
JSON 大体正确但缺字段
字段齐全但 bbox / point 几何错误
```

都会被混成接近同一种失败。GRPO 只能知道“这条比那条差”，但很难知道差在哪一层。

### 阶段 1：课程式格式 / 几何奖励

后续引入了更长的 curriculum，把训练前期更多放在格式与几何约束上。这一步能降低一部分低级格式失败，但仍然以 `parse_fail / bbox_valid / point_valid` 这几个粗粒度指标为主。

### 阶段 2：parser 粒度优化 + loss mask

再之后，parser 开始更细地区分动作输出，并优化 loss mask，使模型不会因为部分格式错误而完全失去有效训练信号。这一步带来了最明显的 `parse_fail` 下降。

### 阶段 3：RL-Struct 风格分层结构奖励

最后加入 RL-Struct 风格多层结构奖励，把格式 reward 拆成：

```text
JSON valid
Required fields
Field format
Geometry valid
Length valid
```

这一步的核心价值是：低级 JSON/schema 不再和高阶 waypoint geometry 混在一起，模型可以在每一层拿到部分奖励，日志也能清楚显示错误集中在哪一层。

## 对比口径

RL-Struct 风格奖励之前，日志只记录粗粒度格式指标：

- `parse_fail`
- `bbox_valid`
- `point_valid`
- `fallback`
- `backtrack_valid`

RL-Struct 风格奖励之后，日志新增分层结构指标：

- `json`: 是否能解析出 JSON object
- `struct`: 必需字段是否齐全
- `field`: 字段类型与 action grammar 是否合法
- `geom`: bbox / point 几何约束是否合法
- `len`: 输出长度是否在合理区间

因此，直接可比指标是 `parse_fail / bbox_valid / point_valid / fallback / backtrack_valid`；新增的 `json / struct / field / geom / len` 用于解释格式约束为什么更稳定。

需要特别注意：这些指标不是互斥分类。

在当前实现中，`parse_fail` 表示 `ParsedAction.ok == False`，也就是“不能作为可执行动作直接使用”。但一个 `parse_fail` 输出仍然可能已经满足部分结构层级，例如：

```text
能解析出 JSON
必需字段齐全
字段类型基本合法
但 bbox / point 几何无效，或 action 不可执行
```

这种样本会同时计入：

```text
parse_fail = 1
json / struct / field = 1
geom = 0
```

因此，`parse_fail=22.4%` 和 `JSON valid=97.7%` 并不矛盾。更准确地说，当前的 `parse_fail` 已经不是“JSON 语法失败率”，而是“最终动作不可执行率”。

另外，`geom` 与 `bbox_valid / point_valid` 也不是完全同一个指标：

- `geom` 来自 parser 的结构几何 bit，主要检查 bbox/point 是否为合法数组、范围是否正确、point 是否在 bbox 内。
- `bbox_valid / point_valid` 来自 policy/controller 侧的可执行 waypoint gate，通常还会受到视角、depth、fallback、controller 可用性等约束影响。

所以 `Geometry valid=67.0%` 高于 `bbox_valid=46.5%` 是合理的：前者表示“输出结构几何合法”，后者更接近“这个 waypoint 能否被当前控制器有效使用”。

## 数据来源

| 阶段 | 设计含义 | 日志路径 | action-dist 行数 | 决策数 |
|---|---|---:|---:|
| Simple format reward | 最初的简单 parse_fail / bbox_valid 奖励 | `logs/20260701-201525-episode-overfit-824/train.log` | 4 | 143 |
| Long curriculum | 更长的格式/几何课程，但仍是粗粒度指标 | `logs/20260703-193328-episode-overfit-824-longcurriculum-40-70-100-gpu235-depth/train.log` | 234 | 8515 |
| Schema-mask | parser 粒度优化 + loss mask，最接近 RL-Struct 前 | `logs/20260704-124200-episode-overfit-824-schema-mask-gpu235-depth/train.log` | 130 | 4195 |
| RL-Struct-style reward | JSON / struct / field / geometry / length 分层奖励 | `logs/20260704-204546-episode-overfit-824-rlstruct-reward-resume-gpu235-depth/train.log` | 223 | 7593 |

说明：`schema-mask` 是最接近 RL-Struct 前的版本，已经包含 parser 粒度与 loss mask 优化，因此它是更严格的 immediate baseline；`long curriculum` 更适合作为“加入格式强化之前”的弱基线。

## 共同指标对比

| 指标 | Simple format reward | Long curriculum | Schema-mask | RL-Struct-style reward |
|---|---:|---:|---:|---:|
| parse_fail ↓ | 64.3% | 51.8% | 22.3% | 22.4% |
| bbox_valid ↑ | 21.0% | 27.1% | 43.5% | 46.5% |
| point_valid ↑ | 21.0% | 27.1% | 43.5% | 46.5% |
| fallback | 11.2% | 11.2% | 22.5% | 20.5% |
| backtrack_valid | 0.7% | 0.9% | 0.7% | 1.0% |

## 分阶段收益

### 从简单格式奖励到 long curriculum

| 指标 | Simple format reward | Long curriculum | 变化 |
|---|---:|---:|---:|
| parse_fail ↓ | 64.3% | 51.8% | -12.5 |
| bbox_valid ↑ | 21.0% | 27.1% | +6.1 |
| point_valid ↑ | 21.0% | 27.1% | +6.1 |

这一阶段说明：单纯依靠 `parse_fail` 惩罚太粗，延长格式/几何课程可以改善格式，但提升幅度有限。

### 从 long curriculum 到 schema-mask

| 指标 | Long curriculum | Schema-mask | 变化 |
|---|---:|---:|---:|
| parse_fail ↓ | 51.8% | 22.3% | -29.5 |
| bbox_valid ↑ | 27.1% | 43.5% | +16.4 |
| point_valid ↑ | 27.1% | 43.5% | +16.4 |

这一阶段是粗粒度格式错误下降最明显的阶段。它说明 parser 粒度和 loss mask 对训练信号非常关键：模型不再把大量“部分正确”的输出当作完全失败处理。

### 从 schema-mask 到 RL-Struct-style reward

| 指标 | Schema-mask | RL-Struct-style reward | 变化 |
|---|---:|---:|---:|
| parse_fail ↓ | 22.3% | 22.4% | +0.1 |
| bbox_valid ↑ | 43.5% | 46.5% | +3.0 |
| point_valid ↑ | 43.5% | 46.5% | +3.0 |
| fallback ↓ | 22.5% | 20.5% | -2.0 |

这一阶段不能被表述为“parse_fail 大幅下降”，因为低级格式错误已经在 schema-mask 阶段被大幅压低。RL-Struct 风格奖励的优势主要体现在两个方面：

- 高阶格式质量小幅提升：`bbox_valid / point_valid` 从 `43.5%` 提升到 `46.5%`
- 错误归因更清楚：可以区分 JSON、字段完整性、字段类型、几何和长度，而不是只看到一个 `parse_fail`

## RL-Struct 后的分层格式准确率

| 分层指标 | 准确率 |
|---|---:|
| JSON valid | 97.7% |
| Required fields / structure valid | 97.0% |
| Field format valid | 94.4% |
| Geometry valid | 67.0% |
| Length valid | 100.0% |

## 结论

加入 RL-Struct 风格奖励后，格式约束的主要收益不是简单把 `parse_fail` 进一步压低，而是把格式学习拆成了可诊断、可奖励的层级。

从严格 immediate baseline 看：

- `parse_fail` 基本持平：`22.3% -> 22.4%`
- `bbox_valid` 提升：`43.5% -> 46.5%`
- `point_valid` 提升：`43.5% -> 46.5%`
- `fallback` 略降：`22.5% -> 20.5%`

这说明 parser 粒度与 loss mask 已经解决了大部分粗粒度 parse failure；RL-Struct 风格奖励进一步强化的是 bbox / point 这类结构内部有效性，而不是 JSON 语法本身。

从更早的 long curriculum baseline 看：

- `parse_fail` 从 `51.8%` 降到 `22.4%`
- `bbox_valid` 从 `27.1%` 提升到 `46.5%`
- `point_valid` 从 `27.1%` 提升到 `46.5%`

这说明整套格式约束改造，包括课程式格式奖励、parser 粒度、loss mask 与 RL-Struct 风格奖励，对格式稳定性是有效的。更准确地说：

```text
parse_fail 的大幅下降主要来自 parser 粒度 + loss mask；
RL-Struct 风格奖励主要提供分层监督、可诊断性和高阶 geometry 约束的小幅提升。
```

## 关键观察

RL-Struct 后的分层指标显示：

```text
JSON valid      ≈ 97.7%
Struct valid    ≈ 97.0%
Field valid     ≈ 94.4%
Geometry valid  ≈ 67.0%
Length valid    ≈ 100.0%
```

因此当前主要问题已经不是“模型不会输出 JSON”，而是：

```text
模型大多能输出合法 JSON 和完整字段，
但 bbox / point 的几何有效性仍然不足，
并且导航成功率仍受 stop 时机与 waypoint 语义质量限制。
```

换句话说，RL-Struct 风格奖励有效地把错误从“低级格式错误”推到了“高阶几何与导航语义错误”。这对后续训练是正向的，因为 GRPO 现在可以获得更细粒度的格式 reward，而不是把所有失败都混成一个 `parse_fail`。

## 图表绘制 Prompt

下面这段 prompt 可直接用于 GPT Image / 科研风格图表生成。

```text
请绘制一张科研论文风格的对比柱状图，标题为“Effect of RL-Struct-style Rewards on LaViRA Waypoint Format Reliability”。

图中包含两个子图：

子图 A：格式约束演进曲线。
横轴为指标：parse_fail（越低越好）、bbox_valid（越高越好）、point_valid（越高越好）、fallback（越低越好）。
纵轴为百分比，范围 0 到 100。
每个指标显示四组柱：
1. Simple format reward：parse_fail 64.3，bbox_valid 21.0，point_valid 21.0，fallback 11.2
2. Long curriculum：parse_fail 51.8，bbox_valid 27.1，point_valid 27.1，fallback 11.2
3. Schema-mask：parse_fail 22.3，bbox_valid 43.5，point_valid 43.5，fallback 22.5
4. RL-Struct-style reward：parse_fail 22.4，bbox_valid 46.5，point_valid 46.5，fallback 20.5

子图 B：Post-RLStruct 的分层结构准确率。
横轴为层级：JSON valid、Required fields、Field format、Geometry valid、Length valid。
纵轴为百分比，范围 0 到 100。
数值分别为：97.7、97.0、94.4、67.0、100.0。

视觉风格要求：
科研学术风格，白色背景，细网格线，清晰图例，柱体使用低饱和蓝色、绿色、橙色。
在每个柱体顶部标注百分比数值。
重点用淡红色箭头标注 parse_fail 从 64.3% 降到 22.4%，用淡绿色箭头标注 bbox_valid 从 21.0% 提升到 46.5%。
同时用小注释标出：parse_fail 的主要下降来自 parser 粒度与 loss mask；RL-Struct-style reward 的主要价值是分层监督与 geometry 约束。
图注写中文：“从简单格式奖励到 RL-Struct 风格分层奖励，LaViRA waypoint 输出的低级格式错误显著减少；RL-Struct 将格式学习拆分为 JSON、结构、字段、几何和长度五层，使剩余瓶颈更清楚地暴露为 waypoint 几何有效性。”
除术语 JSON、RL-Struct、LaViRA、waypoint、parse_fail、bbox_valid、point_valid、schema-mask、loss mask 外，其余文字使用中文。
```

## 一句话总结

格式约束的整体演进是有效的：从简单格式奖励到 parser/loss mask 再到 RL-Struct 风格分层奖励，`parse_fail` 从 `64.3%` 降到 `22.4%`，`bbox_valid / point_valid` 从 `21.0%` 提升到 `46.5%`。其中 RL-Struct 风格奖励的核心贡献是分层监督和诊断能力，使剩余错误集中暴露为 waypoint geometry，而不是低级 JSON/schema 崩塌。

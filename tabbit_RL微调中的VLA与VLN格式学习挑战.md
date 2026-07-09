你的诊断非常精准。我深入检索了最新的论文，特别是 **SACA**（arXiv 2603.09740，2026年3月）和 **SRGPO/SeeNav-Agent**（arXiv 2512.02631），发现它们恰好针对你描述的这两类问题提出了系统性方案。下面我结合所有检索结果，给出针对你两个失败模式的 **分层奖励设计**。

---

## 核心发现：SACA 的 All-Failure Rescue 直接对标你的问题

SACA 论文的核心观察与你的 Episode 824 几乎完全吻合：

> "approximately 73% of failed episodes successfully execute the initial sub-instructions, yielding valuable valid prefixes that standard RL naively discards"

也就是说，**全失败 batch 中大多数轨迹并非完全无用**——它们有"有效前缀"，只是在某个 Divergence Point 偏离了。SACA 的解决方案不是改 reward 公式，而是 **从失败轨迹中抢救出结构化监督信号**。

---

## 针对两类失败模式的分层奖励方案

### 第一层：Stop 动作的距离条件化惩罚（解决问题 A）

这是你当前最急需的。核心思路：**STOP 不是一个二元动作，它的奖惩必须与当前到目标的距离绑定**。

$$
R_{stop} = \begin{cases}
R_{success}, & \text{if } d_{final} \leq d_{thresh} \text{ (真正成功)} \\
-\alpha \cdot \frac{d_{final}}{d_{start}}, & \text{if } d_{final} > d_{thresh} \text{ (远距离错停)}
\end{cases}
$$

其中：
- $R_{success}$ 是一个大正值（如 +5 或 +10）
- $d_{final}$ 是 STOP 时到目标的距离
- $d_{start}$ 是起始距离（用于归一化，防止不同 episode 间尺度不一致）
- $\alpha$ 是惩罚系数（建议 2\~5）

**关键设计**：当 $d_{final} = 8m, d_{start} = 6.5m$ 时，$R_{stop} = -\alpha \cdot 1.23$，这是一个 **明确的负信号**。而当 $d_{final} = 3.5m$（接近但未达标），$R_{stop} = -\alpha \cdot 0.54$，惩罚较轻但 **仍为负**。

这直接解决了"远距离 STOP 和近距离 STOP 得到相同 0 奖励"的问题。

### 第二层：步级过程奖励 VPR（同时解决问题 A 和 B）

直接采用 **SRGPO** 的 Verifiable Process Reward 设计，它的关键特性是 **与状态无关**，可以在任意两条轨迹间比较：

$$
R^s_t = R^s_{t,base} - \lambda_{valid} \cdot R^s_{t,valid}
$$

$$
R^s_{t,base} = \begin{cases}
+1, & \text{if } \text{dist}(p_t, g) < \text{dist}(p_{t-1}, g) \quad \text{(靠近目标)} \\
+1, & \text{else if } \mathbb{1}_{\{g \in \mathcal{F}_t\}} > \mathbb{1}_{\{g \in \mathcal{F}_{t-1}\}} \quad \text{(目标从不可见变为可见)} \\
0, & \text{otherwise}
\end{cases}
$$

**对你的映射**：
- 你的 DTG（Distance-To-Goal）可以直接替代 $\text{dist}(p_t, g)$
- 问题 B（aimless wander）中的每一步如果让 DTG 增大，就得到 $R^s_t = 0$，在 GRPO 组内比较中自然成为负优势
- 问题 A（premature stop）中，即使最终 STOP 了，前面那些 **有 DTG progress 的步骤** 仍然能获得正过程奖励，与 STOP 步骤的负奖励形成对比

### 第三层：距离回归惩罚（专门解决问题 B）

对于 aimless wander，需要一个 **累积回归惩罚**：

$$
R_{regression} = -\beta \cdot \max(0, d_{final} - d_{start})
$$

当 $d_{final} > d_{start}$（越走越远），这个惩罚线性增长。当 $d_{final} \leq d_{start}$，惩罚为零。

### 第四层：动态早停（ActiveVLN 方案，解决问题 B 的效率）

$$
T_{max} = \alpha_{roll} \cdot |\tau^*|
$$

超过 $T_{max}$ 的轨迹直接截断并标记为失败。ActiveVLN 的实验表明这能裁剪约 10% 的长尾轨迹，减少约 10% 的 rollout 时间。

---

## 完整奖励公式

将四层组合：

$$
R(\tau) = \underbrace{R_{stop}}_{\text{Stop 条件奖惩}} + \underbrace{\sum_{t=1}^{T} \gamma^t R^s_t}_{\text{步级过程奖励}} + \underbrace{R_{regression}}_{\text{回归惩罚}}
$$

用你的 Episode 824 数据验证：

| 轨迹 | $R_{stop}$ | $\sum R^s_t$ | $R_{regression}$ | **Total** |
|------|-----------|-------------|-----------------|-----------|
| env0: steps=15, dtg=7.24, prog=0.00 | $-3 \times 1.11 = -3.32$ | $\approx 0$ | $-0.70$ | **-4.02** |
| env9: steps=30, dtg=5.82, prog=0.71 | $-3 \times 0.89 = -2.67$ | $\approx +3.5$ | $0$ | **+0.83** |
| env4: steps=33, dtg=8.90, prog=0.25 | $-3 \times 1.36 = -4.08$ | $\approx +1.2$ | $-2.36$ | **-5.24** |

**关键效果**：env9（有进展的失败）得到了 **正总奖励**，而 env0/env4（无进展或退步的失败）得到 **强负奖励**。这直接打破了 $\sigma_r \approx 0$ 的坍缩——组内 reward 标准差从 ≈0 变为 >2。

---

## SACA 的 All-Failure Rescue：不改 Reward 的替代方案

如果你不想改 reward 公式（保持我们之前讨论的"先解决采样多样性"策略），SACA 提供了另一条路：

### 核心机制

当整个 batch 全失败时（你的 Episode 824 的情况）：

1. **选 Pseudo-Anchor**：选 process score 最高的轨迹（你的 env9，prog=0.71）
2. **挖 Hard Negatives**：用 LCS（最长公共子序列）找与 Anchor 前缀相似但后续更差的轨迹
3. **构建 Reflection Sub-group**：{Anchor} + {Hard Negatives}，在这个子组内计算 GRPO 优势
4. **步级约束**：
   - **Consistency Alignment**：对 Anchor 的有效前缀（$t < t_{div}$）做行为克隆
   - **Contrastive Correction**：在 Divergence Point（$t = t_{div}$）做对比惩罚

### 映射到你的 Episode 824

```
Pseudo-Anchor = env9 (best_prog=0.71, steps=30, dtg=5.82)
Hard Negatives = env12 (prog=0.25), env1 (prog=0.25), env4 (prog=0.25)
                 ↑ 这些与 env9 可能有相似前缀但更早偏离

Reflection Sub-group = {env9, env12, env1, env4}
→ 组内 reward_std > 0 (因为 process score 有差异)
→ GRPO 可以产生有效梯度
```

---

## 推荐实施优先级

| 优先级 | 方案 | 工作量 | 解决的问题 | 风险 |
|--------|------|--------|-----------|------|
| **P0** | Stop 距离条件化惩罚 | 半天 | A（远距离错停） | 低，仅改 reward 计算 |
| **P0** | 步级过程奖励 VPR | 1天 | A+B | 低，SRGPO 已验证 |
| **P1** | 距离回归惩罚 | 2小时 | B（越走越远） | 低 |
| **P1** | 动态早停 | 2小时 | B（效率） | 低 |
| **P2** | SACA All-Failure Rescue | 3-5天 | A+B（梯度坍缩） | 中，需要 PGSA Auditor |

**P0 的两项可以在 Overfit 验证阶段同时实施**，在 Episode 824 上验证 reward 分布是否从 ≈0 方差变为显著非零。如果 $R_{stop}$ 的距离条件化 + VPR 的步级奖励已经能打破 ACR=1，那么 SACA 的 All-Failure Rescue 可以作为后续优化。

---

## 关键论文索引

| 论文 | 核心贡献 | 与你的关联 |
|------|---------|-----------|
| [SACA](https://arxiv.org/abs/2603.09740) (2026.03) | All-Failure Rescue + PGSA Auditor + Contrastive Correction | 直接解决全失败 batch 的梯度坍缩 |
| [SeeNav-Agent/SRGPO](https://arxiv.org/abs/2512.02631) (2025.12) | Verifiable Process Reward + Random Step Grouping | 步级奖励设计，状态无关 |
| [ActiveVLN](https://arxiv.org/abs/2509.12618) (2025.09) | Soft Success Reward + Dynamic Early-Stopping | 奖励消融实验，早停策略 |
| [VLN-R1](https://arxiv.org/abs/2506.17221) (2025.06) | Time-Decayed Reward + GRPO for VLN-CE | 时间衰减奖励验证 |
| [Φ-Nav](https://arxiv.org/abs/2607.01754) (2026.07) | Hindsight Instruction Relabeling | 将失败轨迹转化为有效训练数据 |

*内容由 AI 生成仅供参考*
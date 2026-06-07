现在可以给你一个系统的判断了：

汇总分析
1. "Worker exhausted: all 6 episodes done" 的真相
这是 eval env 的正常行为，不是 bug：

val_check_interval=2 → 每 2 步运行一次 eval
eval 有 6 个 env，单次跑完就全部 done → 触发 exhausted 消息
train env 的 cyclic reshuffle 工作正常（pool 0→9 多次出现）
2. 奖励信号变化
趋势	说明
rewards 6.6→10.1	在涨，但即将触及上限（格式奖励上限=20×0.5=10）
nDTW train 0.035→0.002	在下降！Step 8 几乎归零
nDTW eval 0.086→0.102	eval 有微弱改善，但非常小
success 全程 = 0.0	无一成功
关键问题：rewards 提升来自模型学会输出格式合规的 JSON（format_reward=0.5），而不是导航进步。nDTW 在训练集上反而下降，说明模型在过拟合格式输出，失去了导航探索性。

3. 梯度/训练稳定性
指标	现象	诊断
ratio 0.506~0.833	每步训练后策略漂移严重	micro_batch=1，96次更新，漂移累积
grad_norm 16~42	波动大，clip前梯度极大	reward std 1.3~1.7 导致 advantage 方差大
approx_kl 正负交替	理论上 KL≥0，负值异常	可能是近似公式在 ratio<1 时数值问题
policy_loss 普遍为负	策略在"优化"	但优化方向是 format reward，非导航
最紧迫的问题
format reward 把训练拉歪了：模型用 8 步学会了输出合法 JSON（rewards→10），但 nDTW 和 SR 为 0，梯度几乎全来自格式信号。建议现在考虑：把 format_reward_coef 降为 0（或 0.1），让 nDTW/SR 主导信号，看 agent 是否会开始真正朝目标移动。
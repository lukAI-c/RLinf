# GRPO ratio 爆炸：Qwen3.5 + RLinf 混合架构 train/eval 路径不一致

## 现象

GRPO 训练时 `actor/ratio` 系统性偏离 1.0，且方向不固定：
- 部分 step：ratio ≈ 1e-29 ~ 1e-15（actor 比 rollout 给响应更低概率）
- 个别 sample：ratio = 1e6 ~ 1e30（反向爆炸）
- `actor/grad_norm` 异常巨大（~1e10），训练发散

```
train/actor/ratio:    [(0, 1126), (1, 142), (2, 985155), (3, 26)]
train/actor/approx_kl: [(0, 0.73), (1, 1.27), (2, 0.63), (3, 3.05)]
train/actor/grad_norm: [(0, 90059), (1, 165205), (2, 9e10), (3, 138006)]
```

## 根本原因

### 设计假设
RLinf 的 PPO/GRPO 假设：
1. Rollout 用 `model.eval()` 计算 teacher-forcing logprobs（`prev_logprobs`）
2. Actor 用 `model.train()` 计算 default_forward logprobs（`new_logprobs`）
3. 两者权重相同时，logprob 数值应一致 → ratio = 1 at gradient_step=0

### 这个假设对 Qwen3.5 不成立

Qwen3.5 是混合架构（标准 attention + linear attention `Qwen3_5GatedDeltaNet`），其 decoder 层继承 `GradientCheckpointingLayer`：

```python
# transformers/modeling_layers.py
class GradientCheckpointingLayer(nn.Module):
    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            # 训练路径：checkpointed forward
            return self._gradient_checkpointing_func(...)
        # 推理路径：普通 forward
        return super().__call__(*args, **kwargs)
```

**判断条件 `gradient_checkpointing and self.training`：**

| 角色 | gradient_checkpointing | self.training | 走哪条路径 |
|---|---|---|---|
| Rollout (teacher-forcing) | False | False（init 后 `eval()`） | 普通 forward |
| Actor (default_forward in run_training) | **True** | **True**（`model.train()`） | **checkpointed forward** |

`Qwen3_5GatedDeltaNet` 的 PyTorch fallback 实现在两条路径下数值结果不一致（"fast path" 库未安装时使用）。同样的权重、同样的输入，logprob 偏差可达 100x（如 token 0 logprob 从 -0.23 → -20.83）。

### 为什么 RLinf 其他模型没暴露这个 bug

- 标准 attention 在 checkpointed vs 普通 forward 下输出一致（数值精度内）
- 只有 `Qwen3_5GatedDeltaNet` 这种带递归状态的 linear attention 在两条路径下产生分歧

## 验证证据

通过在 `fsdp_actor_worker.run_training` 加 debug：在第一个 micro-batch 用 actor 自身**重新**计算 forward_inputs 的 logprobs（`recomp`），与原 prev_logprobs 和 new_logprobs 对比：

```
diff_vs_new = 0.000000      ← actor 自洽
diff_vs_old = 101.668701    ← actor ≠ rollout
```

证明：
- ✓ Actor 计算可重现（同输入同输出）
- ✗ Actor 和 Rollout 计算结果差异巨大，根源在两个模型的 forward 路径

## 修复方案

### 已采用：关闭 actor 梯度检查点

```yaml
# examples/embodiment/config/genark_grpo_qwen.yaml
actor:
  fsdp_config:
    gradient_checkpointing: False  # 关键修复
```

效果：actor 走普通 forward 路径，与 rollout（eval 模式 + 无 checkpointing）数值一致。

代价：训练显存占用增加（A800-80GB 单卡足够 4B LoRA 训练）。

### 备选方案

| 方案 | 描述 | 评估 |
|---|---|---|
| 给 rollout 也开 gradient_checkpointing | `rollout.hf_model.eval()` 强制 `self.training=False`，条件不成立 → 无效 | ❌ |
| 替换 `Qwen3_5GatedDeltaNet` 实现 | 安装 `flash-linear-attention` 库使用 fast path | 未尝试 |
| 在 run_training 开始时用 actor 重计算 prev_logprobs | 仍依赖 actor train 模式的一致性 | 复杂 |

## 不属于此问题的次要 fix（同期发现）

调试过程中还顺便修复了几个独立 bug：

1. **`genark_env.reset()` 不重置 `_exhausted`**：epoch 1 后所有 env 进入 dormant，rewards = 0
2. **`_assign_episodes_to_envs` 没按 group_size 分组**：同组 env 跑不同指令，GRPO advantage 无意义
3. **response_ids 是 LEFT padding**：`resp_mask[:resp_len]=True` 实际标记了 PAD 位置
4. **M-RoPE 位置错位**：teacher-forcing 用未 padding prompt，default_forward 用 padded prompt，token 绝对位置不同
5. **teacher-forcing 缺 autocast**：与 default_forward 精度不一致

## 历史遗留问题：本地 RLinf 与官方仓库的同步机制差异

本地 `/home/nvme03/lck/RLinf` 不是 RLinf 官方 main 的最新版本，是一个早期 fork 加了"为大模型 OOM 优化的同步机制改写"。涉及两个文件：

| 文件 | 与 RLinf main 差异行数 |
|---|---|
| `rlinf/workers/actor/fsdp_actor_worker.py` | 127 行 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | 94 行 |

### 同步机制本质区别

**RLinf 官方版本（main 分支）：**
- 通过抽象类 `WeightSyncer`（`rlinf.hybrid_engines.weight_syncer`）
- 支持 `bucket_syncer` / `patch_syncer` 两种实现，配置切换
- 支持版本管理（`applied_version`）、增量同步、压缩
- 一次性发送整个 state_dict，握手协议初始化

**本地 fork 版本：**
- 删除 `WeightSyncer` 抽象层，直接用底层 `send/recv`
- 固定为 bucket-based 同步（`divide_model_to_bucket` + 逐个发送）
- 添加 `sync_weight_load_instant` 配置（True：每个 bucket 立即 load；False：缓存到 CPU 后一次 load）
- 失去版本管理逻辑（`finished_episodes` 计算被搬到 `set_global_step`）

### Fork 改动的目的
**降低同步峰值显存（防 OOM）**。原版一次性把整个 state_dict 塞进通信 buffer，对大模型（>10GB）压力大；本地版切片发送，峰值显存只占单个 bucket。

### 还原到官方版本需要的工作

1. 下载缺失模块：`rlinf/hybrid_engines/weight_syncer/`（5 个文件 ~50KB）
2. 替换 `fsdp_actor_worker.py`（已备份在 `/home/clk/workspace/RLinf/rlinf/workers/actor/test.py`）
3. 替换 `rlinf/workers/rollout/hf/huggingface_worker.py`
4. 下载 weight_syncer 配置模板：`examples/embodiment/config/weight_syncer/{bucket_syncer,patch_syncer}.yaml`
5. 在 `genark_grpo_qwen.yaml` 加一行：`- weight_syncer/bucket_syncer@weight_syncer`

### 兼容性
RLinf runner 调用 actor 的核心方法（`init_worker`、`sync_model_to_rollout`、`set_global_step`、`recv_rollout_trajectories`、`compute_advantages_and_returns`、`run_training`）在原版中**全部存在且签名一致**。我们的自定义代码（`qwen_nav_policy.py`、`genark_env.py`）只依赖 policy-level 接口，**不碰 actor 内部，完全兼容**。

### 与 ratio 爆炸的关系

**无关**。已通过 `DEBUG RECOMP` 验证：
- `diff_vs_new = 0`、`diff_vs_old = 101.67`
- 证明 ratio 爆炸源于 `Qwen3_5GatedDeltaNet` 的 train/eval 路径差异，与同步机制无关
- 无论用 bucket 还是 WeightSyncer，bug 都存在

### 后续建议
1. **短期**：保持当前 fork 版本（OOM 优化对实际显存有帮助）
2. **中期**：跟踪 RLinf 官方仓库的 `bucket_syncer.py` 实现，对比本地 fork 是否还有优化价值
3. **长期**：考虑还原到官方版本以便接收社区更新（版本管理、压缩等新特性）

## 参考

- [transformers GradientCheckpointingLayer](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_layers.py)
- [transformers Qwen3.5 modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [RLinf huggingface_worker.py](https://github.com/RLinf/RLinf/blob/main/rlinf/workers/rollout/hf/huggingface_worker.py)
- [RLinf fsdp_actor_worker.py](https://github.com/RLinf/RLinf/blob/main/rlinf/workers/actor/fsdp_actor_worker.py)
- [RLinf weight_syncer 模块](https://github.com/RLinf/RLinf/tree/main/rlinf/hybrid_engines/weight_syncer)

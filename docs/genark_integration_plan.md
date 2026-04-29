# GenArk → RLinf 集成方案

## 背景

**GenArk** 是基于 Genesis 渲染器的 VLN（视觉语言导航）评测环境，使用 Matterport3D 场景 + 自然语言指令，通过 UniNaVid（7B VLM）进行导航。本文档描述如何将其集成进 RLinf 分布式 RL 训练框架，支持用 PPO/GRPO 对 UniNaVid 进行 RL 微调。

---

## 架构对比：从 HTTP 服务端/客户端 到 RLinf Ray Channel

### 原 GenArk 架构（HTTP）

```
orchestrator.py (CPU — 调度)
    │
    ├── model_server.py  ←── GPU 2  (UniNaVid 推理，Flask HTTP 服务端)
    │        ↑ HTTP POST /create_session, /step, /delete_session
    ├── env_worker.py    ←── GPU 3  (Genesis 渲染 + NavMesh，HTTP 客户端)
    ├── env_worker.py    ←── GPU 4
    ├── env_worker.py    ←── GPU 5
    ├── env_worker.py    ←── GPU 6
    └── env_worker.py    ←── GPU 7
```

通信路径（每步）：`env_worker` 将 RGB batch 序列化为 bytes → HTTP POST → `model_server` 反序列化 → 推理 → 返回 JSON actions。

### 新 RLinf 架构（Ray Channel）

```
RLinf Runner (Ray — 替代 orchestrator.py)
    │
    ├── Rollout Worker  ←── GPU 2  (UniNaVid 推理，替代 model_server.py)
    │        ↑↓ Ray Channel（进程间零拷贝，无 HTTP、无序列化）
    ├── Env Worker      ←── GPU 3  (GenarkVecEnv，替代 env_worker.py × 5)
    ├── Env Worker      ←── GPU 4   Genesis BatchRenderer
    ├── Env Worker      ←── GPU 5   + NavMesh 物理
    ├── Env Worker      ←── GPU 6
    └── Env Worker      ←── GPU 7
    │
    └── Actor Worker    ←── GPU X  (UniNaVid FSDP 训练，eval-only 时不启动)
```

通信路径（每步）：Env Worker 通过 Ray Channel 直接传 tensor（或 shared memory）到 Rollout Worker，无 HTTP、无端口管理。

### 组件对应关系

| 原 GenArk 组件 | 新 RLinf 组件 | GPU | 通信协议 |
|---|---|---|---|
| `orchestrator.py` | RLinf Runner | CPU (Ray) | — |
| `model_server.py` (HTTP 服务端) | **Rollout Worker** (`UniNaVidPolicy`) | GPU 2 | Ray Channel |
| `env_worker.py` × N (HTTP 客户端) | **Env Worker** (`GenarkVecEnv`) | GPU 3-7 | Ray Channel |
| HTTP `/create_session` | `UniNaVidPolicy.reset_env_cache()` | — | 无网络 |
| HTTP `/step` (RGB bytes → JSON) | `predict_action_batch(env_obs)` | — | 内存传递 |
| `PARALLEL.PORT: 8100` | 无（Ray 自动管理通信） | — | — |

**主要优势**：
- 去掉 RGB→bytes→HTTP→bytes→tensor 的序列化往返，每步节省数十毫秒
- 无需手动管理端口、等待 server 健康检查（`client.wait_healthy()`）
- Env/Rollout 可跨 GPU 灵活调度，后续上 PPO 训练只需加 Actor Worker

---

## 两套系统功能对比

| 维度 | GenArk 现状 | RLinf 集成后 |
|---|---|---|
| 环境接口 | HTTP 通信（orchestrator → env_worker → model_server） | `gym.Env` 子类，`reset()`/`step()` 返回 `torch.Tensor` |
| 策略推理 | 独立 model_server 进程（HTTP） | Rollout Worker（Ray Channel，直接 tensor 传递） |
| 并行化 | Genesis BatchRenderer（N 个 env 同场景） | RLinf 向量化 env（`num_envs` 个并行 worker） |
| 训练 | 无（仅评测） | PPO/GRPO actor-critic，FSDP 分布式训练 |
| 奖励 | 最终 SPL/nDTW 指标 | 稠密奖励（geo_dist delta）+ 稀疏成功奖励 |
| GPU 分配 | 手动写 YAML（MODEL_GPU / WORKER_GPUS） | `component_placement` + `CUDA_VISIBLE_DEVICES` |

---

## 核心设计原则

**解耦仿真与推理**：将 GenArk 的"Genesis 渲染 + NavMesh 物理"放入 RLinf Env Worker，UniNaVid 推理交给 RLinf Rollout Worker。

```
┌──────────────────────────────────────────────────────────────┐
│                      RLinf Runner (Ray)                       │
│                                                               │
│   Actor Worker        Rollout Worker         Env Worker       │
│  ┌──────────┐       ┌─────────────┐       ┌─────────────┐    │
│  │ UniNaVid │       │  UniNaVid   │◄─────►│  GenArk     │    │
│  │  (FSDP)  │       │ (推理/eval) │ Ray   │  VecEnv     │    │
│  │ training │       │             │Channel│  Genesis    │    │
│  └──────────┘       └─────────────┘       │  BatchRend  │    │
│  (Step 3b，         (Step 3a 已实现)       └─────────────┘    │
│   训练时启动)                               (Step 1 已实现)    │
└──────────────────────────────────────────────────────────────┘
```

---

## 实施步骤

### Step 1 — 抽离 GenArk Gym 环境层（预计 2-3 天）

**目标**：把 `env_worker.py` 的 Genesis 渲染 + NavMesh 物理封装成 RLinf 兼容的 `gym.Env`。

**文件结构**：

```
rlinf/envs/genark/
├── __init__.py
├── genark_env.py        # GenarkVecEnv — 主 gym.Env 实现
├── scene_loader.py      # mesh/navmesh 加载（从 env_worker.py 提取）
├── physics.py           # NavMesh 碰撞 + 滑动（从 env_worker.py 提取）
└── renderer.py          # Genesis BatchRenderer 封装
```

**`GenarkVecEnv` 接口**：

```python
class GenarkVecEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes):
        # observation_space: Dict(
        #   rgb:              Box(uint8,  (num_envs, 3, H, W)),
        #   instruction_ids:  Box(int64,  (num_envs, max_seq_len)),
        #   instruction_mask: Box(bool,   (num_envs, max_seq_len)),
        # )
        # action_space: Box(int64, (num_envs,), low=0, high=3)
        #   0=stop, 1=forward 0.25m, 2=turn_left 30°, 3=turn_right 30°
        ...

    def reset(self, env_idx=None) -> tuple[dict, dict]:
        # 返回 GPU tensors，shape (num_envs, ...)
        ...

    def step(self, actions: torch.Tensor) -> tuple:
        # actions: (num_envs,) int64
        # 返回 obs, reward, terminated, truncated, info
        ...
```

**奖励设计**：
- 稠密：`reward = prev_geo_dist - curr_geo_dist`（进度奖励）
- 稀疏：`success_bonus * I[action==stop AND dist_to_goal < 3.0m]`
- info 字典带 SPL / nDTW / SDTW（用于评估日志）

**注册到 RLinf**：

1. `rlinf/envs/__init__.py` → `SupportedEnvType.GENARK = "genark"`，`get_env_cls()` 加分支
2. `rlinf/envs/action_utils.py` → `prepare_actions()` 加 genark 分支（4-way argmax）
3. 新建 `examples/embodiment/config/env/genark_r2r.yaml`

**参考实现**：`HabitatEnv`（同为导航任务）位于 `rlinf/envs/habitat/habitat_env.py`

---

### Step 2 — 配置文件（0.5 天）

新建 `examples/embodiment/config/env/genark_r2r.yaml`：

```yaml
env_type: genark
seed: 0
group_size: 1
reward_coef: 1.0
use_rel_reward: True
total_num_envs: null
auto_reset: True
max_steps_per_rollout_epoch: 512
max_episode_steps: 500

init_params:
  config_path: "configs/uninavid_r2r.yaml"    # genark 原始 YAML
  episodes_file: "data/datasets/OpenNav_R2R-CE_100_bertidx.json"
  scene_datasets: "scene_datasets/"
  max_parallel_per_scene: 64                   # Genesis BatchRenderer 并发数
  success_distance: 3.0
```

新建训练入口 `examples/embodiment/config/genark_ppo_uninavid.yaml`（参考 `gsenv_ppo_openpi_pi05.yaml`）：

```yaml
defaults:
  - env/genark_r2r@env.train
  - env/genark_r2r@env.eval
  - model/uninavid@actor.model        # Step 3 新增
  - training_backend/fsdp@actor.fsdp_config

env:
  train:
    total_num_envs: 32
    max_episode_steps: 500
  eval:
    total_num_envs: 32
    auto_reset: False
    only_eval: True
```

---

### Step 3 — UniNaVid 进入 RLinf 策略体系（1-2 周）

这是最费劲的步骤，分两个阶段：

**阶段 3a：推理模式（eval only，先验证）**

- 在 `rlinf/models/embodiment/uninavid/` 包一层 `UniNaVidPolicy(BasePolicy)`
- 实现 `predict_action_batch()` → 调用 UniNaVid forward，argmax 得到 4-way 动作
- 注册到 `rlinf/config.py` 的 `SupportedModel` enum

**阶段 3b：训练模式（FSDP + RL）**

- 实现 `default_forward()` 返回 `(action_logits, log_probs, entropy, value)`
- 关闭 UniNaVid 的 `feat_cache`（训练态需完整梯度，不能用 KV-cache）
- 添加 value head（参考 openpi `value_after_vlm` 模式）
- 支持 FSDP 分片（7B 模型 4-8 卡）

**风险**：UniNaVid 的视频历史 `feat_cache` 是推理优化，训练时必须关掉，否则梯度断裂。

---

### Step 4 — 先跑通 Eval（推荐先做，0.5 天）

**在做训练前，用 eval-only 模式验证环境移植正确性。**

#### GPU 分配配置

对应原来 `PARALLEL.MODEL_GPU: 2, WORKER_GPUS: [3,4,5,6,7]` 的设置，在 RLinf 中通过 `component_placement` + `CUDA_VISIBLE_DEVICES` 实现：

```yaml
# genark_eval_only.yaml — cluster 段
cluster:
  num_nodes: 1
  component_placement:
    # CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 下的相对 index：
    #   rank 0 = 物理 GPU 2  → Rollout Worker（UniNaVid 推理）
    #   rank 1-5 = 物理 GPU 3,4,5,6,7 → Env Worker（Genesis 渲染）
    rollout:
      placement: 0
    env:
      placement: 1-5
```

**placement 索引说明**：
- `placement` 填写的是 `CUDA_VISIBLE_DEVICES` 中的**相对顺序**，不是物理 GPU 编号
- `CUDA_VISIBLE_DEVICES=2,3,4,5,6,7` → rank 0=GPU2, rank 1=GPU3, ..., rank 5=GPU7
- `placement: 1-5` 表示 5 个 Env Worker 进程，各占一张 GPU

**训练时（Step 5）加入 Actor**：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor:
      placement: 0      # GPU 2 — UniNaVid FSDP 训练
    rollout:
      placement: 1      # GPU 3 — UniNaVid 推理
    env:
      placement: 2-5    # GPU 4,5,6,7 — Genesis 渲染
```

启动命令（eval）：

```bash
cd /home/clk/workspace/RLinf
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
EMBODIED_PATH=examples/embodiment \
python examples/embodiment/train_embodied_agent.py \
    --config-name genark_eval_only \
    rollout.model.model_path=/home/nvme03/lck/genark/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \
    env.eval.total_num_envs=40   # 5 GPUs × 8 envs/GPU，按 VRAM 调整
```

**验收标准**：
- `GenarkVecEnv.reset()` 正常返回 RGB + instruction tensors
- `step()` 返回正确 shape 的 obs/reward/terminated
- 成功率接近 genark 原始基线（34% SR，R2R val_unseen）
- SPL / nDTW 指标在 info 字典中正确计算

---

### Step 5 — PPO 训练实跑（1 周调试）

从小规模开始，逐步扩大：

```
Phase 1: num_envs=8,  单场景,  horizon=100  (验证梯度流)
Phase 2: num_envs=32, 多场景,  horizon=500  (正式训练)
Phase 3: num_envs=128, 全数据集              (规模化)
```

**监控指标**：
- reward 曲线、SR（Success Rate）、SPL
- Genesis 渲染吞吐（应 > 100 episodes/s，否则成为瓶颈）
- GPU VRAM（UniNaVid 7B FSDP + Genesis BatchRender 需精心分配）

---

## 关键风险与缓解

| 风险 | 严重程度 | 缓解方案 |
|---|---|---|
| VRAM 压力：7B FSDP + Genesis BatchRender 同卡 | 高 | 训练/推理/渲染分 GPU（已在 placement 中分离）|
| Feat cache 与训练梯度冲突 | 高 | 训练态强制 `run_type="train"`，`feat_cache` 不初始化 |
| Episode 异构（不同场景 navmesh 大小差异） | 中 | 同场景 batching；navmesh 数据 padding |
| Genesis `is_initialized()` 不存在 | 已解决 | 改用 `gs._initialized` 私有属性检查 |
| UniNaVid 非标 HuggingFace 结构 | 已解决 | 确认为 `LlavaLlamaAttForCausalLM`，用 `load_pretrained_model` 加载 |

---

## 文件改动清单（当前进度）

### 新增文件 ✅ 已完成

```
rlinf/envs/genark/__init__.py                        ✅
rlinf/envs/genark/genark_env.py                      ✅  (NavMesh 物理 + Genesis 渲染 + gym 接口)
rlinf/models/embodiment/uninavid/__init__.py          ✅
rlinf/models/embodiment/uninavid/uninavid_policy.py  ✅  (Step 3a eval 推理，Step 3b 训练待实现)
examples/embodiment/config/env/genark_r2r.yaml       ✅
examples/embodiment/config/model/uninavid.yaml       ✅
examples/embodiment/config/genark_eval_only.yaml     ✅  (Step 4 eval-only 入口)
docs/genark_integration_plan.md                      ✅  (本文档)
```

### 新增文件 ⏳ 待完成

```
examples/embodiment/config/genark_ppo_uninavid.yaml  ⏳  (Step 5 完整训练入口)
```

### 修改文件 ✅ 已完成

```
rlinf/envs/__init__.py     → SupportedEnvType.GENARK + get_env_cls() 分支  ✅
rlinf/models/__init__.py   → _build_uninavid + register_model("uninavid")   ✅
```

### 修改文件 ⏳ 待完成

```
rlinf/envs/action_utils.py  → genark 动作预处理分支（4-way argmax）        ⏳
```

---

## 里程碑

| 里程碑 | 状态 | 目标 | 验收标准 |
|---|---|---|---|
| M1 — Env 移植 | ✅ 已完成 | `GenarkVecEnv` gym 接口 | `reset()`/`step()` 返回正确 shape，无报错 |
| M2 — Eval 跑通 | 🔄 进行中 | UniNaVid eval-only 成功率匹配基线 | SR ≈ 34%（val_unseen R2R） |
| M3 — 训练跑通 | ⏳ 待开始 | PPO 梯度正常流动，reward 上升 | loss 下降，SR 高于基线 |
| M4 — 规模化 | ⏳ 待开始 | 128 envs，完整数据集训练 | 吞吐 > 1000 episodes/hr |

# GenArk → RLinf 代码修改总览

本文档记录了将 GenArk 导航环境集成进 RLinf 框架所做的全部改动，分为**新增文件**、**修改原有 RLinf 核心代码**、**环境依赖**三部分。

---

## 一、新增文件（共 6 个）

### 1. `rlinf/envs/genark/__init__.py`

空模块入口，使 `rlinf.envs.genark` 成为 Python 包。

---

### 2. `rlinf/envs/genark/genark_env.py` （799 行，核心）

**功能**：将 GenArk 的 Genesis 渲染 + NavMesh 物理封装为 RLinf 兼容的向量化 gym 环境。

**主要类与函数**：

| 名称 | 说明 |
|---|---|
| `GenarkVecEnv(gym.Env)` | 主环境类，实现 `reset()` / `step()` / `_build_obs()` |
| `_hab_to_genesis()` / `_genesis_to_hab()` | Habitat 坐标系 ↔ Genesis 坐标系互转 |
| `_batch_find_floor()` | 批量 NavMesh 地面碰撞检测（GPU 张量化） |
| `_batch_get_sliding_position()` | NavMesh 滑动碰撞（沿边缘滑动而非完全阻挡） |
| `_load_episodes()` | 加载 R2R-CE JSON episode 文件 |
| `_tokenize_instruction()` | 导航指令文本 → 定长 int64 token 序列 |
| `_update_camera()` | 更新 Genesis BatchRenderer 相机位姿 |
| `_calculate_initial_yaw()` | 从 quaternion 计算初始航向角 |

**关键设计点**：

- **观测空间**：`Dict(rgb: (N,3,H,W) uint8, instruction_ids: (N,L) int64, instruction_mask: (N,L) bool, instruction_text: list[str], task_descriptions: list[str])`
  - 其中 `task_descriptions` 是为了满足 RLinf `_infer_env_batch_size()` 的探测 key
- **动作空间**：`Box(int64, (N,), [0,3])` — 0=stop, 1=forward 0.25m, 2=turn_left 30°, 3=turn_right 30°
- **奖励设计**：稠密奖励 `prev_geo_dist - curr_geo_dist` + 稀疏成功奖励 `success_bonus`
- **指标**：info 字典带 SPL / nDTW / SDTW（用于评估日志）
- **Genesis 初始化保护**：使用 `gs._initialized`（Genesis 无公开 `is_initialized()` API）
- **`seed` 属性**：RLinf `RecordVideo` wrapper 要求 env 有此属性
- **`task_descriptions` key**：RLinf `_infer_env_batch_size()` 通过此 key 推断 batch size

---

### 3. `rlinf/models/embodiment/uninavid/__init__.py`

模块入口，导出 `get_model`。

---

### 4. `rlinf/models/embodiment/uninavid/uninavid_policy.py` （390 行，核心）

**功能**：将 UniNaVid（`LlavaLlamaAttForCausalLM`，7B LLaVA 架构）封装为 RLinf `BasePolicy`，供 Rollout Worker 调用。

**主要类**：

| 名称 | 说明 |
|---|---|
| `_EpisodeCache` | 每个 env slot 的 RGB 历史帧 + pending actions 缓冲 |
| `UniNaVidPolicy(nn.Module, BasePolicy)` | 主策略类 |

**关键方法**：

| 方法 | 说明 |
|---|---|
| `__init__()` | 加载 UniNaVid 模型，**chdir 到 genark 根目录**再调用 `load_pretrained_model`（因 config.json 里有相对路径 `./model_zoo/eva_vit_g.pth`） |
| `reset_env_cache(env_indices)` | 清空指定 env 的 RGB 历史和 feat cache |
| `push_rgb(env_idx, rgb)` | 向指定 env 的历史帧缓冲添加一帧 |
| `_build_input_ids(instruction)` | 构造带视频/图像特殊 token 的 prompt input_ids |
| `_process_rgb_list(rgb_list)` | RGB 帧列表 → vision tensor |
| `_infer_single(env_idx, instruction)` | 单 env 推理，返回 action 列表 |
| `predict_action_batch(env_obs, **kwargs)` | **RLinf rollout worker 接口**，返回 `(actions, result_dict)`，result_dict 含 `prev_logprobs/prev_values/forward_inputs` |
| `default_forward(...)` | Step 3b 训练接口（**占位，尚未实现**，抛 NotImplementedError） |

**接口约定**（与 RLinf Rollout Worker 对齐）：
- 入参：`env_obs={"rgb": (N,3,H,W), "instruction_text": list[str], ...}`
- 出参：`(action_tensor: (N,) int64, result: dict)`
- `pending_actions`：UniNaVid 一次生成多个 action（`max_actions=2`），多余的 action 缓存到下一步使用

---

### 5. `examples/embodiment/config/env/genark_r2r.yaml`

GenArk R2R-CE 环境配置：

```yaml
env_type: genark
init_params:
  episodes_file: /home/nvme03/lck/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json
  scene_datasets: /home/nvme03/lck/genark/scene_datasets/
  # ... 场景、相机、NavMesh 参数
```

---

### 6. `examples/embodiment/config/model/uninavid.yaml`

UniNaVid 模型配置：

```yaml
model_type: "uninavid"
model_path: /home/nvme03/lck/genark/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2
uninavid_src_path: /home/nvme03/lck/genark   # genark 根目录，用于 sys.path 和 chdir
precision: "bf16"
num_action_chunks: 1
is_lora: false
```

---

### 7. `examples/embodiment/config/genark_eval_only.yaml`

Eval-only 主配置，整合所有 defaults：

```yaml
defaults:
  - env/genark_r2r@env.train
  - env/genark_r2r@env.eval
  - model/uninavid@actor.model
  - training_backend/fsdp@actor.fsdp_config
```

关键配置决策：
- `runner.only_eval: True` → 触发无 Actor 的纯评估路径
- `algorithm.loss_type: embodied_dagger` → 绕过 `add_value_head` 强制要求
- `cluster.component_placement` → rollout 在物理 GPU 2，env 在物理 GPU 3（Ray 用物理 GPU 编号，不受 `CUDA_VISIBLE_DEVICES` 影响）
- `algorithm.replay_buffer.*` → dagger actor 初始化所需（即使 only_eval 也会读）

---

## 二、修改原有 RLinf 核心代码（共 4 个文件）

### 1. `rlinf/envs/__init__.py`

**改动**：注册 GenArk 环境类型。

```python
# 新增 enum 值
class SupportedEnvType(Enum):
    ...
    GENARK = "genark"   # +1 行

# 新增 get_env_cls() 分支（+4 行）
elif env_type == SupportedEnvType.GENARK:
    from rlinf.envs.genark.genark_env import GenarkVecEnv
    return GenarkVecEnv
```

**原因**：RLinf 通过 `SupportedEnvType` enum + `get_env_cls()` 工厂函数做 lazy import，所有新环境都要在这里注册。

---

### 2. `rlinf/models/__init__.py`

**改动**：注册 UniNaVid 模型。

```python
# 新增 builder 函数（+5 行）
def _build_uninavid(cfg, torch_dtype):
    from rlinf.models.embodiment.uninavid import get_model
    return get_model(cfg, torch_dtype)

# 新增 register_model 调用（+6 行）
register_model("uninavid", _build_uninavid, category="embodied", force=True)
```

**原因**：RLinf 通过 `_MODEL_REGISTRY` 字典做 lazy import，所有新模型都要在这里注册。

---

### 3. `rlinf/runners/embodied_runner.py`

**改动一**：`init_workers()` 中 actor 为 None 时不初始化（+1 行）

```python
# 原始
self.actor.init_worker().wait()

# 修改后
if self.actor is not None:
    self.actor.init_worker().wait()
```

**改动二**：`run()` 方法开头新增 `only_eval` 快速路径（+10 行）

```python
def run(self):
    if self.cfg.runner.get("only_eval", False):
        eval_metrics = self.evaluate()          # 直接走 eval，跳过训练循环
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.metric_logger.log(data=eval_metrics, step=0)
        self.metric_logger.finish()
        self.stop_logging = True
        self.log_queue.join()
        self.log_thread.join(timeout=1.0)
        return
    # ... 原有训练循环不变
```

**原因**：RLinf 原始 `EmbodiedRunner.run()` **完全没有处理 `only_eval=True`**，会直接进入训练循环，调用 `generate()`（需要 train dst_ranks）和 `bootstrap_step()`（需要 `train_num_envs_per_stage`），两者在 only_eval 模式下均未初始化，导致运行时崩溃。

---

### 4. `examples/embodiment/train_embodied_agent.py`

**改动**：`only_eval=True` 时跳过 Actor 创建（+3 行，重构原有 actor 块）

```python
only_eval = cfg.runner.get("only_eval", False)
if only_eval:
    actor_group = None          # eval 不需要 Actor（7B FSDP + optimizer ~56GB）
else:
    actor_placement = ...       # 原有逻辑
    actor_group = actor_worker_cls.create_group(cfg).launch(...)
```

**原因**：eval-only 模式下 Actor Worker 会加载完整 7B 模型 + FSDP optimizer states（合计 ~56GB），完全浪费显存，且会触发 AdamW warm-up step 导致 OOM。`only_eval=True` 时只需要 Rollout + Env 两个 worker 即可。

---

## 三、环境依赖补充

在 genesis conda 环境中追加安装（原 genesis 环境缺少）：

```bash
pip install "ray[default]>=2.47.0" hydra-core omegaconf gymnasium gym \
            accelerate einops tensorboard wandb huggingface_hub imageio[ffmpeg] pandas pyarrow
pip install -e /home/nvme03/lck/RLinf --no-deps --ignore-requires-python
```

**已知兼容性问题**：
- `torch 2.5.1+cu121`：`pip install` 时会出现 `ImportError: cannot import name 'get_free_symbols'`，这是 torch 2.5.1 内部 `_inductor` 的版本混淆问题，**不影响实际运行**，用 `TORCHDYNAMO_DISABLE=1` 环境变量绕过
- `Python 3.11.15`：超出 RLinf 元数据限制（要求 ≤3.11.14），用 `--ignore-requires-python` 绕过，无实际影响
- `Genesis` 提示 `torch<2.8.0 not supported`：为 Genesis 自身 warning，不影响运行

---

## 四、运行命令

```bash
conda activate genesis
cd /home/nvme03/lck/RLinf

EMBODIED_PATH=examples/embodiment \
TORCHDYNAMO_DISABLE=1 \
python examples/embodiment/train_embodied_agent.py \
    --config-name genark_eval_only \
    env.eval.total_num_envs=1
```

**GPU 分配**（物理编号，Ray 不受 `CUDA_VISIBLE_DEVICES` 影响）：

| Worker | 物理 GPU | 显存占用 |
|---|---|---|
| Rollout（UniNaVid 推理） | GPU 2 | ~14 GB |
| Env（Genesis 渲染） | GPU 3 | ~5 GB |
| Actor | 不启动 | 0 GB |

---

## 五、尚未实现（Step 3b — 训练）

| 项目 | 文件 | 状态 |
|---|---|---|
| `UniNaVidPolicy.default_forward()` | `uninavid_policy.py` | 占位，抛 NotImplementedError |
| 关闭 `feat_cache` 的训练态 forward | `uninavid_policy.py` | 待实现 |
| Value head for PPO actor-critic | `uninavid_policy.py` | 待实现 |
| PPO 训练入口 config | `genark_ppo_uninavid.yaml` | 未创建 |
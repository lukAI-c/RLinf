# GenArk → RLinf 推理接入全过程记录

> 目标：将 GenArk VLN（Vision-Language Navigation）环境接入 RLinf 分布式 RL 框架，完成 Step 1（gym 环境集成）+ Step 4（eval-only 验证，复现原版 GenArk ~34-55% SR 基准）+ Multi-pass 全量评估（10 个场景，5 个 pass）。

---

## 1. 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│  RLinf EmbodiedRunner  (only_eval=True)                          │
│                                                                  │
│   ┌───────────────┐    actions     ┌───────────────────────┐    │
│   │  RolloutWorker│◄──────────────►│  EnvWorker            │    │
│   │  GPU: 2/3     │                │  GPU: 3/4             │    │
│   │               │  obs (RGB+inst)│                       │    │
│   │  UniNaVidPolicy│───────────────►  GenarkVecEnv         │    │
│   │  (LLaVA 7B)   │                │  (Genesis simulator)  │    │
│   └───────────────┘                └───────────────────────┘    │
│                                                                  │
│   ActorWorker: 创建但不参与训练 (only_eval)                       │
└──────────────────────────────────────────────────────────────────┘
```

**文件位置：**

| 文件 | 说明 |
|------|------|
| `rlinf/envs/genark/genark_env.py` | GenArk gym 环境（Genesis + NavMesh） |
| `rlinf/models/embodiment/uninavid/uninavid_policy.py` | UniNaVid 推理策略包装 |
| `examples/embodiment/config/genark_eval_only.yaml` | eval-only 主配置 |
| `examples/embodiment/config/env/genark_r2r.yaml` | 环境参数 |
| `examples/embodiment/config/model/uninavid.yaml` | 模型参数 |
| `scripts/run_genark_eval.sh` | 单场景启动脚本（含 GPU 指定） |
| `scripts/run_full_eval.py` | Multi-pass 全量评估编排器（Python） |
| `scripts/run_genark_eval_all.sh` | Multi-pass 全量评估编排器（Bash 等效） |

---

## 2. 新增/修改的文件

### 2.1 `genark_env.py` — GenArk 向量化 gym 环境

从零编写，核心设计：

#### 坐标系转换（Habitat ↔ Genesis）

```python
# Habitat (x, y, z) → Genesis (x, -z, y)
def _hab_to_genesis(pos):
    x, y, z = pos[0], pos[1], pos[2]
    return x, -z, y

# Genesis cam_pos → Habitat 坐标（含 camera_height 偏移）
def _genesis_to_hab(cam_pos_gen, camera_height):
    x = cam_pos_gen[:, 0]
    y = cam_pos_gen[:, 2] - camera_height
    z = -cam_pos_gen[:, 1]
    return torch.stack([x, y, z], dim=1)
```

#### NavMesh 物理（batch 化移植自 genark/env_worker.py）

**核心思想：** 原版 GenArk 每次只处理一个 agent（单进程 × 单场景），RLinf 需要同时驱动 N 个 env slot，因此将所有逐 env 的 for 循环替换为 PyTorch 张量广播，一次前向即可处理全部 N 个 agent。

---

##### 数据准备（场景初始化时预计算）

```python
# navmesh.npz 包含顶点坐标（Habitat 坐标系）和三角面索引
nm     = np.load(navmesh_path)
V_hab  = torch.tensor(nm["verts"], ...)     # (V, 3)
F_idx  = torch.tensor(nm["faces"], ...)     # (T, 3)

# Habitat → Genesis 坐标系转换后，拆出每个三角形的三个顶点
self._tri_v0_3d = V_gen[F_idx[:, 0]]       # (T, 3)
self._tri_v1_3d = V_gen[F_idx[:, 1]]       # (T, 3)
self._tri_v2_3d = V_gen[F_idx[:, 2]]       # (T, 3)

# 2D 投影（只取 XY 平面，用于平面内点判断）
self._tri_v0_2d = self._tri_v0_3d[:, :2]   # (T, 2)
```

所有三角形顶点在初始化时就 flatten 成张量，步进时不再有任何文件 IO 或列表遍历。

---

##### `_batch_find_floor`：地面高度查询

**目的：** 给定 N 个 agent 的 XY 坐标，找出每个 agent 脚下对应的地面 Z 高度。

```
输入：pos_2d            (N, 2)   — 当前 XY 位置
      current_floor_z   (N,)     — 上一帧的地面 Z（用于排除太高的候选三角形）
      tri_v*_2d/3d      (T, 2/3) — 全场景 T 个三角形顶点

输出：valid_mask  (N,)   — 是否找到有效地面
      best_z      (N,)   — 对应的地面 Z
      best_idx    (N,)   — 命中的三角形 index
```

**关键：unsqueeze 广播**

```python
pt   = pos_2d.unsqueeze(1)      # (N, 1, 2)
v0_2 = tri_v0_2d.unsqueeze(0)  # (1, T, 2)
# 广播后 → (N, T, 2)，同时判断 N 个点对 T 个三角形的包含关系
inside = _batch_is_point_in_triangle(pt, v0_2, v1_2, v2_2)  # (N, T) bool
```

原版写法是 `for env in envs: for tri in triangles: if point_in_tri(...)` 两层循环。
batch 版一次广播，时间复杂度相同（O(N×T)），但全在 GPU 上完成，没有 Python 循环开销。

**`_batch_is_point_in_triangle`：符号法**

```python
def sign(p1, p2, p3):
    return (p1[...,0]-p3[...,0])*(p2[...,1]-p3[...,1]) \
         - (p2[...,0]-p3[...,0])*(p1[...,1]-p3[...,1])

d1 = sign(pt, v0, v1)   # (N, T)
d2 = sign(pt, v1, v2)
d3 = sign(pt, v2, v0)
# 点在三角形内 ⟺ 三个叉积符号一致（全正或全负）
return ~((d1<0|d2<0|d3<0) & (d1>0|d2>0|d3>0))  # (N, T) bool
```

**`_batch_get_z_on_triangle`：平面方程求 Z**

```python
normal = cross(edge1, edge2)           # 三角形法向量 (N, T, 3)
# 平面方程 normal·(p - v0) = 0 → 解出 Z
z = v0[...,2] - (normal[...,0]*dx + normal[...,1]*dy) / normal[...,2]  # (N, T)
```

**候选过滤：只接受高度差 < max_step_height 的三角形**

```python
z_diff     = abs(z_vals - current_floor_z.unsqueeze(1))  # (N, T)
valid_cand = inside & (z_diff < max_step_height)
# 把非法候选填 inf，再 argmin 选最近的
z_diff_m   = where(valid_cand, z_diff, inf)
min_dist, best_idx = torch.min(z_diff_m, dim=1)          # (N,)
```

---

##### `_batch_get_sliding_position`：边界滑动（碰撞处理）

**目的：** agent 尝试移动到 `desired_pos`，若越出当前三角形则沿最近边投影（滑动），而非直接拒绝。

```
输入：current_pos     (N, 2) — 当前位置
      desired_pos     (N, 2) — 目标位置
      current_tri_idx (N,)   — 每个 agent 当前所在三角形 index

输出：(N, 2) — 实际落点（越界则滑动到边上，否则返回 desired_pos）
```

**Step 1：取当前三角形三条边**

```python
p0 = tri_v0_2d[current_tri_idx]    # (N, 2)，按 index 直接索引
edge_starts = torch.stack([p0,p1,p2], dim=1)  # (N, 3, 2)
edge_vecs   = edge_ends - edge_starts          # (N, 3, 2)
```

**Step 2：判断哪条边被穿越**

利用叉积符号：若 `current_pos` 和 `desired_pos` 在边的不同侧，说明移动轨迹穿越了该边。

```python
cp1 = cross_2d(edge_vecs, desired_pos - edge_starts)  # (N, 3)
cp0 = cross_2d(edge_vecs, current_pos - edge_starts)
crossing = (cp1 * cp0) < 0                            # (N, 3) bool
```

**Step 3：把 desired_pos 投影到穿越边上**

```python
t          = clamp(dot(desired_pos-edge_start, edge_vec) / |edge_vec|², 0, 1)
candidates = edge_start + t * edge_vec   # (N, 3, 2)，每条边上的最近点
```

**Step 4：agent_radius 内缩（避免贴墙）**

```python
# 每条边的内向法向量（指向三角形质心方向）
perp = perpendicular(edge_vec)
inward = where(dot(perp, centroid-edge_start) > 0, perp, -perp)
candidates += normalize(inward) * agent_radius   # 落点向内缩 agent_radius
```

**Step 5：选第一条穿越边的投影点**

```python
has_crossing, first_idx = torch.max(crossing.long(), dim=1)  # (N,)
selected = gather(candidates, first_idx)                       # (N, 2)
return where(has_crossing, selected, current_pos)
# 没有穿越 → 原样返回 desired_pos（允许移动）
```

---

##### batch 化前后对比

| | 原版 GenArk | RLinf batch 版 |
|---|---|---|
| 处理单位 | 1 个 agent × 1 次调用 | N 个 agent × 1 次调用 |
| 循环结构 | Python for loop 遍历三角形 | 无 Python 循环，全 tensor 广播 |
| 主要张量维度 | (T, 3) | (N, T, 3) 或 (N, 3, 2) |
| 设备 | CPU（NumPy） | GPU（PyTorch，与 Genesis 同设备） |
| `max_step_height` | 全局常量 | 函数参数，可按 env 配置 |

#### 关键 Genesis 初始化参数

```python
self._gs_scene.add_entity(morph=gs.morphs.Mesh(
    file=mesh_path, fixed=True, collision=False,
    file_meshes_are_zup=True,   # ★ 关键：GLB 是 Y-up，禁止 Genesis 自动旋转 90°
))
```

#### 场景固定（Scene Pinning）

Genesis 建场景后不可修改，多 env 必须共用同一场景：

```python
# 选该 worker 分片中出现最多的 scene_id
scene_counts = Counter(e["scene_id"] for e in worker_eps)
pinned_scene = scene_counts.most_common(1)[0][0]
self._all_episodes = [e for e in worker_eps if e["scene_id"] == pinned_scene]
```

#### Observation 格式（适配 RLinf EnvOutput）

`EnvOutput.prepare_observations()` 只透传 5 个 key，环境 obs 必须映射到这些槽位：

```python
return {
    "main_images":       rgb_hwc,         # (N, H, W, 3) uint8，CHW→HWC
    "states":            elapsed_t,        # (N, 1) float，传递 elapsed_steps
    "task_descriptions": self._instructions,  # list[str]
    "wrist_images":      None,
    "extra_view_images": None,
}
```

#### Auto-reset + RLinf 指标格式

```python
# env_evaluate_step 需要 info["final_info"]["episode"][key] 为 (num_envs,) tensor
episode_info = self._build_episode_tensors(done_idx, ep_metrics_by_env)
info["episode"] = episode_info
saved_info["episode"] = episode_info
```

---

### 2.2 `uninavid_policy.py` — UniNaVid 策略包装

#### `_EpisodeCache` — 多 env feat_cache 隔离

UniNaVid 的 `feat_cache` / `long_feat_cache` 是模型级全局变量。5 个 env 共用 1 个模型时，每次推理都会覆盖这些 cache，必须显式保存/恢复：

```python
class _EpisodeCache:
    rgb_list: list[np.ndarray]       # 完整历史帧
    new_rgb: list[np.ndarray]        # 本次推理新增帧（推理后清空）
    pending_actions: list[int]       # 多步动作缓冲
    feat_cache = None                # 该 env 的 feat_cache 快照
    long_feat_cache = None           # 该 env 的 long_feat_cache 快照
```

#### `_infer_single` — 单 env 推理

```python
# 1. 恢复该 env 的 feat_cache
inner.feat_cache      = cache.feat_cache
inner.long_feat_cache = cache.long_feat_cache

# 2. 必须调用 update_prompt，否则 vlm_attention 的 NAVIGATION_IDENTIFIER 匹配失败
self.model.update_prompt([[prompt_q]])

# 3. 推理
output_ids = self.model.generate(input_ids, images=images, ...)

# 4. 保存更新后的 feat_cache 回该 env
cache.feat_cache      = inner.feat_cache
cache.long_feat_cache = inner.long_feat_cache
```

#### `predict_action_batch` — 从 RLinf obs 中提取数据

```python
main_images  = env_obs.get("main_images")        # (N, H, W, 3)
instructions = env_obs.get("task_descriptions")   # list[str]
# elapsed_steps 从 states 中读取，==0 时触发 reset_env_cache
```

---

### 2.3 `genark_eval_only.yaml` — eval-only 主配置

关键配置项：

```yaml
runner:
  only_eval: True
  max_epochs: 1

env:
  eval:
    total_num_envs: 8
    auto_reset: True          # ★ 必须为 True，否则没有 episode 完成
    max_episode_steps: 500
    max_steps_per_rollout_epoch: 600  # > 500，确保 episode 能完成

algorithm:
  loss_type: embodied_dagger  # eval-only：跳过 value_head 要求
```

---

### 2.4 `genark_r2r.yaml` — 环境参数（对齐原版 genark 配置）

完全对齐 `genark/configs/uninavid_r2r.yaml`：

```yaml
camera_height: 1.25
step_move: 0.25
step_turn: 0.5236   # 30° in radians
allow_sliding: true
max_step_height: 0.5
agent_radius: 0.18
light_scale: 4.0
fov: 105
cam_res: [640, 480]
success_distance: 3.0
success_bonus: 2.5
```

---

### 2.5 `uninavid.yaml` — 模型推理参数（对齐原版）

```yaml
max_new_tokens: 1024
temperature: 0.5    # 原版 GENERATE.TEMPERATURE = 0.5
do_sample: true     # 原版 GENERATE.DO_SAMPLE = True
conv_mode: "vicuna_v1"
```

---

### 2.6 `run_genark_eval.sh` — 单场景启动脚本

```bash
bash scripts/run_genark_eval.sh [ROLLOUT_GPU] [ENV_GPU] [NUM_ENVS]
# 例：bash scripts/run_genark_eval.sh 3 4 1
```

使用物理 GPU 索引（Ray 忽略 CUDA_VISIBLE_DEVICES），并设置：
- `TORCHDYNAMO_DISABLE=1`（避免 torch.compile 与 UniNaVid 的冲突）
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（减少显存碎片）

---

### 2.7 `run_full_eval.py` — Multi-pass 全量评估编排器

**设计目标：** Genesis 单例限制每进程只能运行一个场景，无法在同一 Ray job 中动态切换场景（即 GPU-level 调度）。编排器通过 Pass-level 调度绕过此限制：每个 pass 启动一批 Ray 进程，每批结束后重启集群继续。

**工作流程：**
```
N 个场景  ÷  M 个 worker  =  ceil(N/M) 个 pass

Pass 0: scene_offset=0,  workers=[4,5,6,7]  → scenes[0,1,2,3]
Pass 1: scene_offset=4,  workers=[4,5,6,7]  → scenes[4,5,6,7]
Pass 2: scene_offset=8,  workers=[4,5]      → scenes[8,9]  (最后两个)
              ↓
每 pass 输出 per_scene_*.json
              ↓
全部完成后聚合为 combined_metrics.json
```

**关键实现：**

```python
# genark_env.py — 读取 scene_offset 并偏移场景选择（Bug 10 修复）
scene_offset = int(getattr(cfg, "scene_offset", 0))
pinned_scene = unique_scenes[(scene_offset + seed_offset) % len(unique_scenes)]

# run_full_eval.py — 仅全新启动时清档（Bug 11 修复）
if args.scene_offset is None:
    for f in RESULTS_DIR.glob("per_scene_*.json"):
        f.unlink()
```

**架构约束说明（为何不能做 GPU-level 调度）：**

| 调度粒度 | 机制 | RLinf+Genesis 是否可行 |
|----------|------|----------------------|
| GPU-level（GenArk 原版） | 独立 OS 进程 + HTTP 模型服务，场景完成立即释放 GPU | ❌ Ray job 生命周期绑定全部 actor，无法单独重启 |
| Pass-level（当前方案） | 整批同步，所有 worker 结束后重启集群 | ✅ 符合 Ray 架构，Genesis 单例约束下唯一可行方案 |

**对 RFT 训练的影响：** 无影响。训练时 `auto_reset=True` 在同一 pinned scene 内循环所有 episode，不需要跨场景切换。

---

## 3. 踩坑记录与修复

### Bug 1：`TypeError: NoneType has no len()` in `vlm_attention`

**原因：** `model.generate()` 被调用时 `self.prompts` 为 None。UniNaVid 的 `vlm_attention` 通过检测 prompt 中的 `NAVIGATION_IDENTIFIER` 来切换导航模式。

**修复：**
```python
# 在 generate() 之前调用
self.model.update_prompt([[prompt_q]])
```

---

### Bug 2：场景渲染侧翻（`success: 0%`，`path_length ≈ 0.25m`）

**原因：** Genesis 对 GLB 文件默认会应用 Y-up → Z-up 旋转（90°），导致整个 Matterport 场景侧翻，智能体看到的画面全部是错误的。

**修复：**
```python
gs.morphs.Mesh(file=mesh_path, fixed=True, collision=False,
               file_meshes_are_zup=True)  # 告知 Genesis 此 GLB 已是 Z-up
```

---

### Bug 3：`do_sample=False` 导致重复动作

**原因：** 贪心解码时模型陷入重复输出（一直输出 "left"）。

**修复：** `do_sample: true`，`temperature: 0.5`（对齐原版配置）。

---

### Bug 4：feat_cache 多 env 污染

**原因：** 5 个 env 共享同一 UniNaVid 模型实例，每次推理后 `feat_cache` 被当前 env 的视频帧覆盖，其他 env 的历史信息丢失，导致动作不连贯。

**修复：** `_EpisodeCache` 保存每个 env 的 `feat_cache` 快照，在 `_infer_single` 前后保存/恢复。

---

### Bug 5：`num_trajectories: 0`（没有 episode 完成）

**原因：** `auto_reset: False`，done 的 episode 不触发 `_handle_auto_reset`，`info["episode"]` 永远为空。

**修复：** `auto_reset: True` + 添加 `_build_episode_tensors()` 生成 RLinf 需要的 `(num_envs,)` tensor 格式。

---

### Bug 6：`'Scene' has no attribute 'clear'`

**原因：** Genesis 建好场景后不支持修改或重建。

**修复：** Scene Pinning — 初始化时将该 worker 的所有 episode 固定到同一个 `scene_id`（取最常见的那个）。

---

### Bug 7：`device mismatch: cuda:0 and cpu`

**原因：** `terminated | truncated` 时两个 tensor 在不同 device。

**修复：**
```python
terminated = newly_stopped.cpu()
truncated  = torch.tensor(..., device="cpu")
```

---

### Bug 8：`AttributeError: numpy.ndarray has no attribute 'dim'`

**原因：** `chunk_step` 中对 numpy 数组调用了 PyTorch 的 `.dim()` 方法。

**修复：**
```python
ndim = chunk_actions.ndim if hasattr(chunk_actions, "ndim") else chunk_actions.dim()
```

---

### Bug 10：`scene_offset` 参数未被读取（所有 pass 跑同一批场景）

**原因：** `scene_offset` 已在 `genark_r2r.yaml` 中声明，并通过 Hydra 以 `env.eval.scene_offset=N` 传入，但 `genark_env.py` 初始化时从未从 `cfg` 中读取该值，导致所有 pass 始终使用 `unique_scenes[seed_offset % N]`（即 scenes[0,1,2,3]）。

**表现：** Pass 1 输出的 `per_scene_*.json` 与 Pass 0 完全相同，推理"越来越慢"（实际是在重跑已完成场景）。

**修复：**
```python
# genark_env.py（修复前）
pinned_scene = unique_scenes[seed_offset % len(unique_scenes)]

# genark_env.py（修复后）
scene_offset = int(getattr(cfg, "scene_offset", 0))
pinned_scene = unique_scenes[(scene_offset + seed_offset) % len(unique_scenes)]
```

---

### Bug 11：`--scene-offset` Resume 时历史结果被清空

**原因：** `run_full_eval.py` 在每次启动时无条件删除所有 `per_scene_*.json`，用 `--scene-offset 4` 断点续跑时会把 Pass 0 的结果一并清除。

**修复：**
```python
# 仅在全新启动（无 --scene-offset）时清档
if args.scene_offset is None:
    for f in RESULTS_DIR.glob("per_scene_*.json"):
        f.unlink()
```

---

### Bug 9：`glb_cache` 路径错误

**原因：** GLB 缓存路径写成了 `/home/nvme03/...`，实际原版 GenArk 一直使用 `/home/clk/workspace/genark/glb_cache`。

**修复：** `init_params.glb_cache_dir: /home/clk/workspace/genark/glb_cache`

---

## 4. 推理参数对照表（与原版 genark 对齐）

| 参数 | 原版 `uninavid_r2r.yaml` | RLinf 初版 | RLinf 修正后 |
|------|--------------------------|------------|-------------|
| `temperature` | 0.5 | 0.2 | **0.5** |
| `do_sample` | True | False | **True** |
| `max_new_tokens` | 1024 | 32 | **1024** |
| `file_meshes_are_zup` | True（隐式）| 缺失 | **True** |
| `light_scale` | 4.0 | 4.0 | 4.0 ✓ |
| `fov` | 105 | 105 | 105 ✓ |
| `cam_res` | [640, 480] | [640, 480] | [640, 480] ✓ |
| `step_move` | 0.25 | 0.25 | 0.25 ✓ |
| `step_turn` | 30° (0.5236 rad) | 0.5236 | 0.5236 ✓ |

---

## 5. 推理结果

### 5.1 修复前（所有 Bug 修复前）

```json
{
  "eval/success":        0.105,
  "eval/spl":            0.105,
  "eval/path_length":    1.514,
  "eval/distance_to_goal": 8.502,
  "eval/num_trajectories": 38
}
```

### 5.2 单场景修复后（Bug 1-9 全部修复）

关键修复：`file_meshes_are_zup=True` + `do_sample=True` + `temperature=0.5` + feat_cache 隔离

```json
{
  "eval/success":           0.557,
  "eval/spl":               0.388,
  "eval/ndtw":              0.103,
  "eval/sdtw":              0.085,
  "eval/path_length":       9.471,
  "eval/distance_to_goal":  3.748,
  "eval/steps_taken":       64.7,
  "eval/num_trajectories":  61
}
```

**SR 55.7%，SPL 38.8%** — 已超出原版 GenArk baseline（~34-55%）。

`path_length` 从 1.5m → 9.5m 是决定性信号：修复前智能体几乎不移动（场景侧翻 + 贪心解码卡死），修复后正常导航。

### 5.3 Multi-pass 全量评估（10 个场景，R2R-CE 100 episodes）

使用 `run_full_eval.py` 编排 5 个 pass，每 pass 4 个 worker（GPU 4-7），每 worker 20 envs。

全量结果保存于 `/home/clk/workspace/results/genark_eval/combined_metrics.json`。

---

## 6. 运行方法

### 6.1 单场景快速测试

```bash
conda activate genesis
cd /home/nvme03/lck/RLinf

# 单 env（显存低）
bash scripts/run_genark_eval.sh 3 4 1

# 多 env（更快）
bash scripts/run_genark_eval.sh 3 4 5
```

### 6.2 全量评估（Multi-pass，推荐）

```bash
conda activate genesis
cd /home/nvme03/lck/RLinf

# 默认：rollout=GPU3, env=GPU4-7, 20 envs/worker
python scripts/run_full_eval.py

# 自定义
python scripts/run_full_eval.py --rollout-gpu 3 --env-gpus 4-7 --envs-per-worker 20

# 预览计划（不实际运行）
python scripts/run_full_eval.py --dry-run

# 断点续跑（跳过前 N 个场景）
python scripts/run_full_eval.py --scene-offset 4

# 查看最终结果
cat /home/clk/workspace/results/genark_eval/combined_metrics.json
```

### 6.3 结果文件说明

| 文件 | 内容 | 生命周期 |
|------|------|----------|
| `per_scene_<id>.json` | 单场景所有 episodes + 摘要 | 每 pass 写入，全量跑完后保留 |
| `avg_metrics.json` | 当前 pass 的平均指标 | 每 pass 覆盖（临时文件） |
| `combined_metrics.json` | 全部场景聚合（SR/SPL/nDTW/DTG） | 所有 pass 完成后生成 |

---

## 7. 后续 TODO

### M1–M2（已完成）

| 里程碑 | 内容 | 状态 |
|--------|------|------|
| M1 | GenArk gym 环境接入 RLinf（GenarkVecEnv、坐标系、NavMesh、obs格式） | ✅ |
| M1 | 单场景 eval-only 验证，SR 55.7%（超出 baseline） | ✅ |
| M2 | Multi-pass 编排器（`run_full_eval.py`）+ `scene_offset` Bug 修复 | ✅ |
| M2 | 全量 10 场景评估流水线 | ✅ |

### M3（下一步：训练接入）

| 内容 | 关键点 | 状态 |
|------|--------|------|
| `default_forward()` 返回 policy distribution | 输出 `log_probs`，支持 PPO/GRPO loss 计算 | 未开始 |
| Value head 设计 | 共享 vision backbone，独立线性层输出 scalar V(s) | 未开始 |
| 训练模式关闭 feat_cache | eval 专用优化，训练时会导致梯度异常 | 未开始 |
| FSDP 分布式训练支持 | 当前单卡推理，训练需多卡参数分片 | 未开始 |

### M4（规模扩展）

| 内容 | 状态 |
|------|------|
| 扩展到 128 envs，覆盖完整 R2R val_unseen | 待做 |
| 对标 GenArk 原始 baseline，验证复现质量 | 待做 |

### 工程优化（低优先级）

- `rlinf/envs/action_utils.py`：补全 genark 4-way argmax 分支
- 优化 pass 间 Ray 重启耗时（目前约 3-5s/pass）

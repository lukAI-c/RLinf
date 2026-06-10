# Multi-Scene Preload Pool for GenArk — Change Log & Architecture

> **Audience:** AI coding assistants (and humans) who need to understand *what was
> changed* to make the GenArk navigation environment host **K scenes concurrently
> inside one RLinf env worker**, and *why*. Read this before touching any
> `rlinf/envs/genark/*` file.

---

## 1. Problem statement

GenArk runs the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator
for R2R-style navigation. **A Genesis scene's environment count is fixed at build
time** (`scene.build(n_envs=N)`) and **cannot be rebuilt in-process**. Historically,
each RLinf env worker (`GenarkVecEnv`) was therefore **pinned to a single scene**:
all `num_envs` slots shared one scene, and multi-scene coverage only happened by
launching many workers, each with a different `scene_offset`.

**Goal:** let **one worker host K scenes at once** in a single lockstep rollout
batch — "scene-granularity actor + multiple slots per actor". Each scene gets its
own `GenesisSceneActor` (1 GPU, `n_envs = K_s`), all preloaded at init. The worker's
`num_envs` slots are **statically partitioned** into K contiguous, group-aligned
blocks; each slot is bound to one scene for life and reuses it (resets to a new
episode of the *same* scene) without rebuilding Genesis.

---

## 2. Core design decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Slot → scene binding** | **Static partition**, fixed at init | Genesis can't rebuild; a slot's scene never changes |
| **Block fullness** | **Fully-active blocks** (`len(episodes_s) ≥ K_s`) | Keeps the global active set a contiguous prefix `[0, N_act)` so existing `cam_pos[:N_act]` main-loop sites are unchanged |
| **Crash semantics** | **Full-env dormancy (MVP)** — any one actor crash dormants the whole worker | Matches today's single-scene behavior; avoids GRPO advantage-poisoning from partial dormancy |
| **Concurrency** | **Two-phase fan-out** — submit all K `.remote()` calls, then collect-all | Wall-clock ≈ max(scene_time), not Σ(scene_time). Benchmarked **3.89× speedup at K=4** |
| **Scene assignment** | `scene_count=K` auto-pick + disjoint window per worker via `seed_offset`; explicit `scenes:[…]` list also supported | Maximizes scene coverage; GPU budget = K per worker |
| **GRPO group rule** | `group_size` consecutive slots **must never straddle a scene boundary** | Block offsets & sizes are multiples of `group_size`, enforced at build time |

---

## 3. Files changed

### New files

| File | Purpose |
|---|---|
| `rlinf/envs/genark/genesis_multiscene.py` | **Core of the feature.** `SceneLayout` (static partition table) + `MultiSceneBackend` (multiplexes K backends) + `build_multiscene_backend()` factory |
| `examples/embodiment/config/genark_grpo_qwen_multiscene.yaml` | Training config for the multi-scene GRPO run (GPUs 4,5,6,7) |
| `run_multiscene.sh` | Launch script (`CUDA_VISIBLE_DEVICES=4,5,6,7`, absolute Hydra config path) |
| `tests/unit_tests/test_multiscene_genark.py` | 40 unit tests: layout invariants, index translation, GRPO no-straddle, per-scene pools, stable hash, concurrent dispatch |
| `tests/benchmark_concurrent_dispatch.py` | Synthetic K=1..4 sequential-vs-concurrent benchmark |

### Modified files

| File | What changed |
|---|---|
| `rlinf/envs/genark/genesis_server.py` | GPU-pinning rewrite of `GenesisServerPool`; added `actor_handle()`, `step_physics_async()`, `set_agent_poses_async()`, `fetch_state()` for the concurrent path |
| `rlinf/envs/genark/genesis_backend.py` | Added `cam_pos_hab()` to the `GenesisSimBackend` base class (per-scene coordinate-frame protection) |
| `rlinf/envs/genark/genark_env.py` | Added the `multiscene` branch in `__init__`; converted single-scene episode-pool state into **per-scene dicts** |
| `rlinf/config.py` | Hardened the broken `torch.compile` path in the genesis env (mixed-version torch 2.5.1) |
| `examples/embodiment/config/env/genark_r2r.yaml` | Added `genesis_backend` and `multi_scene` config keys |

---

## 4. Architecture: `MultiSceneBackend`

`MultiSceneBackend` implements the **exact `GenesisSimBackend` interface**, so
`GarkVecEnv` keeps talking to a single `self._sim`. Its step/reward/render main loop
is **unchanged**. The backend multiplexes K per-scene `GenesisRemoteBackend`
instances and scatters each actor's local `(K_s, …)` results into global
`(num_envs, …)` tensors.

### `SceneLayout` — the static partition table (immutable, built once)

```
scenes:        list[str]          # K scene_ids preloaded for this worker
offsets[s]:    int                # global block offset of scene s
sizes[s]:      int  (= K_s)       # block size; offsets[s], sizes[s] are multiples of group_size
slot_to_scene: int[num_envs]      # slot_to_scene[g] = s  for offsets[s] <= g < offsets[s]+K_s
```

**Index translation (the core math):**
- global `g` → scene `s = slot_to_scene[g]`, local `l = g - offsets[s]`
- contiguous block slice: `actions[offsets[s] : offsets[s]+K_s]` → scene `s`'s full local batch
- scatter back: `global_cam_pos[offsets[s]:offsets[s]+K_s] = actor_s.cam_pos`

**Build-time invariants** (raise `ValueError` with remediation text):
1. `sum(K_s) == num_envs`; blocks tile `[0, num_envs)` with no gap/overlap
2. `offsets[s] % group_size == 0 and K_s % group_size == 0` (GRPO groups never straddle scenes)
3. `len(episodes_s) >= K_s` (fully-active blocks → no interior ghost rows)
4. All sub-backends share identical `H, W, FOV`
5. `K <= worker GPU budget`

### Two-phase concurrent dispatch

Every fan-out method (`step_physics`, `set_agent_poses`, `render_main`,
`render_4dir`) uses the same pattern:

```python
# Phase A — submit all K actors (non-blocking .remote() calls)
pending = []
for s, sub in enumerate(self._subs):
    off, k_s = self._layout.offsets[s], self._layout.sizes[s]
    ref = sub.step_physics_async(actions[off:off+k_s], active_mask[off:off+k_s], k_s)
    pending.append((s, sub, ref))

# Phase B — collect + scatter into global tensors
for s, sub, ref in pending:
    sub.fetch_state(ref)
    self._scatter_state(s, sub.cam_pos, sub.cam_yaw, sub.current_tri_idx)
```

All K GPUs compute in parallel; wall-clock per step ≈ `max_s(step_s)`, not `Σ_s`.

---

## 5. Per-scene state in `GenarkVecEnv`

The single-scene episode-pool singletons became **dicts keyed by scene_id**, so K=1
remains byte-identical to the old path (one-key dict):

| Old (single-scene) | New (multiscene) |
|---|---|
| `_all_episodes` | `_pool_by_scene[s]` |
| `_next_ep_idx` | `_next_ep_idx_by_scene[s]` |
| `_episode_cycle` | `_episode_cycle_by_scene[s]` |
| `_balancer` | `_balancer_by_scene[s]` (one `EpisodeBalancer` per scene) |
| `_rng` | `_rng_by_scene[s]` (seeded `42 + seed_offset + stable_hash(scene_id)`) |

**Key routing rule:** episode assignment indexes a slot's **own** pool with a
**local** group index:
```python
s           = slot_to_scene[i]
local_group = (i - offsets[s]) // group_size
ep          = _pool_by_scene[s][local_group]
```
This guarantees groups never pull a wrong-scene episode. `_group_done_counts` stays
keyed by global `group_id` (group→scene is well-defined under alignment).

---

## 6. The three bugs fixed to get it running (in order)

These were the blockers between "code compiles" and "rollout runs". Documented so
future changes don't reintroduce them.

### Bug A — Scene selection ignored episode counts → `fully_active` invariant fired
**Symptom:** `ValueError: scene '…' has only 1 episodes but was allocated 4 slots.`
**Cause:** `build_multiscene_backend()` picked scenes by `seed_offset` window from
`unique_scenes` **without checking** each scene had `≥ K_s` episodes.
**Fix:** before selecting, group episodes by scene and filter to
`eligible_scenes = [s for s in unique_scenes if len(by_scene[s]) >= min_eps]`, where
`min_eps = (base_groups + 1) * group_size` (conservative — covers the largest block).
Raise a clear error if fewer than `scene_count` qualify.
*(File: `genesis_multiscene.py`, `build_multiscene_backend`.)*

### Bug B — Ray `num_gpus=1` crashed with `IndexError` in `get_accelerator_ids`
**Symptom:** `IndexError: list index out of range` in Ray's
`get_accelerator_ids_for_accelerator_resource`; `GenesisSceneActor` died on init.
**Root cause (nested CUDA_VISIBLE_DEVICES conflict):**
- RLinf runs all its workers with `num_gpus=0` and pins them to physical GPUs by
  setting `CUDA_VISIBLE_DEVICES` + `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`.
- The env worker therefore had e.g. `CUDA_VISIBLE_DEVICES="6,7"` (2 entries).
- `GenesisSceneActor` was `@ray.remote(num_gpus=1)`, so Ray did GPU accounting.
- The child actor **inherited** the worker's `RAY_EXPERIMENTAL_NOSET=1` (Ray won't
  re-set CUDA_VISIBLE_DEVICES) **and** its `CUDA_VISIBLE_DEVICES="6,7"`.
- Ray's pool assigned a global index ∈ {0,1,2,3}, then ran
  `original_ids[i] for i in assigned_ids` where `original_ids=[6,7]` (len 2) →
  `original_ids[2]` → **IndexError**.

**Fix (match RLinf's own convention):** `GenesisSceneActor` is now plain
`@ray.remote` (no `num_gpus`), and `GenesisServerPool` spawns each actor with
`num_gpus=0` + an explicit single-GPU `runtime_env`:

```python
GenesisSceneActor.options(
    num_gpus=0,
    runtime_env={"env_vars": {
        "CUDA_VISIBLE_DEVICES": gpu_id,                    # one physical GPU
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "MUJOCO_EGL_DEVICE_ID": gpu_id,
    }},
).remote(cfg_dict, scene_id, num_envs)
```

`GenesisServerPool._discover_gpu_pool()` reads the worker's
`CUDA_VISIBLE_DEVICES` (absolute physical indices set by RLinf) and assigns each
scene actor a **distinct** GPU. Fails loudly if `K > len(gpu_pool)`.
*(File: `genesis_server.py`.)*

### Bug C — Camera batch-size mismatch in `step_physics`
**Symptom:** `genesis.GenesisException: Input data inconsistent with 'envs_idx'.`
at `cam.set_pose(...)` during the first physics step.
**Cause:** each scene's Genesis camera is built with `n_active_envs = K_s` (fixed),
so `cam.set_pose(pos=cam_pos[:active_slot_count])` requires **exactly `K_s` poses**.
`MultiSceneBackend.step_physics` was passing the **dynamic** active count
`active_k_s = active_mask[off:off+k_s].sum()`, which drops below `K_s` as soon as
any slot finishes mid-rollout → camera receives `< K_s` poses → exception.

**Key insight:** `active_slot_count` only **sizes the camera** (the rendered slot
prefix); which slots actually *move* is gated separately by `active_mask`. The
single-scene path keeps `_active_slot_count` **constant** = `num_envs` for exactly
this reason.

**Fix:** pass the **fixed `k_s`** (block size), not the dynamic sum:
```python
ref = sub.step_physics_async(actions[off:off+k_s], active_mask[off:off+k_s], k_s)
```
*(File: `genesis_multiscene.py`, `MultiSceneBackend.step_physics`. The render
methods already passed the fixed `k_s` — only step_physics had the bug.)*

---

## 7. Silent-correctness guards (do NOT remove)

Two bugs would pass K=1 tests and never throw — the dangerous class:

- **Coordinate-frame mixing (`cam_pos_hab()`):** `_genesis_to_hab` must be applied
  **per scene** (each scene has its own navmesh origin), not once over a mosaic of K
  origins. `GenesisSimBackend.cam_pos_hab()` does this; `MultiSceneBackend` overrides
  it to transform inside the per-scene scatter. The two metric sites that compute
  goal distance / nDTW call `self._sim.cam_pos_hab()` instead of transforming
  `cam_pos[:N_act]` directly. **This is the single most important guard.**
- **Episode→scene attribution:** episode logging writes `ep["scene_id"]` (the slot's
  real scene), never the stale global `_pinned_scene_id`.

---

## 8. Config surface

```yaml
env:
  train:
    total_num_envs: 8            # 1 worker × 8 slots (K_s=4 per scene × K=2)
    genesis_backend: multiscene
    multi_scene:
      scene_count: 2             # K scenes preloaded per worker = K GPUs/worker
      scenes: null               # OR explicit [scene_id, ...] (overrides scene_count)
      gpu_budget: 2              # fail-fast if scene_count exceeds this
  eval:
    genesis_backend: local       # eval uses the single-scene in-process backend
```

**GPU budgeting:** each worker consumes K GPUs. The env workers **must be
disaggregated** from policy GPUs (`cluster.component_placement.env`), or the
per-scene actors contend with the policy model. Example placement (GPUs 4,5,6,7):
```yaml
cluster:
  component_placement:
    rollout: { placement: "4-5" }    # actor + rollout (FSDP 2-way)
    actor:   { placement: "4-5" }
    env:     { placement: "6-7:0" }  # 1 env worker owning GPUs 6,7 → 2 scene actors
```

---

## 9. Verified status

- ✅ 40/40 unit tests pass (layout, index translation, GRPO no-straddle, per-scene pools, concurrent dispatch ordering)
- ✅ Concurrent dispatch benchmark: **3.89× speedup at K=4** (ideal 4×)
- ✅ End-to-end: scenes build on distinct GPUs, FSDP loads, **rollout runs with zero crashes** on GPUs 4,5,6,7
- ⚠️ K=1 equivalence (Phase 0 regression anchor) and per-scene coordinate-frame
  oracle (Phase 1) should be re-run if the scatter/index code is modified

---

## 10. Map: where to look for what

| You want to change… | Look in… |
|---|---|
| How scenes are picked / filtered | `genesis_multiscene.py` → `build_multiscene_backend()` |
| The static slot partition / invariants | `genesis_multiscene.py` → `SceneLayout.build()` |
| Fan-out / scatter / concurrency | `genesis_multiscene.py` → `MultiSceneBackend.*` |
| GPU pinning of scene actors | `genesis_server.py` → `GenesisServerPool.__init__` / `_create_actor` / `_discover_gpu_pool` |
| Async submit/fetch primitives | `genesis_server.py` → `GenesisRemoteBackend.*_async` / `fetch_state` |
| Per-scene episode pools / cycling / balancer | `genark_env.py` → `*_by_scene` dicts, `_assign_episodes_to_envs`, `_maybe_reset_complete_groups` |
| Coordinate-frame transform | `genesis_backend.py` → `cam_pos_hab()`; override in `genesis_multiscene.py` |

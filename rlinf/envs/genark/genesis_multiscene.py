"""
MultiSceneBackend — preload K Genesis scenes in one RLinf env worker.

Architecture
------------
One GenesisSceneActor (1 GPU) is spawned per scene, all preloaded at init via
GenesisServerPool.  The worker's ``num_envs`` slots are statically partitioned
into K contiguous, group-aligned blocks — one block per scene.  Each slot is
bound to its scene for the worker's lifetime; within a block, slot reuse (reset
to a new episode of the SAME scene) works via set_agent_poses, without ever
rebuilding Genesis.

MultiSceneBackend implements the full GenesisSimBackend interface so GenarkVecEnv
talks to it identically to a single-scene backend.  Fan-out and scatter happen
inside this class, invisible to the caller.  Each fan-out is two-phase (submit
all K scenes' Ray calls, then collect), so the K actors execute in parallel on
their K GPUs — per-step wall-clock is ~max over scenes, not the sum.

Bug protections implemented here (see plan §Bug catalog):
  #1  cam_pos_hab() applies _genesis_to_hab PER SCENE so coordinate frames don't mix.
  #2  SceneLayout asserts group-alignment (off_s % gs == 0, K_s % gs == 0).
  #4  Per-scene active_k_s passed to actors (never global N_act).
  #5  Fully-active block invariant: len(eps_s) >= K_s enforced at build.
  #7  MVP full-env dormancy: any actor crash raises aggregated SceneCrashError.
  #9  gpu_budget passed to GenesisServerPool.
  #12 All actors must share cam_h, cam_w, fov — asserted at init.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.envs.genark.genesis_backend import (
    GenesisSimBackend,
    _genesis_to_hab,
    _hab_to_genesis,
    _calculate_initial_yaw,
)
from rlinf.envs.genark.genesis_server import (
    GenesisServerPool,
    GenesisRemoteBackend,
    SceneCrashError,
)


# ---------------------------------------------------------------------------
# SceneLayout — immutable slot-partition table
# ---------------------------------------------------------------------------

@dataclass
class SceneLayout:
    """Static mapping between global slot indices and per-scene local indices.

    Built once at worker init; never mutated.

    Invariants enforced in __init__:
      - sum(K_s) == num_envs
      - blocks tile [0, num_envs) with no gaps or overlaps
      - off_s % group_size == 0  and  K_s % group_size == 0  (Bug #2)
      - len(eps_s) >= K_s  (fully-active blocks, Bug #5)
      - K (scene count) <= gpu_budget if given  (Bug #9)
    """

    scenes: list[str]                   # K scene_ids in order
    offsets: list[int]                  # off_s per scene, len=K
    sizes: list[int]                    # K_s per scene, len=K
    slot_to_scene_idx: np.ndarray       # (num_envs,) int: global slot -> scene index s
    num_envs: int
    group_size: int

    @classmethod
    def build(
        cls,
        scenes: list[str],
        episode_counts: list[int],       # len(eps_s) for each scene, in scenes order
        num_envs: int,
        group_size: int,
        gpu_budget: Optional[int] = None,
        scenes_per_gpu: int = 1,
    ) -> "SceneLayout":
        """Distribute num_envs across K scenes into group-aligned contiguous blocks.

        Each block size = round(num_envs / K) aligned down to group_size;
        the last block absorbs the remainder (also group-aligned so sum==num_envs).
        Raises ValueError with a concrete remediation message on any invariant violation.
        """
        gs = max(group_size, 1)
        K  = len(scenes)
        if K == 0:
            raise ValueError("SceneLayout: scenes list is empty.")

        # Bug #9: early GPU budget check. Capacity = gpu_budget × scenes_per_gpu
        # (scenes_per_gpu>1 packs multiple scene actors onto each physical GPU).
        spg = max(int(scenes_per_gpu), 1)
        if gpu_budget is not None and K > gpu_budget * spg:
            raise ValueError(
                f"SceneLayout: {K} scenes requested but capacity is gpu_budget"
                f"({gpu_budget}) × scenes_per_gpu({spg}) = {gpu_budget * spg}. "
                f"Reduce multi_scene.scene_count or raise scenes_per_gpu / gpu_budget."
            )

        if num_envs % gs != 0:
            raise ValueError(
                f"SceneLayout: num_envs={num_envs} is not divisible by group_size={gs}. "
                f"Set num_envs to a multiple of group_size."
            )

        n_groups = num_envs // gs  # total groups
        base_groups_per_scene = n_groups // K
        extra_groups = n_groups % K  # first extra_groups scenes get one more group

        offsets: list[int] = []
        sizes: list[int]   = []
        off = 0
        for s in range(K):
            ng = base_groups_per_scene + (1 if s < extra_groups else 0)
            k_s = ng * gs
            offsets.append(off)
            sizes.append(k_s)
            off += k_s

        assert off == num_envs, f"SceneLayout: block sum {off} != num_envs {num_envs}"

        # Bug #2: group-alignment (off_s and K_s must be multiples of gs)
        for s in range(K):
            if offsets[s] % gs != 0:
                raise ValueError(
                    f"SceneLayout: scene {s} offset {offsets[s]} not divisible by "
                    f"group_size={gs}. Adjust num_envs or scene_count."
                )
            if sizes[s] % gs != 0:
                raise ValueError(
                    f"SceneLayout: scene {s} block size {sizes[s]} not divisible by "
                    f"group_size={gs}. Adjust num_envs or scene_count."
                )
            if sizes[s] == 0:
                raise ValueError(
                    f"SceneLayout: scene {s} got 0 slots. "
                    f"num_envs={num_envs} may be too small for {K} scenes."
                )

        # Bug #5: fully-active blocks
        for s, scene_id in enumerate(scenes):
            eps_count = episode_counts[s]
            k_s = sizes[s]
            if eps_count < k_s:
                raise ValueError(
                    f"SceneLayout: scene '{scene_id}' has only {eps_count} episodes "
                    f"but was allocated {k_s} slots. "
                    f"Either reduce scene_count, reduce num_envs, or use a scene with "
                    f">= {k_s} episodes."
                )

        slot_to_scene_idx = np.empty(num_envs, dtype=np.int32)
        for s in range(K):
            slot_to_scene_idx[offsets[s]:offsets[s] + sizes[s]] = s

        return cls(
            scenes=scenes,
            offsets=offsets,
            sizes=sizes,
            slot_to_scene_idx=slot_to_scene_idx,
            num_envs=num_envs,
            group_size=gs,
        )

    # --- Query helpers -------------------------------------------------------

    def scene_idx_of(self, global_slot: int) -> int:
        return int(self.slot_to_scene_idx[global_slot])

    def scene_id_of(self, global_slot: int) -> str:
        return self.scenes[self.scene_idx_of(global_slot)]

    def local_idx(self, global_slot: int) -> int:
        s = self.scene_idx_of(global_slot)
        return global_slot - self.offsets[s]

    def global_idx(self, scene_idx: int, local_slot: int) -> int:
        return self.offsets[scene_idx] + local_slot

    def scene_local_group(self, global_slot: int) -> int:
        """Local group index within the scene block for a given global slot."""
        s = self.scene_idx_of(global_slot)
        return (global_slot - self.offsets[s]) // self.group_size

    def partition_env_idx(self, env_idx: list[int]) -> dict[int, tuple[list[int], list[int]]]:
        """Split env_idx into per-scene {scene_idx: (global_indices, local_indices)}."""
        result: dict[int, tuple[list[int], list[int]]] = {}
        for g in env_idx:
            s = int(self.slot_to_scene_idx[g])
            l = g - self.offsets[s]
            if s not in result:
                result[s] = ([], [])
            result[s][0].append(g)
            result[s][1].append(l)
        return result


# ---------------------------------------------------------------------------
# MultiSceneBackend
# ---------------------------------------------------------------------------

class MultiSceneBackend(GenesisSimBackend):
    """Multiplexes K GenesisRemoteBackend instances behind the GenesisSimBackend interface.

    GenarkVecEnv calls this exactly as it calls a single-scene backend.
    Internally, each method partitions the global (num_envs,…) tensors by scene
    block, fans out to the appropriate per-scene GenesisRemoteBackend, and scatters
    the results back into global tensors.

    Concurrency model:
      Each fan-out method (step_physics / render_main / render_4dir /
      set_agent_poses) is two-phase: it submits every scene's .remote() call
      first (non-blocking), then collects all results. The K actors therefore
      run in parallel on their K GPUs, so per-step wall-clock is ~max over
      scenes rather than the sum. This concurrency is fully INTERNAL — every
      method still returns a fully-assembled (num_envs, …) result and no Ray
      ObjectRef ever leaks to GenarkVecEnv, so _use_async_render stays False and
      the "ref type single→dict" bug class is still avoided.

    MVP constraints (safe to lift in later phases):
      - Full-env dormancy: any actor crash raises SceneCrashError immediately.
    """

    def __init__(
        self,
        cfg: DictConfig,
        layout: SceneLayout,
        device: torch.device,
        gpu_budget: Optional[int] = None,
        scenes_per_gpu: int = 1,
    ):
        self._layout = layout
        self._device = device
        K = len(layout.scenes)

        # Build pool with all K scenes preloaded. scenes_per_gpu>1 packs multiple
        # scene actors onto each physical GPU (see GenesisServerPool).
        scene_assignments = {
            layout.scenes[s]: layout.sizes[s]
            for s in range(K)
        }
        self._pool = GenesisServerPool(
            cfg, scene_assignments,
            gpu_budget=gpu_budget,
            scenes_per_gpu=scenes_per_gpu,
        )

        # One isolated GenesisRemoteBackend per scene (each carries _num_envs=K_s).
        # Bug #4 guard: each sub-backend's _num_envs=K_s so its internal reshape
        # uses the per-scene slot count, never the global num_envs.
        self._subs: list[GenesisRemoteBackend] = [
            GenesisRemoteBackend(
                pool=self._pool,
                scene_id=layout.scenes[s],
                num_envs=layout.sizes[s],
                device=device,
            )
            for s in range(K)
        ]

        # Bug #12: all actors must share H, W (so scatter is shape-consistent).
        cam_h_set = {sub._cam_h for sub in self._subs}
        cam_w_set = {sub._cam_w for sub in self._subs}
        if len(cam_h_set) != 1 or len(cam_w_set) != 1:
            raise ValueError(
                f"MultiSceneBackend: scenes have heterogeneous camera dims "
                f"(H={cam_h_set}, W={cam_w_set}). All scenes must share the same cfg."
            )
        self._cam_h = next(iter(cam_h_set))
        self._cam_w = next(iter(cam_w_set))

        # Global tensors: (num_envs, ...) assembled from per-scene actor states.
        N = layout.num_envs
        self._cam_pos_t         = torch.zeros(N, 3, dtype=torch.float32, device=device)
        self._cam_yaw_t         = torch.zeros(N,    dtype=torch.float32, device=device)
        self._current_tri_idx_t = torch.zeros(N,    dtype=torch.int64,   device=device)

        # Pull initial state from each actor into the global tensors.
        for s in range(K):
            self._scatter_state(s, self._subs[s].cam_pos, self._subs[s].cam_yaw,
                                self._subs[s].current_tri_idx)

    # --- Properties ----------------------------------------------------------

    @property
    def cam_pos(self) -> torch.Tensor:
        return self._cam_pos_t

    @property
    def cam_yaw(self) -> torch.Tensor:
        return self._cam_yaw_t

    @property
    def current_tri_idx(self) -> torch.Tensor:
        return self._current_tri_idx_t

    @property
    def device(self) -> torch.device:
        return self._device

    # --- cam_pos_hab: per-scene coordinate frame (Bug #1) --------------------

    def cam_pos_hab(self, camera_height: float) -> torch.Tensor:
        """Return cam_pos in Habitat coords, applying _genesis_to_hab PER SCENE.

        Overrides the base-class default (which applies the transform globally)
        so that each scene block uses its own navmesh datum.  Without this,
        positions from different scenes are mixed in a single transform — silently
        producing wrong distances and nDTW because Genesis and Habitat origins
        differ across scans.
        """
        result = torch.zeros(self._layout.num_envs, 3,
                             dtype=torch.float32, device=self._device)
        for s, sub in enumerate(self._subs):
            off = self._layout.offsets[s]
            k_s = self._layout.sizes[s]
            result[off:off + k_s] = _genesis_to_hab(sub.cam_pos, camera_height)
        return result

    # --- Scene lifecycle (no-op: actors are built with scene preloaded) ------

    def load_scene(self, scene_id: str, n_active_envs: int) -> None:
        pass  # scenes preloaded in __init__ via GenesisSceneActor.__init__

    # --- Agent pose management -----------------------------------------------

    def set_agent_poses(
        self,
        env_idx: list[int],
        positions: torch.Tensor,   # (len(env_idx), 3) float32
        yaws: torch.Tensor,        # (len(env_idx),) float32
    ) -> None:
        """Partition env_idx by scene, reset each scene's poses concurrently.

        Two-phase: submit all touched scenes' set_agent_poses (non-blocking),
        then collect + scatter, so the per-scene navmesh snapping overlaps.
        """
        partitioned = self._layout.partition_env_idx(env_idx)
        # Build a row-index map: env_idx[j] -> row j in positions/yaws
        row_of = {g: j for j, g in enumerate(env_idx)}

        # Phase A — submit all touched scenes (non-blocking).
        pending = []  # (s, ref)
        for s, (globals_, locals_) in partitioned.items():
            rows = [row_of[g] for g in globals_]
            pos_s = positions[rows]
            yaw_s = yaws[rows]
            ref = self._subs[s].set_agent_poses_async(locals_, pos_s, yaw_s)
            pending.append((s, ref))

        # Phase B — collect + scatter.
        crashed: list[SceneCrashError] = []
        for s, ref in pending:
            try:
                self._subs[s].fetch_state(ref)
                self._scatter_state(s, self._subs[s].cam_pos,
                                    self._subs[s].cam_yaw,
                                    self._subs[s].current_tri_idx)
            except SceneCrashError as e:
                crashed.append(e)

        if crashed:
            raise SceneCrashError(
                scene_id=crashed[0].scene_id,
                cause=Exception(
                    f"{len(crashed)} scene actor(s) crashed during set_agent_poses: "
                    + ", ".join(str(e) for e in crashed)
                ),
            )

    # --- Physics step --------------------------------------------------------

    def step_physics(
        self,
        actions: torch.Tensor,       # (num_envs,) int64
        active_mask: torch.Tensor,   # (num_envs,) bool
        active_slot_count: int,      # global; we compute per-scene below
    ) -> None:
        """Fan out physics step to each scene actor with per-scene active counts.

        Two-phase concurrent dispatch:
          A. Submit every scene's step_physics .remote() (non-blocking) so all K
             actors start computing in parallel on their own GPUs.
          B. Collect each result and scatter into the global tensors.
        Wall-clock per step is therefore ~max_s(step_s) + collect, not Σ_s(step_s).
        """
        # Phase A — submit all K actors (non-blocking).
        pending = []  # (s, sub, ref)
        for s, sub in enumerate(self._subs):
            off = self._layout.offsets[s]
            k_s = self._layout.sizes[s]
            # active_slot_count sizes the camera (built with n_active_envs == k_s
            # under the fully-active invariant); it is FIXED at k_s, not the live
            # active_mask sum. Slots that finish mid-rollout are gated by
            # active_mask inside step_physics — they are still rendered, exactly as
            # the single-scene path keeps _active_slot_count constant. Passing the
            # dynamic sum here makes set_pose receive < k_s poses → Genesis raises
            # "Input data inconsistent with 'envs_idx'". (Bug #4 was about never
            # passing the *global* N_act across scene boundaries — k_s is per-scene.)
            ref = sub.step_physics_async(
                actions[off:off + k_s],
                active_mask[off:off + k_s],
                k_s,
            )
            pending.append((s, sub, ref))

        # Phase B — collect + scatter (blocks until each scene's actor finishes).
        crashed: list[SceneCrashError] = []
        for s, sub, ref in pending:
            try:
                sub.fetch_state(ref)
                self._scatter_state(s, sub.cam_pos, sub.cam_yaw, sub.current_tri_idx)
            except SceneCrashError as e:
                crashed.append(e)

        if crashed:
            # MVP: full-env dormancy on any actor crash.
            # Re-raise as a single SceneCrashError so GenarkVecEnv enters dormant.
            raise SceneCrashError(
                scene_id=crashed[0].scene_id,
                cause=Exception(
                    f"{len(crashed)} scene actor(s) crashed: "
                    + ", ".join(str(e) for e in crashed)
                ),
            )

    # --- Rendering (sync-merge MVP) ------------------------------------------

    def render_main(self, active_slot_count: int) -> np.ndarray:
        """Render all scenes concurrently, concatenate into (num_envs, H, W, 3).

        Two-phase: submit all K render requests (non-blocking), then fetch + place.
        Rendering is usually the dominant per-step cost, so overlapping the K GPUs
        here is the biggest wall-clock win. From GenarkVecEnv's view this call is
        still synchronous (no refs leak out → _use_async_render stays False).
        """
        N   = self._layout.num_envs
        out = np.zeros((N, self._cam_h, self._cam_w, 3), dtype=np.uint8)

        # Phase A — submit (non-blocking); fully-active invariant ⇒ active_k_s == k_s.
        pending = []  # (s, sub, ref)
        for s, sub in enumerate(self._subs):
            k_s = self._layout.sizes[s]
            pending.append((s, sub, sub.render_main_async(k_s)))

        # Phase B — fetch + place.
        crashed: list[SceneCrashError] = []
        for s, sub, ref in pending:
            off = self._layout.offsets[s]
            k_s = self._layout.sizes[s]
            try:
                block = sub.fetch_render_main(ref)   # (k_s, H, W, 3)
                # Bug #4 / #12: assert shape before writing
                assert block.shape == (k_s, self._cam_h, self._cam_w, 3), (
                    f"render_main scene {s}: expected {(k_s, self._cam_h, self._cam_w, 3)}, "
                    f"got {block.shape}"
                )
                out[off:off + k_s] = block
            except SceneCrashError as e:
                crashed.append(e)

        if crashed:
            raise SceneCrashError(
                scene_id=crashed[0].scene_id,
                cause=Exception(
                    f"{len(crashed)} scene actor(s) crashed during render: "
                    + ", ".join(str(e) for e in crashed)
                ),
            )
        return out

    def render_4dir(self, active_slot_count: int) -> Optional[np.ndarray]:
        """Render 4-dir views for all scenes concurrently, or None if any is None.

        Two-phase: submit all K (non-blocking), then fetch. 4-dir is 3-4× the
        render cost, so concurrency matters most here.
        """
        N   = self._layout.num_envs
        K   = len(self._layout.scenes)

        # Phase A — submit all K (non-blocking).
        pending = []  # (s, sub, ref)
        for s, sub in enumerate(self._subs):
            k_s = self._layout.sizes[s]
            pending.append((s, sub, sub.render_4dir_async(k_s)))

        # Phase B — fetch (preserve per-scene slot s in blocks[s]).
        crashed: list[SceneCrashError] = []
        blocks: list[Optional[np.ndarray]] = [None] * K
        for s, sub, ref in pending:
            try:
                blocks[s] = sub.fetch_render_4dir(ref)   # (k_s, 3, H, W, 3) or None
            except SceneCrashError as e:
                crashed.append(e)

        if crashed:
            raise SceneCrashError(
                scene_id=crashed[0].scene_id,
                cause=Exception(
                    f"{len(crashed)} scene actor(s) crashed during render_4dir: "
                    + ", ".join(str(e) for e in crashed)
                ),
            )

        # If any scene has 4-dir disabled, return None for the whole batch.
        if any(b is None for b in blocks):
            return None

        out = np.zeros((N, 3, self._cam_h, self._cam_w, 3), dtype=np.uint8)
        for s in range(K):
            off = self._layout.offsets[s]
            k_s = self._layout.sizes[s]
            block = blocks[s]
            assert block.shape == (k_s, 3, self._cam_h, self._cam_w, 3), (
                f"render_4dir scene {s}: expected {(k_s, 3, self._cam_h, self._cam_w, 3)}, "
                f"got {block.shape}"
            )
            out[off:off + k_s] = block
        return out

    # NOTE: render_main / render_4dir fan out concurrently INTERNALLY (submit all
    # K, then fetch) but return a single fully-assembled array. We deliberately do
    # NOT expose a public render_main_async, so GenarkVecEnv keeps
    # _use_async_render=False and never sees per-scene refs — the "ref type
    # single→dict" bug class stays eliminated while we still get cross-GPU overlap.

    # --- Health --------------------------------------------------------------

    def is_scene_healthy(self) -> bool:
        """MVP: all actors must be healthy (full-env dormancy model)."""
        return all(sub.is_scene_healthy() for sub in self._subs)

    # --- Internal helpers ----------------------------------------------------

    def _scatter_state(
        self,
        s: int,
        cam_pos: torch.Tensor,       # (K_s, 3)
        cam_yaw: torch.Tensor,       # (K_s,)
        current_tri_idx: torch.Tensor,  # (K_s,)
    ) -> None:
        """Write per-scene actor state into the global (num_envs,...) tensors."""
        off = self._layout.offsets[s]
        k_s = self._layout.sizes[s]
        self._cam_pos_t[off:off + k_s]         = cam_pos.to(self._device)
        self._cam_yaw_t[off:off + k_s]         = cam_yaw.to(self._device)
        self._current_tri_idx_t[off:off + k_s] = current_tri_idx.to(self._device)


# ---------------------------------------------------------------------------
# Factory helper used by GenarkVecEnv.__init__
# ---------------------------------------------------------------------------

def _stable_scene_hash(scene_id: str) -> int:
    """Deterministic int hash for a scene_id (used to seed per-scene RNG)."""
    return int(hashlib.md5(scene_id.encode()).hexdigest(), 16) % (2**31)


def build_multiscene_backend(
    cfg: DictConfig,
    all_episodes: list[dict],
    num_envs: int,
    group_size: int,
    seed_offset: int,
    device: torch.device,
    gpu_budget: Optional[int] = None,
    scenes_per_gpu: int = 1,
) -> tuple["MultiSceneBackend", "SceneLayout", dict[str, list[dict]]]:
    """Select K scenes, build SceneLayout, instantiate MultiSceneBackend.

    Returns (backend, layout, pool_by_scene) where pool_by_scene maps
    scene_id -> shuffled episode list for that scene.  GenarkVecEnv stores
    this dict as _pool_by_scene to drive per-scene episode cycling.

    Scene selection:
      - cfg.multi_scene.scenes: explicit list of scene_ids (takes precedence)
      - cfg.multi_scene.scene_count: pick first N unique scenes, offset by
        seed_offset so workers get disjoint windows (worker w -> scenes[w*N:(w+1)*N])
    """
    multi_cfg = getattr(cfg, "multi_scene", None)
    if multi_cfg is None:
        raise ValueError(
            "genesis_backend='multiscene' requires cfg.multi_scene to be set."
        )

    # Group episodes by scene early so we can filter before selection.
    by_scene: dict[str, list] = {}
    for _ep in all_episodes:
        by_scene.setdefault(_ep["scene_id"], []).append(_ep)

    unique_scenes = sorted({e["scene_id"] for e in all_episodes})
    explicit_scenes = list(getattr(multi_cfg, "scenes", None) or [])

    if explicit_scenes:
        scenes = explicit_scenes
    else:
        scene_count = int(getattr(multi_cfg, "scene_count", 2))
        gs = max(group_size, 1)

        # Minimum episodes per scene = the LARGEST block any scene will get, so
        # it satisfies SceneLayout's fully-active invariant (len(eps_s) >= K_s).
        # Largest block = (base_g + 1) groups only when there ARE remainder groups
        # (extra > 0); when num_envs divides evenly across scenes every block is
        # exactly base_g groups, so requiring base_g+1 over-rejects valid scenes.
        n_groups = num_envs // gs
        base_g   = n_groups // scene_count
        extra    = n_groups %  scene_count
        max_groups_per_scene = base_g + (1 if extra > 0 else 0)
        min_eps = max_groups_per_scene * gs
        eligible_scenes = [s for s in unique_scenes if len(by_scene.get(s, [])) >= min_eps]
        if len(eligible_scenes) < scene_count:
            raise ValueError(
                f"build_multiscene_backend: need {scene_count} scenes with >= {min_eps} "
                f"episodes each, but only {len(eligible_scenes)} qualify out of "
                f"{len(unique_scenes)} total scenes. Reduce scene_count, reduce "
                f"num_envs, or use a richer episodes_file."
            )

        # Disjoint window per worker: worker seed_offset picks window [off:off+K]
        off = (seed_offset * scene_count) % len(eligible_scenes)
        scenes = []
        for i in range(scene_count):
            scenes.append(eligible_scenes[(off + i) % len(eligible_scenes)])

    # Shuffle each scene's episodes with its own RNG for reproducibility
    pool_by_scene: dict[str, list[dict]] = {}
    for scene_id in scenes:
        eps = list(by_scene.get(scene_id, []))
        if not eps:
            raise ValueError(
                f"build_multiscene_backend: scene '{scene_id}' has no episodes in "
                f"episodes_file. Check cfg.multi_scene.scenes."
            )
        rng_s = np.random.default_rng(seed=42 + seed_offset + _stable_scene_hash(scene_id))
        rng_s.shuffle(eps)
        pool_by_scene[scene_id] = eps

    episode_counts = [len(pool_by_scene[sid]) for sid in scenes]

    layout = SceneLayout.build(
        scenes=scenes,
        episode_counts=episode_counts,
        num_envs=num_envs,
        group_size=group_size,
        gpu_budget=gpu_budget,
        scenes_per_gpu=scenes_per_gpu,
    )

    backend = MultiSceneBackend(
        cfg, layout, device,
        gpu_budget=gpu_budget,
        scenes_per_gpu=scenes_per_gpu,
    )
    return backend, layout, pool_by_scene

"""
GenesisEnvServer — Ray-based remote Genesis backend.

Provides crash-isolated Genesis simulation via Ray actors.
Each GenesisSceneActor runs in its own process + GPU, so a SIGSEGV in one
scene does not affect other scenes or the RLinf training process.

Usage (from GenarkVecEnv via genesis_backend_type="remote"):

    pool = GenesisServerPool(cfg, {"mp3d/zsNo4HB9uLZ/...": 12})
    backend = GenesisRemoteBackend(pool, scene_id="mp3d/zsNo4HB9uLZ/...", ...)
    env = GenarkVecEnv(cfg, num_envs=12, seed_offset=0, ..., backend=backend)
"""

from __future__ import annotations

import os

import numpy as np
import torch
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from rlinf.envs.genark.genesis_backend import (
    GenesisSimBackend,
    GenesisLocalBackend,
    _genesis_to_hab,
    _hab_to_genesis,
    _calculate_initial_yaw,
)

try:
    import ray
    import ray.exceptions
except ImportError as e:
    raise ImportError(
        "GenesisEnvServer requires Ray. Install with: pip install ray"
    ) from e


class SceneCrashError(RuntimeError):
    """Raised by GenesisRemoteBackend when the remote actor crashes mid-call.

    GenarkVecEnv catches this and enters dormant mode for the affected envs
    while the pool rebuilds the actor in the background.
    """
    def __init__(self, scene_id: str, cause: BaseException):
        super().__init__(
            f"GenesisSceneActor crashed for scene '{scene_id}': {cause}"
        )
        self.scene_id = scene_id


# ---------------------------------------------------------------------------
# Ray remote actor — one per Genesis scene, one GPU
# ---------------------------------------------------------------------------

@ray.remote
class GenesisSceneActor:
    """Holds one Genesis scene in a dedicated Ray worker process.

    Wraps GenesisLocalBackend so all Genesis state stays inside this actor.
    Communication is via CPU numpy arrays to avoid CUDA IPC complexity.
    """

    def __init__(self, cfg_dict: dict, scene_id: str, num_envs: int):
        cfg = OmegaConf.create(cfg_dict)
        self._backend = GenesisLocalBackend(cfg, num_envs)
        self._backend.load_scene(scene_id, n_active_envs=num_envs)
        self._num_envs = num_envs
        self._cam_h    = self._backend._cam_h
        self._cam_w    = self._backend._cam_w

    def set_agent_poses(
        self,
        env_idx: list[int],
        positions_np: np.ndarray,   # (len(env_idx), 3) float32 CPU
        yaws_np: np.ndarray,        # (len(env_idx),) float32 CPU
    ) -> dict:
        """Set poses, snap to navmesh. Returns snapped state as CPU numpy."""
        dev = self._backend.device
        self._backend.set_agent_poses(
            env_idx,
            torch.from_numpy(positions_np).to(dev),
            torch.from_numpy(yaws_np).to(dev),
        )
        return self._get_state_dict()

    def step_physics(
        self,
        actions_np: np.ndarray,       # (N,) int64 CPU
        active_mask_np: np.ndarray,   # (N,) bool CPU
        active_slot_count: int,
    ) -> dict:
        """Physics step. Returns updated agent state as CPU numpy."""
        assert active_slot_count <= self._num_envs, (
            f"GenesisSceneActor.step_physics: active_slot_count={active_slot_count} "
            f"> actor num_envs={self._num_envs}. Pass per-scene local count, not global N_act."
        )
        dev = self._backend.device
        self._backend.step_physics(
            torch.from_numpy(actions_np).to(dev),
            torch.from_numpy(active_mask_np).to(dev),
            active_slot_count,
        )
        return self._get_state_dict()

    def render_main(self, active_slot_count: int) -> bytes:
        """Returns (N, H, W, 3) uint8 as raw bytes."""
        assert active_slot_count <= self._num_envs, (
            f"GenesisSceneActor.render_main: active_slot_count={active_slot_count} "
            f"> actor num_envs={self._num_envs}. Pass per-scene local count, not global N_act."
        )
        rgb = self._backend.render_main(active_slot_count)
        return rgb.tobytes()

    def render_4dir(self, active_slot_count: int) -> Optional[bytes]:
        """Returns (N, 3, H, W, 3) uint8 as raw bytes, or None."""
        assert active_slot_count <= self._num_envs, (
            f"GenesisSceneActor.render_4dir: active_slot_count={active_slot_count} "
            f"> actor num_envs={self._num_envs}. Pass per-scene local count, not global N_act."
        )
        extras = self._backend.render_4dir(active_slot_count)
        if extras is None:
            return None
        return extras.tobytes()

    def render_main_with_depth(self, active_slot_count: int) -> tuple:
        """Returns (rgb_bytes, depth_bytes): rgb=(N,H,W,3) uint8, depth=(N,H,W) float32."""
        assert active_slot_count <= self._num_envs, (
            f"GenesisSceneActor.render_main_with_depth: active_slot_count={active_slot_count} "
            f"> actor num_envs={self._num_envs}."
        )
        rgb, depth = self._backend.render_main_with_depth(active_slot_count)
        return rgb.tobytes(), depth.tobytes()

    def get_state(self) -> dict:
        """Return full agent state for initial sync."""
        d = self._get_state_dict()
        d["num_envs"] = self._num_envs
        d["cam_h"]    = self._cam_h
        d["cam_w"]    = self._cam_w
        return d

    def health_check(self) -> bool:
        return True

    def _get_state_dict(self) -> dict:
        return {
            "cam_pos":         self._backend.cam_pos.cpu().numpy(),
            "cam_yaw":         self._backend.cam_yaw.cpu().numpy(),
            "current_tri_idx": self._backend.current_tri_idx.cpu().numpy(),
        }


# ---------------------------------------------------------------------------
# Server pool — manages multiple scene actors
# ---------------------------------------------------------------------------

class GenesisServerPool:
    """Maps scene_id → GenesisSceneActor, with automatic crash recovery.

    Parameters
    ----------
    cfg : DictConfig
        Full env cfg (passed to GenesisLocalBackend inside each actor).
    scene_assignments : dict[str, int]
        {scene_id: num_envs} — one actor per entry.
    """

    def __init__(self, cfg: DictConfig, scene_assignments: dict[str, int],
                 gpu_budget: Optional[int] = None,
                 scenes_per_gpu: int = 1):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # scenes_per_gpu: how many scene actors may share one physical GPU.
        #   =1 → strict 1:1 (max parallelism, the original behavior).
        #   >1 → oversubscribe: K scenes packed onto ceil(K/spg) GPUs. Scenes on
        #        the same GPU serialize on its compute/render engine, but Genesis
        #        sim is light (~1% of step time), so this trades a small per-step
        #        cost for wider scene coverage on fewer GPUs.
        self._scenes_per_gpu = max(int(scenes_per_gpu), 1)

        # Bug #9: fail fast (ValueError) instead of hanging on Ray GPU allocation.
        # Capacity = gpu_budget × scenes_per_gpu. Exceeding it would silently
        # triple-book a GPU; fail loudly instead.
        k = len(scene_assignments)
        if gpu_budget is not None and k > gpu_budget * self._scenes_per_gpu:
            raise ValueError(
                f"GenesisServerPool: {k} scenes requested but capacity is "
                f"gpu_budget({gpu_budget}) × scenes_per_gpu({self._scenes_per_gpu}) "
                f"= {gpu_budget * self._scenes_per_gpu}. Reduce multi_scene.scene_count "
                f"or raise multi_scene.scenes_per_gpu / gpu_budget."
            )

        self._cfg               = cfg
        self._cfg_dict          = OmegaConf.to_container(cfg, resolve=True)
        self._scene_assignments = scene_assignments
        self._actors: dict[str, ray.actor.ActorHandle] = {}

        # ── GPU pinning ──────────────────────────────────────────────────────
        # RLinf runs all workers with num_gpus=0 and pins them to physical GPUs
        # via CUDA_VISIBLE_DEVICES (RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
        # tells Ray not to override). We mirror that here: spawn each scene actor
        # with num_gpus=0 and an explicit single-GPU CUDA_VISIBLE_DEVICES drawn
        # from the env worker's own visible-device pool. Using Ray's num_gpus=1
        # accounting instead crashes (IndexError in get_accelerator_ids) because
        # the actor inherits the worker's restricted CUDA_VISIBLE_DEVICES while
        # Ray hands out global pool indices that exceed that restricted list.
        self._gpu_pool = self._discover_gpu_pool()
        capacity = len(self._gpu_pool) * self._scenes_per_gpu
        if capacity < k:
            raise ValueError(
                f"GenesisServerPool: {k} scenes requested but capacity is "
                f"{len(self._gpu_pool)} GPU(s) × scenes_per_gpu({self._scenes_per_gpu}) "
                f"= {capacity} (CUDA_VISIBLE_DEVICES="
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}). "
                f"Reduce multi_scene.scene_count, raise scenes_per_gpu, or widen the "
                f"env worker placement."
            )
        # scene_id -> physical GPU id (string). Round-robin packs scenes onto the
        # pool: with scenes_per_gpu>1 this assigns multiple scenes to each GPU
        # (scene i → gpu_pool[i % len(pool)], so [g0,g1,g0,g1] for 4 scenes/2 GPUs).
        self._scene_gpu: dict[str, str] = {}
        for i, scene_id in enumerate(scene_assignments):
            self._scene_gpu[scene_id] = self._gpu_pool[i % len(self._gpu_pool)]
        if self._scenes_per_gpu > 1:
            print(
                f"[GenesisServerPool] oversubscribe: {k} scenes on "
                f"{len(self._gpu_pool)} GPU(s) (scenes_per_gpu={self._scenes_per_gpu}); "
                f"same-GPU scenes serialize on compute/render.",
                flush=True,
            )

        for scene_id, num_envs in scene_assignments.items():
            self._actors[scene_id] = self._create_actor(scene_id, num_envs)

    @staticmethod
    def _discover_gpu_pool() -> list[str]:
        """Physical GPU ids this env worker may use, from CUDA_VISIBLE_DEVICES.

        RLinf sets CUDA_VISIBLE_DEVICES to the worker's absolute physical GPU
        indices. We split it so each scene actor can claim one distinct GPU.
        """
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd.strip() == "":
            # Fall back to Ray's GPU count if the worker didn't restrict devices.
            try:
                n = int(ray.cluster_resources().get("GPU", 0))
            except Exception:
                n = 0
            return [str(i) for i in range(n)]
        return [tok.strip() for tok in cvd.split(",") if tok.strip() != ""]

    def _create_actor(self, scene_id: str, num_envs: int) -> ray.actor.ActorHandle:
        gpu_id = self._scene_gpu[scene_id]
        return GenesisSceneActor.options(
            num_gpus=0,
            runtime_env={
                "env_vars": {
                    "CUDA_VISIBLE_DEVICES": gpu_id,
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "MUJOCO_EGL_DEVICE_ID": gpu_id,
                }
            },
        ).remote(self._cfg_dict, scene_id, num_envs)

    def get_actor(self, scene_id: str) -> ray.actor.ActorHandle:
        """Return the actor for a scene, rebuilding it if it has crashed.

        Performs a blocking health_check round-trip. Use this on the SYNC path.
        The concurrent fan-out path uses actor_handle() instead (see below).
        """
        actor = self._actors[scene_id]
        try:
            ray.get(actor.health_check.remote(), timeout=2.0)
        except Exception:
            self.rebuild_actor(scene_id)
        return self._actors[scene_id]

    def actor_handle(self, scene_id: str) -> ray.actor.ActorHandle:
        """Return the current actor handle WITHOUT a health_check round-trip.

        Used by the concurrent fan-out path (MultiSceneBackend): blocking on a
        per-scene health_check before each submit would serialize the K submits
        and defeat cross-GPU parallelism. Crash detection still happens on the
        fetch side via GenesisRemoteBackend._safe_get (RayActorError -> rebuild
        + SceneCrashError), so no crash is missed — it's just detected one step
        later, which the full-env dormancy model already tolerates.
        """
        return self._actors[scene_id]

    def rebuild_actor(self, scene_id: str) -> None:
        """Forcibly replace a crashed actor with a fresh one (async — returns immediately)."""
        num_envs = self._scene_assignments[scene_id]
        self._actors[scene_id] = self._create_actor(scene_id, num_envs)

    def is_healthy(self, scene_id: str) -> bool:
        """Return True if actor responds to health_check within 2 s."""
        try:
            ray.get(self._actors[scene_id].health_check.remote(), timeout=2.0)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Remote backend — thin client used by GenarkVecEnv
# ---------------------------------------------------------------------------

class GenesisRemoteBackend(GenesisSimBackend):
    """Implements GenesisSimBackend by forwarding calls to a GenesisSceneActor.

    Maintains local CPU copies of cam_pos/cam_yaw/current_tri_idx so that
    GenarkVecEnv metric computations don't incur extra Ray round-trips.
    """

    def __init__(
        self,
        pool: GenesisServerPool,
        scene_id: str,
        num_envs: int,
        device: torch.device,
    ):
        self._pool      = pool
        self._scene_id  = scene_id
        self._num_envs  = num_envs
        self._device    = device

        # Sync initial state from actor
        actor  = self._pool.get_actor(scene_id)
        state  = self._safe_get(actor.get_state.remote())
        self._cam_h = state["cam_h"]
        self._cam_w = state["cam_w"]
        self._sync_state(state)

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

    # --- Health --------------------------------------------------------------

    def is_scene_healthy(self) -> bool:
        return self._pool.is_healthy(self._scene_id)

    # --- Scene lifecycle -----------------------------------------------------

    def load_scene(self, scene_id: str, n_active_envs: int) -> None:
        # Scene is loaded in actor __init__; this is a no-op for remote backend.
        pass

    # --- Agent pose management -----------------------------------------------

    def set_agent_poses(
        self,
        env_idx: list[int],
        positions: torch.Tensor,
        yaws: torch.Tensor,
    ) -> None:
        actor  = self._pool.get_actor(self._scene_id)
        state  = self._safe_get(actor.set_agent_poses.remote(
            env_idx,
            positions.cpu().numpy(),
            yaws.cpu().numpy(),
        ))
        self._sync_state(state)

    def set_agent_poses_async(
        self,
        env_idx: list[int],
        positions: torch.Tensor,
        yaws: torch.Tensor,
    ):
        """Submit a pose reset; return a Ray ObjectRef immediately (non-blocking).

        Pair with fetch_state() to collect the snapped state and sync local tensors.
        """
        actor = self._pool.actor_handle(self._scene_id)
        return actor.set_agent_poses.remote(
            env_idx,
            positions.cpu().numpy(),
            yaws.cpu().numpy(),
        )

    # --- Physics step --------------------------------------------------------

    def step_physics(
        self,
        actions: torch.Tensor,
        active_mask: torch.Tensor,
        active_slot_count: int,
    ) -> None:
        actor  = self._pool.get_actor(self._scene_id)
        state  = self._safe_get(actor.step_physics.remote(
            actions.cpu().numpy(),
            active_mask.cpu().numpy(),
            active_slot_count,
        ))
        self._sync_state(state)

    def step_physics_async(
        self,
        actions: torch.Tensor,
        active_mask: torch.Tensor,
        active_slot_count: int,
    ):
        """Submit a physics step; return a Ray ObjectRef immediately (non-blocking).

        Uses actor_handle (no health_check round-trip) so MultiSceneBackend can
        submit all K scenes back-to-back and let them run in parallel across GPUs.
        Pair with fetch_state() to collect the result and sync local tensors.
        """
        actor = self._pool.actor_handle(self._scene_id)
        return actor.step_physics.remote(
            actions.cpu().numpy(),
            active_mask.cpu().numpy(),
            active_slot_count,
        )

    def fetch_state(self, ref) -> None:
        """Block on a step_physics / set_agent_poses ref and sync local tensors.

        Crash-safe: _safe_get rebuilds the actor and raises SceneCrashError if
        the remote actor died while we were waiting.
        """
        self._sync_state(self._safe_get(ref))

    # --- Rendering -----------------------------------------------------------

    def render_main(self, active_slot_count: int) -> np.ndarray:
        actor = self._pool.get_actor(self._scene_id)
        raw   = self._safe_get(actor.render_main.remote(active_slot_count))
        return (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape(self._num_envs, self._cam_h, self._cam_w, 3)
            .copy()
        )

    def render_4dir(self, active_slot_count: int) -> Optional[np.ndarray]:
        actor = self._pool.get_actor(self._scene_id)
        raw   = self._safe_get(actor.render_4dir.remote(active_slot_count))
        if raw is None:
            return None
        return (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape(self._num_envs, 3, self._cam_h, self._cam_w, 3)
            .copy()
        )

    # --- Async rendering (Phase 3) -------------------------------------------

    def render_main_async(self, active_slot_count: int):
        """Submit render request; return Ray ObjectRef immediately (non-blocking).

        Uses actor_handle (no health_check round-trip) so K scenes can render
        concurrently across GPUs. Crash detection happens in fetch_render_main.
        """
        actor = self._pool.actor_handle(self._scene_id)
        return actor.render_main.remote(active_slot_count)

    def render_4dir_async(self, active_slot_count: int):
        """Submit 4-dir render request; return Ray ObjectRef or None if disabled.

        Uses actor_handle (no health_check round-trip) for concurrent fan-out.
        """
        actor = self._pool.actor_handle(self._scene_id)
        return actor.render_4dir.remote(active_slot_count)

    def render_main_with_depth_async(self, active_slot_count: int):
        """Submit RGB+depth render; return Ray ObjectRef (non-blocking)."""
        actor = self._pool.actor_handle(self._scene_id)
        return actor.render_main_with_depth.remote(active_slot_count)

    def fetch_render_main(self, ref, timeout: float = 30.0) -> np.ndarray:
        """Block until async render result is ready, then deserialise."""
        raw = self._safe_get(ref, timeout=timeout)
        return (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape(self._num_envs, self._cam_h, self._cam_w, 3)
            .copy()
        )

    def fetch_render_4dir(self, ref, timeout: float = 30.0) -> Optional[np.ndarray]:
        """Block until async 4-dir result is ready, then deserialise. Returns None if ref is None."""
        if ref is None:
            return None
        raw = self._safe_get(ref, timeout=timeout)
        if raw is None:
            return None
        return (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape(self._num_envs, 3, self._cam_h, self._cam_w, 3)
            .copy()
        )

    def fetch_render_main_with_depth(
        self, ref, timeout: float = 30.0
    ) -> tuple:
        """Block until RGB+depth result is ready. Returns (rgb (N,H,W,3) uint8, depth (N,H,W) float32)."""
        rgb_bytes, depth_bytes = self._safe_get(ref, timeout=timeout)
        rgb = (
            np.frombuffer(rgb_bytes, dtype=np.uint8)
            .reshape(self._num_envs, self._cam_h, self._cam_w, 3)
            .copy()
        )
        depth = (
            np.frombuffer(depth_bytes, dtype=np.float32)
            .reshape(self._num_envs, self._cam_h, self._cam_w)
            .copy()
        )
        return rgb, depth

    # --- Internal helpers ----------------------------------------------------

    def _safe_get(self, ref, timeout: float = 60.0):
        """ray.get() with crash detection.

        If the actor raises RayActorError (SIGSEGV / OOM / killed), we
        immediately trigger a pool rebuild and raise SceneCrashError so
        GenarkVecEnv can enter dormant mode gracefully.
        """
        try:
            return ray.get(ref, timeout=timeout)
        except (
            ray.exceptions.RayActorError,
            ray.exceptions.WorkerCrashedError,
            ray.exceptions.RayTaskError,
        ) as exc:
            self._pool.rebuild_actor(self._scene_id)
            raise SceneCrashError(self._scene_id, exc) from exc

    def _sync_state(self, state: dict) -> None:
        """Update local tensor copies from actor-returned state dict."""
        self._cam_pos_t = torch.from_numpy(
            state["cam_pos"].copy()
        ).to(self._device)
        self._cam_yaw_t = torch.from_numpy(
            state["cam_yaw"].copy()
        ).to(self._device)
        self._current_tri_idx_t = torch.from_numpy(
            state["current_tri_idx"].copy()
        ).to(self._device)

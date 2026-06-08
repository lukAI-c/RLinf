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

@ray.remote(num_gpus=1)
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
        dev = self._backend.device
        self._backend.step_physics(
            torch.from_numpy(actions_np).to(dev),
            torch.from_numpy(active_mask_np).to(dev),
            active_slot_count,
        )
        return self._get_state_dict()

    def render_main(self, active_slot_count: int) -> bytes:
        """Returns (N, H, W, 3) uint8 as raw bytes."""
        rgb = self._backend.render_main(active_slot_count)
        return rgb.tobytes()

    def render_4dir(self, active_slot_count: int) -> Optional[bytes]:
        """Returns (N, 3, H, W, 3) uint8 as raw bytes, or None."""
        extras = self._backend.render_4dir(active_slot_count)
        if extras is None:
            return None
        return extras.tobytes()

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

    def __init__(self, cfg: DictConfig, scene_assignments: dict[str, int]):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self._cfg               = cfg
        self._cfg_dict          = OmegaConf.to_container(cfg, resolve=True)
        self._scene_assignments = scene_assignments
        self._actors: dict[str, ray.actor.ActorHandle] = {}

        for scene_id, num_envs in scene_assignments.items():
            self._actors[scene_id] = self._create_actor(scene_id, num_envs)

    def _create_actor(self, scene_id: str, num_envs: int) -> ray.actor.ActorHandle:
        return GenesisSceneActor.remote(self._cfg_dict, scene_id, num_envs)

    def get_actor(self, scene_id: str) -> ray.actor.ActorHandle:
        """Return the actor for a scene, rebuilding it if it has crashed."""
        actor = self._actors[scene_id]
        try:
            ray.get(actor.health_check.remote(), timeout=2.0)
        except Exception:
            self.rebuild_actor(scene_id)
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
        """Submit render request; return Ray ObjectRef immediately (non-blocking)."""
        actor = self._pool.get_actor(self._scene_id)
        return actor.render_main.remote(active_slot_count)

    def render_4dir_async(self, active_slot_count: int):
        """Submit 4-dir render request; return Ray ObjectRef or None if disabled."""
        actor = self._pool.get_actor(self._scene_id)
        return actor.render_4dir.remote(active_slot_count)

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
            ray.exceptions.RayWorkerError,
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

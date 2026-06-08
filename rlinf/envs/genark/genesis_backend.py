"""
GenesisSimBackend — abstract interface + local implementation.

Encapsulates all Genesis simulator calls (scene build, camera render) and
NavMesh geometry (floor-finding, sliding collision). GenarkVecEnv uses only
this interface; it never calls gs.* directly.

Two implementations:
  GenesisLocalBackend  — in-process Genesis (current behaviour, default)
  GenesisRemoteBackend — thin Ray-RPC client (defined in genesis_server.py)
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# NavMesh physics helpers  (moved from genark_env.py)
# ---------------------------------------------------------------------------

def _batch_cross_2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _batch_is_point_in_triangle(pt, v0, v1, v2):
    def sign(p1, p2, p3):
        return (
            (p1[..., 0] - p3[..., 0]) * (p2[..., 1] - p3[..., 1])
            - (p2[..., 0] - p3[..., 0]) * (p1[..., 1] - p3[..., 1])
        )
    d1, d2, d3 = sign(pt, v0, v1), sign(pt, v1, v2), sign(pt, v2, v0)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def _batch_get_z_on_triangle(v0, v1, v2, x, y):
    edge1  = v1 - v0
    edge2  = v2 - v0
    normal = torch.cross(edge1, edge2, dim=-1)
    mask   = torch.abs(normal[..., 2]) < 1e-6
    d_x    = x - v0[..., 0]
    d_y    = y - v0[..., 1]
    num    = normal[..., 0] * d_x + normal[..., 1] * d_y
    den    = normal[..., 2]
    den    = torch.where(torch.abs(den) < 1e-6, torch.ones_like(den), den)
    z      = v0[..., 2] - num / den
    return torch.where(mask.expand_as(z), torch.full_like(z, float("inf")), z)


def _batch_find_floor(
    pos_2d, current_floor_z,
    tri_v0_2d, tri_v1_2d, tri_v2_2d,
    tri_v0_3d, tri_v1_3d, tri_v2_3d,
    max_step_height: float = 0.5,
):
    pt    = pos_2d.unsqueeze(1)
    v0_2  = tri_v0_2d.unsqueeze(0)
    v1_2  = tri_v1_2d.unsqueeze(0)
    v2_2  = tri_v2_2d.unsqueeze(0)
    inside = _batch_is_point_in_triangle(pt, v0_2, v1_2, v2_2)

    v0_3  = tri_v0_3d.unsqueeze(0)
    v1_3  = tri_v1_3d.unsqueeze(0)
    v2_3  = tri_v2_3d.unsqueeze(0)
    z_vals = _batch_get_z_on_triangle(v0_3, v1_3, v2_3, pt[..., 0], pt[..., 1])

    z_diff     = torch.abs(z_vals - current_floor_z.unsqueeze(1))
    valid_cand = inside & (z_diff < max_step_height)
    z_diff_m   = torch.where(valid_cand, z_diff, torch.full_like(z_diff, float("inf")))

    min_dist, best_idx = torch.min(z_diff_m, dim=1)
    valid_mask = min_dist != float("inf")
    best_z     = torch.gather(z_vals, 1, best_idx.unsqueeze(1)).squeeze(1)
    return valid_mask, best_z, best_idx


def _batch_get_sliding_position(
    current_pos, desired_pos, current_tri_idx,
    tri_v0_2d, tri_v1_2d, tri_v2_2d,
    agent_radius: float = 0.0,
):
    p0 = tri_v0_2d[current_tri_idx]
    p1 = tri_v1_2d[current_tri_idx]
    p2 = tri_v2_2d[current_tri_idx]

    edge_starts = torch.stack([p0, p1, p2], dim=1)
    edge_ends   = torch.stack([p1, p2, p0], dim=1)
    edge_vecs   = edge_ends - edge_starts

    p_des  = desired_pos.unsqueeze(1)
    p_cur  = current_pos.unsqueeze(1)
    cp1    = _batch_cross_2d(edge_vecs, p_des - edge_starts)
    cp0    = _batch_cross_2d(edge_vecs, p_cur - edge_starts)
    crossing = (cp1 * cp0) < 0

    v_ap  = p_des - edge_starts
    dot   = (v_ap * edge_vecs).sum(dim=-1)
    sq    = (edge_vecs * edge_vecs).sum(dim=-1)
    sq_s  = torch.where(sq < 1e-6, torch.ones_like(sq), sq)
    t     = torch.clamp(dot / sq_s, 0.0, 1.0)
    candidates = edge_starts + t.unsqueeze(-1) * edge_vecs

    if agent_radius > 0.0:
        centroid  = ((p0 + p1 + p2) / 3.0).unsqueeze(1)
        perp_a    = torch.stack([-edge_vecs[..., 1], edge_vecs[..., 0]], dim=-1)
        perp_b    = -perp_a
        to_ctr    = centroid - edge_starts
        dot_a     = (perp_a * to_ctr).sum(dim=-1)
        inward    = torch.where(dot_a.unsqueeze(-1) > 0, perp_a, perp_b)
        inward_len = torch.norm(inward, dim=-1, keepdim=True).clamp_min(1e-6)
        inward_unit = inward / inward_len
        candidates  = candidates + inward_unit * agent_radius

    has_crossing, first_idx = torch.max(crossing.long(), dim=1)
    selected = torch.gather(
        candidates, 1, first_idx.view(-1, 1, 1).expand(-1, 1, 2)
    ).squeeze(1)
    return torch.where(has_crossing.view(-1, 1).bool(), selected, current_pos)


def _calculate_initial_yaw(rotations: torch.Tensor) -> torch.Tensor:
    """Habitat quaternion [x,y,z,w] → Genesis yaw (radians)."""
    x, y, z, w = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    dir_x_g =  -2 * w * y
    dir_y_g = -(y * y - w * w)
    return torch.atan2(dir_y_g, dir_x_g)


def _hab_to_genesis(pos):
    x, y, z = pos[0], pos[1], pos[2]
    return x, -z, y


def _genesis_to_hab(cam_pos_gen: torch.Tensor, camera_height: float) -> torch.Tensor:
    x = cam_pos_gen[:, 0]
    y = cam_pos_gen[:, 2] - camera_height
    z = -cam_pos_gen[:, 1]
    return torch.stack([x, y, z], dim=1)


def _update_camera(cam, cam_pos: torch.Tensor, cam_yaw: torch.Tensor):
    dir_x  = torch.cos(cam_yaw)
    dir_y  = torch.sin(cam_yaw)
    lookat = cam_pos.clone()
    lookat[:, 0] += dir_x
    lookat[:, 1] += dir_y
    lookat[:, 2]  = cam_pos[:, 2]
    up = torch.zeros_like(cam_pos)
    up[:, 2] = 1.0
    cam.set_pose(pos=cam_pos, lookat=lookat, up=up)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class GenesisSimBackend(ABC):
    """Encapsulates all Genesis physics/render + NavMesh geometry.

    GenarkVecEnv communicates with Genesis exclusively through this interface.
    Two implementations: GenesisLocalBackend (in-process) and
    GenesisRemoteBackend (Ray RPC, defined in genesis_server.py).

    State ownership
    ---------------
    Backend owns: cam_pos, cam_yaw, current_tri_idx, NavMesh triangles,
                  Genesis scene/renderer/camera.
    GenarkVecEnv owns: episode assignment, GRPO groups, reward tracking,
                       active_mask, slot_done, elapsed_steps.
    """

    # Action constants (same as GenarkVecEnv)
    STOP         = 0
    MOVE_FORWARD = 1
    TURN_LEFT    = 2
    TURN_RIGHT   = 3

    # --- State properties (backend-owned, read-only for GenarkVecEnv) -------

    @property
    @abstractmethod
    def cam_pos(self) -> torch.Tensor:
        """(num_envs, 3) float32, Genesis coordinate system."""
        ...

    @property
    @abstractmethod
    def cam_yaw(self) -> torch.Tensor:
        """(num_envs,) float32, radians."""
        ...

    @property
    @abstractmethod
    def current_tri_idx(self) -> torch.Tensor:
        """(num_envs,) int64, current NavMesh triangle index per env."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    # --- Scene lifecycle -----------------------------------------------------

    @abstractmethod
    def load_scene(self, scene_id: str, n_active_envs: int) -> None:
        """Load mesh + navmesh, call gs.Scene.build(n_envs=n_active_envs).
        May only be called once per backend instance (Genesis cannot rebuild)."""
        ...

    # --- Agent pose management -----------------------------------------------

    @abstractmethod
    def set_agent_poses(
        self,
        env_idx: list[int],
        positions: torch.Tensor,   # (len(env_idx), 3) float32, Genesis coords
        yaws: torch.Tensor,        # (len(env_idx),) float32, radians
    ) -> None:
        """Set agent poses, snap to NavMesh floor, update Genesis camera.
        positions come from episode start_position converted via _hab_to_genesis.
        After this call, cam_pos[env_idx] and cam_yaw[env_idx] hold snapped values.
        """
        ...

    # --- Physics step --------------------------------------------------------

    @abstractmethod
    def step_physics(
        self,
        actions: torch.Tensor,        # (N,) int64: STOP/FWD/LEFT/RIGHT
        active_mask: torch.Tensor,    # (N,) bool: envs that should move
        active_slot_count: int,
    ) -> None:
        """Apply rotation + forward movement using NavMesh collision.
        Updates cam_pos, cam_yaw, current_tri_idx in-place.
        Updates Genesis camera pose (needed before next render).
        Does NOT render — caller must call render_main / render_4dir.
        """
        ...

    # --- Rendering -----------------------------------------------------------

    @abstractmethod
    def render_main(self, active_slot_count: int) -> np.ndarray:
        """Render front-view. Returns (num_envs, H, W, 3) uint8 numpy.
        Ghost slots (index >= active_slot_count) are zero-padded."""
        ...

    @abstractmethod
    def render_4dir(self, active_slot_count: int) -> Optional[np.ndarray]:
        """Render left/right/behind extra views.
        Returns (num_envs, 3, H, W, 3) uint8 numpy, or None if
        enable_4dir_render is False."""
        ...

    # --- Coordinate helpers (exposed so GenarkVecEnv can convert metrics) ----

    def genesis_to_hab(self, cam_pos_gen: torch.Tensor, camera_height: float) -> torch.Tensor:
        return _genesis_to_hab(cam_pos_gen, camera_height)

    def hab_to_genesis(self, pos):
        return _hab_to_genesis(pos)

    def calculate_initial_yaw(self, rotations: torch.Tensor) -> torch.Tensor:
        return _calculate_initial_yaw(rotations)


# ---------------------------------------------------------------------------
# Local (in-process) implementation
# ---------------------------------------------------------------------------

class GenesisLocalBackend(GenesisSimBackend):
    """In-process Genesis backend — identical behaviour to original GenarkVecEnv.

    All Genesis calls happen in the same process/GPU as the EnvWorker.
    This is the default backend; behaviour is bit-for-bit identical to the
    previous monolithic GenarkVecEnv implementation.
    """

    def __init__(self, cfg, num_envs: int):
        self._num_envs     = num_envs
        self._cam_w        = int(tuple(getattr(cfg, "cam_res", (640, 480)))[0])
        self._cam_h        = int(tuple(getattr(cfg, "cam_res", (640, 480)))[1])
        self._fov          = int(getattr(cfg, "fov", 105))
        self._light_scale  = float(getattr(cfg, "light_scale", 4.0))
        self._enable_4dir  = bool(getattr(cfg, "enable_4dir_render", False))
        self._step_move    = float(getattr(cfg, "step_move", 0.25))
        self._step_turn    = float(getattr(cfg, "step_turn", math.radians(30.0)))
        self._allow_sliding = bool(getattr(cfg, "allow_sliding", True))
        self._max_step_height = float(getattr(cfg, "max_step_height", 0.5))
        self._agent_radius = float(getattr(cfg, "agent_radius", 0.18))
        self._camera_height = float(getattr(cfg, "camera_height", 1.25))
        self._scene_datasets = str(cfg.init_params.scene_datasets)
        self._glb_cache_dir  = getattr(cfg.init_params, "glb_cache_dir",
                                       "/home/clk/workspace/genark/glb_cache")

        # Genesis objects — created here, scene built in load_scene()
        try:
            import genesis as gs
        except ImportError as e:
            raise ImportError(
                "GenesisLocalBackend requires the Genesis simulator."
            ) from e
        self._gs = gs

        if not gs._initialized:
            gs.init(backend=gs.cuda)

        self._renderer = gs.renderers.BatchRenderer(use_rasterizer=True)
        self._gs_scene = gs.Scene(
            renderer=self._renderer,
            show_viewer=False,
            vis_options=gs.options.VisOptions(
                ambient_light=(1.0, 1.0, 1.0),
                plane_reflection=False,
            ),
        )
        self._cam = self._gs_scene.add_camera(
            res=(self._cam_w, self._cam_h),
            fov=self._fov,
            GUI=False,
        )
        self._gs_ready = False

        # NavMesh geometry tensors — populated in load_scene()
        self._tri_v0_2d = self._tri_v1_2d = self._tri_v2_2d = None
        self._tri_v0_3d = self._tri_v1_3d = self._tri_v2_3d = None

        # Agent state tensors — initialised lazily in set_agent_poses()
        self._cam_pos_t:         Optional[torch.Tensor] = None
        self._cam_yaw_t:         Optional[torch.Tensor] = None
        self._current_tri_idx_t: Optional[torch.Tensor] = None

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
        return self._gs.device

    # --- Scene lifecycle -----------------------------------------------------

    def load_scene(self, scene_id: str, n_active_envs: int) -> None:
        if self._gs_ready:
            raise RuntimeError(
                f"GenesisLocalBackend: attempted to load scene '{scene_id}' but "
                "scene is already built. Genesis does not support scene rebuilding."
            )

        scan_name = os.path.basename(os.path.dirname(scene_id))

        # Resolve mesh path: GLB cache first, OBJ fallback
        glb_path = os.path.join(self._glb_cache_dir, f"{scan_name}.glb")
        if os.path.exists(glb_path):
            mesh_path = glb_path
        else:
            obj_dir   = os.path.join(self._scene_datasets, scan_name, "matterport_mesh")
            obj_files = [f for f in os.listdir(obj_dir) if f.endswith(".obj")]
            if not obj_files:
                raise FileNotFoundError(
                    f"No mesh for '{scan_name}'. Tried:\n"
                    f"  {glb_path}\n  {obj_dir}/*.obj"
                )
            mesh_path = os.path.join(obj_dir, obj_files[0])

        self._gs_scene.add_entity(morph=self._gs.morphs.Mesh(
            file=mesh_path, fixed=True, collision=False,
            file_meshes_are_zup=True,
        ))

        # Load navmesh and convert to Genesis coordinate system
        navmesh_path = os.path.join(self._scene_datasets, scan_name, "navmesh.npz")
        nm     = np.load(navmesh_path)
        device = self._gs.device
        V_hab  = torch.tensor(nm["verts"], device=device, dtype=torch.float32)
        F_idx  = torch.tensor(nm["faces"], device=device, dtype=torch.long)

        V_gen       = torch.zeros_like(V_hab)
        V_gen[:, 0] =  V_hab[:, 0]
        V_gen[:, 1] = -V_hab[:, 2]
        V_gen[:, 2] =  V_hab[:, 1]

        self._tri_v0_3d = V_gen[F_idx[:, 0]]
        self._tri_v1_3d = V_gen[F_idx[:, 1]]
        self._tri_v2_3d = V_gen[F_idx[:, 2]]
        self._tri_v0_2d = self._tri_v0_3d[:, :2]
        self._tri_v1_2d = self._tri_v1_3d[:, :2]
        self._tri_v2_2d = self._tri_v2_3d[:, :2]

        # Build scene — only active_slot_count envs need VRAM
        self._gs_scene.build(n_envs=n_active_envs, env_spacing=(0.0, 0.0))
        self._gs_ready      = True
        self._n_active_envs = n_active_envs

    # --- Agent pose management -----------------------------------------------

    def set_agent_poses(
        self,
        env_idx: list[int],
        positions: torch.Tensor,   # (len(env_idx), 3) float32, Genesis coords
        yaws: torch.Tensor,        # (len(env_idx),) float32
    ) -> None:
        device = self._gs.device
        N = self._num_envs

        # Lazy initialisation of state tensors
        if self._cam_pos_t is None:
            self._cam_pos_t         = torch.zeros(N, 3, dtype=torch.float32, device=device)
            self._cam_yaw_t         = torch.zeros(N,    dtype=torch.float32, device=device)
            self._current_tri_idx_t = torch.zeros(N,    dtype=torch.long,    device=device)

        idx_t = torch.tensor(env_idx, device=device)
        self._cam_pos_t[idx_t] = positions.to(device)
        self._cam_yaw_t[idx_t] = yaws.to(device)

        # Snap to NavMesh floor
        valid, new_z, tri_idx = _batch_find_floor(
            self._cam_pos_t[idx_t, :2],
            self._cam_pos_t[idx_t, 2] - self._camera_height,
            self._tri_v0_2d, self._tri_v1_2d, self._tri_v2_2d,
            self._tri_v0_3d, self._tri_v1_3d, self._tri_v2_3d,
            self._max_step_height,
        )
        self._cam_pos_t[idx_t, 2]      = new_z + self._camera_height
        self._current_tri_idx_t[idx_t] = tri_idx

        _update_camera(
            self._cam,
            self._cam_pos_t[:self._n_active_envs],
            self._cam_yaw_t[:self._n_active_envs],
        )

    # --- Physics step --------------------------------------------------------

    def step_physics(
        self,
        actions: torch.Tensor,        # (N,) int64
        active_mask: torch.Tensor,    # (N,) bool
        active_slot_count: int,
    ) -> None:
        # Rotation
        self._cam_yaw_t[(actions == self.TURN_LEFT)  & active_mask] += self._step_turn
        self._cam_yaw_t[(actions == self.TURN_RIGHT) & active_mask] -= self._step_turn

        # Forward movement with NavMesh collision
        fwd_m = (actions == self.MOVE_FORWARD) & active_mask
        if fwd_m.any():
            fwd_idx = fwd_m.nonzero(as_tuple=True)[0]
            dx = torch.cos(self._cam_yaw_t[fwd_idx])
            dy = torch.sin(self._cam_yaw_t[fwd_idx])
            desired = torch.stack([
                self._cam_pos_t[fwd_idx, 0] + dx * self._step_move,
                self._cam_pos_t[fwd_idx, 1] + dy * self._step_move,
            ], dim=1)
            cur_floor_z = self._cam_pos_t[fwd_idx, 2] - self._camera_height

            valid, new_z, new_tri = _batch_find_floor(
                desired, cur_floor_z,
                self._tri_v0_2d, self._tri_v1_2d, self._tri_v2_2d,
                self._tri_v0_3d, self._tri_v1_3d, self._tri_v2_3d,
                self._max_step_height,
            )

            if self._allow_sliding:
                invalid = ~valid
                if invalid.any():
                    inv_local = invalid.nonzero(as_tuple=True)[0]
                    slide_pos = _batch_get_sliding_position(
                        self._cam_pos_t[fwd_idx[inv_local], :2],
                        desired[inv_local],
                        self._current_tri_idx_t[fwd_idx[inv_local]],
                        self._tri_v0_2d, self._tri_v1_2d, self._tri_v2_2d,
                        self._agent_radius,
                    )
                    sv, sz, st = _batch_find_floor(
                        slide_pos, cur_floor_z[inv_local],
                        self._tri_v0_2d, self._tri_v1_2d, self._tri_v2_2d,
                        self._tri_v0_3d, self._tri_v1_3d, self._tri_v2_3d,
                        self._max_step_height,
                    )
                    if sv.any():
                        can_slide = inv_local[sv]
                        desired[can_slide]  = slide_pos[sv]
                        new_z[can_slide]    = sz[sv]
                        new_tri[can_slide]  = st[sv]
                        valid[can_slide]    = True

            moved = fwd_idx[valid]
            self._cam_pos_t[moved, 0]      = desired[valid, 0]
            self._cam_pos_t[moved, 1]      = desired[valid, 1]
            self._cam_pos_t[moved, 2]      = new_z[valid] + self._camera_height
            self._current_tri_idx_t[moved] = new_tri[valid]

        _update_camera(
            self._cam,
            self._cam_pos_t[:active_slot_count],
            self._cam_yaw_t[:active_slot_count],
        )

    # --- Rendering -----------------------------------------------------------

    def _render_and_scale(self, active_slot_count: int) -> torch.Tensor:
        """Render front view for active slots, apply light scale. Returns GPU tensor."""
        rgb_raw, _, _, _ = self._cam.render(
            rgb=True, depth=False, segmentation=False, force_render=True
        )
        if rgb_raw.dtype == torch.uint8:
            rgb_float = rgb_raw.float() / 255.0
        else:
            rgb_float = rgb_raw
        return torch.clamp(rgb_float * self._light_scale * 255.0, 0, 255).byte()

    def _pad_ghost_slots(self, rgb_active: torch.Tensor, active_slot_count: int) -> torch.Tensor:
        """Pad ghost slots with zeros so output shape is always (num_envs, ...)."""
        if active_slot_count < self._num_envs:
            pad_shape = (self._num_envs - active_slot_count,) + rgb_active.shape[1:]
            pad = torch.zeros(pad_shape, dtype=torch.uint8, device=rgb_active.device)
            return torch.cat([rgb_active, pad], dim=0)
        return rgb_active

    def render_main(self, active_slot_count: int) -> np.ndarray:
        _update_camera(
            self._cam,
            self._cam_pos_t[:active_slot_count],
            self._cam_yaw_t[:active_slot_count],
        )
        rgb_active = self._render_and_scale(active_slot_count)
        rgb_batch  = self._pad_ghost_slots(rgb_active, active_slot_count)
        return rgb_batch.cpu().numpy()

    def render_4dir(self, active_slot_count: int) -> Optional[np.ndarray]:
        if not self._enable_4dir:
            return None
        if active_slot_count == 0:
            return None

        deltas = [math.radians(90.0), math.radians(-90.0), math.radians(180.0)]
        front_yaw = self._cam_yaw_t[:active_slot_count]
        extras_np = []

        for d_yaw in deltas:
            rotated_yaw = front_yaw + d_yaw
            _update_camera(self._cam, self._cam_pos_t[:active_slot_count], rotated_yaw)
            rgb_active = self._render_and_scale(active_slot_count)
            rgb_padded = self._pad_ghost_slots(rgb_active, active_slot_count)
            extras_np.append(rgb_padded.cpu().numpy())

        # Restore front camera pose
        _update_camera(
            self._cam,
            self._cam_pos_t[:active_slot_count],
            front_yaw,
        )

        # (3, N, H, W, 3) → (N, 3, H, W, 3)
        return np.stack(extras_np, axis=1)

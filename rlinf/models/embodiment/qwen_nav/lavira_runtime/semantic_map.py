"""Legacy sparse mapper retained only for ``map_backend=sparse_ab`` diagnostics.

The production LaViRA path uses the byte-identical LHX mapper vendored under
``rlinf.third_party.lavira_rft.source`` through ``LaviraSourceCore``. Do not
extend this module to reimplement or approximate LHX mapping behavior again.
It remains available only for historical Genesis compatibility and numerical
A/B diagnostics against the source backend.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np
from skimage.morphology import binary_closing, disk, remove_small_objects

from ..action_parser import ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT
from ..lavira_map import FMMPlanner, TURN_ANGLE_DEG, _angle_and_direction
from .source_port import collision_check_fmm


BASE_CLASSES = (
    "chair", "couch", "plant", "bed", "toilet", "tv", "table", "oven",
    "sink", "refrigerator", "book", "clock", "vase", "cup", "bottle", "floor",
)
NAVIGABLE_CLASSES = {
    "stair", "stairs", "step", "steps", "stairway", "stairways", "staircase",
    "staircases", "floor", "ground", "walkway",
}


@dataclass(frozen=True)
class MapPose:
    row: int
    col: int


class LaviraSemanticMap:
    """Persistent open-vocabulary semantic map for one Genesis environment slot.

    Channels follow the upstream layout: obstacle, explored, current agent,
    past agent, then one dynamically allocated channel per detected class.
    """

    map_size_cm = 2400
    resolution_cm = 5
    grid_size = 480
    center = 240
    cells_per_m = 20.0
    frame_width = 160
    frame_height = 120
    # Match lavira-rft MAP.VISION_RANGE / MAP_RESOLUTION: a 100-cell local
    # voxel window, i.e. 5m at the default 5cm resolution.
    vision_range_cells = 100
    height_min_cm = -40
    height_max_cm = 360
    obstacle_min_height_cm = 25
    exp_pred_threshold = 1.0
    map_pred_threshold = 1.0
    cat_pred_threshold = 5.0

    def __init__(
        self,
        hfov_deg: float = 79.0,
        camera_height: float = 0.88,
        map_size_cm: int = 2400,
        resolution_cm: int = 5,
        frame_width: int = 160,
        frame_height: int = 120,
    ) -> None:
        self.hfov_deg = float(hfov_deg)
        self.camera_height = float(camera_height)
        self.map_size_cm = int(map_size_cm)
        self.resolution_cm = int(resolution_cm)
        self.grid_size = self.map_size_cm // self.resolution_cm
        self.center = self.grid_size // 2
        self.cells_per_m = 100.0 / self.resolution_cm
        self.mapping_range_m = self.vision_range_cells / self.cells_per_m
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self._fx = self.frame_width / (2.0 * math.tan(math.radians(self.hfov_deg / 2.0)))
        vfov = 2.0 * math.atan(
            self.frame_height / self.frame_width * math.tan(math.radians(self.hfov_deg / 2.0))
        )
        self._fy = self.frame_height / (2.0 * math.tan(vfov / 2.0))
        self.reset()

    def reset(
        self,
        hab_x: float = 0.0,
        hab_z: float = 0.0,
        yaw_rad: float = 0.0,
    ) -> None:
        self.origin_x, self.origin_z = float(hab_x), float(hab_z)
        # LHX mapping starts each episode at (map_center, map_center, 0 deg).
        # Keep the external Habitat world pose, but express all map geometry in
        # that initial-agent frame so mapping, collision and FMM share exactly
        # the same basis as full_pose in ZS_Evaluator_mp.
        self.origin_yaw = float(yaw_rad)
        self.obstacle = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.explored = np.zeros_like(self.obstacle)
        self.current_agent = np.zeros_like(self.obstacle)
        self.past_agent = np.zeros_like(self.obstacle)
        self.collision = np.zeros_like(self.obstacle, dtype=bool)
        self.semantic: OrderedDict[str, np.ndarray] = OrderedDict()
        self.detected_classes: list[str] = []
        self.last_pose: tuple[float, float, float] | None = None
        self.last_action: int | None = None
        self.last_forward_collision = False
        self.last_collision_audit: dict | None = None
        self.last_fmm_audit: dict | None = None
        # LaViRA computes _process_map once after each real observation and
        # passes the resulting traversable grid to FMM.  Keep that same cache
        # boundary instead of re-running morphology for every replay action.
        self._traversible_cache: np.ndarray | None = None
        self._traversible_before_collision: np.ndarray | None = None
        # Kept only for debug-map rendering. It is never read by planning.
        self.agent_path_xz: list[tuple[float, float]] = []
        self.initialized = True

    @property
    def channels(self) -> np.ndarray:
        semantic = list(self.semantic.values())
        return np.stack(
            [self.obstacle, self.explored, self.current_agent, self.past_agent, *semantic], axis=0
        )

    def _world_to_source_xy(self, x, z):
        dx = np.asarray(x) - self.origin_x
        dz = np.asarray(z) - self.origin_z
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        source_x = dx * c - dz * s
        source_y = -dx * s - dz * c
        return source_x, source_y

    def _world_to_map_float(self, x: float, z: float) -> np.ndarray:
        source_x, source_y = self._world_to_source_xy(x, z)
        return np.asarray([
            self.center + float(source_y) * self.cells_per_m,
            self.center + float(source_x) * self.cells_per_m,
        ], dtype=np.float32)

    def world_to_map(self, x: float, z: float) -> MapPose:
        row_col = self._world_to_map_float(x, z)
        return MapPose(
            int(np.clip(row_col[0], 0, self.grid_size - 1)),
            int(np.clip(row_col[1], 0, self.grid_size - 1)),
        )

    def map_to_world(self, row: float, col: float) -> tuple[float, float]:
        source_x = (col - self.center) / self.cells_per_m
        source_y = (row - self.center) / self.cells_per_m
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return (
            self.origin_x + source_x * c - source_y * s,
            self.origin_z - source_x * s - source_y * c,
        )

    def _relative_yaw(self, yaw_rad: float) -> float:
        return math.atan2(
            math.sin(float(yaw_rad) - self.origin_yaw),
            math.cos(float(yaw_rad) - self.origin_yaw),
        )

    def _register_classes(self, masks: Mapping[str, np.ndarray]) -> None:
        for label in masks:
            name = str(label).strip().lower()
            if name and name not in self.semantic:
                self.semantic[name] = np.zeros_like(self.obstacle)
                self.detected_classes.append(name)

    def update(
        self,
        depth_m: np.ndarray,
        hab_x: float,
        hab_z: float,
        gen_yaw_rad: float,
        semantic_masks: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        """Fuse one RGB-D frame through LaViRA-style sparse voxel splatting.

        This is a source-aligned sparse approximation of the upstream
        ``point_cloud -> splat_feat_nd -> height projection`` path.  Unlike the
        source mapper, it does not materialise and transform a local dense
        ``100 x 100 x 80`` voxel tensor, so it is not bitwise equivalent near
        clipping and interpolation boundaries.
        """
        if depth_m is None or depth_m.ndim != 2:
            return
        masks = semantic_masks or {}
        self._register_classes(masks)
        depth_source = depth_m.astype(np.float32, copy=False)
        ds = depth_source.shape[1] // self.frame_width
        if ds > 1:
            # Exact LHX _preprocess_state sampling grid.
            depth = depth_source[ds // 2::ds, ds // 2::ds]
        else:
            depth = depth_source
        depth = depth[:self.frame_height, :self.frame_width]
        if depth.shape != (self.frame_height, self.frame_width):
            raise ValueError(
                "depth cannot be source-downsampled to configured map frame: "
                f"input={depth_source.shape} output={depth.shape} "
                f"expected={(self.frame_height, self.frame_width)}"
            )
        h, w = depth.shape
        yy, xx = np.indices((h, w), dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0.1) & (depth <= self.mapping_range_m)
        if not np.any(valid):
            return

        # Genesis yaw maps to Habitat forward=(cos(yaw), -sin(yaw)). The local
        # window mirrors LaViRA's 100 x 100-cell voxel crop: 5m forward and
        # +/-2.5m laterally at the default map resolution.
        d = depth[valid]
        # Match LHX depth_utils.get_camera_matrix() principal point.
        cx = (w - 1.0) / 2.0
        cy = (h - 1.0) / 2.0
        right = (xx[valid] - cx) * d / self._fx
        forward = d
        height = self.camera_height + (cy - yy[valid]) * d / self._fy
        local = np.abs(right) <= self.mapping_range_m / 2.0
        if not np.any(local):
            return
        right, forward, height = right[local], forward[local], height[local]
        c, s = math.cos(gen_yaw_rad), math.sin(gen_yaw_rad)
        # Genesis→Habitat reflects Genesis Y into Habitat Z.  Therefore an
        # image-space right ray maps to (+sin(yaw), +cos(yaw)) in Habitat XZ.
        wx = float(hab_x) + forward * c + right * s
        wz = float(hab_z) - forward * s + right * c
        source_x, source_y = self._world_to_source_xy(wx, wz)
        rows_f = self.center + source_y * self.cells_per_m
        cols_f = self.center + source_x * self.cells_per_m
        height_voxel = (height * 100.0 - self.height_min_cm) / self.resolution_cm
        in_height = (height_voxel >= 0.0) & (
            height_voxel < (self.height_max_cm - self.height_min_cm) / self.resolution_cm
        )
        in_map = (
            (rows_f >= 0.0) & (rows_f < self.grid_size - 1) &
            (cols_f >= 0.0) & (cols_f < self.grid_size - 1) & in_height
        )
        if not np.any(in_map):
            return
        rows_f, cols_f, height_voxel = (
            rows_f[in_map], cols_f[in_map], height_voxel[in_map]
        )
        # Trilinear voxel splatting followed by a height sum. The all-height
        # projection sums to one per valid point; the obstacle band receives
        # only mass in LaViRA's [25cm, agent-height+1cm] height interval.
        obstacle_lo = (self.obstacle_min_height_cm - self.height_min_cm) / self.resolution_cm
        obstacle_hi = (
            (self.camera_height * 100.0 + 1.0 - self.height_min_cm) / self.resolution_cm
        )
        z0 = np.floor(height_voxel)
        z_fraction = height_voxel - z0
        obstacle_weight = (
            (1.0 - z_fraction) * ((z0 >= obstacle_lo) & (z0 < obstacle_hi))
            + z_fraction * (((z0 + 1.0) >= obstacle_lo) & ((z0 + 1.0) < obstacle_hi))
        ).astype(np.float32)
        explored_delta = self._bilinear_splat(rows_f, cols_f, np.ones_like(rows_f, dtype=np.float32))
        obstacle_delta = self._bilinear_splat(rows_f, cols_f, obstacle_weight)
        explored_evidence = np.clip(
            explored_delta / self.exp_pred_threshold, 0.0, 1.0
        )
        obstacle_evidence = np.clip(
            obstacle_delta / self.map_pred_threshold, 0.0, 1.0
        )
        # Exact LHX map fusion: map_pred = max(old_map, translated).
        self.explored = np.maximum(self.explored, explored_evidence)
        self.obstacle = np.maximum(self.obstacle, obstacle_evidence)

        for label, raw_mask in masks.items():
            name = str(label).strip().lower()
            if name not in self.semantic:
                continue
            mask_source = np.asarray(raw_mask, dtype=np.float32)
            if ds > 1:
                mask = mask_source[ds // 2::ds, ds // 2::ds]
            else:
                mask = mask_source
            mask = mask[:h, :w]
            if mask.shape != (h, w):
                raise ValueError(
                    f"semantic mask shape {mask_source.shape} does not match depth "
                    f"sampling grid {(h, w)}"
                )
            projected = mask[valid][local][in_map]
            if np.any(projected > 0):
                # LHX writes semantics from agent_height_proj, the same
                # 25cm..camera-height band used for obstacle evidence.
                semantic_delta = self._bilinear_splat(
                    rows_f,
                    cols_f,
                    obstacle_weight * projected,
                )
                semantic_evidence = np.clip(
                    semantic_delta / self.cat_pred_threshold, 0.0, 1.0
                )
                self.semantic[name] = np.maximum(
                    self.semantic[name], semantic_evidence
                )

        self.current_agent.fill(0.0)
        agent = self.world_to_map(hab_x, hab_z)
        rs = slice(max(0, agent.row - 1), min(self.grid_size, agent.row + 2))
        cs = slice(max(0, agent.col - 1), min(self.grid_size, agent.col + 2))
        self.current_agent[rs, cs] = 1.0
        self.past_agent[rs, cs] = 1.0
        if (
            not self.agent_path_xz
            or math.hypot(
                float(hab_x) - self.agent_path_xz[-1][0],
                float(hab_z) - self.agent_path_xz[-1][1],
            ) >= 0.02
        ):
            self.agent_path_xz.append((float(hab_x), float(hab_z)))
        self.last_forward_collision = False
        if self.last_pose is not None and self.last_action == ACTION_FORWARD:
            if math.hypot(float(hab_x) - self.last_pose[0], float(hab_z) - self.last_pose[1]) < 0.20:
                self._mark_collision(
                    self.last_pose,
                    (float(hab_x), float(hab_z), float(gen_yaw_rad)),
                )
                self.last_forward_collision = True
        self.last_pose = (float(hab_x), float(hab_z), float(gen_yaw_rad))

    def _bilinear_splat(
        self, rows_f: np.ndarray, cols_f: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """Sparse 2D remainder of LaViRA's trilinear voxel splat."""
        out = np.zeros_like(self.obstacle)
        if len(rows_f) == 0:
            return out
        row0 = np.floor(rows_f).astype(np.int32)
        col0 = np.floor(cols_f).astype(np.int32)
        row_fraction = rows_f - row0
        col_fraction = cols_f - col0
        for dr, dc, weights in (
            (0, 0, (1.0 - row_fraction) * (1.0 - col_fraction)),
            (1, 0, row_fraction * (1.0 - col_fraction)),
            (0, 1, (1.0 - row_fraction) * col_fraction),
            (1, 1, row_fraction * col_fraction),
        ):
            np.add.at(out, (row0 + dr, col0 + dc), values * weights)
        return out

    def _mark_collision(
        self,
        last_pose: tuple[float, float, float],
        current_pose: tuple[float, float, float],
    ) -> None:
        """Apply lavira-rft's ``collision_check_fmm`` mask verbatim."""
        # Preserve fractional map coordinates. LHX collision_check_fmm derives
        # its 11x11 ring offset from full_pose floats; converting to MapPose
        # first silently discarded that information.
        last = self._world_to_map_float(last_pose[0], last_pose[1])
        current = self._world_to_map_float(current_pose[0], current_pose[1])
        collision = collision_check_fmm(
            last,
            current,
            self._relative_yaw(current_pose[2]),
            self.collision.shape,
            cells_per_m=self.cells_per_m,
        )
        prior = self.collision.copy()
        self.collision |= collision
        self.last_collision_audit = {
            "last_map_rc": [float(last[0]), float(last[1])],
            "current_map_rc": [float(current[0]), float(current[1])],
            "relative_yaw_rad": float(self._relative_yaw(current_pose[2])),
            "candidate_cells": int(np.count_nonzero(collision)),
            "new_cells": int(np.count_nonzero(self.collision & ~prior)),
            "total_cells": int(np.count_nonzero(self.collision)),
        }

    def set_last_action(self, action: int | None) -> None:
        self.last_action = action

    def traversible(self) -> np.ndarray:
        """Return the cached LaViRA-RFT ``_process_map`` result.

        The cache is rebuilt after ``LaviraNavigationController.observe``.
        This mirrors the source evaluator's ``self.traversable`` field: FMM
        may be called repeatedly while replaying a macro action, but map
        morphology is evaluated only after the next observation is fused.
        """
        if self._traversible_cache is not None:
            return self._traversible_cache
        self._traversible_cache = self._compute_traversible()
        return self._traversible_cache

    def rebuild_traversible(self) -> np.ndarray:
        """Refresh the cached traversibility grid after a map update."""
        self._traversible_cache = self._compute_traversible()
        return self._traversible_cache

    def _compute_traversible(self) -> np.ndarray:
        """LaViRA-RFT ``_process_map`` semantic traversibility rule."""
        # LHX calls remove_small_objects once on the full channel tensor.
        full_map = remove_small_objects(self.channels.astype(bool), min_size=64)
        obstacles = full_map[0]
        explored = full_map[1]
        objects = np.zeros_like(obstacles)
        navigable = np.zeros_like(obstacles)
        for semantic_index, label in enumerate(self.semantic):
            cleaned = full_map[4 + semantic_index]
            if label in NAVIGABLE_CLASSES:
                navigable |= cleaned
            else:
                objects |= cleaned

        footprint = disk(3)
        obstacles_closed = binary_closing(obstacles, footprint=footprint)
        objects_closed = binary_closing(objects, footprint=footprint)
        navigable &= ~objects
        navigable_closed = binary_closing(navigable, footprint=footprint)

        blocked = objects_closed | obstacles_closed
        blocked[navigable_closed] = False
        blocked = remove_small_objects(blocked, min_size=64)
        blocked = binary_closing(blocked, footprint=footprint)
        traversible = ~blocked

        # Preserve the source operation order: floor is formed from raw
        # obstacle/object channels and can reopen cells introduced only by
        # morphological closing in ``blocked``.
        free_mask = ~(obstacles | objects)
        free_mask |= navigable
        floor = explored & free_mask
        floor = binary_closing(remove_small_objects(floor, min_size=400), footprint=disk(3))
        traversible |= floor
        # Match LaViRA-RFT evaluator order: semantic navigability may override
        # geometry in _process_map(), but collision_check_fmm is applied after
        # that result and therefore always wins.
        self._traversible_before_collision = traversible.copy()
        traversible[self.collision] = False
        return traversible.astype(np.float32)

    def fmm_action(
        self,
        hab_x: float,
        hab_z: float,
        gen_yaw_rad: float,
        goal_x: float,
        goal_z: float,
        goal_threshold_m: float = 1.0,
    ) -> int | None:
        trav = self.traversible()
        # LHX Policy._get_action keeps full_pose-derived map coordinates as
        # floats. Their fractional part controls the 11x11 FMM annulus. The
        # previous adapter rounded these to MapPose integers before planning.
        agent = self._world_to_map_float(hab_x, hab_z)
        agent[0] = np.clip(agent[0], 0, self.grid_size - 1)
        agent[1] = np.clip(agent[1], 0, self.grid_size - 1)
        goal = self.world_to_map(goal_x, goal_z)
        requested_goal = np.asarray([goal.row, goal.col], dtype=np.int32)
        requested_goal_traversible = bool(trav[goal.row, goal.col])
        if trav[goal.row, goal.col] == 0:
            candidates = np.argwhere(trav > 0)
            if len(candidates) == 0:
                self.last_fmm_audit = {
                    "status": "no_traversible_cells",
                    "agent_map_rc": agent.astype(float).tolist(),
                    "requested_goal_map_rc": requested_goal.tolist(),
                }
                return None
            goal_arr = candidates[np.argmin(np.sum((candidates - np.asarray([goal.row, goal.col])) ** 2, axis=1))]
        else:
            goal_arr = np.asarray([goal.row, goal.col])
        planner = FMMPlanner(trav)
        try:
            planner.set_goal(goal_arr)
            # Policy._get_action passes None as fixed_destination. The source
            # evaluator owns arrival through its outer 0.75 m distance check;
            # the local FMM call only chooses the next primitive direction.
            # Set the open-waypoint threshold below every finite distance to
            # reproduce the source function's effective ``stop=False`` branch.
            planner.wp_thresh = -np.inf
            row, col, stop = planner.get_short_term_goal(
                agent.astype(np.float32), None
            )
        except Exception as exc:
            self.last_fmm_audit = {
                "status": "planner_error",
                "error": type(exc).__name__,
                "agent_map_rc": agent.astype(float).tolist(),
                "requested_goal_map_rc": requested_goal.tolist(),
                "nearest_goal_map_rc": np.asarray(goal_arr).astype(int).tolist(),
            }
            return None
        if stop:
            self.last_fmm_audit = {
                "status": "source_stop",
                "agent_map_rc": agent.astype(float).tolist(),
                "requested_goal_map_rc": requested_goal.tolist(),
                "nearest_goal_map_rc": np.asarray(goal_arr).astype(int).tolist(),
            }
            return None
        target = np.asarray([row - agent[0], col - agent[1]], dtype=np.float32)
        if np.linalg.norm(target) < 1e-6:
            self.last_fmm_audit = {
                "status": "zero_waypoint_vector",
                "agent_map_rc": agent.astype(float).tolist(),
                "requested_goal_map_rc": requested_goal.tolist(),
                "nearest_goal_map_rc": np.asarray(goal_arr).astype(int).tolist(),
                "short_term_goal_map_rc": [int(row), int(col)],
            }
            return None
        # Reuse the established LaViRA coordinate conversion and action rule
        # from ``lavira_map.OccupancyMap._single_fmm_step``.  The FMM map is
        # row/column while Genesis yaw is Cartesian; hand-deriving this cross
        # product here previously inverted LEFT and RIGHT.
        h_input_deg = -math.degrees(self._relative_yaw(gen_yaw_rad))
        hvec = np.asarray(
            [math.cos(math.radians(h_input_deg)), math.sin(math.radians(h_input_deg))],
            dtype=np.float64,
        )
        heading = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64) @ hvec
        angle, action = _angle_and_direction(heading, target, TURN_ANGLE_DEG)
        agent_cell = np.floor(agent).astype(np.int32)
        fmm_dist = getattr(planner, "fmm_dist", None)
        fmm_distance_at_agent = (
            float(fmm_dist[agent_cell[0], agent_cell[1]])
            if fmm_dist is not None else None
        )
        radius = 5
        r0, r1 = max(0, agent_cell[0] - radius), min(self.grid_size, agent_cell[0] + radius + 1)
        c0, c1 = max(0, agent_cell[1] - radius), min(self.grid_size, agent_cell[1] + radius + 1)
        before_collision = self._traversible_before_collision
        local_before_collision = (
            int(np.count_nonzero(before_collision[r0:r1, c0:c1]))
            if before_collision is not None else None
        )
        self.last_fmm_audit = {
            "status": "action",
            "agent_map_rc": agent.astype(float).tolist(),
            "requested_goal_map_rc": requested_goal.tolist(),
            "requested_goal_traversible": requested_goal_traversible,
            "nearest_goal_map_rc": np.asarray(goal_arr).astype(int).tolist(),
            "goal_snapped": bool(np.any(np.asarray(goal_arr) != requested_goal)),
            "collision_cells_total": int(np.count_nonzero(self.collision)),
            "collision_cells_local_11x11": int(np.count_nonzero(self.collision[r0:r1, c0:c1])),
            "traversible_cells_before_collision_local_11x11": local_before_collision,
            "traversible_cells_local_11x11": int(np.count_nonzero(trav[r0:r1, c0:c1])),
            "fmm_distance_at_agent": fmm_distance_at_agent,
            "short_term_goal_map_rc": [int(row), int(col)],
            "heading_vector_rc": heading.astype(float).tolist(),
            "waypoint_vector_rc": target.astype(float).tolist(),
            "relative_angle_deg": float(angle),
            "primitive_action": int(action),
        }
        return action

    def preview_fmm_reachability(
        self,
        hab_x: float,
        hab_z: float,
        goal_x: float,
        goal_z: float,
    ) -> dict:
        """Read-only FMM connectivity check for the diagnostic sparse map."""
        traversible = self.traversible()
        goal = self.world_to_map(goal_x, goal_z)
        agent = np.floor(self._world_to_map_float(hab_x, hab_z)).astype(np.int64)
        agent[0] = np.clip(agent[0], 0, traversible.shape[0] - 1)
        agent[1] = np.clip(agent[1], 0, traversible.shape[1] - 1)
        if not bool(traversible[goal.row, goal.col]):
            return {
                "reachable": False, "status": "endpoint_blocked",
                "goal_map_rc": [int(goal.row), int(goal.col)],
                "agent_map_rc": agent.astype(int).tolist(),
                "distance_at_agent": None,
            }
        if not bool(traversible[agent[0], agent[1]]):
            return {
                "reachable": False, "status": "agent_blocked",
                "goal_map_rc": [int(goal.row), int(goal.col)],
                "agent_map_rc": agent.astype(int).tolist(),
                "distance_at_agent": None,
            }
        try:
            planner = FMMPlanner(traversible)
            planner.set_goal(np.asarray([goal.row, goal.col], dtype=np.int64))
            distances = np.asarray(planner.fmm_dist)
            distance = float(distances[agent[0], agent[1]])
            maximum = float(np.max(distances))
            has_masked_cells = bool(np.any(traversible == 0))
            reachable = bool(
                np.isfinite(distance)
                and (
                    not has_masked_cells
                    or distance < maximum - np.finfo(distances.dtype).eps
                )
            )
        except Exception as exc:
            return {
                "reachable": False, "status": "planner_error",
                "error": type(exc).__name__,
                "goal_map_rc": [int(goal.row), int(goal.col)],
                "agent_map_rc": agent.astype(int).tolist(),
                "distance_at_agent": None,
            }
        return {
            "reachable": reachable,
            "status": "reachable" if reachable else "disconnected",
            "goal_map_rc": [int(goal.row), int(goal.col)],
            "agent_map_rc": agent.astype(int).tolist(),
            "distance_at_agent": distance,
            "unreachable_fill_distance": maximum if has_masked_cells else None,
        }

    def render_debug_map(
        self,
        *,
        gt_reference_path: np.ndarray | None = None,
        goal_xz: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Render the accumulated 2D map for diagnostics only.

        The returned BGR image intentionally makes observed map evidence and
        dataset GT visually distinct. Neither GT nor this rendering enters the
        policy, planner, reward, or action path.
        """
        explored = self.explored.astype(bool)
        blocked = (self.obstacle > 0.0) | self.collision
        traversible = self.traversible().astype(bool)
        image = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        image[:] = (28, 28, 28)  # unexplored
        image[explored] = (82, 82, 82)
        image[explored & traversible] = (145, 145, 145)
        image[blocked] = (20, 20, 20)

        semantic_colors = (
            (190, 80, 255), (255, 170, 30), (80, 210, 80),
            (220, 80, 80), (230, 80, 210), (80, 190, 230),
        )
        for idx, channel in enumerate(self.semantic.values()):
            mask = channel.astype(bool)
            if not np.any(mask):
                continue
            color = np.asarray(semantic_colors[idx % len(semantic_colors)], dtype=np.uint8)
            image[mask] = ((image[mask].astype(np.uint16) + color) // 2).astype(np.uint8)

        def _polyline(points: np.ndarray, color: tuple[int, int, int], width: int) -> None:
            if len(points) < 2:
                return
            pixels = np.asarray(
                [[self.world_to_map(float(x), float(z)).col,
                  self.world_to_map(float(x), float(z)).row] for x, z in points],
                dtype=np.int32,
            )
            cv2.polylines(image, [pixels], isClosed=False, color=color, thickness=width, lineType=cv2.LINE_AA)

        if gt_reference_path is not None and len(gt_reference_path):
            gt = np.asarray(gt_reference_path, dtype=np.float32)
            if gt.ndim == 2 and gt.shape[1] >= 3:
                gt = gt[:, [0, 2]]
            _polyline(gt, (0, 165, 255), 2)  # orange: GT reference path
            start = self.world_to_map(float(gt[0, 0]), float(gt[0, 1]))
            end = self.world_to_map(float(gt[-1, 0]), float(gt[-1, 1]))
            cv2.circle(image, (start.col, start.row), 5, (0, 255, 0), -1)
            cv2.circle(image, (end.col, end.row), 5, (0, 0, 255), -1)

        if self.agent_path_xz:
            _polyline(np.asarray(self.agent_path_xz, dtype=np.float32), (255, 255, 0), 2)  # cyan: agent
            current = self.world_to_map(*self.agent_path_xz[-1])
            cv2.circle(image, (current.col, current.row), 4, (255, 0, 0), -1)
        if goal_xz is not None:
            goal = self.world_to_map(*goal_xz)
            cv2.drawMarker(image, (goal.col, goal.row), (0, 255, 255), cv2.MARKER_CROSS, 11, 2)

        legend = (
            "gray=explored  black=blocked  cyan=agent  orange=GT  "
            "green=start  red=goal  yellow=FMM target"
        )
        cv2.rectangle(image, (0, 0), (self.grid_size, 20), (0, 0, 0), -1)
        cv2.putText(image, legend, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def save_debug_snapshot(
        self,
        png_path: str,
        *,
        npz_path: str | None = None,
        gt_reference_path: np.ndarray | None = None,
        goal_xz: tuple[float, float] | None = None,
    ) -> None:
        """Persist a debug-only map image and, optionally, raw map channels."""
        image = self.render_debug_map(
            gt_reference_path=gt_reference_path,
            goal_xz=goal_xz,
        )
        cv2.imwrite(str(png_path), image)
        if npz_path is None:
            return
        arrays: dict[str, np.ndarray] = {
            "obstacle": self.obstacle,
            "explored": self.explored,
            "collision": self.collision.astype(np.uint8),
            "agent_path_xz": np.asarray(self.agent_path_xz, dtype=np.float32),
            "origin_xz": np.asarray([self.origin_x, self.origin_z], dtype=np.float32),
        }
        for idx, (label, channel) in enumerate(self.semantic.items()):
            arrays[f"semantic_{idx}_{label}"] = channel
        np.savez_compressed(npz_path, **arrays)

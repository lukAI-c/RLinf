"""Thin RLinf lifecycle adapter around the pinned LHX navigation core."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import cv2
import numpy as np
import torch
from skimage.morphology import binary_closing, disk, remove_small_objects

from .loader import load_source_core


def _namespace(**values):
    return SimpleNamespace(**values)


def _source_config(
    device: torch.device,
    results_dir: Path,
    *,
    hfov_deg: float,
    camera_height: float,
    map_size_cm: int,
    resolution_cm: int,
    frame_width: int,
    frame_height: int,
):
    map_cfg = _namespace(
        HFOV=float(hfov_deg),
        MIN_Z=2,
        DEVICE=device,
        DU_SCALE=1,
        # LHX R2R overrides this to True. Policy._get_action also constructs
        # its FMM debug field only on this branch before writing it.
        VISUALIZE=True,
        PRINT_IMAGES=False,
        FRAME_WIDTH=int(frame_width),
        FRAME_HEIGHT=int(frame_height),
        VISION_RANGE=100,
        MAP_RESOLUTION=int(resolution_cm),
        MAP_SIZE_CM=int(map_size_cm),
        GLOBAL_DOWNSCALING=2,
        CAT_PRED_THRESHOLD=5.0,
        EXP_PRED_THRESHOLD=1.0,
        MAP_PRED_THRESHOLD=1.0,
        NUM_ENVIRONMENTS=1,
        AGENT_HEIGHT=float(camera_height),
        CENTER_RESET_STEPS=25,
        RESULTS_DIR=str(results_dir),
    )
    return _namespace(
        MAP=map_cfg,
        EVAL=_namespace(
            FMM_WAYPOINT_THRESHOLD=2.0,
            FMM_GOAL_THRESHOLD=1.0,
            DECISION_THRESHOLD=0.4,
            SCORE_THRESHOLD=0.5,
            VALUE_THRESHOLD=0.30,
            CHANGE_THRESHOLD=-0.03,
        ),
        TASK_CONFIG=_namespace(SIMULATOR=_namespace(TURN_ANGLE=30.0)),
        RESULTS_DIR=str(results_dir),
    )


class LaviraSourceCore:
    """Run LHX Semantic_Mapping, map processing, collision and Policy FMM.

    World/RPC ownership remains in RLinf. All map and planner calculations are
    delegated to the byte-identical source modules loaded by ``loader.py``.
    """

    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        results_dir: str | Path = "/tmp/rlinf_lavira_source",
        hfov_deg: float = 79.0,
        camera_height: float = 0.88,
        map_size_cm: int = 2400,
        resolution_cm: int = 5,
        frame_width: int = 160,
        frame_height: int = 120,
    ) -> None:
        self.source = load_source_core()
        self.device = torch.device(device)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        trace_root = os.environ.get("RLINF_LAVIRA_SOURCE_TRACE_DIR", "").strip()
        trace_max_frames = os.environ.get(
            "RLINF_LAVIRA_SOURCE_TRACE_MAX_FRAMES", "0"
        ).strip()
        self.trace_dir = (
            Path(trace_root) / str(os.getpid()) / str(id(self))
            if trace_root else None
        )
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_max_frames = int(trace_max_frames or 0)
        self.resolution_cm = int(resolution_cm)
        self.grid_size = int(map_size_cm) // self.resolution_cm
        self.center = self.grid_size // 2
        self.cells_per_m = 100.0 / self.resolution_cm
        self.config = _source_config(
            self.device,
            self.results_dir,
            hfov_deg=hfov_deg,
            camera_height=camera_height,
            map_size_cm=map_size_cm,
            resolution_cm=resolution_cm,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        self.mapper = self.source.Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapper.eval()
        self.mapper._rlinf_trace_enabled = self.trace_dir is not None
        self.policy = self.source.FusionMapPolicy(self.config, self.grid_size)
        self.policy.reset()
        self.fmm_episode_label = "0"
        self.detected_classes = self.source.OrderedSet()
        self.reset()

    def reset(self, hab_x: float = 0.0, hab_z: float = 0.0, yaw_rad: float = 0.0) -> None:
        self.origin_x = float(hab_x)
        self.origin_z = float(hab_z)
        self.origin_yaw = float(yaw_rad)
        self.detected_classes = self.source.OrderedSet()
        self.mapper.reset()
        self.mapper.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        self.policy.reset()
        self.collision = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self._traversible = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self._traversible_before_collision = self._traversible.copy()
        self.floor = np.zeros_like(self._traversible)
        self.last_pose: tuple[float, float, float] | None = None
        self.last_source_pose: np.ndarray | None = None
        self.last_full_pose: np.ndarray | None = None
        self.last_action: int | None = None
        self.last_forward_collision = False
        self.last_collision_audit = None
        self.last_fmm_audit = None
        self.step = 0
        self.agent_path_xz: list[tuple[float, float]] = []
        self.last_depth_m: np.ndarray | None = None
        self.last_semantic_masks: dict[str, np.ndarray] = {}
        self.last_external_pose: np.ndarray | None = None

    def _world_to_source_pose(self, x: float, z: float, yaw: float) -> np.ndarray:
        dx, dz = float(x) - self.origin_x, float(z) - self.origin_z
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        source_x = dx * c - dz * s
        source_y = -dx * s - dz * c
        heading = math.atan2(math.sin(yaw - self.origin_yaw), math.cos(yaw - self.origin_yaw))
        return np.asarray([source_x, source_y, heading], dtype=np.float32)

    @staticmethod
    def _relative_sensor_pose(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        x1, y1, o1 = previous
        x2, y2, o2 = current
        theta = math.atan2(float(y2 - y1), float(x2 - x1)) - float(o1)
        distance = math.hypot(float(x2 - x1), float(y2 - y1))
        return np.asarray(
            [distance * math.cos(theta), distance * math.sin(theta), float(o2 - o1)],
            dtype=np.float32,
        )

    @staticmethod
    def _source_depth_cm(depth_m: np.ndarray) -> np.ndarray:
        # Exact ZS_Evaluator_mp._preprocess_depth semantics, with the bridge's
        # metric depth first mapped back to Habitat's normalized sensor range.
        normalized = np.clip((np.asarray(depth_m, dtype=np.float32) - 0.1) / 4.9, 0.0, 1.0)
        depth = normalized.copy()
        for column in range(depth.shape[1]):
            values = depth[:, column]
            values[values == 0.0] = values.max()
        depth[depth > 0.99] = 0.0
        depth[depth == 0.0] = 1.0
        return 10.0 + depth * 490.0

    def _register_masks(self, masks: Mapping[str, np.ndarray]) -> None:
        for label in masks:
            name = str(label).strip().lower()
            if name:
                self.detected_classes.add(name)

    def _build_source_observation(
        self, depth_m: np.ndarray, masks: Mapping[str, np.ndarray]
    ) -> torch.Tensor:
        self._register_masks(masks)
        depth_cm = self._source_depth_cm(depth_m)
        ds = depth_cm.shape[1] // self.config.MAP.FRAME_WIDTH
        depth_small = depth_cm[ds // 2::ds, ds // 2::ds][
            : self.config.MAP.FRAME_HEIGHT, : self.config.MAP.FRAME_WIDTH
        ]
        semantic = np.zeros(
            (len(self.detected_classes), *depth_small.shape), dtype=np.float32
        )
        for label, raw_mask in masks.items():
            name = str(label).strip().lower()
            index = self.detected_classes.index(name)
            mask = np.asarray(raw_mask, dtype=np.float32)
            semantic[index] = mask[ds // 2::ds, ds // 2::ds][
                : depth_small.shape[0], : depth_small.shape[1]
            ]
        rgb = np.zeros((3, *depth_small.shape), dtype=np.float32)
        state = np.concatenate((rgb, depth_small[None], semantic), axis=0)
        return torch.from_numpy(state[None]).float().to(self.device)

    def update(
        self,
        depth_m: np.ndarray,
        hab_x: float,
        hab_z: float,
        yaw_rad: float,
        semantic_masks: Mapping[str, np.ndarray] | None = None,
        *,
        publish_planner_state: bool = True,
    ) -> None:
        masks = semantic_masks or {}
        self.last_depth_m = np.asarray(depth_m, dtype=np.float32).copy()
        self.last_semantic_masks = {
            str(label): np.asarray(mask, dtype=np.float32).copy()
            for label, mask in masks.items()
        }
        self.last_external_pose = np.asarray(
            [hab_x, hab_z, yaw_rad], dtype=np.float32
        )
        current = self._world_to_source_pose(hab_x, hab_z, yaw_rad)
        previous = current if self.last_source_pose is None else self.last_source_pose
        sensor_pose = self._relative_sensor_pose(current, previous)
        observation = self._build_source_observation(depth_m, masks)
        pose_tensor = torch.from_numpy(sensor_pose[None]).float().to(self.device)
        with torch.no_grad():
            self.mapper(observation, pose_tensor, self.step)
            full_map, full_pose, _ = self.mapper.update_map(
                self.step, self.detected_classes, 0
            )
            self.mapper.one_step_full_map.fill_(0.0)
            self.mapper.one_step_local_map.fill_(0.0)

        trace_this_frame = self.trace_dir is not None and (
            self.trace_max_frames <= 0 or self.step < self.trace_max_frames
        )
        self.mapper._rlinf_trace_enabled = trace_this_frame
        if trace_this_frame:
            trace = dict(getattr(self.mapper, "_rlinf_trace", {}))
            labels = list(getattr(self.detected_classes, "order", ()))
            floor_channel = 4 + labels.index("floor") if "floor" in labels else None
            np.savez_compressed(
                self.trace_dir / f"frame_{self.step:04d}.npz",
                raw_depth_m=np.asarray(depth_m, dtype=np.float32),
                source_pose=current,
                previous_source_pose=previous,
                sensor_pose=sensor_pose,
                external_pose=np.asarray([hab_x, hab_z, yaw_rad], dtype=np.float32),
                fp_obstacle=trace.get("fp_obstacle"),
                local_obstacle_before=trace.get("local_obstacle_before"),
                translated_obstacle=trace.get("translated_obstacle"),
                local_obstacle_after=trace.get("local_obstacle_after"),
                full_obstacle=np.asarray(full_map[0, 0], dtype=np.float32),
                full_explored=np.asarray(full_map[0, 1], dtype=np.float32),
                full_floor=(
                    np.asarray(full_map[0, floor_channel], dtype=np.float32)
                    if floor_channel is not None
                    else np.empty((0,), dtype=np.float32)
                ),
                full_pose=np.asarray(full_pose[0], dtype=np.float32),
            )
            self.mapper._rlinf_trace = {}

        self.last_forward_collision = False
        if self.last_full_pose is not None and self.last_action == 1:
            collision = self.source.map_utils.collision_check_fmm(
                self.last_full_pose,
                full_pose[0],
                self.resolution_cm,
                self.collision.shape,
            ).astype(bool)
            prior = self.collision.copy()
            self.collision |= collision
            self.last_forward_collision = bool(
                np.linalg.norm(current[:2] - previous[:2]) < 0.2
            )
            self.last_collision_audit = {
                "candidate_cells": int(collision.sum()),
                "new_cells": int(np.count_nonzero(self.collision & ~prior)),
                "total_cells": int(self.collision.sum()),
            }

        self.last_source_pose = current
        self.last_full_pose = full_pose[0].copy()
        self.last_pose = (float(hab_x), float(hab_z), float(yaw_rad))
        # LHX get_panorama() advances the mapper for all 12 turn frames but
        # leaves the evaluator's outer full_map/full_pose snapshot untouched.
        # A later ordinary primitive publishes the mapper's accumulated state.
        if publish_planner_state:
            self.full_map = full_map[0]
            self.full_pose = full_pose[0]
        self.step += 1
        if not self.agent_path_xz or math.hypot(
            hab_x - self.agent_path_xz[-1][0], hab_z - self.agent_path_xz[-1][1]
        ) >= 0.02:
            self.agent_path_xz.append((float(hab_x), float(hab_z)))

    def _process_map(self) -> None:
        # Source-identical body extracted from ZS_Evaluator_mp._process_map.
        navigable_index = self.source.map_utils.process_navigable_classes(
            self.detected_classes
        )
        not_navigable_index = [
            i for i in range(len(self.detected_classes)) if i not in navigable_index
        ]
        full_map = remove_small_objects(self.full_map.astype(bool), min_size=64)
        obstacles = full_map[0].astype(bool)
        explored = full_map[1].astype(bool)
        semantic = full_map[4:]
        objects = np.sum(semantic[not_navigable_index], axis=0).astype(bool)
        footprint = disk(3)
        obstacles_closed = binary_closing(obstacles, footprint=footprint)
        objects_closed = binary_closing(objects, footprint=footprint)
        navigable = np.logical_or.reduce(semantic[navigable_index])
        navigable = np.logical_and(navigable, np.logical_not(objects))
        navigable_closed = binary_closing(navigable, footprint=footprint)
        untraversable = np.logical_or(objects_closed, obstacles_closed)
        untraversable[navigable_closed == 1] = 0
        untraversable = remove_small_objects(untraversable, min_size=64)
        untraversable = binary_closing(untraversable, footprint=footprint)
        traversible = np.logical_not(untraversable)
        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        floor = explored * free_mask
        floor = remove_small_objects(floor.astype(bool), min_size=400)
        floor = binary_closing(floor, footprint=footprint)
        traversible = np.logical_or(floor, traversible)
        explored_closed = binary_closing(explored, footprint=footprint)
        contours, _ = cv2.findContours(
            explored_closed.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour_image = np.zeros(full_map.shape[-2:], dtype=np.uint8)
        contour_image = cv2.drawContours(
            contour_image, contours, -1, (255, 255, 255), thickness=3
        )
        frontiers = np.logical_and(floor, contour_image)
        frontiers = remove_small_objects(frontiers.astype(bool), min_size=64)
        self._traversible_before_collision = traversible.copy()
        traversible[self.collision] = False
        self._traversible = traversible
        self.floor = floor
        self.frontiers = frontiers.astype(np.uint8)

    @property
    def obstacle(self) -> np.ndarray:
        return self.full_map[0]

    @property
    def explored(self) -> np.ndarray:
        return self.full_map[1]

    @property
    def semantic(self) -> dict[str, np.ndarray]:
        labels = self.detected_classes.order[:-1]
        return {label: self.full_map[4 + i] for i, label in enumerate(labels)}

    @property
    def channels(self) -> np.ndarray:
        return self.full_map

    @property
    def current_agent(self) -> np.ndarray:
        return self.full_map[2]

    @property
    def past_agent(self) -> np.ndarray:
        return self.full_map[3]

    def _relative_yaw(self, yaw_rad: float) -> float:
        return math.atan2(
            math.sin(float(yaw_rad) - self.origin_yaw),
            math.cos(float(yaw_rad) - self.origin_yaw),
        )

    def traversible(self) -> np.ndarray:
        return self._traversible.astype(np.float32)

    def rebuild_traversible(self) -> np.ndarray:
        self._process_map()
        return self.traversible()

    def set_last_action(self, action: int | None) -> None:
        self.last_action = action

    def configure_fmm_output(
        self,
        results_dir: str | Path,
        episode_id: str | int,
        trial_id: str | int,
    ) -> None:
        """Route source FMM images to a stable episode/trial directory."""

        def _safe_component(value: str | int) -> str:
            component = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
            return component or "unknown"

        trial_component = _safe_component(trial_id)
        if trial_component.isdigit():
            trial_component = f"{int(trial_component):03d}"
        self.fmm_episode_label = (
            f"{_safe_component(episode_id)}_trial_{trial_component}"
        )
        output_root = Path(results_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        self.config.RESULTS_DIR = str(output_root)
        self.policy.config.RESULTS_DIR = str(output_root)

    def _world_to_source_xy(self, x: float, z: float) -> tuple[float, float]:
        return tuple(self._world_to_source_pose(x, z, self.origin_yaw)[:2])

    def world_to_map(self, x: float, z: float):
        sx, sy = self._world_to_source_xy(x, z)
        return _namespace(
            row=int(np.clip(self.center + sy * self.cells_per_m, 0, self.grid_size - 1)),
            col=int(np.clip(self.center + sx * self.cells_per_m, 0, self.grid_size - 1)),
        )

    def map_to_world(self, row: float, col: float) -> tuple[float, float]:
        sx = (col - self.center) / self.cells_per_m
        sy = (row - self.center) / self.cells_per_m
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return (
            self.origin_x + sx * c - sy * s,
            self.origin_z - sx * s - sy * c,
        )

    def fmm_action(
        self,
        hab_x: float,
        hab_z: float,
        yaw_rad: float,
        goal_x: float,
        goal_z: float,
        _goal_threshold_m: float = 1.0,
    ) -> int:
        goal = self.world_to_map(goal_x, goal_z)
        action = self.policy._get_action(
            self.full_pose,
            np.asarray([goal.row, goal.col]),
            self.full_map,
            self._traversible.copy(),
            self.collision,
            self.step,
            self.fmm_episode_label,
            list(self.detected_classes.order[:-1]),
            False,
        )
        agent_row_raw = int(self.full_pose[1] * self.cells_per_m)
        agent_col_raw = int(self.full_pose[0] * self.cells_per_m)
        fmm_height, fmm_width = self.policy.fmm_dist.shape
        agent_row_sampled = int(np.clip(agent_row_raw, 0, fmm_height - 1))
        agent_col_sampled = int(np.clip(agent_col_raw, 0, fmm_width - 1))
        self.last_fmm_audit = {
            "status": "source_policy",
            "requested_goal_map_rc": [goal.row, goal.col],
            "collision_cells_total": int(self.collision.sum()),
            "agent_map_rc_raw": [agent_row_raw, agent_col_raw],
            "agent_map_rc_sampled": [agent_row_sampled, agent_col_sampled],
            "agent_map_out_of_bounds": bool(
                agent_row_raw != agent_row_sampled
                or agent_col_raw != agent_col_sampled
            ),
            "fmm_distance_at_agent": float(
                self.policy.fmm_dist[agent_row_sampled, agent_col_sampled]
            ),
            "primitive_action": int(action),
        }
        return int(action)

    def preview_fmm_reachability(
        self,
        hab_x: float,
        hab_z: float,
        goal_x: float,
        goal_z: float,
    ) -> dict:
        """Evaluate source-FMM connectivity without changing planner state.

        This uses the pinned source ``FMMPlanner`` and the same full-pose map
        coordinates as ``FusionMapPolicy._get_action``.  It deliberately does
        not call the live policy, so ``fmm_dist``, debug output, and the next
        primitive action remain untouched.
        """
        del hab_x, hab_z  # Source planning owns pose through ``full_pose``.
        goal = self.world_to_map(goal_x, goal_z)
        traversible = self._traversible.copy()
        if not bool(traversible[goal.row, goal.col]):
            return {
                "reachable": False,
                "status": "endpoint_blocked",
                "goal_map_rc": [int(goal.row), int(goal.col)],
                "distance_at_agent": None,
            }
        if getattr(self, "full_pose", None) is None:
            return {
                "reachable": False,
                "status": "planner_pose_unavailable",
                "goal_map_rc": [int(goal.row), int(goal.col)],
                "distance_at_agent": None,
            }
        position = np.asarray([
            float(self.full_pose[1]) * self.cells_per_m,
            float(self.full_pose[0]) * self.cells_per_m,
        ], dtype=np.float32)
        agent = np.floor(position).astype(np.int64)
        agent[0] = np.clip(agent[0], 0, traversible.shape[0] - 1)
        agent[1] = np.clip(agent[1], 0, traversible.shape[1] - 1)
        if not bool(traversible[agent[0], agent[1]]):
            return {
                "reachable": False,
                "status": "agent_blocked",
                "goal_map_rc": [int(goal.row), int(goal.col)],
                "agent_map_rc": agent.astype(int).tolist(),
                "distance_at_agent": None,
            }
        try:
            planner = self.source.FMMPlanner(
                self.config, traversible, visualize=False
            )
            planner.set_goal(np.asarray([goal.row, goal.col], dtype=np.int64))
            distances = np.asarray(planner.fmm_dist)
            distance = float(distances[agent[0], agent[1]])
            maximum = float(np.max(distances))
            # Source set_goal fills every masked (obstacle or disconnected)
            # cell with max(reachable_distance)+1.  If no cell is masked, the
            # entire grid is connected and no fill sentinel exists.
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
                "reachable": False,
                "status": "planner_error",
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
        """Render diagnostics only; planning continues to use source arrays."""
        obstacle = (self.obstacle > 0).astype(np.uint8)
        explored = (self.explored > 0).astype(np.uint8)
        canvas = np.full((*obstacle.shape, 3), 245, dtype=np.uint8)
        canvas[explored > 0] = (220, 220, 220)
        canvas[obstacle > 0] = (45, 45, 45)
        canvas[self.collision] = (0, 0, 255)

        def polyline(points, color, thickness):
            if points is None or len(points) < 2:
                return
            pixels = [
                (self.world_to_map(float(p[0]), float(p[2] if len(p) > 2 else p[1])))
                for p in points
            ]
            xy = np.asarray([[p.col, p.row] for p in pixels], dtype=np.int32)
            cv2.polylines(canvas, [xy], False, color, thickness)

        polyline(gt_reference_path, (0, 180, 0), 2)
        polyline(np.asarray(self.agent_path_xz, dtype=np.float32), (255, 255, 0), 2)
        if goal_xz is not None:
            goal = self.world_to_map(*goal_xz)
            cv2.drawMarker(
                canvas, (goal.col, goal.row), (0, 255, 255),
                markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2,
            )
        return np.flipud(canvas)

    def save_debug_snapshot(
        self,
        png_path: str,
        *,
        npz_path: str | None = None,
        gt_reference_path: np.ndarray | None = None,
        goal_xz: tuple[float, float] | None = None,
    ) -> None:
        image = self.render_debug_map(
            gt_reference_path=gt_reference_path, goal_xz=goal_xz
        )
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(png_path, image)
        if npz_path:
            labels = np.asarray(list(self.last_semantic_masks), dtype=np.str_)
            masks = (
                np.stack([self.last_semantic_masks[label] for label in labels], axis=0)
                if len(labels)
                else np.empty((0, 0, 0), dtype=np.float32)
            )
            np.savez_compressed(
                npz_path,
                channels=self.channels,
                traversible=self._traversible,
                collision=self.collision,
                floor=self.floor,
                agent_path_xz=np.asarray(self.agent_path_xz, dtype=np.float32),
                depth_m=self.last_depth_m,
                semantic_labels=labels,
                semantic_masks=masks,
                pose_hab=self.last_external_pose,
                previous_action=np.asarray(
                    -1 if self.last_action is None else self.last_action,
                    dtype=np.int64,
                ),
            )

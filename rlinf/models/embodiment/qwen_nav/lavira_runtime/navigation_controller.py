"""High-level RLinf lifecycle adapter around the vendored LHX navigation core.

With ``map_backend=source``, mapping and FMM execute copied LHX source. The
decision lifecycle, waypoint memory, STOP checks, and batched rollout boundary
remain RLinf adaptations and therefore are not claimed to be byte-identical to
``ZS_Evaluator_mp.py``. Only VLM decisions enter RL; primitive FMM actions are
replay steps.
"""

from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
from typing import Optional

import numpy as np

from ..action_parser import (
    ACTION_FORWARD,
    ACTION_PANORAMA_SCAN,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
)
from .observation import LaviraObservation
from .semantic_map import BASE_CLASSES, LaviraSemanticMap
from .source_port import check_blocked_directions
from .waypoint_memory import WaypointMemory
from ..canonical_targets import PORTAL_CLASSES


class NavigationState(str, Enum):
    SCANNING = "scanning"
    NEED_DECISION = "need_decision"
    TURNING = "turning"
    NAVIGATING = "navigating"
    BACKTRACKING = "backtracking"


class LaviraNavigationController:
    def __init__(self, cfg) -> None:
        self.map_backend = str(getattr(cfg, "map_backend", "source"))
        self.map_device = str(getattr(cfg, "map_device", "cpu"))
        self.initial_scan_turns = int(getattr(cfg, "initial_scan_turns", 12))
        self.max_steps_to_target = int(getattr(cfg, "max_steps_to_target", 15))
        # Match LaViRA's Habitat R2R geometry.  The bridge also passes these
        # values explicitly, but the defaults must stay consistent when this
        # controller is used outside the bridge.
        self.hfov_deg = float(getattr(cfg, "hfov_deg", 79.0))
        self.camera_height = float(getattr(cfg, "camera_height", 0.88))
        self.map_size_cm = int(getattr(cfg, "map_size_cm", 2400))
        self.map_resolution_cm = int(getattr(cfg, "map_resolution_cm", 5))
        self.map_frame_width = int(getattr(cfg, "map_frame_width", 160))
        self.map_frame_height = int(getattr(cfg, "map_frame_height", 120))
        map_visualization_cfg = getattr(cfg, "map_visualization", None)
        self.target_reached_threshold_m = float(
            getattr(cfg, "target_reached_threshold_m", 0.75)
        )
        self.fmm_goal_threshold_m = float(getattr(cfg, "fmm_goal_threshold_m", 1.0))
        self.layered_backtrack_radius_m = float(
            getattr(cfg, "layered_backtrack_radius_m", 6.0)
        )
        self.area_goal_min_distance_m = float(
            getattr(cfg, "area_goal_min_distance_m", 1.5)
        )
        self.area_goal_max_distance_m = float(
            getattr(cfg, "area_goal_max_distance_m", 3.0)
        )
        self.area_goal_preferred_distance_m = float(
            getattr(cfg, "area_goal_preferred_distance_m", 2.1)
        )
        self.area_goal_min_ray_ratio = float(
            getattr(cfg, "area_goal_min_ray_ratio", 0.8)
        )
        self.reset()

    def reset(self) -> None:
        self.state = NavigationState.SCANNING
        self.scan_turns_done = 0
        if hasattr(self, "map"):
            self.map.reset()
        else:
            self.map = self._create_map()
        self.goal_xz: Optional[tuple[float, float]] = None
        # LHX emits one FMM primitive in the same branch that creates a local
        # waypoint. Its outer 0.75m check is reached only on the next loop,
        # after env.step() and the normal planner-state refresh.
        self.goal_just_set = False
        self.goal_direction = ""
        self.turn_queue: list[int] = []
        self.pending_target = None
        self.waypoints = WaypointMemory()
        self.steps_to_goal = 0
        # LaViRA updates this flag from every valid VLM response.  It controls
        # the semantic query set for subsequent real observations.
        self.stair = False
        self._backtrack_replan_waypoint_id: Optional[int] = None
        self._backtrack_replan_ready = False
        self._current_waypoint_id: Optional[int] = None
        self._removed_waypoint_ids: list[int] = []
        self._going_to_stop = False
        self._stop_check_after_scan = False
        self._stop_check_ready = False
        # Debug-only provenance for the VLM point -> projected FMM goal path.
        # This state is never consulted by navigation.
        self.audit_decision_id = 0
        self.audit_request = None
        self.audit_projection = None
        self._navigation_feedback: Optional[str] = None

    def _create_map(self):
        if self.map_backend == "source":
            from rlinf.third_party.lavira_rft.source_core import LaviraSourceCore

            results_dir = (
                Path("/tmp/rlinf_lavira_source")
                / str(os.getpid())
                / str(id(self))
            )
            return LaviraSourceCore(
                device=self.map_device,
                results_dir=results_dir,
                hfov_deg=self.hfov_deg,
                camera_height=self.camera_height,
                map_size_cm=self.map_size_cm,
                resolution_cm=self.map_resolution_cm,
                frame_width=self.map_frame_width,
                frame_height=self.map_frame_height,
            )
        if self.map_backend == "sparse_ab":
            return LaviraSemanticMap(
                hfov_deg=self.hfov_deg,
                camera_height=self.camera_height,
                map_size_cm=self.map_size_cm,
                resolution_cm=self.map_resolution_cm,
                frame_width=self.map_frame_width,
                frame_height=self.map_frame_height,
            )
        raise ValueError(
            f"unsupported LaViRA map_backend={self.map_backend!r}; "
            "expected 'source' or diagnostic-only 'sparse_ab'"
        )

    @property
    def semantic_query_classes(self) -> list[str]:
        if self.stair:
            return ["stairs"]
        classes = list(BASE_CLASSES)
        if self.state == NavigationState.SCANNING:
            classes.append("stairs")
        return classes

    def observe(
        self,
        observation: LaviraObservation,
        semantic_masks=None,
        *,
        publish_planner_state: bool = True,
    ) -> None:
        if self.map.last_pose is None:
            self.map.reset(
                observation.hab_x,
                observation.hab_z,
                observation.front_yaw,
            )
        if self.map_backend == "source":
            self.map.update(
                observation.front_depth,
                observation.hab_x,
                observation.hab_z,
                observation.front_yaw,
                semantic_masks,
                publish_planner_state=publish_planner_state,
            )
        else:
            self.map.update(
                observation.front_depth,
                observation.hab_x,
                observation.hab_z,
                observation.front_yaw,
                semantic_masks,
            )
        # Regular observations publish the same outer planner snapshot that
        # ZS_Evaluator_mp refreshes after env.step(). Panorama frames only
        # advance Semantic_Mapping; get_panorama() does not call _process_map().
        if publish_planner_state:
            self.map.rebuild_traversible()

    def configure_fmm_output(
        self,
        results_dir: str | Path,
        episode_id: str | int,
        trial_id: str | int,
    ) -> None:
        """Configure source-policy diagnostics for the current trial."""
        if self.map_backend == "source":
            self.map.configure_fmm_output(results_dir, episode_id, trial_id)

    def export_wm_snapshot(self) -> dict:
        """Copy runtime state for an isolated counterfactual branch."""
        from .wm_snapshot import export_runtime_snapshot

        return export_runtime_snapshot(self)

    def restore_wm_snapshot(self, snapshot: dict) -> None:
        """Restore a previously exported counterfactual runtime state."""
        from .wm_snapshot import restore_runtime_snapshot

        restore_runtime_snapshot(self, snapshot)

    def needs_decision(self) -> bool:
        return self.state == NavigationState.NEED_DECISION

    def available_waypoint_ids(self, observation: LaviraObservation) -> list[int]:
        # ZS_Evaluator_mp appends the current waypoint before querying the VLM,
        # then exposes every prior waypoint within 6m. It does not permanently
        # filter failed branches from this normal-decision candidate list.
        return [
            node.id
            for node in self.waypoints.nodes
            if node.id != self._current_waypoint_id
            if self._waypoint_distance(observation, node) <= self.layered_backtrack_radius_m
        ]

    @property
    def current_waypoint_id(self) -> Optional[int]:
        return self._current_waypoint_id

    def begin_decision_waypoint(
        self,
        observation: LaviraObservation,
        waypoint_id: int,
    ) -> int:
        """Append the current panorama anchor before the VLM decision."""
        if self._current_waypoint_id is not None:
            return self._current_waypoint_id
        node = self.waypoints.add(
            x=float(observation.hab_x),
            z=float(observation.hab_z),
            yaw=float(observation.front_yaw),
            action="",
            target="",
            progress="",
            waypoint_id=waypoint_id,
        )
        self._current_waypoint_id = node.id
        return node.id

    def consume_removed_waypoint_ids(self) -> list[int]:
        removed = self._removed_waypoint_ids
        self._removed_waypoint_ids = []
        return removed

    @staticmethod
    def _waypoint_distance(observation: LaviraObservation, node) -> float:
        return float(np.hypot(observation.hab_x - node.x, observation.hab_z - node.z))

    def blocked_directions(self, observation: LaviraObservation) -> set[str]:
        """Source evaluator's depth-based directional action filter."""
        return check_blocked_directions(observation.depth_by_direction)

    def preview_fmm_reachability(
        self,
        observation: LaviraObservation,
        goal_xz,
    ) -> dict:
        """Return a side-effect-free source-FMM connectivity audit."""
        if goal_xz is None or len(goal_xz) != 2:
            return {
                "reachable": False,
                "status": "missing_goal",
                "distance_at_agent": None,
            }
        return self.map.preview_fmm_reachability(
            float(observation.hab_x), float(observation.hab_z),
            float(goal_xz[0]), float(goal_xz[1]),
        )

    def select_grounding_candidate(
        self,
        observation: LaviraObservation,
        raw_action: str,
        candidates: list[dict],
        *,
        stair=False,
        target_region: str = "any",
        min_ray_ratio: float = 0.8,
    ) -> tuple[Optional[dict], list[dict]]:
        """Prefer strong geometry while preserving source depth-backoff.

        Map validity is used to select among boxes, not to reject the action.
        If no candidate is geometrically strong, the highest-confidence local
        DINO box is passed to the unchanged LHX projection path.
        """
        direction_index = observation.view_index(raw_action)
        directional_observation = LaviraObservation(
            rgb_by_direction=[observation.rgb_by_direction[direction_index]] * 4,
            depth_by_direction=np.repeat(
                observation.depth_by_direction[direction_index][None, ...], 4, axis=0
            ),
            yaw_by_direction=np.full(
                4, observation.yaw_by_direction[direction_index], dtype=np.float32
            ),
            hab_x=float(observation.hab_x),
            hab_z=float(observation.hab_z),
            hfov_deg=float(observation.hfov_deg),
        )
        evaluated: list[dict] = []
        for source_candidate in candidates or []:
            candidate = dict(source_candidate)
            candidate["depth_valid"] = False
            candidate["sample_depth_m"] = None
            candidate["goal_xz"] = None
            candidate["goal_map_rc"] = None
            candidate["endpoint_traversible"] = False
            candidate["ray_ratio"] = 0.0
            candidate["backoff_count"] = None
            candidate["selection_mode"] = ""

            if bool(candidate.get("scene_region", False)):
                candidate["geometry_status"] = "scene_region"
                evaluated.append(candidate)
                continue
            candidate["region_matches"] = bool(
                target_region == "any" or candidate.get("region") == target_region
            )
            bbox = candidate.get("bbox_2d")
            if bbox is None or len(bbox) != 4:
                candidate["geometry_status"] = "invalid_bbox"
                evaluated.append(candidate)
                continue
            sample = directional_observation.front_target_sample(
                None, bbox, stair=stair
            )
            depth_valid = bool(
                sample is not None
                and np.isfinite(float(sample[2]))
                and float(sample[2]) > 0.0
            )
            candidate["depth_valid"] = depth_valid
            candidate["sample_depth_m"] = float(sample[2]) if depth_valid else None
            if not depth_valid:
                candidate["geometry_status"] = "invalid_depth"
                evaluated.append(candidate)
                continue
            projection = self._evaluate_lhx_projection(
                directional_observation, None, bbox, stair
            )
            goal = projection.get("goal")
            if goal is None:
                candidate["geometry_status"] = "projection_failed"
                evaluated.append(candidate)
                continue
            map_target = self.map.world_to_map(*goal)
            row, col = int(map_target.row), int(map_target.col)
            endpoint_ok = bool(projection.get("endpoint_traversible", False))
            candidate["goal_xz"] = [float(goal[0]), float(goal[1])]
            candidate["goal_map_rc"] = [row, col]
            candidate["endpoint_traversible"] = endpoint_ok
            candidate["backoff_count"] = int(projection.get("backoff_count", 0))
            candidate["lhx_accept_reason"] = str(
                projection.get("accepted_reason", "")
            )
            ray_ratio = (
                self._ray_passability(directional_observation, goal)
                if endpoint_ok else 0.0
            )
            candidate["ray_ratio"] = float(ray_ratio)
            if endpoint_ok and ray_ratio >= float(min_ray_ratio):
                candidate["geometry_status"] = "valid"
            elif not endpoint_ok:
                candidate["geometry_status"] = "endpoint_blocked"
            else:
                candidate["geometry_status"] = "ray_blocked"
            evaluated.append(candidate)

        localized = [
            candidate for candidate in evaluated
            if candidate.get("geometry_status") not in {"scene_region", "invalid_bbox"}
        ]
        matching = [
            candidate for candidate in localized
            if candidate.get("region_matches", True)
        ]
        selection_pool = matching if matching else localized
        valid = [
            candidate for candidate in selection_pool
            if candidate["geometry_status"] == "valid"
        ]
        selected = max(
            valid if valid else selection_pool,
            key=lambda candidate: (
                float(candidate.get("confidence", 0.0)),
                -int(candidate.get("index", 0)),
            ),
        ) if selection_pool else None
        if selected is not None:
            selected["selection_mode"] = (
                "geometry_valid" if valid else "lhx_passthrough"
            )
            selected["geometry_selected"] = True
        return selected, evaluated

    def _evaluate_lhx_projection(
        self,
        observation: LaviraObservation,
        point_2d,
        bbox_2d,
        stair=False,
    ) -> dict:
        """Preview the exact source backoff acceptance without changing state."""
        depth = np.asarray(observation.front_depth, dtype=np.float32).copy()
        depth[~np.isfinite(depth)] = 0.0
        traversible = np.asarray(self.map.traversible()) > 0
        max_depth = float(depth.max()) if depth.size else 0.0
        max_backoffs = max(1, int(np.ceil(max_depth / 0.1)) + 1)
        for backoff_count in range(max_backoffs):
            current_max_depth = float(depth.max()) if depth.size else 0.0
            goal = observation.project_front_target(
                point_2d, bbox_2d, stair=stair, depth_override=depth
            )
            if goal is not None:
                map_target = self.map.world_to_map(*goal)
                row, col = int(map_target.row), int(map_target.col)
                endpoint_ok = bool(
                    0 <= row < traversible.shape[0]
                    and 0 <= col < traversible.shape[1]
                    and traversible[row, col]
                )
                depth_exhausted = current_max_depth < 0.1
                if endpoint_ok or depth_exhausted:
                    return {
                        "goal": goal,
                        "endpoint_traversible": endpoint_ok,
                        "backoff_count": backoff_count,
                        "accepted_reason": (
                            "source_depth_exhausted"
                            if depth_exhausted and not endpoint_ok
                            else (
                                "initial_traversible"
                                if backoff_count == 0 else "backoff_traversible"
                            )
                        ),
                    }
            depth = depth - 0.1
        return {
            "goal": None,
            "endpoint_traversible": False,
            "backoff_count": max_backoffs,
            "accepted_reason": "projection_failed",
        }

    def select_area_target(
        self,
        observation: LaviraObservation,
        raw_action: str,
        *,
        stair=False,
    ) -> tuple[Optional[dict], list[dict]]:
        """Select a medium-range free-space point for an area target."""
        direction_index = observation.view_index(raw_action)
        directional_observation = LaviraObservation(
            rgb_by_direction=[observation.rgb_by_direction[direction_index]] * 4,
            depth_by_direction=np.repeat(
                observation.depth_by_direction[direction_index][None, ...], 4, axis=0
            ),
            yaw_by_direction=np.full(
                4, observation.yaw_by_direction[direction_index], dtype=np.float32
            ),
            hab_x=float(observation.hab_x),
            hab_z=float(observation.hab_z),
            hfov_deg=float(observation.hfov_deg),
        )
        traversible = np.asarray(self.map.traversible()) > 0
        candidates: list[dict] = []
        index = 0
        for y in (600.0, 675.0, 750.0, 825.0):
            for x in (350.0, 425.0, 500.0, 575.0, 650.0):
                point = [x, y]
                sample = directional_observation.front_target_sample(
                    point, None, stair=stair
                )
                candidate = {
                    "index": index,
                    "candidate_type": "area",
                    "point_2d": point,
                    "geometry_status": "invalid_depth",
                    "sample_depth_m": None,
                    "goal_xz": None,
                    "goal_map_rc": None,
                    "goal_distance_m": None,
                    "endpoint_traversible": False,
                    "ray_ratio": 0.0,
                    "clearance_ratio": 0.0,
                    "depth_valid_ratio": 0.0,
                    "depth_mad_m": None,
                }
                index += 1
                if (
                    sample is None
                    or not np.isfinite(float(sample[2]))
                    or float(sample[2]) <= 0.0
                ):
                    candidates.append(candidate)
                    continue
                candidate["sample_depth_m"] = float(sample[2])
                sample_u, sample_v = int(sample[0]), int(sample[1])
                depth = directional_observation.front_depth
                h, w = depth.shape
                patch = depth[
                    max(0, sample_v - 3):min(h, sample_v + 4),
                    max(0, sample_u - 3):min(w, sample_u + 4),
                ]
                valid_depth = patch[
                    np.isfinite(patch) & (patch > 0.0) & (patch <= 5.0)
                ]
                candidate["depth_valid_ratio"] = float(
                    valid_depth.size / patch.size
                ) if patch.size else 0.0
                if valid_depth.size:
                    median = float(np.median(valid_depth))
                    candidate["depth_mad_m"] = float(
                        np.median(np.abs(valid_depth - median))
                    )
                goal = directional_observation.project_front_target(
                    point, None, stair=stair
                )
                if goal is None:
                    candidate["geometry_status"] = "projection_failed"
                    candidates.append(candidate)
                    continue
                distance_m = float(np.hypot(
                    float(goal[0]) - observation.hab_x,
                    float(goal[1]) - observation.hab_z,
                ))
                map_target = self.map.world_to_map(*goal)
                row, col = int(map_target.row), int(map_target.col)
                in_bounds = (
                    0 <= row < traversible.shape[0]
                    and 0 <= col < traversible.shape[1]
                )
                endpoint_ok = bool(in_bounds and traversible[row, col])
                candidate.update({
                    "goal_xz": [float(goal[0]), float(goal[1])],
                    "goal_map_rc": [row, col],
                    "goal_distance_m": distance_m,
                    "endpoint_traversible": endpoint_ok,
                })
                if not (
                    self.area_goal_min_distance_m
                    <= distance_m
                    <= self.area_goal_max_distance_m
                ):
                    candidate["geometry_status"] = "distance_out_of_range"
                    candidates.append(candidate)
                    continue
                if not endpoint_ok:
                    candidate["geometry_status"] = "endpoint_blocked"
                    candidates.append(candidate)
                    continue
                ray_ratio = self._ray_passability(directional_observation, goal)
                candidate["ray_ratio"] = float(ray_ratio)
                candidate["clearance_ratio"] = self._target_clearance_ratio(
                    traversible, row, col
                )
                candidate["geometry_status"] = (
                    "valid"
                    if ray_ratio >= self.area_goal_min_ray_ratio
                    else "ray_blocked"
                )
                candidates.append(candidate)

        valid = [
            candidate for candidate in candidates
            if candidate["geometry_status"] == "valid"
        ]
        selected = max(valid, key=lambda candidate: (
            float(candidate["ray_ratio"]),
            float(candidate["clearance_ratio"]),
            float(candidate["depth_valid_ratio"]),
            -float(candidate["depth_mad_m"] or 0.0),
            -abs(float(candidate["point_2d"][0]) - 500.0),
            -abs(
                float(candidate["goal_distance_m"])
                - self.area_goal_preferred_distance_m
            ),
            -int(candidate["index"]),
        )) if valid else None
        if selected is not None:
            selected["selection_mode"] = "area_geometry"
            selected["geometry_selected"] = True
        return selected, candidates

    def _target_clearance_ratio(
        self,
        traversible: np.ndarray,
        row: int,
        col: int,
    ) -> float:
        resolution_m = float(getattr(self.map, "resolution_cm", 5)) / 100.0
        radius = max(1, int(np.ceil(0.25 / max(resolution_m, 1e-3))))
        r1, r2 = max(0, row - radius), min(traversible.shape[0], row + radius + 1)
        c1, c2 = max(0, col - radius), min(traversible.shape[1], col + radius + 1)
        window = traversible[r1:r2, c1:c2]
        return float(window.mean()) if window.size else 0.0

    def accept_navigation(self, raw_action: str, point_2d, bbox_2d, target: str, progress: str,
                          observation: LaviraObservation, stair=False,
                          waypoint_id: Optional[int] = None,
                          canonical_mode: bool = False,
                          stop_after_reach: bool = False) -> bool:
        if raw_action not in {
            "navigate to forward", "navigate to left", "navigate to right", "navigate to behind",
        }:
            return False
        turns = {
            "navigate to forward": [],
            "navigate to left": [ACTION_TURN_LEFT] * 3,
            "navigate to right": [ACTION_TURN_RIGHT] * 3,
            "navigate to behind": [ACTION_TURN_LEFT] * 6,
        }
        # LaViRA-RFT projects only after these physical turns have completed.
        self.audit_decision_id += 1
        self.audit_request = {
            "decision_id": self.audit_decision_id,
            "raw_action": raw_action,
            "point_2d": list(point_2d) if point_2d is not None else None,
            "bbox_2d": list(bbox_2d) if bbox_2d is not None else None,
            "target": str(target),
            "progress": str(progress),
            "stair": bool(stair),
            "agent_xz": [float(observation.hab_x), float(observation.hab_z)],
            "agent_yaw": float(observation.front_yaw),
        }
        self.audit_projection = None
        self.pending_target = (
            point_2d, bbox_2d, raw_action, stair, str(target), bool(canonical_mode)
        )
        self._going_to_stop = bool(stop_after_reach)
        self._stop_check_after_scan = False
        self._stop_check_ready = False
        self.stair = bool(stair)
        if waypoint_id is not None:
            if self.waypoints.update(
                waypoint_id,
                action=raw_action,
                target=str(target),
                progress=str(progress),
            ) is None:
                self.waypoints.add(
                    x=float(observation.hab_x), z=float(observation.hab_z),
                    yaw=float(observation.front_yaw), action=raw_action,
                    target=str(target), progress=str(progress), waypoint_id=waypoint_id,
                )
            self._current_waypoint_id = waypoint_id
        self.turn_queue = list(turns[raw_action])
        self.goal_xz, self.goal_direction = None, raw_action
        self.goal_just_set = True
        self.steps_to_goal, self.state = 0, NavigationState.TURNING
        return True

    def accept_backtrack(self, waypoint_id: Optional[int], observation: LaviraObservation) -> bool:
        waypoint = self.waypoints.get(waypoint_id)
        if waypoint is None:
            return False
        self.goal_xz = (waypoint.x, waypoint.z)
        self.goal_just_set = True
        self.goal_direction = f"backtrack to {waypoint_id}"
        self.pending_target = None
        self._going_to_stop = False
        self._stop_check_after_scan = False
        self._stop_check_ready = False
        self.turn_queue = []
        self.steps_to_goal = 0
        self.waypoints.mark_failed_branch(waypoint.id)
        self._backtrack_replan_waypoint_id = waypoint.id
        self._backtrack_replan_ready = False
        self.state = NavigationState.BACKTRACKING
        return True

    def schedule_backtrack_replan(self, waypoint_id: Optional[int]) -> bool:
        """Queue LHX's immediate second LLM decision without simulator motion."""
        waypoint = self.waypoints.get(waypoint_id)
        if waypoint is None:
            return False
        self.waypoints.mark_failed_branch(waypoint.id)
        self._backtrack_replan_waypoint_id = waypoint.id
        self._backtrack_replan_ready = True
        self.goal_xz = None
        self.goal_just_set = False
        self.pending_target = None
        self._going_to_stop = False
        self._stop_check_after_scan = False
        self._stop_check_ready = False
        self.turn_queue = []
        self.steps_to_goal = 0
        self.state = NavigationState.NEED_DECISION
        return True

    def accept_backtrack_replan(
        self,
        waypoint_id: Optional[int],
        anchor_observation: LaviraObservation,
        raw_action: str,
        point_2d,
        bbox_2d,
        target: str,
        progress: str,
        stair=False,
        replan_waypoint_id: Optional[int] = None,
        canonical_mode: bool = False,
    ) -> bool:
        """Port LHX's immediate second-chance backtrack path.

        The source evaluator does not first replay primitives back to the old
        waypoint.  It marks the intervening branch failed, asks the VLM to
        replan from that waypoint's stored panorama, projects the new target
        in that waypoint's pose, then FMM-navigates to the projected target.
        """
        waypoint = self.waypoints.get(waypoint_id)
        if waypoint is None:
            return False
        if raw_action not in {
            "navigate to forward", "navigate to left", "navigate to right", "navigate to behind",
        }:
            return False

        direction_index = anchor_observation.view_index(raw_action)
        directional_observation = LaviraObservation(
            rgb_by_direction=[anchor_observation.rgb_by_direction[direction_index]] * 4,
            depth_by_direction=np.repeat(
                anchor_observation.depth_by_direction[direction_index][None, ...], 4, axis=0
            ),
            yaw_by_direction=np.full(
                4, anchor_observation.yaw_by_direction[direction_index], dtype=np.float32
            ),
            hab_x=float(anchor_observation.hab_x),
            hab_z=float(anchor_observation.hab_z),
            hfov_deg=float(anchor_observation.hfov_deg),
        )
        goal = self._project_target_with_portal_candidates(
            directional_observation, point_2d, bbox_2d, stair, target,
            canonical_mode=canonical_mode,
        )
        if goal is None:
            return False

        self.waypoints.mark_failed_branch(waypoint.id)
        if replan_waypoint_id is not None:
            self.waypoints.add(
                x=float(waypoint.x), z=float(waypoint.z), yaw=float(waypoint.yaw),
                action=raw_action, target=str(target), progress=str(progress),
                waypoint_id=replan_waypoint_id,
            )
            self._current_waypoint_id = replan_waypoint_id
        self.audit_decision_id += 1
        self.audit_request = {
            "decision_id": self.audit_decision_id,
            "raw_action": raw_action,
            "point_2d": list(point_2d) if point_2d is not None else None,
            "bbox_2d": list(bbox_2d) if bbox_2d is not None else None,
            "target": str(target),
            "progress": str(progress),
            "stair": bool(stair),
            "agent_xz": [float(anchor_observation.hab_x), float(anchor_observation.hab_z)],
            "agent_yaw": float(anchor_observation.front_yaw),
            "source_backtrack_waypoint_id": int(waypoint.id),
        }
        self.audit_projection = {"status": "source_backtrack_replan", "goal_xz": list(goal)}
        self.stair = bool(stair)
        self.goal_xz = goal
        self.goal_just_set = True
        self.goal_direction = f"replan from {waypoint.id}: {raw_action}"
        self.pending_target = None
        self.turn_queue = []
        self.steps_to_goal = 0
        self.state = NavigationState.NAVIGATING
        return True

    def consume_backtrack_replan_waypoint_id(self) -> Optional[int]:
        """Return the reached backtrack point exactly once for source replan."""
        if not self._backtrack_replan_ready:
            return None
        self._backtrack_replan_ready = False
        return self._backtrack_replan_waypoint_id

    def consume_navigation_feedback(self) -> Optional[str]:
        feedback = self._navigation_feedback
        self._navigation_feedback = None
        return feedback

    def consume_stop_check_ready(self) -> bool:
        ready = self._stop_check_ready
        self._stop_check_ready = False
        return ready

    def resume_after_stop_rejection(self) -> None:
        self._going_to_stop = False
        self._stop_check_after_scan = False
        self._stop_check_ready = False
        self.state = NavigationState.SCANNING
        self.scan_turns_done = 0
        self.goal_xz = None
        self.goal_just_set = False
        self.map.set_last_action(None)

    def next_primitive(self, observation: LaviraObservation) -> Optional[int]:
        if self.state == NavigationState.SCANNING:
            if self.scan_turns_done < self.initial_scan_turns:
                # The environment-side panorama primitive performs the source
                # evaluator's 12 real turns and returns all scan frames in one
                # observation.  Sending ordinary TURN_LEFT here would replay
                # the turns one by one and rerun GroundedSAM/map fusion on each
                # intermediate frame before the first VLM decision.
                self.scan_turns_done = self.initial_scan_turns
                self._record_primitive(ACTION_PANORAMA_SCAN)
                return ACTION_PANORAMA_SCAN
            self.state = NavigationState.NEED_DECISION
            if self._stop_check_after_scan:
                self._stop_check_after_scan = False
                self._stop_check_ready = True
            self.map.set_last_action(None)
            return None

        if self.state == NavigationState.TURNING:
            if self.turn_queue:
                action = self.turn_queue.pop(0)
                self.map.set_last_action(action)
                return action
            point, bbox, _raw, stair, target, canonical_mode = self.pending_target
            self.pending_target = None
            goal = self._project_target_with_portal_candidates(
                observation, point, bbox, stair, target,
                canonical_mode=canonical_mode,
            )
            if goal is None:
                if not isinstance(self.audit_projection, dict):
                    self.audit_projection = {"status": "projection_failed"}
                self.goal_just_set = False
                self.state = NavigationState.NEED_DECISION
                self._navigation_feedback = (
                    "[Navigation feedback] The previous target could not be "
                    "projected to a traversible map cell. Choose a different "
                    "visible target or direction."
                )
                self.map.set_last_action(None)
                return None
            self.goal_xz = goal
            map_goal = self.map.world_to_map(*goal)
            traversible = self.map.traversible()
            projection_audit = dict(self.audit_projection or {})
            projection_audit.update({
                "status": "projected",
                "goal_xz": [float(goal[0]), float(goal[1])],
                "goal_map_rc": [int(map_goal.row), int(map_goal.col)],
                "goal_traversible": bool(traversible[map_goal.row, map_goal.col]),
                "goal_explored": bool(self.map.explored[map_goal.row, map_goal.col]),
                "goal_obstacle": float(self.map.obstacle[map_goal.row, map_goal.col]),
                "goal_collision": bool(self.map.collision[map_goal.row, map_goal.col]),
            })
            self.audit_projection = projection_audit
            self.state = NavigationState.NAVIGATING

        if self.state not in (NavigationState.NAVIGATING, NavigationState.BACKTRACKING) or self.goal_xz is None:
            return None
        agent_cell = self.map.world_to_map(observation.hab_x, observation.hab_z)
        target_cell = self.map.world_to_map(*self.goal_xz)
        distance_cells = float(np.hypot(
            target_cell.row - agent_cell.row,
            target_cell.col - agent_cell.col,
        ))
        reached_threshold_cells = (
            self.target_reached_threshold_m * 100.0 / self.map_resolution_cm
        )
        # ZS_Evaluator_mp compares integer map cells before invoking FMM:
        # distance_to_target < 75cm / map_resolution.
        if self.steps_to_goal >= self.max_steps_to_target:
            # Source evaluator's effective local variable is 15 primitive
            # steps, removes the unfinished waypoint, and returns to high-level
            # waypoint selection.
            removed = self.waypoints.pop_latest()
            if removed is not None:
                self._removed_waypoint_ids.append(removed.id)
            self._goal_reached()
            return self.next_primitive(observation)
        if not self.goal_just_set and distance_cells < reached_threshold_cells:
            self._goal_reached()
            return self.next_primitive(observation)
        action = self.map.fmm_action(
            observation.hab_x,
            observation.hab_z,
            observation.front_yaw,
            self.goal_xz[0],
            self.goal_xz[1],
            self.fmm_goal_threshold_m,
        )
        if action is None:
            # The source FMM always returns an integer primitive. Keep the
            # adapter failure path separate from the outer 0.75m arrival test.
            was_backtracking = self.state == NavigationState.BACKTRACKING
            self.state = NavigationState.SCANNING
            self.scan_turns_done = 0
            self.goal_xz = None
            self.goal_just_set = False
            self.map.set_last_action(None)
            if was_backtracking and self._backtrack_replan_waypoint_id is not None:
                self._backtrack_replan_ready = True
            return self.next_primitive(observation)
        self.steps_to_goal += 1
        self.goal_just_set = False
        self._record_primitive(action)
        return action

    def audit_snapshot(self, observation: LaviraObservation, primitive: Optional[int]) -> dict:
        """Return debug-only state for one policy tick without changing control."""
        snapshot = {
            "decision_id": self.audit_decision_id,
            "controller_state": self.state.value,
            "steps_to_goal": self.steps_to_goal,
            "goal_just_set": bool(self.goal_just_set),
            "primitive_action": None if primitive is None else int(primitive),
            "agent_xz": [float(observation.hab_x), float(observation.hab_z)],
            "agent_yaw": float(observation.front_yaw),
            "last_forward_collision": bool(self.map.last_forward_collision),
            "collision": self.map.last_collision_audit,
            "fmm": self.map.last_fmm_audit,
            "request": self.audit_request,
            "projection": self.audit_projection,
        }
        if self.goal_xz is not None:
            snapshot["active_goal_xz"] = [float(self.goal_xz[0]), float(self.goal_xz[1])]
            snapshot["goal_distance_m"] = float(np.hypot(
                observation.hab_x - self.goal_xz[0], observation.hab_z - self.goal_xz[1]
            ))
        return snapshot

    def _project_traversible_target(
        self,
        observation: LaviraObservation,
        point_2d,
        bbox_2d,
        stair,
    ) -> Optional[tuple[float, float]]:
        """Back off depth until projection reaches a traversible map cell."""
        depth = np.asarray(observation.front_depth, dtype=np.float32).copy()
        finite = depth[np.isfinite(depth)]
        initial_stats = {
            "min": float(finite.min()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
        }
        depth[~np.isfinite(depth)] = 0.0
        traversible = self.map.traversible()
        max_depth = float(depth.max()) if depth.size else 0.0
        max_backoffs = max(1, int(np.ceil(max_depth / 0.1)) + 1)
        attempts = []
        for backoff_count in range(max_backoffs):
            current_max_depth = float(depth.max()) if depth.size else 0.0
            sample = observation.front_target_sample(
                point_2d, bbox_2d, stair=stair, depth_override=depth
            )
            goal = observation.project_front_target(
                point_2d,
                bbox_2d,
                stair=stair,
                depth_override=depth,
            )
            attempt = {
                "backoff_count": backoff_count,
                "sample_pixel_uv": (
                    [int(sample[0]), int(sample[1])] if sample is not None else None
                ),
                "sample_depth_m": float(sample[2]) if sample is not None else None,
                "goal_xz": None,
                "goal_map_rc": None,
                "goal_traversible": False,
            }
            if goal is not None:
                target = self.map.world_to_map(*goal)
                attempt.update({
                    "goal_xz": [float(goal[0]), float(goal[1])],
                    "goal_map_rc": [int(target.row), int(target.col)],
                    "goal_traversible": bool(traversible[target.row, target.col]),
                })
            attempts.append(attempt)
            if goal is not None:
                goal_is_traversible = traversible[target.row, target.col] != 0
                depth_is_exhausted = current_max_depth < 0.1
                # Exact ZS_Evaluator_mp termination condition:
                #
                #   traversable[target] == 1 or depth_image.max() < 0.1
                #
                # The source accepts the final projected target even when its
                # map cell is blocked. Policy._get_action then snaps it to the
                # nearest traversible cell before invoking FMM.
                if goal_is_traversible or depth_is_exhausted:
                    self.audit_projection = {
                        "status": "projected",
                        "accepted_reason": (
                            "source_depth_exhausted"
                            if depth_is_exhausted and not goal_is_traversible
                            else (
                                "initial_traversible"
                                if backoff_count == 0 else "backoff_traversible"
                            )
                        ),
                        "backoff_count": backoff_count,
                        "initial_depth_m": initial_stats,
                        "attempts": attempts,
                    }
                    return goal
            # Keep the source subtraction semantics. Clamping to zero changes
            # which positive sub-0.1 sample is used on the final iteration.
            depth = depth - 0.1
        self.audit_projection = {
            "status": "projection_failed",
            "accepted_reason": "depth_exhausted",
            "backoff_count": max_backoffs,
            "initial_depth_m": initial_stats,
            "attempts": attempts,
            "final_failed_goal_map_rc": (
                attempts[-1]["goal_map_rc"] if attempts else None
            ),
        }
        return None

    @staticmethod
    def _portal_candidate_points(bbox_2d) -> list[list[float]]:
        if bbox_2d is None or len(bbox_2d) != 4:
            return []
        x1, y1, x2, y2 = [float(v) for v in bbox_2d]
        return [
            [x1 + fx * (x2 - x1), y1 + fy * (y2 - y1)]
            for fy in (0.45, 0.70, 0.90)
            for fx in (0.05, 0.50, 0.95)
        ]

    def _ray_passability(self, observation: LaviraObservation, goal) -> float:
        traversible = np.asarray(self.map.traversible()) > 0
        to_float = getattr(self.map, "_world_to_map_float", None)
        if to_float is None:
            agent = self.map.world_to_map(observation.hab_x, observation.hab_z)
            agent_rc = np.asarray([agent.row, agent.col], dtype=np.float32)
        else:
            agent_rc = np.asarray(
                to_float(observation.hab_x, observation.hab_z), dtype=np.float32
            )
        target = self.map.world_to_map(*goal)
        dr = float(target.row) - float(agent_rc[0])
        dc = float(target.col) - float(agent_rc[1])
        count = max(2, int(np.ceil(max(abs(dr), abs(dc)))) + 1)
        rows = np.rint(np.linspace(float(agent_rc[0]), float(target.row), count)).astype(int)
        cols = np.rint(np.linspace(float(agent_rc[1]), float(target.col), count)).astype(int)
        valid = (
            (rows >= 0) & (rows < traversible.shape[0]) &
            (cols >= 0) & (cols < traversible.shape[1])
        )
        if not np.any(valid):
            return 0.0
        resolution_m = float(getattr(self.map, "resolution_cm", 5)) / 100.0
        skip = min(int(round(0.25 / max(resolution_m, 1e-3))), count - 1)
        values = traversible[rows[skip:][valid[skip:]], cols[skip:][valid[skip:]]]
        return float(values.mean()) if values.size else 0.0

    @staticmethod
    def _portal_candidate_rank(candidate: dict) -> tuple[float, int, float, int]:
        return (
            float(candidate["ray_ratio"]),
            -int(candidate["backoff_count"]),
            -float(candidate["center_offset"]),
            -int(candidate["index"]),
        )

    def _project_target_with_portal_candidates(
        self, observation: LaviraObservation, point_2d, bbox_2d, stair, target: str,
        canonical_mode: bool = False,
    ) -> Optional[tuple[float, float]]:
        if (not canonical_mode or str(target).strip().lower() not in PORTAL_CLASSES
                or bbox_2d is None):
            return self._project_traversible_target(observation, point_2d, bbox_2d, stair)

        candidates = []
        for index, candidate_point in enumerate(self._portal_candidate_points(bbox_2d)):
            goal = self._project_traversible_target(
                observation, candidate_point, None, stair
            )
            projection = dict(self.audit_projection or {})
            if goal is None:
                candidates.append({
                    "index": index, "point_2d": candidate_point,
                    "status": "projection_failed", "ray_ratio": 0.0,
                })
                continue
            map_goal = self.map.world_to_map(*goal)
            endpoint_ok = bool(self.map.traversible()[map_goal.row, map_goal.col])
            ray_ratio = self._ray_passability(observation, goal) if endpoint_ok else 0.0
            candidates.append({
                "index": index,
                "point_2d": candidate_point,
                "goal_xz": [float(goal[0]), float(goal[1])],
                "goal_map_rc": [int(map_goal.row), int(map_goal.col)],
                "endpoint_traversible": endpoint_ok,
                "ray_ratio": ray_ratio,
                "backoff_count": int(projection.get("backoff_count", 10**6)),
                "center_offset": float(
                    ((index % 3) / 2.0 - 0.5) ** 2
                    + ((index // 3) / 2.0 - 0.5) ** 2
                ),
                "status": "valid" if endpoint_ok and ray_ratio >= 0.8 else "rejected",
            })

        valid = [candidate for candidate in candidates if candidate["status"] == "valid"]
        if valid:
            chosen = max(valid, key=self._portal_candidate_rank)
            self.audit_projection = {
                "status": "portal_candidate",
                "selected_index": int(chosen["index"]),
                "selected_point_2d": chosen["point_2d"],
                "candidates": candidates,
            }
            return tuple(chosen["goal_xz"])

        fallback = self._project_traversible_target(
            observation, point_2d, bbox_2d, stair
        )
        projection = dict(self.audit_projection or {})
        projection.update({"portal_candidates": candidates, "portal_fallback": True})
        self.audit_projection = projection
        return fallback

    def _record_primitive(self, action: int) -> None:
        self.map.set_last_action(action)

    def _goal_reached(self) -> None:
        was_backtracking = self.state == NavigationState.BACKTRACKING
        verify_stop_after_scan = self._going_to_stop
        self._going_to_stop = False
        if self._current_waypoint_id is not None:
            node = self.waypoints.get(self._current_waypoint_id)
            if node is not None:
                node.reached = True
        self._current_waypoint_id = None
        self.goal_just_set = False
        # The source evaluator acquires a fresh physical panorama before each
        # new high-level waypoint decision.
        self.state = NavigationState.SCANNING
        self.scan_turns_done = 0
        self._stop_check_after_scan = verify_stop_after_scan
        self._stop_check_ready = False
        self.goal_xz = None
        self.map.set_last_action(None)
        if was_backtracking and self._backtrack_replan_waypoint_id is not None:
            self._backtrack_replan_ready = True

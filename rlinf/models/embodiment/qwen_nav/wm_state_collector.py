"""Default-off collection of real policy-induced NavigationWM V6 states."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from tools.lavira_world_model_data.schema_v6 import (
    STATE_SCHEMA_VERSION,
    atomic_write_json,
    stable_state_id,
    validate_panorama,
    validate_state_record,
)

from .lavira_runtime.wm_snapshot import save_runtime_snapshot


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "value"):
        return _json_value(value.value)
    return str(value)


def parsed_action_payload(parsed: Any, raw_response: str) -> dict[str, Any]:
    return {
        "raw_response": str(raw_response),
        "action_type": str(getattr(parsed, "action_type", "")),
        "direction": str(getattr(parsed, "raw_dir", "") or ""),
        "target": str(getattr(parsed, "target", "") or ""),
        "raw_target": str(getattr(parsed, "raw_target", "") or ""),
        "progress_analysis": str(getattr(parsed, "progress", "") or ""),
        "reasoning_plan_action": str(getattr(parsed, "reasoning_plan_action", "") or ""),
        "planning": str(getattr(parsed, "planning", "") or ""),
        "action": str(getattr(parsed, "raw_dir", "") or getattr(parsed, "action_type", "") or ""),
        "bbox_2d": _json_value(getattr(parsed, "bbox_2d", None)),
        "point_2d": _json_value(getattr(parsed, "point_2d", None)),
        "waypoint_id": _json_value(getattr(parsed, "waypoint_id", None)),
        "stop": bool(getattr(parsed, "stop", False)),
        "stair": _json_value(getattr(parsed, "stair", False)),
        "parse_ok": bool(getattr(parsed, "ok", False)),
        "parse_error": _json_value(getattr(parsed, "err", None)),
    }


def controller_payload(controller: Any, observation: Any) -> dict[str, Any]:
    return {
        "state": str(controller.state.value),
        "steps_to_goal": int(controller.steps_to_goal),
        "goal_just_set": bool(controller.goal_just_set),
        "goal_xz": _json_value(controller.goal_xz),
        "goal_direction": str(controller.goal_direction),
        "current_waypoint_id": _json_value(controller.current_waypoint_id),
        "available_backtrack_ids": controller.available_waypoint_ids(observation),
        "blocked_directions": sorted(controller.blocked_directions(observation)),
        "failed_waypoints": [
            int(node.id) for node in controller.waypoints.nodes
            if node.failed or node.failed_dir
        ],
        "last_forward_collision": bool(controller.map.last_forward_collision),
        "map_step": int(controller.map.step),
    }


def history_payload(cache: Any, limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    for node in cache.waypoints[-max(0, int(limit)):]:
        rows.append({
            "waypoint_id": int(node.id),
            "step": int(node.step),
            "action": str(node.action),
            "target": str(node.target),
            "progress": str(node.progress_analysis),
            "pose_hab": [float(node.pose_hab[0]), float(node.pose_hab[1])],
            "failed": bool(node.failed),
            "failed_direction": bool(node.failed_dir),
        })
    return rows


def backtrack_anchor_payload(
    cache: Any, available_waypoint_ids: list[int],
) -> dict[str, np.ndarray] | None:
    """Serialize the exact RGB-D observations used by source backtrack replan."""
    records = {int(node.id): node for node in cache.waypoints}
    waypoint_ids: list[int] = []
    rgb_rows: list[np.ndarray] = []
    depth_rows: list[np.ndarray] = []
    pose_rows: list[np.ndarray] = []
    for waypoint_id in map(int, available_waypoint_ids):
        node = records.get(waypoint_id)
        if node is None or len(node.anchor_views) != 4:
            raise ValueError(
                f"backtrack waypoint {waypoint_id} is missing four anchor RGB views"
            )
        depth = np.asarray(node.anchor_depth_by_direction, dtype=np.float32)
        if depth.ndim != 3 or depth.shape[0] != 4:
            raise ValueError(
                f"backtrack waypoint {waypoint_id} has invalid anchor depth {depth.shape}"
            )
        rgb = np.stack([
            np.asarray(image, dtype=np.uint8) for image in node.anchor_views
        ])
        if rgb.shape[:3] != depth.shape:
            raise ValueError(
                f"backtrack waypoint {waypoint_id} RGB/depth shape mismatch: "
                f"{rgb.shape} vs {depth.shape}"
            )
        waypoint_ids.append(waypoint_id)
        rgb_rows.append(rgb)
        depth_rows.append(depth)
        pose_rows.append(np.asarray(node.pose_hab, dtype=np.float32))
    if not waypoint_ids:
        return None
    return {
        "waypoint_ids": np.asarray(waypoint_ids, dtype=np.int64),
        "rgb": np.stack(rgb_rows).astype(np.uint8, copy=False),
        "depth_m": np.stack(depth_rows).astype(np.float32, copy=False),
        "pose_hab": np.stack(pose_rows).astype(np.float32, copy=False),
    }


class OnPolicyStateCollector:
    """Write immutable decision-state directories without changing execution."""

    def __init__(self, cfg: Any) -> None:
        self.enabled = bool(getattr(cfg, "enabled", False))
        raw_output_dir = str(getattr(cfg, "output_dir", "")).strip()
        self.output_dir = Path(raw_output_dir) if raw_output_dir else Path()
        self.save_source_snapshot = bool(
            getattr(cfg, "save_source_snapshot", True)
        )
        # Policy-cache snapshots are opt-in because they can contain a much
        # larger Python history than the source mapper snapshot.  P0 oracle
        # collection enables this explicitly; normal V6 collection remains
        # byte-compatible when the field is absent.
        self.save_policy_snapshot = bool(
            getattr(cfg, "save_policy_snapshot", False)
        )
        self.max_decisions_per_episode = int(
            getattr(cfg, "max_decisions_per_episode", 12)
        )
        self.history_length = int(getattr(cfg, "history_length", 8))
        self.run_id = str(getattr(cfg, "run_id", "")).strip()
        self.contract_id = str(
            getattr(cfg, "contract_id", "") or self.run_id
        ).strip()
        self.policy_do_sample = bool(getattr(cfg, "policy_do_sample", False))
        self.policy_sampling_temperature = float(
            getattr(cfg, "policy_sampling_temperature", 0.0)
        )
        self.policy_sampling_seed = int(
            getattr(cfg, "policy_sampling_seed", 42)
        )
        self.policy_history_max_frames = int(
            getattr(cfg, "policy_history_max_frames", 1)
        )
        self.guidance_policy_id = str(
            getattr(cfg, "guidance_policy_id", "")
        ).strip()
        self._counts: dict[tuple[str, str, str], int] = {}
        if self.enabled and not raw_output_dir:
            raise ValueError("NavigationWM data collection requires output_dir")

    def _qualified_trial_id(self, trial_id: str) -> str:
        raw = str(trial_id)
        return f"{raw}@{self.run_id}" if self.run_id else raw

    def reset_trial(self, scene_id: str, episode_id: str, trial_id: str) -> None:
        self._counts[
            (str(scene_id), str(episode_id), self._qualified_trial_id(trial_id))
        ] = 0

    def next_decision_index(
        self, scene_id: str, episode_id: str, trial_id: str
    ) -> int:
        return self._counts.get(
            (
                str(scene_id), str(episode_id),
                self._qualified_trial_id(trial_id),
            ), 0
        )

    def collect(
        self,
        *,
        episode_id: str,
        trial_id: str,
        scene_id: str,
        instruction: str,
        decision_index: int,
        sim_step: int,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        yaw_rad: np.ndarray,
        pose_hab: np.ndarray,
        position_genesis: np.ndarray,
        parsed: Any,
        raw_response: str,
        cache: Any,
        controller: Any,
        observation: Any,
        policy_snapshot: dict[str, Any] | None = None,
        guidance_mode: str = "off",
        guidance_interventions_before_state: int = 0,
    ) -> Path | None:
        if not self.enabled:
            return None
        qualified_trial_id = self._qualified_trial_id(trial_id)
        trial_key = (str(scene_id), str(episode_id), qualified_trial_id)
        count = self._counts.get(trial_key, 0)
        if self.max_decisions_per_episode > 0 and count >= self.max_decisions_per_episode:
            return None
        state_id = stable_state_id(
            str(scene_id), str(episode_id), qualified_trial_id,
            int(decision_index), int(sim_step)
        )
        final_dir = self.output_dir / f"state_{state_id}"
        if (final_dir / "_SUCCESS").exists():
            self._counts[trial_key] = count + 1
            return final_dir
        temporary = final_dir.with_name(f".{final_dir.name}.tmp.{os.getpid()}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            panorama_path = temporary / "panorama.npz"
            np.savez_compressed(
                panorama_path,
                rgb=np.asarray(rgb, dtype=np.uint8),
                depth_m=np.asarray(depth_m, dtype=np.float32),
                yaw_rad=np.asarray(yaw_rad, dtype=np.float32),
                pose_hab=np.asarray(pose_hab, dtype=np.float32),
                position_genesis=np.asarray(position_genesis, dtype=np.float32),
            )
            available_backtrack_ids = controller.available_waypoint_ids(observation)
            backtrack_anchors = backtrack_anchor_payload(
                cache, available_backtrack_ids
            )
            backtrack_anchor_name = None
            if backtrack_anchors is not None:
                backtrack_anchor_name = "backtrack_anchors.npz"
                np.savez_compressed(
                    temporary / backtrack_anchor_name, **backtrack_anchors
                )
            snapshot_name = None
            if self.save_source_snapshot:
                snapshot_name = "source_snapshot.pt.gz"
                save_runtime_snapshot(
                    temporary / snapshot_name, controller.export_wm_snapshot()
                )
            policy_snapshot_name = None
            if self.save_policy_snapshot:
                if policy_snapshot is None:
                    raise ValueError(
                        "save_policy_snapshot=true requires policy_snapshot"
                    )
                policy_snapshot_name = "policy_snapshot.pt.gz"
                from rlinf.research.cf_foresight.policy_snapshot import (
                    save_policy_snapshot,
                )
                save_policy_snapshot(
                    temporary / policy_snapshot_name, policy_snapshot
                )
            interventions_before_state = int(
                guidance_interventions_before_state
            )
            guided_history = interventions_before_state > 0
            if guided_history and not self.guidance_policy_id:
                raise ValueError(
                    "guided V6 state requires immutable guidance_policy_id"
                )
            record = {
                "schema_version": STATE_SCHEMA_VERSION,
                "state_id": state_id,
                "episode_id": str(episode_id),
                "trial_id": qualified_trial_id,
                "scene_id": str(scene_id),
                "instruction": str(instruction),
                "decision_index": int(decision_index),
                "sim_step": int(sim_step),
                "pose_hab": np.asarray(pose_hab, dtype=np.float32).tolist(),
                "position_genesis": np.asarray(
                    position_genesis, dtype=np.float32
                ).tolist(),
                "qwen": parsed_action_payload(parsed, raw_response),
                "history": history_payload(cache, self.history_length),
                "controller": controller_payload(controller, observation),
                "source_snapshot_saved": snapshot_name is not None,
                "policy_snapshot_saved": policy_snapshot_name is not None,
                "assets": {
                    "panorama_npz": "panorama.npz",
                    "source_snapshot": snapshot_name,
                    "policy_snapshot": policy_snapshot_name,
                    "backtrack_anchors_npz": backtrack_anchor_name,
                },
                "collection": {
                    "state_source": (
                        "on_policy_guided" if guided_history
                        else "on_policy_qwen"
                    ),
                    "policy_execution_mutated": guided_history,
                    "raw_trial_id": str(trial_id),
                    "run_id": self.run_id,
                    "contract_id": self.contract_id,
                    "policy_do_sample": self.policy_do_sample,
                    "policy_sampling_temperature": self.policy_sampling_temperature,
                    "policy_sampling_seed": self.policy_sampling_seed,
                    "policy_history_max_frames": self.policy_history_max_frames,
                    "guidance_mode": str(guidance_mode),
                    "guidance_policy_id": self.guidance_policy_id,
                    "guidance_interventions_before_state": (
                        interventions_before_state
                    ),
                },
            }
            validate_state_record(record)
            validate_panorama(panorama_path)
            atomic_write_json(temporary / "state.json", record)
            (temporary / "_SUCCESS").write_text("ok\n")
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            temporary.replace(final_dir)
            self._counts[trial_key] = count + 1
            return final_dir
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

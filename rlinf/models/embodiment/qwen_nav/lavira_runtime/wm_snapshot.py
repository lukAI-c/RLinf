"""Serializable snapshots for counterfactual LaViRA branch evaluation.

The snapshot adapter deliberately lives outside the vendored LHX sources. It
copies mutable runtime state without changing mapping, projection, or FMM
semantics. Static model weights and configuration are never serialized.
"""

from __future__ import annotations

import copy
import gzip
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .waypoint_memory import WaypointMemory


SNAPSHOT_SCHEMA_VERSION = "lavira-runtime-wm-snapshot-v1"

_MAPPER_TENSORS = (
    "feat",
    "init_grid",
    "full_map",
    "one_step_full_map",
    "local_map",
    "one_step_local_map",
    "full_pose",
    "local_pose",
)
_MAPPER_ARRAYS = (
    "origins", "lmb", "state", "visited_vis", "last_loc", "curr_loc"
)
_SOURCE_ARRAYS = (
    "collision",
    "_traversible",
    "_traversible_before_collision",
    "floor",
    "frontiers",
    "last_source_pose",
    "last_full_pose",
    "last_depth_m",
    "last_external_pose",
    "full_map",
    "full_pose",
)
_SOURCE_SCALARS = (
    "origin_x",
    "origin_z",
    "origin_yaw",
    "last_pose",
    "last_action",
    "last_forward_collision",
    "last_collision_audit",
    "last_fmm_audit",
    "step",
    "fmm_episode_label",
)
_CONTROLLER_FIELDS = (
    "state",
    "scan_turns_done",
    "goal_xz",
    "goal_just_set",
    "goal_direction",
    "turn_queue",
    "pending_target",
    "steps_to_goal",
    "stair",
    "_backtrack_replan_waypoint_id",
    "_backtrack_replan_ready",
    "_current_waypoint_id",
    "_removed_waypoint_ids",
    "_going_to_stop",
    "_stop_check_after_scan",
    "_stop_check_ready",
    "audit_decision_id",
    "audit_request",
    "audit_projection",
    "_navigation_feedback",
)


def _clone_array(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _copy_waypoints(memory: WaypointMemory) -> dict[str, Any]:
    return {
        "next_id": int(memory._next_id),
        "nodes": [
            {
                "id": int(node.id),
                "x": float(node.x),
                "z": float(node.z),
                "yaw": float(node.yaw),
                "action": str(node.action),
                "target": str(node.target),
                "progress": str(node.progress),
                "parent_id": node.parent_id,
                "failed_directions": sorted(node.failed_directions),
                "failed": bool(node.failed),
                "failed_dir": bool(node.failed_dir),
                "reached": bool(node.reached),
            }
            for node in memory.nodes
        ],
    }


def _restore_waypoints(payload: dict[str, Any]) -> WaypointMemory:
    memory = WaypointMemory()
    for row in payload.get("nodes", []):
        node = memory.add(
            x=float(row["x"]),
            z=float(row["z"]),
            yaw=float(row["yaw"]),
            action=str(row["action"]),
            target=str(row["target"]),
            progress=str(row["progress"]),
            parent_id=row.get("parent_id"),
            waypoint_id=int(row["id"]),
        )
        node.failed_directions = set(row.get("failed_directions", []))
        node.failed = bool(row.get("failed", False))
        node.failed_dir = bool(row.get("failed_dir", False))
        node.reached = bool(row.get("reached", False))
    memory._next_id = int(payload.get("next_id", memory._next_id))
    return memory


def export_runtime_snapshot(controller: Any) -> dict[str, Any]:
    """Export mutable controller and source-map state to CPU-owned values."""
    source_map = controller.map
    if not hasattr(source_map, "mapper") or not hasattr(source_map, "policy"):
        raise TypeError("world-model snapshots require map_backend=source")
    mapper = source_map.mapper
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "map_backend": str(controller.map_backend),
        "controller": {
            field: copy.deepcopy(getattr(controller, field))
            for field in _CONTROLLER_FIELDS
        },
        "waypoints": _copy_waypoints(controller.waypoints),
        "source": {
            "scalars": {
                field: copy.deepcopy(getattr(source_map, field, None))
                for field in _SOURCE_SCALARS
            },
            "arrays": {
                field: _clone_array(getattr(source_map, field, None))
                for field in _SOURCE_ARRAYS
            },
            "agent_path_xz": copy.deepcopy(source_map.agent_path_xz),
            "semantic_masks": {
                str(label): _clone_array(mask)
                for label, mask in source_map.last_semantic_masks.items()
            },
            "detected_classes": list(source_map.detected_classes.order[:-1]),
            "policy": {
                "fmm_dist": _clone_array(source_map.policy.fmm_dist),
                "fixed_destination": copy.deepcopy(source_map.policy.fixed_destination),
                "max_destination_socre": float(source_map.policy.max_destination_socre),
                "max_destination_confidence": float(
                    getattr(source_map.policy, "max_destination_confidence", -1.0)
                ),
            },
        },
        "mapper": {
            "tensors": {
                field: _cpu_tensor(getattr(mapper, field))
                for field in _MAPPER_TENSORS
                if hasattr(mapper, field)
            },
            "arrays": {
                field: _clone_array(getattr(mapper, field, None))
                for field in _MAPPER_ARRAYS
            },
            "vis_classes": copy.deepcopy(getattr(mapper, "vis_classes", [])),
        },
    }
    return snapshot


def restore_runtime_snapshot(controller: Any, snapshot: dict[str, Any]) -> None:
    """Restore a snapshot into an existing, identically configured controller."""
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported LaViRA runtime snapshot schema")
    if str(controller.map_backend) != snapshot.get("map_backend"):
        raise ValueError("snapshot map backend does not match controller")
    source_map = controller.map
    mapper = source_map.mapper
    device = source_map.device

    for field, value in snapshot["controller"].items():
        setattr(controller, field, copy.deepcopy(value))
    controller.waypoints = _restore_waypoints(snapshot["waypoints"])

    for field, value in snapshot["source"]["scalars"].items():
        setattr(source_map, field, copy.deepcopy(value))
    for field, value in snapshot["source"]["arrays"].items():
        setattr(source_map, field, _clone_array(value))
    source_map.agent_path_xz = copy.deepcopy(snapshot["source"]["agent_path_xz"])
    source_map.last_semantic_masks = {
        str(label): _clone_array(mask)
        for label, mask in snapshot["source"]["semantic_masks"].items()
    }
    detected = source_map.source.OrderedSet()
    for label in snapshot["source"]["detected_classes"]:
        detected.add(str(label))
    source_map.detected_classes = detected
    policy = snapshot["source"]["policy"]
    source_map.policy.fmm_dist = _clone_array(policy["fmm_dist"])
    source_map.policy.fixed_destination = copy.deepcopy(policy["fixed_destination"])
    source_map.policy.max_destination_socre = float(policy["max_destination_socre"])
    source_map.policy.max_destination_confidence = float(
        policy["max_destination_confidence"]
    )

    for field, value in snapshot["mapper"]["tensors"].items():
        setattr(mapper, field, value.detach().to(device).clone())
    for field, value in snapshot["mapper"]["arrays"].items():
        setattr(mapper, field, _clone_array(value))
    mapper.vis_classes = copy.deepcopy(snapshot["mapper"]["vis_classes"])


def save_runtime_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    """Atomically persist a runtime snapshot."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        torch.save(snapshot, handle)
    temporary.replace(output)


def load_runtime_snapshot(path: str | Path) -> dict[str, Any]:
    """Load a trusted locally generated runtime snapshot onto CPU."""
    with gzip.open(Path(path), "rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=False)

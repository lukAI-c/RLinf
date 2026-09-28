#!/usr/bin/env python3
"""Record the same deterministic primitive trace in the LHX Habitat setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _capture_rgbd(env, label: str, output_dir: Path) -> dict:
    """Save one exact agent state and its Habitat front RGB-D observation."""
    from PIL import Image

    state = env.sim.get_agent_state()
    raw = env.sim.get_sensor_observations()
    rgb = np.asarray(raw["rgb"])[..., :3].copy()
    depth = np.asarray(raw["depth"], dtype=np.float32).copy()
    if depth.ndim == 3:
        depth = depth[..., 0]
    Image.fromarray(rgb).save(output_dir / (label + ".png"))
    np.save(output_dir / (label + "_depth.npy"), depth)
    rotation = state.rotation
    return {
        "label": label,
        "rgb": label + ".png",
        "depth": label + "_depth.npy",
        "position_hab": np.asarray(state.position, dtype=np.float64).tolist(),
        "rotation_xyzw": [
            float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)
        ],
        "yaw": float(_quat_yaw(rotation)),
    }

from habitat_qwen_server import HabitatQwenBridge, _quat_yaw


DEFAULT_ACTIONS = [1, 1, 2, 1, 3, 1, 1, 1, 2, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lavira-root", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--episode-id", default="259")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--actions", default=",".join(map(str, DEFAULT_ACTIONS)))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--capture-dir",
        help="Optional directory for front RGB-D captures at start and after every action.",
    )
    cli = parser.parse_args()

    args = SimpleNamespace(
        lavira_root=cli.lavira_root,
        dataset_path=cli.dataset_path,
        gt_path=cli.gt_path,
        scenes_dir=cli.scenes_dir,
        split="val_unseen",
        gpu_id=cli.gpu_id,
        gpu_ids=str(cli.gpu_id),
        num_envs=1,
        worker_mode=False,
        width=640,
        height=480,
        hfov=79.0,
        camera_height=0.88,
        depth_max=5.0,
        forward_step=0.25,
        turn_angle=30.0,
        max_episode_steps=300,
        max_decisions=0,
        success_distance=3.0,
        metrics_path="",
    )
    bridge = HabitatQwenBridge(args)
    captures = []
    try:
        bridge.reset([str(cli.episode_id)], ["primitive_probe"])
        env = bridge._envs[0]
        requested = np.asarray(env.current_episode.start_position, dtype=np.float64)
        state = env.sim.get_agent_state()
        initial = np.asarray(state.position, dtype=np.float64)
        pathfinder = env.sim.pathfinder
        navmesh_snap = np.asarray(pathfinder.snap_point(requested), dtype=np.float64)
        actions = [int(value) for value in cli.actions.split(",") if value.strip()]
        names = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
        capture_dir = Path(cli.capture_dir) if cli.capture_dir else None
        if capture_dir is not None:
            capture_dir.mkdir(parents=True, exist_ok=True)
            captures.append(_capture_rgbd(env, "frame_000", capture_dir))
        trace = []
        for step_index, action in enumerate(actions, start=1):
            before = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
            env.step({"action": names[action]})
            after_state = env.sim.get_agent_state()
            after = np.asarray(after_state.position, dtype=np.float64)
            snapped_after = np.asarray(pathfinder.snap_point(after), dtype=np.float64)
            moved = float(np.linalg.norm(after[[0, 2]] - before[[0, 2]]))
            trace.append(
                {
                    "step": step_index,
                    "action": action,
                    "position": after.tolist(),
                    "yaw": float(_quat_yaw(after_state.rotation)),
                    "moved_m": moved,
                    "collision": bool(getattr(env.sim, "previous_step_collided", False)),
                    "is_navigable": bool(pathfinder.is_navigable(after)),
                    "navmesh_snap_position": snapped_after.tolist(),
                    "navmesh_snap_error_m": float(np.linalg.norm(snapped_after - after)),
                    "navmesh_snap_valid": bool(np.all(np.isfinite(snapped_after))),
                }
            )
            if capture_dir is not None:
                captures.append(_capture_rgbd(
                    env, "frame_%03d" % step_index, capture_dir
                ))
        result = {
            "simulator": "habitat",
            "episode_id": str(cli.episode_id),
            "scene_id": str(env.current_episode.scene_id),
            "requested_start_position": requested.tolist(),
            "snapped_start_position": initial.tolist(),
            "pathfinder_start_snap": navmesh_snap.tolist(),
            "start_snap_error_m": float(np.linalg.norm(initial - requested)),
            "actions": actions,
            "trace": trace,
            "captures": captures,
        }
    finally:
        bridge.close()

    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if cli.capture_dir:
        (Path(cli.capture_dir) / "manifest.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Record a deterministic primitive trace from the Genesis navigation backend."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rlinf.envs.genark.genesis_backend import GenesisLocalBackend, _batch_find_floor


DEFAULT_ACTIONS = [1, 1, 2, 1, 3, 1, 1, 1, 2, 1]


def _episode(path: str, episode_id: str) -> dict:
    payload = json.loads(Path(path).read_text())
    episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    return next(ep for ep in episodes if str(ep.get("episode_id")) == episode_id)


def _position(backend: GenesisLocalBackend) -> list[float]:
    return backend.cam_pos_hab(0.88)[0].detach().cpu().tolist()


def _navmesh_snap(backend: GenesisLocalBackend) -> tuple[list[float], float, bool]:
    current = backend.cam_pos[:1]
    valid, floor_z, _ = _batch_find_floor(
        current[:, :2],
        current[:, 2] - backend._camera_height,
        backend._tri_v0_2d,
        backend._tri_v1_2d,
        backend._tri_v2_2d,
        backend._tri_v0_3d,
        backend._tri_v1_3d,
        backend._tri_v2_3d,
        backend._max_step_height,
    )
    snapped = current.clone()
    if bool(valid[0]) and bool(torch.isfinite(floor_z[0])):
        snapped[0, 2] = floor_z[0] + backend._camera_height
    snapped_hab = backend.cam_pos_hab(backend._camera_height)[0].clone()
    snapped_hab[1] = snapped[0, 2] - backend._camera_height
    current_hab = backend.cam_pos_hab(backend._camera_height)[0]
    error = float(torch.linalg.vector_norm(snapped_hab - current_hab).cpu())
    return snapped_hab.detach().cpu().tolist(), error, bool(valid[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--episode-id", default="259")
    parser.add_argument("--scene-datasets", required=True)
    parser.add_argument("--glb-cache-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--actions", default=",".join(map(str, DEFAULT_ACTIONS)))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    episode = _episode(args.episodes, args.episode_id)
    cfg = SimpleNamespace(
        cam_res=(64, 48),
        fov=79,
        light_scale=4.0,
        enable_4dir_render=False,
        step_move=0.25,
        step_turn=math.radians(30.0),
        allow_sliding=True,
        max_step_height=0.5,
        agent_radius=0.10,
        camera_height=0.88,
        depth_min=0.1,
        depth_max=5.0,
        init_params=SimpleNamespace(
            scene_datasets=args.scene_datasets,
            glb_cache_dir=args.glb_cache_dir,
            glb_atlas_dir=None,
            glb_atlas_scans=[],
        ),
    )
    backend = GenesisLocalBackend(cfg, num_envs=1)
    backend.load_scene(str(episode["scene_id"]), n_active_envs=1)
    requested_hab = np.asarray(episode["start_position"], dtype=np.float32)
    requested_gen_values = list(backend.hab_to_genesis(requested_hab))
    # Episode positions are agent-floor positions. GenesisLocalBackend stores
    # the camera pose, matching GenarkVecEnv._init_agent_poses().
    requested_gen_values[2] += cfg.camera_height
    requested_gen = torch.tensor(
        [requested_gen_values], dtype=torch.float32, device=backend.device
    )
    rotation = torch.tensor(
        [episode["start_rotation"]], dtype=torch.float32, device=backend.device
    )
    yaw = backend.calculate_initial_yaw(rotation)
    backend.set_agent_poses([0], requested_gen, yaw)

    panorama_position_before = np.asarray(_position(backend), dtype=np.float64)
    panorama_yaw_before = float(backend.cam_yaw[0].detach().cpu())
    panorama_rgb, panorama_depth, panorama_yaw = (
        backend.render_panorama_with_depth(1)
    )
    panorama_position_after = np.asarray(_position(backend), dtype=np.float64)
    panorama_yaw_after = float(backend.cam_yaw[0].detach().cpu())

    actions = [int(value) for value in args.actions.split(",") if value.strip()]
    trace = []
    initial_position = _position(backend)
    for step_index, action in enumerate(actions, start=1):
        before = np.asarray(_position(backend), dtype=np.float64)
        before_yaw = float(backend.cam_yaw[0].detach().cpu())
        action_t = torch.tensor([action], dtype=torch.long, device=backend.device)
        backend.step_physics(
            action_t,
            torch.ones(1, dtype=torch.bool, device=backend.device),
            1,
        )
        after = np.asarray(_position(backend), dtype=np.float64)
        snapped_position, snap_error, snap_valid = _navmesh_snap(backend)
        moved = float(np.linalg.norm(after[[0, 2]] - before[[0, 2]]))
        trace.append(
            {
                "step": step_index,
                "action": action,
                "position": after.tolist(),
                "yaw": float(backend.cam_yaw[0].detach().cpu()),
                "moved_m": moved,
                "collision": bool(action == 1 and moved < 0.249),
                "triangle_index": int(backend.current_tri_idx[0].detach().cpu()),
                "navmesh_snap_position": snapped_position,
                "navmesh_snap_error_m": snap_error,
                "navmesh_snap_valid": snap_valid,
                "yaw_before": before_yaw,
            }
        )

    result = {
        "simulator": "genesis",
        "episode_id": str(args.episode_id),
        "scene_id": str(episode["scene_id"]),
        "requested_start_position": requested_hab.tolist(),
        "snapped_start_position": initial_position,
        "start_snap_error_m": float(
            np.linalg.norm(np.asarray(initial_position) - requested_hab)
        ),
        "atomic_panorama": {
            "rgb_shape": list(panorama_rgb.shape),
            "depth_shape": list(panorama_depth.shape),
            "yaw_shape": list(panorama_yaw.shape),
            "position_gap_m": float(
                np.linalg.norm(panorama_position_after - panorama_position_before)
            ),
            "yaw_gap_rad": float(panorama_yaw_after - panorama_yaw_before),
            "first_heading_delta_rad": float(
                panorama_yaw[0, 0] - panorama_yaw_before
            ),
            "last_heading_delta_rad": float(
                panorama_yaw[0, 11] - panorama_yaw_before
            ),
            "depth_min_m": float(np.nanmin(panorama_depth[0])),
            "depth_max_m": float(np.nanmax(panorama_depth[0])),
            "rgb_nonzero_ratio": float(np.count_nonzero(panorama_rgb[0]) / panorama_rgb[0].size),
        },
        "actions": actions,
        "trace": trace,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

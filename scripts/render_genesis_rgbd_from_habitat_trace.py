#!/usr/bin/env python3
"""Render Genesis RGB-D at exact states captured from Habitat.

The input manifest is produced by ``probe_habitat_primitives.py --capture-dir``.
It intentionally bypasses Genesis navigation snapping: photometric calibration
needs the Habitat camera pose, not the closest Genesis traversible pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from rlinf.envs.genark.genesis_backend import (
    GenesisLocalBackend,
    _update_camera,
)


def _parse_rgb(value: str) -> tuple[float, float, float]:
    values = tuple(float(token) for token in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated RGB values")
    return values


def _parse_matrix(value: str) -> tuple[tuple[float, float, float], ...]:
    values = tuple(float(token) for token in value.split(","))
    if len(values) != 9:
        raise argparse.ArgumentTypeError("expected nine comma-separated matrix values")
    return tuple(tuple(values[row * 3:(row + 1) * 3]) for row in range(3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-datasets", required=True)
    parser.add_argument("--glb-cache-dir", required=True)
    parser.add_argument("--mp3d-glb-cache-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--light-scale", type=float, default=1.0)
    parser.add_argument("--ambient-light", type=_parse_rgb, default=(1.0, 1.0, 1.0))
    parser.add_argument(
        "--color-matrix",
        type=_parse_matrix,
        default=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    parser.add_argument("--color-bias", type=_parse_rgb, default=(0.0, 0.0, 0.0))
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    captures = manifest.get("captures", [])
    if not captures:
        raise ValueError("manifest contains no captures")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = SimpleNamespace(
        cam_res=(640, 480), fov=79.0, light_scale=float(args.light_scale),
        ambient_light=args.ambient_light, rgb_linear_to_srgb=True,
        rgb_color_matrix=args.color_matrix,
        rgb_color_bias=args.color_bias, enable_4dir_render=False,
        step_move=0.25, step_turn=np.deg2rad(30.0), allow_sliding=True,
        max_step_height=0.5, agent_radius=0.1, camera_height=0.88,
        depth_min=0.1, depth_max=5.0,
        init_params=SimpleNamespace(
            scene_datasets=args.scene_datasets, glb_cache_dir=args.glb_cache_dir,
            mp3d_glb_cache_dir=args.mp3d_glb_cache_dir,
            glb_atlas_dir=None, glb_atlas_scans=[],
        ),
    )
    backend = GenesisLocalBackend(cfg, num_envs=1)
    backend.load_scene(args.scene_id, n_active_envs=1)
    rendered = []
    for capture in captures:
        position_hab = np.asarray(capture["position_hab"], dtype=np.float32)
        position_gen = list(backend.hab_to_genesis(position_hab))
        position_gen[2] += cfg.camera_height
        pose = torch.as_tensor([position_gen], dtype=torch.float32, device=backend.device)
        rotation = torch.as_tensor(
            [capture["rotation_xyzw"]], dtype=torch.float32, device=backend.device
        )
        yaw = backend.calculate_initial_yaw(rotation)
        # Direct state assignment avoids changing the Habitat pose through the
        # Genesis navmesh projection used by navigation rollouts.
        backend._cam_pos_t = pose
        backend._cam_yaw_t = yaw
        _update_camera(backend._cam, pose, yaw)
        rgb_raw, _, _, _ = backend._cam.render(
            rgb=True, depth=False, segmentation=False, force_render=True
        )
        if rgb_raw.ndim == 4:
            rgb_raw = rgb_raw[0]
        linear = rgb_raw.float() / 255.0 if rgb_raw.dtype == torch.uint8 else rgb_raw.float()
        _, depth_raw, _, _ = backend._cam.render(
            rgb=False, depth=True, segmentation=False, force_render=False
        )
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[0]
        label = str(capture["label"])
        np.save(output_dir / (label + "_linear.npy"), linear.cpu().numpy())
        np.save(output_dir / (label + "_depth.npy"), depth_raw.float().cpu().numpy())
        Image.fromarray(backend.render_main(1)[0]).save(output_dir / (label + ".png"))
        rendered.append({
            "label": label,
            "linear": label + "_linear.npy",
            "depth": label + "_depth.npy",
            "rgb": label + ".png",
            "position_hab": capture["position_hab"],
            "rotation_xyzw": capture["rotation_xyzw"],
            "yaw": float(yaw[0].cpu()),
        })
    result = {
        "scene_id": args.scene_id,
        "mesh_path": backend._mesh_path,
        "source_manifest": str(manifest_path.resolve()),
        "light_scale": backend._light_scale,
        "ambient_light": list(cfg.ambient_light),
        "rgb_color_matrix": backend._rgb_color_matrix.tolist(),
        "rgb_color_bias": backend._rgb_color_bias.tolist(),
        "frames": rendered,
    }
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify that enabling Genesis depth does not change policy RGB output.

Run on an explicitly chosen idle GPU, for example:
  CUDA_VISIBLE_DEVICES=7 /home/clk/miniconda3/envs/genesis-vllm/bin/python \
    scripts/verify_genesis_rgb_depth_parity.py \
    --config-name genark_eval_qwen_zeroshot \
    --output-dir logs/genesis_rgb_depth_parity
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from rlinf.envs.genark.genesis_backend import GenesisLocalBackend


DEFAULT_SCENE = "mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb"
# Episode 824 start pose, already expressed in Genesis coordinates.
DEFAULT_POSITION = (1.81156, -4.66185, 0.0998907)
DEFAULT_YAW = 0.51


def _black_fraction(rgb: np.ndarray) -> float:
    return float((rgb.max(axis=-1) <= 2).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="genark_eval_qwen_zeroshot")
    parser.add_argument("--config-dir", default="examples/embodiment/config")
    parser.add_argument("--scene-id", default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", default="logs/genesis_rgb_depth_parity")
    parser.add_argument("--x", type=float, default=DEFAULT_POSITION[0])
    parser.add_argument("--y", type=float, default=DEFAULT_POSITION[1])
    parser.add_argument("--z", type=float, default=DEFAULT_POSITION[2])
    parser.add_argument("--yaw", type=float, default=DEFAULT_YAW)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("EMBODIED_PATH", str(repo_root / "examples" / "embodiment"))
    config_dir = (repo_root / args.config_dir).resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name)

    env_cfg = cfg.env.eval
    backend = GenesisLocalBackend(env_cfg, num_envs=1)
    backend.load_scene(args.scene_id, n_active_envs=1)
    position = torch.tensor([[args.x, args.y, args.z]], dtype=torch.float32)
    yaw = torch.tensor([args.yaw], dtype=torch.float32)
    backend.set_agent_poses([0], position, yaw)

    rgb_only = backend.render_main(active_slot_count=1)
    rgb_depth, depth = backend.render_main_with_depth(active_slot_count=1)
    mae = np.abs(rgb_only.astype(np.float32) - rgb_depth.astype(np.float32)).mean()
    valid_depth = np.isfinite(depth) & (depth > 0)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_only[0]).save(output_dir / "rgb_only.png")
    Image.fromarray(rgb_depth[0]).save(output_dir / "rgb_with_depth.png")
    np.save(output_dir / "depth.npy", depth[0])

    print(f"rgb_mae={mae:.8f}")
    print(f"rgb_only_black_fraction={_black_fraction(rgb_only[0]):.6f}")
    print(f"rgb_with_depth_black_fraction={_black_fraction(rgb_depth[0]):.6f}")
    print(f"depth_finite_positive_fraction={valid_depth[0].mean():.6f}")
    print(f"depth_min_positive={depth[0][valid_depth[0]].min() if valid_depth.any() else 'n/a'}")
    print(f"output_dir={output_dir}")
    if mae != 0.0:
        raise SystemExit("RGB parity check failed: RGB changed when depth was enabled.")
    if not valid_depth.any():
        raise SystemExit("Depth check failed: no finite positive depth values.")


if __name__ == "__main__":
    main()

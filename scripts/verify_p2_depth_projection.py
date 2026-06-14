"""
P2 depth-pipeline + bbox→world projection visual verification script.

Usage:
    CUDA_VISIBLE_DEVICES=4 GENESIS_HEADLESS=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python scripts/verify_p2_depth_projection.py \
        --out_dir /tmp/p2_verify \
        --num_steps 30

Output per step (when depth obs is present and bbox is drawn):
    step_NNN_rgb.png   — RGB + red bbox + projected world-coord annotation
    step_NNN_depth.png — Depth heatmap (viridis) + bbox ROI highlight
    step_NNN.txt       — numeric diagnostics (depth stats, pose, world goal)

Projection sanity checks printed to stdout after run:
    - Depth median at bbox should be 0.5–10 m (indoor scene range)
    - world_goal should be plausibly forward of agent (same hemisphere as facing dir)
"""

import argparse
import json
import math
import os
import sys

import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf

# ── Try to import visualization libs (PIL + matplotlib are optional) ──────────
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    print("WARNING: Pillow not found. Install with: pip install Pillow")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("WARNING: matplotlib not found. Install with: pip install matplotlib")

from rlinf.envs.genark.genark_env import GenarkVecEnv as GenArkVecEnv
from rlinf.models.embodiment.qwen_nav.lavira_depth_utils import (
    project_bbox_to_world,
    build_camera_K,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test bboxes: draw several at different image positions to exercise projection
# All in 0-1000 normalised space (same as VLM output).
# ─────────────────────────────────────────────────────────────────────────────
TEST_BBOXES = {
    "centre":     [400, 350, 600, 650],   # image centre
    "right":      [650, 350, 850, 600],   # right of centre → world_z < agent_z
    "left":       [150, 350, 350, 600],   # left  of centre → world_z > agent_z
    "far_centre": [470, 200, 530, 350],   # upper centre (distant object)
}


def _make_env(out_dir: str):
    """Instantiate a minimal single-env GenArkVecEnv with depth enabled."""
    cfg = OmegaConf.create({
        # simulator
        "camera_height":    1.25,
        "step_move":        0.25,
        "step_turn":        math.radians(30.0),
        "allow_sliding":    True,
        "max_step_height":  0.5,
        "agent_radius":     0.18,
        "light_scale":      4.0,
        "fov":              105,
        "cam_res":          [640, 480],
        "max_seq_len":      256,
        # episode
        "success_distance": 3.0,
        "success_bonus":    2.5,
        "use_rel_reward":   True,
        "reward_mode":      "geo_progress",
        "ndtw_coef":        1.0,
        "sr_coef":          10.0,
        "geo_coef":         1.0,
        "decision_level_ndtw": False,
        # P2: depth obs ON
        "enable_depth_obs": True,
        "enable_4dir_render": False,
        "bbox_reward_coef": 0.0,
        # misc
        "auto_reset":       True,
        "ignore_terminations": False,
        "max_episode_steps": 40,
        "max_steps_per_rollout_epoch": 200,
        "seed":             42,
        "group_size":       1,
        "reward_coef":      1.0,
        "format_reward":    0.0,
        "record_bbox_diagnostics": False,
        # genesis backend (single scene)
        "genesis_backend":  "local",
        "scene_offset":     0,
        # init params
        "init_params": OmegaConf.create({
            "episodes_file":    "/home/nvme03/lck/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json",
            "scene_datasets":   "/home/nvme03/lck/genark/scene_datasets",
            "glb_cache_dir":    "/home/clk/workspace/genark/glb_cache",
        }),
        # video (disabled)
        "video_cfg": OmegaConf.create({"save_video": False, "video_base_dir": out_dir}),
    })
    env = GenArkVecEnv(cfg, num_envs=1, seed_offset=0, total_num_processes=1)
    return env


def _save_rgb_annotated(rgb_hw3: np.ndarray, bboxes_annotated: list, step: int, out_dir: str):
    """Draw bboxes + world-coord text on RGB, save PNG."""
    if not _HAS_PIL:
        return
    img = Image.fromarray(rgb_hw3.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    colors = ["red", "lime", "cyan", "yellow"]
    for idx, (label, bbox_1000, world_goal, depth_med) in enumerate(bboxes_annotated):
        H, W = rgb_hw3.shape[:2]
        x1 = int(bbox_1000[0] / 1000 * W)
        y1 = int(bbox_1000[1] / 1000 * H)
        x2 = int(bbox_1000[2] / 1000 * W)
        y2 = int(bbox_1000[3] / 1000 * H)
        color = colors[idx % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if world_goal is not None:
            text = f"{label}\n({world_goal[0]:.1f},{world_goal[1]:.1f})m d={depth_med:.1f}m"
        else:
            text = f"{label}\nno valid depth"
        draw.text((x1 + 2, y1 + 2), text, fill=color)
    path = os.path.join(out_dir, f"step_{step:03d}_rgb.png")
    img.save(path)
    return path


def _save_depth_annotated(depth_hw: np.ndarray, bboxes_1000: dict, step: int, out_dir: str):
    """Save depth heatmap (viridis) + bbox outlines."""
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # Clip to indoor range for better contrast
    clipped = np.clip(depth_hw, 0, 10.0)
    masked = np.where(clipped > 0, clipped, np.nan)
    im = ax.imshow(masked, cmap="viridis", vmin=0, vmax=10)
    plt.colorbar(im, ax=ax, label="depth (m)")
    H, W = depth_hw.shape
    colors = ["red", "lime", "cyan", "yellow"]
    for idx, (label, bbox_1000) in enumerate(bboxes_1000.items()):
        x1 = bbox_1000[0] / 1000 * W
        y1 = bbox_1000[1] / 1000 * H
        w  = (bbox_1000[2] - bbox_1000[0]) / 1000 * W
        h  = (bbox_1000[3] - bbox_1000[1]) / 1000 * H
        rect = matplotlib.patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=colors[idx % len(colors)], facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1 + 2, y1 + 10, label, color=colors[idx % len(colors)], fontsize=8)
    # Depth stats
    valid = depth_hw[depth_hw > 0]
    stats = f"valid={len(valid)}px  min={valid.min():.2f}  med={np.median(valid):.2f}  max={valid.max():.2f}m" if len(valid) else "no valid depth"
    ax.set_title(f"Step {step:03d}  depth heatmap\n{stats}", fontsize=9)
    path = os.path.join(out_dir, f"step_{step:03d}_depth.png")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _import_patches():
    try:
        import matplotlib.patches
        return matplotlib.patches
    except Exception:
        return None


def run_verify(out_dir: str, num_steps: int):
    os.makedirs(out_dir, exist_ok=True)
    K = build_camera_K()
    print(f"Camera K =\n{K}\n")

    print("Initialising GenArkVecEnv with enable_depth_obs=True ...")
    env = _make_env(out_dir)
    obs, _ = env.reset()

    # Patch matplotlib.patches import for _save_depth_annotated
    import importlib
    try:
        import matplotlib.patches as _patches  # noqa
    except ImportError:
        pass

    results = []
    action_seq = [1, 1, 1, 3, 1, 1, 2, 1, 1, 1, 3, 3, 1, 1, 1, 2, 1] * 5  # forward+turns

    for step in range(num_steps):
        # ── Extract obs ───────────────────────────────────────────────────
        import torch
        main_img = obs.get("main_images")
        states   = obs.get("states")
        extra    = obs.get("extra_view_images")

        if main_img is None:
            continue

        rgb_np = (main_img[0].cpu().numpy() if isinstance(main_img, torch.Tensor)
                  else np.asarray(main_img)[0])  # (H, W, 3)

        # Depth
        depth_hw = None
        if extra is not None:
            arr = (extra.cpu().numpy() if isinstance(extra, torch.Tensor) else np.asarray(extra))
            if arr.shape[1] == 1 and arr.dtype == np.float32:
                depth_hw = arr[0, 0, :, :, 0]   # (H, W)

        # Pose from extended states (N, 4) = [elapsed, hab_x, hab_z, yaw]
        pose = None
        if states is not None:
            s = (states.cpu().numpy() if isinstance(states, torch.Tensor) else np.asarray(states))
            if s.shape[1] >= 4:
                pose = s[0, 1:4]  # [hab_x, hab_z, gen_yaw_rad]

        # ── Diagnostics ──────────────────────────────────────────────────
        diag_lines = [f"step={step:03d}"]
        if depth_hw is not None:
            valid_d = depth_hw[depth_hw > 0]
            diag_lines.append(f"  depth: shape={depth_hw.shape} dtype={depth_hw.dtype} "
                               f"valid_px={len(valid_d)} "
                               f"min={valid_d.min():.3f} med={np.median(valid_d):.3f} "
                               f"max={valid_d.max():.3f} m" if len(valid_d) else "  depth: no valid pixels")
        else:
            diag_lines.append("  depth: NOT in obs (extra_view_images shape mismatch or None)")

        if pose is not None:
            diag_lines.append(f"  pose: hab_x={pose[0]:.3f} hab_z={pose[1]:.3f} "
                               f"yaw={pose[2]:.4f}rad ({math.degrees(pose[2]):.1f}deg)")
        else:
            diag_lines.append("  pose: NOT in states (states.shape[1] < 4)")

        # ── Projection for each test bbox ────────────────────────────────
        annotated = []
        if depth_hw is not None and pose is not None:
            for label, bbox_1000 in TEST_BBOXES.items():
                roi_x1 = max(0, int(bbox_1000[0] / 1000 * depth_hw.shape[1]))
                roi_x2 = min(depth_hw.shape[1], int(bbox_1000[2] / 1000 * depth_hw.shape[1]))
                roi_y1 = max(0, int(bbox_1000[1] / 1000 * depth_hw.shape[0]))
                roi_y2 = min(depth_hw.shape[0], int(bbox_1000[3] / 1000 * depth_hw.shape[0]))
                roi = depth_hw[roi_y1:roi_y2, roi_x1:roi_x2]
                valid_roi = roi[(roi > 0) & np.isfinite(roi) & (roi < 15.0)]
                depth_med = float(np.median(valid_roi)) if len(valid_roi) else -1.0

                world = project_bbox_to_world(
                    bbox_1000, depth_hw,
                    gen_yaw_rad=float(pose[2]),
                    hab_x=float(pose[0]),
                    hab_z=float(pose[1]),
                )
                annotated.append((label, bbox_1000, world, depth_med))
                if world is not None:
                    diag_lines.append(
                        f"  [{label}] depth_med={depth_med:.2f}m  "
                        f"world=({world[0]:.2f},{world[1]:.2f})m"
                    )
                else:
                    diag_lines.append(f"  [{label}] depth_med={depth_med:.2f}m  world=NONE (invalid depth)")

        # ── Save images ───────────────────────────────────────────────────
        rgb_path = dep_path = None
        if len(annotated) > 0:
            rgb_path = _save_rgb_annotated(rgb_np, annotated, step, out_dir)
            dep_path = _save_depth_annotated(depth_hw, TEST_BBOXES, step, out_dir)
        elif depth_hw is not None and _HAS_MPL:
            dep_path = _save_depth_annotated(depth_hw, {}, step, out_dir)

        txt_path = os.path.join(out_dir, f"step_{step:03d}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(diag_lines) + "\n")

        print("\n".join(diag_lines))
        if rgb_path:
            print(f"  → {rgb_path}")
        if dep_path:
            print(f"  → {dep_path}")

        results.append({
            "step": step,
            "has_depth": depth_hw is not None,
            "has_pose":  pose is not None,
        })

        # Step env
        action = action_seq[step % len(action_seq)]
        import torch as _torch
        a = _torch.tensor([action], dtype=_torch.int64)
        obs, _, _, _, _ = env.step(a)

    # ── Summary ───────────────────────────────────────────────────────────
    n_with_depth = sum(r["has_depth"] for r in results)
    n_with_pose  = sum(r["has_pose"]  for r in results)
    print(f"\n{'='*60}")
    print(f"P2 Verification Summary ({num_steps} steps)")
    print(f"  Steps with depth obs  : {n_with_depth}/{num_steps}")
    print(f"  Steps with pose in states: {n_with_pose}/{num_steps}")
    print(f"  Output dir: {out_dir}")
    print(f"\nSanity checks to confirm manually:")
    print(f"  1. depth PNGs — walls/floors should appear at 1–8 m (viridis: purple=close, yellow=far)")
    print(f"  2. 'centre' bbox world goal should be approx agent_pos + (depth, 0) when yaw≈0")
    print(f"  3. 'right' bbox world_z < 'centre' world_z (right side of image = hab -Z direction)")
    print(f"  4. 'left'  bbox world_z > 'centre' world_z")
    print(f"{'='*60}")

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"steps": results, "n_with_depth": n_with_depth,
                   "n_with_pose": n_with_pose}, f, indent=2)
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",   default="/tmp/p2_verify")
    parser.add_argument("--num_steps", type=int, default=30)
    args = parser.parse_args()
    run_verify(args.out_dir, args.num_steps)


if __name__ == "__main__":
    main()

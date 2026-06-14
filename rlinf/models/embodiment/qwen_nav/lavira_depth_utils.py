"""
P2: bbox → world (Habitat XZ) coordinate projection.

Adapted from LaViRA vlnce_baselines/utils/depth_utils.py:get_world_xz_from_pixel.

Coordinate conventions
----------------------
Genesis world:  X forward (yaw=0), Y left (yaw=+π/2), Z up.
Habitat world:  X = Genesis X,  Z = −Genesis Y,  Y = height (up).
Genesis cam_yaw θ: forward = (cos θ, sin θ) in Genesis XY plane.

In the Habitat XZ plane the same forward vector is (cos θ, −sin θ).
The projection formula below uses heading = −gen_yaw so that
forward = (cos(−θ), sin(−θ)) = (cos θ, −sin θ)  ← correct.

The original LaViRA function had a misleading variable name ('heading_rad')
but actually expected *degrees*; here we accept radians directly (no deg2rad).
"""

import math
from typing import Optional

import numpy as np

# Default render geometry — must match genesis_backend.py defaults.
_DEFAULT_RENDER_W: int   = 640
_DEFAULT_RENDER_H: int   = 480
_DEFAULT_HFOV_DEG: float = 105.0
# Hard cap for valid depth (metres).  GenArk indoor scenes rarely exceed 15 m.
_MAX_DEPTH_M: float = 15.0


def build_camera_K(
        hfov_deg: float = _DEFAULT_HFOV_DEG,
        width:    int   = _DEFAULT_RENDER_W,
        height:   int   = _DEFAULT_RENDER_H,
) -> np.ndarray:
    """Return 3×3 camera intrinsics K (float64)."""
    fx = width  / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
    vfov_rad = 2.0 * math.atan(height / width * math.tan(math.radians(hfov_deg / 2.0)))
    fy = height / (2.0 * math.tan(vfov_rad / 2.0))
    return np.array(
        [[fx, 0.0, width / 2.0],
         [0.0, fy,  height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


# Module-level default K (built once).
_DEFAULT_K: np.ndarray = build_camera_K()


def project_bbox_to_world(
        bbox_2d:     list,            # [x1,y1,x2,y2] in 0-1000 normalised space
        depth_hw:    np.ndarray,      # (H, W) float32, metric metres
        gen_yaw_rad: float,           # Genesis cam_yaw (rad); 0=+X, CCW positive
        hab_x:       float,           # Habitat agent X (metres)
        hab_z:       float,           # Habitat agent Z (metres) = −Genesis Y
        K:           Optional[np.ndarray] = None,
        render_w:    int   = _DEFAULT_RENDER_W,
        render_h:    int   = _DEFAULT_RENDER_H,
) -> Optional[np.ndarray]:
    """
    Project bbox centre to Habitat world (XZ) coordinates.

    Args:
        bbox_2d:     [x1, y1, x2, y2] in 0-1000 normalised image space.
        depth_hw:    Depth map (H, W) float32, metric metres.
        gen_yaw_rad: Genesis camera yaw in radians.
        hab_x:       Agent Habitat X coordinate (metres).
        hab_z:       Agent Habitat Z coordinate (metres).
        K:           3×3 camera intrinsics; defaults to build_camera_K().
        render_w/h:  Render resolution matching depth_hw.

    Returns:
        np.ndarray [world_x, world_z] in metres, or None if depth is invalid.
    """
    if K is None:
        K = _DEFAULT_K

    x1, y1, x2, y2 = bbox_2d
    H, W = depth_hw.shape

    # Compute median depth over the bbox ROI (robust to noise/holes).
    px1 = max(0, min(int(round(x1 / 1000.0 * W)),     W - 1))
    px2 = max(px1 + 1, min(int(round(x2 / 1000.0 * W)), W))
    py1 = max(0, min(int(round(y1 / 1000.0 * H)),     H - 1))
    py2 = max(py1 + 1, min(int(round(y2 / 1000.0 * H)), H))

    roi = depth_hw[py1:py2, px1:px2]
    valid_mask = (roi > 0) & np.isfinite(roi) & (roi < _MAX_DEPTH_M)
    valid_depths = roi[valid_mask]
    if len(valid_depths) == 0:
        return None
    depth = float(np.median(valid_depths))

    # Projection ray: horizontal centre, vertical bottom of bbox.
    # Bottom-centre ≈ where the target meets the floor → more accurate approach dist.
    # (LaViRA ZS_Evaluator convention: bbox bottom-centre as the target pixel.)
    u = (x1 + x2) / 2.0 / 1000.0 * render_w   # horizontal centre
    v = y2          / 1000.0 * render_h          # bottom edge

    # Back-project to camera frame: X-right, Y-down, Z-forward.
    K_inv = np.linalg.inv(K)
    cam   = depth * (K_inv @ np.array([u, v, 1.0], dtype=np.float64))
    local_x = float(cam[0])   # rightward in camera frame
    local_z = float(cam[2])   # forward  in camera frame

    # Genesis yaw → Habitat heading: hab forward = (cos θ, −sin θ) in XZ.
    # Equivalently, use heading = −gen_yaw in the standard formula.
    heading = -gen_yaw_rad
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    world_x = hab_x + local_z * cos_h + local_x * sin_h
    world_z = hab_z + local_z * sin_h - local_x * cos_h

    return np.array([world_x, world_z], dtype=np.float32)

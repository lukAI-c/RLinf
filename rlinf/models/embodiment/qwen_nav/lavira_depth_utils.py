"""
RLINF/LHX ADAPTER: bbox → world (Habitat XZ) coordinate projection.

The LHX mapper/FMM algorithm bodies are copied verbatim under
``rlinf.third_party.lavira_rft.source`` and must not be reimplemented here.
This module remains active only where RLinf translates Qwen/GroundedSAM
geometry into simulator coordinates before handing state to the source core.

Adapted from LaViRA vlnce_baselines/utils/depth_utils.py:get_world_xz_from_pixel.

Coordinate conventions
----------------------
Genesis world:  X forward (yaw=0), Y left (yaw=+π/2), Z up.
Habitat world:  X = Genesis X,  Z = −Genesis Y,  Y = height (up).
Genesis cam_yaw θ: forward = (cos θ, sin θ) in Genesis XY plane.

In the Habitat XZ plane the same forward vector is (cos θ, −sin θ).
The Genesis→Habitat Z reflection also reverses the camera-right axis:
camera right is (sin θ, cos θ) in Habitat XZ.  The projection formula below
uses heading = −gen_yaw for forward and explicitly applies that reflected
right axis.  This avoids mirroring every off-centre DINO target across the
agent's forward ray.

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
# LaViRA preprocesses target-projection depth with MIN_DEPTH=0.1m and
# MAX_DEPTH=5.0m before calling get_world_xz_from_pixel.
_MAX_DEPTH_M: float = 5.0


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
    valid_mask = (roi >= 0.1) & np.isfinite(roi) & (roi <= _MAX_DEPTH_M)
    valid_depths = roi[valid_mask]
    if len(valid_depths) == 0:
        return None
    median_depth = float(np.median(valid_depths))
    depth_diff = np.abs(roi - median_depth)
    depth_diff[~valid_mask] = np.inf
    roi_y, roi_x = np.unravel_index(np.argmin(depth_diff), depth_diff.shape)
    u = px1 + int(roi_x)
    v = py1 + int(roi_y)
    depth = float(roi[roi_y, roi_x])

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

    world_x = hab_x + local_z * cos_h - local_x * sin_h
    world_z = hab_z + local_z * sin_h + local_x * cos_h

    return np.array([world_x, world_z], dtype=np.float32)


def project_point_to_world(
        point_2d:    list,            # [x,y] in 0-1000 normalised space
        depth_hw:    np.ndarray,      # (H, W) float32, metric metres
        gen_yaw_rad: float,           # Genesis cam_yaw (rad); 0=+X, CCW positive
        hab_x:       float,           # Habitat agent X (metres)
        hab_z:       float,           # Habitat agent Z (metres) = −Genesis Y
        K:           Optional[np.ndarray] = None,
        render_w:    int   = _DEFAULT_RENDER_W,
        render_h:    int   = _DEFAULT_RENDER_H,
        window_px:   int | None = None,
) -> Optional[np.ndarray]:
    """
    Project a normalized point_2d to Habitat world (XZ) coordinates.

    Depth is taken as the median in a small window around the point so one bad
    depth pixel does not kill the planner target. Returns None if depth is invalid.
    """
    if K is None:
        K = _DEFAULT_K

    sample = sample_point_depth(point_2d, depth_hw, window_px=window_px)
    if sample is None:
        return None
    u, v, depth = sample

    K_inv = np.linalg.inv(K)
    cam = depth * (K_inv @ np.array([float(u), float(v), 1.0], dtype=np.float64))
    local_x = float(cam[0])
    local_z = float(cam[2])

    heading = -gen_yaw_rad
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    world_x = hab_x + local_z * cos_h - local_x * sin_h
    world_z = hab_z + local_z * sin_h + local_x * cos_h
    return np.array([world_x, world_z], dtype=np.float32)


def sample_point_depth(
        point_2d: list,
        depth_hw: np.ndarray,
        window_px: int | None = None,
) -> Optional[tuple[int, int, float]]:
    """Return the exact pixel and median depth used by point projection."""
    x, y = point_2d
    H, W = depth_hw.shape
    # ZS_Evaluator_mp converts GroundingDINO's pixel bbox with ``int`` before
    # projection.  GroundedSAM stores the same coordinates in 0-1000 space,
    # so truncation here is the exact inverse used by the source path.
    u = max(0, min(int(x / 1000.0 * W), W - 1))
    v = max(0, min(int(y / 1000.0 * H), H - 1))

    window_sizes = (3, 5, 7, 9) if window_px is None else (max(1, int(window_px)),)
    for window_size in window_sizes:
        half = window_size // 2
        x1, x2 = max(0, u - half), min(W, u + half + 1)
        y1, y2 = max(0, v - half), min(H, v + half + 1)
        roi = depth_hw[y1:y2, x1:x2]
        # Match get_world_xz_from_pixel(): depth backoff deliberately keeps
        # positive samples below MIN_DEPTH so the source can accept its final
        # near-agent target once the full depth image drops below 0.1 m.
        valid_mask = (roi > 0.0) & np.isfinite(roi) & (roi <= _MAX_DEPTH_M)
        valid_depths = roi[valid_mask]
        if len(valid_depths):
            return u, v, float(np.median(valid_depths))
    return None

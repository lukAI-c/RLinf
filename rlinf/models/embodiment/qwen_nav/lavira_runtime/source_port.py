"""RLinf/LHX boundary helpers that are still lifecycle adaptations.

Mapping and FMM now execute vendored LHX source directly. The helpers below
cover behavior embedded inside LHX's monolithic evaluator that cannot be
called independently yet. Treat them as parity-sensitive adapter code, not as
proof that the complete inference chain is byte-identical to LHX.
"""

from __future__ import annotations

import math

import numpy as np


_DIRECTION_NAMES = ("forward", "left", "behind", "right")
_SOURCE_DEPTH_MIN_M = 0.1
_SOURCE_DEPTH_MAX_M = 5.0


def check_blocked_directions(depth_by_direction: np.ndarray | None) -> set[str]:
    """Port of ``VLMReasoningAgent._check_blocked_directions``.

    ``depth_by_direction`` uses the source prompt order
    ``[forward, left, behind, right]`` and contains metric Genesis depth.  The
    source evaluator receives Habitat normalized depth, so invert Habitat's
    configured 0.1--5.0m linear mapping before applying the exact 0.15 test.
    """
    blocked_directions: set[str] = set()
    if depth_by_direction is None:
        return blocked_directions

    depth_views = np.asarray(depth_by_direction)
    if depth_views.ndim != 3:
        return blocked_directions

    for index, name in enumerate(_DIRECTION_NAMES):
        if index >= len(depth_views):
            break
        d_img = depth_views[index]
        if d_img.ndim == 3:
            d_img = d_img[:, :, 0]
        if d_img.ndim != 2:
            continue
        # Habitat-Sim maps normalized depth linearly over [MIN_DEPTH,
        # MAX_DEPTH]. RLinf's bridge exposes metric metres, so reconstruct the
        # exact values consumed by LHX's _check_blocked_directions().
        normalized = (
            d_img - _SOURCE_DEPTH_MIN_M
        ) / (_SOURCE_DEPTH_MAX_M - _SOURCE_DEPTH_MIN_M)
        h, w = normalized.shape
        center_d = normalized[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
        valid_mask = center_d > 0.01
        if np.any(valid_mask) and float(np.mean(center_d[valid_mask])) < 0.15:
            blocked_directions.add(name)
    return blocked_directions


# The three functions below are copied from lavira-rft's map_utils.py, with
# imports removed because the Genesis adapter already provides map coordinates.
def get_mask(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 <= step_size ** 2 \
               and ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 > (step_size - 1) ** 2:
                mask[i, j] = 1
    mask[size // 2, size // 2] = 1
    return mask


def get_collision_mask(known_vector: np.ndarray, mask_data: np.ndarray, angle_threshold: float):
    collision_map = np.zeros_like(mask_data)
    center = np.array(mask_data.shape) // 2
    rows, cols = np.indices(mask_data.shape)
    rows_from_center = rows - center[0]
    cols_from_center = cols - center[1]
    nonzero_indices = np.nonzero(mask_data)
    vectors = np.array([rows_from_center[nonzero_indices], cols_from_center[nonzero_indices]])
    vector_lengths = np.linalg.norm(vectors, axis=0)
    known_vector_length = np.linalg.norm(known_vector)
    rotation_matrix = np.array([[0, -1], [1, 0]])
    known_vector = np.dot(rotation_matrix, known_vector)
    known_vector_expanded = np.tile(known_vector[:, np.newaxis], vectors.shape[1])
    cos_angles = np.sum(known_vector_expanded * vectors, axis=0) / (
        known_vector_length * vector_lengths + 1e-10
    )
    angles_rad = np.arccos(np.clip(cos_angles, -1.0, 1.0))
    angles_deg = np.degrees(angles_rad)
    collision_map[
        nonzero_indices[0][angles_deg <= angle_threshold],
        nonzero_indices[1][angles_deg <= angle_threshold],
    ] = 1
    return collision_map


def collision_check_fmm(
    last_rc: np.ndarray,
    current_rc: np.ndarray,
    yaw_rad: float,
    map_shape: tuple[int, int],
    *,
    cells_per_m: float,
    collision_threshold_m: float = 0.2,
) -> np.ndarray:
    """Genesis-coordinate adapter for source ``collision_check_fmm``.

    The copied source algorithm operates on map row/column coordinates.  This
    adapter supplies those coordinates directly rather than recreating Habitat
    ``sensor_pose`` conversion.
    """
    collision_map = np.zeros(map_shape, dtype=bool)
    displacement = float(np.linalg.norm(current_rc - last_rc))
    if displacement >= collision_threshold_m * cells_per_m:
        return collision_map

    x, y = current_rc
    dx, dy = float(x - int(x)), float(y - int(y))
    mask = get_mask(dx, dy, scale=1, step_size=5)
    # Source uses heading=-pose_heading before ``angle_to_vector``.  Genesis
    # yaw=0 points +X, which maps to increasing map columns under this basis.
    heading = -float(yaw_rad)
    heading_vector = np.asarray([math.cos(heading), math.sin(heading)])
    collision_mask = get_collision_mask(heading_vector, mask, 32)
    x, y = int(x), int(y)
    if x - 5 >= 0 and x + 6 < map_shape[0] and y - 5 >= 0 and y + 6 < map_shape[1]:
        collision_map[x - 5: x + 6, y - 5: y + 6] = collision_mask.astype(bool)
    return collision_map

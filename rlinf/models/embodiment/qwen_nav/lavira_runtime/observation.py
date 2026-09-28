"""Direction-aware RGB-D adapter at the RLinf/LHX ownership boundary.

The vendored LHX mapper/FMM consumes the resulting state, but projection and
simulator-coordinate conversion remain RLinf integration code. Keep this
module parity-tested against LHX rather than adding another mapper or planner
implementation here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from ..lavira_depth_utils import (
    build_camera_K,
    project_point_to_world,
    sample_point_depth,
)


_DIRECTION_INDEX = {"forward": 0, "left": 1, "behind": 2, "right": 3}


@dataclass
class LaviraObservation:
    """One agent observation in canonical ``[front, left, behind, right]`` order."""

    rgb_by_direction: list[Image.Image]
    depth_by_direction: np.ndarray
    yaw_by_direction: np.ndarray
    hab_x: float
    hab_z: float
    hfov_deg: float = 79.0

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_by_direction)
        if depth.ndim != 3 or depth.shape[0] != 4:
            raise ValueError(
                "depth_by_direction must have shape [4,H,W], "
                f"got {depth.shape}"
            )
        self.depth_by_direction = depth.astype(np.float32, copy=False)
        height, width = depth.shape[1:]
        self.camera_k = build_camera_K(self.hfov_deg, width, height)

    @property
    def front_depth(self) -> np.ndarray:
        return self.depth_by_direction[0]

    @property
    def front_yaw(self) -> float:
        return float(self.yaw_by_direction[0])

    def view_index(self, raw_action: str) -> int:
        direction = str(raw_action or "navigate to forward").replace("navigate to ", "")
        return _DIRECTION_INDEX.get(direction, 0)

    def project_target(
        self,
        raw_action: str,
        point_2d: Optional[list[float]] = None,
        bbox_2d: Optional[list[float]] = None,
    ) -> Optional[tuple[float, float]]:
        idx = self.view_index(raw_action)
        depth = self.depth_by_direction[idx]
        yaw = float(self.yaw_by_direction[idx])
        if point_2d is not None:
            result = project_point_to_world(
                point_2d, depth, yaw, self.hab_x, self.hab_z, K=self.camera_k
            )
            if result is not None:
                return float(result[0]), float(result[1])
        if bbox_2d is not None:
            x1, _y1, x2, y2 = bbox_2d
            result = project_point_to_world(
                [(float(x1) + float(x2)) / 2.0, float(y2)],
                depth,
                yaw,
                self.hab_x,
                self.hab_z,
                K=self.camera_k,
            )
            if result is not None:
                return float(result[0]), float(result[1])
        return None

    def project_front_target(
        self,
        point_2d: Optional[list[float]] = None,
        bbox_2d: Optional[list[float]] = None,
        stair=False,
        depth_override: Optional[np.ndarray] = None,
    ) -> Optional[tuple[float, float]]:
        """Project an upstream LaViRA waypoint after its turn macro finishes."""
        depth = self.front_depth if depth_override is None else depth_override
        # LaViRA-RFT deliberately ignores point_2d for stair transitions and
        # uses the bbox top-centre instead (the visible stair edge is the
        # reliable approach cue).
        if point_2d is not None and stair not in ("up", "down"):
            result = project_point_to_world(
                point_2d, depth, self.front_yaw, self.hab_x, self.hab_z,
                K=self.camera_k,
            )
            if result is not None:
                return float(result[0]), float(result[1])
        if bbox_2d is not None:
            # LaViRA uses the bbox top-centre for stairs and bottom-centre for
            # ordinary targets, because the visible stair edge is the useful
            # approach cue rather than its floor contact point.
            if stair in ("up", "down"):
                x1, y1, x2, _y2 = bbox_2d
                result = project_point_to_world(
                    [(float(x1) + float(x2)) / 2.0, float(y1)],
                    depth, self.front_yaw, self.hab_x, self.hab_z,
                    K=self.camera_k,
                )
                if result is not None:
                    return float(result[0]), float(result[1])
            x1, _y1, x2, y2 = bbox_2d
            result = project_point_to_world(
                [(float(x1) + float(x2)) / 2.0, float(y2)],
                depth,
                self.front_yaw,
                self.hab_x,
                self.hab_z,
                K=self.camera_k,
            )
            if result is not None:
                return float(result[0]), float(result[1])
        return None

    def front_target_sample(
        self,
        point_2d: Optional[list[float]],
        bbox_2d: Optional[list[float]],
        stair=False,
        depth_override: Optional[np.ndarray] = None,
    ) -> Optional[tuple[int, int, float]]:
        """Return the pixel/depth used by the source-aligned front projection."""
        depth = self.front_depth if depth_override is None else depth_override
        target_point = point_2d
        if target_point is None or stair in ("up", "down"):
            if bbox_2d is None:
                return None
            x1, y1, x2, y2 = bbox_2d
            target_point = [
                (float(x1) + float(x2)) / 2.0,
                float(y1) if stair in ("up", "down") else float(y2),
            ]
        return sample_point_depth(target_point, depth)

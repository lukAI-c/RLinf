"""Persistent navigation graph, deliberately separate from image history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PIL import Image


@dataclass
class WaypointNode:
    id: int
    x: float
    z: float
    yaw: float
    action: str
    target: str
    progress: str
    parent_id: Optional[int]
    arrival_image: Optional[Image.Image] = None
    failed_directions: set[str] = field(default_factory=set)
    failed: bool = False
    failed_dir: bool = False
    reached: bool = False


class WaypointMemory:
    def __init__(self) -> None:
        self._nodes: list[WaypointNode] = []
        self._next_id = 0

    def reset(self) -> None:
        self._nodes.clear()
        self._next_id = 0

    @property
    def nodes(self) -> list[WaypointNode]:
        return self._nodes

    def add(self, x: float, z: float, yaw: float, action: str, target: str, progress: str,
            image: Optional[Image.Image] = None, parent_id: Optional[int] = None,
            waypoint_id: Optional[int] = None) -> WaypointNode:
        node_id = self._next_id if waypoint_id is None else int(waypoint_id)
        node = WaypointNode(node_id, x, z, yaw, action, target, progress, parent_id, image)
        self._next_id = max(self._next_id, node_id + 1)
        self._nodes.append(node)
        return node

    def get(self, waypoint_id: Optional[int]) -> Optional[WaypointNode]:
        return next((node for node in self._nodes if node.id == waypoint_id), None)

    def update(
        self,
        waypoint_id: int,
        *,
        action: str,
        target: str,
        progress: str,
    ) -> Optional[WaypointNode]:
        node = self.get(waypoint_id)
        if node is None:
            return None
        node.action = str(action)
        node.target = str(target)
        node.progress = str(progress)
        return node

    def pop_latest(self) -> Optional[WaypointNode]:
        return self._nodes.pop() if self._nodes else None

    def mark_failed_branch(self, waypoint_id: int) -> bool:
        """Port source backtrack bookkeeping onto persistent waypoint nodes."""
        for index, node in enumerate(self._nodes):
            if node.id != waypoint_id:
                continue
            node.failed_dir = True
            for abandoned in self._nodes[index + 1:]:
                abandoned.failed = True
            return True
        return False

"""Immutable same-episode assignment for Robostral Phase-1 / Gate A-B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FixedEpisodeAssignment:
    """Walk a frozen episode list. No weighted sampling, no balancer ties."""

    def __init__(
        self,
        episodes: list[dict],
        groups: list[dict],
        *,
        scene_id: str | None = None,
    ) -> None:
        self._eps = {str(ep.get("episode_id")): ep for ep in episodes}
        self._groups = [dict(group) for group in groups]
        if not self._groups:
            raise ValueError("episode_assignment.groups must not be empty")
        self._scene_id = None if scene_id is None else str(scene_id)
        self._cursor = 0
        self._seen: dict[str, int] = {key: 0 for key in self._eps}
        self._passes = 0
        for group in self._groups:
            episode_id = str(group["episode_id"])
            if episode_id not in self._eps:
                raise ValueError(
                    f"episode_assignment references unknown episode_id={episode_id}"
                )
            assigned_scene = str(self._eps[episode_id].get("scene_id", ""))
            if self._scene_id and assigned_scene and assigned_scene != self._scene_id:
                raise ValueError(
                    "episode_assignment scene mismatch: "
                    f"file={self._scene_id} episode={episode_id} data={assigned_scene}"
                )

    @classmethod
    def from_file(cls, path: str | Path, episodes: list[dict]) -> "FixedEpisodeAssignment":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            episodes,
            payload.get("groups") or [],
            scene_id=payload.get("scene_id"),
        )

    def set_global_step(self, step: int) -> None:
        """Point at assignment group ``step`` unless that group was already consumed.

        Constructor priming calls ``next_episode()`` for group 0. A later
        ``set_global_step(0)`` must not rewind the cursor back onto group 0.
        Resume at step ``k`` on a fresh object sets the cursor to ``k``.
        """
        n = len(self._groups)
        step = max(0, int(step))
        desired = step % n
        if self._cursor > desired:
            self._passes = max(self._passes, step // n)
            return
        self._cursor = desired
        self._passes = step // n

    def next_episode(self) -> tuple[dict, bool]:
        if self._cursor >= len(self._groups):
            self._cursor = 0
            self._passes += 1
            new_pass = True
        else:
            new_pass = False
        group = self._groups[self._cursor]
        self._cursor += 1
        episode_id = str(group["episode_id"])
        self._seen[episode_id] = self._seen.get(episode_id, 0) + 1
        return self._eps[episode_id], new_pass

    def record(self, episode_id: Any, success: bool) -> None:
        return None

    def mark_seen(self, episode_id: Any) -> None:
        key = str(episode_id)
        self._seen[key] = self._seen.get(key, 0) + 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "cursor": int(self._cursor),
            "passes": int(self._passes),
            "seen": dict(self._seen),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._cursor = int(state.get("cursor", 0))
        self._passes = int(state.get("passes", 0))
        seen = state.get("seen") or {}
        self._seen = {str(key): int(value) for key, value in seen.items()}

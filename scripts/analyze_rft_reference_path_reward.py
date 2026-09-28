#!/usr/bin/env python3
"""Counterfactually score saved LaViRA audit trajectories with RFT path shaping."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from rlinf.envs.genark.genark_env import _reference_path_potential


def _load_episode(dataset: Path, episode_id: str) -> dict:
    opener = gzip.open if dataset.suffix == ".gz" else open
    with opener(dataset, "rt") as handle:
        data = json.load(handle)
    episodes = data.get("episodes", data) if isinstance(data, dict) else data
    matches = [
        episode
        for episode in episodes
        if str(episode.get("episode_id")) == str(episode_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one episode_id={episode_id}, found {len(matches)}"
        )
    return matches[0]


def _decision_positions(audit_path: Path) -> list[tuple[int, np.ndarray, str]]:
    positions: dict[int, np.ndarray] = {}
    actions: dict[int, str] = {}
    with audit_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            decision_id = int(row.get("decision_id", 0))
            if decision_id <= 0 or "agent_xz" not in row:
                continue
            x, z = row["agent_xz"]
            positions[decision_id] = np.array([x, 0.0, z], dtype=float)
            request = row.get("request") or {}
            if request.get("raw_action"):
                actions[decision_id] = str(request["raw_action"])
    return [
        (decision_id, position, actions.get(decision_id, ""))
        for decision_id, position in sorted(positions.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-id", default="609")
    parser.add_argument("--lateral-penalty", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=2.5)
    args = parser.parse_args()

    episode = _load_episode(args.dataset, args.episode_id)
    reference_path = np.asarray(episode["reference_path"], dtype=float)
    start = np.asarray(episode["start_position"], dtype=float)
    initial_potential, _, _ = _reference_path_potential(
        start, reference_path, args.lateral_penalty
    )

    audit_paths = sorted(args.log_dir.glob(
        "lavira_maps/worker_*/env_*/episode_*/projection_audit.jsonl"
    ))
    if not audit_paths:
        raise FileNotFoundError(f"No projection_audit.jsonl below {args.log_dir}")

    for audit_path in audit_paths:
        previous = initial_potential
        deltas = []
        rows = []
        for decision_id, position, action in _decision_positions(audit_path):
            potential, along, lateral = _reference_path_potential(
                position, reference_path, args.lateral_penalty
            )
            delta = float(np.clip(potential - previous, -args.clip, args.clip))
            deltas.append(delta)
            rows.append(
                {
                    "decision_id": decision_id,
                    "action": action,
                    "delta": round(delta, 4),
                    "along_track": round(along, 4),
                    "lateral": round(lateral, 4),
                }
            )
            previous = potential
        env_name = next(
            part for part in audit_path.parts if part.startswith("env_")
        )
        print(json.dumps({
            "env": env_name,
            "total": round(sum(deltas), 4),
            "positive": sum(delta > 0 for delta in deltas),
            "negative": sum(delta < 0 for delta in deltas),
            "decisions": rows,
        }))


if __name__ == "__main__":
    main()

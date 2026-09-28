#!/usr/bin/env python3
"""Pair deterministic Genesis/Habitat primitive traces step by step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genesis", required=True)
    parser.add_argument("--habitat", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    genesis = json.loads(Path(args.genesis).read_text())
    habitat = json.loads(Path(args.habitat).read_text())
    if genesis["actions"] != habitat["actions"]:
        raise ValueError("primitive action sequences differ")
    paired = []
    for left, right in zip(genesis["trace"], habitat["trace"]):
        position_gap = float(
            np.linalg.norm(np.asarray(left["position"]) - np.asarray(right["position"]))
        )
        snap_gap = float(
            np.linalg.norm(
                np.asarray(left["navmesh_snap_position"])
                - np.asarray(right["navmesh_snap_position"])
            )
        )
        paired.append(
            {
                "step": left["step"],
                "action": left["action"],
                "position_gap_m": position_gap,
                "genesis_collision": left["collision"],
                "habitat_collision": right["collision"],
                "collision_match": left["collision"] == right["collision"],
                "genesis_navmesh_snap_error_m": left["navmesh_snap_error_m"],
                "habitat_navmesh_snap_error_m": right["navmesh_snap_error_m"],
                "navmesh_snap_position_gap_m": snap_gap,
                "navmesh_snap_valid_match": (
                    left["navmesh_snap_valid"] == right["navmesh_snap_valid"]
                ),
                "genesis_position": left["position"],
                "habitat_position": right["position"],
            }
        )
    result = {
        "episode_id": genesis["episode_id"],
        "start_position_gap_m": float(
            np.linalg.norm(
                np.asarray(genesis["snapped_start_position"])
                - np.asarray(habitat["snapped_start_position"])
            )
        ),
        "max_position_gap_m": max(row["position_gap_m"] for row in paired),
        "max_navmesh_snap_position_gap_m": max(
            row["navmesh_snap_position_gap_m"] for row in paired
        ),
        "collision_match_count": sum(row["collision_match"] for row in paired),
        "navmesh_snap_valid_match_count": sum(
            row["navmesh_snap_valid_match"] for row in paired
        ),
        "step_count": len(paired),
        "paired": paired,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

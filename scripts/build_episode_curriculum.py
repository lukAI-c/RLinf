#!/usr/bin/env python3
"""Build an episode curriculum manifest from repeated frozen-policy evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ACTIVE_CLEAN_TYPES = {"clean", "recovered"}
MEANINGFUL_REACH_TYPES = ACTIVE_CLEAN_TYPES | {"proximity"}


def classify(clean_count: int, reach_count: int, mean_progress: float) -> str:
    if clean_count >= 3:
        return "clean_seed"
    if reach_count >= 2:
        return "stop_frontier"
    if reach_count >= 1 or mean_progress > 0:
        return "nav_frontier"
    return "hard"


def build_manifest(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scene_id"]), str(row["episode_id"]))].append(row)

    episodes = []
    tier_counts: dict[str, int] = defaultdict(int)
    for (scene_id, episode_id), trials in sorted(grouped.items()):
        clean_count = sum(
            str(trial.get("success_type", "")) in ACTIVE_CLEAN_TYPES
            for trial in trials
        )
        reach_count = sum(
            str(trial.get("success_type", "")) in MEANINGFUL_REACH_TYPES
            for trial in trials
        )
        mean_progress = sum(float(trial.get("dtg_progress", 0.0)) for trial in trials) / len(trials)
        tier = classify(clean_count, reach_count, mean_progress)
        tier_counts[tier] += 1
        episodes.append(
            {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "tier": tier,
                "active_clean_count": clean_count,
                "meaningful_reach_count": reach_count,
                "mean_dtg_progress": mean_progress,
                "num_trials": len(trials),
            }
        )

    return {
        "schema_version": 1,
        "classification": {
            "clean_seed": "active_clean_count >= 3",
            "stop_frontier": "active_clean_count < 3 and meaningful_reach_count >= 2",
            "nav_frontier": "active_clean_count < 3, meaningful_reach_count < 2, and (meaningful_reach_count >= 1 or mean_dtg_progress > 0)",
            "hard": "active_clean_count == 0, meaningful_reach_count == 0, and mean_dtg_progress <= 0",
            "active_clean_types": sorted(ACTIVE_CLEAN_TYPES),
            "meaningful_reach_types": sorted(MEANINGFUL_REACH_TYPES),
            "lucky_start_excluded": True,
        },
        "tier_counts": dict(sorted(tier_counts.items())),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("input must be a non-empty JSON list of trial rows")
    manifest = build_manifest(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest['episodes'])} episodes to {args.output}")
    print(f"tier_counts={manifest['tier_counts']}")


if __name__ == "__main__":
    main()

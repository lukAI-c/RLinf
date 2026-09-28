"""Compare paired frozen-policy Habitat and Genesis episode metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


METRIC_KEYS = (
    "success",
    "spl",
    "ndtw",
    "sdtw",
    "distance_to_goal",
    "path_length",
    "steps_taken",
)


def _scene_id(value):
    scene = str(value or "")
    for prefix in (
        "/root/data/scene_datasets/",
        "/home/clk/workspace/lavira-rft/data/scene_datasets/",
        "/home/nvme01/uni-lavira/data/scene_datasets/",
    ):
        if scene.startswith(prefix):
            return scene[len(prefix):]
    return scene


def _records(path: Path):
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    for key in ("all_episode_metrics", "all_episode_trials", "episodes", "trials", "records"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("%s does not contain an episode record list" % path)


def _by_episode(path: Path):
    result = defaultdict(list)
    for row in _records(path):
        key = (_scene_id(row.get("scene_id")), str(row.get("episode_id")))
        result[key].append(row)
    return dict(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genesis", type=Path, required=True)
    parser.add_argument("--habitat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    genesis = _by_episode(args.genesis)
    habitat = _by_episode(args.habitat)
    rows = []
    common_keys = sorted(
        set(genesis) & set(habitat),
        key=lambda item: (item[0], len(item[1]), item[1]),
    )
    for scene_id, episode_id in common_keys:
        left_trials, right_trials = genesis[(scene_id, episode_id)], habitat[(scene_id, episode_id)]
        for trial_index, (left, right) in enumerate(
            zip(left_trials, right_trials), start=1
        ):
            row = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "trial_index": trial_index,
            }
            for key in METRIC_KEYS:
                if key in left and key in right:
                    row["genesis_%s" % key] = left[key]
                    row["habitat_%s" % key] = right[key]
                    row["delta_%s" % key] = float(right[key]) - float(left[key])
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "genesis_records": sum(len(v) for v in genesis.values()),
                "habitat_records": sum(len(v) for v in habitat.values()),
                "paired_records": len(rows),
                "unpaired_genesis": [
                    {"scene_id": scene, "episode_id": episode}
                    for scene, episode in sorted(set(genesis) - set(habitat))
                ],
                "unpaired_habitat": [
                    {"scene_id": scene, "episode_id": episode}
                    for scene, episode in sorted(set(habitat) - set(genesis))
                ],
                "paired": rows,
            },
            indent=2,
        )
    )
    print("paired records:", len(rows))


if __name__ == "__main__":
    main()

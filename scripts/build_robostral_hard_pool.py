#!/usr/bin/env python3
"""Build an immutable trial-level Robostral hard-pool manifest.

Input must be trial-level frozen-policy rows, not per-episode averages.
Each episode must have exactly five trials unless --allow-other-trial-count.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rlinf.envs.genark.hard_pool import build_hard_pool_manifest
from rlinf.envs.genark.terminal_navigation_score import (
    DEFAULT_CLEAN_STOP_BONUS,
    DEFAULT_DISTANCE_FLOOR_M,
)


def _load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    for key in ("trials", "rows", "episodes"):
        if isinstance(payload.get(key), list):
            # A finished eval dump may already be episode-aggregated. Reject
            # that shape unless each item still looks like a single trial.
            rows = payload[key]
            if rows and "num_trials" in rows[0] and "trials" not in rows[0]:
                raise ValueError(
                    f"{path} looks like per-episode aggregates; pass trial-level rows"
                )
            if rows and isinstance(rows[0].get("trials"), list):
                flattened = []
                for episode in rows:
                    flattened.extend(episode["trials"])
                return flattened
            return rows
    raise ValueError(f"Cannot find trial rows in {path}")


def build_manifest(*args, **kwargs):
    return build_hard_pool_manifest(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-policy-checkpoint", required=True)
    parser.add_argument("--source-policy-fingerprint", required=True)
    parser.add_argument("--distance-floor-m", type=float, default=DEFAULT_DISTANCE_FLOOR_M)
    parser.add_argument("--clean-stop-bonus", type=float, default=DEFAULT_CLEAN_STOP_BONUS)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument("--required-trials", type=int, default=5)
    args = parser.parse_args()

    rows = _load_rows(args.trials)
    manifest = build_manifest(
        rows,
        source_policy_checkpoint=args.source_policy_checkpoint,
        source_policy_fingerprint=args.source_policy_fingerprint,
        distance_floor_m=args.distance_floor_m,
        clean_stop_bonus=args.clean_stop_bonus,
        success_distance=args.success_distance,
        required_trials=args.required_trials,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    buckets: dict[str, int] = defaultdict(int)
    for episode in manifest["episodes"]:
        buckets[episode["bucket"]] += 1
    print(json.dumps({"n_episodes": len(manifest["episodes"]), "buckets": dict(buckets)}))


if __name__ == "__main__":
    main()

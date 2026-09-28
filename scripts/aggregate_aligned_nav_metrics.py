#!/usr/bin/env python3
"""Aggregate Genesis and Habitat navigation evaluation records identically.

The two simulators write slightly different raw records, but the evaluation
contract is shared here:

* ``success`` / ``distance_success`` means final distance <= success distance;
* ``habitat_success`` is an optional simulator-native metric;
* ``all_episode_metrics.json`` always contains raw trial rows;
* ``per_episode_summary.json`` contains one mean row per episode;
* ``avg_metrics.json`` is the direct mean over raw trials.

This module is intentionally an orchestration utility.  It does not alter
policy actions or simulator termination behavior.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


NUMERIC_KEYS = (
    "success",
    "distance_success",
    "habitat_success",
    "oracle_success",
    "spl",
    "ndtw",
    "sdtw",
    "distance_to_goal",
    "path_length",
    "steps_taken",
    "decision_count",
    "primitive_step_count",
    "parse_fail_count",
    "dtg_progress",
)

DIAGNOSTIC_NUMERIC_KEYS = (
    "start_distance_to_goal",
    "stop_step",
    "stop_dtg",
    "min_dtg",
    "best_dtg_progress",
    "final_regression",
    "early_stop",
    "reward_sum",
)
ALL_NUMERIC_KEYS = NUMERIC_KEYS + DIAGNOSTIC_NUMERIC_KEYS
OPTIONAL_NUMERIC_KEYS = {
    "habitat_success",
    "decision_count",
    "primitive_step_count",
    *DIAGNOSTIC_NUMERIC_KEYS,
}


def _number(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def _episode_key(row: dict[str, Any]) -> tuple[str, str]:
    return _scene_id(row.get("scene_id", "")), str(row.get("episode_id", ""))


def _scene_id(value: Any) -> str:
    """Convert host/container scene paths to the dataset-relative id."""
    scene = str(value or "")
    prefixes = (
        "/root/data/scene_datasets/",
        "/home/clk/workspace/lavira-rft/data/scene_datasets/",
        "/home/nvme01/uni-lavira/data/scene_datasets/",
    )
    for prefix in prefixes:
        if scene.startswith(prefix):
            scene = scene[len(prefix):]
            break
    return scene


def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw trial while preserving simulator-specific fields."""
    result = dict(row)
    result["scene_id"] = _scene_id(result.get("scene_id", ""))

    # Both evaluators use final-distance success.  Habitat additionally keeps
    # its native success in habitat_success for a strict simulator comparison.
    distance_success = result.get("distance_success", result.get("success"))
    if result.get("success") is None and distance_success is not None:
        result["success"] = distance_success
    if result.get("distance_success") is None and result.get("success") is not None:
        result["distance_success"] = result["success"]

    for key in NUMERIC_KEYS:
        if key in result:
            value = _number(result[key], None if key in OPTIONAL_NUMERIC_KEYS else 0.0)
            if value is not None:
                result[key] = value
    for key in DIAGNOSTIC_NUMERIC_KEYS:
        if key in result:
            value = _number(result[key], None)
            if value is not None:
                result[key] = value
    # Keep an identical JSON schema for Genesis (no Habitat-native success)
    # and Habitat (no Genesis reward diagnostics).
    for key in OPTIONAL_NUMERIC_KEYS:
        result.setdefault(key, None)

    result.setdefault("n_trials", 1)
    result.setdefault("success_rate", result.get("success", 0.0))
    return result


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("episodes", payload.get("trials", []))
    else:
        rows = []
    if not isinstance(rows, list):
        raise ValueError("evaluation payload must contain a list of episodes/trials")
    return [_normalise_row(row) for row in rows if isinstance(row, dict)]


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(sum(float(value) for value in values) / len(values))


def _aggregate_episode(key: tuple[str, str], trials: list[dict[str, Any]]) -> dict[str, Any]:
    scene_id, episode_id = key
    episode: dict[str, Any] = {
        "scene_id": scene_id,
        "episode_id": episode_id,
        "n_trials": len(trials),
    }
    for metric in ALL_NUMERIC_KEYS:
        value = _mean(trials, metric)
        if value is not None:
            episode[metric] = value
    episode["success_rate"] = float(episode.get("success", 0.0))

    # Retain termination information in the episode summary.  This is a count
    # when an episode has repeated stochastic trials, and remains useful for
    # Genesis records that already carry the full terminal diagnosis.
    for field in ("termination_cause", "success_type"):
        counts = Counter(str(row[field]) for row in trials if row.get(field) not in (None, ""))
        if counts:
            episode[f"{field}_counts"] = dict(sorted(counts.items()))
            episode[field] = counts.most_common(1)[0][0]

    return episode


def aggregate_rows(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    expected_episodes: int | None = None,
    expected_keys: set[tuple[str, str]] | None = None,
    source: str = "unknown",
    max_episode_steps: int | None = None,
) -> dict[str, Any]:
    rows = [_normalise_row(row) for row in rows]
    if not rows:
        raise ValueError("no evaluation rows found")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_episode_key(row)].append(row)

    actual_keys = set(grouped)
    if expected_keys is not None and actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            f"evaluation episode set mismatch: missing={missing[:10]} extra={extra[:10]}"
        )
    if expected_episodes is not None and len(actual_keys) != expected_episodes:
        raise RuntimeError(
            f"evaluation incomplete: expected {expected_episodes} unique episodes, "
            f"got {len(actual_keys)} from {len(rows)} trials"
        )

    episodes = [
        _aggregate_episode(key, grouped[key])
        for key in sorted(grouped)
    ]
    scene_trials: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene_trials[str(row.get("scene_id", ""))].append(row)
    per_scene = []
    for scene_id in sorted(scene_trials):
        scene_episode_rows = [episode for episode in episodes if episode["scene_id"] == scene_id]
        summary = {
            metric: _mean(scene_trials[scene_id], metric)
            for metric in ALL_NUMERIC_KEYS
        }
        per_scene.append({
            "scene_id": scene_id,
            "n_episodes": len(scene_episode_rows),
            "n_trials": len(scene_trials[scene_id]),
            "summary": summary,
            "termination_cause_counts": dict(Counter(
                str(row.get("termination_cause"))
                for row in scene_trials[scene_id]
                if row.get("termination_cause") not in (None, "")
            )),
            "success_type_counts": dict(Counter(
                str(row.get("success_type"))
                for row in scene_trials[scene_id]
                if row.get("success_type") not in (None, "")
            )),
        })

    averages = {
        metric: _mean(rows, metric)
        for metric in ALL_NUMERIC_KEYS
    }
    averages.update({
        "num_episodes": len(episodes),
        "num_scenes": len(per_scene),
        "n_trials": len(rows),
        "eval_repeats_per_episode": max(
            int(episode["n_trials"]) for episode in episodes
        ),
        "source": source,
    })
    if max_episode_steps is not None:
        averages["max_episode_steps"] = int(max_episode_steps)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "all_episode_metrics.json").write_text(json.dumps(rows, indent=2))
    (root / "per_episode_summary.json").write_text(json.dumps(episodes, indent=2))
    (root / "per_scene_metrics.json").write_text(json.dumps(per_scene, indent=2))
    (root / "avg_metrics.json").write_text(json.dumps(averages, indent=2))
    (root / "metrics_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "source": source,
        "success_definition": "final distance_to_goal <= success_distance",
        "habitat_success_definition": "Habitat-native success when available",
        "aggregation": "direct mean over raw trials",
        "n_episodes": len(episodes),
        "n_trials": len(rows),
        "n_scenes": len(per_scene),
        "max_episode_steps": max_episode_steps,
    }, indent=2))
    (root / "combined_metrics.json").write_text(json.dumps({
        "combined": averages,
        "per_scene": per_scene,
    }, indent=2))
    return averages


def aggregate_files(
    input_files: Iterable[str | Path],
    output_dir: str | Path,
    expected_episodes: int | None = None,
    source: str = "unknown",
    max_episode_steps: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in input_files:
        with Path(path).open() as handle:
            rows.extend(_rows_from_payload(json.load(handle)))
    return aggregate_rows(
        rows,
        output_dir,
        expected_episodes=expected_episodes,
        source=source,
        max_episode_steps=max_episode_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--source", default="unknown")
    parser.add_argument("--max-episode-steps", type=int)
    args = parser.parse_args()
    avg = aggregate_files(
        args.input_files,
        args.output_dir,
        expected_episodes=args.expected_episodes,
        source=args.source,
        max_episode_steps=args.max_episode_steps,
    )
    print(json.dumps(avg, indent=2))


if __name__ == "__main__":
    main()

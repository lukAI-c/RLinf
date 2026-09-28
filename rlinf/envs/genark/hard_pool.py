"""Trial-level hard-pool manifest construction."""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from rlinf.envs.genark.terminal_navigation_score import (
    DISTANCE_CONTRACT,
    HARD_POOL_TRAINING_BUCKETS,
    classify_hard_pool_bucket,
    compute_terminal_navigation_score,
    diagnostic_is_clean_stop,
    reward_contract_dict,
)


def build_hard_pool_manifest(
    rows: list[dict],
    *,
    source_policy_checkpoint: str,
    source_policy_fingerprint: str,
    distance_floor_m: float,
    clean_stop_bonus: float,
    success_distance: float,
    required_trials: int,
) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        scene = str(row.get("scene_id") or "")
        episode = str(row.get("episode_id") or "")
        if not scene or not episode:
            raise ValueError(f"trial row missing scene_id/episode_id: {row!r}")
        grouped[(scene, episode)].append(row)

    episodes = []
    for (scene_id, episode_id), trials in sorted(grouped.items()):
        if len(trials) != required_trials:
            raise ValueError(
                f"{scene_id} ep={episode_id} has {len(trials)} trials, "
                f"expected {required_trials}"
            )
        scores = []
        dtgs = []
        clean_flags = []
        progresses = []
        for trial in trials:
            dtg = float(trial.get("distance_to_goal", trial.get("final_dtg")))
            if not math.isfinite(dtg):
                raise ValueError(
                    f"non-finite Euclidean DTG for {scene_id} ep={episode_id}"
                )
            clean = diagnostic_is_clean_stop(trial, success_distance=success_distance)
            score = compute_terminal_navigation_score(
                dtg,
                clean,
                distance_floor_m=distance_floor_m,
                clean_stop_bonus=clean_stop_bonus,
            )
            scores.append(score)
            dtgs.append(dtg)
            clean_flags.append(clean)
            start_dtg = trial.get("start_distance_to_goal", trial.get("start_dtg"))
            if start_dtg is not None and math.isfinite(float(start_dtg)):
                progresses.append(float(start_dtg) - dtg)
        score_t = torch.tensor(scores, dtype=torch.float32)
        bucket = classify_hard_pool_bucket(
            clean_stop_count=int(sum(clean_flags)),
            terminal_score_std=float(score_t.std(unbiased=True).item()),
            max_final_euclidean_dtg_m=float(max(dtgs)),
        )
        episodes.append(
            {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "bucket": bucket,
                "num_trials": len(trials),
                "clean_stop_count": int(sum(clean_flags)),
                "proximity_without_stop_count": sum(
                    (not clean)
                    and dtg < success_distance
                    and str(trial.get("termination_cause", "")) != "stop"
                    for trial, clean, dtg in zip(trials, clean_flags, dtgs)
                ),
                "wrong_stop_count": sum(
                    str(trial.get("termination_cause", "")) == "stop" and not clean
                    for trial, clean in zip(trials, clean_flags)
                ),
                "mean_final_dtg": float(sum(dtgs) / len(dtgs)),
                "max_final_dtg": float(max(dtgs)),
                "terminal_score_mean": float(score_t.mean().item()),
                "terminal_score_std": float(score_t.std(unbiased=True).item()),
                "mean_dtg_progress": (
                    float(sum(progresses) / len(progresses)) if progresses else 0.0
                ),
                "in_training_pool": bucket in HARD_POOL_TRAINING_BUCKETS,
            }
        )
    return {
        "schema_version": 1,
        "source_policy_checkpoint": source_policy_checkpoint,
        "source_policy_fingerprint": source_policy_fingerprint,
        "distance_contract": DISTANCE_CONTRACT,
        "reward_contract": reward_contract_dict(
            distance_floor_m=distance_floor_m,
            clean_stop_bonus=clean_stop_bonus,
        ),
        "episodes": episodes,
    }

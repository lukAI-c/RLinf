"""Robostral-inspired Euclidean endpoint navigation score.

Robostral Navigate (arXiv 2607.20785v3, §3.3) uses
``-max(2, geodesic_dist_to_goal)`` as a terminal reward. GenArk does not
expose a geodesic query, so this module uses the Habitat-coordinate Euclidean
``distance_to_goal`` that ``genark_env`` already logs.

The training objective is a single endpoint score. Clean STOP remains in the
model output and env termination protocol, but is an evaluation metric, not a
reward term. ``clean_stop_bonus`` defaults to 0. A nonzero bonus is retained
only as an opt-in ablation.

This file is the single source of truth for:

- the scalar score
- terminal-GRPO advantages
- hard-pool bucket classification
- launcher / runtime contract checks
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

DISTANCE_CONTRACT = "genark_habitat_coordinate_euclidean_v1"
DEFAULT_DISTANCE_FLOOR_M = 2.0
DEFAULT_CLEAN_STOP_BONUS = 0.0
TERMINAL_GRPO_STD_EPS = 1e-6


def compute_terminal_navigation_score(
    final_euclidean_dtg_m: float,
    clean_stop: bool,
    *,
    distance_floor_m: float = DEFAULT_DISTANCE_FLOOR_M,
    clean_stop_bonus: float = DEFAULT_CLEAN_STOP_BONUS,
) -> float:
    """Return ``-max(floor, DTG) + bonus * clean_stop``.

    Training uses ``bonus=0``. Final Euclidean DTG is the only learning
    signal: reach the floor region and stay there. ``clean_stop`` still
    affects the score only when an ablation sets a nonzero bonus.
    """
    if not math.isfinite(float(final_euclidean_dtg_m)):
        raise ValueError(
            "terminal navigation score requires a finite Euclidean "
            f"distance_to_goal, got {final_euclidean_dtg_m!r}"
        )
    if not math.isfinite(float(distance_floor_m)) or float(distance_floor_m) < 0:
        raise ValueError(f"distance_floor_m must be a finite >= 0, got {distance_floor_m!r}")
    if not math.isfinite(float(clean_stop_bonus)):
        raise ValueError(f"clean_stop_bonus must be finite, got {clean_stop_bonus!r}")
    score = -max(float(distance_floor_m), float(final_euclidean_dtg_m))
    if clean_stop:
        score += float(clean_stop_bonus)
    return float(score)


def score_from_completed_diagnostic(
    diag: Mapping[str, Any],
    *,
    distance_floor_m: float = DEFAULT_DISTANCE_FLOOR_M,
    clean_stop_bonus: float = DEFAULT_CLEAN_STOP_BONUS,
    success_distance: float = 3.0,
) -> float:
    """Score one completed-episode diagnostic using the Euclidean DTG field."""
    dtg = diag.get("distance_to_goal", diag.get("final_dtg", None))
    if dtg is None:
        raise ValueError(
            "completed diagnostic is missing Euclidean distance_to_goal: "
            f"scene={diag.get('scene_id')} episode={diag.get('episode_id')} "
            f"trial={diag.get('trial_index', diag.get('trial_id'))} "
            f"env={diag.get('env_id')}"
        )
    dtg_f = float(dtg)
    if not math.isfinite(dtg_f):
        raise ValueError(
            "completed diagnostic has a non-finite Euclidean distance_to_goal: "
            f"scene={diag.get('scene_id')} episode={diag.get('episode_id')} "
            f"trial={diag.get('trial_index', diag.get('trial_id'))} "
            f"env={diag.get('env_id')} dtg={dtg!r}"
        )
    clean = diagnostic_is_clean_stop(diag, success_distance=success_distance)
    return compute_terminal_navigation_score(
        dtg_f,
        clean,
        distance_floor_m=distance_floor_m,
        clean_stop_bonus=clean_stop_bonus,
    )


def diagnostic_is_clean_stop(
    diag: Mapping[str, Any],
    *,
    success_distance: float = 3.0,
) -> bool:
    """Return the fail-closed clean-STOP label from a completed diagnostic."""
    stored = diag.get("clean_stop_success")
    if stored is not None:
        return float(stored) > 0.5
    cause = str(diag.get("termination_cause", diag.get("cause", "")))
    dtg = diag.get("distance_to_goal", diag.get("final_dtg"))
    if dtg is None:
        return False
    dtg_f = float(dtg)
    return bool(cause == "stop" and math.isfinite(dtg_f) and dtg_f < float(success_distance))


def classify_replay_group(
    scores: Sequence[float],
    clean_stops: Sequence[bool],
    final_dtgs: Sequence[float],
    *,
    distance_floor_m: float = DEFAULT_DISTANCE_FLOOR_M,
    std_eps: float = TERMINAL_GRPO_STD_EPS,
) -> str:
    """Label one offline-replay group. Exclusive, report-only taxonomy."""
    k = int(sum(bool(flag) for flag in clean_stops))
    n = len(scores)
    if n == 0:
        return "empty"
    if 1 <= k <= n - 1:
        return "mixed_stop_support"
    if k == 0 and all(float(dtg) <= float(distance_floor_m) for dtg in final_dtgs):
        return "flat_missed_stop"
    std = _sample_std(scores)
    if std > float(std_eps):
        return "navigation_informative"
    return "uninformative"


def classify_hard_pool_bucket(
    *,
    clean_stop_count: int,
    terminal_score_std: float,
    max_final_euclidean_dtg_m: float,
) -> str:
    """Assign exactly one hard-pool bucket. Ordered if/elif is the contract."""
    if clean_stop_count >= 4:
        return "mastered"
    if 1 <= clean_stop_count <= 3 and terminal_score_std >= 0.1:
        return "mixed_support"
    if clean_stop_count == 0 and max_final_euclidean_dtg_m <= 2.0:
        return "flat_missed_stop"
    if (
        clean_stop_count == 0
        and terminal_score_std >= 0.5
        and max_final_euclidean_dtg_m > 2.0
    ):
        return "navigation_hard"
    if clean_stop_count == 0 and terminal_score_std < 0.1:
        return "uninformative"
    return "deferred"


HARD_POOL_TRAINING_BUCKETS = frozenset({"mixed_support", "navigation_hard"})


def compute_terminal_grpo_outcome_advantages(
    episode_terminal_score: torch.Tensor,
    group_size: int,
    *,
    episode_ids: Sequence[Any] | None = None,
    scene_ids: Sequence[Any] | None = None,
    std_eps: float = TERMINAL_GRPO_STD_EPS,
) -> torch.Tensor:
    """Return per-trajectory GRPO advantages from terminal scores.

    ``A_i = (S_i - mean(S)) / (sample_std(S) + 1e-6)``. Groups whose sample
    standard deviation is below ``std_eps`` receive zero advantage.
    """
    if group_size < 2:
        raise ValueError("decision_terminal_grpo requires group_size >= 2")
    scores = episode_terminal_score.reshape(-1).to(dtype=torch.float32)
    if scores.numel() % group_size != 0:
        raise ValueError(
            "decision_terminal_grpo requires batch="
            f"{scores.numel()} divisible by group_size={group_size}"
        )
    if not torch.isfinite(scores).all():
        raise ValueError("decision_terminal_grpo received a non-finite terminal score")
    _validate_same_episode_group(episode_ids, scene_ids, group_size)
    grouped = scores.reshape(-1, group_size)
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, unbiased=True, keepdim=True)
    centered = grouped - mean
    advantages = centered / (std + float(std_eps))
    advantages = torch.where(
        std < float(std_eps),
        torch.zeros_like(advantages),
        advantages,
    )
    return advantages.reshape_as(scores)


def _validate_same_episode_group(
    episode_ids: Sequence[Any] | None,
    scene_ids: Sequence[Any] | None,
    group_size: int,
) -> None:
    if episode_ids is None and scene_ids is None:
        return
    n = None
    if episode_ids is not None:
        n = len(list(episode_ids))
    if scene_ids is not None:
        scene_n = len(list(scene_ids))
        if n is None:
            n = scene_n
        elif scene_n != n:
            raise ValueError(
                "episode_ids and scene_ids must have the same length: "
                f"{n} vs {scene_n}"
            )
    if n is None or n % group_size != 0:
        raise ValueError(
            "terminal-GRPO identity lists must be divisible by "
            f"group_size={group_size}, got n={n}"
        )
    for start in range(0, n, group_size):
        end = start + group_size
        if episode_ids is not None:
            unique_eps = {str(item) for item in list(episode_ids)[start:end]}
            if len(unique_eps) != 1:
                raise ValueError(
                    "decision_terminal_grpo mixed episode IDs in one group: "
                    f"{sorted(unique_eps)}"
                )
        if scene_ids is not None:
            unique_scenes = {str(item) for item in list(scene_ids)[start:end]}
            if len(unique_scenes) != 1:
                raise ValueError(
                    "decision_terminal_grpo mixed scene IDs in one group: "
                    f"{sorted(unique_scenes)}"
                )


def _sample_std(values: Sequence[float]) -> float:
    tensor = torch.tensor([float(v) for v in values], dtype=torch.float32)
    if tensor.numel() < 2:
        return 0.0
    return float(tensor.std(unbiased=True).item())


def freeze_completed_episode_outcome(
    result: Any,
    completed_diag: Mapping[str, Any] | None,
    *,
    distance_floor_m: float = DEFAULT_DISTANCE_FLOOR_M,
    clean_stop_bonus: float = DEFAULT_CLEAN_STOP_BONUS,
    success_distance: float = 3.0,
) -> bool:
    """Set ``episode_success`` and ``episode_terminal_score`` atomically.

    Both tensors are written together or neither is written. A half-frozen
    result is an error, not a silent repair.
    """
    has_success = getattr(result, "episode_success", None) is not None
    has_score = getattr(result, "episode_terminal_score", None) is not None
    if has_success and has_score:
        return True
    if has_success != has_score:
        raise RuntimeError(
            "partially frozen episode outcome: "
            f"episode_success={'set' if has_success else 'missing'} "
            f"episode_terminal_score={'set' if has_score else 'missing'} "
            f"scene={None if completed_diag is None else completed_diag.get('scene_id')} "
            f"episode={None if completed_diag is None else completed_diag.get('episode_id')} "
            f"trial={None if completed_diag is None else completed_diag.get('trial_index', completed_diag.get('trial_id'))} "
            f"env={None if completed_diag is None else completed_diag.get('env_id')}"
        )
    if not isinstance(completed_diag, Mapping):
        return False
    score = score_from_completed_diagnostic(
        completed_diag,
        distance_floor_m=distance_floor_m,
        clean_stop_bonus=clean_stop_bonus,
        success_distance=success_distance,
    )
    clean = diagnostic_is_clean_stop(
        completed_diag, success_distance=success_distance
    )
    result.episode_success = torch.tensor([[[float(clean)]]], dtype=torch.float32)
    result.episode_terminal_score = torch.tensor([[[float(score)]]], dtype=torch.float32)
    if "terminal_navigation_score" not in completed_diag:
        # Diagnostics are treated as read-mostly. Copy-on-write keeps the
        # authoritative score next to the Euclidean DTG that produced it.
        if hasattr(completed_diag, "__setitem__"):
            completed_diag["terminal_navigation_score"] = float(score)
    return True


def reward_contract_dict(
    *,
    distance_floor_m: float = DEFAULT_DISTANCE_FLOOR_M,
    clean_stop_bonus: float = DEFAULT_CLEAN_STOP_BONUS,
) -> dict[str, float]:
    return {
        "distance_floor_m": float(distance_floor_m),
        "clean_stop_bonus": float(clean_stop_bonus),
    }


def contracts_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    atol: float = 1e-8,
) -> bool:
    try:
        return (
            abs(float(left["distance_floor_m"]) - float(right["distance_floor_m"])) <= atol
            and abs(float(left["clean_stop_bonus"]) - float(right["clean_stop_bonus"]))
            <= atol
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate_robostral_rft_contract(
    cfg: Any,
    *,
    manifest: Mapping[str, Any] | None = None,
    loaded_policy_fingerprint: str | None = None,
) -> list[str]:
    """Return human-readable contract violations. Empty list means pass."""
    errors: list[str] = []
    env_train = _cfg_get(cfg, "env", "train") or {}
    algorithm = _cfg_get(cfg, "algorithm") or {}
    adv_type = str(_mapping_get(algorithm, "adv_type", "")).lower()
    hard_pool = _mapping_get(env_train, "hard_pool")
    curriculum = _mapping_get(env_train, "episode_curriculum")
    assignment = _mapping_get(env_train, "episode_assignment")
    score_cfg = _mapping_get(env_train, "terminal_navigation_score")
    hard_enabled = bool(_mapping_get(hard_pool, "enabled", False))
    curr_enabled = bool(_mapping_get(curriculum, "enabled", False))
    assign_enabled = bool(_mapping_get(assignment, "enabled", False))
    score_enabled = bool(_mapping_get(score_cfg, "enabled", False))

    if hard_enabled and curr_enabled:
        errors.append("hard_pool and episode_curriculum cannot both be enabled")
    if assign_enabled and curr_enabled:
        errors.append("episode_assignment and episode_curriculum cannot both be enabled")
    if assign_enabled and hard_enabled:
        errors.append("episode_assignment and hard_pool cannot both be enabled")

    group_size = int(_mapping_get(algorithm, "group_size", 0) or 0)
    env_group = int(_mapping_get(env_train, "group_size", 0) or 0)
    total_envs = int(_mapping_get(env_train, "total_num_envs", 0) or 0)
    requires_same_episode_group = (
        adv_type == "decision_terminal_grpo" or hard_enabled or assign_enabled
    )
    if requires_same_episode_group:
        if group_size < 2:
            errors.append(
                f"group_size must be >= 2, got algorithm.group_size={group_size}"
            )
        if env_group and group_size and env_group != group_size:
            errors.append(
                f"env.train.group_size={env_group} != algorithm.group_size={group_size}"
            )
        if total_envs and group_size and total_envs % group_size != 0:
            errors.append(
                f"total_num_envs={total_envs} is not divisible by group_size={group_size}"
            )

    if adv_type == "decision_terminal_grpo":
        if not score_enabled:
            errors.append(
                "decision_terminal_grpo requires "
                "env.train.terminal_navigation_score.enabled=true"
            )
        if float(_mapping_get(env_train, "reference_path_progress_coef", 0.0) or 0.0) != 0.0:
            errors.append("decision_terminal_grpo forbids a nonzero reference-path coefficient")
        if bool(_mapping_get(env_train, "reference_path_reward_enabled", False)):
            errors.append("decision_terminal_grpo forbids reference_path_reward_enabled")
        if bool(_mapping_get(env_train, "missed_stop_aux_reward_enabled", False)):
            errors.append("decision_terminal_grpo forbids missed_stop_aux_reward_enabled")
        if float(_mapping_get(algorithm, "rloo_aux_coef", 0.0) or 0.0) != 0.0:
            errors.append("decision_terminal_grpo forbids a nonzero rloo_aux_coef")
        if bool(_mapping_get(env_train, "process_reward_enabled", False)):
            errors.append("decision_terminal_grpo forbids process_reward_enabled")
        if bool(_mapping_get(env_train, "nav_terminal_reward_enabled", False)):
            errors.append("decision_terminal_grpo forbids nav_terminal_reward_enabled")

    if manifest is not None:
        expected = reward_contract_dict(
            distance_floor_m=float(
                _mapping_get(score_cfg, "distance_floor_m", DEFAULT_DISTANCE_FLOOR_M)
            ),
            clean_stop_bonus=float(
                _mapping_get(score_cfg, "clean_stop_bonus", DEFAULT_CLEAN_STOP_BONUS)
            ),
        )
        if not contracts_match(manifest.get("reward_contract") or {}, expected):
            errors.append(
                "manifest reward_contract does not match the configured score contract"
            )
        if loaded_policy_fingerprint:
            source = str(manifest.get("source_policy_fingerprint") or "")
            if source and source != str(loaded_policy_fingerprint):
                errors.append(
                    "manifest source_policy_fingerprint does not match the "
                    "loaded policy weights"
                )
    return errors


def _cfg_get(cfg: Any, *keys: str) -> Any:
    cur = cfg
    for key in keys:
        cur = _mapping_get(cur, key)
        if cur is None:
            return None
    return cur


def _mapping_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)

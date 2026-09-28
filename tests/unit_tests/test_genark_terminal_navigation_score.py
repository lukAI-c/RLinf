import torch

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult
from rlinf.envs.genark.terminal_navigation_score import (
    classify_hard_pool_bucket,
    classify_replay_group,
    compute_terminal_navigation_score,
    freeze_completed_episode_outcome,
    validate_robostral_rft_contract,
)


def test_timeout_beyond_floor_is_negative_dtg():
    assert compute_terminal_navigation_score(5.0, False) == -5.0


def test_timeout_inside_floor_clips_to_minus_two():
    assert compute_terminal_navigation_score(1.5, False) == -2.0


def test_clean_stop_does_not_change_endpoint_score_by_default():
    assert compute_terminal_navigation_score(1.5, True) == -2.0
    assert compute_terminal_navigation_score(1.5, False) == -2.0


def test_opt_in_clean_stop_bonus_still_adds_one_meter():
    assert compute_terminal_navigation_score(
        1.5, True, clean_stop_bonus=1.0
    ) == -1.0


def test_wrong_stop_keeps_distance_penalty():
    assert compute_terminal_navigation_score(5.0, False) == -5.0
    assert compute_terminal_navigation_score(5.0, True) == -5.0


def test_clean_and_missed_stop_inside_floor_share_endpoint_score():
    missed = compute_terminal_navigation_score(1.5, False)
    clean = compute_terminal_navigation_score(1.5, True)
    assert missed == -2.0
    assert clean == -2.0
    assert compute_terminal_navigation_score(
        1.5, True, clean_stop_bonus=1.0
    ) != missed


def test_both_freeze_paths_write_identical_tensors():
    diag = {
        "scene_id": "Z6",
        "episode_id": "586",
        "trial_index": 1,
        "env_id": 0,
        "distance_to_goal": 1.5,
        "clean_stop_success": 1.0,
        "termination_cause": "stop",
    }
    flush_result = EmbodiedRolloutResult()
    fallback_result = EmbodiedRolloutResult()

    assert freeze_completed_episode_outcome(flush_result, diag) is True
    assert freeze_completed_episode_outcome(fallback_result, dict(diag)) is True
    assert torch.equal(flush_result.episode_success, fallback_result.episode_success)
    assert torch.equal(
        flush_result.episode_terminal_score, fallback_result.episode_terminal_score
    )
    assert float(flush_result.episode_success.item()) == 1.0
    assert float(flush_result.episode_terminal_score.item()) == -2.0


def test_partially_frozen_outcome_raises_atomicity_error():
    result = EmbodiedRolloutResult(
        episode_success=torch.tensor([[[1.0]]]),
    )
    try:
        freeze_completed_episode_outcome(
            result,
            {
                "distance_to_goal": 1.5,
                "clean_stop_success": 1.0,
                "episode_id": "586",
                "env_id": 0,
            },
        )
    except RuntimeError as exc:
        assert "partially frozen" in str(exc)
    else:
        raise AssertionError("expected atomicity error")


def test_terminal_score_survives_second_freeze_without_another_diag():
    result = EmbodiedRolloutResult()
    diag = {
        "distance_to_goal": 5.0,
        "clean_stop_success": 0.0,
        "termination_cause": "timeout",
        "episode_id": "824",
        "env_id": 3,
    }
    assert freeze_completed_episode_outcome(result, diag) is True
    first_score = result.episode_terminal_score.clone()
    assert freeze_completed_episode_outcome(result, None) is True
    assert torch.equal(result.episode_terminal_score, first_score)


def test_hard_pool_classifier_is_exclusive_and_deterministic():
    assert classify_hard_pool_bucket(
        clean_stop_count=4, terminal_score_std=1.31, max_final_euclidean_dtg_m=1.0
    ) == "mastered"
    assert classify_hard_pool_bucket(
        clean_stop_count=2, terminal_score_std=1.31, max_final_euclidean_dtg_m=1.0
    ) == "mixed_support"
    assert classify_hard_pool_bucket(
        clean_stop_count=0, terminal_score_std=0.0, max_final_euclidean_dtg_m=1.5
    ) == "flat_missed_stop"
    assert classify_hard_pool_bucket(
        clean_stop_count=0, terminal_score_std=0.8, max_final_euclidean_dtg_m=5.0
    ) == "navigation_hard"
    assert classify_hard_pool_bucket(
        clean_stop_count=0, terminal_score_std=0.01, max_final_euclidean_dtg_m=8.0
    ) == "uninformative"
    assert classify_hard_pool_bucket(
        clean_stop_count=2, terminal_score_std=0.01, max_final_euclidean_dtg_m=4.0
    ) == "deferred"


def test_replay_group_labels_do_not_confuse_stop_with_navigation():
    assert (
        classify_replay_group([-5.0, -4.0, -3.0, -6.0], [False] * 4, [5.0, 4.0, 3.0, 6.0])
        == "navigation_informative"
    )
    assert (
        classify_replay_group([-1.0, -2.0, -2.0, -2.0], [True, False, False, False], [1.0, 1.5, 1.2, 0.8])
        == "mixed_stop_support"
    )
    assert (
        classify_replay_group([-2.0] * 4, [False] * 4, [1.5, 1.2, 0.8, 1.9])
        == "flat_missed_stop"
    )


def test_contract_rejects_curriculum_plus_hard_pool_and_aux_rewards():
    errors = validate_robostral_rft_contract(
        {
            "env": {
                "train": {
                    "hard_pool": {"enabled": True},
                    "episode_curriculum": {"enabled": True},
                    "terminal_navigation_score": {"enabled": True},
                    "reference_path_reward_enabled": True,
                    "group_size": 8,
                    "total_num_envs": 8,
                }
            },
            "algorithm": {
                "adv_type": "decision_terminal_grpo",
                "group_size": 8,
                "rloo_aux_coef": 0.5,
            },
        }
    )
    joined = " ".join(errors)
    assert "hard_pool and episode_curriculum" in joined
    assert "reference_path_reward_enabled" in joined
    assert "rloo_aux_coef" in joined

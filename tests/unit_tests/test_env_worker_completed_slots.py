import numpy as np
import torch

import pytest

from rlinf.workers.env.env_worker import (
    _mask_completed_episode_observations,
    _validate_decision_rloo_terminal_group,
    _validate_decision_terminal_grpo_group,
)


def test_completed_slots_stay_dormant_after_group_reset():
    obs = {
        "task_descriptions": ["old reset trial", "still running", "new reset trial"],
        "episode_active": torch.tensor([True, True, True]),
        "states": torch.arange(12).reshape(3, 4),
    }

    masked = _mask_completed_episode_observations(
        obs, torch.tensor([True, False, True])
    )

    assert masked["task_descriptions"] == ["", "still running", ""]
    assert masked["episode_active"].tolist() == [False, True, False]
    assert torch.equal(masked["states"], obs["states"])
    assert obs["task_descriptions"] == [
        "old reset trial", "still running", "new reset trial"
    ]
    assert obs["episode_active"].tolist() == [True, True, True]


def test_completed_slot_mask_supports_numpy_activity():
    masked = _mask_completed_episode_observations(
        {
            "task_descriptions": ["done", "active"],
            "episode_active": np.asarray([True, True]),
        },
        torch.tensor([True, False]),
    )

    assert masked["task_descriptions"] == ["", "active"]
    assert masked["episode_active"].tolist() == [False, True]


def _terminal_diag(env_i: int, *, generation: int = 3, clean: bool = False):
    return {
        "episode_id": "609",
        "trial_index": env_i + 1,
        "rft_group_id": 0,
        "rft_group_generation": generation,
        "clean_stop_success": float(clean),
    }


def test_decision_rloo_terminal_group_accepts_one_complete_generation():
    diagnostics = [
        _terminal_diag(env_i, clean=env_i == 0) for env_i in range(8)
    ]

    _validate_decision_rloo_terminal_group(
        torch.tensor([1.0] + [0.0] * 7),
        diagnostics,
        expected_size=8,
        stage_id=0,
    )


def test_decision_rloo_terminal_group_rejects_partial_or_mixed_data():
    diagnostics = [_terminal_diag(env_i) for env_i in range(8)]
    with pytest.raises(RuntimeError, match="incomplete"):
        _validate_decision_rloo_terminal_group(
            torch.zeros(7), diagnostics[:7], expected_size=8, stage_id=0
        )

    diagnostics[-1] = _terminal_diag(7, generation=4)
    with pytest.raises(RuntimeError, match="mixed terminal groups"):
        _validate_decision_rloo_terminal_group(
            torch.zeros(8), diagnostics, expected_size=8, stage_id=0
        )


def test_decision_rloo_terminal_group_rejects_outcome_mismatch():
    diagnostics = [
        _terminal_diag(env_i, clean=env_i == 0) for env_i in range(8)
    ]

    with pytest.raises(RuntimeError, match="outcome disagrees"):
        _validate_decision_rloo_terminal_group(
            torch.zeros(8), diagnostics, expected_size=8, stage_id=0
        )


def _terminal_score_diag(env_i: int, *, episode_id: str = "609", dtg: float = 5.0, clean: bool = False):
    return {
        "episode_id": episode_id,
        "scene_id": "mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
        "trial_index": env_i + 1,
        "rft_group_id": 0,
        "rft_group_generation": 3,
        "clean_stop_success": float(clean),
        "distance_to_goal": dtg,
        "termination_cause": "stop" if clean else "timeout",
        "env_id": env_i,
    }


def test_decision_terminal_grpo_group_accepts_same_episode_scores():
    diagnostics = [_terminal_score_diag(env_i, dtg=5.0 - env_i * 0.2) for env_i in range(8)]
    scores = torch.tensor([[[-5.0 + env_i * 0.2] for env_i in range(8)]])
    success = torch.zeros(1, 8, 1)
    _validate_decision_terminal_grpo_group(
        success,
        scores,
        diagnostics,
        expected_size=8,
        stage_id=0,
    )


def test_decision_terminal_grpo_group_rejects_mixed_episodes():
    diagnostics = [_terminal_score_diag(env_i) for env_i in range(8)]
    diagnostics[-1]["episode_id"] = "824"
    with pytest.raises(RuntimeError, match="mixed or missing episode"):
        _validate_decision_terminal_grpo_group(
            torch.zeros(8),
            torch.full((8,), -5.0),
            diagnostics,
            expected_size=8,
            stage_id=0,
        )


def test_decision_terminal_grpo_accepts_two_same_episode_groups():
    diagnostics = []
    scores = []
    success = []
    for group_i, episode_id in enumerate(("824", "447")):
        for env_i in range(8):
            dtg = 5.0 - env_i * 0.2
            diagnostics.append(
                _terminal_score_diag(
                    env_i,
                    episode_id=episode_id,
                    dtg=dtg,
                )
            )
            diagnostics[-1]["rft_group_id"] = group_i
            diagnostics[-1]["trial_index"] = env_i + 1
            scores.append(-dtg)
            success.append(0.0)
    _validate_decision_terminal_grpo_group(
        torch.tensor(success),
        torch.tensor(scores),
        diagnostics,
        expected_size=16,
        stage_id=0,
        group_size=8,
    )


def test_decision_rloo_rejects_mixed_ids_inside_one_of_two_groups():
    diagnostics = [_terminal_diag(env_i) for env_i in range(8)]
    diagnostics += [_terminal_diag(env_i) for env_i in range(8)]
    for diag in diagnostics[8:]:
        diag["rft_group_id"] = 1
    diagnostics[-1]["rft_group_generation"] = 9
    with pytest.raises(RuntimeError, match="mixed terminal groups"):
        _validate_decision_rloo_terminal_group(
            torch.zeros(16),
            diagnostics,
            expected_size=16,
            stage_id=0,
            group_size=8,
        )

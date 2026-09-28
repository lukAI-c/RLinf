from collections import defaultdict

import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult
from rlinf.workers.env.env_worker import EnvWorker


def _cfg(mode: str = "adaptive_pos"):
    return OmegaConf.create(
        {
            "algorithm": {
                "group_size": 4,
                "reinforce_ada": {
                    "enabled": True,
                    "mode": mode,
                    "initial_candidates": 4,
                    "retry_batch_size": 4,
                    "max_candidates": 8,
                    "train_group_size": 4,
                    "min_successes": 1,
                    "min_failures": 3,
                    "only_episode_overfit": True,
                },
            },
            "env": {
                "train": {
                    "episode_overfit": {"enabled": True},
                    "max_episode_steps": 80,
                }
            },
        }
    )


def _worker(mode: str = "adaptive_pos") -> EnvWorker:
    worker = EnvWorker.__new__(EnvWorker)
    worker.cfg = _cfg(mode)
    worker.stage_num = 1
    worker.env_list = [object()]
    worker._reinforce_ada_pool_results = None
    worker._reinforce_ada_pool_diags = None
    worker._reinforce_ada_attempts = 0
    worker._reinforce_ada_ready_results = None
    worker._reinforce_ada_ready_diags = None
    return worker


def _result(reward: float) -> EmbodiedRolloutResult:
    result = EmbodiedRolloutResult(max_episode_length=80)
    result.rewards.append(torch.tensor([[reward]], dtype=torch.float32))
    return result


def _diag(index: int, *, success: bool = False, kind: str = "wrong_stop") -> dict:
    components = {"dense_geo": 0.1, "wrong_stop": -1.0 if kind == "wrong_stop" else 0.0}
    if success:
        components = {"success": 5.0, "endpoint": 0.7}
    return {
        "env_id": index,
        "episode_id": 824,
        "scene_id": "scene",
        "success": float(success),
        "clean_stop_success": float(success),
        "success_type": "clean" if success else kind,
        "best_dtg_progress": float(index),
        "final_dtg": 1.0 if success else 4.0 + index,
        "min_dtg": 1.0,
        "final_regression": float(index),
        "steps_taken": 10.0 + index,
        "ndtw": 0.9 if success else 0.1,
        "reward_components": components,
    }


def test_pos_selection_requires_true_success_and_keeps_one_anchor():
    worker = _worker()
    results = [_result(-1.0), _result(-0.5), _result(4.0), _result(-0.2), _result(-0.8)]
    diags = [
        _diag(0, kind="wrong_stop"),
        _diag(1, kind="no_stop"),
        _diag(2, success=True),
        _diag(3, kind="wrong_stop"),
        _diag(4, kind="wrong_stop"),
    ]

    selected, selected_diags, should_train = worker._select_reinforce_ada_pos_group(
        defaultdict(list), 0, results, diags
    )

    assert should_train
    assert len(selected) == 4
    assert sum(float(diag.get("success", 0.0)) > 0.5 for diag in selected_diags) == 1
    assert selected_diags[0]["original_env_id"] == 2


def test_static_pos_skips_an_all_failure_pool():
    worker = _worker("static_pos")
    results = [_result(-1.0), _result(-0.5), _result(-0.2), _result(-0.8)]
    diags = [_diag(i, kind="wrong_stop") for i in range(4)]

    selected, selected_diags, should_train = worker._select_reinforce_ada_pos_group(
        defaultdict(list), 0, results, diags
    )

    assert not should_train
    assert selected == []
    assert selected_diags == []


def test_adaptive_collection_retries_then_emits_once_success_exists():
    worker = _worker("adaptive_pos")
    first = [_result(-1.0), _result(-0.5), _result(-0.2), _result(-0.8)]
    first_diags = {i: _diag(i, kind="wrong_stop") for i in range(4)}
    metrics = defaultdict(list)

    outcome = worker._collect_reinforce_ada_attempt(metrics, [first], [first_diags])
    assert outcome["status"] == "retry"
    assert outcome["candidate_count"] == 4

    second = [_result(4.0), _result(-0.4), _result(-0.3), _result(-0.6)]
    second_diags = {
        0: _diag(0, success=True),
        1: _diag(1, kind="wrong_stop"),
        2: _diag(2, kind="no_stop"),
        3: _diag(3, kind="wrong_stop"),
    }
    outcome = worker._collect_reinforce_ada_attempt(metrics, [second], [second_diags])

    assert outcome["status"] == "ready_train"
    assert outcome["candidate_count"] == 8
    assert outcome["first_success_attempt"] == 2
    assert worker._reinforce_ada_ready_results is not None
    assert len(worker._reinforce_ada_ready_results[0]) == 4

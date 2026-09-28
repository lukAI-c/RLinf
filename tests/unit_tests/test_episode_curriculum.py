import numpy as np
import pytest

from types import SimpleNamespace

from rlinf.envs.genark.genark_env import (
    EpisodeBalancer,
    GenarkVecEnv,
    _filter_overfit_episode,
)


def _episode(episode_id):
    return {
        "episode_id": episode_id,
        "instruction": {"instruction_text": f"instruction {episode_id}"},
        "start_position": [0.0, 0.0, 0.0],
        "goals": [{"position": [1.0, 0.0, 0.0]}],
    }


def test_episode_curriculum_switches_tier_at_global_step():
    episodes = [_episode(1), _episode(2), _episode(3)]
    balancer = EpisodeBalancer(
        episodes,
        np.random.default_rng(7),
        curriculum_tiers={"1": "clean_seed", "2": "stop_frontier", "3": "hard"},
        curriculum_stages=[
            {"name": "seed", "until_step": 2, "weights": {"clean_seed": 1.0}},
            {"name": "stop", "until_step": 4, "weights": {"stop_frontier": 1.0}},
            {"name": "hard", "until_step": None, "weights": {"hard": 1.0}},
        ],
    )

    assert balancer.next_episode()[0]["episode_id"] == 1
    assert balancer.next_episode()[0]["episode_id"] == 1
    balancer.set_global_step(2)
    assert balancer.next_episode()[0]["episode_id"] == 2
    balancer.set_global_step(4)
    assert balancer.next_episode()[0]["episode_id"] == 3


def test_episode_curriculum_renormalizes_over_available_tiers():
    episodes = [_episode(1), _episode(2)]
    balancer = EpisodeBalancer(
        episodes,
        np.random.default_rng(11),
        curriculum_tiers={"1": "clean_seed", "2": "stop_frontier"},
        curriculum_stages=[
            {
                "name": "mixed",
                "until_step": None,
                "weights": {"clean_seed": 0.25, "stop_frontier": 0.75, "hard": 0.5},
            }
        ],
    )

    selected = {balancer.next_episode()[0]["episode_id"] for _ in range(20)}
    assert selected <= {1, 2}
    assert selected == {1, 2}


def test_episode_curriculum_rejects_incomplete_manifest():
    with pytest.raises(ValueError, match="missing episode IDs"):
        EpisodeBalancer(
            [_episode(1), _episode(2)],
            np.random.default_rng(3),
            curriculum_tiers={"1": "clean_seed"},
            curriculum_stages=[
                {"name": "seed", "until_step": None, "weights": {"clean_seed": 1.0}}
            ],
        )


def _assignment_env(curriculum_enabled: bool):
    env = GenarkVecEnv.__new__(GenarkVecEnv)
    env.num_envs = 8
    env.group_size = 8
    env.episode_curriculum_enabled = curriculum_enabled
    env._slot_active = np.ones(8, dtype=bool)
    env._scene_layout = None
    env._pinned_scene_id = "scene"
    env._pool_by_scene = {"scene": [_episode(586), _episode(824)]}
    env._episodes = [None] * 8
    env._instructions = [""] * 8
    env._assign_current_trial_indices = lambda indices: None
    return env


def test_curriculum_assignment_survives_outer_bootstrap_reset():
    env = _assignment_env(curriculum_enabled=True)
    env._assign_episodes_to_envs()
    assert {ep["episode_id"] for ep in env._episodes} == {586}

    selected = env._pool_by_scene["scene"][1]
    for i in range(env.num_envs):
        env._episodes[i] = selected
        env._instructions[i] = selected["instruction"]["instruction_text"]

    env._assign_episodes_to_envs()
    assert {ep["episode_id"] for ep in env._episodes} == {824}
    assert set(env._instructions) == {"instruction 824"}


def test_non_curriculum_assignment_keeps_static_pool_behavior():
    env = _assignment_env(curriculum_enabled=False)
    selected = env._pool_by_scene["scene"][1]
    env._episodes = [selected] * env.num_envs

    env._assign_episodes_to_envs()
    assert {ep["episode_id"] for ep in env._episodes} == {586}


def test_overfit_episode_ids_preserve_requested_order():
    episodes = [
        {
            "episode_id": str(ep_id),
            "scene_id": "mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
            "instruction": {"instruction_text": f"go {ep_id}"},
        }
        for ep_id in (469, 586, 824, 1301)
    ]
    filtered = _filter_overfit_episode(
        episodes,
        SimpleNamespace(
            episode_overfit=SimpleNamespace(
                enabled=True,
                scene_id="mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
                episode_ids=["586", "824", "1301"],
            ),
            group_size=1,
            total_num_envs=3,
        ),
    )
    assert [ep["episode_id"] for ep in filtered] == ["586", "824", "1301"]

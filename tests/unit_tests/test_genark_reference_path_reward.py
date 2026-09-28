import numpy as np

from rlinf.envs.genark.genark_env import (
    GenarkVecEnv,
    _missed_stop_aux_reward,
    _normalized_reference_path_potential,
    _reference_path_potential,
)


def test_reference_path_potential_rewards_along_track_progress():
    path = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, -3.0]],
        dtype=float,
    )
    start, _, _ = _reference_path_potential([0.0, 0.0, 0.0], path)
    forward, along, lateral = _reference_path_potential([1.5, 0.0, 0.0], path)

    assert forward - start == 1.5
    assert along == 1.5
    assert lateral == 0.0


def test_reference_path_potential_penalizes_lateral_drift_and_reversal():
    path = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    on_path, _, _ = _reference_path_potential([2.0, 0.0, 0.0], path)
    drifted, _, lateral = _reference_path_potential([2.0, 0.0, 1.0], path)
    reversed_position, _, _ = _reference_path_potential([1.0, 0.0, 0.0], path)

    assert drifted == on_path - 1.0
    assert lateral == 1.0
    assert reversed_position - on_path == -1.0


def test_normalized_reference_path_potential_is_bounded_by_route_length():
    path = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    midpoint, along, lateral = _normalized_reference_path_potential(
        [2.0, 0.0, 0.0], path
    )
    far_off_path, _, _ = _normalized_reference_path_potential(
        [2.0, 0.0, 100.0], path
    )

    assert midpoint == 0.5
    assert along == 2.0
    assert lateral == 0.0
    assert far_off_path == -1.0


def test_normalized_reference_path_potential_supports_custom_origin():
    path = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=float)
    origin, _, _ = _reference_path_potential([16.0, 0.0, 0.0], path)

    start, _, _ = _normalized_reference_path_potential(
        [16.0, 0.0, 0.0],
        path,
        origin_potential=origin,
        normalization_length=4.0,
    )
    one_meter_forward, _, _ = _normalized_reference_path_potential(
        [17.0, 0.0, 0.0],
        path,
        origin_potential=origin,
        normalization_length=4.0,
    )
    one_meter_backward, _, _ = _normalized_reference_path_potential(
        [15.0, 0.0, 0.0],
        path,
        origin_potential=origin,
        normalization_length=4.0,
    )

    assert start == 0.0
    assert one_meter_forward == 0.25
    assert one_meter_backward == -0.25


def test_missed_stop_aux_reward_only_penalizes_non_stop_inside_radius():
    kwargs = {
        "success_distance": 3.0,
        "penalty": 0.25,
        "eligible_episode": True,
    }
    assert _missed_stop_aux_reward(
        decision_start_dtg=2.5, is_stop=False, **kwargs
    ) == -0.25
    assert _missed_stop_aux_reward(
        decision_start_dtg=2.5, is_stop=True, **kwargs
    ) == 0.0
    assert _missed_stop_aux_reward(
        decision_start_dtg=3.5, is_stop=False, **kwargs
    ) == 0.0
    assert _missed_stop_aux_reward(
        decision_start_dtg=2.5,
        is_stop=False,
        **{**kwargs, "eligible_episode": False},
    ) == 0.0


def test_missed_stop_terminal_pending_is_per_env():
    env = object.__new__(GenarkVecEnv)
    env._missed_stop_terminal_pending = np.asarray([False, True, False])

    assert not env.missed_stop_terminal_pending(0)
    assert env.missed_stop_terminal_pending(1)
    assert not env.missed_stop_terminal_pending(2)

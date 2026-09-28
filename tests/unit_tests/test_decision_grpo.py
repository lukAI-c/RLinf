import torch

from rlinf.algorithms.advantages import (
    compute_decision_aux_rloo_advantages,
    compute_decision_grpo_advantages,
    compute_decision_maxrl_aux_rloo_advantages,
    compute_decision_maxrl_advantages,
    compute_decision_maxrl_outcome_advantages,
    compute_decision_rloo_advantages,
    compute_decision_rloo_aux_rloo_advantages,
    compute_decision_rloo_outcome_advantages,
    compute_decision_terminal_grpo_advantages,
)
from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.data.embodied_io_struct import (
    EmbodiedRolloutResult,
    convert_trajectories_to_batch,
)


def test_decision_grpo_uses_reward_to_go_and_masks_finished_slots():
    rewards = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    dones = torch.tensor(
        [[False, False, False, False], [False, False, False, False],
         [False, True, False, False], [True, True, True, True]]
    )
    loss_mask = torch.tensor(
        [[True, True, True, True], [True, True, True, True], [True, False, True, True]]
    )

    advantages, _ = compute_decision_grpo_advantages(
        rewards=rewards, dones=dones, loss_mask=loss_mask, group_size=4
    )

    assert advantages.shape == rewards.shape
    assert advantages[2, 1] == 0
    assert torch.isfinite(advantages).all()
    assert advantages[1, 1] < 0


def test_decision_grpo_zeroes_collapsed_or_single_member_decisions():
    rewards = torch.zeros(2, 4)
    dones = torch.tensor(
        [[False, False, False, False], [False, False, False, False], [True, True, True, True]]
    )
    loss_mask = torch.tensor(
        [[True, True, True, True], [True, False, False, False]]
    )

    advantages, _ = compute_decision_grpo_advantages(
        rewards=rewards, dones=dones, loss_mask=loss_mask, group_size=4
    )

    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_decision_grpo_preserves_embodied_decision_axis():
    rewards = torch.tensor([[[0.0], [0.0], [0.0], [0.0]], [[0.0], [-1.0], [1.0], [0.0]]])
    dones = torch.tensor(
        [
            [[False], [False], [False], [False]],
            [[False], [False], [False], [False]],
            [[True], [True], [True], [True]],
        ]
    )
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    output = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="decision_grpo",
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        group_size=4,
        reward_type="action_level",
        gamma=1.0,
    )

    assert output["advantages"].shape == rewards.shape
    assert output["advantages"][1, 1, 0] < 0


def test_decision_grpo_normalizes_each_group_independently():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0, 101.0, 102.0, 103.0, 104.0]])
    dones = torch.tensor(
        [[False] * 8, [True] * 8]
    )
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    advantages, _ = compute_decision_grpo_advantages(
        rewards=rewards, dones=dones, loss_mask=loss_mask, group_size=4
    )

    assert torch.allclose(advantages[:, :4], advantages[:, 4:])


def test_decision_grpo_stops_returns_at_episode_boundary():
    rewards = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [100.0, 1.0, 1.0, 1.0]]
    )
    dones = torch.tensor(
        [[False, False, False, False], [True, False, False, False], [True, True, True, True]]
    )
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    advantages, _ = compute_decision_grpo_advantages(
        rewards=rewards, dones=dones, loss_mask=loss_mask, group_size=4
    )

    # Env 0 terminates after decision 0, so its later reward cannot affect
    # decision 0's return or advantage.
    assert advantages[0, 0] < 0


def test_decision_maxrl_outcome_advantages_cover_all_success_counts():
    k0 = compute_decision_maxrl_outcome_advantages(
        torch.tensor([0.0, 0.0, 0.0, 0.0]), group_size=4
    )
    k1 = compute_decision_maxrl_outcome_advantages(
        torch.tensor([1.0, 0.0, 0.0, 0.0]), group_size=4
    )
    k2 = compute_decision_maxrl_outcome_advantages(
        torch.tensor([1.0, 1.0, 0.0, 0.0]), group_size=4
    )
    k4 = compute_decision_maxrl_outcome_advantages(
        torch.tensor([1.0, 1.0, 1.0, 1.0]), group_size=4
    )

    assert torch.equal(k0, torch.zeros_like(k0))
    assert torch.allclose(k1, torch.tensor([3.0, -1.0, -1.0, -1.0]))
    assert torch.allclose(k2, torch.tensor([1.0, 1.0, -1.0, -1.0]))
    assert torch.equal(k4, torch.zeros_like(k4))


def test_decision_rloo_is_bounded_and_zero_mean_for_rare_success():
    outcome = compute_decision_rloo_outcome_advantages(
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        group_size=8,
    )

    assert torch.allclose(outcome[:1], torch.tensor([1.0]), atol=1e-5)
    assert torch.allclose(
        outcome[1:], torch.full((7,), -1.0 / 7.0), atol=1e-5
    )
    assert torch.allclose(outcome.mean(), torch.tensor(0.0), atol=1e-5)
    assert outcome.abs().max() <= 1.0


def test_decision_aux_rloo_uses_other_trajectories_as_baseline():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    dones = torch.tensor([[False] * 4, [True] * 4])
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    advantages = compute_decision_aux_rloo_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        group_size=4,
    )

    assert torch.allclose(
        advantages, torch.tensor([[1.0, -1 / 3, -1 / 3, -1 / 3]])
    )


def test_decision_maxrl_aux_rloo_uses_auxiliary_signal_for_all_failure_group():
    rewards = torch.tensor([[0.5, 0.0, -0.5, 0.0]])
    dones = torch.tensor([[False] * 4, [True] * 4])
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    advantages, _ = compute_decision_maxrl_aux_rloo_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        episode_success=torch.zeros(4),
        group_size=4,
        maxrl_aux_coef=0.25,
    )

    assert advantages[0, 0] > 0
    assert advantages[0, 2] < 0
    assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)


def test_decision_maxrl_aux_rloo_does_not_double_count_clean_stop():
    rewards = torch.zeros(1, 4)
    dones = torch.tensor([[False] * 4, [True] * 4])
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    advantages, _ = compute_decision_maxrl_aux_rloo_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        episode_success=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        group_size=4,
    )

    assert torch.allclose(advantages, torch.tensor([[3.0, -1.0, -1.0, -1.0]]))


def test_decision_rloo_aux_rloo_bounds_rare_clean_stop_and_uses_aux_on_k0():
    dones = torch.tensor([[False] * 8, [True] * 8])
    loss_mask = torch.ones(1, 8, dtype=torch.bool)

    clean_stop_advantages, _ = compute_decision_rloo_aux_rloo_advantages(
        rewards=torch.zeros(1, 8),
        dones=dones,
        loss_mask=loss_mask,
        episode_success=torch.tensor([1.0] + [0.0] * 7),
        group_size=8,
        rloo_aux_coef=0.25,
    )
    assert torch.allclose(clean_stop_advantages[0, 0], torch.tensor(1.0))
    assert torch.allclose(
        clean_stop_advantages[0, 1:],
        torch.full((7,), -1.0 / 7.0),
    )

    all_failure_advantages, _ = compute_decision_rloo_aux_rloo_advantages(
        rewards=torch.tensor([[0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        dones=dones,
        loss_mask=loss_mask,
        episode_success=torch.zeros(8),
        group_size=8,
        rloo_aux_coef=0.25,
    )
    assert all_failure_advantages[0, 0] > 0
    assert all_failure_advantages[0, 2] < 0
    assert torch.allclose(
        all_failure_advantages.mean(), torch.tensor(0.0), atol=1e-6
    )


def test_decision_maxrl_aux_rloo_registry_preserves_decision_axis():
    rewards = torch.tensor(
        [
            [[0.5], [0.0], [-0.5], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
        ]
    )
    dones = torch.zeros(3, 4, 1, dtype=torch.bool)
    dones[-1] = True
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="decision_maxrl_aux_rloo",
        reward_type="action_level",
        rewards=rewards,
        dones=dones,
        values=None,
        loss_mask=loss_mask,
        loss_mask_sum=None,
        episode_success=torch.zeros(1, 4, 1),
        group_size=4,
        gamma=1.0,
        maxrl_aux_coef=0.25,
    )

    assert result["advantages"].shape == rewards.shape
    assert result["advantages"][0, 0, 0] > 0
    assert result["advantages"][0, 2, 0] < 0


def test_decision_rloo_aux_rloo_registry_preserves_decision_axis():
    rewards = torch.tensor(
        [
            [[0.5], [0.0], [-0.5], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
        ]
    )
    dones = torch.zeros(3, 4, 1, dtype=torch.bool)
    dones[-1] = True
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="decision_rloo_aux_rloo",
        reward_type="action_level",
        rewards=rewards,
        dones=dones,
        values=None,
        loss_mask=loss_mask,
        loss_mask_sum=None,
        episode_success=torch.tensor([[[1.0], [0.0], [0.0], [0.0]]]),
        group_size=4,
        gamma=1.0,
        rloo_aux_coef=0.25,
    )

    assert result["advantages"].shape == rewards.shape
    assert result["advantages"][0, 0, 0] > 1.0
    assert torch.all(result["advantages"][:, 1:, 0] < 0)


def test_decision_maxrl_masks_finished_decisions_and_keeps_k0_process_signal():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0]])
    dones = torch.tensor(
        [[False, False, False, False], [False, True, False, False], [True, True, True, True]]
    )
    loss_mask = torch.tensor(
        [[True, True, True, True], [True, False, True, True]]
    )

    advantages, _ = compute_decision_maxrl_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        episode_success=torch.zeros(4),
        group_size=4,
    )

    assert advantages[1, 1] == 0
    assert torch.isfinite(advantages).all()
    assert advantages.abs().sum() > 0


def test_decision_rloo_outcome_covers_all_success_counts():
    k0 = compute_decision_rloo_outcome_advantages(
        torch.tensor([0.0, 0.0, 0.0, 0.0]), group_size=4
    )
    k1 = compute_decision_rloo_outcome_advantages(
        torch.tensor([1.0, 0.0, 0.0, 0.0]), group_size=4
    )
    k2 = compute_decision_rloo_outcome_advantages(
        torch.tensor([1.0, 1.0, 0.0, 0.0]), group_size=4
    )
    k4 = compute_decision_rloo_outcome_advantages(
        torch.tensor([1.0, 1.0, 1.0, 1.0]), group_size=4
    )

    assert torch.equal(k0, torch.zeros_like(k0))
    assert torch.allclose(k1, torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3]))
    assert torch.allclose(k2, torch.tensor([2 / 3, 2 / 3, -2 / 3, -2 / 3]))
    assert torch.equal(k4, torch.zeros_like(k4))


def test_decision_rloo_combines_outcome_and_process_signal():
    rewards = torch.zeros(2, 4)
    dones = torch.tensor([[False] * 4, [False] * 4, [True] * 4])
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    advantages, _ = compute_decision_rloo_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        episode_success=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        group_size=4,
    )

    assert torch.allclose(advantages[:, 0], torch.ones(2))
    assert torch.allclose(advantages[:, 1:], torch.full((2, 3), -1 / 3))


def test_episode_success_survives_trajectory_split_and_batch_conversion():
    result = EmbodiedRolloutResult(
        episode_success=torch.tensor([[[1.0], [0.0], [1.0], [0.0]]])
    )

    pieces = result.to_splited_trajectories(split_size=2)
    batch = convert_trajectories_to_batch(pieces)

    assert torch.equal(batch["episode_success"], result.episode_success)


def test_decision_maxrl_registry_preserves_decision_axis():
    rewards = torch.zeros(2, 4, 1)
    dones = torch.tensor(
        [
            [[False], [False], [False], [False]],
            [[False], [False], [False], [False]],
            [[True], [True], [True], [True]],
        ]
    )
    output = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="decision_maxrl",
        rewards=rewards,
        dones=dones,
        loss_mask=torch.ones_like(rewards, dtype=torch.bool),
        episode_success=torch.tensor([[[1.0], [0.0], [0.0], [0.0]]]),
        group_size=4,
        reward_type="action_level",
    )

    assert output["advantages"].shape == rewards.shape
    assert torch.allclose(output["advantages"][:, 0, 0], torch.tensor([3.0, 3.0]))
    assert torch.allclose(output["advantages"][:, 1:, 0], -torch.ones(2, 3))


def test_decision_rloo_registry_preserves_decision_axis():
    rewards = torch.zeros(2, 4, 1)
    dones = torch.tensor(
        [
            [[False], [False], [False], [False]],
            [[False], [False], [False], [False]],
            [[True], [True], [True], [True]],
        ]
    )
    output = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="decision_rloo",
        rewards=rewards,
        dones=dones,
        loss_mask=torch.ones_like(rewards, dtype=torch.bool),
        episode_success=torch.tensor([[[1.0], [0.0], [0.0], [0.0]]]),
        group_size=4,
        reward_type="action_level",
    )

    assert output["advantages"].shape == rewards.shape
    assert torch.allclose(output["advantages"][:, 0, 0], torch.ones(2))
    assert torch.allclose(
        output["advantages"][:, 1:, 0], torch.full((2, 3), -1 / 3)
    )


def test_decision_terminal_grpo_is_zero_mean_for_mixed_scores():
    scores = torch.tensor([-5.0, -2.0, -2.0, -1.0])
    rewards = torch.tensor([[10.0, -7.0, 3.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
    dones = torch.tensor([[False] * 4, [False] * 4, [True] * 4])
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    advantages, _ = compute_decision_terminal_grpo_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        episode_terminal_score=scores,
        group_size=4,
    )

    assert torch.isfinite(advantages).all()
    assert torch.allclose(advantages.mean(dim=1), torch.zeros(2), atol=1e-5)
    assert advantages[0, 0] < 0
    assert advantages[0, 3] > 0
    assert torch.equal(advantages[0], advantages[1])


def test_decision_terminal_grpo_two_groups_of_eight_broadcast():
    scores = torch.tensor(
        [-5.0, -4.0, -3.0, -2.0, -6.0, -5.0, -4.0, -3.0]
        + [-2.0, -2.0, -2.0, -2.0, -8.0, -7.0, -6.0, -1.0]
    )
    rewards = torch.zeros(1, 16)
    dones = torch.tensor([[False] * 16, [True] * 16])
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    advantages, _ = compute_decision_terminal_grpo_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        episode_terminal_score=scores,
        group_size=8,
        episode_ids=["824"] * 8 + ["447"] * 8,
    )

    assert advantages.shape == rewards.shape
    assert torch.isfinite(advantages).all()
    assert torch.allclose(advantages[0, :8].mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(advantages[0, 8:].mean(), torch.tensor(0.0), atol=1e-5)


def test_decision_terminal_grpo_zeroes_equal_score_group():
    rewards = torch.ones(2, 4)
    dones = torch.tensor([[False] * 4, [False] * 4, [True] * 4])
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    advantages, _ = compute_decision_terminal_grpo_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        episode_terminal_score=torch.full((4,), -2.0),
        group_size=4,
    )

    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_decision_terminal_grpo_ignores_per_decision_rewards():
    scores = torch.tensor([-5.0, -4.0, -3.0, -2.0])
    dones = torch.tensor([[False] * 4, [True] * 4])
    loss_mask = torch.ones(1, 4, dtype=torch.bool)

    left, _ = compute_decision_terminal_grpo_advantages(
        rewards=torch.zeros(1, 4),
        dones=dones,
        loss_mask=loss_mask,
        episode_terminal_score=scores,
        group_size=4,
    )
    right, _ = compute_decision_terminal_grpo_advantages(
        rewards=torch.tensor([[100.0, -50.0, 7.0, 0.0]]),
        dones=dones,
        loss_mask=loss_mask,
        episode_terminal_score=scores,
        group_size=4,
    )

    assert torch.equal(left, right)


def test_decision_terminal_grpo_rejects_mixed_episode_ids():
    rewards = torch.zeros(1, 4)
    dones = torch.tensor([[False] * 4, [True] * 4])
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    try:
        compute_decision_terminal_grpo_advantages(
            rewards=rewards,
            dones=dones,
            loss_mask=loss_mask,
            episode_terminal_score=torch.tensor([-5.0, -4.0, -3.0, -2.0]),
            group_size=4,
            episode_ids=["586", "586", "824", "586"],
        )
    except ValueError as exc:
        assert "mixed episode IDs" in str(exc)
    else:
        raise AssertionError("expected mixed episode IDs to raise")


def test_episode_terminal_score_survives_trajectory_split_and_batch_conversion():
    result = EmbodiedRolloutResult(
        episode_success=torch.tensor([[[1.0], [0.0], [1.0], [0.0]]]),
        episode_terminal_score=torch.tensor([[[-1.0], [-2.0], [-1.0], [-5.0]]]),
    )

    pieces = result.to_splited_trajectories(split_size=2)
    batch = convert_trajectories_to_batch(pieces)

    assert torch.equal(batch["episode_success"], result.episode_success)
    assert torch.equal(batch["episode_terminal_score"], result.episode_terminal_score)


def test_decision_terminal_grpo_registry_preserves_decision_axis():
    output = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="decision_terminal_grpo",
        rewards=torch.zeros(2, 4, 1),
        dones=torch.tensor(
            [
                [[False], [False], [False], [False]],
                [[False], [False], [False], [False]],
                [[True], [True], [True], [True]],
            ]
        ),
        loss_mask=torch.ones(2, 4, 1, dtype=torch.bool),
        episode_terminal_score=torch.tensor([[[-5.0], [-2.0], [-2.0], [-1.0]]]),
        group_size=4,
        reward_type="action_level",
    )

    assert output["advantages"].shape == (2, 4, 1)
    assert torch.isfinite(output["advantages"]).all()
    assert output["advantages"][0, 0, 0] < 0
    assert output["advantages"][0, 3, 0] > 0

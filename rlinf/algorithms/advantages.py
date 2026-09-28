# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.utils.utils import masked_mean


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz].
        values (torch.Tensor): Value function estimates. Shape: [seq_len, bsz].
        dones (torch.Tensor): Done flags (1 if episode ended, else 0).
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            delta = (
                rewards[step]
                + gamma * values[step + 1] * (~dones[step + 1])
                - values[step]
            )

        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae if critic_free else gae + values[step]

    advantages = returns - values[:-1] if not critic_free else returns

    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    return advantages, returns


@register_advantage("grpo")
def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None


@register_advantage("decision_grpo")
def compute_decision_grpo_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    gamma: float = 1.0,
    **kwargs,
):
    """Compute group-relative advantages independently for each decision.

    ``rewards`` is [T, B], where each entry is the reward accumulated by one
    LLM decision.  Unlike trajectory GRPO, this keeps T intact and uses the
    discounted reward-to-go at each decision before normalising within its
    GRPO group.
    """
    returns = torch.zeros_like(rewards)
    future_return = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        future_return = rewards[step] + gamma * future_return * (~dones[step + 1])
        returns[step] = future_return

    num_groups = rewards.shape[1] // group_size
    grouped_returns = returns.reshape(-1, num_groups, group_size)
    valid = loss_mask.bool().reshape(-1, num_groups, group_size)
    valid_count = valid.sum(dim=-1, keepdim=True)
    valid_returns = torch.where(valid, grouped_returns, torch.zeros_like(grouped_returns))
    mean = valid_returns.sum(dim=-1, keepdim=True) / valid_count.clamp_min(1)
    centered = torch.where(valid, grouped_returns - mean, torch.zeros_like(grouped_returns))
    # Match the existing GRPO convention: sample standard deviation within each
    # group. Decisions with fewer than two live trajectories have no baseline.
    variance = centered.square().sum(dim=-1, keepdim=True) / (valid_count - 1).clamp_min(1)
    advantages = centered / (variance.sqrt() + 1e-6)
    advantages = torch.where(valid_count >= 2, advantages, torch.zeros_like(advantages))
    advantages = torch.where(valid, advantages, torch.zeros_like(advantages))

    return advantages.reshape_as(rewards), None


def compute_decision_aux_rloo_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Leave-one-out advantages for decision-level auxiliary returns.

    Unlike decision GRPO, this estimator neither includes the current sample in
    its baseline nor divides by a random group standard deviation. It therefore
    preserves the scale chosen by the bounded auxiliary reward.
    """
    returns = torch.zeros_like(rewards)
    future_return = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        future_return = rewards[step] + gamma * future_return * (~dones[step + 1])
        returns[step] = future_return

    num_groups = rewards.shape[1] // group_size
    grouped_returns = returns.reshape(-1, num_groups, group_size)
    valid = loss_mask.bool().reshape(-1, num_groups, group_size)
    valid_returns = torch.where(valid, grouped_returns, torch.zeros_like(grouped_returns))
    valid_count = valid.sum(dim=-1, keepdim=True)
    other_count = valid_count - valid.to(dtype=valid_count.dtype)
    other_sum = valid_returns.sum(dim=-1, keepdim=True) - valid_returns
    other_mean = other_sum / other_count.clamp_min(1)
    advantages = grouped_returns - other_mean
    advantages = torch.where(valid & (other_count > 0), advantages, torch.zeros_like(advantages))
    return advantages.reshape_as(rewards)


def compute_decision_maxrl_outcome_advantages(
    episode_success: torch.Tensor,
    group_size: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return the binary-success MaxRL control-variate advantage per trajectory.

    For a fixed-N group with K successful trajectories, the estimator is
    ``(Y - K / N) / (K / N)`` when K > 0.  A fully failed group has no MaxRL
    outcome signal, so it returns zero and can still learn from process reward.
    """
    success = episode_success.reshape(-1).to(dtype=torch.float32)
    if success.numel() % group_size != 0:
        raise ValueError(
            f"Decision-MaxRL requires batch={success.numel()} divisible by "
            f"group_size={group_size}"
        )
    grouped = success.reshape(-1, group_size)
    p_hat = grouped.mean(dim=-1, keepdim=True)
    outcome = torch.where(
        p_hat > 0,
        (grouped - p_hat) / (p_hat + epsilon),
        torch.zeros_like(grouped),
    )
    return outcome.reshape_as(success)


def compute_decision_rloo_outcome_advantages(
    episode_success: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Return a leave-one-out binary-outcome advantage per trajectory.

    Each trajectory uses the other ``N - 1`` outcomes as its baseline:
    ``A_i = Y_i - mean(Y_{-i})``. The baseline is independent of trajectory
    ``i``'s sampled action, and binary outcomes keep this estimator in [-1, 1].
    """
    if group_size < 2:
        raise ValueError("Decision-RLOO requires group_size >= 2")
    success = episode_success.reshape(-1).to(dtype=torch.float32)
    if success.numel() % group_size != 0:
        raise ValueError(
            f"Decision-RLOO requires batch={success.numel()} divisible by "
            f"group_size={group_size}"
        )
    grouped = success.reshape(-1, group_size)
    leave_one_out_mean = (
        grouped.sum(dim=-1, keepdim=True) - grouped
    ) / float(group_size - 1)
    return (grouped - leave_one_out_mean).reshape_as(success)


@register_advantage("decision_maxrl")
def compute_decision_maxrl_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    loss_mask: torch.Tensor,
    episode_success: torch.Tensor,
    group_size: int,
    gamma: float = 1.0,
    maxrl_process_coef: float = 1.0,
    maxrl_epsilon: float = 1e-6,
    **kwargs,
):
    """Combine binary-outcome MaxRL with decision-level process advantages.

    Outcome credit is trajectory-wide and comes only from the terminal success
    verifier.
    """
    if episode_success is None:
        raise ValueError("decision_maxrl requires explicit episode_success labels")

    process_advantages, _ = compute_decision_grpo_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        group_size=group_size,
        gamma=gamma,
    )
    outcome = compute_decision_maxrl_outcome_advantages(
        episode_success=episode_success,
        group_size=group_size,
        epsilon=maxrl_epsilon,
    ).to(device=rewards.device, dtype=rewards.dtype)
    total = outcome.unsqueeze(0) + float(maxrl_process_coef) * process_advantages
    return total * loss_mask.to(dtype=total.dtype), None


@register_advantage("decision_rloo")
def compute_decision_rloo_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    loss_mask: torch.Tensor,
    episode_success: torch.Tensor,
    group_size: int,
    gamma: float = 1.0,
    rloo_process_coef: float = 1.0,
    **kwargs,
):
    """Combine bounded RLOO outcome credit with decision process advantages."""
    if episode_success is None:
        raise ValueError("decision_rloo requires explicit episode_success labels")

    process_advantages, _ = compute_decision_grpo_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        group_size=group_size,
        gamma=gamma,
    )
    outcome = compute_decision_rloo_outcome_advantages(
        episode_success=episode_success,
        group_size=group_size,
    ).to(device=rewards.device, dtype=rewards.dtype)
    total = outcome.unsqueeze(0) + float(rloo_process_coef) * process_advantages
    return total * loss_mask.to(dtype=total.dtype), None


@register_advantage("decision_maxrl_aux_rloo")
def compute_decision_maxrl_aux_rloo_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    loss_mask: torch.Tensor,
    episode_success: torch.Tensor,
    group_size: int,
    gamma: float = 1.0,
    maxrl_aux_coef: float = 0.25,
    maxrl_epsilon: float = 1e-6,
    **kwargs,
):
    """Optimize clean-STOP MaxRL plus a bounded auxiliary RLOO objective.

    ``episode_success`` is the sole outcome channel. ``rewards`` must contain
    auxiliary rewards only, so clean STOP is not counted a second time.
    """
    if episode_success is None:
        raise ValueError(
            "decision_maxrl_aux_rloo requires explicit episode_success labels"
        )

    auxiliary = compute_decision_aux_rloo_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        group_size=group_size,
        gamma=gamma,
    )
    outcome = compute_decision_maxrl_outcome_advantages(
        episode_success=episode_success,
        group_size=group_size,
        epsilon=maxrl_epsilon,
    ).to(device=rewards.device, dtype=rewards.dtype)
    total = outcome.unsqueeze(0) + float(maxrl_aux_coef) * auxiliary
    return total * loss_mask.to(dtype=total.dtype), None


@register_advantage("decision_rloo_aux_rloo")
def compute_decision_rloo_aux_rloo_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    loss_mask: torch.Tensor,
    episode_success: torch.Tensor,
    group_size: int,
    gamma: float = 1.0,
    rloo_aux_coef: float = 0.25,
    **kwargs,
):
    """Optimize bounded clean-STOP RLOO plus auxiliary-return RLOO.

    ``episode_success`` is the sole binary outcome channel. ``rewards`` contains
    only decision-level shaping rewards, so clean STOP is not counted twice.
    Both baselines exclude the trajectory whose advantage is being estimated.
    """
    if episode_success is None:
        raise ValueError(
            "decision_rloo_aux_rloo requires explicit episode_success labels"
        )

    auxiliary = compute_decision_aux_rloo_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        group_size=group_size,
        gamma=gamma,
    )
    outcome = compute_decision_rloo_outcome_advantages(
        episode_success=episode_success,
        group_size=group_size,
    ).to(device=rewards.device, dtype=rewards.dtype)
    total = outcome.unsqueeze(0) + float(rloo_aux_coef) * auxiliary
    return total * loss_mask.to(dtype=total.dtype), None


@register_advantage("decision_terminal_grpo")
def compute_decision_terminal_grpo_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    loss_mask: torch.Tensor,
    episode_terminal_score: torch.Tensor,
    group_size: int,
    episode_ids=None,
    scene_ids=None,
    **kwargs,
):
    """Broadcast one Euclidean terminal-score GRPO advantage per trajectory.

    ``rewards`` is ignored. The estimator uses only ``episode_terminal_score``.
    """
    if episode_terminal_score is None:
        raise ValueError(
            "decision_terminal_grpo requires explicit episode_terminal_score labels"
        )
    from rlinf.envs.genark.terminal_navigation_score import (
        compute_terminal_grpo_outcome_advantages,
    )

    outcome = compute_terminal_grpo_outcome_advantages(
        episode_terminal_score=episode_terminal_score,
        group_size=group_size,
        episode_ids=episode_ids,
        scene_ids=scene_ids,
    ).to(device=rewards.device, dtype=rewards.dtype)
    total = outcome.unsqueeze(0).expand_as(rewards)
    return total * loss_mask.to(dtype=total.dtype), None


@register_advantage("grpo_dynamic")
def compute_grpo_dynamic_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    idx_to_traj: list[int],
    advantage_mode: str = "turn",  # "trajectory" or "turn"
    **kwargs,
):
    """
    Compute GRPO advantages for multi-turn multi-agent scenarios.

    IMPORTANT: This function computes advantages PER QUESTION, not globally.
    - idx_to_traj maps turn_idx -> global_traj_idx (e.g., [0,0,1,1,2,2,3,3,4,4,...,15,15])
    - Trajectories 0-3 belong to question 0, 4-7 to question 1, etc.
    - We must compute GRPO separately for each question's group_size trajectories

    Two advantage computation modes:
    1. "trajectory": Trajectory-level GRPO (Method 1)
       - Compute mean/std over group_size trajectory rewards per question
       - Broadcast same advantage to all turns in a trajectory
       - Example: Q0 has 4 trajs with 1,2,3,4 turns. Compute GRPO over 4 traj rewards,
                  then assign traj0_adv to its 1 turn, traj1_adv to its 2 turns, etc.

    2. "turn": Turn-level GRPO (Method 2)
       - Compute mean/std over all turns within each question
       - Example: Q0 has 4 trajs with 1,2,3,4 turns = 10 turns total.
                  Compute GRPO over these 10 turn rewards (currently all same within traj).
       - Future-proof: works when turns have different rewards within same trajectory

    Args:
        rewards: Shape [num_sequence, 1] after preprocessing (num_sequence = total turns)
        loss_mask: Shape [seq_len, num_sequence] after preprocessing
        group_size: Number of trajectories per question (e.g., 4)
        idx_to_traj: List mapping turn_idx -> global_traj_idx
        advantage_mode: "trajectory" or "turn"

    Returns:
        advantages: Shape [seq_len, num_sequence]
    """
    num_sequence = len(idx_to_traj)

    rewards_flat = rewards.squeeze(-1)

    assert rewards_flat.numel() == num_sequence, (
        f"Rewards size mismatch: {rewards_flat.numel()} != {num_sequence}"
    )

    num_trajectories = max(idx_to_traj) + 1
    num_questions = num_trajectories // group_size
    assert num_trajectories % group_size == 0, (
        f"num_trajectories {num_trajectories} not divisible by group_size {group_size}"
    )

    turn_advantages = torch.zeros(
        num_sequence, dtype=rewards.dtype, device=rewards.device
    )

    if advantage_mode == "trajectory":
        # Aggregate turn rewards into per-trajectory rewards first.
        trajectory_rewards = torch.zeros(
            num_trajectories, dtype=rewards.dtype, device=rewards.device
        )
        trajectory_counts = torch.zeros(
            num_trajectories, dtype=torch.long, device=rewards.device
        )

        for turn_idx, traj_idx in enumerate(idx_to_traj):
            trajectory_rewards[traj_idx] += rewards_flat[turn_idx]
            trajectory_counts[traj_idx] += 1

        # Step 1: Average rewards per trajectory.
        trajectory_rewards = trajectory_rewards / trajectory_counts.clamp(min=1).float()

        # Step 2: reshape to [num_questions, group_size] for per-question GRPO.
        trajectory_rewards_grouped = trajectory_rewards.view(num_questions, group_size)

        # Step 3: compute per-question mean and std.
        per_question_mean = trajectory_rewards_grouped.mean(
            dim=-1, keepdim=True
        )  # [num_questions, 1]
        per_question_std = trajectory_rewards_grouped.std(
            dim=-1, keepdim=True
        )  # [num_questions, 1]

        # Step 4: normalize within each question group.
        normalized_trajectory_rewards = (
            trajectory_rewards_grouped - per_question_mean
        ) / (per_question_std + 1e-6)  # [num_questions, group_size]

        # Step 5: flatten back to [num_trajectories].
        normalized_trajectory_rewards = normalized_trajectory_rewards.view(-1)

        # Step 6: broadcast trajectory advantages to all turns in that trajectory.
        for turn_idx, traj_idx in enumerate(idx_to_traj):
            turn_advantages[turn_idx] = normalized_trajectory_rewards[traj_idx]

    elif advantage_mode == "turn":
        # Step 1: map each turn to its owning question.
        turn_to_question = torch.tensor(
            [idx_to_traj[i] // group_size for i in range(num_sequence)],
            dtype=torch.long,
            device=rewards.device,
        )

        # Step 2: normalize turn rewards within each question group.
        for question_idx in range(num_questions):
            question_mask = turn_to_question == question_idx
            question_turn_rewards = rewards_flat[question_mask]

            # Step 3: compute mean and std for all turns in this question.
            question_mean = question_turn_rewards.mean()
            question_std = question_turn_rewards.std()

            # Step 4: normalize turn rewards within the question.
            normalized_question_rewards = (question_turn_rewards - question_mean) / (
                question_std + 1e-6
            )

            # Step 5: write normalized turn-level advantages back.
            turn_advantages[question_mask] = normalized_question_rewards

    else:
        raise ValueError(
            f"Invalid advantage_mode: {advantage_mode}. Must be 'trajectory' or 'turn'"
        )

    advantages = torch.zeros_like(
        loss_mask, dtype=rewards.dtype
    ) + turn_advantages.view(1, -1)
    advantages = advantages * loss_mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        rewards (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = rewards.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        rewards = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(loss_mask).float()  # [L, B]
    seq_length = loss_mask.size(0)
    mask_flipped = loss_mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=0, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [1, B]

    r_matrix = r_matrix.scatter_(dim=0, index=eos_indices, src=rewards)  # [L, B]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [L, B]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[0]), dim=0).flip(dims=[0])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None


@register_advantage("raw")
def compute_raw_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    normalize_advantages: bool = False,
    **kwargs,
):
    """
    Return raw rewards or normalized rewards.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        normalize_advantages (bool): Whether to normalize advantages.

    Returns:
        torch.Tensor: advantages
    """
    if rewards.ndim == 2:
        rewards = rewards.reshape(-1)
    advantages = rewards.unsqueeze(0).expand_as(loss_mask) * loss_mask

    # Simple baseline subtraction (mean of valid advantages)
    if normalize_advantages:
        valid = advantages[loss_mask.bool()]
        if valid.numel() > 0:
            advantages = (advantages - valid.mean()) / (valid.std() + 1e-5)

    return advantages, None

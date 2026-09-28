# Decision-MaxRL Multi-Decision Navigation RL

## Goal

`decision_maxrl` gives continuous navigation decisions two complementary credit
signals without changing PPO loss or clipping:

\[
A_{i,t}^{total}=A_i^{outcome}+\lambda_{proc}A_{i,t}^{process}.
\]

`A_i^{outcome}` is a binary-success Maximum Likelihood Reinforcement Learning
(MaxRL) signal. `A_{i,t}^{process}` is the existing decision-level GRPO
reward-to-go signal built from DTG, path, wrong-stop, and no-stop rewards.

## Fixed-N MaxRL Outcome

For one episode group with `N` on-policy trajectories, terminal verifier labels
`Y_i in {0, 1}`, and `K = sum_i Y_i`:

\[
\hat p=K/N,\qquad
A_i^{outcome}=
\begin{cases}
\dfrac{Y_i-\hat p}{\hat p+\epsilon}, & K>0,\\
0, & K=0.
\end{cases}
\]

This is the zero-mean control-variate form of the fixed-N MaxRL estimator.  It
gives `N=4, K=1` a success advantage near `+3` and each failure near `-1`.
For `K=0`, no outcome signal is invented; the process term remains active.

## Data Contract

GenArk writes the completed episode diagnostic before reset. EnvWorker freezes
its `success` field into `Trajectory.episode_success` with shape `[1, B, 1]`.
The label is carried through trajectory merge, actor-rank split, and batch
alignment. It is removed immediately after advantage calculation so it never
enters decision-level PPO minibatch flattening.

The label is not inferred from shaped reward. A success is only a genuine
terminal environment success: STOP within `success_distance`. Wrong-stop,
timeout, and no-stop are zero.

## Training Configuration

The episode-overfit launch uses:

```yaml
algorithm:
  adv_type: decision_maxrl
  group_size: 4
  rollout_epoch: 1
  maxrl_process_coef: 1.0
  maxrl_epsilon: 1.0e-6
  normalize_advantages: false
env.train:
  sr_coef: 0.0
```

`sr_coef` is zero because success enters once, through the binary MaxRL outcome.
The overfit launch also sets `nav_endpoint_coef: 0.0`; its endpoint term is
success-only and would otherwise duplicate this outcome credit. Path and
DTG-based signals remain process reward.
Candidate selection and Reinforce-Ada remain disabled for this fixed-N first
stage. All `N` trajectories are used, come from one episode and one policy
version, and are kept in one single-rank actor group.

## Validation Sequence

1. Unit tests cover `K=0`, `K=1`, `K=2`, `K=4`, loss-mask behavior, and
   `episode_success` trajectory transport.
2. Episode 259 smoke run for two or three optimizer updates with `N=4`.
3. If the channel and signs are correct, increase to fixed `N=8`, with one
   single actor rank and `group_size=8`, for ten to twenty updates.

Expected behavior:

- `K=0`: outcome advantage is zero; valid decision process advantages can still
  be nonzero.
- `K>0`: successful trajectories receive positive outcome advantage and failed
  trajectories receive negative outcome advantage.
- STOP after a terminal decision has no further loss because `loss_mask` zeros
  the remaining decisions.

## Deliberate Scope Boundary

This first version does not use success-dependent early stopping, candidate
selection, Reinforce-Ada retries, global normalization, or actor-loss changes.
Those mechanisms alter the fixed-N estimator and belong to a later sampling
study after the binary-success channel and Decision-MaxRL objective are proven
on the overfit task.

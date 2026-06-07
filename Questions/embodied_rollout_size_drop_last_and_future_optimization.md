# Embodied Rollout Size Drop-Last and Future Optimization

## Background

In GenArk embodied RFT, the actor currently receives rollout trajectories from
`EnvWorker`/`RolloutWorker`, converts them into `rollout_batch`, then flattens the
first two dimensions for actor training.

In `rlinf/workers/actor/fsdp_actor_worker.py`, the actor computes:

```python
rollout_size = (
    self.rollout_batch["prev_logprobs"].shape[0]
    * self.rollout_batch["prev_logprobs"].shape[1]
)
```

After `process_nested_dict_for_train()`, tensors are flattened with:

```python
value.reshape(-1, *value.shape[2:])
```

So for embodied training:

```text
rollout_size ~= rollout time steps x rollout batch/env slots
```

## Why Rollout Size Is Not Stable

Unlike fixed-size offline batches, GenArk rollout size can vary between runs and
between scenes.

Main causes:

1. **Scene episode count limits active slots**

   Example log:

   ```text
   active_slots=14/18
   ```

   Even if `env.train.total_num_envs=18`, a fixed scene with only 14 episodes can
   provide at most 14 active env slots. The rest are ghost/dormant slots.

2. **Episodes terminate early**

   Example logs showed many episodes ending early:

   ```text
   steps=0
   steps=1
   steps=2
   steps=3
   ...
   steps=22
   ```

   This means the actual amount of trainable rollout data can be much smaller
   than `max_steps_per_rollout_epoch * total_num_envs`.

3. **Worker exhaustion**

   When all episodes in the pinned scene finish, the worker enters dormant mode:

   ```text
   Worker exhausted: ... all 14 episodes done. Entering dormant mode.
   ```

4. **Different scenes have different episode counts**

   Because each EnvWorker pins one scene, changing `scene_offset` or adding
   multiple scenes can change the effective number of active slots and therefore
   the final rollout size.

## Failure Observed

With:

```text
actor.global_batch_size=96
actor.micro_batch_size=1
world_size=2
```

The actor uses:

```text
batch_size_per_rank = actor.global_batch_size / world_size = 48
```

One run produced:

```text
rollout_size = 360
```

The old actor code required:

```python
assert rollout_size % batch_size_per_rank == 0
```

This failed:

```text
AssertionError: 360 is not divisible by 48
```

## Current Fix: Automatic Drop-Last

Implemented in:

```text
rlinf/workers/actor/fsdp_actor_worker.py
```

The actor now computes:

```python
batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
usable_rollout_size = (rollout_size // batch_size_per_rank) * batch_size_per_rank
shuffle_id = torch.randperm(rollout_size, generator=g)[:usable_rollout_size]
```

If the rollout size is not divisible by `batch_size_per_rank`, the actor drops
the shuffled tail samples.

Example:

```text
rollout_size=360
batch_size_per_rank=48
usable_rollout_size=336
dropped=24
```

This keeps:

- static `actor.global_batch_size`
- static gradient accumulation
- stable per-rank batch shape
- no manual tuning per scene/run

Rank 0 prints a diagnostic message:

```text
[FSDPActor] Dropping tail rollout samples for static global batch: ...
```

## Trade-Off

The fix is intentionally conservative.

Pros:

- Minimal code change.
- Avoids run failures caused by variable rollout size.
- Keeps training batch size and gradient accumulation stable.
- Common behavior for minibatch training.

Cons:

- Some rollout samples are discarded.
- Drop fraction depends on scene/episode dynamics.
- It does not solve scene sampling imbalance by itself.

## Future Optimizations

### 1. Dynamic Effective Global Batch Size

Instead of dropping samples, choose an effective `batch_size_per_rank` each
iteration that divides the actual rollout size.

Example:

```text
rollout_size=360
configured batch_size_per_rank=48
dynamic batch_size_per_rank=45
effective_global_batch_size=90
```

Pros:

- Keeps all samples.

Cons:

- Gradient accumulation changes between iterations.
- Effective batch size changes across scenes/runs.
- More moving parts for training stability.

### 2. Fixed-Shape Rollout With Loss Mask

Keep rollout tensor shape fixed even after episodes terminate, and rely on
`loss_mask` to exclude inactive/dormant samples.

Pros:

- Stable rollout size.
- No sample dropping.
- Cleaner interaction with fixed global batches.

Cons:

- Needs careful handling so dormant samples do not affect reward, advantage,
  normalization, or loss.
- More code paths to audit.

### 3. Multi-Scene / Multi-EnvWorker Sampling

Current setup pins one EnvWorker to one scene. A scene with few episodes limits
active slots, e.g. `active_slots=14/18`.

Future direction:

- multiple EnvWorkers
- each worker pinned to a different scene
- balance active slots across scenes
- avoid over-sampling scenes with fewer/shorter episodes

This is the more complete solution for scene-level sampling balance.

### 4. Scene-Aware Sampling Accounting

Track per-scene effective samples:

```text
scene_id
active_slots
episode_count
valid_loss_mask_sum
sample_count_after_drop_last
drop_fraction
```

Use this to decide whether to:

- rotate scenes
- adjust scene weights
- refill exhausted workers
- report scene imbalance during training

## Current Recommendation

Keep automatic drop-last for now so training can proceed robustly.

Later, if sample efficiency or scene balance becomes a bottleneck, prioritize:

1. fixed-shape rollout with loss mask, or
2. dynamic effective global batch size, plus
3. multi-scene EnvWorker balancing.


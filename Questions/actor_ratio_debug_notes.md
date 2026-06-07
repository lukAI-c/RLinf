# Actor Ratio Debug Notes

## Purpose

This temporary debug instrumentation was added to diagnose why PPO `actor/ratio`
still explodes in QwenNav LoRA RFT even after excluding obvious actor/rollout
forward-path mismatch.

The current suspicion is that token-level logprob differences are being amplified
by `logprob_type: action_level`, where 64 response-token logprobs are summed
before computing:

```python
ratio = exp(new_action_logprob - old_action_logprob)
```

This means a moderate average token-level difference can become a very large
action-level ratio.

## Files Touched

- `rlinf/workers/actor/fsdp_actor_worker.py`

## Debug Code Added

### 1. `_debug_actor_ratio(...)`

Added inside `EmbodiedFSDPActor`, immediately before `run_training`.

This method prints one line per inspected micro-batch on actor rank 0 only.
It does not modify tensors or training behavior.

It reports:

- `token_log_ratio(mean/min/max)`: statistics of `new_token_logprob - old_token_logprob`
  on valid response tokens.
- `action_log_ratio(mean/min/max)`: statistics after reshaping by `action_dim`
  and summing token log ratios.
- `action_ratio(mean/max)`: `exp(action_log_ratio)`.
- `max_idx`: sample/chunk index with the largest action log ratio.
- `valid_tokens`: number of valid response tokens for the max-ratio sample.
- `loss_mask`: loss mask value for the max-ratio action.
- `old_sum` / `new_sum`: old and new action-level logprob sums.
- `tokens_head`: first response token ids for the max-ratio action.

Log prefix:

```text
[FSDPActor][ratio-debug]
```

### 2. Debug Counter in `run_training`

Added:

```python
actor_ratio_debug_prints = 0
```

The debug only considers actions where `loss_mask=True`. It prints the first 8
valid debug samples, and then continues printing only when the largest valid
action ratio in a micro-batch exceeds `actor.ratio_debug_threshold` (default
`100.0`):

```python
if self._debug_actor_ratio(..., printed_count=actor_ratio_debug_prints):
    actor_ratio_debug_prints += 1
```

This keeps logs small while focusing on samples that actually affect the PPO
loss/metrics.

### 3. `global_batch_idx`

The rollout dataloader loop was changed from:

```python
for train_global_batch in rollout_dataloader_iter:
```

to:

```python
for global_batch_idx, train_global_batch in enumerate(rollout_dataloader_iter):
```

This only provides a batch index for the debug log.

## How To Read The Debug Output

Example:

```text
[FSDPActor][ratio-debug] gb=0 mb=0 token_log_ratio(mean/min/max)=0.4200/-1.2000/2.1000 valid_action_log_ratio(mean/min/max)=26.8000/-3.0000/44.5000 valid_action_ratio(mean/max)=4.36e+11/2.12e+19 ...
```

Interpretation:

- If `token_log_ratio` is already huge, the issue is token-level old/new logprob
  mismatch.
- If `token_log_ratio` is moderate but `valid_action_log_ratio` is huge, the
  issue is action-level summation amplification.
- If `valid_tokens` is unexpectedly high or low, inspect `response_mask` and
  action JSON tokenization.
- If `loss_mask=0` for max-ratio samples but metrics are still affected, inspect
  loss mask shape/broadcasting before `policy_loss`.

## How To Remove Later

Remove these temporary debug-only pieces from `rlinf/workers/actor/fsdp_actor_worker.py`:

1. Delete the whole `_debug_actor_ratio(...)` method.
2. Delete `actor_ratio_debug_prints = 0` in `run_training`.
3. Change the dataloader loop back if desired:

```python
for train_global_batch in rollout_dataloader_iter:
```

or keep `enumerate(...)` if another debug still needs `global_batch_idx`.

4. Delete the call block:

```python
if self._debug_actor_ratio(..., printed_count=actor_ratio_debug_prints):
    actor_ratio_debug_prints += 1
```

## Related Non-Debug Change

The same file also contains a separate functional change for automatic
`drop_last` when `rollout_size` is not divisible by the per-rank global batch
size. That change is not part of this debug instrumentation and should not be
removed together with the ratio debug unless intentionally reverting it.

Its log prefix is:

```text
[FSDPActor] Dropping tail rollout samples for static global batch
```

# Legacy LaViRA Runtime Notes (Superseded)

> This document describes the retired sparse NumPy mapper path. It is retained
> only as migration history and must not be used as the production contract.
> The active architecture is documented in
> `docs/lavira_source_core_architecture.md`.

## Intent

This migration ports LaViRA's navigation mechanism from
`/home/clk/workspace/lavira-rft` main at `64e2a55` into a
Genesis/RLinf adapter. It does not import Habitat's evaluator loop. Ray,
EnvWorker, Decision-MaxRL, multiscene execution, and the existing reward to
gradient path remain unchanged.

## Runtime boundary

`rlinf/models/embodiment/qwen_nav/lavira_runtime/` owns the new state machine:

- `LaviraObservation` is the only RGB-D projection interface. It receives
  canonical `[front, left, behind, right]` RGB, depth, and yaw.
- `WaypointMemory` keeps all prior coordinate nodes independently of visual
  history, matching the source `visited_targets` lifetime.
- `LaviraNavigationController` owns `NEED_DECISION`, `NAVIGATING`, and
  `BACKTRACKING`, plus persistent local subgoals and the source evaluator's
  target timeout/arrival transitions.
- The former lightweight `lavira_controller` is permanently disabled. The
  complete semantic-map runtime is the only controller path for
  `lavira_runtime.enabled=true`.

The template repository is not modified.

## Reuse and adaptation audit

The runtime reuses the existing RLinf LaViRA port wherever the source contract
does not depend on Habitat:

- `lavira_map.FMMPlanner` and its coordinate/action conversion are the single
  FMM implementation. `LaviraSemanticMap.fmm_action()` delegates its final
  heading-to-action decision to `lavira_map._angle_and_direction`; it must not
  duplicate row/column cross-product math.
- `lavira_depth_utils` remains the single Genesis-to-Habitat RGB-D projection
  adapter and is shared by both the legacy occupancy controller and this
  runtime.
- The controller keeps a Genesis-specific state wrapper because the source
  evaluator owns Habitat stepping, prompt construction, and a blocking
  interaction loop. Those parts cannot be copied without bypassing RLinf's
  decision/logprob/replay contract.

There are two intentional Genesis adaptations rather than source copies:

- Collision footprints are painted from metric Genesis/Habitat XZ poses;
  upstream reconstructs the equivalent footprint from Habitat pose tensors.
- RGB-D map fusion is a sparse NumPy implementation of LaViRA's voxel-splat
  projection, avoiding its Habitat/Torch batch mapper while retaining the same
  5m local window, height band, and semantic traversibility rules.

The source VLM-waypoint execution path has no LRLR detector, repeated-subgoal
filter, or collision-count recovery. Those policies are intentionally absent.
Forward collisions only update the persistent collision map, after which FMM
replans against the updated traversibility map.

## Genesis observation contract

Genesis transports a **front** metric depth map through `wrist_images` and
four RGB views through `main_images` + `extra_view_images` in canonical
`[front, left, behind, right]` order. The runtime turns the agent physically
before it projects a left/right/behind waypoint, so its fresh front depth is
always paired with the matching yaw. This is why the launch configuration uses
`enable_depth_obs=true`, `enable_4dir_render=true`, and
`enable_4dir_depth_obs=false`.

## Decision and STOP boundary

Only a parsed model `stop=true` becomes environment `ACTION_STOP`. FMM arrival,
backtrack arrival, and the source 30-step subgoal timeout move the controller
to a new decision state and never enqueue STOP. Controller-generated
FMM primitives are emitted as non-decision replay steps, so only VLM
NAVIGATE/BACKTRACK/STOP responses receive policy logprobs and advantages.

The source evaluator does not call `env.step()` when it has no executable
action. RLinf's batched interface represents this with `ACTION_NOOP`: no
physics, elapsed step, path point, or navigation reward is produced, and the
slot requests another high-level decision on the next policy call.

Waypoint projection follows the source depth-backoff loop: if the projected
cell is blocked, subtract 0.1m from the depth image and project again. A failed
projection or unavailable primitive produces the same no-action transition;
it must never be converted into a controller-generated FORWARD or an invented
recovery prompt.

`backtrack to <id>` follows the source evaluator's action-first contract: once
the waypoint id parses, the controller executes the backtrack even if the model
also filled in explanatory `target`/`bbox_2d` fields. Those fields are ignored
by control and receive no geometry-format reward unless they follow the source
empty/zero convention. This prevents a recoverable model formatting lapse from
being converted into a forward fallback exactly when backtracking is needed.

## Enablement

The default is disabled. A runtime experiment must enable both model and
environment sides:

```bash
++actor.model.lavira_runtime.enabled=true \
++env.train.enable_depth_obs=true \
++env.train.enable_4dir_render=true \
++env.train.enable_4dir_depth_obs=false
```

For evaluation, apply the equivalent `env.eval.*` overrides. Start with one
environment on episode 259, then run four grouped environments, then a 2-3
optimizer-step RFT smoke test.

## Acceptance checks

- Side and rear targets are physically turned into the front view before
  depth/yaw projection.
- A planner return never terminates an episode.
- Forward collisions are painted into the map and affect the next FMM plan;
  they do not activate an additional recovery policy.
- Every prior waypoint within the source 6m radius remains a valid backtrack
  candidate independently of the visual-history frame limit.
- Controller replay adds no model logprob or Decision-MaxRL shape changes.

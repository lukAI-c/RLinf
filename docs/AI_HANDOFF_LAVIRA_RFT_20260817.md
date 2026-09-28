# LaViRA-on-RLinf RFT AI Handoff

> Snapshot date: 2026-08-17 (Asia/Shanghai)
>
> Main repository: `/home/clk/workspace/RLinf`
>
> Original LHX reference repository: `/home/lhx/workspace/lavira-code`
>
> This document is an operational handoff for the next coding AI. It separates
> current code from historical experiments and proposed work. Do not infer the
> current state from an old chat message or log directory name.

## 1. Non-Negotiable Rules

1. **Never modify the LHX repository.** `/home/lhx/workspace/lavira-code` is a
   read-only behavioral reference. Source-derived code lives under
   `RLinf/rlinf/third_party/lavira_rft`.
2. **Do not reset the RLinf worktree.** It is intentionally very dirty. Many of
   the active LaViRA/RFT files are untracked. A broad `git reset`, checkout, or
   clean would destroy current work.
3. **Measure current code, not remembered code.** Launcher flags override YAML.
   Log-directory names are not reliable descriptions of enabled behavior.
4. **Keep strict reproduction and robust training adaptations explicit.** Never
   silently describe an adapter heuristic as original LHX behavior.
5. **Use clean STOP as the binary RFT outcome.** Proximity at timeout is useful
   diagnostic/process information, but is not an outcome success.
6. **After every training run, audit gradients and behavior.** Finite losses and
   nonzero gradients do not prove the policy improved.
7. **Keep changes small.** The user explicitly rejects complex retry state
   machines and scene-specific patches unless a simpler measured alternative
   has failed.
8. **Performance changes require full-run A/B.** Local microbenchmarks have been
   misleading. Preserve fixed-input semantic equality and roll back changes that
   do not improve representative rollout wall time.
9. **Do not expose credentials.** One neighboring repository was observed with
   a credential embedded in its Git remote URL. Never paste remote URLs with
   credentials into logs, documents, or chat; rotate/redact the credential.

## 2. Executive Status

### 2.1 Objective

The project ports the LHX LaViRA navigation inference stack into RLinf so a
Qwen3.5-VL navigation checkpoint can be trained with online RFT in Genesis while
retaining the important LHX perception, semantic-map, waypoint, collision, and
FMM behavior.

The immediate objective is no longer just episode 609 overfitting. The current
formal experiment uses a frozen `checkpoint-1200` evaluation to split episodes
by difficulty, then trains on original full instructions and original episode
starts. The long-term objective is navigation improvement that generalizes
beyond one episode, especially reliable **clean STOP**.

### 2.2 Repository State

- Branch: `genesis-env-server`
- HEAD at snapshot: `98076e2` (`feat: improve GenArk navigation reward rollout`)
- Important recent commits:
  - `d1d3cb6`: GenArk LaViRA online RFT pipeline
  - `e535bd2`: DTG curriculum work
  - `8f195ed`: occupancy/FMM navigation core
  - `3f0112e`: multiscene configuration alignment
- The active implementation contains many modified and untracked files. Run
  `git status --short` before every edit and never assume HEAD contains the
  operational system.

### 2.3 Latest Formal Run

Latest run inspected:

```text
/home/clk/workspace/RLinf/logs/
  20260816-205104-sft1200-episode-tier-curriculum-rloo-rft
```

It resumed from an earlier `global_step_6`, completed through step 24, and saved
checkpoints at steps 8, 10, 12, 14, 16, 18, 20, 22, and 24. The final checkpoint
is:

```text
/home/clk/workspace/RLinf/logs/20260816-205104-sft1200-episode-tier-curriculum-rloo-rft/
  genark_grpo_qwen_multiscene/checkpoints/global_step_24
```

The run completed without a fatal error. The final rollout took about 28:09,
actor training about 713 seconds, weight synchronization about 63 seconds, and
the complete step about 2526 seconds. This is functional, but not a proof of
learning quality.

### 2.4 Latest Learning Evidence

Across inspected groups from steps 7-24:

- Episode 586 produced clean STOP groups with approximately `K=4/8`, `6/8`, and
  `6/8` on several visits.
- Episodes 1301, 824, 1133, 576, 207, and 469 generally produced `K=0/8` clean
  STOP groups.
- Many environment lines report `success=1` for proximity/evaluator success
  while the group diagnostic still reports `success_count=0`. This is expected
  under the fail-closed contract: only explicit STOP inside the threshold is a
  clean outcome.
- The last group, episode 469, had `K=0`, one wrong STOP, seven no-STOP
  trajectories, reward std about 0.423, auxiliary advantage std about 0.191,
  and grad norm about 0.0048. It had a valid auxiliary update but no binary
  outcome learning signal.

**Current conclusion:** the chain runs and episode 586 supplies learnable STOP
examples, but the curriculum has not demonstrated stable transfer of clean STOP
to the other episodes. Most groups still update from reference-path auxiliary
advantages rather than clean-STOP outcome differences.

## 3. Current Formal Training Configuration

The source of truth is:

```text
/home/clk/workspace/RLinf/launch_episode_curriculum_rft.sh
```

At this snapshot its important defaults are:

| Item | Current value |
|---|---|
| Base checkpoint | `/home/lhx/workspace/test_model/output/qwen3.5_v2_unfreeze_vit_nav_target/v0-20260709-081633/checkpoint-1200` |
| Scene | `mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb` only |
| Manifest | `episode_curriculum_sft1200_final5.json` |
| Env placement | physical GPU `3`, process 0 |
| Rollout placement | physical GPUs `6-7`, two ranks, nominally 4 envs each |
| Actor placement | physical GPU `5` |
| Group size | 8 |
| Actor global batch | 32 |
| Actor micro batch | 1 |
| Actor LR | `2.5e-7` |
| Gradient checkpointing | false |
| Rollout temperature | 1.0 |
| Max training steps | 24 |
| Max rollout primitive steps | 200 |
| Max rollout decisions | 20 |
| Save interval | 2 |
| Early stop KL | 0.03 |
| Advantage | `decision_rloo_aux_rloo` |
| Reference-path coefficient | 1.0 |
| Auxiliary RLOO coefficient | 0.5 |
| GSAM placement | local inside rollout by default; remote service disabled |
| GSAM visualization | false |
| Map visualization | false; save interval 50 if enabled |
| Termination shadow judge | disabled |

### 3.1 Curriculum

The 100-episode frozen-evaluation manifest contains:

- 8 `clean_seed`
- 46 `stop_frontier`
- 39 `nav_frontier`
- 7 `hard`

Classification rules in the artifact:

```text
clean_seed:    active_clean_count >= 3
stop_frontier: active_clean_count < 3 and meaningful_reach_count >= 2
nav_frontier:  active_clean_count < 3, meaningful_reach_count < 2, and
               (meaningful_reach_count >= 1 or mean_dtg_progress > 0)
hard:          active_clean_count == 0, meaningful_reach_count == 0, and
               mean_dtg_progress <= 0
```

The current launcher restricts training to one scene, so only these 11 manifest
episodes are eligible:

| Episode | Tier | Meaningful reach count in 5 frozen trials |
|---:|---|---:|
| 586 | clean_seed | 4 |
| 1301 | stop_frontier | 2 |
| 824 | stop_frontier | 2 |
| 1133 | nav_frontier | 1 |
| 207 | nav_frontier | 1 |
| 432 | nav_frontier | 0, positive mean progress |
| 469 | nav_frontier | 0, positive mean progress |
| 576 | nav_frontier | 1 |
| 550 | hard | 0 |
| 559 | hard | 0 |
| 705 | hard | 0 |

Stage schedule:

```text
steps  1-6:  clean_seed    {clean_seed: 1.00}
steps  7-12: stop_frontier {clean_seed: 0.25, stop_frontier: 0.75}
steps 13-18: nav_frontier  {clean_seed: 0.20, stop_frontier: 0.30,
                            nav_frontier: 0.50}
steps 19+:   hard_mix      {clean_seed: 0.15, stop_frontier: 0.20,
                            nav_frontier: 0.45, hard: 0.20}
```

The curriculum must persist its current stage/episode through bootstrap reset
and checkpoint resume. Relevant tests are
`test_episode_curriculum.py` and
`test_embodied_runner_curriculum_checkpoint.py`.

### 3.2 Reward and Outcome Contract

The current formal launcher disables most legacy rewards:

```text
geo=0, nav terminal off, nav path=0, endpoint=0, GSAM=0, nDTW=0,
format/json/structure/field/length/bbox=0, SR=0,
wrong-stop penalty=0, parse-fail penalty=0, no-stop penalty=0,
generic process reward off.
```

Enabled signals:

1. **Reference-path normalized potential**
   - progress coefficient: 1.0
   - lateral penalty: 1.0
   - per-decision delta clip: 0.25
2. **Missed-STOP auxiliary event**
   - enabled
   - penalty: 0.25
   - `terminate_on_missed_stop=true`
3. **Binary RLOO outcome**
   - one only for explicit clean STOP within success distance
   - proximity at timeout maps to zero
4. **Advantage combination**
   - bounded decision RLOO for clean STOP outcome
   - a separate RLOO over auxiliary/reference-path rewards
   - auxiliary coefficient 0.5

For a binary outcome `Y` and group size `N`, decision RLOO is:

```text
A_i = Y_i - mean(Y_j for j != i)
```

For `K=1, N=8`, this gives `+1` to the successful trajectory and `-1/7` to
each failure. For `K=0` or `K=8`, all outcome advantages are zero. This is the
central unresolved issue: homogeneous groups cannot learn from binary outcome
RLOO, even when all eight are successful.

### 3.3 Prompt and Visual Query Mode

The model currently uses the original seven-field `lavira_waypoint` contract:

```json
{
  "progress_analysis": "...",
  "reasoning_plan_action": "...",
  "planning": "...",
  "action": "navigate to forward",
  "stop": false,
  "stair": false,
  "target": "..."
}
```

Do not claim the current launcher is strict LHX. It additionally enables:

```text
grounded_sam.canonicalize_lavira_waypoint_query=true
grounded_sam.reject_scene_region_source_waypoint=true
waypoint_scene_box_area_ratio=0.65
waypoint_scene_edge_margin_ratio=0.02
```

Therefore the Qwen output schema is LHX-like, but the target passed to DINO may
be canonicalized, and scene-sized boxes may be rejected. These are RLinf
execution adaptations. The YAML default for canonicalization is false; the
launcher overrides it to true.

## 4. End-to-End Execution Chain

### 4.1 High-Level Flow

```text
Genesis / GenArk observations
  -> EnvWorker group lifecycle and episode curriculum
  -> two rollout ranks (4 env slots each)
  -> QwenNavPolicy per-slot runtime
  -> 12-turn physical panorama when requested
  -> GroundingDINO + RepViT-SAM semantic perception
  -> source-derived LHX Semantic_Mapping fusion
  -> Qwen high-level 7-field decision
  -> GroundingDINO target localization
  -> metric-depth projection / depth backoff
  -> LHX-derived map target + collision map + FMM
  -> primitive TURN_LEFT / TURN_RIGHT / MOVE_FORWARD / STOP
  -> trajectory rewards and clean-STOP labels
  -> decision RLOO + auxiliary RLOO
  -> FSDP actor replay/update
  -> weights synchronized to rollout workers
```

### 4.2 Environment Layer

Primary files:

- `rlinf/envs/genark/genark_env.py`
  - R2R episode reset and observations
  - Genesis primitive stepping
  - atomic 12-view panorama contract
  - path/DTG/nDTW/success diagnostics
  - reference-path reward and missed-STOP event
  - episode curriculum selection/state
- `rlinf/envs/genark/genesis_backend.py`
  - backend abstraction and observation transport
- `rlinf/envs/genark/genesis_multiscene.py`
  - scene loading, grouping, and multi-scene orchestration
- `rlinf/envs/genark/genesis_server.py`
- `rlinf/envs/genark/genesis_zmq_server.py`
  - process/RPC boundaries for Genesis

Important contracts:

- Camera: 640x480, HFOV 79 degrees, camera height 0.88 m.
- Depth is metric and capped/configured at 5 m before the LHX preprocessing
  contract is applied.
- Panorama is 12 real left turns with RGB/depth frames and consumes primitive
  step budget.
- Completed slots must publish `episode_active=false`; policy must return NOOP
  for inactive slots and must not continue Qwen inference.
- Episode/trial identity must include both `episode_id` and `trial_id`; episode
  ID alone is not unique under repeated evaluation.

### 4.3 Policy and Perception Layer

Primary directory:

```text
rlinf/models/embodiment/qwen_nav/
```

Key files:

- `qwen_nav_policy.py`
  - main batched policy bridge
  - per-env runtime ownership
  - Qwen processor/generation inputs
  - active-slot filtering
  - visual replay tensors for actor training
  - optional termination shadow collection
- `waypoint_runtime.py`
  - waypoint decision flow, history, query normalization, and policy helpers
- `action_parser.py`
  - seven-field output parsing and action constants
- `prompts.py`
  - LHX-style prompt, optional canonical prompt, optional termination-shadow
    prompt; do not change prompt without a frozen A/B
- `grounded_sam.py`
  - GroundingDINO queries, bbox selection/rejection, RepViT-SAM masks, local and
    remote execution
- `canonical_targets.py`
  - optional exact alias mapping, including examples such as
    `archway exit -> archway`, `doorway to hallway -> doorway`, and
    `hallway floor -> hallway`
- `lavira_depth_utils.py`
  - camera intrinsics, bbox bottom-center pixel, metric projection, backoff
- `policy_diagnostics.py`
  - action/parse/grounding audit counters

Visual roles:

- Qwen chooses direction/action and emits a free-text target.
- DINO grounds that target in the selected RGB image.
- SAM supplies semantic masks for mapping; strict LHX waypoint geometry uses
  the highest-confidence DINO bbox bottom-center rather than a SAM bottom band.
- Depth maps the pixel to a world/map waypoint.

### 4.4 Runtime Controller

Primary directory:

```text
rlinf/models/embodiment/qwen_nav/lavira_runtime/
```

Key files:

- `navigation_controller.py`
  - state machine: scan, decision, turn, navigate, backtrack
  - waypoint acceptance and depth backoff
  - 0.75 m reached test and 15-primitive local goal timeout
  - collision/FMM primitive selection
- `observation.py`
  - typed RGB/depth/pose observation and 79-degree projection contract
- `source_port.py`
  - interface to source-derived mapper/FMM backend
- `waypoint_memory.py`
  - failed waypoint, layered history, backtrack, second-chance state
- `semantic_map.py`
  - earlier RLinf NumPy/sparse mapper path; not the current `map_backend=source`
    implementation
- `fmm_planner.py`
  - earlier RLinf planner path; distinguish it from source backend planner
- `wm_snapshot.py`
  - default-off research snapshot support

Critical lifecycle fix:

- When a new waypoint is accepted, `goal_just_set=true`.
- The controller must execute at least one source FMM primitive before allowing
  the 0.75 m reached check for that new goal.
- Without this guard, a depth-backoff target at 0.52-0.66 m is marked reached in
  the same tick, causing panorama -> Qwen -> DINO loops with almost no motion.
- This historical regression expanded one rollout from roughly 30 minutes to
  more than 64 minutes and increased panorama work by about 3.4x.

Current code detail: inspect `navigation_controller.py` before claiming exact
ordering. At this snapshot the timeout check appears before the reached check in
the current function, while `goal_just_set` protects the first primitive. Compare
this explicitly against the current LHX loop when changing lifecycle behavior.

### 4.5 Source-Derived LHX Backend

Primary directory:

```text
rlinf/third_party/lavira_rft/
```

- `source_core.py`: RLinf adapter around the source-derived modules
- `loader.py`: import/path isolation
- `source/vlnce_baselines/map/mapping.py`: Semantic_Mapping
- `source/vlnce_baselines/models/Policy.py`: map processing/traversibility
- `source/vlnce_baselines/models/fmm_planner.py`: FMM planner
- `source/vlnce_baselines/utils/depth_utils.py`: depth/point-cloud utilities
- `source/vlnce_baselines/utils/map_utils.py`: map helpers
- `source/habitat_extensions/pose_utils.py`: pose conversions

The source backend is the accepted current mapper/FMM path. Preserve its
per-frame fusion order. Panorama frames update mapper state sequentially. The
`publish_planner_state` flag controls when a new planner snapshot is exposed;
panorama-local mapper updates and outer-loop planner publication are distinct.

The original comparison points in LHX are mainly:

- `vlnce_baselines/ZS_Evaluator_mp.py`
- `vlnce_baselines/utils/prompts_vln.py`
- `vlnce_baselines/map/mapping.py`
- `vlnce_baselines/models/Policy.py`
- `vlnce_baselines/models/fmm_planner.py`

Always read both versions before editing. Source-derived files in RLinf are not
proof that the complete outer loop is identical; RLinf still uses a batched
state machine and NOOP slot semantics.

### 4.6 Worker and Training Layer

- `rlinf/workers/env/env_worker.py`
  - owns grouped trajectories, completed-slot padding, bootstrap/reset,
    decision rewards, clean STOP/proximity labels, group diagnostics
- `rlinf/algorithms/advantages.py`
  - Decision-MaxRL, decision RLOO, and auxiliary RLOO estimators
- `rlinf/algorithms/registry.py`
  - advantage type registration
- `rlinf/workers/actor/fsdp_actor_worker.py`
  - reconstructs visual-language actor inputs, recomputes logprobs, applies
    PPO-style loss, logs gradient/ratio/KL diagnostics
- `rlinf/workers/rollout/vllm/vllm_embodied_worker.py`
  - vLLM rollout generation and embodied-policy bridge
- `rlinf/workers/rollout/hf/huggingface_worker.py`
  - HuggingFace rollout/debug alternative
- `rlinf/runners/embodied_runner.py`
  - synchronize -> rollout -> advantage -> actor update -> checkpoint loop
- `rlinf/data/embodied_io_struct.py`
  - tensors and metadata exchanged between stages

### 4.7 Actor Visual Replay Integrity

A serious historical bug truncated `pixel_values` when actual visual patches
exceeded a stored capacity, while leaving `image_grid_thw` describing the full
image. Rollout behavior could look correct while actor replay trained on corrupt
visual inputs.

The current fix is fail-closed:

- preserve all patches when capacity is sufficient;
- reject a grid/patch count mismatch;
- raise instead of silently truncating capacity overflow;
- optionally validate with `RLINF_VALIDATE_VISUAL_REPLAY`.

Relevant test: `tests/unit_tests/test_qwen_nav_visual_replay.py`.

Interpretation consequence: learning conclusions from older runs produced before
this fix are not trustworthy even if their rollout maps looked normal. The
2026-08-16 run uses the corrected chain.

## 5. LHX Alignment: What Is and Is Not Aligned

### 5.1 Accepted Alignment

- `checkpoint-1200` and original seven-field LaViRA prompt format
- original image aspect ratio and processor pixel bounds
- 12-turn panorama and F/L/B/R selection order
- metric depth preprocessing in the 0.1-5 m contract
- HFOV 79 degrees and camera height 0.88 m
- DINO bbox bottom-center waypoint convention in source/strict path
- source-derived Semantic_Mapping, map processing, collision, and FMM modules
- 0.75 m waypoint reached threshold
- 15 primitive local waypoint lifetime
- new goal receives one FMM primitive before reached check
- continuous history cadence and backtrack/second-chance concepts
- done slots stop inference
- `(episode_id, trial_id)` identity for repeated evaluation

### 5.2 Structural Differences That Remain

- LHX runs a synchronous single-environment Python loop. RLinf has batched
  policies, independent env slots, distributed rollout ranks, and NOOP for an
  inactive slot.
- `NavigationController` is an RLinf state-machine rewrite; it is not direct
  execution of `ZS_Evaluator_mp.py`.
- The source mapper/FMM modules are copied/adapted, but invocation and snapshot
  publication are mediated by `source_core.py`.
- Policy history and failed-waypoint lifetimes cross RLinf policy ticks.
- Current launcher canonicalizes free targets and rejects scene-region boxes;
  original LHX passes the raw target to DINO and chooses the highest-confidence
  bbox.
- Genesis rendering is not Habitat rendering. RGB appearance, depth rasterization,
  collision physics, and episode step accounting must be separately validated.
- `ACTION_NOOP` is an RLinf transport semantic, not an LHX action.

### 5.3 Strict-vs-Robust Configuration Debt

The code supports strict-like and robust adaptations, but they are not yet
packaged as two obvious top-level modes. A future cleanup should expose, without
changing algorithms:

```text
strict_lhx:
  raw target, highest-confidence bbox, source fallback behavior,
  source snapshot timing

robust_train:
  canonical query, scene-region rejection, explicit stale-snapshot policy,
  low-frequency audit output
```

Do not perform this refactor while debugging learning unless a focused test
proves configuration equivalence.

## 6. Problems Found and Their Status

### 6.1 Fixed P0 Problems

1. **Depth units mixed between env.step and simulator observation**
   - normalized depth was incorrectly treated as metric depth;
   - caused target projections millimeters from the agent;
   - current contract uses explicit metric preprocessing.
2. **Projection used implicit 105-degree HFOV instead of configured 79**
   - current runtime passes explicit intrinsics.
3. **Instruction double nesting**
   - prompts previously received strings like `['instruction']`;
   - current descriptions are flat strings per env.
4. **Done slots continued inference**
   - current `episode_active`/NOOP contract stops further model calls.
5. **Repeated episode metrics overwritten by episode ID**
   - current repeated evaluation uses `(episode_id, trial_id)`.
6. **GroundedSAM target used SAM bottom-band point**
   - source/strict waypoint path uses DINO highest-confidence bbox bottom-center;
     SAM remains semantic-map mask provider.
7. **Waypoint bbox-only result dropped**
   - bbox-only detection can enter navigation/history instead of degrading to a
     direction primitive.
8. **New waypoint immediately marked reached**
   - fixed by `goal_just_set` first-FMM-action guard.
9. **Actor visual replay truncation**
   - fixed with exact grid/patch validation and fail-closed behavior.
10. **Curriculum state lost through bootstrap reset/resume**
    - persistence logic and tests were added.

### 6.2 Current P0/P1 Problems

1. **Clean STOP outcome support is sparse.** Most current episodes produce
   homogeneous `K=0/8` groups. Binary RLOO gives zero outcome advantage.
2. **Reference-path auxiliary reward can move the policy without teaching STOP.**
   It creates gradients in all-failure groups, but those gradients may reinforce
   movement while STOP remains absent.
3. **Curriculum effectiveness is not established.** Episode 586 improves, but
   transfer to stop/nav/hard tiers is weak and high variance.
4. **Base SFT capability is limited.** Prior audits found weak direction/STOP
   behavior and poor support for exact target scenes; RFT cannot learn a missing
   behavior unless exploration produces discriminative samples or another
   principled signal supplies it.
5. **Current launcher is not strict LHX despite its prompt mode.** Canonical query
   and scene-box rejection must be accounted for in every comparison.
6. **Single-scene curriculum is not the full 100-episode curriculum.** It sees 11
   episodes in Z6MF only.
7. **Performance remains expensive.** A typical rollout is roughly 28-31 minutes,
   and an actor update can add around 12 minutes. Systems optimization must not
   alter perception/map/action outputs.
8. **Controller snapshot/reached/timeout order needs an explicit trace-level
   invariant test.** Historical claims of exact outer-loop alignment were too
   strong.

## 7. Historical Experiments and Lessons

### 7.1 Canonical Schema and Portal/Area Geometry

An expanded canonical schema (`target_class`, region, attributes) and custom
portal/area geometry were explored. They overloaded the 4B model, created
scene-sized hallway/corridor detections, and diverged from LHX. The active prompt
was returned to the seven-field form. Some execution-only canonical target
mapping remains enabled in the current curriculum launcher.

Lesson: normalize DINO queries only at a narrow adapter boundary. Do not require
a new output schema without a matching SFT model.

### 7.2 GroundingDINO Tensor Batch

A true two-image tensor batch produced about 2.02x speedup in a fixed-input local
benchmark with nearly identical boxes/masks. In a complete rollout it did not
improve wall time and at points made it worse because CPU preprocessing,
cross-env synchronization, SAM serialization, and scheduling dominated. The
user ultimately requested the DINO batch path be rolled back to the original
single-image behavior for precision and simplicity.

Lesson: do not resurrect tensor batching from the local 2x number alone. Prove
end-to-end benefit with the current two-rank topology and exact semantic A/B.

### 7.3 SAM Batch

RepViT-SAM's predictor interface did not provide a stable multi-image encoder
batch. Same-image multi-box decoder batching was considered, but SAM was not the
dominant end-to-end bottleneck. No accepted general SAM batching optimization is
currently part of the baseline.

### 7.4 GroundedSAM Multi-Service RPC

Multiple per-GPU services removed single-service queueing but duplicated CUDA
contexts and did not implement true model batch. Four services on one GPU could
oversubscribe CPU/GPU resources. Current launcher defaults to local GSAM inside
the two rollout ranks; remote service code remains optional and disabled.

### 7.5 Map Fusion Across-Env Threading

Thread-pool parallel map fusion had good-looking offline numbers but no full
rollout gain, and could be slower because each PyTorch/OpenMP call already used
internal threading. Execution-framework changes were rolled back.

Lesson: preserve per-frame/source fusion order and avoid Python thread pools
around internally parallel kernels without whole-run evidence.

### 7.6 Demand-Driven Rendering and Panorama Cache

Ordinary steps rendered only front view and reused exact-pose panoramas. A
40-step smoke completed in 188 seconds, but the full run reached only about step
106 after 25.5 minutes and could not meet the 30-minute gate. It also exposed a
`scan_valid` lifecycle gap. The branch was restored byte-for-byte and rejected.

Evidence: `.engramory-memory/demand-render-rollout-optimization-rejected.md`.

### 7.7 Lifecycle Regression

The incorrect same-tick 0.75 m reached check caused:

```text
depth backoff -> 0.56 m goal -> immediate reached -> panorama -> Qwen/DINO
-> another 0.56 m goal -> immediate reached
```

One abnormal run had about 173 high-level decisions, 180 panoramas, 2160
panorama semantic frames, and only 81 physical movement primitives, compared
with a healthy baseline around 50 decisions, 53 panoramas, 636 semantic frames,
and 1563 movement primitives. This is why lifecycle call order is both a quality
and throughput invariant.

### 7.8 Episode 609 Diagnosis

Repeated analysis showed that initial-direction errors were primarily Qwen
visual/topological decisions, not DINO alone. In representative 8-way sampling,
only one trajectory chose the correct initial right exit while most chose left
or behind. DINO could correctly detect an archway in the wrong selected view.
Ambiguous targets such as `hallway floor` and `archway exit` were a secondary
problem. Even a correct first exit often failed to maintain the correct hallway
direction and no clean STOP was produced reliably.

Lesson: a correct bbox inside a Qwen-selected wrong view does not fix high-level
direction reasoning. Keep LA (Qwen direction) and VA (DINO grounding) audits
separate.

### 7.9 Reverse Curriculum and Synthetic Sub-Instructions

Several episode-609 curricula changed starts along the GT path or manufactured
sub-instructions. They created distribution mismatch with the single full
instruction used in SFT and produced confusing map/trajectory results. The user
requested their removal.

Current curriculum invariants:

- original episode start;
- original full instruction;
- no `goal_approach -> hall_transition -> original` sequence;
- no stop repair resampling;
- no teacher STOP suffix;
- no extra BC correction.

### 7.10 RLOO vs MaxRL

The temporary engineering fixes that capped rare MaxRL advantages, suppressed
all-failure process reward, and skipped zero-advantage optimizer steps were
removed. They reduced symptoms but did not solve missing successful samples.

Decision RLOO was introduced as a distinct estimator, not a silent MaxRL change.
It bounds the rare-success signal but still collapses to zero for homogeneous
groups. This is mathematically correct, not an implementation bug.

### 7.11 WAM / World-Model Research

Motion-causal research is separate from active RFT:

- a metric-depth future-motion probe passed an initial gate;
- matched RGB-only/no-future controls later showed only about 7.75% benefit and
  forward-collision AUROC around 0.483;
- a Qwen causal auxiliary experiment worsened EPE and was stopped;
- geometry-action human-annotation gate remains pending.

Do not enable WAM, counterfactual foresight, or termination shadow by default in
the formal RFT until their independent gates pass.

## 8. Memory Store

Memory root:

```text
/home/clk/workspace/RLinf/.engramory-memory/
```

Files:

1. `MEMORY.md`
   - pointer-only index; do not put detailed state directly in this file.
2. `lavira-rft-experiment-ledger-2026-07.md`
   - accepted alignment state;
   - canonical/geometry experiments;
   - GSAM RPC and DINO batching results;
   - Map Fusion experiment;
   - lifecycle regression;
   - render experiment;
   - actor-data audits;
   - episode 609 reward/learning curves;
   - clean-STOP/RLOO decisions.
3. `demand-render-rollout-optimization-rejected.md`
   - complete reason the demand-render/cache branch must not be reused as the
     baseline.
4. `rft-gradient-policy-audit.md`
   - mandatory post-run signal, stability, collapse, and behavioral audit;
   - fail-closed clean STOP contract;
   - RLOO mathematical definition;
   - simplicity constraint.
5. `navigation-wam-motion-causal-research.md`
   - gate-ordered motion/world-model research and negative results.

Read all five before proposing another optimization. Memory is evidence and
preference context, not authority over current code; verify every claim against
the launcher and implementation.

## 9. Recent Conversation Summary

The recent work progressed through these phases:

1. **LHX/RLinf audit:** found depth-unit, HFOV, instruction nesting, done-slot,
   repeated-trial metric, image aspect-ratio, prompt, GroundedSAM point, and
   semantic-map discrepancies.
2. **Source alignment:** copied/adapted LHX Semantic_Mapping, Policy map
   processing, RepViT-SAM, collision and FMM into `third_party/lavira_rft` and
   built a stateful RLinf controller around them.
3. **Map diagnosis:** traced accumulated obstacle fusion, closed corridors, stale
   traversibility, depth-backoff, and map-target/FMM fields using episodes 259
   and 609.
4. **Visual decision work:** separated Qwen direction selection from DINO target
   grounding; explored canonical vocabulary and portal/area geometry; returned
   the model output to the original seven fields because no matching canonical
   SFT model exists.
5. **Throughput work:** tried multi-service GSAM, DINO batch, SAM batch ideas,
   Map Fusion threading, split GPU placement, and demand rendering. Full-run
   evidence rejected most of these despite favorable microbenchmarks.
6. **Lifecycle alignment:** restored 0.75 m reached and 15-step waypoint limits,
   then fixed the newly introduced same-tick reached regression with
   `goal_just_set`.
7. **RFT signal work:** compared MaxRL and bounded RLOO, made clean STOP the only
   binary outcome, added mandatory advantage/gradient diagnostics, and retained
   reference-path auxiliary reward to avoid fully zero gradients.
8. **Curriculum work:** abandoned GT-start reverse curriculum and synthetic
   sub-instructions; evaluated checkpoint-1200 five times per episode; built the
   four-tier manifest; fixed curriculum state persistence; trained the current
   single-scene schedule.
9. **Actor-chain validation:** discovered and fixed visual patch truncation in
   replay; old RFT learning results before this fix are not reliable.
10. **Termination research:** proposed and partially implemented a DITA-style
    shadow judge, then explicitly paused it. It remains disabled and collect-only
    when used; it must not modify action or reward in formal training.
11. **Current evidence:** the corrected formal run completed to step 24. Episode
    586 yields clean STOP samples; most other episodes remain all-failure for the
    binary outcome. Curriculum transfer is therefore unproven.

An external-paper/code comparison for arXiv `2607.20785` was requested near the
end of the prior work. Do not assume its conclusions are integrated here unless
a separate evidence note or code change is found.

## 10. `/home/clk` Project and Code Topology

`/home/clk/workspace` is a symlink-backed workspace (currently resolving into
NVMe storage). Treat model/data/log/cache directories differently from source
repositories.

### 10.1 Source Repositories and Their Roles

- `/home/clk/workspace/RLinf`
  - active project; all formal RFT work happens here.
- `/home/clk/workspace/Genesis`
  - upstream Genesis simulator checkout.
- `/home/clk/workspace/genark`
  - standalone Genesis navigation integration, MP3D caches, episode datasets,
    and simulator assets used by RLinf.
- `/home/clk/workspace/lavira-rft`
  - older/alternate LaViRA RFT working tree; not the current formal source of
    truth.
- `/home/clk/workspace/LaWAM`
  - world/action-model research repository.
- `/home/clk/workspace/NaVid-VLN-CE`
  - navigation baseline/reference repository.
- `/home/clk/workspace/Uni-NaVid`
  - navigation baseline/reference repository.
- `/home/clk/workspace/habitat-lab`
  - Habitat simulator/evaluation stack.
- `/home/clk/workspace/habitat-gs`
  - Gaussian-splatting Habitat work; audit/redact its Git remote credential.
- `/home/clk/workspace/TurboVLA`
  - VLA research repository.
- `/home/clk/workspace/awr`
  - AWR-related research/reference code.
- `/home/clk/workspace/3DGS`
  - 3D Gaussian Splatting source/research.
- `/home/clk/workspace/template`
  - design notes such as GRPO, prompt, rollout/reward plans; reference only.

### 10.2 Data, Assets, Outputs, and Temporary Areas

- `model_zoo`: model assets/checkpoints
- `mp3d_nerfstudio`, `mp3d_opennav100`: datasets/derived scene content
- `mesh_out`, `navmesh_out`: generated geometry/navigation meshes
- `results`: evaluation outputs
- `third_party`: shared external dependencies
- `ray_tmp*`, `tmp*`: runtime temporary directories; do not delete while a run
  is active
- `worldvln_ab`: world/VLN A/B artifacts
- `r`: miscellaneous research/output workspace; inspect before use

This is a functional inventory, not permission to delete. Before cleaning disk,
check running PIDs/tmux/Ray sessions and open log paths.

### 10.3 RLinf Top-Level Structure

```text
RLinf/
  rlinf/                 core package
  examples/embodiment/   Hydra configs and training entrypoint
  scripts/               evaluation, alignment, conversion, profiling tools
  tools/                 research/offline data-generation tools
  tests/                 unit and end-to-end tests
  docs/                  design, status, and handoff documents
  assets/                GroundedSAM and experiment assets
  data/                  local datasets/artifacts
  logs/                  run directories and checkpoints
  launch_*.sh            experiment launchers
  .engramory-memory/     durable evidence/preferences, git-ignored
```

### 10.4 `rlinf/` Package Structure

```text
rlinf/
  agents/          generic reasoning/search agents
  algorithms/      reward and advantage estimators
  data/            rollout/embodied tensor structures and datasets
  envs/            GenArk, Habitat, robot/simulation environments
  hybrid_engines/  FSDP/Megatron/vLLM/SGLang execution engines
  models/          embodied policies, including qwen_nav
  research/        default-off counterfactual/world-model studies
  runners/         top-level training loops
  scheduler/       cluster, placement, channels, collectives, workers
  third_party/     isolated source-derived dependencies, including LaViRA
  utils/           metrics, checkpoints, conversion, resharding
  workers/         actor, env, rollout, inference, reward, SFT workers
```

### 10.5 Key Configs and Launchers

- `launch_episode_curriculum_rft.sh`: current formal launcher
- `launch_episode_609_overfit_rft.sh`: historical episode-609 experiment; do not
  assume its flags match the formal launcher
- `examples/embodiment/config/genark_grpo_qwen_multiscene.yaml`: base RFT config
- `examples/embodiment/config/model/qwen_nav.yaml`: model/runtime defaults
- `examples/embodiment/config/env/genark_r2r.yaml`: environment defaults
- `examples/embodiment/config/episode_curriculum_sft1200_final5.json`: frozen
  five-trial curriculum artifact
- `examples/embodiment/train_embodied_agent.py`: entrypoint

### 10.6 Alignment and Diagnostic Scripts

- `scripts/compare_fixed_lavira_input.py`: fixed-input perception/runtime A/B
- `scripts/compare_lavira_mapper_frames.py`: mapper frame replay/comparison
- `scripts/compare_primitive_traces.py`: primitive trace comparison
- `scripts/run_lhx_original_open100.sh`: original LHX evaluation wrapper
- `scripts/run_lhx_rlinf_habitat_decision_alignment.sh`: decision alignment
- `scripts/verify_genesis_rgb_depth_parity.py`: Habitat/Genesis sensor parity
- `scripts/calibrate_genesis_rgb.py`: RGB calibration
- `scripts/grounded_sam_server.py`: optional remote perception service
- `scripts/build_episode_curriculum.py`: frozen-eval manifest generation
- `scripts/analyze_rft_reference_path_reward.py`: reward/trajectory audit
- `scripts/analyze_termination_shadow.py`: disabled shadow-judge analysis
- `scripts/aggregate_aligned_nav_metrics.py`: repeated-trial metric aggregation

### 10.7 Research-Only Code

- `rlinf/research/cf_foresight/`
- `tools/cf_foresight_*`
- `tools/lavira_world_model_data/`
- `rlinf/models/embodiment/qwen_nav/wam_motion/`
- `future_frame_projection.py`
- `navigation_wm_client.py`
- `wm_state_collector.py`

These paths are not the formal RFT default. Keep them behind explicit switches.

## 11. Logs, Maps, Metrics, and Checkpoints

A run directory can contain:

```text
train.log                         combined Ray/worker/training output
lavira_maps/                      source/runtime map visualization
grounded_sam/                     bbox/mask/patch audits when enabled
lavira_alignment/.../projection_audit.jsonl
fmm_fields/                       source FMM visualizations
termination_shadow/               optional shadow-judge artifacts
genark_grpo_qwen_multiscene/checkpoints/global_step_N/actor
```

Map conventions used in prior analysis:

- yellow cross: visual/Qwen waypoint target
- green: GT/reference path overlay, only when the correct episode ID is supplied
- blue: actual FMM-controlled agent trajectory

Missing GT overlay is often a visualization configuration issue, not an
algorithm change. `MAP_EPISODE_ID` defaults to null in the current launcher to
avoid drawing episode 586 over a different curriculum episode.

Repeated-evaluation metric contract:

- `all_episode_metrics.json`: one row per `(episode_id, trial_id)`
- `per_episode_summary.json`: mean over trials for each episode
- `avg_metrics.json`: mean over all trial rows

## 12. Mandatory Post-Run Audit

For every completed run, report all four layers.

### 12.1 Signal Quality

- group reward mean/std
- ACR and low-reward-std rates
- clean STOP count `K`, `p_hat`, all-failure rate
- `advantages_valid_mean`, std, absolute mean
- max-absolute / mean-absolute ratio
- effective sample ratio
- positive/negative/nonfinite fractions

### 12.2 Update Stability

- policy loss and total loss
- grad norm
- ratio and ratio absolute deviation
- approximate KL and early-stop status
- clip fraction
- action-token NLL and logprob std

Approximate KL may be slightly negative because it is a sample estimator; a
large or persistent negative value should trigger metric/formula inspection.
Do not read `action_token_nll` as full vocabulary entropy.

### 12.3 Collapse Evidence

- forward/left/right/behind/backtrack/STOP distribution
- parse failure and JSON/field conformance
- bbox/GSAM/fallback rates
- checkpoint-to-checkpoint change, not one minibatch only

### 12.4 Behavior

- clean STOP SR and proximity separately
- nDTW, final DTG, minimum DTG, best progress, along-track progress
- wrong initial direction recovery
- wrong STOP/no STOP/timeout counts
- compare to frozen checkpoint-1200 and immediately preceding checkpoint on the
  same episode set and sampling settings

## 13. Recommended Next Work

### 13.1 First: Establish a Trustworthy Behavioral Curve

1. Evaluate frozen `checkpoint-1200`, `global_step_6`, `12`, `18`, and `24` on
   the same 11 Z6MF episodes with at least five fixed trial seeds.
2. Report clean STOP, proximity, no STOP, wrong STOP, nDTW, best progress, and
   action distribution per tier.
3. Do not infer curriculum success from training groups because the sampled
   episode changes each step.
4. If only episode 586 improves, record the curriculum as memorization/nontransfer.

### 13.2 Second: Isolate STOP Outcome from Movement Auxiliary Learning

Run a small controlled comparison from checkpoint-1200:

```text
A: current reference-path coef 1.0, auxiliary coef 0.5
B: same rollout data and outcome, lower auxiliary influence
```

The objective is to test whether auxiliary movement updates suppress STOP, not
to introduce another curriculum or repair mechanism. Keep the episode schedule,
prompt, mapper, and controller identical.

### 13.3 Third: Consider a Continuous Terminal Outcome Only After A/B

The binary clean-STOP RLOO estimator is unidentifiable for homogeneous groups.
A principled minimal extension is a continuous terminal navigation outcome plus
an explicit clean-STOP bonus, followed by within-group leave-one-out/standardized
advantage. This can distinguish two all-failure trajectories without teacher
actions or repair sampling.

Before implementation, answer:

- Is the signal based only on environment-observable navigation quality?
- Does it reward getting close without allowing timeout proximity to masquerade
  as clean success?
- Is STOP still explicitly more valuable than merely entering the goal radius?
- Does it duplicate the existing reference-path auxiliary reward?

Do not add it until the current checkpoint curve proves the binary/auxiliary
failure mode.

### 13.4 Termination Shadow Judge

The DITA-style shadow code currently exists but is disabled. If resumed, it must
first be diagnostic only:

- measure direct Qwen STOP probability;
- collect hidden states at clean-terminal, missed-terminal, wrong-STOP, and
  nonterminal states;
- train episode-disjoint balanced probes;
- compare probe AUC to native LM-head STOP-logit AUC;
- do not alter prompt, action, reward, or termination during the diagnostic.

Avoid hard AUC thresholds. Treat representation and LM-head gaps as continuous,
noisy evidence. Do not add repair resampling, teacher suffix BC, or wrong-STOP
teacher actions without a new justified design.

### 13.5 Throughput

Do not restart with DINO batching, Map Fusion threading, or demand rendering.
Profile a complete current run first. The accepted performance target is a real
200-step rollout under 30 minutes with identical decisions/maps/actions on fixed
inputs and a meaningful improvement over the immediate baseline.

## 14. First Actions for the Next AI

Run these in order before changing code:

```bash
cd /home/clk/workspace/RLinf
git branch --show-current
git rev-parse --short HEAD
git status --short
sed -n '1,440p' launch_episode_curriculum_rft.sh
cat .engramory-memory/MEMORY.md
cat .engramory-memory/rft-gradient-policy-audit.md
tail -200 logs/20260816-205104-sft1200-episode-tier-curriculum-rloo-rft/train.log
```

Then inspect:

```text
rlinf/models/embodiment/qwen_nav/qwen_nav_policy.py
rlinf/models/embodiment/qwen_nav/waypoint_runtime.py
rlinf/models/embodiment/qwen_nav/lavira_runtime/navigation_controller.py
rlinf/third_party/lavira_rft/source_core.py
rlinf/envs/genark/genark_env.py
rlinf/workers/env/env_worker.py
rlinf/algorithms/advantages.py
rlinf/workers/actor/fsdp_actor_worker.py
```

Before a training launch:

1. Check `nvidia-smi` and confirm physical placements are actually free.
2. Check `tmux ls`, active Ray sessions, launcher PIDs, and log paths.
3. Confirm `BASE_MODEL`, `RFT_RESUME_DIR`, and
   `RFT_INITIAL_GLOBAL_STEP` are mutually consistent.
4. Confirm no old process is writing the target log or temp directory.
5. Run the focused unit tests below.

Suggested focused suite:

```bash
/home/clk/miniconda3/envs/genesis/bin/python -m pytest \
  tests/unit_tests/test_episode_curriculum.py \
  tests/unit_tests/test_embodied_runner_curriculum_checkpoint.py \
  tests/unit_tests/test_genark_success_contract.py \
  tests/unit_tests/test_genark_reference_path_reward.py \
  tests/unit_tests/test_genark_panorama_contract.py \
  tests/unit_tests/test_qwen_nav_visual_replay.py \
  tests/unit_tests/test_qwen_nav_lavira_waypoint_parser.py \
  tests/unit_tests/test_lavira_source_core.py \
  tests/unit_tests/test_lavira_runtime.py
```

Some environments may lack optional `supervision`/GroundedSAM dependencies. A
dependency failure must be separated from an assertion failure; do not report a
suite as passing without listing skips/import failures.

## 15. Definition of Done for the Next Milestone

A next training change is successful only if all of the following hold:

1. Original full instructions and starts are preserved.
2. Fixed-input Qwen/DINO/projection/map/FMM outputs remain unchanged unless that
   component is the explicit experimental variable.
3. Actor visual replay grid/patch validation passes.
4. The post-update fixed evaluation improves clean STOP or navigation behavior
   on more than episode 586.
5. Wrong STOP and no-STOP are reported separately.
6. Advantage concentration, KL, gradient norm, and action distribution remain
   healthy.
7. Full rollout wall time does not regress unexpectedly.
8. The result is written into `.engramory-memory` with exact log/checkpoint paths,
   and rejected code is cleanly removed or disabled behind an explicit switch.

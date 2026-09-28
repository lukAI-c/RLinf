# Robostral-Inspired RLinf RFT Implementation Plan (Revision 4)

Snapshot: 2026-09-12. Endpoint-only is closed as a *reward* experiment.
Do not spend more runs on densify / held-out of \(S=-\max(2,DTG)\) alone.
Next active experiment is STOP as a second stage, then CISPO.

Revision 3 snapshot was 2026-09-04.

Primary references:

- Robostral Navigate: <https://arxiv.org/html/2607.20785v3>
- CISPO / MiniMax-M1: <https://arxiv.org/html/2506.13585> (still deferred)
- Standing decision: `docs/ROBOSTRAL_ENDPOINT_DECISION_20260823.md`
- Train-set freeze: `docs/ROBOSTRAL_ENDPOINT_SCORE_REPORT_20260818.md`
- Held-out of 6-ep model: `docs/ROBOSTRAL_HELDOUT_INTERIM_20260824.md`
- 4-scene step-4 held-out: `docs/ROBOSTRAL_POOL55_STEP4_HELDOUT.md`

This is an adaptation of Robostral Navigate, not a paper reproduction.
GenArk has no geodesic query. Distance is Habitat-coordinate Euclidean DTG.

## 1. Standing Decision

Keep a single endpoint score. Do not add more reward channels.

```text
S_i = -max(2 m, final Euclidean DTG_i)
```

- `clean_stop_bonus=0`
- STOP remains in the JSON schema and env termination
- `QWEN_NAV_LOSS_MASK_INCLUDE_STOP=0` (stop value tokens off PPO)
- `terminate_on_missed_stop=false`
- Old shaping off: reference-path, missed-STOP aux, process, nDTW, nav-terminal, `rloo_aux_coef=0`
- Advantage: `decision_terminal_grpo` on same-episode groups of 8
- Start every formal run from `checkpoint-1200`
- Launcher: `launch_robostral_terminal_rft.sh` (do not overwrite `launch_episode_curriculum_rft.sh`)

Episode **586** is eval-only (anti-forgetting). Do not train it.

## 2. Non-Negotiable Boundaries

Do not modify `/home/lhx/workspace/lavira-code`, LHX-derived mapper / fusion /
FMM / controller, camera geometry, 12-frame panorama, 15-primitive lifecycle,
seven-field prompt, parser, GroundedSAM selection, episode starts or
instructions.

Do not introduce teacher STOP, STOP-repair resampling, BC suffixes,
historical-success replay, synthetic sub-instructions, cross-episode GRPO
baselines, or extra reward terms in the same run.

Do not `git reset` / checkout / clean the RLinf worktree.

Proximity at timeout is a diagnostic, never RFT success.

## 3. What Is Proven

Frozen eval, same 6 Z6 episodes × 5 trials, `checkpoint-1200` vs endpoint
`global_step_6` (`logs/20260817-205113-sft1200-robostral-terminal-rft` after
cursor/bonus-0 rerun path; numbers from
`docs/ROBOSTRAL_ENDPOINT_SCORE_REPORT_20260818.md`):

| metric | SFT-1200 | endpoint step-6 |
|---|---:|---:|
| mean final DTG | 9.56 m | 7.31 m |
| nDTW | 0.083 | 0.104 |
| path length | 17.70 m | 16.92 m |
| SPL | 0.163 | 0.217 |
| clean STOP | 6/30 | 6/30 |

All five unsaturated train episodes improved DTG. Trajectory shape and
efficiency moved with DTG. Clean STOP did not keep degrading.

Old curriculum + `decision_rloo_aux_rloo` is not the formal strategy: 15/18
groups were clean-STOP all-failure; most updates came from the auxiliary path.

Engineering that must stay:

- `episode_terminal_score` frozen atomically with `episode_success`
- GRPO: `A = (S - mean) / (sample_std + 1e-6)`, `keepdim` on std, multiple
  groups of 8 allowed in one batch
- Actor not collocated with vLLM when batch > 8
- `episode_blocklist` for 586 and held-out IDs

## 4. What Failed or Must Not Be Repeated

| Item | Evidence | Action |
|---|---|---|
| `clean_stop_bonus=1` as main objective | Gate A.5: DTG better, STOP suppressed | Do not restore |
| STOP credit routing / dual advantage | Replaced by bonus=0 + stop-token off PPO | Do not implement |
| Same 5-ep seed repeats | Hung 9 days, 0 steps; does not answer transfer | Do not rerun |
| Other-scene pool expansion (4 scenes / 55 eps) | Step-4 held-out DTG drop driven by episode **705** only; nDTW worse; 207/432 worse | Do not finish 12 steps; do not stack more MP3D scenes to fake Z6 transfer |
| Claiming STOP improved | 5/6 clean STOPs are still 586 | Eval metric only |
| Geodesic / navmesh DTG | Not in GenArk | Out of scope |
| Missed-STOP truncation | Fights endpoint continuation | Keep false |
| CISPO as next change | Cannot create signal; confounds target vs optimizer | Deferred |

Held-out of the 6-ep model (`207, 432, 550, 559, 705`): DTG 11.29 → 10.71 m.
That is the current generalization fact.

## 5. P0 Status (closed 2026-09-12)

**Name:** Z6 train-set densify, frozen held-out gate.

**Question (original):** Can endpoint-only transfer to unseen Z6 episodes if we
train more Z6 episodes, without touching the held-out set?

**Close-out:** The reward formula itself is validated on the train set
(6-ep freeze DTG 9.56 → 7.31 m). Further generalization tests of this
*same* scalar are not an innovation. The 12×1-visit coverage check
improved held-out DTG at step 6 (−1.47 m) then regressed at step 10
(+1.51 m); clean STOP on held-out stayed 0/25. Do **not** launch another
densify or 12×1-visit run. Do **not** use more held-out DTG of
endpoint-only as a keep gate.

Hard-pool is the same reward with different sampling. Skip it until STOP
stage and CISPO are done.

### 5.1 Frozen eval set (immutable for this experiment)

```text
held-out: 207, 432, 550, 559, 705
anchor:   586
protocol: 5 trials, temperature 1.0, terminate_on_missed_stop=false
SFT baseline already exists:
  logs/20260824-014436-robostral-gate-a5-sft-heldout
```

Do not train these six IDs. Do not change this list mid-run.

### 5.2 Train set

Pin scene `mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb`.

OpenNav-100 only has 11 Z6 episodes. After removing the six eval IDs, five
remain (`824, 1301, 1133, 576, 469`). That set already failed to transfer.
**Do not relaunch that 5-ep assignment as the next formal run.**

Required before launch:

1. Inventory every Z6 episode available in the local datasets / caches beyond
   OpenNav-100.
2. Train pool = all Z6 IDs except the six frozen eval IDs.
3. If the usable train pool is still ≤ 8 episodes, **stop and report**. Do not
   pad with other scenes. The next decision is then data, not another 6-step
   overfit.

Same-episode groups of 8. Blocklist the six eval IDs in `episode_blocklist`.

### 5.3 Throughput (required, not optional)

The 32-env / 2-vLLM layout spent ~2.5 h/step waiting on generation
(`max_num_seqs=8`, 16 envs per rank, 12-frame panorama). Genesis itself was
~6 min.

For this experiment:

- `total_num_envs = 8` (one GRPO group) unless rollout world size ≥ 4
- Rollout GPUs ≥ 2, `max_num_seqs` ≥ envs per rank
- Actor on its own GPU(s); never collocate with vLLM
- Target: step wall time in the same band as the old 8-env run (~40 min), not
  2.5 h

Do not drop the 12-frame panorama.

### 5.4 Schedule and gates

Start from `checkpoint-1200`. Save every 2 steps.

Every 6 actor updates, freeze and evaluate the immutable held-out+anchor set.
Compare to the SFT held-out dump (recompute endpoint score with bonus=0).

Report per episode: clean STOP, wrong STOP, proximity-without-STOP, mean/min
final DTG, endpoint score, nDTW, path length, SPL.

**Keep going only if** held-out mean final DTG improves by ≥ 1.0 m vs SFT
**and** at least two held-out episodes improve, **and** 586 stays 5/5 clean
STOP. A single-episode mean (the 705 failure mode) does not count.

**Stop if** held-out DTG worsens for two consecutive evals, or 586 clean STOP
drops, or nDTW falls while DTG “improves” only via longer wandering.

Do not restore STOP bonus to rescue a failed held-out.

### 5.5 What this experiment does not answer

- Habitat clean-STOP SR as a training objective
- Whether CISPO beats PPO
- Whether hard-pool scheduling beats cyclic Z6 sampling

## 6. Remaining Experiments (P0 densify no longer a blocker)

1. **STOP as a second stage (active).** Separate run. Init from endpoint
   `logs/20260823-132620-sft1200-robostral-terminal-rft/.../global_step_6`.
   Launcher: `launch_robostral_stop_stage.sh`.
   `RFT_CLEAN_STOP_BONUS=1.0`, `QWEN_NAV_LOSS_MASK_INCLUDE_STOP=1`.
   Train the same 5 unsaturated Z6 episodes (not 586 / held-out).
   Gate: clean STOP up on those train episodes without DTG collapse;
   586 stays 5/5. Held-out DTG is a diagnostic, not a keep/stop of this
   stage. Do not mix this into another densify run.
2. **CISPO** after STOP stage. Isolated loss A/B. The codebase does not
   yet expose `algorithm.loss_type=actor_cispo` for this trainer; implement
   that flag without rewriting the existing `actor` loss. Same endpoint
   init, bonus=0, STOP off PPO, so optimizer is the only change.
3. **Hard-pool on Z6** (deferred past STOP + CISPO). Same reward, different
   sampling. Trial-level 5-eval, exclusive buckets, train only
   `mixed_support` and `navigation_hard`. Not next.

## 7. Permanently Deferred

- Robostral pixel pointing / metric displacement fallback (new SFT schema)
- Robostral diffusion low-level policy (conflicts with LHX FMM)
- Prefix-tree attention as a substitute for rollout signal
- Hundreds-of-env async orchestration
- DITA judge, STOP classifiers, teacher correction, BC repair
- Mixing nDTW / reference-path / process into the training score
- Re-enabling missed-STOP truncation
- Per-trajectory loss-length normalization unless diagnostics first show bias
- Gate B (terminal-GRPO vs `decision_rloo_aux_rloo` on the old 6-ep assignment)
  — superseded by the freeze vs SFT-1200 on those six, and by the held-out
  failure. Do not spend a 12-step A/B there.
- Finishing `pool55-split3` steps 5–12

## 8. Coding-Agent Delivery Order (current)

1. STOP stage: `launch_robostral_stop_stage.sh`, 8 env, 6 steps, freeze-eval
   the 5 train IDs + 586 (clean STOP / DTG). Held-out 5 is optional diagnostic.
2. CISPO isolated A/B (implement `actor_cispo` if missing, then one run).
3. Hard-pool only after those two.

Each stage remains independently reversible. Do not modify LHX-derived
code or add a heuristic fallback. Do not relaunch Z6 densify.

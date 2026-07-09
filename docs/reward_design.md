# Reward Design for Decision-Level Navigation RFT

## Overview

The total reward assigned to each LLM decision $d$ is:

$$r_d = r_d^{\text{nav}} + r_d^{\text{fmt}}$$

where $r_d^{\text{nav}}$ is the navigation reward and $r_d^{\text{fmt}}$ is the format reward.

---

## Reward Components

### 1. Navigation Reward $r_d^{\text{nav}}$

Two modes are supported, selected via `reward_mode`.

---

#### Mode A: Geodesic Progress $r_d^{\text{geo}}$ *(current)*

$$r_d^{\text{geo}} = \sum_{t=t_{\text{start}}}^{t_{\text{end}}} \left( \mathcal{D}_{\text{geo}}(s_{t-1}) - \mathcal{D}_{\text{geo}}(s_t) \right) + \mathcal{B}_{\text{succ}} \cdot \mathbf{1}[\text{STOP} \wedge \mathcal{D}_{\text{geo}}(s_{t_{\text{end}}}) < \delta_{\text{succ}}]$$

- $\mathcal{D}_{\text{geo}}(s_t)$: geodesic distance to goal at env step $t$
- $\mathcal{B}_{\text{succ}} = 2.5$: sparse success bonus
- $\delta_{\text{succ}} = 3.0\,\text{m}$: success distance threshold
- The sum telescopes: equals $\mathcal{D}_{\text{geo}}(s_{t_{\text{start}}-1}) - \mathcal{D}_{\text{geo}}(s_{t_{\text{end}}})$, i.e., total geodesic progress over the macro-action

**Properties**: dense signal at every env step; unit = meters; typical range $\pm 0.3$–$1.5\,\text{m}$ per decision.

---

#### Mode B: nDTW + Success Rate Delta $r_d^{\text{ndtw}}$ *(optional)*

$$\boxed{r_d^{\text{ndtw}} = \lambda_{\text{nDTW}} \cdot \Delta\eta_d + \lambda_{\text{SR}} \cdot \Delta\rho_d}$$

where the nDTW similarity $\eta$ and its per-decision delta $\Delta\eta_d$ are:

$$\eta(P, \hat{P}) = \exp\!\left( -\frac{\text{DTW}(P,\, \hat{P})}{|\hat{P}| \cdot \delta_{\text{succ}}} \right), \qquad \Delta\eta_d = \eta(P_{0:t_{\text{end}}},\, \hat{P}) - \eta(P_{0:t_{\text{start}}-1},\, \hat{P})$$

and the success rate delta $\Delta\rho_d$ is:

$$\Delta\rho_d = \rho_{t_{\text{end}}} - \rho_{t_{\text{start}}-1}, \qquad \rho_t = \mathbf{1}[\text{STOP} \wedge \mathcal{D}_{\text{geo}}(s_t) < \delta_{\text{succ}}]$$

| Symbol | Value | Meaning |
|---|---|---|
| $\lambda_{\text{nDTW}}$ | 1.0 | nDTW delta weight |
| $\lambda_{\text{SR}}$ | 10.0 | Success rate delta weight |
| $P_{0:t}$ | — | Agent path up to step $t$; all intermediate positions appended at each env step |
| $\hat{P}$ | — | Ground-truth reference path |
| $\delta_{\text{succ}}$ | 3.0 m | Success distance threshold |

**Computation timing** (decision-level rollout, `decision_level_ndtw=true`):

One LLM inference produces a macro-action of $K$ env steps. Rather than computing $\eta$ at every env step and summing the deltas (which telescopes to the same value when $\gamma=1$), we compute $\eta$ **once per decision** at flush time:

| Event | Action |
|---|---|
| Each env step $t \in [t_{\text{start}}, t_{\text{end}}]$ | Append $s_t$ to $P$; compute $\Delta\rho_t$ (accumulated into $\Delta\rho_d$) |
| Decision flush | Compute $\eta(P_{0:t_{\text{end}}}, \hat{P})$ once via fastdtw; assign $r_d^{\text{ndtw}}$ |

**Why this is equivalent** (telescoping, $\gamma=1$):
$$\sum_{t=t_{\text{start}}}^{t_{\text{end}}} \bigl(\eta(P_{0:t}, \hat{P}) - \eta(P_{0:t-1}, \hat{P})\bigr) = \eta(P_{0:t_{\text{end}}}, \hat{P}) - \eta(P_{0:t_{\text{start}}-1}, \hat{P}) = \Delta\eta_d$$

**Benefits**: fastdtw calls reduced from $K$ to $1$ per decision; credit assignment granularity aligned with LLM decision. Path $P$ still includes all intermediate positions, so detours within a macro-action remain penalized by DTW.

---

### 2. Format Reward $r_d^{\text{fmt}}$

$$r_d^{\text{fmt}} = \lambda_{\text{fmt}} \cdot \mathbf{1}[\text{JSON parse succeeds}]$$

$$\lambda_{\text{fmt}} = 0.5$$

**Purpose**: breaks the cold-start problem. The model receives gradient signal for producing valid JSON output even before learning to navigate. Applied per env step within the decision window; accumulated into $r_d^{\text{fmt}}$ alongside navigation reward.

**Caution**: $\lambda_{\text{fmt}}$ dominates $r_d^{\text{geo}}$ in early training ($\lambda_{\text{fmt}} \gg |\Delta\mathcal{D}_{\text{geo}}|$ per step). Recommended schedule:

| Training stage | $\lambda_{\text{fmt}}$ |
|---|---|
| Early (cold-start) | 0.5 |
| Mid | 0.1 |
| Late (optional) | 0.0 |

---

## Evaluation Metrics (not used in training)

Computed at episode end; reported in logs only.

| Metric | Formula | Meaning |
|---|---|---|
| Success | $\mathbf{1}[\text{STOP} \wedge \mathcal{D}_{\text{geo}}(s_T) < \delta_{\text{succ}}]$ | Binary navigation success |
| SPL | $\text{Success} \times \dfrac{\ell^*}{\max(\ell^*, \ell)}$ | Success weighted by path length |
| DTG | $\mathcal{D}_{\text{geo}}(s_T)$ (m) | Distance to goal at episode end |
| nDTW | $\eta(P_{0:T},\, \hat{P})$ | Trajectory similarity to GT |
| sDTW | $\eta(P_{0:T},\, \hat{P}) \times \text{Success}$ | Success-weighted nDTW |

where $\ell^*$ is the initial geodesic distance (optimal path length) and $\ell$ is the actual path length.

---

## Reward Signal Summary

$$r_d = \underbrace{r_d^{\text{nav}}}_{\text{navigation}} + \underbrace{r_d^{\text{fmt}}}_{\text{format}}$$

| Mode | $r_d^{\text{nav}}$ formula | Signal density | Cold-start |
|---|---|---|---|
| $r^{\text{geo}}$ *(current)* | $\mathcal{D}_{\text{geo}}(s_{t_{\text{start}}-1}) - \mathcal{D}_{\text{geo}}(s_{t_{\text{end}}})$ | Dense (every step) | ✓ (with $r^{\text{fmt}}$) |
| $r^{\text{ndtw}}$ | $\lambda_{\text{nDTW}}\,\Delta\eta_d + \lambda_{\text{SR}}\,\Delta\rho_d$ | Medium (trajectory quality) | ✓ (with $r^{\text{fmt}}$) |

---

## GRPO Advantage Computation

Within each group of $G = 3$ envs sharing the same episode/task, the advantage for decision $d$ of env $i$ is:

$$A_{i,d} = \frac{R_i - \bar{R}}{\sigma_R + \epsilon}, \quad \bar{R} = \frac{1}{G}\sum_{j=1}^{G} R_j, \quad \sigma_R = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(R_j - \bar{R})^2}$$

where $R_i = \sum_{d=1}^{D} r_{i,d}$ is the cumulative episode return for env $i$, and $\epsilon = 10^{-6}$.

The PPO-clip objective for each decision step is:

$$\mathcal{L}^{\text{CLIP}}(\theta) = -\mathbb{E}\!\left[\min\!\left(\frac{\pi_\theta(a_d|s_d)}{\pi_{\theta_{\text{old}}}(a_d|s_d)} A_{i,d},\; \text{clip}\!\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}, 1-\varepsilon, 1+\varepsilon\right) A_{i,d}\right) \cdot \mathbf{1}[\text{loss\_mask}_d]\right]$$

Blank-padded decisions (dormant env steps) have $\text{loss\_mask}_d = 0$ and do not contribute to the loss.

---

## Current Reward Profile Design

As of the LaViRA / GenArk RFT overfit experiments, reward selection is no
longer driven by a fixed step-based curriculum.  The old sequence
`format -> geometry -> local_nav -> full_nav` was useful while the base model
could not reliably emit the required LaViRA waypoint schema, but it became a
poor fit once a format-stable checkpoint was available:

- Re-running from a checkpoint repeatedly re-entered early curriculum stages.
- `local_nav` rewarded DTG progress and wrong-stop avoidance, but kept
  `sr = 0`, so correct STOP was not explicitly rewarded.
- Format and geometry shaping started to compete with the actual navigation
  objective.

The current code therefore uses an explicit `reward_profile`:

```yaml
reward_profile: nav              # default for online RFT
# reward_profile: format_learning # only for cold-start format warm-up
```

### `format_learning`

This profile is only for cold-start checkpoints or prompt/schema changes.  It
teaches the model how to produce valid LaViRA waypoint outputs, not how to
navigate.

| Component | Role |
|---|---|
| JSON / struct / field-format reward | Main positive signal |
| bbox / point reward | Geometry-format shaping |
| parse-fail penalty | Strong format correction |
| DTG / nDTW / SR / wrong-stop | Disabled |

### `nav`

This is the main task-learning profile.  It assumes the checkpoint already emits
mostly valid actions and focuses reward on navigation and correct STOP.

| Component | Role |
|---|---|
| `sr_coef` | Positive reward for correct STOP near the goal |
| `wrong_stop_penalty` | Penalty for premature STOP far from the goal |
| `decision_dtg_coef` | Dense decision-level progress toward the goal |
| `ndtw_coef` | Path-quality shaping against the reference trajectory |
| `process_reward_enabled` | Optional DTG best-progress / regression shaping |
| format / bbox rewards | Low-weight guardrails against schema regression |
| GroundedSAM reward | Disabled by default; diagnostics only unless explicitly enabled |

The intended training flow is:

1. Use `format_learning` only when the model cannot reliably produce the
   expected output schema.
2. Resume from a format-stable checkpoint with `reward_profile: nav`.
3. Do not use step counters to decide whether the model should learn formatting
   or navigation; choose the profile based on checkpoint capability.

The legacy `reward_curriculum` config is retained only for backward
compatibility.  New GenArk / LaViRA RFT runs should prefer `reward_profile`.

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

"""
GenArk vectorized gym environment for RLinf.

Wraps Genesis batch-renderer + Matterport3D NavMesh physics into a gym.Env
that is compatible with RLinf's EmbodiedRunner pipeline.

Observation space (per env):
  rgb               : uint8  (3, H, W)      — first-person RGB camera
  instruction_ids   : int64  (max_seq_len,) — tokenised instruction
  instruction_mask  : bool   (max_seq_len,) — attention mask

Action space (per env):
  int64 scalar: 0=stop, 1=forward 0.25 m, 2=turn_left 30°, 3=turn_right 30°

Reward:
  dense  : prev_geo_dist - curr_geo_dist  (progress toward goal)
  sparse : +success_bonus when STOP called within success_distance of goal
"""

from __future__ import annotations

import copy
import gzip
import json
import math
import os
from pathlib import Path
from typing import Optional, Union

import gym
import numpy as np
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from rlinf.envs.genark.genesis_backend import (
    GenesisSimBackend,
    GenesisLocalBackend,
    _calculate_initial_yaw,
    _hab_to_genesis,
    _genesis_to_hab,
)


# ---------------------------------------------------------------------------
# Episode / scene loader
# ---------------------------------------------------------------------------

def _load_episodes(episodes_file: str) -> list[dict]:
    """Load R2R-CE-style episode JSON (plain or gzipped)."""
    path = Path(episodes_file)
    if not path.exists():
        raise FileNotFoundError(f"Episodes file not found: {episodes_file}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(str(path), "rt") as f:
        data = json.load(f)
    # Habitat dataset format: {"episodes": [...]}
    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    # Flat list format used by genark OpenNav files
    return data


def _group_by_scene(episodes: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for ep in episodes:
        sid = ep["scene_id"]
        groups.setdefault(sid, []).append(ep)
    return groups


def _filter_overfit_episode(episodes: list[dict], cfg) -> list[dict]:
    """Optionally restrict the pool to one fixed episode for RL sanity checks."""
    overfit_cfg = getattr(cfg, "episode_overfit", None)
    if overfit_cfg is None or not bool(getattr(overfit_cfg, "enabled", False)):
        return episodes

    target_scene = getattr(overfit_cfg, "scene_id", None)
    target_ep = getattr(overfit_cfg, "episode_id", None)
    if target_scene is None or target_ep is None:
        raise ValueError(
            "episode_overfit.enabled=true requires episode_overfit.scene_id and "
            "episode_overfit.episode_id."
        )

    target_scene = str(target_scene)
    target_ep = str(target_ep)
    filtered = [
        ep for ep in episodes
        if str(ep.get("scene_id", "")) == target_scene
        and str(ep.get("episode_id", "")) == target_ep
    ]
    if not filtered:
        sample = [
            (str(ep.get("scene_id", "")), str(ep.get("episode_id", "")))
            for ep in episodes[:5]
        ]
        raise ValueError(
            "episode_overfit: requested episode not found: "
            f"scene_id={target_scene!r}, episode_id={target_ep!r}. "
            f"First available examples: {sample}"
        )
    if len(filtered) > 1:
        print(
            f"[GenArk][episode-overfit] WARNING: found {len(filtered)} matching "
            f"records for scene={target_scene} ep={target_ep}; using all matches.",
            flush=True,
        )
    ep = filtered[0]
    instr = (ep.get("instruction") or {}).get("instruction_text", "")
    print(
        f"[GenArk][episode-overfit] enabled: scene={target_scene} "
        f"episode_id={target_ep} matches={len(filtered)} "
        f"instruction={instr!r}",
        flush=True,
    )
    group_size = int(getattr(cfg, "group_size", 1) or 1)
    total_num_envs = int(getattr(cfg, "total_num_envs", group_size) or group_size)
    n_groups = max(1, math.ceil(total_num_envs / max(group_size, 1)))
    if len(filtered) < n_groups:
        filtered = [copy.deepcopy(filtered[i % len(filtered)]) for i in range(n_groups)]
        print(
            f"[GenArk][episode-overfit] duplicated fixed episode to "
            f"{len(filtered)} group entries for total_num_envs={total_num_envs}, "
            f"group_size={group_size}.",
            flush=True,
        )
    return filtered


class EpisodeBalancer:
    """Scene-internal episode balancer: selects next episode by (seen_count, success_rate) lexicographic order.

    seen_count = number of times the episode was assigned to a GRPO group.
    success_rate = EMA of per-trajectory success outcomes.
    Cold-start: seen=0, sr=0.0 → unseen episodes are highest priority (coverage-first).

    Curriculum: if curriculum_dtg_start is set, only episodes with Euclidean
    start-to-goal distance <= (curriculum_dtg_start + passes * curriculum_dtg_step)
    are considered. Falls back to the full pool if no episodes fit the window.
    """

    def __init__(self, episodes: list[dict], rng, ema_alpha: float = 0.3,
                 curriculum_dtg_start: float | None = None,
                 curriculum_dtg_step: float = 5.0):
        self._eps   = {e.get("episode_id", i): e for i, e in enumerate(episodes)}
        self._seen  = {k: 0   for k in self._eps}
        self._sr    = {k: 0.0 for k in self._eps}
        self._rng   = rng
        self._alpha = ema_alpha
        self._last  = None
        self._passes = 0  # number of full coverage sweeps completed

        # Pre-compute Euclidean start→goal distance for each episode (curriculum use).
        self._dtg: dict = {}
        for k, e in self._eps.items():
            sp = np.array(e.get("start_position", [0.0, 0.0, 0.0]), dtype=float)
            gp = np.array((e.get("goals") or [{}])[0].get("position", [0.0, 0.0, 0.0]), dtype=float)
            self._dtg[k] = float(np.linalg.norm(sp - gp))

        self._curriculum_dtg_start = curriculum_dtg_start  # None = disabled
        self._curriculum_dtg_step  = float(curriculum_dtg_step)

    def _curriculum_max_dtg(self) -> float:
        if self._curriculum_dtg_start is None:
            return float("inf")
        return self._curriculum_dtg_start + self._passes * self._curriculum_dtg_step

    def next_episode(self) -> tuple[dict, bool]:
        """Return (episode_dict, new_pass).

        new_pass=True when this call pushed the global minimum seen_count up by 1
        (i.e. every episode has now been covered at least N+1 times).
        """
        cands = [k for k in self._eps if k != self._last] or list(self._eps)

        # Curriculum filtering: restrict to episodes within current DTG window.
        max_dtg = self._curriculum_max_dtg()
        if max_dtg < float("inf"):
            easy = [k for k in cands if self._dtg[k] <= max_dtg]
            if easy:
                cands = easy
            # else: fallback to full pool (don't starve at scene startup)

        min_key = min((self._seen[k], self._sr[k]) for k in cands)
        tied = [k for k in cands if (self._seen[k], self._sr[k]) == min_key]
        k = tied[int(self._rng.integers(len(tied)))]
        prev_min = min(self._seen.values())
        self._seen[k] += 1
        self._last = k
        new_pass = min(self._seen.values()) > prev_min
        if new_pass:
            self._passes += 1
        return self._eps[k], new_pass

    def record(self, ep_id, success: bool) -> None:
        """Update EMA success rate for an episode after a trajectory completes."""
        if ep_id in self._sr:
            self._sr[ep_id] = (1 - self._alpha) * self._sr[ep_id] + self._alpha * float(success)

    def mark_seen(self, ep_id) -> None:
        """Pre-credit an episode that was assigned during initial layout."""
        if ep_id in self._seen:
            self._seen[ep_id] += 1


# ---------------------------------------------------------------------------
# Simple tokenizer placeholder — replace with real tokenizer when available
# ---------------------------------------------------------------------------

def _tokenize_instruction(text: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Placeholder byte-level tokenizer (ASCII codepoints).
    Replace with the real UniNaVid tokenizer once available.
    Returns (token_ids, attention_mask) as int64 / bool numpy arrays.
    """
    ids = np.array([ord(c) for c in text[:max_len]], dtype=np.int64)
    pad_len = max_len - len(ids)
    ids = np.pad(ids, (0, pad_len))
    mask = np.zeros(max_len, dtype=bool)
    mask[: len(text[:max_len])] = True
    return ids, mask


# ---------------------------------------------------------------------------
# GenarkVecEnv
# ---------------------------------------------------------------------------

class GenarkVecEnv(gym.Env):
    """
    Vectorised Genesis navigation environment for RLinf.

    num_envs environments run in the same Genesis scene; all share the same
    navmesh but can have different episodes. Episodes are drawn from a
    flat JSON list (genark OpenNav format or Habitat gzipped dataset).

    Constructor signature matches other RLinf envs (HabitatEnv, FrankaSimEnv).
    """

    # Actions
    STOP        = 0
    MOVE_FORWARD = 1
    TURN_LEFT   = 2
    TURN_RIGHT  = 3

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info=None,
        backend: GenesisSimBackend = None,
    ):
        super().__init__()
        self.cfg                 = cfg
        self.num_envs            = num_envs
        self.seed_offset         = seed_offset
        self.total_num_processes = total_num_processes

        # --- simulator params ---
        self.camera_height   = float(getattr(cfg, "camera_height",   1.25))
        self.step_move       = float(getattr(cfg, "step_move",       0.25))
        self.step_turn       = float(getattr(cfg, "step_turn",       math.radians(30.0)))
        self.allow_sliding   = bool( getattr(cfg, "allow_sliding",   True))
        self.max_step_height = float(getattr(cfg, "max_step_height", 0.5))
        self.agent_radius    = float(getattr(cfg, "agent_radius",    0.18))
        self.light_scale     = float(getattr(cfg, "light_scale",     4.0))
        self.success_distance = float(getattr(cfg, "success_distance", 3.0))
        self.success_bonus   = float(getattr(cfg, "success_bonus",   2.5))
        self.use_rel_reward  = bool( getattr(cfg, "use_rel_reward",  True))
        # Reward mode: "geo_progress"  — dense d2g delta + success_bonus on stop
        #              "ndtw_sr_delta" — per-step nDTW delta + SR delta (RFT)
        #              "geo_ndtw"      — geo_progress + decision-level nDTW + SR (recommended for RFT)
        #                  geo_progress creates within-group variance for random policies;
        #                  nDTW refines path quality; SR rewards success.
        #              "decision_nav"  — decision-level DTG + decision-level nDTW + SR (no step-level geo)
        #                  DTG reward = clip(DTG_before - DTG_after, -dtg_clip, dtg_clip) per decision,
        #                  with asymmetric scaling (backward penalized at 0.5×) to preserve exploration.
        # See docs/genark_rft_caveats.md C4-C6 for known issues with ndtw_sr_delta.
        self.reward_mode     = str(getattr(cfg, "reward_mode", "geo_progress"))
        self.geo_coef        = float(getattr(cfg, "geo_coef",  1.0))
        self.ndtw_coef       = float(getattr(cfg, "ndtw_coef", 1.0))
        self.sr_coef         = float(getattr(cfg, "sr_coef",   10.0))
        # decision_nav mode params
        self.decision_dtg_coef  = float(getattr(cfg, "decision_dtg_coef",  1.0))
        self.decision_dtg_clip  = float(getattr(cfg, "decision_dtg_clip",  2.0))
        # decision_level_ndtw: when True and reward_mode="ndtw_sr_delta", skip per-step
        # nDTW computation in step(). nDTW delta is computed once per LLM decision via
        # compute_decision_ndtw_reward(), called by env_worker at decision flush time.
        # SR delta is still computed per env step (it accumulates correctly via summation).
        # Benefit: 1 fastdtw call per decision instead of N_macro calls; credit assignment
        # granularity matches LLM decision granularity.
        self.decision_level_ndtw = bool(getattr(cfg, "decision_level_ndtw", False))
        # Format reward: added once per LLM decision when policy outputs valid JSON,
        # capped per episode.  LaViRA waypoint has a large schema; a single one-time
        # bonus is too weak when parse_fail is common, while an uncapped per-decision
        # bonus can dominate navigation.  The cap keeps schema learning useful but
        # bounded during overfit reward exploration.
        self.format_reward_coef = float(getattr(cfg, "format_reward_coef", 0.0))
        self.format_reward_cap = float(getattr(cfg, "format_reward_cap", 1.0))
        self.json_reward_coef = float(
            getattr(cfg, "json_reward_coef", max(self.format_reward_coef, 0.05))
        )
        self.struct_reward_coef = float(
            getattr(cfg, "struct_reward_coef", max(self.format_reward_coef, 0.05))
        )
        self.field_format_reward_coef = float(
            getattr(cfg, "field_format_reward_coef", max(self.format_reward_coef, 0.05))
        )
        self.length_reward_coef = float(getattr(cfg, "length_reward_coef", 0.02))
        self.structured_reward_cap = float(
            getattr(cfg, "structured_reward_cap", self.format_reward_cap)
        )
        self.bbox_reward_cap = float(
            getattr(cfg, "bbox_reward_cap", self.format_reward_cap)
        )
        # B1: extra format reward when policy outputs valid bbox_2d.  In
        # lavira_waypoint mode the sentinel is emitted only for valid NAVIGATE
        # waypoint geometry, so this acts as bbox/point shaping.
        # Communicated via action sentinel >= 10 (ACTION_PARSE_OK_HAS_BBOX_BASE).
        self.bbox_reward_coef = float(getattr(cfg, "bbox_reward_coef", 0.0))
        rcfg = getattr(cfg, "reward_curriculum", None)
        # Reward profiles:
        #   "format_learning": schema / bbox warm-up for cold-start checkpoints.
        #   "nav":             task reward for navigation and correct STOP.
        #
        # Legacy reward_curriculum is mapped to format_learning for backward
        # compatibility, but new runs should set reward_profile explicitly.
        legacy_curriculum_enabled = bool(
            getattr(rcfg, "enabled", False) if rcfg is not None else False
        )
        self.reward_profile = str(
            getattr(
                cfg,
                "reward_profile",
                "format_learning" if legacy_curriculum_enabled else "nav",
            )
        )
        self._global_step = 0
        self._last_reward_profile: str | None = None
        # B-infra: when True, render front depth and expose via extra_view_images.
        self._enable_depth_obs = bool(getattr(cfg, "enable_depth_obs", False))
        # --- Penalties to discourage premature / illegal terminations ---
        # parse_fail_penalty: applied when the model fails to produce valid JSON.
        # The action is clamped to MOVE_FORWARD (episode continues) but a negative
        # reward signals "your output was malformed, fix it".
        self.parse_fail_penalty = float(getattr(cfg, "parse_fail_penalty", -1.0))
        # wrong_stop_penalty: applied when the model calls STOP but is still far
        # from the goal (DTG > success_distance * wrong_stop_dist_factor).
        # Episode still ends (STOP is honored); penalty makes "give-up stop" costly.
        self.wrong_stop_penalty = float(getattr(cfg, "wrong_stop_penalty", -0.5))
        self.wrong_stop_dist_factor = float(getattr(cfg, "wrong_stop_dist_factor", 1.5))
        # Optional experiment-1 STOP shaping: replace the flat wrong_stop penalty
        # with a distance-conditioned penalty.  Disabled by default so baseline
        # runs keep exactly the old reward semantics.
        self.conditional_wrong_stop_penalty = bool(
            getattr(cfg, "conditional_wrong_stop_penalty", False)
        )
        self.conditional_wrong_stop_ratio_min = float(
            getattr(cfg, "conditional_wrong_stop_ratio_min", 0.5)
        )
        self.conditional_wrong_stop_ratio_max = float(
            getattr(cfg, "conditional_wrong_stop_ratio_max", 2.0)
        )
        # Lightweight SACA-style process reward.  Disabled by default: when enabled,
        # reward new best DTG progress and lightly penalize regressing after getting
        # close.  This gives all-failure GRPO groups nonzero relative signal without
        # introducing an external auditor.
        self.process_reward_enabled = bool(getattr(cfg, "process_reward_enabled", False))
        self.process_progress_coef = float(getattr(cfg, "process_progress_coef", 0.2))
        self.process_progress_cap = float(getattr(cfg, "process_progress_cap", 1.0))
        self.process_regression_penalty = float(getattr(cfg, "process_regression_penalty", -0.2))
        self.process_regression_margin = float(getattr(cfg, "process_regression_margin", 1.0))
        # GroundedSAM reward is intentionally gated by DTG progress; raw detection
        # confidence is diagnostic only and never used directly as reward.
        self.gsam_reward_enabled = bool(getattr(cfg, "gsam_reward_enabled", False))
        self.gsam_detect_progress_bonus = float(getattr(cfg, "gsam_detect_progress_bonus", 0.05))
        # 4-direction rendering: when True, in addition to the front view stored
        # in `main_images`, also render left/right/behind and put them in
        # `extra_view_images` shape (num_envs, 3, H, W, 3) uint8.
        # 3× extra render cost per step. Required for QwenNavPolicy lavira-style
        # 4-dir prompt (see docs/genark_rft_caveats.md C10).
        self.enable_4dir_render = bool(getattr(cfg, "enable_4dir_render", False))
        self.max_episode_steps = int(getattr(cfg, "max_episode_steps", 500))
        self.auto_reset      = bool( getattr(cfg, "auto_reset",      True))
        # GRPO: how many envs share one episode. group_size envs get the same
        # instruction/start so their rewards are comparable within the group.
        self.group_size      = int(  getattr(cfg, "group_size",       1))
        # cyclic_episode_sampling: when True, the episode pool is reshuffled and
        # replayed from the start once exhausted (train mode).  When False, groups
        # enter dormant once the pool is exhausted (eval / single-pass mode).
        self.cyclic_episode_sampling = bool(
            getattr(cfg, "cyclic_episode_sampling", False)
        )
        self.is_eval = bool(getattr(cfg, "is_eval", False))
        self.eval_roll_through_episode_pool = bool(
            getattr(cfg, "eval_roll_through_episode_pool", False)
        )
        self.eval_repeats_per_episode = max(
            1, int(getattr(cfg, "eval_repeats_per_episode", 1))
        )
        self.episode_balanced_sampling = bool(
            getattr(cfg, "episode_balanced_sampling", False)
        )
        _c = getattr(cfg, "curriculum_dtg_start", None)
        self._curriculum_dtg_start = float(_c) if _c is not None else None
        self._curriculum_dtg_step  = float(getattr(cfg, "curriculum_dtg_step", 5.0))

        # --- observation params ---
        cam_res              = tuple(getattr(cfg, "cam_res", (640, 480)))
        self.cam_w, self.cam_h = cam_res
        self.fov             = int(getattr(cfg, "fov", 105))
        self.max_seq_len     = int(getattr(cfg, "max_seq_len", 256))

        # --- data ---
        init_p             = cfg.init_params
        self.episodes_file = str(init_p.episodes_file)
        self.scene_datasets = str(init_p.scene_datasets)

        # --- spaces ---
        import gym.spaces as S
        self.observation_space = S.Dict({
            "rgb": S.Box(
                low=0, high=255,
                shape=(3, self.cam_h, self.cam_w),
                dtype=np.uint8,
            ),
            "instruction_ids": S.Box(
                low=0, high=65535,
                shape=(self.max_seq_len,),
                dtype=np.int64,
            ),
            "instruction_mask": S.Box(
                low=0, high=1,
                shape=(self.max_seq_len,),
                dtype=bool,
            ),
        })
        self.action_space = S.Discrete(4)

        # --- internal state ---
        self._gs_ready        = False
        # cam_pos, cam_yaw, current_tri_idx are owned by _sim (backend);
        # GenarkVecEnv accesses them via self._sim.cam_pos etc.
        self._active_mask     = None     # (num_envs,) bool on backend device — set after backend init
        self._goal_pos_t      = None     # (num_envs, 3) habitat coords on backend device
        self._prev_geo_dist   = None     # (num_envs,) float32 on backend device
        self._elapsed_steps   = np.zeros(num_envs, dtype=np.int32)
        self._instructions    = [""] * num_envs
        self._episodes        = [None] * num_envs
        self._pred_path_lists = [[] for _ in range(num_envs)]
        self._distances_lists = [[] for _ in range(num_envs)]
        # ndtw_sr_delta reward mode: track per-env previous-step nDTW and SR
        self._prev_ndtw       = np.zeros(num_envs, dtype=np.float32)
        self._prev_sr         = np.zeros(num_envs, dtype=np.float32)
        # 4-dir extras: (num_envs, 3, H, W, 3) uint8 numpy — order [left, right, behind]
        self._current_rgb_extras = None
        self._stop_called     = [False] * num_envs
        self._current_rgb     = None     # cached last RGB (CPU numpy)
        self._current_depth   = None     # cached last depth (CPU numpy, float32) when enable_depth_obs

        # --- Episode-end diagnostic trackers ---
        # Raw action history per env (includes PARSE_FAIL=4 sentinel, NOT clamped).
        # Cleared on every episode (re)start; used to classify termination reason.
        self._action_history: list[list[int]] = [[] for _ in range(num_envs)]
        # Start distance-to-goal for success_type classification in ep-diag.
        self._start_dtg: list[float] = [0.0] * num_envs
        # Toggle: set GENARK_EP_DIAG=0 to silence per-episode diagnostic line.
        self._ep_diag_enabled = os.environ.get("GENARK_EP_DIAG", "1") != "0"

        # Option A — single-pass: each active slot runs exactly one episode.
        # _slot_active[i]: this slot has a real episode (first active_slot_count slots).
        # _slot_done[i]:   this slot has completed its episode (stop or timeout).
        # When all active slots are done → _exhausted = True → dormant_step().
        self._slot_active = np.zeros(num_envs, dtype=bool)   # set after scene pinning
        self._slot_done   = np.zeros(num_envs, dtype=bool)
        self._episode_log: list[dict] = []
        self._last_completed_diag_by_env: list[dict | None] = [None] * num_envs
        self._completed_diag_event_by_env: list[dict | None] = [None] * num_envs
        # One-shot bridge for terminal decision rewards.  In cyclic train mode a
        # finished GRPO group can be reset inside step(); env_worker computes the
        # decision reward immediately after step(), so preserve the just-finished
        # episode state before _init_agent_poses() overwrites it.
        self._pending_decision_reward_snapshot_by_env: list[dict | None] = [
            None
        ] * num_envs
        self._gsam_reward_diag_counts = self._empty_gsam_reward_diag_counts()
        self._exhausted = False
        self._dummy_obs_cache = None

        # Load episode list and group by scene.
        # Genesis builds ONE scene per process — all envs share it.
        # Episodes must therefore come from a single scene_id (single-scene path),
        # or from K scenes (multiscene path, see genesis_backend="multiscene").
        #
        # Single-scene assignment: deterministic round-robin over the sorted unique
        # scene list. With N workers and M scenes:
        #   - if N <= M: each worker pins to a distinct scene (covers N/M)
        #   - if N >  M: workers wrap around (some scenes get >1 worker)
        all_episodes = _filter_overfit_episode(
            _load_episodes(self.episodes_file), cfg
        )
        unique_scenes = sorted({e["scene_id"] for e in all_episodes})
        if not unique_scenes:
            raise ValueError(
                f"No episodes available after filtering episodes_file={self.episodes_file}"
            )
        scene_offset = int(getattr(cfg, "scene_offset", 0))

        genesis_backend_type = str(getattr(cfg, "genesis_backend", "local"))

        # _scene_layout: set only in multiscene mode; None in single-scene mode.
        # _pool_by_scene: dict[scene_id -> list[episode]] for each scene in use.
        #   Single-scene: one-key dict so per-scene paths work for both modes.
        # _rng_by_scene: dict[scene_id -> np.Generator] — independent per-scene RNG.
        self._scene_layout = None
        self._pool_by_scene: dict[str, list[dict]] = {}
        self._rng_by_scene:  dict[str, np.random.Generator] = {}
        self._next_ep_idx_by_scene:  dict[str, int] = {}
        self._episode_cycle_by_scene: dict[str, int] = {}
        self._balancer_by_scene: dict[str, "EpisodeBalancer | None"] = {}

        gs = max(self.group_size, 1)

        if genesis_backend_type == "multiscene":
            # --- Multi-scene path ---
            import torch
            from rlinf.envs.genark.genesis_multiscene import (
                build_multiscene_backend,
                SceneLayout,
                _stable_scene_hash,
            )
            _multi_cfg = getattr(cfg, "multi_scene", None)
            gpu_budget = getattr(_multi_cfg, "gpu_budget", None)
            if gpu_budget is not None:
                gpu_budget = int(gpu_budget)
            scenes_per_gpu = int(getattr(_multi_cfg, "scenes_per_gpu", 1) or 1)

            ms_backend, layout, pool_by_scene = build_multiscene_backend(
                cfg=cfg,
                all_episodes=all_episodes,
                num_envs=num_envs,
                group_size=gs,
                seed_offset=seed_offset,
                device=torch.device("cuda"),
                gpu_budget=gpu_budget,
                scenes_per_gpu=scenes_per_gpu,
            )
            backend = ms_backend
            self._scene_layout = layout

            for scene_id in layout.scenes:
                eps = pool_by_scene[scene_id]
                self._pool_by_scene[scene_id]   = eps
                # Per-scene RNG (independent of other scenes for reproducibility)
                self._rng_by_scene[scene_id]    = np.random.default_rng(
                    seed=42 + seed_offset + _stable_scene_hash(scene_id)
                )
                k_s = layout.sizes[layout.scenes.index(scene_id)]
                n_initial_groups_s = k_s // gs
                self._next_ep_idx_by_scene[scene_id]   = n_initial_groups_s
                self._episode_cycle_by_scene[scene_id] = 0
                self._balancer_by_scene[scene_id]      = None
                if self.episode_balanced_sampling and self.cyclic_episode_sampling:
                    bal = EpisodeBalancer(eps, self._rng_by_scene[scene_id],
                                         curriculum_dtg_start=self._curriculum_dtg_start,
                                         curriculum_dtg_step=self._curriculum_dtg_step)
                    for g in range(min(n_initial_groups_s, len(eps))):
                        bal.mark_seen(eps[g].get("episode_id", g))
                    self._balancer_by_scene[scene_id] = bal

            # Mark all slots active (fully-active invariant: K_s <= len(eps_s))
            self._active_slot_count = num_envs
            self._slot_active[:num_envs] = True

            # Compatibility aliases used in single-scene code paths.
            # The first scene is used for attributes that don't affect correctness
            # in multiscene (e.g. logging fallback). Real per-slot scene is in layout.
            self._pinned_scene_id = layout.scenes[0]
            self._all_episodes    = []   # not used in multiscene; per-scene pools are

            print(
                f"[GenArk][multiscene] Worker {seed_offset}/{total_num_processes}: "
                f"K={len(layout.scenes)} scenes, num_envs={num_envs}, "
                f"group_size={gs}, scenes={layout.scenes}",
                flush=True,
            )

        else:
            # --- Single-scene path (local / remote / zmq) ---
            pinned_scene = unique_scenes[(scene_offset + seed_offset) % len(unique_scenes)]
            scene_eps = [e for e in all_episodes if e["scene_id"] == pinned_scene]

            rng = np.random.default_rng(seed=42 + seed_offset)
            rng.shuffle(scene_eps)

            self._pool_by_scene[pinned_scene]   = scene_eps
            self._rng_by_scene[pinned_scene]    = rng
            self._pinned_scene_id = pinned_scene
            self._all_episodes    = scene_eps   # kept for backward-compat read paths

            active_slot_count = min(num_envs, len(scene_eps))
            self._slot_active[:active_slot_count] = True
            self._active_slot_count = active_slot_count

            n_initial_groups = (active_slot_count + gs - 1) // gs
            self._next_ep_idx_by_scene[pinned_scene]   = n_initial_groups
            self._episode_cycle_by_scene[pinned_scene] = 0
            self._balancer_by_scene[pinned_scene]      = None
            if self.episode_balanced_sampling and self.cyclic_episode_sampling:
                bal = EpisodeBalancer(scene_eps, rng,
                                     curriculum_dtg_start=self._curriculum_dtg_start,
                                     curriculum_dtg_step=self._curriculum_dtg_step)
                for g in range(min(n_initial_groups, len(scene_eps))):
                    bal.mark_seen(scene_eps[g].get("episode_id", g))
                self._balancer_by_scene[pinned_scene] = bal

            print(f"[GenArk] Worker {seed_offset}/{total_num_processes}: "
                  f"pinned to scene '{pinned_scene}' "
                  f"({len(scene_eps)} eps, "
                  f"active_slots={active_slot_count}/{num_envs}, "
                  f"{len(unique_scenes)} unique scenes total)", flush=True)

        # _group_done_counts: group_id → count of done envs waiting for group-mates.
        self._group_done_counts: dict[int, int] = {}

        # --- Legacy single-scene accessors (kept for backward-compat) ---
        # These resolve from the per-scene dicts for K=1; in multiscene they are
        # only accessed in code paths guarded by _scene_layout is None.
        def _single_scene_attr(attr: str):
            """Return the value for the single scene (K=1 path only)."""
            sid = self._pinned_scene_id
            return {
                "_next_ep_idx":   self._next_ep_idx_by_scene,
                "_episode_cycle": self._episode_cycle_by_scene,
                "_balancer":      self._balancer_by_scene,
                "_rng":           self._rng_by_scene,
            }[attr][sid]

        # Initialise backend.  Caller can pass a backend= explicitly.
        self._genesis_server_proc = None   # subprocess.Popen, set for zmq mode

        if backend is None:
            if genesis_backend_type == "remote":
                from rlinf.envs.genark.genesis_server import (
                    GenesisServerPool,
                    GenesisRemoteBackend,
                )
                import torch
                pool = GenesisServerPool(
                    cfg,
                    {self._pinned_scene_id: self._active_slot_count},
                )
                backend = GenesisRemoteBackend(
                    pool,
                    scene_id=self._pinned_scene_id,
                    num_envs=num_envs,
                    device=torch.device("cuda"),
                )

            elif genesis_backend_type == "zmq":
                from rlinf.envs.genark.genesis_zmq_server import (
                    GenesisZMQBackend,
                    launch_zmq_server,
                )
                import torch, hashlib
                scene_hash = hashlib.md5(
                    self._pinned_scene_id.encode()
                ).hexdigest()[:8]
                socket_base = f"genesis_{scene_hash}_{seed_offset}"
                self._genesis_server_proc = launch_zmq_server(
                    cfg,
                    self._pinned_scene_id,
                    self._active_slot_count,
                    socket_base,
                )
                backend = GenesisZMQBackend(
                    socket_base,
                    scene_id=self._pinned_scene_id,
                    num_envs=num_envs,
                    device=torch.device("cuda"),
                )

            else:
                backend = GenesisLocalBackend(cfg, num_envs)

        self._sim: GenesisSimBackend = backend

        # Async render pipeline + crash isolation state (Phase 3 / Phase 4).
        from rlinf.envs.genark.genesis_server import SceneCrashError as _SceneCrashError
        self._SceneCrashError = _SceneCrashError
        # Any backend that exposes render_main_async supports the async pipeline.
        self._use_async_render: bool = hasattr(self._sim, "render_main_async")
        self._pending_rgb_ref        = None   # Ray ObjectRef for in-flight render_main
        self._pending_rgb_extras_ref = None   # Ray ObjectRef for in-flight render_4dir

        # Phase 4: scene crash isolation.
        # When the remote actor crashes, _scene_crashed is set True and all
        # steps enter dormant until the actor is rebuilt and re-initialised.
        self._scene_crashed: bool         = False
        self._crash_recovery_cooldown: int = 0   # steps to skip before next health poll

        # Pick initial scene and episodes, then build Genesis scene
        self._current_scene_id = None
        self._scene_episode_pool: list[dict] = []
        self._assign_episodes_to_envs()

    # ------------------------------------------------------------------
    # Scene loading — delegates to backend
    # ------------------------------------------------------------------

    def _load_scene(self, scene_id: str):
        """Load mesh + navmesh via backend, then mark scene as ready."""
        self._sim.load_scene(scene_id, n_active_envs=self._active_slot_count)
        self._gs_ready = True
        self._current_scene_id = scene_id

    # ------------------------------------------------------------------
    # Episode assignment helpers
    # ------------------------------------------------------------------

    def set_global_step(self, step: int) -> None:
        self._global_step = int(step)

    def _reward_coeffs(self) -> dict[str, float]:
        """Effective reward coefficients for the selected reward profile."""
        profile = self.reward_profile
        if profile != self._last_reward_profile:
            print(
                f"[GenArk][reward-profile] step={self._global_step} profile={profile}",
                flush=True,
            )
            self._last_reward_profile = profile

        if profile == "format_learning":
            return {
                "ndtw": 0.0,
                "sr": 0.0,
                "dtg": 0.0,
                "json": max(self.json_reward_coef, 0.20),
                "struct": max(self.struct_reward_coef, 0.20),
                "field_format": max(self.field_format_reward_coef, 0.20),
                "length": max(self.length_reward_coef, 0.05),
                "format": max(self.format_reward_coef, 0.05),
                "bbox": max(self.bbox_reward_coef, 0.50),
                "parse_fail": min(self.parse_fail_penalty, -1.0),
                "wrong_stop": 0.0,
            }
        if profile != "nav":
            raise ValueError(
                f"Unknown reward_profile={profile!r}; expected 'nav' or "
                "'format_learning'."
            )
        return {
            "ndtw": self.ndtw_coef,
            "sr": self.sr_coef,
            "dtg": self.decision_dtg_coef,
            "json": self.json_reward_coef,
            "struct": self.struct_reward_coef,
            "field_format": self.field_format_reward_coef,
            "length": self.length_reward_coef,
            "format": self.format_reward_coef,
            "bbox": self.bbox_reward_coef,
            "parse_fail": self.parse_fail_penalty,
            "wrong_stop": self.wrong_stop_penalty,
        }

    def _assign_episodes_to_envs(self, env_idx: Optional[list[int]] = None):
        """Assign episodes to slots. With group_size>1, every group_size consecutive
        slots share the same episode so GRPO can compare rewards within a group.

        Multi-scene: slot i is assigned from its own scene's pool using a LOCAL
        group index within the scene block, so episodes never cross scene boundaries.
        Single-scene: equivalent to the old global-index path (K=1 one-key dict).
        """
        gs      = max(self.group_size, 1)
        indices = list(range(self.num_envs)) if env_idx is None else env_idx
        layout  = self._scene_layout  # None in single-scene mode

        for i in indices:
            if not self._slot_active[i]:
                continue  # ghost slot — no episode assigned

            if layout is not None:
                # Multi-scene: index into the slot's own scene pool
                scene_id    = layout.scene_id_of(i)
                pool        = self._pool_by_scene[scene_id]
                local_group = layout.scene_local_group(i)   # group within scene block
            else:
                # Single-scene: global group index into the one pool
                scene_id    = self._pinned_scene_id
                pool        = self._pool_by_scene[scene_id]
                local_group = i // gs

            if local_group < len(pool):
                self._episodes[i]     = pool[local_group]
                self._instructions[i] = pool[local_group]["instruction"]["instruction_text"]
            else:
                self._slot_active[i] = False  # shouldn't happen, safety guard

        # P0 guard: all active envs in the same group must share the exact same
        # episode object AND belong to the same scene (Bug #2 runtime backstop).
        assigned = {i for i in indices if self._slot_active[i] and self._episodes[i] is not None}
        groups_assigned: dict[int, list[int]] = {}
        for i in assigned:
            gid = i // gs
            groups_assigned.setdefault(gid, []).append(i)
        for gid, members in groups_assigned.items():
            ep_objects = {id(self._episodes[j]) for j in members}
            assert len(ep_objects) == 1, (
                f"[P0] _assign_episodes_to_envs: group {gid} split across episodes: "
                f"{[self._episodes[j].get('episode_id') for j in members]}"
            )
            if layout is not None:
                # Bug #2 runtime backstop: all group members must be in the same scene
                scene_ids_in_group = {layout.scene_id_of(j) for j in members}
                assert len(scene_ids_in_group) == 1, (
                    f"[P0] group {gid} straddles scenes {scene_ids_in_group}. "
                    f"SceneLayout invariant violated — check group_size alignment."
                )

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
    ) -> tuple[dict, dict]:
        if env_idx is None:
            env_idx = list(range(self.num_envs))
        elif isinstance(env_idx, (int, np.integer)):
            env_idx = [int(env_idx)]
        else:
            env_idx = list(env_idx)

        # Reset exhaustion flags for the envs being reset so episodes can replay.
        # _slot_done[i] must be cleared before _assign_episodes_to_envs so that
        # the exhaustion check in step() doesn't immediately re-enter dormant mode.
        for i in env_idx:
            self._slot_done[i] = False
        if all(not self._slot_done[i] for i in range(self.num_envs)
               if self._slot_active[i]):
            self._exhausted = False

        self._assign_episodes_to_envs(env_idx)

        # Ensure scene is loaded (use first active slot's episode)
        active_ep = next((self._episodes[i] for i in env_idx
                          if self._slot_active[i] and self._episodes[i] is not None), None)
        if active_ep is not None and active_ep["scene_id"] != self._current_scene_id:
            self._load_scene(active_ep["scene_id"])

        # Only operate on active slots — ghost slots have no episode assigned
        active_idx = [i for i in env_idx if self._slot_active[i] and self._episodes[i] is not None]

        self._elapsed_steps[active_idx] = 0
        for i in active_idx:
            self._stop_called[i] = False
            self._pred_path_lists[i] = []
            self._distances_lists[i] = []
            self._action_history[i] = []

        # Initialise tensors for active slots only
        self._init_agent_poses(active_idx)

        # Clear any in-flight async render refs so _build_obs() forces a sync render.
        self._pending_rgb_ref        = None
        self._pending_rgb_extras_ref = None
        self._current_rgb            = None
        self._current_rgb_extras     = None
        self._current_depth          = None
        # If actor was rebuilt externally, allow reset to clear crash flag.
        if self._scene_crashed and self._sim.is_scene_healthy():
            self._scene_crashed = False

        obs  = self._build_obs()
        return obs, {}

    def _init_agent_poses(self, env_idx: list[int]):
        device = self._sim.device
        N      = self.num_envs

        # Lazy-init GRPO metric tensors (owned by GenarkVecEnv, not backend)
        if self._active_mask is None:
            self._active_mask          = torch.ones(N,     dtype=torch.bool,    device=device)
            self._goal_pos_t           = torch.zeros(N, 3, dtype=torch.float32, device=device)
            self._prev_geo_dist        = torch.zeros(N,    dtype=torch.float32, device=device)
            self._last_parse_ok        = torch.zeros(N,    dtype=torch.bool,    device=device)
            self._last_has_bbox        = torch.zeros(N,    dtype=torch.bool,    device=device)
            self._last_json_valid      = torch.zeros(N,    dtype=torch.bool,    device=device)
            self._last_struct_ok       = torch.zeros(N,    dtype=torch.bool,    device=device)
            self._last_field_format_ok = torch.zeros(N,    dtype=torch.bool,    device=device)
            self._last_length_ok       = torch.zeros(N,    dtype=torch.bool,    device=device)
            self._dtg_decision_start   = torch.zeros(N,    dtype=torch.float32, device=device)
            self._min_dtg_t            = torch.zeros(N,    dtype=torch.float32, device=device)
            self._process_progress_paid = torch.zeros(N,   dtype=torch.float32, device=device)
            # Cumulative format reward already paid this episode, used to cap
            # dense schema shaping during LaViRA waypoint overfit.
            self._format_reward_given  = torch.zeros(N,    dtype=torch.float32, device=device)
            self._bbox_reward_given    = torch.zeros(N,    dtype=torch.float32, device=device)
            self._structured_reward_given = torch.zeros(N, dtype=torch.float32, device=device)

        # Build position/yaw tensors from episode data for valid slots
        valid_idx = [i for i in env_idx if self._episodes[i] is not None]
        if not valid_idx:
            return

        positions = torch.zeros(len(valid_idx), 3, dtype=torch.float32)
        yaws      = torch.zeros(len(valid_idx),    dtype=torch.float32)

        for j, i in enumerate(valid_idx):
            ep  = self._episodes[i]
            # Bug #2 runtime check: slot must hold an episode for its own scene.
            if self._scene_layout is not None:
                expected_scene = self._scene_layout.scene_id_of(i)
                assert ep["scene_id"] == expected_scene, (
                    f"[P0] _init_agent_poses: slot {i} has episode scene_id "
                    f"'{ep['scene_id']}' but belongs to scene block '{expected_scene}'. "
                    f"Episode pool routing bug in _assign_episodes_to_envs."
                )
            sp  = ep["start_position"]
            gx, gy, gz = _hab_to_genesis(sp)
            positions[j] = torch.tensor([gx, gy, gz + self.camera_height])
            rot          = torch.tensor(ep["start_rotation"], dtype=torch.float32)
            yaws[j]      = _calculate_initial_yaw(rot.unsqueeze(0))[0]

            goal_p = ep["goals"][0]["position"]
            self._goal_pos_t[i] = torch.tensor(goal_p, dtype=torch.float32, device=device)
            self._active_mask[i] = True

        # Backend: set poses + snap to navmesh + update Genesis camera
        self._sim.set_agent_poses(valid_idx, positions, yaws)

        # Read back snapped positions for metric initialisation.
        # Bug #1: use cam_pos_hab() so coordinate frames are correct per scene.
        # In single-scene mode this is identical to _genesis_to_hab(cam_pos, height).
        # In multi-scene mode, MultiSceneBackend applies the transform per block.
        init_hab = self._sim.cam_pos_hab(self.camera_height)
        for i in valid_idx:
            d = torch.norm(init_hab[i] - self._goal_pos_t[i]).item()
            self._distances_lists[i] = [d]
            self._pred_path_lists[i] = [init_hab[i].cpu().numpy()]
            self._prev_geo_dist[i]   = d
            self._start_dtg[i]       = d
            self._dtg_decision_start[i] = d
            self._min_dtg_t[i] = d
            self._process_progress_paid[i] = 0.0
            self._prev_ndtw[i] = 1.0
            self._prev_sr[i]   = 0.0
            self._format_reward_given[i] = 0.0
            self._bbox_reward_given[i] = 0.0
            self._structured_reward_given[i] = 0.0

    def step(
        self,
        actions: Union[torch.Tensor, np.ndarray, list],
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Apply actions and advance the simulation one step.

        Args:
            actions: (num_envs,) int64 — 0=stop, 1=fwd, 2=left, 3=right

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Dormant mode: pool exhausted, return no-op terminated steps.
        # Skip Genesis render and metric computation. info["episode"] is empty
        # so RLinf's eval aggregator does not count repeats.
        if self._exhausted:
            return self._dormant_step()

        # Phase 4: scene crash recovery.
        # If the remote actor crashed, stay dormant until it is rebuilt.
        if self._scene_crashed:
            return self._dormant_step_crash_recovery()

        device = self._sim.device
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long, device=device)
        else:
            actions = actions.to(device=device, dtype=torch.long)

        # Decode action sentinels BEFORE clamping.
        # Sentinel table:
        #   0-3  : normal action, parse_ok=True,  has_bbox=False
        #   4    : ACTION_PARSE_FAIL, parse_ok=False, has_bbox=False
        #   5    : schema_ok but geometry/action invalid; parse_ok=True,
        #          has_bbox=False, execute as parse-fail fallback
        #   10-13: ACTION_PARSE_OK_HAS_BBOX_BASE + true_action, parse_ok=True, has_bbox=True
        #   20+  : structured parse-fail bitmask; fallback action but partial
        #          RL-Struct rewards (JSON/fields/format/geometry/length) apply.
        _HAS_BBOX_BASE = 10
        _SCHEMA_OK_PARSE_FAIL = 5
        _STRUCTURED_FAIL_BASE = 20
        _R_JSON = 1 << 0
        _R_STRUCT = 1 << 1
        _R_FIELD = 1 << 2
        _R_GEOM = 1 << 3
        _R_LENGTH = 1 << 4
        actions = actions.clone()
        structured_mask = actions >= _STRUCTURED_FAIL_BASE
        structured_bits = torch.zeros_like(actions)
        structured_bits[structured_mask] = actions[structured_mask] - _STRUCTURED_FAIL_BASE
        has_bbox_mask = (actions >= _HAS_BBOX_BASE) & (actions < _HAS_BBOX_BASE + 4)
        schema_ok_parse_fail_mask = actions == _SCHEMA_OK_PARSE_FAIL
        normal_ok_mask = (actions < 4) | has_bbox_mask
        parse_ok_mask = (
            normal_ok_mask
            | schema_ok_parse_fail_mask
            | ((structured_bits & _R_FIELD) != 0)
        )
        json_valid_mask = normal_ok_mask | schema_ok_parse_fail_mask | ((structured_bits & _R_JSON) != 0)
        struct_ok_mask = normal_ok_mask | schema_ok_parse_fail_mask | ((structured_bits & _R_STRUCT) != 0)
        field_format_ok_mask = normal_ok_mask | schema_ok_parse_fail_mask | ((structured_bits & _R_FIELD) != 0)
        geometry_ok_mask = has_bbox_mask | ((structured_bits & _R_GEOM) != 0)
        length_ok_mask = normal_ok_mask | ((structured_bits & _R_LENGTH) != 0)
        # Recover true action from has_bbox sentinel (11→1, 12→2, 13→3)
        actions[has_bbox_mask] = actions[has_bbox_mask] - _HAS_BBOX_BASE
        actions[schema_ok_parse_fail_mask] = 4
        actions[structured_mask] = 4
        # Store for compute_decision_ndtw_reward() to use as format/bbox bonus gate.
        self._last_parse_ok = parse_ok_mask.to(dtype=torch.bool, device=self._last_parse_ok.device)
        self._last_has_bbox = has_bbox_mask.to(dtype=torch.bool, device=self._last_has_bbox.device)
        self._last_json_valid = json_valid_mask.to(dtype=torch.bool, device=self._last_json_valid.device)
        self._last_struct_ok = struct_ok_mask.to(dtype=torch.bool, device=self._last_struct_ok.device)
        self._last_field_format_ok = field_format_ok_mask.to(dtype=torch.bool, device=self._last_field_format_ok.device)
        self._last_length_ok = length_ok_mask.to(dtype=torch.bool, device=self._last_length_ok.device)

        # Record raw action (incl. parse-fail sentinel) into per-env history for
        # episode-end diagnostics and zero-shot difficulty metrics.
        raw_acts_cpu = actions.detach().cpu().tolist()
        active_undone_pre = self._slot_active & ~self._slot_done
        for _i, _a in enumerate(raw_acts_cpu):
            if active_undone_pre[_i]:
                self._action_history[_i].append(int(_a))

        # Parse-fail policy (Lavira-style): keep navigating, do NOT terminate.
        # Any remaining sentinel value >= 4 is clamped to MOVE_FORWARD so the
        # episode continues. Done/ghost slots still go to STOP.
        actions[actions >= 4] = self.MOVE_FORWARD  # parse-fail → default forward
        done_or_ghost = torch.tensor(
            self._slot_done | ~self._slot_active, dtype=torch.bool, device=device
        )
        actions[done_or_ghost] = self.STOP

        # Only increment elapsed_steps for active+undone slots
        active_undone_np = self._slot_active & ~self._slot_done
        self._elapsed_steps[active_undone_np] += 1

        # --- Mark newly stopped agents ---
        newly_stopped = (actions == self.STOP) & self._active_mask
        self._active_mask[newly_stopped] = False
        for i in newly_stopped.nonzero(as_tuple=True)[0].tolist():
            if self._slot_active[i] and not self._slot_done[i]:
                self._stop_called[i] = True

        # --- Physics + Render (wrapped for Phase 4 crash isolation) ---
        try:
            self._sim.step_physics(actions, self._active_mask, self._active_slot_count)
        except self._SceneCrashError as e:
            print(f"[GenArk][crash] scene actor crashed during step_physics: {e}", flush=True)
            self._scene_crashed = True
            self._crash_recovery_cooldown = 50
            self._pending_rgb_ref = self._pending_rgb_extras_ref = None
            return self._dormant_step_crash_recovery()

        # --- Render via backend ---
        try:
            if self._enable_depth_obs:
                # Depth obs path: synchronous render_main_with_depth + 4-dir RGB.
                self._current_rgb, self._current_depth = self._sim.render_main_with_depth(
                    self._active_slot_count
                )
                self._current_rgb_extras = self._sim.render_4dir(self._active_slot_count)
            elif self._use_async_render:
                # Phase 3 async pipeline:
                # 1. Fetch the render submitted at the *previous* step (LLM generate has
                #    been running for ~120s in parallel, so the result is ready instantly).
                if self._pending_rgb_ref is not None:
                    self._current_rgb = self._sim.fetch_render_main(self._pending_rgb_ref)
                    self._current_rgb_extras = self._sim.fetch_render_4dir(
                        self._pending_rgb_extras_ref
                    )
                # 2. Submit this step's render asynchronously (returns immediately).
                self._pending_rgb_ref        = self._sim.render_main_async(self._active_slot_count)
                self._pending_rgb_extras_ref = self._sim.render_4dir_async(self._active_slot_count)
            else:
                self._current_rgb        = self._sim.render_main(self._active_slot_count)
                self._current_rgb_extras = self._sim.render_4dir(self._active_slot_count)
        except self._SceneCrashError as e:
            print(f"[GenArk][crash] scene actor crashed during render: {e}", flush=True)
            self._scene_crashed = True
            self._crash_recovery_cooldown = 50
            self._pending_rgb_ref = self._pending_rgb_extras_ref = None
            return self._dormant_step_crash_recovery()

        # --- Metrics ---
        N_act = self._active_slot_count
        # Bug #1: use cam_pos_hab() so each scene block uses its own navmesh datum.
        # Single-scene: identical to _genesis_to_hab(cam_pos, height).
        # Multi-scene: MultiSceneBackend applies transform per scene block.
        curr_hab_act = self._sim.cam_pos_hab(self.camera_height)[:N_act]
        curr_dist_act = torch.norm(
            curr_hab_act - self._goal_pos_t[:N_act].to(curr_hab_act.device), dim=1
        )
        # Pad with zeros for ghost slots so shapes stay (num_envs,)
        if N_act < self.num_envs:
            pad_d = torch.zeros(self.num_envs - N_act, device=curr_dist_act.device)
            curr_dist = torch.cat([curr_dist_act, pad_d], dim=0)
        else:
            curr_dist = curr_dist_act

        for i in range(N_act):
            if self._active_mask[i]:
                self._pred_path_lists[i].append(curr_hab_act[i].cpu().numpy())
                self._distances_lists[i].append(curr_dist_act[i].item())
                if hasattr(self, "_min_dtg_t") and self._min_dtg_t is not None:
                    self._min_dtg_t[i] = min(
                        float(self._min_dtg_t[i].item()),
                        float(curr_dist_act[i].item()),
                    )

        # --- Reward ---
        if self.reward_mode == "ndtw_sr_delta":
            # Per-step SR delta (always); nDTW delta only when not decision_level_ndtw.
            # When decision_level_ndtw=True, nDTW is computed once per LLM decision
            # via compute_decision_ndtw_reward(), called by env_worker at flush time.
            reward = self._compute_ndtw_sr_delta_reward(
                curr_dist=curr_dist,
                newly_stopped=newly_stopped,
                N_act=N_act,
                include_ndtw=not self.decision_level_ndtw,
            )
        elif self.reward_mode == "geo_ndtw":
            # Combined: dense geo_progress (per step) + SR delta (per step)
            # + decision-level nDTW (deferred to env_worker flush via compute_decision_ndtw_reward).
            # geo_progress creates within-group variance even for random policies,
            # preventing zero-gradient cold-start; nDTW shapes path quality.
            geo_reward = self.geo_coef * (self._prev_geo_dist.to(curr_dist.device) - curr_dist)
            sr_and_ndtw = self._compute_ndtw_sr_delta_reward(
                curr_dist=curr_dist,
                newly_stopped=newly_stopped,
                N_act=N_act,
                include_ndtw=False,  # nDTW deferred to decision flush
            )
            reward = geo_reward + sr_and_ndtw
        elif self.reward_mode == "decision_nav":
            # Decision-level DTG + decision-level nDTW + terminal STOP reward.
            # All navigation/STOP rewards are settled in compute_decision_ndtw_reward()
            # at decision flush time.  Do not compute SR here: STOP marks
            # _active_mask=False before this point, which would drop successful STOP.
            reward = torch.zeros_like(curr_dist)
        else:
            # Default: dense d2g progress + sparse success bonus
            reward = self._prev_geo_dist.to(curr_dist.device) - curr_dist
            just_succeeded = newly_stopped & (curr_dist < self.success_distance)
            reward[just_succeeded] += self.success_bonus
        self._prev_geo_dist = curr_dist.clone()

        # --- Parse-fail penalty (episode continues, malformed output costs) ---
        coeffs = self._reward_coeffs()
        parse_fail_penalty = coeffs["parse_fail"]
        if parse_fail_penalty != 0.0:
            active_undone_t = torch.tensor(
                active_undone_np, dtype=torch.bool, device=reward.device,
            )
            pf_mask = (~self._last_parse_ok.to(reward.device)) & active_undone_t
            if pf_mask.any():
                reward[pf_mask] += parse_fail_penalty

        # --- Wrong-stop penalty (STOP issued when still far from goal) ---
        # FIXED small penalty (no distance scaling). Scaling by DTG made "explore
        # farther then stop" hurt MORE than "stop in place" (-3.5 @ 8.9m), actively
        # teaching the agent not to leave spawn. A flat penalty discourages premature
        # STOP without punishing exploration distance.
        wrong_stop_penalty = coeffs["wrong_stop"]
        if wrong_stop_penalty != 0.0 and self.reward_mode != "decision_nav":
            ws_mask = newly_stopped & (curr_dist.to(newly_stopped.device) > self.success_distance)
            if ws_mask.any():
                reward[ws_mask.to(reward.device)] += -abs(wrong_stop_penalty)

        # --- Termination / truncation (active+undone slots only) ---
        newly_timed_out = torch.tensor(
            (self._elapsed_steps >= self.max_episode_steps) & active_undone_np,
            dtype=torch.bool, device="cpu",
        )
        terminated = newly_stopped.cpu() & torch.tensor(active_undone_np, device="cpu")
        truncated  = newly_timed_out

        # --- Info ---
        success_np = (
            (np.array([self._stop_called[i] for i in range(self.num_envs)])
             & (curr_dist.cpu().numpy() < self.success_distance))
        ).astype(float)
        info = {
            "distance_to_goal": curr_dist.cpu().numpy(),
            "success":          success_np,
            "elapsed_steps":    self._elapsed_steps.copy(),
        }

        # --- Single-pass episode completion ---
        dones = (terminated | truncated).numpy()
        newly_done_idx = [i for i in np.where(dones)[0].tolist()
                          if self._slot_active[i] and not self._slot_done[i]]
        if newly_done_idx:
            obs, info = self._handle_slot_done(newly_done_idx, self._build_obs(), info)
            # Group-level reset: when all envs in a group are done, immediately
            # reset the whole group to the next episode (same scene, next pool entry).
            # This runs BEFORE the exhaustion check so reset envs clear _slot_done
            # and the exhaustion flag is not set prematurely.
            should_roll_eval_pool = (
                self.is_eval
                and self.eval_roll_through_episode_pool
                and not self.cyclic_episode_sampling
            )
            if self.group_size > 1 or should_roll_eval_pool:
                self._maybe_reset_complete_groups(newly_done_idx)
                obs = self._build_obs()  # rebuild after potential pose reset
        else:
            obs = self._build_obs()

        # Check full exhaustion
        if not self._exhausted and not np.any(self._slot_active & ~self._slot_done):
            self._exhausted = True
            self._dump_per_scene_metrics()
            scene_label = (
                str(self._scene_layout.scenes)
                if self._scene_layout is not None
                else f"'{self._pinned_scene_id}'"
            )
            print(f"[GenArk] Worker exhausted: scene={scene_label} "
                  f"all {int(self._slot_active.sum())} episodes done. "
                  "Entering dormant mode.", flush=True)

        return (
            obs,
            reward.cpu(),
            terminated,
            truncated,
            info,
        )

    def _build_obs(self) -> dict:
        """Assemble observation tensors from current state."""
        device = self._sim.device

        # RGB (+ optional depth) from last render (or first render after reset)
        if self._current_rgb is None:
            if self._enable_depth_obs:
                # Depth obs: render RGB + depth together; 4-dir still RGB-only.
                self._current_rgb, self._current_depth = self._sim.render_main_with_depth(
                    self._active_slot_count
                )
                self._current_rgb_extras = self._sim.render_4dir(self._active_slot_count)
            elif self._use_async_render:
                # First frame after reset: submit and immediately fetch (no pipelining yet).
                ref  = self._sim.render_main_async(self._active_slot_count)
                ref4 = self._sim.render_4dir_async(self._active_slot_count)
                self._current_rgb        = self._sim.fetch_render_main(ref)
                self._current_rgb_extras = self._sim.fetch_render_4dir(ref4)
            else:
                self._current_rgb        = self._sim.render_main(self._active_slot_count)
                self._current_rgb_extras = self._sim.render_4dir(self._active_slot_count)

        # (N, H, W, 3) → (N, 3, H, W) as uint8 tensor on GPU
        rgb_np  = self._current_rgb
        rgb_t   = torch.from_numpy(
            np.ascontiguousarray(rgb_np.transpose(0, 3, 1, 2))
        ).to(device)

        # Tokenise instructions
        ids_list, mask_list = [], []
        for inst in self._instructions:
            ids, mask = _tokenize_instruction(inst, self.max_seq_len)
            ids_list.append(ids)
            mask_list.append(mask)
        ids_t  = torch.tensor(np.stack(ids_list),  dtype=torch.int64,  device=device)
        mask_t = torch.tensor(np.stack(mask_list), dtype=torch.bool,   device=device)

        # RLinf's EnvOutput.prepare_observations() only keeps these 5 keys:
        #   main_images, wrist_images, extra_view_images, states, task_descriptions
        # Everything else is discarded before reaching the rollout worker.
        # Map our data into these slots:
        #   rgb (N,3,H,W) CHW → main_images (N,H,W,3) HWC  (RLinf standard)
        #   elapsed_steps      → states  (used by UniNaVidPolicy for cache reset)
        #   instruction text   → task_descriptions

        # CHW → HWC
        rgb_hwc = rgb_t.permute(0, 2, 3, 1).contiguous()

        # states: (N, 4) float32
        #   [:, 0] elapsed_steps  — used by policy for episode-reset detection
        #   [:, 1] cam_pos_hab_x  — Habitat X (metres); P2 depth projection
        #   [:, 2] cam_pos_hab_z  — Habitat Z (metres) = −Genesis Y; P2
        #   [:, 3] cam_yaw        — Genesis yaw (radians), 0=+X CCW; P2
        cam_hab   = self._sim.cam_pos_hab(self.camera_height)  # (N, 3) tensor
        cam_yaw_t = self._sim.cam_yaw                          # (N,)   tensor
        states_np = np.stack([
            self._elapsed_steps.astype(np.float32),
            cam_hab[:, 0].cpu().numpy().astype(np.float32),
            cam_hab[:, 2].cpu().numpy().astype(np.float32),
            cam_yaw_t.cpu().numpy().astype(np.float32),
        ], axis=1)  # (N, 4)
        states_t = torch.from_numpy(states_np).to(device)

        # extra_view_images: either depth obs (B-infra) or 4-dir RGB, never both.
        if self._enable_depth_obs and self._current_depth is not None:
            # Depth: shape (N, 1, H, W, 1) float32 — borrowed channel for B-infra.
            # Policy identifies it as depth by dtype=float32 and shape[1]==1.
            depth_np = self._current_depth  # (N, H, W) float32
            extra_t = torch.from_numpy(
                depth_np[:, None, :, :, None].astype(np.float32)
            ).to(device)
        elif self._current_rgb_extras is not None:
            # 4-dir extras: (N, 3, H, W, 3) uint8 → put into extra_view_images
            # Order: [left, right, behind] (front is in main_images)
            extra_t = torch.from_numpy(
                np.ascontiguousarray(self._current_rgb_extras)
            ).to(device)
        else:
            extra_t = None

        task_descriptions = [
            self._instructions[i]
            if self._slot_active[i] and not self._slot_done[i]
            else ""
            for i in range(self.num_envs)
        ]

        return {
            "main_images":     rgb_hwc,           # (N, H, W, 3) uint8 — front view
            "states":          states_t,           # (N, 4) float — [elapsed, hab_x, hab_z, yaw]
            "task_descriptions": task_descriptions,  # empty string marks dormant slots
            # Kept for any direct callers that bypass prepare_observations
            "wrist_images":    None,
            "extra_view_images": extra_t,         # depth (N,1,H,W,1) float32 or 4-dir (N,3,H,W,3) uint8
        }

    def _handle_slot_done(
        self, done_idx: list[int], final_obs: dict, info: dict
    ) -> tuple[dict, dict]:
        """Single-pass completion: emit metrics, mark slot done, NO episode reset."""
        saved_obs  = copy.deepcopy(final_obs)
        saved_info = copy.deepcopy(info)

        ep_metrics_by_env = self._compute_episode_metrics(done_idx)

        for i in done_idx:
            self._slot_done[i] = True
            self._elapsed_steps[i] = 0
            m   = ep_metrics_by_env.get(i, {})
            ep_id = self._episodes[i].get("episode_id", f"_env{i}")
            # Bug #10: write the slot's actual scene_id, not the stale _pinned_scene_id.
            slot_scene_id = (
                self._scene_layout.scene_id_of(i)
                if self._scene_layout is not None
                else self._pinned_scene_id
            )
            acts = self._action_history[i]
            parse_fail_count = sum(1 for a in acts if a >= 4)
            last_action = acts[-1] if acts else None
            termination_cause = "stop" if last_action == self.STOP else "timeout"
            success_val = int(m.get("success", 0))
            start_dtg = float(self._start_dtg[i])
            final_dtg = float(m.get("distance_to_goal", 0.0))
            dists = self._distances_lists[i]
            min_dtg = float(min(dists)) if dists else final_dtg
            best_dtg_progress = start_dtg - min_dtg
            final_regression = final_dtg - min_dtg
            stop_step = len(acts) if termination_cause == "stop" else 0
            stop_dtg = final_dtg if termination_cause == "stop" else 0.0
            early_stop = bool(
                termination_cause == "stop"
                and stop_step <= 5
                and final_dtg > self.success_distance
            )
            trial_index = self._completed_trial_count(slot_scene_id, ep_id) + 1
            if success_val:
                if start_dtg < self.success_distance:
                    success_type = "lucky_start"
                elif parse_fail_count > 0:
                    success_type = "recovered"
                else:
                    success_type = "clean"
            else:
                success_type = "wrong_stop" if termination_cause == "stop" else "no_stop"
            ep_record = {
                "env_id": int(i),
                "episode_id": ep_id,
                "scene_id":   slot_scene_id,
                "trial_index": int(trial_index),
                "parse_fail_count": int(parse_fail_count),
                "success_type": success_type,
                "termination_cause": termination_cause,
                "start_distance_to_goal": start_dtg,
                "dtg_progress": start_dtg - final_dtg,
                "stop_step": int(stop_step),
                "stop_dtg": float(stop_dtg),
                "min_dtg": float(min_dtg),
                "best_dtg_progress": float(best_dtg_progress),
                "final_regression": float(final_regression),
                "early_stop": float(early_stop),
                **{k: float(v) for k, v in m.items()},
            }
            self._episode_log.append(ep_record)
            self._last_completed_diag_by_env[i] = ep_record
            self._completed_diag_event_by_env[i] = dict(ep_record)
            self._pending_decision_reward_snapshot_by_env[i] = {
                "episode_id": ep_id,
                "ep": self._episodes[i],
                "pred": np.array(self._pred_path_lists[i], dtype=float).copy(),
                "curr_ndtw": float(m.get("ndtw", self._prev_ndtw[i])),
                "prev_ndtw": float(self._prev_ndtw[i]),
                "dtg_before": float(self._dtg_decision_start[i].item()),
                "curr_dtg": final_dtg,
                "termination_cause": termination_cause,
                "success": float(m.get("success", 0.0)),
                "min_dtg": min_dtg,
                "start_dtg": start_dtg,
                "process_paid": float(self._process_progress_paid[i].item()),
                "last_json_valid": bool(self._last_json_valid[i]),
                "last_struct_ok": bool(self._last_struct_ok[i]),
                "last_field_format_ok": bool(self._last_field_format_ok[i]),
                "last_length_ok": bool(self._last_length_ok[i]),
                "last_has_bbox": bool(self._last_has_bbox[i]),
                "structured_paid": float(self._structured_reward_given[i].item()),
                "bbox_paid": float(self._bbox_reward_given[i].item()),
            }
            balancer = self._balancer_by_scene.get(slot_scene_id)
            if balancer is not None:
                balancer.record(ep_id, bool(m.get("success", 0)))
            print(
                f"[GenArk] Episode done  env={i}  ep={ep_id}"
                f"  success={m.get('success', 0):.0f}"
                f"  SPL={m.get('spl', 0):.3f}"
                f"  DTG={m.get('distance_to_goal', 0):.2f}m"
                f"  steps={m.get('steps_taken', 0)}",
                flush=True,
            )

            # --- Episode-end termination-reason diagnostic ---
            if self._ep_diag_enabled:
                last = acts[-1] if acts else None
                n_pf = parse_fail_count
                cause = termination_cause
                # Compact action string: STOP=S FWD=F LEFT=L RIGHT=R PARSE=P
                _LET = {0: "S", 1: "F", 2: "L", 3: "R", 4: "P"}
                seq = "".join(_LET.get(min(a, 4), "?") for a in acts)
                if len(seq) > 60:
                    seq = seq[:30] + "..." + seq[-15:]
                # Compact DTG curve: 5 sample points
                if dists:
                    n = len(dists)
                    samp_idx = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
                    dtg_str = "→".join(f"{dists[k]:.2f}" for k in samp_idx)
                else:
                    dtg_str = "(no dist data)"
                print(
                    f"[GenArk][ep-diag] env={i} ep={ep_id} "
                    f"cause={cause} success_type={success_type} "
                    f"steps={len(acts)} parse_fail={n_pf} "
                    f"start_dtg={start_dtg:.2f}m "
                    f"final_dtg={m.get('distance_to_goal', 0):.2f}m "
                    f"stop_dtg={stop_dtg:.2f}m "
                    f"min_dtg={min_dtg:.2f}m "
                    f"best_prog={best_dtg_progress:.2f}m "
                    f"final_reg={final_regression:.2f}m "
                    f"early_stop={int(early_stop)} "
                    f"ndtw={m.get('ndtw', 0):.3f} "
                    f"dtg_curve=[{dtg_str}] "
                    f"acts={seq}",
                    flush=True,
                )

        episode_info = self._build_episode_tensors(done_idx, ep_metrics_by_env)
        info["episode"]        = episode_info
        saved_info["episode"]  = episode_info
        info["final_observation"]  = saved_obs
        info["final_info"]         = saved_info
        info["_final_observation"] = np.isin(
            np.arange(self.num_envs), done_idx
        ).astype(bool)
        return final_obs, info

    def _completed_trial_count(self, scene_id: str, episode_id) -> int:
        """Number of completed eval trials for one scene/episode pair."""
        ep_key = str(episode_id)
        return sum(
            1
            for rec in self._episode_log
            if str(rec.get("scene_id", "")) == str(scene_id)
            and str(rec.get("episode_id", "")) == ep_key
        )

    def _maybe_reset_complete_groups(self, newly_done_idx: list[int]) -> None:
        """Group-level reset: when all envs in a GRPO group finish their episode,
        reset the entire group to the next episode from the pool.

        Each group of `group_size` envs shares one episode so GRPO can compare
        rewards within the group. Resetting them together to a NEW episode
        (same scene, different task) preserves this invariant across resets.

        Multi-scene: each group resolves its scene via slot_to_scene_idx and draws
        the next episode from that scene's independent pool — groups never straddle
        scene boundaries (enforced by SceneLayout group-alignment invariants).

        If the episode pool is exhausted, the group stays dormant (existing behavior).
        """
        gs = max(self.group_size, 1)
        layout = self._scene_layout  # None in single-scene mode

        for env_i in newly_done_idx:
            group_id    = env_i // gs
            group_start = group_id * gs
            group_envs  = [
                j for j in range(group_start, group_start + gs)
                if j < self.num_envs and self._slot_active[j]
            ]

            self._group_done_counts[group_id] = (
                self._group_done_counts.get(group_id, 0) + 1
            )
            if self._group_done_counts[group_id] < len(group_envs):
                continue  # still waiting for group-mates

            # All active envs in the group are done
            del self._group_done_counts[group_id]

            # Resolve which scene this group belongs to.
            scene_id = (
                layout.scene_id_of(group_start)
                if layout is not None
                else self._pinned_scene_id
            )

            # Bug #2 runtime backstop: all group members must be in the same scene.
            if layout is not None:
                scene_ids_in_group = {layout.scene_id_of(j) for j in group_envs}
                assert len(scene_ids_in_group) == 1, (
                    f"[P0] _maybe_reset_complete_groups: group {group_id} straddles "
                    f"scenes {scene_ids_in_group}. SceneLayout alignment broken."
                )

            # Resolve per-scene episode-pool state
            pool     = self._pool_by_scene[scene_id]
            rng      = self._rng_by_scene[scene_id]
            balancer = self._balancer_by_scene.get(scene_id)
            eval_repeat_mode = (
                self.is_eval
                and self.eval_roll_through_episode_pool
                and not self.cyclic_episode_sampling
                and balancer is None
            )

            if balancer is not None:
                # Balanced mode: EpisodeBalancer picks next episode by
                # (seen_count, success_rate) lexicographic order.
                next_ep, new_pass = balancer.next_episode()
                ep_id = next_ep.get("episode_id", "?")
                if new_pass:
                    self._episode_cycle_by_scene[scene_id] += 1
                    cycle = self._episode_cycle_by_scene[scene_id]
                    summ  = self._summarize_episode_log()
                    print(
                        f"[GenArk][episode-cycle] scene={scene_id} "
                        f"cycle={cycle} (balanced) "
                        f"sr={summ.get('success', 0):.3f} "
                        f"spl={summ.get('spl', 0):.3f} "
                        f"ndtw={summ.get('ndtw', 0):.3f} "
                        f"n={summ.get('num_episodes', 0)}",
                        flush=True,
                    )
            else:
                current_ep = self._episodes[group_envs[0]]
                current_ep_id = (
                    current_ep.get("episode_id", None)
                    if current_ep is not None else None
                )
                completed_trials = (
                    self._completed_trial_count(scene_id, current_ep_id)
                    if current_ep_id is not None else 0
                )
                if eval_repeat_mode and current_ep is not None and completed_trials < self.eval_repeats_per_episode:
                    next_ep = current_ep
                    ep_id = current_ep.get("episode_id", "?")
                elif self._next_ep_idx_by_scene[scene_id] >= len(pool):
                    if self.cyclic_episode_sampling:
                        # Reshuffle and restart this scene's pool (train mode).
                        # Avoid immediately repeating the episode just run.
                        last_ep_id = (self._episodes[group_envs[0]] or {}).get(
                            "episode_id", None
                        )
                        rng.shuffle(pool)
                        if (
                            last_ep_id is not None
                            and len(pool) > 1
                            and pool[0].get("episode_id") == last_ep_id
                        ):
                            self._pool_by_scene[scene_id] = pool[1:] + pool[:1]
                            pool = self._pool_by_scene[scene_id]
                        self._next_ep_idx_by_scene[scene_id] = 0
                        next_ep_idx = 0
                        self._episode_cycle_by_scene[scene_id] += 1
                        cycle = self._episode_cycle_by_scene[scene_id]
                        summ  = self._summarize_episode_log()
                        print(
                            f"[GenArk][episode-cycle] scene={scene_id} "
                            f"cycle={cycle} "
                            f"reshuffled {len(pool)} episodes | "
                            f"sr={summ.get('success', 0):.3f} "
                            f"spl={summ.get('spl', 0):.3f} "
                            f"ndtw={summ.get('ndtw', 0):.3f} "
                            f"n={summ.get('num_episodes', 0)}",
                            flush=True,
                        )
                        next_ep = pool[self._next_ep_idx_by_scene[scene_id]]
                        ep_id = next_ep.get(
                            "episode_id", self._next_ep_idx_by_scene[scene_id]
                        )
                        self._next_ep_idx_by_scene[scene_id] += 1
                    else:
                        # Single-pass mode (eval) — group enters dormant permanently.
                        print(
                            f"[GenArk][group-reset] group={group_id} scene={scene_id} "
                            f"pool exhausted (next_ep_idx={self._next_ep_idx_by_scene[scene_id]}, "
                            f"pool_size={len(pool)}); entering dormant.",
                            flush=True,
                        )
                        continue

                else:
                    # Assign the next pooled episode to all envs in the group.
                    next_ep = pool[self._next_ep_idx_by_scene[scene_id]]
                    ep_id   = next_ep.get("episode_id", self._next_ep_idx_by_scene[scene_id])
                    self._next_ep_idx_by_scene[scene_id] += 1

            for j in group_envs:
                self._episodes[j]     = next_ep
                self._instructions[j] = next_ep["instruction"]["instruction_text"]
                self._slot_done[j]    = False
                self._elapsed_steps[j] = 0
                self._stop_called[j]  = False
                self._action_history[j] = []
                self._pred_path_lists[j] = []
                self._distances_lists[j] = []

            # P0 guard: all envs in this group must share the exact same episode object.
            assert len({id(self._episodes[j]) for j in group_envs}) == 1, (
                f"[P0] group {group_id} split across episodes after reset: "
                f"{[self._episodes[j].get('episode_id') for j in group_envs]}"
            )

            # Re-initialise agent poses and metrics for the reset group
            self._init_agent_poses(group_envs)

            if balancer is not None:
                _seen_vals = list(balancer._seen.values())
                _pool_info = (f"balanced seen min={min(_seen_vals)} max={max(_seen_vals)}"
                              f" passes={balancer._passes}")
            elif eval_repeat_mode:
                completed = self._completed_trial_count(scene_id, ep_id)
                remaining = len(pool) - self._next_ep_idx_by_scene[scene_id]
                _pool_info = (
                    f"trial {completed + 1}/{self.eval_repeats_per_episode}, "
                    f"pool remaining: {remaining}"
                )
            else:
                remaining = len(pool) - self._next_ep_idx_by_scene[scene_id]
                _pool_info = f"pool remaining: {remaining}"
            print(
                f"[GenArk][group-reset] group={group_id} scene={scene_id} → ep={ep_id} "
                f"envs={group_envs} ({_pool_info})",
                flush=True,
            )

    def _dormant_step_crash_recovery(
        self,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Dormant step issued while a remote scene actor is being rebuilt.

        Every `_crash_recovery_cooldown` calls we check if the actor is healthy
        again.  Once healthy, we re-initialise agent poses and clear the crash
        flag so normal stepping resumes on the next call.
        """
        if self._crash_recovery_cooldown > 0:
            self._crash_recovery_cooldown -= 1
        else:
            # Poll health via backend abstraction (works for remote + zmq).
            self._crash_recovery_cooldown = 50
            if self._sim.is_scene_healthy():
                print(
                    f"[GenArk][crash-recovery] scene healthy again, "
                    "re-initialising agent poses.",
                    flush=True,
                )
                try:
                    active_idx = [
                        i for i in range(self.num_envs)
                        if self._slot_active[i] and self._episodes[i] is not None
                    ]
                    if active_idx:
                        self._init_agent_poses(active_idx)
                    self._scene_crashed          = False
                    self._pending_rgb_ref        = None
                    self._pending_rgb_extras_ref = None
                    self._current_rgb            = None
                    self._current_rgb_extras     = None
                except Exception as e:
                    print(
                        f"[GenArk][crash-recovery] re-init failed: {e}, "
                        "staying dormant.",
                        flush=True,
                    )
        return self._dormant_step()

    def _dormant_step(self) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Cheap no-op step issued after the worker's episode pool is exhausted.

        Returns same-shape tensors as a real step but skips Genesis render
        and inference-relevant work. info["episode"] is intentionally empty
        so RLinf's eval aggregator ignores these steps.
        """
        N = self.num_envs
        # Build / reuse a tiny dummy obs in the same key layout as _build_obs.
        if self._dummy_obs_cache is None:
            dev = self._sim.device
            dummy_rgb = torch.zeros((N, self.cam_h, self.cam_w, 3),
                                    dtype=torch.uint8, device=dev)
            dummy_states = torch.zeros((N, 4), dtype=torch.float32, device=dev)
            # extra_view_images: (N, 3, H, W, 3) uint8 when enable_4dir_render, else None.
            # wrist_images: GenArk has no wrist camera → always None.
            dummy_extra = (
                torch.zeros((N, 3, self.cam_h, self.cam_w, 3),
                            dtype=torch.uint8, device=dev)
                if self.enable_4dir_render else None
            )
            self._dummy_obs_cache = {
                "main_images":       dummy_rgb,
                "states":            dummy_states,
                "task_descriptions": [""] * N,
                "wrist_images":      None,
                "extra_view_images": dummy_extra,
            }
        reward     = torch.zeros(N, dtype=torch.float32, device="cpu")
        # IMPORTANT: terminated=False, truncated=False — we do NOT signal
        # episode end. If we did, RLinf's aggregator would treat each dormant
        # step as a fresh episode completion and add zeros to the SR average.
        # Instead the env quietly idles until max_steps_per_rollout_epoch.
        terminated = torch.zeros(N, dtype=torch.bool, device="cpu")
        truncated  = torch.zeros(N, dtype=torch.bool, device="cpu")
        info = {
            "distance_to_goal": np.zeros(N, dtype=np.float32),
            "success":          np.zeros(N, dtype=np.float32),
            "elapsed_steps":    np.zeros(N, dtype=np.int32),
        }
        return self._dummy_obs_cache, reward, terminated, truncated, info

    def _dump_per_scene_metrics(self):
        """Dump this worker's per-episode metrics to a JSON file under the
        eval log directory (alongside avg_metrics.json)."""
        try:
            log_dir = getattr(self.cfg.video_cfg, "video_base_dir", None)
            if log_dir:
                # video_base_dir is .../video/eval — go up to .../genark_eval
                base = os.path.dirname(os.path.dirname(log_dir))
            else:
                base = "/tmp"
            os.makedirs(base, exist_ok=True)
            if self._scene_layout is not None:
                # Multi-scene: use a worker-level name listing all scenes
                scan = "multiscene_" + "_".join(
                    os.path.basename(os.path.dirname(sid))
                    for sid in self._scene_layout.scenes
                )
            else:
                scan = os.path.basename(os.path.dirname(self._pinned_scene_id))
            out_path = os.path.join(base, f"per_scene_{scan}.json")
            unique_episode_count = len({
                (str(e.get("scene_id", "")), str(e.get("episode_id", "")))
                for e in self._episode_log
            })
            payload = {
                "scene_id":  (
                    self._scene_layout.scenes
                    if self._scene_layout is not None
                    else self._pinned_scene_id
                ),
                "scan_name": scan,
                "n_episodes": unique_episode_count,
                "n_trials": len(self._episode_log),
                "n_unique_episodes": unique_episode_count,
                "episodes":  self._episode_log,
                "summary":   self._summarize_episode_log(),
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[GenArk] Per-scene metrics saved → {out_path}", flush=True)
            self._dump_zero_shot_difficulty_summary(base)
        except Exception as e:
            print(f"[GenArk] WARNING: could not dump per-scene metrics: {e}",
                  flush=True)

    def _dump_zero_shot_difficulty_summary(self, base: str) -> None:
        """Merge eval trials and emit per-episode averaged difficulty summaries.

        Multiple env workers may finish at different times, so this is deliberately
        idempotent. The final worker to exhaust will write the complete merged view.
        """
        trials: list[dict] = []
        for path in sorted(Path(base).glob("per_scene_*.json")):
            try:
                with open(path, "r") as f:
                    payload = json.load(f)
            except Exception:
                continue
            for ep in payload.get("episodes", []):
                trials.append(ep)

        if not trials:
            return

        trials_path = os.path.join(base, "all_episode_trials.json")
        with open(trials_path, "w") as f:
            json.dump({
                "n_trials": len(trials),
                "trials": trials,
            }, f, indent=2)

        grouped: dict[tuple[str, str], list[dict]] = {}
        for trial in trials:
            key = (str(trial.get("scene_id", "")), str(trial.get("episode_id", "")))
            grouped.setdefault(key, []).append(trial)

        numeric_keys = [
            "success", "spl", "ndtw", "sdtw",
            "distance_to_goal", "path_length", "steps_taken",
            "parse_fail_count", "dtg_progress",
        ]
        episodes: list[dict] = []
        for (scene_id, ep_id), ep_trials in sorted(grouped.items()):
            avg = {
                k: float(sum(float(t.get(k, 0.0)) for t in ep_trials) / len(ep_trials))
                for k in numeric_keys
            }
            success_type_counts: dict[str, int] = {}
            for t in ep_trials:
                st = str(t.get("success_type", "unknown"))
                success_type_counts[st] = success_type_counts.get(st, 0) + 1
            episodes.append({
                "scene_id": scene_id,
                "episode_id": ep_id,
                "n_trials": len(ep_trials),
                **avg,
                "success_rate": avg["success"],
                "success_type_counts": success_type_counts,
            })

        all_path = os.path.join(base, "all_episode_metrics.json")
        with open(all_path, "w") as f:
            json.dump({
                "n_episodes": len(episodes),
                "n_trials": len(trials),
                "eval_repeats_per_episode": self.eval_repeats_per_episode,
                "episodes": episodes,
            }, f, indent=2)

        def _mean(key: str) -> float:
            return float(sum(float(e.get(key, 0.0)) for e in episodes) / len(episodes))

        primary_buckets = {"easy": [], "medium": [], "hard": []}
        tags = {
            "format_hard": [],
            "stop_hard": [],
            "exploration_hard": [],
        }
        for ep in episodes:
            success_rate = float(ep.get("success_rate", ep.get("success", 0.0)))
            ndtw = float(ep.get("ndtw", 0.0))
            dtg_progress = float(ep.get("dtg_progress", 0.0))
            success_type_counts = ep.get("success_type_counts", {})
            ep_ref = {
                "scene_id": ep.get("scene_id", ""),
                "episode_id": ep.get("episode_id", ""),
                "n_trials": ep.get("n_trials", 0),
                "success_rate": success_rate,
            }

            if success_rate >= 0.5:
                primary_buckets["easy"].append(ep_ref)
            elif success_rate > 0.0 or ndtw >= 0.35 or dtg_progress > 0.0:
                primary_buckets["medium"].append(ep_ref)
            else:
                primary_buckets["hard"].append(ep_ref)

            if float(ep.get("parse_fail_count", 0.0)) > 0.0:
                tags["format_hard"].append(ep_ref)
            if int(success_type_counts.get("wrong_stop", 0)) > 0:
                tags["stop_hard"].append(ep_ref)
            if int(success_type_counts.get("no_stop", 0)) > 0:
                tags["exploration_hard"].append(ep_ref)

        summary = {
            "n_episodes": len(episodes),
            "n_trials": len(trials),
            "eval_repeats_per_episode": self.eval_repeats_per_episode,
            "mean_metrics": {
                k: _mean(k)
                for k in numeric_keys
            },
            "primary_buckets": {
                k: {"count": len(v), "episodes": v}
                for k, v in primary_buckets.items()
            },
            "diagnostic_tags": {
                k: {"count": len(v), "episodes": v}
                for k, v in tags.items()
            },
        }
        summary_path = os.path.join(base, "difficulty_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"[GenArk] Zero-shot difficulty summary saved → {summary_path} "
            f"(episodes={len(episodes)}, trials={len(trials)})",
            flush=True,
        )

    def get_grpo_process_diagnostics(self) -> list[dict]:
        """Return lightweight per-slot diagnostics for GRPO group monitoring.

        Read-only helper used by EnvWorker after a decision-rollout collection.
        It intentionally exposes only scalars that are cheap to aggregate and
        never changes reward or episode state.
        """
        out: list[dict] = []
        for i in range(self.num_envs):
            rec = (
                self._last_completed_diag_by_env[i]
                if i < len(self._last_completed_diag_by_env)
                else None
            )
            dists = self._distances_lists[i]
            if rec is not None:
                start_dtg = float(rec.get("start_distance_to_goal", 0.0))
                curr_dtg = float(rec.get("distance_to_goal", start_dtg))
                min_dtg = float(rec.get("min_dtg", curr_dtg))
                best_dtg_progress = float(
                    rec.get("best_dtg_progress", max(0.0, start_dtg - min_dtg))
                )
                final_regression = float(rec.get("final_regression", curr_dtg - min_dtg))
            else:
                start_dtg = float(self._start_dtg[i]) if i < len(self._start_dtg) else 0.0
                curr_dtg = float(dists[-1]) if dists else start_dtg
                min_dtg = float(min(dists)) if dists else curr_dtg
                best_dtg_progress = max(0.0, start_dtg - min_dtg)
                final_regression = curr_dtg - min_dtg
            episode_id = (
                rec.get("episode_id")
                if rec is not None
                else (self._episodes[i] or {}).get("episode_id", "")
            )
            out.append({
                "env_id": int(i),
                "episode_id": episode_id,
                "success": float(rec.get("success", 0.0)) if rec is not None else 0.0,
                "success_type": str(rec.get("success_type", "")) if rec is not None else "",
                "termination_cause": str(rec.get("termination_cause", "")) if rec is not None else "",
                "start_dtg": start_dtg,
                "curr_dtg": curr_dtg,
                "min_dtg": min_dtg,
                "best_dtg_progress": best_dtg_progress,
                "final_regression": final_regression,
            })
        return out

    def pop_completed_episode_diagnostic(self, env_id: int) -> dict | None:
        """Return and clear the latest completion diagnostic for one env slot."""
        if env_id < 0 or env_id >= len(self._completed_diag_event_by_env):
            return None
        rec = self._completed_diag_event_by_env[env_id]
        self._completed_diag_event_by_env[env_id] = None
        return dict(rec) if isinstance(rec, dict) else None

    @staticmethod
    def _empty_gsam_reward_diag_counts() -> dict[str, float]:
        return {
            "decisions": 0.0,
            "used": 0.0,
            "detected": 0.0,
            "detected_and_progress": 0.0,
            "detected_but_regressed": 0.0,
            "bonus": 0.0,
        }

    def pop_gsam_reward_diagnostics(self) -> dict[str, float]:
        """Return and reset GroundedSAM reward-gate diagnostics."""
        counts = self._gsam_reward_diag_counts
        self._gsam_reward_diag_counts = self._empty_gsam_reward_diag_counts()
        n = max(1.0, float(counts.get("decisions", 0.0)))
        if counts.get("decisions", 0.0) <= 0:
            return {}
        return {
            "gsam/used_rate": float(counts["used"] / n),
            "gsam/detect_rate": float(counts["detected"] / n),
            "gsam/detected_and_progress_rate": float(counts["detected_and_progress"] / n),
            "gsam/detected_but_regressed_rate": float(counts["detected_but_regressed"] / n),
            "gsam/bonus_count": float(counts["bonus"]),
        }

    def _summarize_episode_log(self) -> dict:
        if not self._episode_log:
            return {}
        keys = ["success", "spl", "ndtw", "sdtw",
                "distance_to_goal", "path_length", "steps_taken",
                "parse_fail_count", "dtg_progress"]
        n = len(self._episode_log)
        return {
            k: float(sum(e.get(k, 0.0) for e in self._episode_log) / n)
            for k in keys
        } | {"num_episodes": n}

    def _build_episode_tensors(
        self, done_idx: list[int], ep_metrics_by_env: dict
    ) -> dict:
        """Build per-env episode metric tensors compatible with RLinf env_worker."""
        keys = ["success", "spl", "ndtw", "sdtw", "distance_to_goal",
                "path_length", "steps_taken"]
        arrays = {k: np.zeros(self.num_envs, dtype=np.float32) for k in keys}
        for i in done_idx:
            m = ep_metrics_by_env.get(i, {})
            for k in keys:
                arrays[k][i] = float(m.get(k, 0.0))
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in arrays.items()}

    def _compute_ndtw_sr_delta_reward(
        self,
        curr_dist: torch.Tensor,
        newly_stopped: torch.Tensor,
        N_act: int,
        include_ndtw: bool = True,
    ) -> torch.Tensor:
        """
        RFT reward = ndtw_coef * (nDTW_t - nDTW_{t-1}) + sr_coef * (SR_t - SR_{t-1}).

        - per-step SR  : 1 if (just stopped AND d2g < success_distance) else 0
        - per-step nDTW: fastdtw on current partial pred_path vs full GT reference_path
                         (skipped when include_ndtw=False, i.e. decision_level_ndtw=True;
                          nDTW delta is instead computed once per decision via
                          compute_decision_ndtw_reward())

        Caveats: see docs/genark_rft_caveats.md C4 (per-step nDTW unstable),
        C5 (SR delta sparse), C6 (scale imbalance).
        """
        N = self.num_envs
        reward_np = np.zeros(N, dtype=np.float32)

        curr_dist_np = curr_dist.detach().cpu().numpy()
        stopped_np = newly_stopped.detach().cpu().numpy()

        for i in range(N_act):
            if not self._active_mask[i]:
                continue

            # SR delta — binary 0→1 at success step (always computed per env step).
            # Gate: episodes BORN within success_distance ("lucky_start") get NO SR —
            # reaching the goal you spawned next to is not navigation, and the flat
            # +sr_coef bonus otherwise creates a degenerate "STOP immediately" optimum
            # (curriculum spawns ≤5m → free +2). Only born-far episodes earn SR.
            born_far = self._start_dtg[i] >= self.success_distance
            curr_sr = 1.0 if (born_far and stopped_np[i] and curr_dist_np[i] < self.success_distance) else 0.0
            sr_delta = curr_sr - self._prev_sr[i]
            self._prev_sr[i] = curr_sr

            if include_ndtw:
                # nDTW delta — fastdtw on partial path vs GT reference
                ep = self._episodes[i]
                pred = self._pred_path_lists[i]
                curr_ndtw = 1.0
                if ep is not None and len(pred) >= 2:
                    gt = np.array(ep.get("reference_path", [[0, 0, 0]]), dtype=float)
                    pred_arr = np.array(pred)
                    try:
                        dtw_d = fastdtw(pred_arr, gt, dist=euclidean)[0]
                        curr_ndtw = float(np.exp(-dtw_d / (len(gt) * self.success_distance)))
                    except Exception:
                        curr_ndtw = self._prev_ndtw[i]
                ndtw_delta = curr_ndtw - self._prev_ndtw[i]
                self._prev_ndtw[i] = curr_ndtw
            else:
                ndtw_delta = 0.0  # deferred to compute_decision_ndtw_reward()

            coeffs = self._reward_coeffs()
            reward_np[i] = coeffs["ndtw"] * ndtw_delta + coeffs["sr"] * sr_delta

        return torch.from_numpy(reward_np).to(curr_dist.device)

    def _wrong_stop_reward(self, *, curr_dtg: float, start_dtg: float) -> float:
        """Penalty for STOP outside the success radius.

        Baseline behavior is the original flat penalty.  When
        conditional_wrong_stop_penalty is enabled, farther wrong STOPs are
        penalized more strongly via curr_dtg / start_dtg, clipped for stability.
        """
        base = -abs(self._reward_coeffs()["wrong_stop"])
        if not self.conditional_wrong_stop_penalty:
            return base
        denom = max(float(start_dtg), 1e-6)
        ratio = float(curr_dtg) / denom
        ratio = float(
            np.clip(
                ratio,
                self.conditional_wrong_stop_ratio_min,
                self.conditional_wrong_stop_ratio_max,
            )
        )
        return base * ratio

    def compute_decision_ndtw_reward(
        self,
        env_indices: list[int],
        policy_diag_by_env: Optional[dict[int, dict]] = None,
    ) -> torch.Tensor:
        """
        Compute decision-level rewards (nDTW delta + optional DTG delta + format bonus)
        once per LLM decision for the given env indices.

        Called by env_worker at decision flush time (when is_decision=True).
        At that point _pred_path_lists[i] already contains all positions from the
        last decision up to the current env step, including intermediate macro steps.
        This means detours within a macro action are correctly captured by DTW.

        For reward_mode="decision_nav", also computes decision-level DTG reward:
            delta = DTG_before_decision - DTG_after_decision
            reward = clip(delta * dtg_coef, -dtg_clip, dtg_clip), asymmetric:
                     positive delta (progress) × 1.0, negative delta (regress) × 0.5

        Updates self._prev_ndtw[i] and self._dtg_decision_start[i] for each env.
        Returns a [num_envs] float tensor; envs not in env_indices get 0.
        """
        reward_np = np.zeros(self.num_envs, dtype=np.float32)
        coeffs = self._reward_coeffs()
        for i in env_indices:
            snapshot = self._pending_decision_reward_snapshot_by_env[i]
            if snapshot is None and not self._active_mask[i]:
                continue
            if snapshot is not None:
                self._pending_decision_reward_snapshot_by_env[i] = None
                ep = snapshot.get("ep")
                pred = snapshot.get("pred", [])
                prev_ndtw = float(snapshot.get("prev_ndtw", self._prev_ndtw[i]))
                dtg_before = float(snapshot.get("dtg_before", 0.0))
                curr_dtg = float(snapshot.get("curr_dtg", dtg_before))
                termination_cause = str(snapshot.get("termination_cause", ""))
                min_dtg = float(snapshot.get("min_dtg", curr_dtg))
                start_dtg = float(snapshot.get("start_dtg", curr_dtg))
                process_paid = float(snapshot.get("process_paid", 0.0))
                last_json_valid = bool(snapshot.get("last_json_valid", False))
                last_struct_ok = bool(snapshot.get("last_struct_ok", False))
                last_field_format_ok = bool(
                    snapshot.get("last_field_format_ok", False)
                )
                last_length_ok = bool(snapshot.get("last_length_ok", False))
                last_has_bbox = bool(snapshot.get("last_has_bbox", False))
                structured_paid = float(snapshot.get("structured_paid", 0.0))
                bbox_paid = float(snapshot.get("bbox_paid", 0.0))
            else:
                ep = self._episodes[i]
                pred = self._pred_path_lists[i]
                prev_ndtw = float(self._prev_ndtw[i])
                dtg_before = float(self._dtg_decision_start[i])
                dists = self._distances_lists[i]
                curr_dtg = float(dists[-1]) if dists else dtg_before
                termination_cause = ""
                min_dtg = float(self._min_dtg_t[i].item())
                start_dtg = float(self._start_dtg[i])
                process_paid = float(self._process_progress_paid[i].item())
                last_json_valid = bool(self._last_json_valid[i])
                last_struct_ok = bool(self._last_struct_ok[i])
                last_field_format_ok = bool(self._last_field_format_ok[i])
                last_length_ok = bool(self._last_length_ok[i])
                last_has_bbox = bool(self._last_has_bbox[i])
                structured_paid = float(self._structured_reward_given[i].item())
                bbox_paid = float(self._bbox_reward_given[i].item())

            # --- nDTW delta ---
            curr_ndtw = 1.0
            if snapshot is not None:
                curr_ndtw = float(snapshot.get("curr_ndtw", prev_ndtw))
            elif ep is not None and len(pred) >= 2:
                gt = np.array(ep.get("reference_path", [[0, 0, 0]]), dtype=float)
                pred_arr = np.array(pred)
                try:
                    dtw_d = fastdtw(pred_arr, gt, dist=euclidean)[0]
                    curr_ndtw = float(np.exp(-dtw_d / (len(gt) * self.success_distance)))
                except Exception:
                    curr_ndtw = prev_ndtw
            ndtw_r = coeffs["ndtw"] * (curr_ndtw - prev_ndtw)
            if snapshot is None:
                self._prev_ndtw[i] = curr_ndtw

            # --- decision-level DTG reward (decision_nav mode only) ---
            dtg_r = 0.0
            decision_progress = 0.0
            if self.reward_mode == "decision_nav":
                delta = dtg_before - curr_dtg  # positive = made progress toward goal
                decision_progress = delta
                dtg_coef = coeffs["dtg"]
                scaled = delta * dtg_coef if delta > 0 else 0.5 * delta * dtg_coef
                dtg_r = float(np.clip(scaled, -self.decision_dtg_clip, self.decision_dtg_clip))
                if snapshot is None:
                    self._dtg_decision_start[i] = curr_dtg  # update for next decision

            terminal_stop_r = 0.0
            if self.reward_mode == "decision_nav" and snapshot is not None:
                is_stop = termination_cause == "stop"
                born_far = start_dtg >= self.success_distance
                if is_stop and born_far and curr_dtg < self.success_distance:
                    terminal_stop_r += coeffs["sr"]
                elif is_stop and curr_dtg > self.success_distance:
                    terminal_stop_r += self._wrong_stop_reward(
                        curr_dtg=curr_dtg,
                        start_dtg=start_dtg,
                    )

            # --- RL-Struct style structured-output reward ---
            # Split the old monolithic format bonus into progressive components:
            # JSON validity → required fields → field/value format → length guard.
            # This prevents "invalid JSON" and "point outside bbox" from sharing
            # the same reward signal, while the per-episode cap keeps structure as
            # a guardrail instead of letting it dominate navigation.
            struct_r = 0.0
            if last_json_valid:
                struct_r += coeffs["json"]
            if last_struct_ok:
                struct_r += coeffs["struct"]
            if last_field_format_ok:
                struct_r += coeffs["field_format"]
            if last_length_ok:
                struct_r += coeffs["length"]
            if struct_r > 0.0:
                remaining = max(0.0, self.structured_reward_cap - structured_paid)
                struct_r = min(struct_r, remaining)
                if struct_r > 0.0 and snapshot is None:
                    self._structured_reward_given[i] += struct_r
            # --- bbox reward (B1: bonus when policy outputs valid bbox_2d) ---
            # In lavira_waypoint mode this sentinel requires valid NAVIGATE bbox
            # geometry; parser already enforces point_2d inside bbox_2d.
            bbox_r = 0.0
            bbox_coef = coeffs["bbox"]
            if bbox_coef > 0 and last_has_bbox:
                remaining = max(0.0, self.bbox_reward_cap - bbox_paid)
                bbox_r = min(bbox_coef, remaining)
                if bbox_r > 0.0 and snapshot is None:
                    self._bbox_reward_given[i] += bbox_r

            process_r = 0.0
            if self.process_reward_enabled:
                best_progress = max(0.0, start_dtg - min_dtg)
                capped_progress = min(best_progress, max(0.0, self.process_progress_cap))
                new_progress = max(0.0, capped_progress - process_paid)
                if new_progress > 0.0:
                    process_r += self.process_progress_coef * new_progress
                    if snapshot is None:
                        self._process_progress_paid[i] += new_progress
                regression = curr_dtg - min_dtg
                if regression > self.process_regression_margin:
                    process_r += self.process_regression_penalty

            gsam_r = 0.0
            diag = policy_diag_by_env.get(i, {}) if policy_diag_by_env else {}
            if diag:
                gsam_used = bool(diag.get("grounded_sam_used", False))
                gsam_detected = bool(diag.get("grounded_sam_detected", False))
                is_stop = bool(diag.get("is_stop", False))
                progress_positive = decision_progress > 0.0
                counts = self._gsam_reward_diag_counts
                counts["decisions"] += 1.0
                counts["used"] += float(gsam_used)
                counts["detected"] += float(gsam_detected)
                counts["detected_and_progress"] += float(
                    gsam_detected and progress_positive and not is_stop
                )
                counts["detected_but_regressed"] += float(
                    gsam_detected and not progress_positive
                )
            if self.gsam_reward_enabled and diag:
                if (
                    bool(diag.get("grounded_sam_detected", False))
                    and decision_progress > 0.0
                    and not bool(diag.get("is_stop", False))
                ):
                    gsam_r += self.gsam_detect_progress_bonus
                    self._gsam_reward_diag_counts["bonus"] += 1.0

            reward_np[i] = (
                ndtw_r
                + dtg_r
                + terminal_stop_r
                + struct_r
                + bbox_r
                + process_r
                + gsam_r
            )
        return torch.from_numpy(reward_np).float()

    def _compute_episode_metrics(self, env_idx: list[int]) -> dict:
        """Returns {env_idx: metrics_dict} for each done env."""
        metrics = {}
        for i in env_idx:
            ep    = self._episodes[i]
            dists = self._distances_lists[i]
            pred  = np.array(self._pred_path_lists[i])
            gt    = np.array(ep.get("reference_path", [[0, 0, 0]]), dtype=float)

            ep_success = float(
                self._stop_called[i] and dists and dists[-1] < self.success_distance
            )
            gt_length  = dists[0] if dists else 0.0
            if len(pred) > 1:
                path_len = float(np.linalg.norm(np.diff(pred, axis=0), axis=1).sum())
            else:
                path_len = 0.0

            spl = (
                float(ep_success * gt_length / max(gt_length, path_len))
                if max(gt_length, path_len) > 0 else 0.0
            )
            try:
                dtw_d = fastdtw(pred, gt, dist=euclidean)[0]
                ndtw  = float(
                    np.exp(-dtw_d / (len(gt) * self.success_distance))
                )
            except Exception:
                ndtw = 0.0

            metrics[i] = {
                "success":          ep_success,
                "spl":              spl,
                "ndtw":             ndtw,
                "sdtw":             ndtw * ep_success,
                "distance_to_goal": dists[-1] if dists else float("nan"),
                "path_length":      path_len,
                "steps_taken":      len(pred) - 1,
            }
        return metrics

    # ------------------------------------------------------------------
    # Misc gym.Env stubs
    # ------------------------------------------------------------------

    @property
    def seed(self) -> int:
        return self.cfg.seed + self.seed_offset

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        return self._elapsed_steps.sum() == 0

    @is_start.setter
    def is_start(self, value: bool):
        # env_worker sets is_start=True to signal episode beginning; reset elapsed steps
        if value:
            self._elapsed_steps[:] = 0

    @property
    def info_logging_keys(self) -> list[str]:
        return ["success", "spl", "ndtw", "distance_to_goal"]

    def chunk_step(
        self, chunk_actions: torch.Tensor
    ) -> tuple[list, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        RLinf env_worker calls chunk_step(chunk_actions) instead of step().
        chunk_actions shape: (num_envs, num_action_chunks) — for nav, num_action_chunks=1.
        Returns (obs_list, reward_tensor, terminations, truncations, infos_list).
        reward_tensor shape: (num_envs, num_action_chunks)
        """
        ndim = chunk_actions.ndim if hasattr(chunk_actions, "ndim") else chunk_actions.dim()
        num_chunks = chunk_actions.shape[1] if ndim == 2 else 1
        obs_list, infos_list = [], []
        rewards      = torch.zeros(self.num_envs, num_chunks, dtype=torch.float32, device="cpu")
        terminations = torch.zeros(self.num_envs, num_chunks, dtype=torch.bool,    device="cpu")
        truncations  = torch.zeros(self.num_envs, num_chunks, dtype=torch.bool,    device="cpu")

        for c in range(num_chunks):
            actions_c = chunk_actions[:, c] if ndim == 2 else chunk_actions
            obs, reward, term, trunc, info = self.step(actions_c)
            obs_list.append(obs)
            infos_list.append(info)
            rewards[:, c]      = reward.cpu() if isinstance(reward, torch.Tensor) else torch.tensor(reward, dtype=torch.float32)
            terminations[:, c] = term
            truncations[:, c]  = trunc

        return obs_list, rewards, terminations, truncations, infos_list

    def render(self, mode="rgb_array"):
        return self._current_rgb

    def close(self):
        pass

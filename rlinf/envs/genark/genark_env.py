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

# Genesis is a heavy import; fail loudly if missing so the error message
# is clear rather than a cryptic AttributeError later.
try:
    import genesis as gs
except ImportError as e:
    raise ImportError(
        "GenarkVecEnv requires the Genesis simulator. "
        "Install it following the GenArk README before using this env."
    ) from e


# ---------------------------------------------------------------------------
# NavMesh physics helpers  (ported verbatim from genark/env_worker.py)
# ---------------------------------------------------------------------------

def _batch_cross_2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _batch_is_point_in_triangle(pt, v0, v1, v2):
    def sign(p1, p2, p3):
        return (
            (p1[..., 0] - p3[..., 0]) * (p2[..., 1] - p3[..., 1])
            - (p2[..., 0] - p3[..., 0]) * (p1[..., 1] - p3[..., 1])
        )
    d1, d2, d3 = sign(pt, v0, v1), sign(pt, v1, v2), sign(pt, v2, v0)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def _batch_get_z_on_triangle(v0, v1, v2, x, y):
    edge1  = v1 - v0
    edge2  = v2 - v0
    normal = torch.cross(edge1, edge2, dim=-1)
    mask   = torch.abs(normal[..., 2]) < 1e-6
    d_x    = x - v0[..., 0]
    d_y    = y - v0[..., 1]
    num    = normal[..., 0] * d_x + normal[..., 1] * d_y
    den    = normal[..., 2]
    den    = torch.where(torch.abs(den) < 1e-6, torch.ones_like(den), den)
    z      = v0[..., 2] - num / den
    return torch.where(mask.expand_as(z), torch.full_like(z, float("inf")), z)


def _batch_find_floor(
    pos_2d, current_floor_z,
    tri_v0_2d, tri_v1_2d, tri_v2_2d,
    tri_v0_3d, tri_v1_3d, tri_v2_3d,
    max_step_height: float = 0.5,
):
    pt    = pos_2d.unsqueeze(1)
    v0_2  = tri_v0_2d.unsqueeze(0)
    v1_2  = tri_v1_2d.unsqueeze(0)
    v2_2  = tri_v2_2d.unsqueeze(0)
    inside = _batch_is_point_in_triangle(pt, v0_2, v1_2, v2_2)

    v0_3  = tri_v0_3d.unsqueeze(0)
    v1_3  = tri_v1_3d.unsqueeze(0)
    v2_3  = tri_v2_3d.unsqueeze(0)
    z_vals = _batch_get_z_on_triangle(v0_3, v1_3, v2_3, pt[..., 0], pt[..., 1])

    z_diff     = torch.abs(z_vals - current_floor_z.unsqueeze(1))
    valid_cand = inside & (z_diff < max_step_height)
    z_diff_m   = torch.where(valid_cand, z_diff, torch.full_like(z_diff, float("inf")))

    min_dist, best_idx = torch.min(z_diff_m, dim=1)
    valid_mask = min_dist != float("inf")
    best_z     = torch.gather(z_vals, 1, best_idx.unsqueeze(1)).squeeze(1)
    return valid_mask, best_z, best_idx


def _batch_get_sliding_position(
    current_pos, desired_pos, current_tri_idx,
    tri_v0_2d, tri_v1_2d, tri_v2_2d,
    agent_radius: float = 0.0,
):
    p0 = tri_v0_2d[current_tri_idx]
    p1 = tri_v1_2d[current_tri_idx]
    p2 = tri_v2_2d[current_tri_idx]

    edge_starts = torch.stack([p0, p1, p2], dim=1)
    edge_ends   = torch.stack([p1, p2, p0], dim=1)
    edge_vecs   = edge_ends - edge_starts

    p_des  = desired_pos.unsqueeze(1)
    p_cur  = current_pos.unsqueeze(1)
    cp1    = _batch_cross_2d(edge_vecs, p_des - edge_starts)
    cp0    = _batch_cross_2d(edge_vecs, p_cur - edge_starts)
    crossing = (cp1 * cp0) < 0

    v_ap  = p_des - edge_starts
    dot   = (v_ap * edge_vecs).sum(dim=-1)
    sq    = (edge_vecs * edge_vecs).sum(dim=-1)
    sq_s  = torch.where(sq < 1e-6, torch.ones_like(sq), sq)
    t     = torch.clamp(dot / sq_s, 0.0, 1.0)
    candidates = edge_starts + t.unsqueeze(-1) * edge_vecs

    if agent_radius > 0.0:
        centroid  = ((p0 + p1 + p2) / 3.0).unsqueeze(1)
        perp_a    = torch.stack([-edge_vecs[..., 1], edge_vecs[..., 0]], dim=-1)
        perp_b    = -perp_a
        to_ctr    = centroid - edge_starts
        dot_a     = (perp_a * to_ctr).sum(dim=-1)
        inward    = torch.where(dot_a.unsqueeze(-1) > 0, perp_a, perp_b)
        inward_len = torch.norm(inward, dim=-1, keepdim=True).clamp_min(1e-6)
        inward_unit = inward / inward_len
        candidates  = candidates + inward_unit * agent_radius

    has_crossing, first_idx = torch.max(crossing.long(), dim=1)
    selected = torch.gather(
        candidates, 1, first_idx.view(-1, 1, 1).expand(-1, 1, 2)
    ).squeeze(1)
    return torch.where(has_crossing.view(-1, 1).bool(), selected, current_pos)


def _calculate_initial_yaw(rotations: torch.Tensor) -> torch.Tensor:
    """Habitat quaternion [x,y,z,w] → Genesis yaw (radians)."""
    x, y, z, w = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    dir_x_g =  -2 * w * y
    dir_y_g = -(y * y - w * w)
    return torch.atan2(dir_y_g, dir_x_g)


def _hab_to_genesis(pos):
    x, y, z = pos[0], pos[1], pos[2]
    return x, -z, y


def _genesis_to_hab(cam_pos_gen: torch.Tensor, camera_height: float) -> torch.Tensor:
    x = cam_pos_gen[:, 0]
    y = cam_pos_gen[:, 2] - camera_height
    z = -cam_pos_gen[:, 1]
    return torch.stack([x, y, z], dim=1)


def _update_camera(cam, cam_pos: torch.Tensor, cam_yaw: torch.Tensor):
    dir_x  = torch.cos(cam_yaw)
    dir_y  = torch.sin(cam_yaw)
    lookat = cam_pos.clone()
    lookat[:, 0] += dir_x
    lookat[:, 1] += dir_y
    lookat[:, 2]  = cam_pos[:, 2]
    up = torch.zeros_like(cam_pos)
    up[:, 2] = 1.0
    cam.set_pose(pos=cam_pos, lookat=lookat, up=up)


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
        # Format reward: added each step when policy outputs valid JSON (action 0-3).
        # action=4 (ACTION_PARSE_FAIL sentinel) means parse failed → no format reward.
        # Breaks cold-start: model learns to output valid JSON before learning navigation.
        self.format_reward_coef = float(getattr(cfg, "format_reward_coef", 0.0))
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

        # --- internal state (initialised in _init_genesis) ---
        self._gs_ready        = False
        self._cam_pos         = None     # (num_envs, 3) on gs.device
        self._cam_yaw         = None     # (num_envs,)   on gs.device
        self._current_tri_idx = None     # (num_envs,)   on gs.device
        self._active_mask     = None     # (num_envs,)   bool on gs.device
        self._goal_pos_t      = None     # (num_envs, 3) habitat coords on gs.device
        self._prev_geo_dist   = None     # (num_envs,)   float32 on gs.device
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
        self._exhausted = False
        self._dummy_obs_cache = None

        # navmesh geometry tensors — set in _load_scene
        self._tri_v0_2d = self._tri_v1_2d = self._tri_v2_2d = None
        self._tri_v0_3d = self._tri_v1_3d = self._tri_v2_3d = None

        # Load episode list and group by scene.
        # Genesis builds ONE scene per process — all envs share it.
        # Episodes must therefore come from a single scene_id.
        #
        # Scene assignment: deterministic round-robin over the sorted unique
        # scene list. With N workers and M scenes:
        #   - if N <= M: each worker pins to a distinct scene (covers N/M)
        #   - if N >  M: workers wrap around (some scenes get >1 worker)
        # This guarantees no two workers collide on the same scene when N <= M,
        # which is the common case (e.g. 5 workers × 10 scenes → cover 5).
        all_episodes = _load_episodes(self.episodes_file)
        unique_scenes = sorted({e["scene_id"] for e in all_episodes})
        scene_offset = int(getattr(cfg, "scene_offset", 0))
        pinned_scene = unique_scenes[(scene_offset + seed_offset) % len(unique_scenes)]
        scene_eps = [e for e in all_episodes if e["scene_id"] == pinned_scene]

        # Persistent RNG — used for initial shuffle and subsequent cyclic reshuffles.
        # seed_offset makes each worker use a different sequence.
        self._rng = np.random.default_rng(seed=42 + seed_offset)
        self._rng.shuffle(scene_eps)

        self._all_episodes    = scene_eps
        self._pinned_scene_id = pinned_scene
        # Active slots = min(num_envs, actual ep count) — no cycling needed.
        active_slot_count = min(num_envs, len(scene_eps))
        self._slot_active[:active_slot_count] = True
        self._active_slot_count = active_slot_count  # Genesis n_envs
        print(f"[GenArk] Worker {seed_offset}/{total_num_processes}: "
              f"pinned to scene '{pinned_scene}' "
              f"({len(self._all_episodes)} eps, "
              f"active_slots={active_slot_count}/{num_envs}, "
              f"{len(unique_scenes)} unique scenes total)", flush=True)

        # Group-level reset state.
        # When group_size>1, envs in the same group share one episode.
        # When ALL envs in a group finish, the group is reset together to the
        # next episode in the pool (preserving GRPO group semantics).
        # _next_ep_idx: index into _all_episodes for the next group reset.
        #   Initially = number of episodes already assigned (one per group).
        #   Incremented by 1 each time a group resets.
        # _group_done_counts: group_id → number of done envs waiting for group-mates.
        gs = max(self.group_size, 1)
        n_initial_groups = (active_slot_count + gs - 1) // gs  # ceil division
        self._next_ep_idx: int = n_initial_groups
        self._group_done_counts: dict[int, int] = {}
        self._episode_cycle: int = 0  # incremented on each cyclic reshuffle

        # Pick initial scene and episodes, then build Genesis scene
        self._current_scene_id = None
        self._scene_episode_pool: list[dict] = []
        self._init_genesis()
        self._assign_episodes_to_envs()

    # ------------------------------------------------------------------
    # Genesis initialisation  (called once — Genesis init is global)
    # ------------------------------------------------------------------

    def _init_genesis(self):
        if not gs._initialized:
            gs.init(backend=gs.cuda)
        self._renderer = gs.renderers.BatchRenderer(use_rasterizer=True)
        self._gs_scene = gs.Scene(
            renderer=self._renderer,
            show_viewer=False,
            vis_options=gs.options.VisOptions(
                ambient_light=(1.0, 1.0, 1.0),
                plane_reflection=False,
            ),
        )
        # Camera is created once; pose updated each step
        self._cam = self._gs_scene.add_camera(
            res=(self.cam_w, self.cam_h),
            fov=self.fov,
            GUI=False,
        )

    def _load_scene(self, scene_id: str):
        """Load mesh + navmesh for a new scene into the existing Genesis scene.

        scene_id is a Habitat-style path like "mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb".
        Mesh lookup order:
          1. <glb_cache_dir>/<scan_name>.glb   (pre-converted GLB, fastest)
          2. <scene_datasets>/<scan_name>/matterport_mesh/*.obj  (fallback)
        """
        scan_name = os.path.basename(os.path.dirname(scene_id))

        # 1. GLB cache (original genark uses /home/clk/workspace/genark/glb_cache)
        glb_cache_dir = getattr(self.cfg.init_params, "glb_cache_dir",
                                "/home/clk/workspace/genark/glb_cache")
        glb_path = os.path.join(glb_cache_dir, f"{scan_name}.glb")
        if os.path.exists(glb_path):
            mesh_path = glb_path
        else:
            # 2. Fallback: OBJ in matterport_mesh subdirectory
            obj_dir   = os.path.join(self.scene_datasets, scan_name, "matterport_mesh")
            obj_files = [f for f in os.listdir(obj_dir) if f.endswith(".obj")]
            if not obj_files:
                raise FileNotFoundError(
                    f"No mesh for '{scan_name}'. Tried:\n"
                    f"  {glb_path}\n  {obj_dir}/*.obj"
                )
            mesh_path = os.path.join(obj_dir, obj_files[0])

        if self._gs_ready:
            # Genesis does not support modifying a built scene.
            # All episodes are pinned to one scene_id, so this should never fire.
            raise RuntimeError(
                f"Attempted to switch Genesis scene from '{self._current_scene_id}' "
                f"to '{scene_id}'. Genesis does not support scene rebuilding. "
                "All episodes must share the same scene_id."
            )

        self._gs_scene.add_entity(morph=gs.morphs.Mesh(
            file=mesh_path, fixed=True, collision=False,
            # GLB converted from OBJ preserves original Habitat coords (Y-up).
            # Tell Genesis NOT to apply its automatic Y-UP → Z-UP rotation.
            file_meshes_are_zup=True,
        ))

        # Load navmesh
        navmesh_path = os.path.join(
            self.scene_datasets, scan_name, "navmesh.npz"
        )
        nm    = np.load(navmesh_path)
        device = gs.device
        V_hab  = torch.tensor(nm["verts"], device=device, dtype=torch.float32)
        F_idx  = torch.tensor(nm["faces"], device=device, dtype=torch.long)

        V_gen            = torch.zeros_like(V_hab)
        V_gen[:, 0]      =  V_hab[:, 0]
        V_gen[:, 1]      = -V_hab[:, 2]
        V_gen[:, 2]      =  V_hab[:, 1]

        self._tri_v0_3d  = V_gen[F_idx[:, 0]]
        self._tri_v1_3d  = V_gen[F_idx[:, 1]]
        self._tri_v2_3d  = V_gen[F_idx[:, 2]]
        self._tri_v0_2d  = self._tri_v0_3d[:, :2]
        self._tri_v1_2d  = self._tri_v1_3d[:, :2]
        self._tri_v2_2d  = self._tri_v2_3d[:, :2]

        # Build only active_slot_count envs — ghost slots don't need Genesis VRAM
        self._gs_scene.build(n_envs=self._active_slot_count, env_spacing=(0.0, 0.0))
        self._gs_ready = True
        self._current_scene_id = scene_id

    # ------------------------------------------------------------------
    # Episode assignment helpers
    # ------------------------------------------------------------------

    def _assign_episodes_to_envs(self, env_idx: Optional[list[int]] = None):
        """Assign episodes to slots. With group_size>1, every group_size consecutive
        slots share the same episode so GRPO can compare rewards within a group."""
        indices = list(range(self.num_envs)) if env_idx is None else env_idx
        for i in indices:
            if not self._slot_active[i]:
                continue  # ghost slot — no episode assigned
            ep_idx = i // self.group_size  # group_size envs share one episode
            if ep_idx < len(self._all_episodes):
                self._episodes[i] = self._all_episodes[ep_idx]
                self._instructions[i] = (
                    self._all_episodes[ep_idx]["instruction"]["instruction_text"]
                )
            else:
                self._slot_active[i] = False  # shouldn't happen, safety guard

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

        obs  = self._build_obs()
        return obs, {}

    def _init_agent_poses(self, env_idx: list[int]):
        device = gs.device
        N      = self.num_envs

        if self._cam_pos is None:
            self._cam_pos         = torch.zeros(N, 3, dtype=torch.float32, device=device)
            self._cam_yaw         = torch.zeros(N,    dtype=torch.float32, device=device)
            self._current_tri_idx = torch.zeros(N,    dtype=torch.long,    device=device)
            self._active_mask     = torch.ones(N,     dtype=torch.bool,    device=device)
            self._goal_pos_t      = torch.zeros(N, 3, dtype=torch.float32, device=device)
            self._prev_geo_dist        = torch.zeros(N, dtype=torch.float32, device=device)
            self._last_parse_ok        = torch.zeros(N, dtype=torch.bool,    device=device)
            self._dtg_decision_start   = torch.zeros(N, dtype=torch.float32, device=device)

        for i in env_idx:
            ep   = self._episodes[i]
            if ep is None:
                continue  # ghost slot — no episode assigned, skip
            sp   = ep["start_position"]
            gx, gy, gz = _hab_to_genesis(sp)
            self._cam_pos[i]   = torch.tensor(
                [gx, gy, gz + self.camera_height], dtype=torch.float32, device=device
            )
            rot = torch.tensor(ep["start_rotation"], dtype=torch.float32, device=device)
            self._cam_yaw[i]   = _calculate_initial_yaw(rot.unsqueeze(0))[0]
            goal_p             = ep["goals"][0]["position"]
            self._goal_pos_t[i] = torch.tensor(goal_p, dtype=torch.float32, device=device)
            self._active_mask[i] = True

        # Snap active envs to navmesh floor (skip ghost slots that have no pose)
        valid_idx = [i for i in env_idx if self._episodes[i] is not None]
        if not valid_idx:
            _update_camera(self._cam, self._cam_pos[:self._active_slot_count], self._cam_yaw[:self._active_slot_count])
            return

        idx_t  = torch.tensor(valid_idx, device=device)
        valid, new_z, tri_idx = _batch_find_floor(
            self._cam_pos[idx_t, :2],
            self._cam_pos[idx_t, 2] - self.camera_height,
            self._tri_v0_2d, self._tri_v1_2d, self._tri_v2_2d,
            self._tri_v0_3d, self._tri_v1_3d, self._tri_v2_3d,
            self.max_step_height,
        )
        self._cam_pos[idx_t, 2]      = new_z + self.camera_height
        self._current_tri_idx[idx_t] = tri_idx

        # Record initial distances for active slots only
        init_hab = _genesis_to_hab(self._cam_pos, self.camera_height)
        for i in valid_idx:
            d = torch.norm(init_hab[i] - self._goal_pos_t[i]).item()
            self._distances_lists[i] = [d]
            self._pred_path_lists[i] = [init_hab[i].cpu().numpy()]
            self._prev_geo_dist[i]   = d
            self._start_dtg[i]       = d
            self._dtg_decision_start[i] = d
            # ndtw_sr_delta mode: reset per-env previous nDTW/SR (initial path = single point → nDTW=1.0)
            self._prev_ndtw[i] = 1.0
            self._prev_sr[i]   = 0.0

        _update_camera(self._cam, self._cam_pos[:self._active_slot_count], self._cam_yaw[:self._active_slot_count])

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

        device = gs.device
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long, device=device)
        else:
            actions = actions.to(device=device, dtype=torch.long)

        # Detect parse-fail sentinel (ACTION_PARSE_FAIL=4) BEFORE clamping.
        # parse_ok_mask[i]=True means policy output valid JSON for env i.
        parse_ok_mask = (actions < 4)
        # Store for compute_decision_ndtw_reward() to use as format bonus gate.
        self._last_parse_ok = parse_ok_mask.to(dtype=torch.bool, device=self._last_parse_ok.device)

        # Record raw action (incl. parse-fail sentinel) into per-env history,
        # for episode-end termination-reason diagnostics.
        if self._ep_diag_enabled:
            raw_acts_cpu = actions.detach().cpu().tolist()
            active_undone_pre = self._slot_active & ~self._slot_done
            for _i, _a in enumerate(raw_acts_cpu):
                if active_undone_pre[_i]:
                    self._action_history[_i].append(int(_a))

        # Parse-fail policy (Lavira-style): keep navigating, do NOT terminate.
        # Sentinel ACTION_PARSE_FAIL=4 is clamped to MOVE_FORWARD so the episode
        # continues and the agent has a chance to self-recover. The negative reward
        # for parse_fail is applied later in the reward block. Done/ghost slots
        # still go to STOP (they must not move further).
        actions = actions.clone()
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

        # --- Rotation ---
        self._cam_yaw[(actions == self.TURN_LEFT)  & self._active_mask] += self.step_turn
        self._cam_yaw[(actions == self.TURN_RIGHT) & self._active_mask] -= self.step_turn

        # --- Forward movement with NavMesh collision ---
        fwd_m = (actions == self.MOVE_FORWARD) & self._active_mask
        if fwd_m.any():
            fwd_idx = fwd_m.nonzero(as_tuple=True)[0]
            dx = torch.cos(self._cam_yaw[fwd_idx])
            dy = torch.sin(self._cam_yaw[fwd_idx])
            desired = torch.stack([
                self._cam_pos[fwd_idx, 0] + dx * self.step_move,
                self._cam_pos[fwd_idx, 1] + dy * self.step_move,
            ], dim=1)
            cur_floor_z = self._cam_pos[fwd_idx, 2] - self.camera_height

            valid, new_z, new_tri = _batch_find_floor(
                desired, cur_floor_z,
                self._tri_v0_2d, self._tri_v1_2d, self._tri_v2_2d,
                self._tri_v0_3d, self._tri_v1_3d, self._tri_v2_3d,
                self.max_step_height,
            )

            if self.allow_sliding:
                invalid = ~valid
                if invalid.any():
                    inv_local = invalid.nonzero(as_tuple=True)[0]
                    slide_pos = _batch_get_sliding_position(
                        self._cam_pos[fwd_idx[inv_local], :2],
                        desired[inv_local],
                        self._current_tri_idx[fwd_idx[inv_local]],
                        self._tri_v0_2d, self._tri_v1_2d, self._tri_v2_2d,
                        self.agent_radius,
                    )
                    sv, sz, st = _batch_find_floor(
                        slide_pos, cur_floor_z[inv_local],
                        self._tri_v0_2d, self._tri_v1_2d, self._tri_v2_2d,
                        self._tri_v0_3d, self._tri_v1_3d, self._tri_v2_3d,
                        self.max_step_height,
                    )
                    if sv.any():
                        can_slide       = inv_local[sv]
                        desired[can_slide]  = slide_pos[sv]
                        new_z[can_slide]    = sz[sv]
                        new_tri[can_slide]  = st[sv]
                        valid[can_slide]    = True

            moved = fwd_idx[valid]
            self._cam_pos[moved, 0]     = desired[valid, 0]
            self._cam_pos[moved, 1]     = desired[valid, 1]
            self._cam_pos[moved, 2]     = new_z[valid] + self.camera_height
            self._current_tri_idx[moved] = new_tri[valid]

        # --- Render (active slots only; pad ghost slots with zeros) ---
        _update_camera(self._cam, self._cam_pos[:self._active_slot_count], self._cam_yaw[:self._active_slot_count])
        rgb_active, _, _, _ = self._cam.render(
            rgb=True, depth=False, segmentation=False, force_render=True
        )  # (active_slot_count, H, W, 3)
        if rgb_active.dtype == torch.uint8:
            rgb_float = rgb_active.float() / 255.0
        else:
            rgb_float = rgb_active
        rgb_active = torch.clamp(rgb_float * self.light_scale * 255.0, 0, 255).byte()
        # Pad ghost slots with zeros so output shape is always (num_envs, H, W, 3)
        if self._active_slot_count < self.num_envs:
            pad = torch.zeros(
                self.num_envs - self._active_slot_count,
                self.cam_h, self.cam_w, 3,
                dtype=torch.uint8, device=rgb_active.device,
            )
            rgb_batch = torch.cat([rgb_active, pad], dim=0)
        else:
            rgb_batch = rgb_active
        self._current_rgb = rgb_batch.cpu().numpy()  # (num_envs, H, W, 3) uint8

        # 4-dir: render 3 additional views (left/right/behind) and store
        self._render_extra_3dir()

        # --- Metrics ---
        # Metrics only for active slots (geometry tensors are num_envs-shaped but
        # only [:active_slot_count] have valid values from Genesis)
        N_act = self._active_slot_count
        curr_hab_act = _genesis_to_hab(self._cam_pos[:N_act], self.camera_height)
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
            # Decision-level DTG + decision-level nDTW + SR. No step-level geo_progress.
            # DTG reward is deferred to compute_decision_ndtw_reward() alongside nDTW.
            # Per-step: only SR delta (so episode termination is still rewarded on the correct step).
            reward = self._compute_ndtw_sr_delta_reward(
                curr_dist=curr_dist,
                newly_stopped=newly_stopped,
                N_act=N_act,
                include_ndtw=False,  # both DTG and nDTW deferred to decision flush
            )
        else:
            # Default: dense d2g progress + sparse success bonus
            reward = self._prev_geo_dist.to(curr_dist.device) - curr_dist
            just_succeeded = newly_stopped & (curr_dist < self.success_distance)
            reward[just_succeeded] += self.success_bonus
        self._prev_geo_dist = curr_dist.clone()

        # --- Parse-fail penalty (episode continues, malformed output costs) ---
        if self.parse_fail_penalty != 0.0:
            active_undone_t = torch.tensor(
                active_undone_np, dtype=torch.bool, device=reward.device,
            )
            pf_mask = (~self._last_parse_ok.to(reward.device)) & active_undone_t
            if pf_mask.any():
                reward[pf_mask] += self.parse_fail_penalty

        # --- Wrong-stop penalty (STOP issued when still far from goal) ---
        # Penalty scales with DTG: penalty = -wrong_stop_penalty * (DTG / success_distance)
        # e.g. DTG=10m, success=5m → -2.0; DTG=5m → -1.0; DTG=2m → -0.4
        if self.wrong_stop_penalty != 0.0:
            ws_mask = newly_stopped & (curr_dist.to(newly_stopped.device) > self.success_distance)
            if ws_mask.any():
                dist_ratio = curr_dist.to(reward.device) / max(self.success_distance, 1e-6)
                scaled_penalty = -abs(self.wrong_stop_penalty) * dist_ratio
                reward[ws_mask.to(reward.device)] += scaled_penalty[ws_mask.to(reward.device)]

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
            if self.group_size > 1:
                self._maybe_reset_complete_groups(newly_done_idx)
                obs = self._build_obs()  # rebuild after potential pose reset
        else:
            obs = self._build_obs()

        # Check full exhaustion
        if not self._exhausted and not np.any(self._slot_active & ~self._slot_done):
            self._exhausted = True
            self._dump_per_scene_metrics()
            print(f"[GenArk] Worker exhausted: scene='{self._pinned_scene_id}' "
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
        device = gs.device

        # RGB from last render (or zeros on first reset before render)
        if self._current_rgb is None:
            # Initial render after reset — active slots only, pad ghost slots
            _update_camera(self._cam, self._cam_pos[:self._active_slot_count], self._cam_yaw[:self._active_slot_count])
            rgb_raw, _, _, _ = self._cam.render(
                rgb=True, depth=False, segmentation=False, force_render=True
            )
            if rgb_raw.dtype == torch.uint8:
                rgb_float = rgb_raw.float() / 255.0
            else:
                rgb_float = rgb_raw
            rgb_raw = torch.clamp(rgb_float * self.light_scale * 255.0, 0, 255).byte()
            if self._active_slot_count < self.num_envs:
                pad = torch.zeros(
                    self.num_envs - self._active_slot_count,
                    self.cam_h, self.cam_w, 3,
                    dtype=torch.uint8, device=rgb_raw.device,
                )
                rgb_raw = torch.cat([rgb_raw, pad], dim=0)
            self._current_rgb = rgb_raw.cpu().numpy()
            # Lazy-init render: also produce 4-dir extras if enabled
            self._render_extra_3dir()

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
        elapsed_t = torch.from_numpy(
            self._elapsed_steps.astype(np.float32)
        ).unsqueeze(1).to(device)  # (N,1) float — states format

        # 4-dir extras: (N, 3, H, W, 3) uint8 → put into extra_view_images
        # Order: [left, right, behind] (front is in main_images)
        if self._current_rgb_extras is not None:
            extra_t = torch.from_numpy(
                np.ascontiguousarray(self._current_rgb_extras)
            ).to(device)
        else:
            extra_t = None

        return {
            "main_images":     rgb_hwc,           # (N, H, W, 3) uint8 — front view
            "states":          elapsed_t,          # (N, 1) float — carries elapsed_steps
            "task_descriptions": list(self._instructions),  # list[str], len=N
            # Kept for any direct callers that bypass prepare_observations
            "wrist_images":    None,
            "extra_view_images": extra_t,         # (N, 3, H, W, 3) uint8 [L,R,B] or None
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
            self._episode_log.append({
                "episode_id": ep_id,
                "scene_id":   self._pinned_scene_id,
                **{k: float(v) for k, v in m.items()},
            })
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
                acts = self._action_history[i]
                last = acts[-1] if acts else None
                n_pf = sum(1 for a in acts if a >= 4)
                # Classify termination cause. Parse-fail no longer terminates the
                # episode (it is clamped to MOVE_FORWARD); only real STOP or
                # max_episode_steps timeout end an episode.
                if last == self.STOP:
                    cause = "stop"
                else:
                    cause = "timeout"
                # Classify success type for post-hoc quality analysis.
                success_val = int(m.get('success', 0))
                start_dtg = self._start_dtg[i]
                if success_val:
                    if start_dtg < self.success_distance:
                        success_type = "lucky_start"   # started within threshold
                    elif n_pf > 0:
                        success_type = "recovered"     # success despite parse_fails
                    else:
                        success_type = "clean"         # success with all valid actions
                else:
                    success_type = "wrong_stop" if cause == "stop" else "no_stop"
                # Compact action string: STOP=S FWD=F LEFT=L RIGHT=R PARSE=P
                _LET = {0: "S", 1: "F", 2: "L", 3: "R", 4: "P"}
                seq = "".join(_LET.get(min(a, 4), "?") for a in acts)
                if len(seq) > 60:
                    seq = seq[:30] + "..." + seq[-15:]
                # Compact DTG curve: 5 sample points
                dists = self._distances_lists[i]
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

    def _maybe_reset_complete_groups(self, newly_done_idx: list[int]) -> None:
        """Group-level reset: when all envs in a GRPO group finish their episode,
        reset the entire group to the next episode from the pool.

        Each group of `group_size` envs shares one episode so GRPO can compare
        rewards within the group. Resetting them together to a NEW episode
        (same scene, different task) preserves this invariant across resets.

        If the episode pool is exhausted, the group stays dormant (existing behavior).
        """
        for env_i in newly_done_idx:
            group_id = env_i // self.group_size
            self._group_done_counts[group_id] = (
                self._group_done_counts.get(group_id, 0) + 1
            )

            # Count active envs in this group (may be < group_size at scene edge)
            group_start = group_id * self.group_size
            group_envs = [
                j for j in range(group_start, group_start + self.group_size)
                if j < self.num_envs and self._slot_active[j]
            ]
            n_active_in_group = len(group_envs)

            if self._group_done_counts[group_id] < n_active_in_group:
                continue  # still waiting for group-mates

            # All active envs in the group are done
            del self._group_done_counts[group_id]

            if self._next_ep_idx >= len(self._all_episodes):
                if self.cyclic_episode_sampling:
                    # Reshuffle and restart the pool (train mode).
                    # Avoid immediately repeating the episode the group just ran.
                    last_ep_id = (self._episodes[group_envs[0]] or {}).get(
                        "episode_id", None
                    )
                    self._rng.shuffle(self._all_episodes)
                    if (
                        last_ep_id is not None
                        and len(self._all_episodes) > 1
                        and self._all_episodes[0].get("episode_id") == last_ep_id
                    ):
                        self._all_episodes = (
                            self._all_episodes[1:] + self._all_episodes[:1]
                        )
                    self._next_ep_idx = 0
                    self._episode_cycle += 1
                    # Print per-scene summary on each cycle so we can compare scenes.
                    summ = self._summarize_episode_log()
                    print(
                        f"[GenArk][episode-cycle] scene={self._pinned_scene_id} "
                        f"cycle={self._episode_cycle} "
                        f"reshuffled {len(self._all_episodes)} episodes | "
                        f"sr={summ.get('success', 0):.3f} "
                        f"spl={summ.get('spl', 0):.3f} "
                        f"ndtw={summ.get('ndtw', 0):.3f} "
                        f"n={summ.get('num_episodes', 0)}",
                        flush=True,
                    )
                else:
                    # Single-pass mode (eval) — group enters dormant permanently.
                    print(
                        f"[GenArk][group-reset] group={group_id} pool exhausted "
                        f"(next_ep_idx={self._next_ep_idx}, "
                        f"pool_size={len(self._all_episodes)}); entering dormant.",
                        flush=True,
                    )
                    continue

            # Assign the next pooled episode to all envs in the group
            next_ep = self._all_episodes[self._next_ep_idx]
            ep_id = next_ep.get("episode_id", self._next_ep_idx)
            self._next_ep_idx += 1

            for j in group_envs:
                self._episodes[j]     = next_ep
                self._instructions[j] = next_ep["instruction"]["instruction_text"]
                self._slot_done[j]    = False   # clear done → no longer dormant
                self._elapsed_steps[j] = 0
                self._stop_called[j]  = False
                self._action_history[j] = []
                self._pred_path_lists[j] = []
                self._distances_lists[j] = []

            # Re-initialise agent poses and metrics for the reset group
            self._init_agent_poses(group_envs)

            print(
                f"[GenArk][group-reset] group={group_id} → ep={ep_id} "
                f"envs={group_envs} "
                f"(pool remaining: {len(self._all_episodes) - self._next_ep_idx})",
                flush=True,
            )

    def _dormant_step(self) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Cheap no-op step issued after the worker's episode pool is exhausted.

        Returns same-shape tensors as a real step but skips Genesis render
        and inference-relevant work. info["episode"] is intentionally empty
        so RLinf's eval aggregator ignores these steps.
        """
        N = self.num_envs
        # Build / reuse a tiny dummy obs in the same key layout as _build_obs.
        if self._dummy_obs_cache is None:
            dummy_rgb = torch.zeros((N, self.cam_h, self.cam_w, 3),
                                    dtype=torch.uint8, device=gs.device)
            dummy_states = torch.zeros((N, 1), dtype=torch.float32, device=gs.device)
            # extra_view_images: (N, 3, H, W, 3) uint8 when enable_4dir_render, else None.
            # wrist_images: GenArk has no wrist camera → always None.
            dummy_extra = (
                torch.zeros((N, 3, self.cam_h, self.cam_w, 3),
                            dtype=torch.uint8, device=gs.device)
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
            scan = os.path.basename(os.path.dirname(self._pinned_scene_id))
            out_path = os.path.join(base, f"per_scene_{scan}.json")
            payload = {
                "scene_id":  self._pinned_scene_id,
                "scan_name": scan,
                "n_episodes": len(self._episode_log),
                "episodes":  self._episode_log,
                "summary":   self._summarize_episode_log(),
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[GenArk] Per-scene metrics saved → {out_path}", flush=True)
        except Exception as e:
            print(f"[GenArk] WARNING: could not dump per-scene metrics: {e}",
                  flush=True)

    def _summarize_episode_log(self) -> dict:
        if not self._episode_log:
            return {}
        keys = ["success", "spl", "ndtw", "sdtw",
                "distance_to_goal", "path_length", "steps_taken"]
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

    def _render_extra_3dir(self) -> None:
        """
        Render 3 additional camera angles (left=+90°, right=-90°, behind=180°)
        at current cam_pos, store as `self._current_rgb_extras`
        shape (num_envs, 3, H, W, 3) uint8 numpy.

        Front view is rendered by the caller; we only render the 3 extras here.
        Camera pose is restored to front yaw afterward.

        Cost: 3× single-direction render. Disabled unless `enable_4dir_render`.
        """
        if not self.enable_4dir_render:
            self._current_rgb_extras = None
            return

        N_act = self._active_slot_count
        if N_act == 0:
            self._current_rgb_extras = None
            return

        # Yaw deltas for [left, right, behind] (CCW positive, matches TURN_LEFT)
        deltas = [math.radians(90.0), math.radians(-90.0), math.radians(180.0)]

        extras_np = []
        front_yaw = self._cam_yaw[:N_act]
        for d_yaw in deltas:
            rotated_yaw = front_yaw + d_yaw
            _update_camera(self._cam, self._cam_pos[:N_act], rotated_yaw)
            rgb_raw, _, _, _ = self._cam.render(
                rgb=True, depth=False, segmentation=False, force_render=True
            )
            # Apply same light_scale + uint8 conversion as the front render
            if rgb_raw.dtype == torch.uint8:
                rgb_float = rgb_raw.float() / 255.0
            else:
                rgb_float = rgb_raw
            rgb_raw = torch.clamp(rgb_float * self.light_scale * 255.0, 0, 255).byte()
            # Pad ghost slots
            if N_act < self.num_envs:
                pad = torch.zeros(
                    self.num_envs - N_act,
                    self.cam_h, self.cam_w, 3,
                    dtype=torch.uint8, device=rgb_raw.device,
                )
                rgb_raw = torch.cat([rgb_raw, pad], dim=0)
            extras_np.append(rgb_raw.cpu().numpy())  # (N, H, W, 3)

        # Restore front camera pose
        _update_camera(self._cam, self._cam_pos[:N_act], front_yaw)

        # (3, N, H, W, 3) → (N, 3, H, W, 3)
        self._current_rgb_extras = np.stack(extras_np, axis=1)

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

            # SR delta — binary 0→1 at success step (always computed per env step)
            curr_sr = 1.0 if (stopped_np[i] and curr_dist_np[i] < self.success_distance) else 0.0
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

            reward_np[i] = self.ndtw_coef * ndtw_delta + self.sr_coef * sr_delta

        return torch.from_numpy(reward_np).to(curr_dist.device)

    def compute_decision_ndtw_reward(self, env_indices: list[int]) -> torch.Tensor:
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
        for i in env_indices:
            if not self._active_mask[i]:
                continue
            ep = self._episodes[i]
            pred = self._pred_path_lists[i]

            # --- nDTW delta ---
            curr_ndtw = 1.0
            if ep is not None and len(pred) >= 2:
                gt = np.array(ep.get("reference_path", [[0, 0, 0]]), dtype=float)
                pred_arr = np.array(pred)
                try:
                    dtw_d = fastdtw(pred_arr, gt, dist=euclidean)[0]
                    curr_ndtw = float(np.exp(-dtw_d / (len(gt) * self.success_distance)))
                except Exception:
                    curr_ndtw = self._prev_ndtw[i]
            ndtw_r = self.ndtw_coef * (curr_ndtw - self._prev_ndtw[i])
            self._prev_ndtw[i] = curr_ndtw

            # --- decision-level DTG reward (decision_nav mode only) ---
            dtg_r = 0.0
            if self.reward_mode == "decision_nav":
                dists = self._distances_lists[i]
                curr_dtg = float(dists[-1]) if dists else float(self._dtg_decision_start[i])
                dtg_before = float(self._dtg_decision_start[i])
                delta = dtg_before - curr_dtg  # positive = made progress toward goal
                scaled = delta * self.decision_dtg_coef if delta > 0 else 0.5 * delta * self.decision_dtg_coef
                dtg_r = float(np.clip(scaled, -self.decision_dtg_clip, self.decision_dtg_clip))
                self._dtg_decision_start[i] = curr_dtg  # update for next decision

            # --- format reward (one bonus per valid LLM decision) ---
            fmt_r = (
                self.format_reward_coef
                if (self.format_reward_coef > 0 and bool(self._last_parse_ok[i]))
                else 0.0
            )
            reward_np[i] = ndtw_r + dtg_r + fmt_r
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

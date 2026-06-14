"""
P3: occupancy-only map + FMM planner for LaViRA-style navigation.

Self-contained — no LaViRA or Habitat imports.
Ported from:
  vlnce_baselines/map/mapping.py       (occupancy update logic)
  vlnce_baselines/models/fmm_planner.py
  vlnce_baselines/models/Policy.py:_get_action
  vlnce_baselines/utils/map_utils.py

Coordinate conventions (same as lavira_depth_utils.py):
  Habitat world: x=East (+), z=South (+)
  Genesis:       x=East, y=North, z=up; yaw=0 faces East, CCW positive
  Camera frame:  X=right, Y=forward (depth), Z=up

Map layout: GRID_SIZE × GRID_SIZE cells, RES_CM cm/cell
  row=0  → North edge;  row increases South  (row = CENTER - dz * CELLS_PER_M)
  col=0  → West  edge;  col increases East   (col = CENTER + dx * CELLS_PER_M)
  Agent at episode-start origin → cell (CENTER, CENTER)
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import skfmm
from numpy import ma
from skimage.morphology import closing, disk

# ---------------------------------------------------------------------------
# Map geometry (matches LaViRA MAP_SIZE_CM=2400, MAP_RESOLUTION=5)
# ---------------------------------------------------------------------------
GRID_SIZE   = 480       # cells
RES_CM      = 5         # cm per cell
CELLS_PER_M = 100.0 / RES_CM   # 20.0 cells / metre
MAP_CENTER  = GRID_SIZE // 2   # 240

# Obstacle height filter (metres above ground)
OBS_MIN_M = 0.20    # floor below this → not obstacle
OBS_MAX_M = 1.95    # ceiling above → not obstacle

# FMM thresholds (from LaViRA r2r.yaml)
FMM_STEP_SIZE       = 5    # local window radius in cells
FMM_WP_THRESH_M     = 2.0  # waypoint stop threshold (metres)
FMM_GOAL_THRESH_M   = 1.0  # goal-arrival threshold (metres)
FMM_WP_THRESH_CELLS   = FMM_WP_THRESH_M   * CELLS_PER_M   # 40
FMM_GOAL_THRESH_CELLS = FMM_GOAL_THRESH_M * CELLS_PER_M   # 20

TURN_ANGLE_DEG = 30.0   # GenArk step_turn (also LaViRA default)

# Primitive step sizes — must match env config
STEP_MOVE_M   = 0.25
STEP_TURN_RAD = math.radians(30.0)

# Depth sub-sampling: every Nth pixel
DEPTH_SUBSAMPLE = 4

# Minimum explored cells before trusting FMM (avoids planning in empty map)
MIN_TRAVERSIBLE_CELLS = 50


# ---------------------------------------------------------------------------
# FMM helper functions (ported from map_utils.py)
# ---------------------------------------------------------------------------

def _get_mask(sx: float, sy: float, step_size: int = FMM_STEP_SIZE) -> np.ndarray:
    """Annular boolean mask of radius=step_size around sub-pixel offset (sx, sy)."""
    size = step_size * 2 + 1
    mask = np.zeros((size, size), dtype=np.float32)
    cx = size // 2 + sx
    cy = size // 2 + sy
    for i in range(size):
        for j in range(size):
            d2 = ((i + 0.5) - cx) ** 2 + ((j + 0.5) - cy) ** 2
            if (step_size - 1) ** 2 < d2 <= step_size ** 2:
                mask[i, j] = 1.0
    mask[size // 2, size // 2] = 1.0
    return mask


def _get_dist(sx: float, sy: float, step_size: int = FMM_STEP_SIZE) -> np.ndarray:
    """Distance weights within step_size circle (used to break FMM ties)."""
    size = step_size * 2 + 1
    mask = np.full((size, size), 1e-10, dtype=np.float32)
    cx = size // 2 + sx
    cy = size // 2 + sy
    for i in range(size):
        for j in range(size):
            d2 = ((i + 0.5) - cx) ** 2 + ((j + 0.5) - cy) ** 2
            if d2 <= step_size ** 2:
                mask[i, j] = max(5.0, d2 ** 0.5)
    return mask


def _angle_to_vector(angle_deg: float) -> np.ndarray:
    r = math.radians(angle_deg)
    return np.array([math.cos(r), math.sin(r)], dtype=np.float64)


def _angle_and_direction(
    heading_vec: np.ndarray,
    waypoint_vec: np.ndarray,
    turn_angle_deg: float,
) -> Tuple[float, int]:
    """
    Returns (angle_degrees, action).
    action: 1=forward, 2=turn_left, 3=turn_right  (GenArk codes)
    """
    unit_h = heading_vec / (np.linalg.norm(heading_vec) + 1e-8)
    unit_w = waypoint_vec / (np.linalg.norm(waypoint_vec) + 1e-8)
    cross = float(np.cross(unit_h, unit_w))
    dot   = float(np.dot(unit_h, unit_w))
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    half = turn_angle_deg / 2.0
    if cross > 0 and angle_deg >= (half + 0.01):
        return angle_deg, 3   # right
    if cross < 0 and angle_deg >= half:
        return angle_deg, 2   # left
    return angle_deg, 1       # forward


def _get_nearest_nonzero(arr: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Return index of nearest non-zero cell to start (in row-col space)."""
    nz = np.argwhere(arr != 0)
    if len(nz) == 0:
        return start.astype(int)
    dists = np.linalg.norm(nz - start, axis=1)
    return nz[np.argmin(dists)]


# ---------------------------------------------------------------------------
# FMMPlanner — ported from LaViRA fmm_planner.py
# ---------------------------------------------------------------------------

class FMMPlanner:
    """
    Fast Marching Method path planner operating on a 2-D traversibility map.

    Args:
        traversible: (GRID_SIZE, GRID_SIZE) float array; 1 = free, 0 = obstacle.
        step_size:   Local window radius in cells.
        wp_thresh_cells:   Stop if FMM distance (cells) < this value.
        goal_thresh_cells: Treat as arrived if distance < this value.
    """

    def __init__(
        self,
        traversible: np.ndarray,
        step_size: int = FMM_STEP_SIZE,
        wp_thresh_cells: float = FMM_WP_THRESH_CELLS,
        goal_thresh_cells: float = FMM_GOAL_THRESH_CELLS,
    ) -> None:
        self.traversible = traversible.copy().astype(np.float32)
        self.du = step_size
        self.wp_thresh   = wp_thresh_cells
        self.goal_thresh = goal_thresh_cells
        self.fmm_dist: Optional[np.ndarray] = None

    def set_goal(self, goal: np.ndarray) -> None:
        trav_ma = ma.masked_values(self.traversible * 1, 0)
        gx, gy = int(goal[0]), int(goal[1])
        trav_ma[gx, gy] = 0
        dd = skfmm.distance(trav_ma, dx=1)
        dd = ma.filled(dd, float(np.max(dd)) + 1)
        self.fmm_dist = dd

    def get_short_term_goal(
        self,
        position: np.ndarray,
        fixed_destination: Optional[np.ndarray] = None,
    ) -> Tuple[int, int, bool]:
        """
        Returns (goal_row, goal_col, stop_flag).
        stop_flag=True  → agent is close enough to the goal, or can't make progress.
        """
        pad = self.du
        dist = np.pad(
            self.fmm_dist, pad, mode="constant",
            constant_values=float(np.max(self.fmm_dist)),
        )
        x, y   = int(position[0]), int(position[1])
        dx, dy = position[0] - x, position[1] - y
        mask      = _get_mask(dx, dy, self.du)
        xp, yp = x + pad, y + pad
        subset = dist[xp - self.du : xp + self.du + 1,
                      yp - self.du : yp + self.du + 1].copy()
        if subset.shape != mask.shape:
            return x, y, True
        subset *= mask
        subset += (1.0 - mask) * 1e5

        # Mirror LaViRA logic: when fixed_destination provided, ONLY goal_thresh
        # applies (wp_thresh is overridden by the else branch in the original).
        # Without fixed_destination, use wp_thresh for open exploration.
        if fixed_destination is not None:
            stop = subset[self.du, self.du] < self.goal_thresh
        else:
            stop = subset[self.du, self.du] < self.wp_thresh
        if stop:
            return x, y, True

        sx, sy = np.unravel_index(np.argmin(subset), subset.shape)
        return x + (sx - self.du), y + (sy - self.du), False


# ---------------------------------------------------------------------------
# Occupancy Map — one instance per environment slot
# ---------------------------------------------------------------------------

class OccupancyMap:
    """
    Per-episode accumulating occupancy map.

    Usage:
        occ = OccupancyMap()
        occ.reset(hab_x0, hab_z0)          # called at episode start
        occ.update(depth_hw, hab_x, hab_z, gen_yaw_rad)  # every primitive step
        acts = occ.get_fmm_action_seq(gx, gz, hab_x, hab_z, gen_yaw_rad)
    """

    def __init__(
        self,
        hfov_deg: float = 105.0,
        render_w: int   = 640,
        render_h: int   = 480,
        camera_height: float = 1.25,
    ) -> None:
        self._cam_h = camera_height
        self._build_intrinsics(hfov_deg, render_w, render_h)
        self._obstacle = np.zeros((GRID_SIZE, GRID_SIZE), np.float32)
        self._explored = np.zeros((GRID_SIZE, GRID_SIZE), np.float32)
        self._origin_x = 0.0
        self._origin_z = 0.0
        self._initialized = False

    def _build_intrinsics(self, hfov_deg: float, w: int, h: int) -> None:
        fx = w / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
        vfov = 2.0 * math.atan(h / w * math.tan(math.radians(hfov_deg / 2.0)))
        fy = h / (2.0 * math.tan(vfov / 2.0))
        self._fx, self._fy = fx, fy
        self._cx, self._cy = w / 2.0, h / 2.0

    # ---- coordinate helpers ------------------------------------------------

    def _world_to_map(
        self, world_x: np.ndarray, world_z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Habitat world arrays → integer (row, col) arrays."""
        col = (MAP_CENTER + (world_x - self._origin_x) * CELLS_PER_M).astype(np.int32)
        row = (MAP_CENTER - (world_z - self._origin_z) * CELLS_PER_M).astype(np.int32)
        return row, col

    def _single_world_to_map(self, hab_x: float, hab_z: float) -> Tuple[int, int]:
        col = int(MAP_CENTER + (hab_x - self._origin_x) * CELLS_PER_M)
        row = int(MAP_CENTER - (hab_z - self._origin_z) * CELLS_PER_M)
        return row, col

    # ---- lifecycle ---------------------------------------------------------

    def reset(self, hab_x: float = 0.0, hab_z: float = 0.0) -> None:
        """Clear map and set episode origin. Call at episode start."""
        self._obstacle[:] = 0.0
        self._explored[:] = 0.0
        self._origin_x = hab_x
        self._origin_z = hab_z
        self._initialized = True

    # ---- map update from depth observation ---------------------------------

    def update(
        self,
        depth_hw: np.ndarray,
        hab_x: float,
        hab_z: float,
        gen_yaw_rad: float,
    ) -> None:
        """
        Back-project depth_hw into world coords and accumulate obstacle/explored.
        Skips update if reset() has not been called (episode origin unknown).
        """
        if not self._initialized:
            return

        H, W = depth_hw.shape
        s = DEPTH_SUBSAMPLE
        v_idx = np.arange(0, H, s)
        u_idx = np.arange(0, W, s)
        v_g, u_g = np.meshgrid(v_idx, u_idx, indexing="ij")
        d = depth_hw[v_g, u_g].astype(np.float64)

        valid = (d > 0.1) & (d < 15.0) & np.isfinite(d)

        # Camera coords: X=right, Y=forward (depth), Z=up
        cam_x    = (u_g   - self._cx) * d / self._fx
        cam_y    = d
        cam_z_up = (self._cy - v_g) * d / self._fy   # positive = up
        height   = cam_z_up + self._cam_h             # metres above ground

        # World XZ (Habitat) — same formula as lavira_depth_utils.project_bbox_to_world
        heading = -gen_yaw_rad
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        world_x = hab_x + cam_y * cos_h + cam_x * sin_h
        world_z = hab_z + cam_y * sin_h - cam_x * cos_h

        row, col = self._world_to_map(world_x, world_z)
        in_bounds = (
            (row >= 0) & (row < GRID_SIZE) & (col >= 0) & (col < GRID_SIZE)
        )

        # Explored: any valid, in-bounds depth point
        exp_mask = valid & in_bounds
        if exp_mask.any():
            idx = row[exp_mask].ravel() * GRID_SIZE + col[exp_mask].ravel()
            cnt = np.bincount(idx, minlength=GRID_SIZE * GRID_SIZE)
            self._explored = np.clip(
                self._explored + cnt.reshape(GRID_SIZE, GRID_SIZE).astype(np.float32) * 0.05,
                0.0, 1.0,
            )

        # Obstacle: height in obstacle window
        obs_mask = valid & in_bounds & (height > OBS_MIN_M) & (height < OBS_MAX_M)
        if obs_mask.any():
            idx_o = row[obs_mask].ravel() * GRID_SIZE + col[obs_mask].ravel()
            cnt_o = np.bincount(idx_o, minlength=GRID_SIZE * GRID_SIZE)
            self._obstacle += cnt_o.reshape(GRID_SIZE, GRID_SIZE).astype(np.float32)

        # Mark explored around the agent:
        # - 30-cell square (1.5m) to bridge the camera blind spot directly in front
        # - also mark a forward-facing strip to cover the blind zone between agent
        #   and the nearest visible floor pixel (~1.5m ahead at camera height 1.25m)
        ar, ac = self._single_world_to_map(hab_x, hab_z)
        ar = max(0, min(GRID_SIZE - 1, ar))
        ac = max(0, min(GRID_SIZE - 1, ac))
        RAD = 30   # 30 cells = 1.5 m — covers single-camera floor blind spot
        r0, r1 = max(0, ar - RAD), min(GRID_SIZE, ar + RAD + 1)
        c0, c1 = max(0, ac - RAD), min(GRID_SIZE, ac + RAD + 1)
        self._explored[r0:r1, c0:c1] = np.maximum(self._explored[r0:r1, c0:c1], 0.3)

    # ---- traversibility ----------------------------------------------------

    def get_traversible(self) -> np.ndarray:
        """
        Traversible = explored cells that are not strongly marked as obstacle.
        Obstacles dilated by ~20cm (4 cells) to account for agent radius.
        """
        obs_binary = (self._obstacle > 2.0).astype(np.float32)   # ≥3 hits
        obs_dilated = closing(obs_binary, footprint=disk(4)).astype(np.float32)
        traversible = (self._explored > 0.1).astype(np.float32)
        traversible[obs_dilated > 0] = 0.0
        return traversible

    # ---- FMM action planning -----------------------------------------------

    def get_fmm_action_seq(
        self,
        world_goal_x: float,
        world_goal_z: float,
        hab_x: float,
        hab_z: float,
        gen_yaw_rad: float,
        n_steps: int = 3,
    ) -> List[int]:
        """
        Plan up to n_steps primitive actions toward (world_goal_x, world_goal_z).

        Returns list of action ints (GenArk codes):
            0=STOP, 1=FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT

        Falls back to [FORWARD] if map is too sparse or planning fails.
        """
        if not self._initialized:
            return [1]

        traversible = self.get_traversible()
        if float(traversible.sum()) < MIN_TRAVERSIBLE_CELLS:
            return [1]

        # Goal in map coords
        goal_r, goal_c = self._single_world_to_map(world_goal_x, world_goal_z)
        goal_r = max(0, min(GRID_SIZE - 1, goal_r))
        goal_c = max(0, min(GRID_SIZE - 1, goal_c))
        goal_arr = np.array([goal_r, goal_c])

        # Snap goal to nearest traversible cell if needed
        if traversible[goal_r, goal_c] == 0:
            goal_arr = _get_nearest_nonzero(traversible, goal_arr)

        try:
            planner = FMMPlanner(traversible)
            planner.set_goal(goal_arr)
        except Exception as exc:
            print(f"[P3][FMM] set_goal failed: {exc}")
            return [1]

        # Connectivity check: if agent is in a different connected component than
        # the goal, fmm_dist at agent ≈ max(fmm_dist)+1 (filled masked value).
        # In that case fall back to a heading-based action toward the world goal.
        agent_r, agent_c = self._single_world_to_map(hab_x, hab_z)
        agent_r = max(0, min(GRID_SIZE - 1, agent_r))
        agent_c = max(0, min(GRID_SIZE - 1, agent_c))
        max_dist = float(np.max(planner.fmm_dist))
        if planner.fmm_dist[agent_r, agent_c] >= max_dist - 1.0:
            # Agent is disconnected from goal → use heading-based direction action
            return _heading_toward_world_goal(world_goal_x, world_goal_z,
                                              hab_x, hab_z, gen_yaw_rad)

        # Open-loop lookahead: simulate n_steps without new depth
        actions: List[int] = []
        sim_x, sim_z, sim_yaw = hab_x, hab_z, gen_yaw_rad
        for _ in range(n_steps):
            act = self._single_fmm_step(planner, sim_x, sim_z, sim_yaw, goal_arr)
            if act == 0:
                # FMM says goal reached — do NOT emit STOP into the action queue.
                # STOP is VLM's decision; planner just stops extending the sequence.
                break
            actions.append(act)
            sim_x, sim_z, sim_yaw = _simulate_step(act, sim_x, sim_z, sim_yaw)

        return actions if actions else [1]

    def _single_fmm_step(
        self,
        planner: FMMPlanner,
        hab_x: float,
        hab_z: float,
        gen_yaw_rad: float,
        goal_arr: np.ndarray,
    ) -> int:
        """One FMM step: pose → short-term waypoint → heading → primitive action."""
        ar, ac = self._single_world_to_map(hab_x, hab_z)
        ar = max(0, min(GRID_SIZE - 1, ar))
        ac = max(0, min(GRID_SIZE - 1, ac))
        position = np.array([float(ar), float(ac)])

        stg_r, stg_c, stop = planner.get_short_term_goal(
            position, fixed_destination=goal_arr
        )
        if stop:
            return 0  # STOP

        wp_vec = np.array([float(stg_r) - float(ar), float(stg_c) - float(ac)])
        if np.linalg.norm(wp_vec) < 1e-4:
            return 1  # essentially at waypoint → keep moving forward

        # LaViRA heading convention for _get_action:
        #   lavira_heading_deg = degrees(gen_yaw_rad)
        #   heading_input = -lavira_heading_deg
        #   heading_vector = angle_to_vector(heading_input)  →  then rotate 90° CCW
        h_input_deg = -math.degrees(gen_yaw_rad)
        hvec = _angle_to_vector(h_input_deg)
        rot  = np.array([[0.0, -1.0], [1.0, 0.0]])
        heading_vec = rot @ hvec

        _, action = _angle_and_direction(heading_vec, wp_vec, TURN_ANGLE_DEG)
        return action


# ---------------------------------------------------------------------------
# Primitive-step simulator (open-loop lookahead)
# ---------------------------------------------------------------------------

def _heading_toward_world_goal(
    world_goal_x: float,
    world_goal_z: float,
    hab_x: float,
    hab_z: float,
    gen_yaw_rad: float,
    n_steps: int = 3,
) -> List[int]:
    """
    Fallback when FMM map is disconnected: compute the single primitive action
    (turn or forward) that faces the agent toward the world goal using pure
    heading geometry, then repeat n_steps times (exploration behaviour).
    """
    # Direction to goal in Habitat XZ
    dx = world_goal_x - hab_x
    dz = world_goal_z - hab_z
    if abs(dx) < 1e-4 and abs(dz) < 1e-4:
        return [0]  # already at goal

    # Bearing of goal in Habitat coords: East=0, CCW positive
    # hab_z increases South, so "forward to North" = dz<0
    goal_bearing_rad = math.atan2(-dz, dx)   # same convention as gen_yaw_rad

    heading = gen_yaw_rad
    # Angle difference (heading → goal bearing), normalised to [-π, π]
    diff = goal_bearing_rad - heading
    diff = (diff + math.pi) % (2.0 * math.pi) - math.pi

    half_turn = STEP_TURN_RAD / 2.0
    if abs(diff) <= half_turn:
        return [1] * n_steps        # facing goal → move forward
    elif diff > 0:
        # Need to turn CCW (LEFT in Genesis)
        n_turns = max(1, round(abs(diff) / STEP_TURN_RAD))
        return [2] * min(n_turns, n_steps)
    else:
        # Need to turn CW (RIGHT in Genesis)
        n_turns = max(1, round(abs(diff) / STEP_TURN_RAD))
        return [3] * min(n_turns, n_steps)


def _simulate_step(
    action: int, hab_x: float, hab_z: float, gen_yaw_rad: float
) -> Tuple[float, float, float]:
    """
    Predict new Habitat pose after one primitive action.
    FORWARD moves in the direction the agent faces.
    """
    if action == 1:  # FORWARD
        # heading=-gen_yaw in P2 formula; cos/sin are even/odd → same as cos(gen_yaw), -sin(gen_yaw)
        new_x = hab_x + STEP_MOVE_M * math.cos(gen_yaw_rad)
        new_z = hab_z - STEP_MOVE_M * math.sin(gen_yaw_rad)
        return new_x, new_z, gen_yaw_rad
    if action == 2:  # TURN_LEFT  (CCW in Genesis → yaw increases)
        return hab_x, hab_z, gen_yaw_rad + STEP_TURN_RAD
    if action == 3:  # TURN_RIGHT (CW in Genesis → yaw decreases)
        return hab_x, hab_z, gen_yaw_rad - STEP_TURN_RAD
    return hab_x, hab_z, gen_yaw_rad   # STOP / unknown

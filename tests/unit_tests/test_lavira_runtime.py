"""Unit tests for the Genesis-adapted LaViRA navigation runtime."""

from types import SimpleNamespace

import numpy as np
from PIL import Image

from rlinf.models.embodiment.qwen_nav.action_parser import (
    ACTION_FORWARD,
    ACTION_PANORAMA_SCAN,
    ACTION_STOP,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
)
from rlinf.models.embodiment.qwen_nav.lavira_runtime import (
    LaviraNavigationController,
    LaviraObservation,
    NavigationState,
)
from rlinf.models.embodiment.qwen_nav.lavira_map import FMMPlanner
from rlinf.models.embodiment.qwen_nav.lavira_depth_utils import (
    project_bbox_to_world,
    project_point_to_world,
)
from rlinf.models.embodiment.qwen_nav.lavira_runtime import observation as observation_module
from rlinf.models.embodiment.qwen_nav.lavira_runtime.semantic_map import LaviraSemanticMap
from rlinf.models.embodiment.qwen_nav.lavira_runtime.source_port import (
    check_blocked_directions,
    collision_check_fmm,
)


def _cfg(**overrides):
    values = dict(
        map_backend="sparse_ab",
        map_device="cpu",
        max_steps_to_target=15,
        target_reached_threshold_m=0.75,
        initial_scan_turns=12,
        hfov_deg=79.0,
        camera_height=1.25,
        map_frame_width=32,
        map_frame_height=32,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _obs(x=0.0, z=0.0):
    return LaviraObservation(
        rgb_by_direction=[Image.new("RGB", (32, 32)) for _ in range(4)],
        depth_by_direction=np.full((4, 32, 32), 2.0, dtype=np.float32),
        yaw_by_direction=np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float32),
        hab_x=x,
        hab_z=z,
    )


def test_initial_scan_is_one_panorama_primitive():
    controller = LaviraNavigationController(_cfg())
    obs = _obs()
    controller.observe(obs, {"floor": np.ones((32, 32), dtype=bool)})
    actions = [controller.next_primitive(obs) for _ in range(12)]
    assert actions[0] == ACTION_PANORAMA_SCAN
    assert actions[1:] == [None] * 11
    assert controller.next_primitive(obs) is None
    assert controller.state == NavigationState.NEED_DECISION
    assert "floor" in controller.map.detected_classes


def test_controller_never_emits_stop_for_internal_fmm_arrival():
    controller = LaviraNavigationController(_cfg())
    obs = _obs()
    controller.observe(obs)
    controller.state = NavigationState.NAVIGATING
    controller.goal_xz = (0.1, 0.0)  # already at internal goal
    controller.goal_direction = "backtrack"
    controller.map.fmm_action = lambda *_args: (_ for _ in ()).throw(
        AssertionError("outer 0.75m arrival must run before source FMM")
    )
    assert controller.next_primitive(obs) == ACTION_PANORAMA_SCAN
    assert controller.state == NavigationState.SCANNING
    assert ACTION_STOP != ACTION_FORWARD


def test_outer_arrival_uses_lhx_integer_cells_and_strict_threshold():
    controller = LaviraNavigationController(_cfg(initial_scan_turns=0))
    obs = _obs()
    controller.observe(obs)
    controller.map.fmm_action = lambda *_args: ACTION_FORWARD

    controller.state = NavigationState.NAVIGATING
    controller.goal_xz = (0.74, 0.0)
    assert controller.next_primitive(obs) is None
    assert controller.state == NavigationState.NEED_DECISION

    controller.state = NavigationState.NAVIGATING
    controller.goal_xz = (0.75, 0.0)
    assert controller.next_primitive(obs) == ACTION_FORWARD
    assert controller.state == NavigationState.NAVIGATING


def test_turn_before_waypoint_projection():
    controller = LaviraNavigationController(_cfg(initial_scan_turns=12))
    obs = _obs()
    controller.observe(obs)
    controller.next_primitive(obs)  # leave scanning
    assert controller.accept_navigation(
        "navigate to left", [500, 800], [100, 200, 800, 900], "hall", "", obs
    )
    assert [controller.next_primitive(obs) for _ in range(3)] == [ACTION_TURN_LEFT] * 3
    assert controller.goal_xz is None
    controller.next_primitive(obs)
    assert controller.goal_xz is not None


def test_backtrack_returns_to_saved_waypoint_via_fmm():
    controller = LaviraNavigationController(_cfg(initial_scan_turns=12))
    start = _obs()
    controller.observe(start, {"floor": np.ones((32, 32), dtype=bool)})
    controller.next_primitive(start)
    controller.begin_decision_waypoint(start, 4)
    assert controller.accept_navigation(
        "navigate to forward", [500, 800], [100, 200, 800, 900], "hall", "", start,
        waypoint_id=4,
    )
    controller._goal_reached()
    later = _obs(x=2.0, z=0.0)
    controller.begin_decision_waypoint(later, 5)
    assert controller.available_waypoint_ids(later) == [4]
    assert controller.accept_backtrack(4, later)
    assert controller.state == NavigationState.BACKTRACKING
    assert controller.goal_xz == (0.0, 0.0)
    # Source normal decisions keep failed flags selectable; the flags only
    # alter immediate replan history presentation.
    controller.waypoints.mark_failed_branch(4)
    assert controller.available_waypoint_ids(later) == [4]


def test_backtrack_replan_signal_is_emitted_once_after_fmm_arrival():
    controller = LaviraNavigationController(_cfg(initial_scan_turns=12))
    start = _obs()
    controller.observe(start, {"floor": np.ones((32, 32), dtype=bool)})
    controller.next_primitive(start)
    assert controller.accept_navigation(
        "navigate to forward", [500, 800], [100, 200, 800, 900], "hall", "", start,
        waypoint_id=4,
    )
    at_waypoint = _obs(x=0.0, z=0.0)
    assert controller.accept_backtrack(4, at_waypoint)

    # FMM arrival requests the source panorama before the next decision; the
    # replan signal is still emitted exactly once, never an implicit forward.
    assert controller.next_primitive(at_waypoint) == ACTION_PANORAMA_SCAN
    assert controller.state == NavigationState.SCANNING
    assert controller.consume_backtrack_replan_waypoint_id() == 4
    assert controller.consume_backtrack_replan_waypoint_id() is None
    assert controller.waypoints.get(4).failed_dir


def test_source_backtrack_replan_is_deferred_without_physical_motion():
    controller = LaviraNavigationController(_cfg(initial_scan_turns=0))
    prior = _obs(x=0.0, z=0.0)
    current = _obs(x=2.0, z=0.0)
    controller.begin_decision_waypoint(prior, 0)
    controller._goal_reached()
    controller.begin_decision_waypoint(current, 1)

    assert controller.schedule_backtrack_replan(0)
    assert controller.state == NavigationState.NEED_DECISION
    assert controller.goal_xz is None
    assert controller.next_primitive(current) is None
    assert controller.consume_backtrack_replan_waypoint_id() == 0
    assert controller.consume_backtrack_replan_waypoint_id() is None


def test_current_waypoint_is_not_offered_as_backtrack_target():
    controller = LaviraNavigationController(_cfg(initial_scan_turns=0))
    controller.begin_decision_waypoint(_obs(x=0.0, z=0.0), 0)
    assert controller.available_waypoint_ids(_obs(x=0.0, z=0.0)) == []
    controller._goal_reached()
    controller.begin_decision_waypoint(_obs(x=0.5, z=0.0), 1)
    assert controller.available_waypoint_ids(_obs(x=0.5, z=0.0)) == [0]


def test_runtime_defaults_use_single_source_behavior():
    controller = LaviraNavigationController(SimpleNamespace())
    assert controller.map_backend == "source"
    assert controller.max_steps_to_target == 15
    assert controller.target_reached_threshold_m == 0.75
    assert controller.fmm_goal_threshold_m == 1.0


def test_portal_candidate_rank_does_not_prefer_farther_world_goal():
    centered_near = {
        "index": 4,
        "ray_ratio": 1.0,
        "backoff_count": 0,
        "center_offset": 0.0,
        "goal_xz": [1.0, 0.0],
    }
    corner_far = {
        "index": 0,
        "ray_ratio": 1.0,
        "backoff_count": 0,
        "center_offset": 0.5,
        "goal_xz": [5.0, 0.0],
    }

    chosen = max(
        [corner_far, centered_near],
        key=LaviraNavigationController._portal_candidate_rank,
    )
    assert chosen is centered_near


def test_grounding_candidates_require_depth_map_and_ray_validity(monkeypatch):
    controller = LaviraNavigationController(_cfg(initial_scan_turns=0))
    obs = _obs()
    # The high-confidence bbox samples around pixel (16, 16); invalidate every
    # fallback window there so it cannot borrow depth from a neighboring pixel.
    obs.depth_by_direction[0, 12:21, 12:21] = 0.0
    monkeypatch.setattr(
        controller.map,
        "traversible",
        lambda: np.ones((400, 400), dtype=bool),
    )
    monkeypatch.setattr(controller, "_ray_passability", lambda *_args: 1.0)
    candidates = [
        {
            "index": 0,
            "bbox_2d": [400.0, 200.0, 600.0, 500.0],
            "label": "doorway",
            "confidence": 0.95,
            "region": "center",
            "scene_region": False,
        },
        {
            "index": 1,
            "bbox_2d": [100.0, 300.0, 300.0, 700.0],
            "label": "doorway",
            "confidence": 0.65,
            "region": "left",
            "scene_region": False,
        },
        {
            "index": 2,
            "bbox_2d": [0.0, 0.0, 1000.0, 1000.0],
            "label": "doorway",
            "confidence": 0.99,
            "region": "center",
            "scene_region": True,
        },
    ]

    selected, evaluated = controller.select_grounding_candidate(
        obs, "navigate to forward", candidates
    )

    assert selected is not None
    assert selected["index"] == 1
    assert evaluated[0]["geometry_status"] == "invalid_depth"
    assert evaluated[1]["geometry_status"] == "valid"
    assert evaluated[2]["geometry_status"] == "scene_region"


def test_grounding_candidates_fall_through_to_lhx_when_map_rejects_all(monkeypatch):
    controller = LaviraNavigationController(_cfg(initial_scan_turns=0))
    obs = _obs()
    monkeypatch.setattr(
        controller.map,
        "traversible",
        lambda: np.zeros((400, 400), dtype=bool),
    )
    candidates = [
        {
            "index": 0,
            "bbox_2d": [100.0, 200.0, 300.0, 700.0],
            "label": "chair",
            "confidence": 0.91,
            "region": "left",
            "scene_region": False,
        },
        {
            "index": 1,
            "bbox_2d": [600.0, 200.0, 800.0, 700.0],
            "label": "chair",
            "confidence": 0.63,
            "region": "right",
            "scene_region": False,
        },
    ]

    selected, evaluated = controller.select_grounding_candidate(
        obs, "navigate to forward", candidates
    )

    assert selected is not None
    assert selected["index"] == 0
    assert selected["selection_mode"] == "lhx_passthrough"
    assert all(item["geometry_status"] == "endpoint_blocked" for item in evaluated)


def test_area_target_selects_centered_medium_range_free_space(monkeypatch):
    controller = LaviraNavigationController(_cfg(initial_scan_turns=0))
    obs = _obs()
    monkeypatch.setattr(
        controller.map,
        "traversible",
        lambda: np.ones((400, 400), dtype=bool),
    )
    monkeypatch.setattr(controller, "_ray_passability", lambda *_args: 1.0)

    selected, candidates = controller.select_area_target(
        obs, "navigate to forward"
    )

    assert selected is not None
    assert selected["point_2d"][0] == 500.0
    assert 1.5 <= selected["goal_distance_m"] <= 3.0
    assert selected["selection_mode"] == "area_geometry"
    assert any(item["geometry_status"] == "valid" for item in candidates)


def test_area_target_failure_does_not_invent_a_goal(monkeypatch):
    controller = LaviraNavigationController(_cfg(initial_scan_turns=0))
    obs = _obs()
    monkeypatch.setattr(
        controller.map,
        "traversible",
        lambda: np.zeros((400, 400), dtype=bool),
    )

    selected, candidates = controller.select_area_target(
        obs, "navigate to forward"
    )

    assert selected is None
    assert candidates
    assert not any(item["geometry_status"] == "valid" for item in candidates)


def test_stair_flag_controls_semantic_query_lifecycle():
    controller = LaviraNavigationController(_cfg(initial_scan_turns=0))
    obs = _obs()
    assert "stairs" in controller.semantic_query_classes
    controller.next_primitive(obs)
    assert "stairs" not in controller.semantic_query_classes
    assert controller.accept_navigation(
        "navigate to forward", [500, 800], [100, 200, 800, 900],
        "stairs", "", obs, stair=True,
    )
    assert controller.semantic_query_classes == ["stairs"]


def test_source_backtrack_replan_projects_immediately_without_anchor_replay(monkeypatch):
    """LHX replans from the saved panorama before physical backtracking."""
    controller = LaviraNavigationController(_cfg(initial_scan_turns=0))
    anchor = _obs()
    controller.observe(anchor)
    controller.next_primitive(anchor)
    assert controller.accept_navigation(
        "navigate to forward", [500, 800], [100, 200, 800, 900], "hall", "", anchor,
        waypoint_id=4,
    )
    monkeypatch.setattr(
        controller, "_project_traversible_target", lambda *_args, **_kwargs: (3.0, 1.0)
    )

    assert controller.accept_backtrack_replan(
        4,
        anchor,
        "navigate to left",
        [500, 800],
        [100, 200, 800, 900],
        "doorway",
        "try the left exit",
        replan_waypoint_id=5,
    )

    assert controller.state == NavigationState.NAVIGATING
    assert controller.goal_xz == (3.0, 1.0)
    assert controller.goal_direction == "replan from 4: navigate to left"
    assert controller.waypoints.get(4).failed_dir
    assert controller.waypoints.get(5) is not None
    # The old RLinf path emitted this only after returning to waypoint 4.
    # Source parity requires no deferred anchor-arrival prompt.
    assert controller.consume_backtrack_replan_waypoint_id() is None


def test_collision_evidence_replans_through_fmm_without_extra_recovery_policy():
    controller = LaviraNavigationController(_cfg(initial_scan_turns=0))
    obs = _obs()
    controller.observe(obs)
    controller.state = NavigationState.NAVIGATING
    controller.goal_xz = (4.0, 0.0)
    controller.map.fmm_action = lambda *_args: ACTION_FORWARD

    controller.map.last_forward_collision = True
    assert controller.next_primitive(obs) == ACTION_FORWARD
    controller.map.last_forward_collision = True
    assert controller.next_primitive(obs) == ACTION_FORWARD
    assert controller.state == NavigationState.NAVIGATING


def test_collision_cells_override_navigable_semantics():
    semantic_map = LaviraSemanticMap(frame_width=32, frame_height=32)
    semantic_map.explored.fill(1.0)
    semantic_map.semantic["floor"] = np.ones_like(semantic_map.obstacle)
    cell = semantic_map.world_to_map(0.0, 0.0)
    semantic_map.collision[cell.row, cell.col] = True

    assert semantic_map.traversible()[cell.row, cell.col] == 0.0


def test_source_floor_can_reopen_morphology_only_blocked_cells():
    semantic_map = LaviraSemanticMap(frame_width=32, frame_height=32)
    semantic_map.explored[220:261, 220:261] = 1.0
    # A sufficiently large obstacle survives source min-size filtering. Its
    # closing footprint expands beyond the raw obstacle boundary.
    semantic_map.obstacle[230:251, 230:251] = 1.0
    traversible = semantic_map.rebuild_traversible()

    assert traversible[240, 240] == 0.0
    assert traversible[228, 240] == 1.0


def test_source_blocked_direction_filter_uses_real_four_direction_depth():
    depth = np.full((4, 30, 30), 4.0, dtype=np.float32)
    depth[1].fill(0.8)  # (0.8 - 0.1) / 4.9 < source threshold 0.15.

    assert check_blocked_directions(depth) == {"left"}

    depth[1].fill(1.0)  # (1.0 - 0.1) / 4.9 > 0.15, so it remains available.
    assert check_blocked_directions(depth) == set()


def test_source_collision_mask_points_forward_in_genesis_map_basis():
    # yaw=0 means +X, i.e. increasing map column in the Genesis adapter.
    mask = collision_check_fmm(
        np.asarray([240.0, 240.0]),
        np.asarray([240.0, 240.0]),
        0.0,
        (480, 480),
        cells_per_m=20.0,
    )
    assert mask.sum() > 0
    assert mask[240, 244]


def test_source_map_frame_is_anchored_to_episode_initial_heading():
    semantic_map = LaviraSemanticMap(frame_width=32, frame_height=32)
    semantic_map.reset(10.0, -4.0, np.pi / 2)

    # At initial yaw=+90deg, physical forward is world -Z and must remain
    # source +X (increasing map column), exactly like LHX full_pose theta=0.
    forward = semantic_map.world_to_map(10.0, -5.0)
    assert forward.row == semantic_map.center
    assert forward.col == semantic_map.center + 20

    # Physical left from that heading is world -X and maps to source +Y.
    left = semantic_map.world_to_map(9.0, -4.0)
    assert left.row == semantic_map.center + 20
    assert left.col == semantic_map.center

    world = semantic_map.map_to_world(forward.row, forward.col)
    assert np.allclose(world, (10.0, -5.0), atol=1e-6)


def test_source_mapper_uses_center_stride_and_max_fusion():
    semantic_map = LaviraSemanticMap(frame_width=4, frame_height=3)
    semantic_map.reset(0.0, 0.0, 0.0)
    depth = np.zeros((12, 16), dtype=np.float32)
    depth[2::4, 2::4] = 2.0

    semantic_map.update(depth, 0.0, 0.0, 0.0)
    explored_once = semantic_map.explored.copy()
    obstacle_once = semantic_map.obstacle.copy()
    assert explored_once.max() > 0.0

    semantic_map.update(depth, 0.0, 0.0, 0.0)
    assert np.array_equal(semantic_map.explored, explored_once)
    assert np.array_equal(semantic_map.obstacle, obstacle_once)


def test_collision_replans_in_initial_agent_frame_for_nonzero_world_yaw():
    semantic_map = LaviraSemanticMap(frame_width=32, frame_height=32)
    initial_yaw = -2.1796109676361084
    semantic_map.reset(6.4, -9.9, initial_yaw)
    semantic_map.explored.fill(1.0)
    semantic_map.last_pose = (6.4, -9.9, initial_yaw)
    semantic_map.last_action = ACTION_FORWARD
    semantic_map._mark_collision(
        semantic_map.last_pose,
        (6.4, -9.9, initial_yaw),
    )
    semantic_map.rebuild_traversible()

    goal_x = 6.4 + 4.0 * np.cos(initial_yaw)
    goal_z = -9.9 - 4.0 * np.sin(initial_yaw)
    assert semantic_map.fmm_action(
        6.4, -9.9, initial_yaw, goal_x, goal_z
    ) in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT)


def test_source_target_timeout_returns_to_decision_after_fifteen_primitives():
    controller = LaviraNavigationController(_cfg(initial_scan_turns=12))
    obs = _obs()
    controller.observe(obs)
    controller.waypoints.add(0.0, 0.0, 0.0, "navigate to forward", "hall", "")
    controller.state = NavigationState.NAVIGATING
    controller.goal_xz = (4.0, 0.0)
    controller.map.fmm_action = lambda *_args: ACTION_FORWARD

    assert [controller.next_primitive(obs) for _ in range(15)] == [ACTION_FORWARD] * 15
    assert controller.next_primitive(obs) == ACTION_PANORAMA_SCAN
    assert controller.state == NavigationState.SCANNING
    assert controller.waypoints.nodes == []
    assert controller.consume_removed_waypoint_ids() == [0]


def test_new_goal_inside_arrival_radius_executes_one_fmm_primitive_first(monkeypatch):
    controller = LaviraNavigationController(_cfg(initial_scan_turns=12))
    obs = _obs()
    controller.observe(obs)
    monkeypatch.setattr(
        controller,
        "_project_target_with_portal_candidates",
        lambda *_args, **_kwargs: (0.5, 0.0),
    )
    fmm_calls = []

    def fmm_action(*_args, **_kwargs):
        fmm_calls.append(True)
        return ACTION_FORWARD

    monkeypatch.setattr(controller.map, "fmm_action", fmm_action)
    assert controller.accept_navigation(
        "navigate to forward",
        None,
        [250.0, 250.0, 750.0, 750.0],
        "archway",
        "",
        obs,
        waypoint_id=0,
    )

    # LHX creates the target and emits its first _get_action result in the
    # same loop body, before the outer 0.75m check can run.
    assert controller.next_primitive(obs) == ACTION_FORWARD
    assert fmm_calls == [True]
    assert controller.steps_to_goal == 1
    assert not controller.goal_just_set
    assert controller.state == NavigationState.NAVIGATING

    # The next policy tick represents the post-env.step observation. Only now
    # may the source outer loop classify the 0.5m waypoint as reached.
    assert controller.next_primitive(obs) == ACTION_PANORAMA_SCAN
    assert controller.state == NavigationState.SCANNING


def test_source_timeout_precedes_reached_check_at_step_fifteen():
    controller = LaviraNavigationController(_cfg(initial_scan_turns=12))
    obs = _obs()
    controller.observe(obs)
    waypoint = controller.waypoints.add(
        0.0, 0.0, 0.0, "navigate to forward", "hall", ""
    )
    controller._current_waypoint_id = waypoint.id
    controller.state = NavigationState.NAVIGATING
    controller.goal_xz = (0.0, 0.0)
    controller.steps_to_goal = 15

    assert controller.next_primitive(obs) == ACTION_PANORAMA_SCAN
    assert controller.waypoints.nodes == []
    assert controller.consume_removed_waypoint_ids() == [waypoint.id]


def test_source_stop_navigates_then_scans_before_verifier(monkeypatch):
    controller = LaviraNavigationController(_cfg(initial_scan_turns=12))
    start = _obs(x=0.0, z=0.0)
    goal = _obs(x=1.0, z=0.0)
    controller.observe(start)
    monkeypatch.setattr(
        controller,
        "_project_target_with_portal_candidates",
        lambda *_args, **_kwargs: (1.0, 0.0),
    )
    monkeypatch.setattr(
        controller.map, "fmm_action", lambda *_args, **_kwargs: ACTION_FORWARD
    )

    assert controller.accept_navigation(
        "navigate to forward",
        None,
        [250.0, 250.0, 750.0, 750.0],
        "white rug",
        "",
        start,
        waypoint_id=0,
        stop_after_reach=True,
    )
    assert controller.next_primitive(start) == ACTION_FORWARD
    assert not controller.consume_stop_check_ready()

    assert controller.next_primitive(goal) == ACTION_PANORAMA_SCAN
    assert not controller.consume_stop_check_ready()
    assert controller.next_primitive(goal) is None
    assert controller.state == NavigationState.NEED_DECISION
    assert controller.consume_stop_check_ready()
    assert not controller.consume_stop_check_ready()


def test_panorama_observation_fuses_without_publishing_planner_snapshot(
    monkeypatch,
):
    controller = LaviraNavigationController(_cfg(initial_scan_turns=12))
    obs = _obs()
    # Exercise the source-controller publication boundary with a lightweight
    # map double; source_core itself is covered in test_lavira_source_core.py.
    controller.map_backend = "source"
    controller.map.last_pose = (0.0, 0.0, 0.0)
    update_calls = []
    rebuild_calls = []

    monkeypatch.setattr(
        controller.map,
        "update",
        lambda *_args, **kwargs: update_calls.append(kwargs),
    )
    monkeypatch.setattr(
        controller.map,
        "rebuild_traversible",
        lambda: rebuild_calls.append(True),
    )

    controller.observe(obs, publish_planner_state=False)
    assert update_calls == [{"publish_planner_state": False}]
    assert rebuild_calls == []

    controller.observe(obs, publish_planner_state=True)
    assert update_calls[-1] == {"publish_planner_state": True}
    assert rebuild_calls == [True]


def test_fixed_fmm_destination_uses_goal_not_open_waypoint_threshold():
    traversible = np.ones((80, 80), dtype=np.float32)
    planner = FMMPlanner(traversible)
    goal = np.asarray([40, 40], dtype=np.int32)
    position = np.asarray([40, 10], dtype=np.float32)  # 1.5m at 5cm/cell
    planner.set_goal(goal)

    assert planner.get_short_term_goal(position, None)[2]  # 2m open waypoint threshold
    assert not planner.get_short_term_goal(position, goal)[2]  # 1m fixed target threshold


def test_fmm_turn_direction_matches_lavira_for_left_and_right_subgoals(monkeypatch):
    """Genesis yaw -> map row/column conversion must not invert FMM turns."""
    semantic_map = LaviraSemanticMap(frame_width=32, frame_height=32)
    semantic_map.reset(0.0, 0.0)
    semantic_map.traversible = lambda: np.ones((480, 480), dtype=np.float32)

    fixed_destinations = []

    class _Planner:
        def __init__(self, _traversible):
            self.wp_thresh = 0.0

        def set_goal(self, _goal):
            pass

        def get_short_term_goal(self, position, _goal):
            fixed_destinations.append(_goal)
            return position[0] + 4, position[1] + 4, False

    monkeypatch.setattr(
        "rlinf.models.embodiment.qwen_nav.lavira_runtime.semantic_map.FMMPlanner",
        _Planner,
    )
    # At yaw=0, a row/column target down-right is LaViRA's physical LEFT.
    assert semantic_map.fmm_action(0.0, 0.0, 0.0, 1.0, -1.0) == ACTION_TURN_LEFT
    assert fixed_destinations[-1] is None

    class _RightPlanner(_Planner):
        def get_short_term_goal(self, position, _goal):
            return position[0] - 4, position[1] + 4, False

    monkeypatch.setattr(
        "rlinf.models.embodiment.qwen_nav.lavira_runtime.semantic_map.FMMPlanner",
        _RightPlanner,
    )
    assert semantic_map.fmm_action(0.0, 0.0, 0.0, 1.0, 1.0) == ACTION_TURN_RIGHT


def test_fmm_preserves_lhx_fractional_agent_position(monkeypatch):
    semantic_map = LaviraSemanticMap(frame_width=32, frame_height=32)
    semantic_map.reset(0.0, 0.0)
    semantic_map.traversible = lambda: np.ones((480, 480), dtype=np.float32)
    received_positions = []

    class _Planner:
        def __init__(self, _traversible):
            self.wp_thresh = 0.0
            self.fmm_dist = np.ones((480, 480), dtype=np.float32)

        def set_goal(self, _goal):
            pass

        def get_short_term_goal(self, position, _fixed_destination):
            received_positions.append(position.copy())
            return int(position[0]) + 4, int(position[1]) + 4, False

    monkeypatch.setattr(
        "rlinf.models.embodiment.qwen_nav.lavira_runtime.semantic_map.FMMPlanner",
        _Planner,
    )
    semantic_map.fmm_action(0.013, -0.017, 0.0, 1.0, 0.0)

    expected = semantic_map._world_to_map_float(0.013, -0.017)
    np.testing.assert_allclose(received_positions[0], expected)
    assert not np.allclose(received_positions[0], np.floor(received_positions[0]))


def test_fmm_audit_covers_collision_to_primitive_chain():
    semantic_map = LaviraSemanticMap(frame_width=32, frame_height=32)
    semantic_map.reset(0.0, 0.0)
    semantic_map.explored.fill(1.0)
    semantic_map.rebuild_traversible()
    semantic_map._mark_collision((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    semantic_map.rebuild_traversible()

    action = semantic_map.fmm_action(0.0, 0.0, 0.0, 2.0, 0.0)
    audit = semantic_map.last_fmm_audit

    assert action in (ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT)
    assert semantic_map.last_collision_audit["candidate_cells"] > 0
    assert audit["collision_cells_local_11x11"] > 0
    assert audit["nearest_goal_map_rc"]
    assert audit["fmm_distance_at_agent"] is not None
    assert audit["short_term_goal_map_rc"]
    assert audit["primitive_action"] == action


def test_invalid_depth_returns_to_decision_without_navigation_fallback(monkeypatch):
    controller = LaviraNavigationController(_cfg(initial_scan_turns=12))
    obs = _obs()
    controller.observe(obs)
    monkeypatch.setattr(LaviraObservation, "project_front_target", lambda *_args, **_kwargs: None)
    assert controller.accept_navigation(
        "navigate to forward", [500, 800], [100, 200, 800, 900], "hall", "", obs
    )
    assert controller.next_primitive(obs) is None
    assert controller.state == NavigationState.NEED_DECISION
    assert controller.goal_xz is None
    assert "different visible target" in controller.consume_navigation_feedback()
    assert controller.consume_navigation_feedback() is None


def test_target_projection_retries_with_source_depth_backoff(monkeypatch):
    controller = LaviraNavigationController(_cfg(initial_scan_turns=0))
    obs = _obs()
    controller.observe(obs)
    controller.next_primitive(obs)
    seen_depths = []

    def _project(_self, _point, _bbox, stair=False, depth_override=None):
        seen_depths.append(float(depth_override.max()))
        return (1.0, 0.0) if len(seen_depths) == 1 else (0.0, 1.0)

    monkeypatch.setattr(LaviraObservation, "project_front_target", _project)
    traversible = np.zeros((480, 480), dtype=np.float32)
    open_target = controller.map.world_to_map(0.0, 1.0)
    traversible[open_target.row, open_target.col] = 1.0
    controller.map.traversible = lambda: traversible
    goal = controller._project_traversible_target(obs, [500, 500], None, False)
    assert goal == (0.0, 1.0)
    assert len(seen_depths) == 2
    assert np.isclose(seen_depths[0] - seen_depths[1], 0.1)


def test_target_projection_accepts_exhausted_nontraversible_goal_like_source(monkeypatch):
    controller = LaviraNavigationController(_cfg(initial_scan_turns=0))
    obs = _obs()
    obs.depth_by_direction[0].fill(0.21)
    controller.observe(obs)
    controller.next_primitive(obs)
    controller.map.traversible = lambda: np.zeros((480, 480), dtype=np.float32)

    goal = controller._project_traversible_target(obs, [500, 500], None, False)
    assert goal is not None
    assert controller.audit_projection["accepted_reason"] == "source_depth_exhausted"
    assert controller.audit_projection["backoff_count"] == 2
    assert np.isclose(controller.audit_projection["initial_depth_m"]["median"], 0.21)
    assert controller.audit_projection["attempts"]
    assert np.isclose(
        controller.audit_projection["attempts"][-1]["sample_depth_m"], 0.01,
        atol=1e-6,
    )


def test_target_projection_with_nan_depth_terminates_and_audits():
    controller = LaviraNavigationController(_cfg(initial_scan_turns=0))
    obs = _obs()
    obs.depth_by_direction[0].fill(np.nan)
    controller.map.traversible = lambda: np.zeros((480, 480), dtype=np.float32)

    assert controller._project_traversible_target(
        obs, [500, 500], None, False
    ) is None
    assert controller.audit_projection["backoff_count"] == 1
    assert controller.audit_projection["attempts"] == [{
        "backoff_count": 0,
        "sample_pixel_uv": None,
        "sample_depth_m": None,
        "goal_xz": None,
        "goal_map_rc": None,
        "goal_traversible": False,
    }]


def test_observation_projection_uses_configured_hfov():
    depth = np.full((4, 480, 640), 2.0, dtype=np.float32)
    rgb = [Image.new("RGB", (640, 480)) for _ in range(4)]
    yaws = np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float32)
    narrow = LaviraObservation(rgb, depth, yaws, 0.0, 0.0, hfov_deg=79.0)
    wide = LaviraObservation(rgb, depth, yaws, 0.0, 0.0, hfov_deg=105.0)

    narrow_goal = narrow.project_target("navigate to forward", point_2d=[800, 500])
    wide_goal = wide.project_target("navigate to forward", point_2d=[800, 500])
    assert narrow_goal is not None and wide_goal is not None
    assert not np.allclose(narrow_goal, wide_goal)


def test_stair_projection_uses_bbox_top_not_point(monkeypatch):
    def _point_projection(point, *_args, **_kwargs):
        return np.asarray([2.0, 3.0]) if point == [450.0, 200.0] else np.asarray([99.0, 99.0])

    monkeypatch.setattr(observation_module, "project_point_to_world", _point_projection)
    assert _obs().project_front_target(
        point_2d=[500, 900], bbox_2d=[100, 200, 800, 900], stair="up"
    ) == (2.0, 3.0)


def test_point_projection_uses_source_window_expansion_and_five_meter_cap():
    depth = np.zeros((480, 640), dtype=np.float32)
    depth[240, 324] = 2.0  # only the source 9x9 fallback window reaches this
    assert project_point_to_world([500, 500], depth, 0.0, 0.0, 0.0) is not None

    depth[240, 324] = 6.0
    assert project_point_to_world([500, 500], depth, 0.0, 0.0, 0.0) is None


def test_bbox_projection_uses_actual_median_depth_pixel():
    depth = np.zeros((10, 10), dtype=np.float32)
    depth[2, 1] = 1.0
    depth[3, 4] = 2.0
    depth[4, 8] = 3.0
    result = project_bbox_to_world(
        [0, 0, 1000, 1000],
        depth,
        0.0,
        0.0,
        0.0,
        K=np.eye(3),
        render_w=10,
        render_h=10,
    )
    assert result is not None
    assert np.allclose(result, [2.0, 8.0])


def test_point_projection_preserves_reflected_camera_right_axis():
    depth = np.full((10, 10), 2.0, dtype=np.float32)
    K = np.eye(3)

    # At yaw=0, image right is +Habitat Z after Genesis Y→-Habitat Z.
    yaw_zero = project_point_to_world(
        [900, 500], depth, 0.0, 0.0, 0.0, K=K, render_w=10, render_h=10
    )
    assert yaw_zero is not None
    assert np.allclose(yaw_zero, [2.0, 18.0])

    # At yaw=pi/2, camera forward is -Habitat Z and image right is +Habitat X.
    yaw_quarter_turn = project_point_to_world(
        [900, 500], depth, np.pi / 2.0, 0.0, 0.0,
        K=K, render_w=10, render_h=10,
    )
    assert yaw_quarter_turn is not None
    assert np.allclose(yaw_quarter_turn, [18.0, -2.0], atol=1e-6)


def test_debug_map_snapshot_saves_2d_map_and_gt_overlay(tmp_path):
    semantic_map = LaviraSemanticMap(frame_width=32, frame_height=32)
    depth = np.full((32, 32), 2.0, dtype=np.float32)
    semantic_map.update(depth, 0.0, 0.0, 0.0, {"floor": np.ones((32, 32), dtype=bool)})

    png_path = tmp_path / "map.png"
    npz_path = tmp_path / "map.npz"
    semantic_map.save_debug_snapshot(
        str(png_path),
        npz_path=str(npz_path),
        gt_reference_path=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        goal_xz=(1.0, 0.0),
    )

    assert png_path.exists() and png_path.stat().st_size > 0
    saved = np.load(npz_path)
    assert saved["obstacle"].shape == (480, 480)
    assert saved["agent_path_xz"].shape == (1, 2)


def test_semantic_map_uses_lavira_five_meter_projection_window():
    semantic_map = LaviraSemanticMap(frame_width=32, frame_height=32)
    far_depth = np.full((32, 32), 5.1, dtype=np.float32)

    semantic_map.update(far_depth, 0.0, 0.0, 0.0)

    assert not semantic_map.explored.any()
    assert not semantic_map.obstacle.any()

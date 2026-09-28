from types import SimpleNamespace

import pytest
import torch

from rlinf.models.embodiment.qwen_nav.action_parser import (
    ACTION_FORWARD,
    ACTION_NOOP,
    ACTION_PARSE_FAIL,
    ACTION_PARSE_OK_HAS_BBOX_BASE,
    ACTION_TURN_LEFT,
    ACTION_STOP,
    parse_lavira_waypoint_json,
)
from rlinf.models.embodiment.qwen_nav.prompts import (
    LAVIRA_WAYPOINT_SYSTEM_PROMPT,
    build_backtrack_replan_user_content_text,
    build_waypoint_user_content_text,
)
from rlinf.models.embodiment.qwen_nav.qwen_nav_policy import (
    _HistoryCache,
    _WaypointRecord,
    QwenNavPolicy,
    _sample_waypoints,
)
from rlinf.models.embodiment.qwen_nav.grounded_sam import GroundedSAMRefineResult
from rlinf.models.embodiment.qwen_nav.lavira_runtime import NavigationState
from PIL import Image
import numpy as np


def _json(action="navigate to forward", stop=False, target="hallway"):
    stop_txt = "true" if stop else "false"
    return f"""{{
      "progress_analysis": "near hallway",
      "reasoning_plan_action": "go forward",
      "planning": "continue",
      "action": "{action}",
      "stop": {stop_txt},
      "stair": false,
      "target": "{target}"
    }}"""


def test_waypoint_navigate_target_ok():
    parsed = parse_lavira_waypoint_json(_json())
    assert parsed.ok
    assert parsed.action_type == "NAVIGATE"
    assert parsed.actions == [ACTION_FORWARD]
    assert parsed.target == "hallway"


def test_area_geometry_failure_preserves_dino_lhx_fallback():
    parsed = parse_lavira_waypoint_json(
        _json(target="hallway floor")
    )
    dino_candidate = {
        "index": 0,
        "candidate_type": "dino",
        "bbox_2d": [100.0, 200.0, 700.0, 800.0],
        "label": "hallway entrance",
        "confidence": 0.7,
        "region": "center",
        "scene_region": False,
    }
    sam_result = GroundedSAMRefineResult(
        detected=True,
        bbox_2d=list(dino_candidate["bbox_2d"]),
        label="hallway entrance",
        confidence=0.7,
        candidates=[dino_candidate],
    )

    class _Controller:
        def select_area_target(self, *_args, **_kwargs):
            return None, [{"candidate_type": "area", "geometry_status": "endpoint_blocked"}]

        def select_grounding_candidate(self, _observation, _action, candidates, **_kwargs):
            selected = next(item for item in candidates if item.get("bbox_2d") is not None)
            selected["selection_mode"] = "lhx_passthrough"
            return selected, candidates

    result = QwenNavPolicy._apply_grounding_geometry_filter(
        _Controller(), object(), parsed, sam_result
    )

    assert result.detected
    assert result.bbox_2d == dino_candidate["bbox_2d"]
    assert "area_no_valid_goal" in result.fallback_reason
    assert "lhx_dino_passthrough" in result.fallback_reason


def test_area_geometry_success_uses_point_without_dino_bbox():
    parsed = parse_lavira_waypoint_json(_json(target="hallway floor"))

    class _Controller:
        def select_area_target(self, *_args, **_kwargs):
            selected = {
                "point_2d": [500.0, 700.0],
                "selection_mode": "area_geometry",
            }
            return selected, [selected]

        def select_grounding_candidate(self, *_args, **_kwargs):
            raise AssertionError("DINO selection must not override valid area geometry")

    result = QwenNavPolicy._apply_grounding_geometry_filter(
        _Controller(), object(), parsed, None
    )

    assert result.detected
    assert result.point_2d == [500.0, 700.0]
    assert result.bbox_2d is None
    assert result.fallback_reason == "area_geometry"


def test_area_geometry_failure_without_dino_keeps_source_fallback_reachable():
    parsed = parse_lavira_waypoint_json(_json(target="hallway floor"))

    class _Controller:
        def select_area_target(self, *_args, **_kwargs):
            return None, [{"candidate_type": "area", "geometry_status": "ray_blocked"}]

    result = QwenNavPolicy._apply_grounding_geometry_filter(
        _Controller(), object(), parsed, None
    )

    assert not result.detected
    assert result.fallback_reason == "area_no_valid_goal"
    assert result.candidates[0]["geometry_status"] == "ray_blocked"


def test_waypoint_grounding_sends_raw_lhx_query_without_mutating_parsed_target():
    parsed = parse_lavira_waypoint_json(_json(target="archway exit"))
    seen = {}

    class _Refiner:
        def refine(self, _image_rgb, target, target_region, **kwargs):
            seen["target"] = target
            seen["target_region"] = target_region
            seen.update(kwargs)
            return GroundedSAMRefineResult(False, fallback_reason="test")

    policy = object.__new__(QwenNavPolicy)
    policy.prompt_style = "lavira_waypoint"
    policy.canonicalize_lavira_waypoint_query = False
    policy.grounded_sam_enabled = True
    policy._get_grounded_sam = lambda: _Refiner()
    views = [Image.new("RGB", (32, 24)) for _ in range(4)]

    policy._refine_waypoint_with_grounded_sam(parsed, views)

    assert seen["target"] == "archway exit"
    assert seen["target_region"] == "any"
    assert seen["grounding_classes"] == ["archway exit"]
    assert seen["source_lhx"] is True
    assert parsed.target == "archway exit"


def test_waypoint_grounding_can_canonicalize_only_the_dino_query():
    parsed = parse_lavira_waypoint_json(_json(target="archway exit"))
    seen = {}

    class _Refiner:
        def refine(self, _image_rgb, target, target_region, **kwargs):
            seen["target"] = target
            seen["target_region"] = target_region
            seen.update(kwargs)
            return GroundedSAMRefineResult(False, fallback_reason="test")

    policy = object.__new__(QwenNavPolicy)
    policy.prompt_style = "lavira_waypoint"
    policy.canonicalize_lavira_waypoint_query = True
    policy.grounded_sam_enabled = True
    policy._get_grounded_sam = lambda: _Refiner()
    views = [Image.new("RGB", (32, 24)) for _ in range(4)]

    policy._refine_waypoint_with_grounded_sam(parsed, views)

    assert seen["target"] == "archway"
    assert seen["target_region"] == "any"
    assert seen["grounding_classes"] == ["archway"]
    assert seen["source_lhx"] is True
    assert parsed.target == "archway exit"


def test_hallway_entrance_maps_to_fixed_input_verified_archway_query():
    parsed = parse_lavira_waypoint_json(_json(target="hallway entrance"))
    seen = {}

    class _Refiner:
        def refine(self, _image_rgb, target, target_region, **kwargs):
            seen["target"] = target
            seen.update(kwargs)
            return GroundedSAMRefineResult(False, fallback_reason="test")

    policy = object.__new__(QwenNavPolicy)
    policy.prompt_style = "lavira_waypoint"
    policy.canonicalize_lavira_waypoint_query = True
    policy.grounded_sam_enabled = True
    policy._get_grounded_sam = lambda: _Refiner()
    views = [Image.new("RGB", (32, 24)) for _ in range(4)]

    policy._refine_waypoint_with_grounded_sam(parsed, views)

    assert seen["target"] == "archway"
    assert seen["grounding_classes"] == ["archway"]
    assert parsed.target == "hallway entrance"


def test_canonical_grounding_maps_free_form_query():
    parsed = parse_lavira_waypoint_json(_json(target="archway exit"))
    seen = {}

    class _Refiner:
        def refine(self, _image_rgb, target, target_region, **kwargs):
            seen["target"] = target
            seen["target_region"] = target_region
            seen.update(kwargs)
            return GroundedSAMRefineResult(False, fallback_reason="test")

    policy = object.__new__(QwenNavPolicy)
    policy.prompt_style = "canonical_v1"
    policy.grounded_sam_enabled = True
    policy._get_grounded_sam = lambda: _Refiner()
    views = [Image.new("RGB", (32, 24)) for _ in range(4)]

    policy._refine_waypoint_with_grounded_sam(parsed, views)

    assert seen["target"] == "archway"
    assert seen["grounding_classes"] == ["archway"]
    assert seen["source_lhx"] is False


def test_batched_waypoint_grounding_uses_raw_query_and_lhx_stair_aliases():
    seen = []

    class _Refiner:
        def refine(self, _image_rgb, target, target_region, **kwargs):
            seen.append((target, target_region, kwargs))
            return GroundedSAMRefineResult(False, fallback_reason="test")

    policy = object.__new__(QwenNavPolicy)
    torch.nn.Module.__init__(policy)
    policy.prompt_style = "lavira_waypoint"
    policy._grounded_sam = _Refiner()
    policy.grounded_sam_cfg = None
    jobs = [{
        "image_rgb": np.zeros((24, 32, 3), dtype=np.uint8),
        "target": "Stone Steps",
        "target_region": "left",
        "stair": True,
    }]

    policy._refine_grounded_sam_jobs(jobs)

    assert seen[0][0] == "stone steps"
    assert seen[0][1] == "any"
    assert seen[0][2]["grounding_classes"] == [
        "stone steps", "stairs", "stairway", "staircase", "steps"
    ]
    assert seen[0][2]["source_lhx"] is True
    assert jobs[0]["target"] == "Stone Steps"


def test_waypoint_backtrack_ok():
    parsed = parse_lavira_waypoint_json(
        _json(action="backtrack to 7", target="")
    )
    assert parsed.ok
    assert parsed.action_type == "BACKTRACK"
    assert parsed.waypoint_id == 7
    assert parsed.actions == []


def test_waypoint_stop_ok():
    parsed = parse_lavira_waypoint_json(_json(stop=True, target="white rug"))
    assert parsed.ok
    assert parsed.action_type == "STOP"
    assert parsed.actions == [ACTION_STOP]


def test_waypoint_missing_field_invalid():
    parsed = parse_lavira_waypoint_json('{"action": "navigate to forward"}')
    assert not parsed.ok
    assert parsed.err.startswith("missing_fields:")


def test_waypoint_behind_is_executable():
    parsed = parse_lavira_waypoint_json(_json(action="navigate to behind"))
    assert parsed.ok
    assert parsed.action_type == "NAVIGATE"
    assert parsed.actions == [ACTION_TURN_LEFT] * 6 + [ACTION_FORWARD]


def test_waypoint_action_with_description_suffix_invalid():
    parsed = parse_lavira_waypoint_json(
        _json(action="navigate to forward - continue straight ahead")
    )
    assert not parsed.ok
    assert parsed.err == (
        "unknown_direction:navigate to forward - continue straight ahead"
    )


def test_waypoint_rejects_non_string_target():
    parsed = parse_lavira_waypoint_json(_json().replace('"target": "hallway"', '"target": false'))
    assert not parsed.ok
    assert parsed.err == "invalid_type:target"


def test_waypoint_rejects_non_boolean_stop():
    parsed = parse_lavira_waypoint_json(_json().replace('"stop": false', '"stop": "false"'))
    assert not parsed.ok
    assert parsed.err == "invalid_type:stop"


def test_waypoint_rejects_geometry_fields():
    text = _json().replace(
        '      "target": "hallway"',
        '      "bbox_2d": [100, 200, 700, 900],\n'
        '      "point_2d": [400, 800],\n'
        '      "target": "hallway"',
    )
    parsed = parse_lavira_waypoint_json(text)
    assert not parsed.ok
    assert parsed.err.startswith("unexpected_fields:")


def test_waypoint_backtrack_executes_when_visual_fields_describe_route():
    parsed = parse_lavira_waypoint_json(_json(action="backtrack to 7"))
    assert parsed.ok
    assert parsed.action_type == "BACKTRACK"
    assert parsed.waypoint_id == 7
    # Backtracking is executable from the waypoint id alone and never receives
    # model-geometry format credit.
    assert not (parsed.reward_bits & (1 << 3))


def test_waypoint_requires_target_for_navigation():
    parsed = parse_lavira_waypoint_json(_json(target=""))
    assert not parsed.ok
    assert parsed.err == "empty_navigate_target"


def test_waypoint_rejects_wrong_field_order_and_markdown_fence():
    wrong_order = """{
      "action": "navigate to forward",
      "progress_analysis": "near hallway",
      "reasoning_plan_action": "go forward",
      "planning": "continue",
      "stop": false,
      "stair": false,
      "target": "hallway"
    }"""
    parsed = parse_lavira_waypoint_json(wrong_order)
    assert not parsed.ok
    assert parsed.err == "field_order"

    parsed = parse_lavira_waypoint_json(f"```json\n{_json()}\n```")
    assert not parsed.ok
    assert parsed.err == "no_json_found"


def test_waypoint_prompt_matches_lavira_template_sections():
    text = build_waypoint_user_content_text(
        instruction="Walk up the stairs.",
        waypoint_infos=[
            {
                "id": 0,
                "action": "navigate to left",
                "target": "stairs entry",
                "progress_analysis": "Started near the lounge and found stairs.",
                "continuous_count": 5,
            }
        ],
    )

    assert '# Instruction\n"Walk up the stairs."' in text
    assert "On each after-turn view below, the green box and red dot" in text
    assert "Waypoint 0 arrival view (image labeled WP0):" in text
    assert "At Waypoint 0 you turned left." in text
    assert "Waypoint 0 -> Current Position (continuous frames):" in text
    assert 'Your previous progress_analysis: "Started near the lounge and found stairs."' in text
    assert "Current FORWARD view:" in text
    assert "Current BEHIND view:" in text
    assert "backtrack to <waypoint_id> - return to a previous waypoint (Available IDs: 0)" in text
    assert '"reasoning_plan_action": "<one-sentence justification of the planning/action>"' in text
    assert '"reasoning_bbox_point"' not in text
    assert '"bbox_2d"' not in text
    assert '"point_2d"' not in text
    assert "GroundingDINO should detect" in text
    assert '   - "navigate to behind"' not in text
    assert "   - navigate to behind - turn around and go forward" not in text
    assert "   - navigate to forward - continue straight ahead" in text
    assert LAVIRA_WAYPOINT_SYSTEM_PROMPT.startswith(
        "You are an embodied navigation agent."
    )
    assert text.count("<image>") == 11  # arrival + after-turn + 5 continuous + 4 current


def test_waypoint_prompt_allows_behind_only_initially():
    initial = build_waypoint_user_content_text(
        instruction="Walk up the stairs.",
        waypoint_infos=[],
    )
    later = build_waypoint_user_content_text(
        instruction="Walk up the stairs.",
        waypoint_infos=[{
            "id": 0,
            "action": "navigate to behind",
            "target": "stairs",
            "progress_analysis": "Found the stairs.",
            "continuous_count": 0,
        }],
    )

    assert "   - navigate to behind - turn around and go forward" in initial
    assert "   - navigate to behind - turn around and go forward" not in later
    assert "Current BEHIND view:" in later


def test_waypoint_prompt_uses_runtime_backtrack_ids_as_authority():
    text = build_waypoint_user_content_text(
        instruction="Walk up the stairs.",
        waypoint_infos=[
            {"id": 0, "action": "navigate to left", "target": "stairs"},
            {"id": 1, "action": "navigate to forward", "target": "landing"},
        ],
        available_backtrack_ids=[0],
    )

    assert "Available IDs: 0" in text
    assert "Available IDs: 1" not in text


def test_source_layered_history_hides_failed_branch_images():
    image = Image.new("RGB", (8, 8), (255, 0, 0))
    blank = Image.new("RGB", (8, 8), (127, 127, 127))
    records = [
        _WaypointRecord(0, (0.0, 0.0, 0.0), image, image, [], "navigate to forward",
                        [0, 0, 1, 1], [1, 1], "hall", "started"),
        _WaypointRecord(1, (1.0, 0.0, 0.0), image, image, [], "navigate to left",
                        [0, 0, 1, 1], [1, 1], "door", "turned left"),
    ]
    records[0].failed_dir = True
    records[1].failed = True
    images, ids, infos = _sample_waypoints(
        records, 2, current_pose=np.asarray([1.0, 0.0, 0.0]),
        layered_history=True, blank_image=blank,
    )

    assert ids == [0, 1]
    assert infos[0]["failed_dir"] and infos[1]["failed"]
    assert images[0] is image                 # branch point arrival remains
    assert all(frame is blank for frame in images[1:])  # failed turn + branch hidden


def test_source_layered_history_only_uses_live_frames_for_zone_one():
    first = Image.new("RGB", (8, 8), (255, 0, 0))
    second = Image.new("RGB", (8, 8), (0, 255, 0))
    live = Image.new("RGB", (8, 8), (0, 0, 255))
    blank = Image.new("RGB", (8, 8), (127, 127, 127))
    records = [
        _WaypointRecord(0, (0.0, 0.0, 0.0), first, first, [first], "navigate to forward",
                        [0, 0, 1, 1], [1, 1], "hall", "started"),
        _WaypointRecord(1, (1.0, 0.0, 0.0), second, second, [second], "navigate to left",
                        [0, 0, 1, 1], [1, 1], "door", "turned left"),
    ]
    images, _ids, infos = _sample_waypoints(
        records, 2, current_leg_frames=[live],
        current_pose=np.asarray([1.0, 0.0, 0.0]), layered_history=True,
        blank_image=blank,
    )

    # Zone 2: only arrival/turn for older waypoint; its saved continuous
    # frame must not leak into the current decision prompt.
    assert images[:2] == [first, first]
    assert all(frame is blank for frame in images[2:7])
    # Zone 1 is attached solely to the previous waypoint/current leg.
    assert images[7:10] == [second, second, live]
    assert infos[0]["continuous_count"] == 0
    assert infos[1]["continuous_count"] == 1


def test_backtrack_replan_prompt_keeps_source_three_sections():
    text = build_backtrack_replan_user_content_text(
        instruction="Return to the red rug.",
        history_entries=[{"id": 2, "image_count": 7}],
        failed_entries=[{"id": 3, "image_count": 7}],
        previous_action="navigate to left",
        blocked_directions={"forward"},
        padding_image_count=7,
    )

    assert "Navigation History" in text
    assert "Previous Trajectory" in text
    assert "Current 4-directional views" in text
    assert "Trajectory after Backtrack Point (Failed Path)" in text
    assert "navigate to forward" not in text
    assert "navigate to left" in text
    assert text.count("<image>") == 25  # retained + failed + padding + F/L/B/R


def test_policy_replan_uses_real_frames_after_backtrack_point():
    class _Processor:
        def apply_chat_template(self, messages, **_kwargs):
            return "".join(
                item.get("text", "")
                if item["type"] == "text"
                else "<|image_pad|>"
                for item in messages[1]["content"]
            )

    target = Image.new("RGB", (8, 8), (255, 0, 0))
    failed_path_frame = Image.new("RGB", (8, 8), (0, 0, 255))
    blank = Image.new("RGB", (8, 8), (127, 127, 127))
    cache = _HistoryCache()
    cache.history_images = [target, failed_path_frame]
    cache.waypoints = [
        _WaypointRecord(0, (0.0, 0.0, 0.0), target, target, [], "navigate to left",
                        [0, 0, 1, 1], [1, 1], "hall", "before branch",
                        history_end_index=1),
    ]
    policy = object.__new__(QwenNavPolicy)
    policy._blank_image = blank
    policy._waypoint_images_per_record = 7
    policy.history_max_frames = 2
    policy.use_4dir = True
    policy.image_size = (8, 8)
    policy.processor = _Processor()

    text, images = policy._build_backtrack_replan_prompt(
        instruction="Return to the rug.", cache=cache, waypoint_id=0,
        current_view=target, extra_views=None, blocked_directions=None,
    )

    assert "Trajectory after Backtrack Point (Failed Path):" in text
    assert np.array_equal(np.asarray(images[7]), np.asarray(failed_path_frame))
    assert text.count("<image>") == len(images)


def test_policy_executes_source_immediate_replan_instead_of_anchor_replay(monkeypatch):
    """A BACKTRACK response must cause one immediate old-pose replan call."""
    class _Processor:
        def apply_chat_template(self, messages, **_kwargs):
            return "".join(
                item.get("text", "")
                if item["type"] == "text"
                else "<|image_pad|>"
                for item in messages[1]["content"]
            )

    # Build only the inference state used by this regression; no model or GPU
    # is involved because generation is replaced with two deterministic replies.
    policy = object.__new__(QwenNavPolicy)
    torch.nn.Module.__init__(policy)
    policy.collect_forward_inputs = False
    policy._padding_initialized = True
    policy.prompt_style = "lavira_waypoint"
    policy.use_4dir = True
    policy.image_size = (32, 32)
    policy._blank_image = Image.new("RGB", policy.image_size, (127, 127, 127))
    policy._waypoint_images_per_record = 7
    policy.history_max_frames = 2
    policy.history_every_k = 1
    policy.history_action_aware = False
    policy.history_forward_interval = 5
    policy.history_turn_interval = 2
    policy.lavira_runtime_enabled = True
    policy.lavira_source_prompt_alignment = False
    policy.lavira_runtime_cfg = SimpleNamespace(
        initial_scan_turns=0,
        max_steps_to_target=15,
        target_reached_threshold_m=0.75,
        hfov_deg=105.0,
            camera_height=1.25,
            map_frame_width=32,
            map_frame_height=32,
            layered_history=True,
        history_wp_max=8,
        layered_backtrack_radius_m=6.0,
    )
    policy.lavira_layered_history = True
    policy.lavira_history_wp_max = 8
    policy.lavira_backtrack_radius_m = 6.0
    policy.lavira_backtrack_second_chance = True
    policy.grounded_sam_enabled = False
    policy.stop_double_check_enabled = False
    policy.json_retry_enabled = False
    policy._parse_fail_debug_printed = 0
    policy._parse_fail_debug_limit = 0
    policy._per_env_cache = {}
    policy._lavira_runtime = {}
    policy._grounded_sam_episode_counts = {}
    policy._lavira_map_episode_counts = {}
    policy._wm_state_collector = None
    policy.lavira_map_visualize = False
    policy._action_stats_flush_every = 0
    policy.processor = _Processor()
    policy._save_lavira_map_snapshot = lambda *_args, **_kwargs: None
    policy._write_lavira_projection_audit = lambda *_args, **_kwargs: None
    policy._save_grounded_sam_diagnostic = lambda *_args, **_kwargs: None
    policy._refine_waypoint_with_grounded_sam = lambda *_args, **_kwargs: None

    cache = policy._get_cache(0)
    anchor = Image.new("RGB", (32, 32), (255, 0, 0))
    cache.history_images = [anchor]
    cache.waypoints = [
        _WaypointRecord(
            0, (0.0, 0.0, 0.0), anchor, anchor, [],
            "navigate to forward", [100, 200, 800, 900], [500, 800],
            "hall", "started", history_end_index=1,
            anchor_views=[anchor] * 4,
            anchor_depth_by_direction=np.full((4, 32, 32), 2.0, dtype=np.float32),
        )
    ]
    cache.next_waypoint_id = 1
    controller = policy._get_lavira_runtime(0)
    controller.state = NavigationState.NEED_DECISION
    controller.waypoints.add(
        0.0, 0.0, 0.0, "navigate to forward", "hall", "started", waypoint_id=0
    )
    monkeypatch.setattr(
        controller, "_project_traversible_target", lambda *_args, **_kwargs: (2.0, 0.0)
    )
    monkeypatch.setattr(
        controller.map, "fmm_action", lambda *_args, **_kwargs: ACTION_FORWARD
    )

    outputs = [
        _json(action="backtrack to 0", target=""),
        _json(action="navigate to left", target="doorway"),
    ]
    calls = []

    def _generate(prompts, images):
        calls.append((prompts, images))
        text = outputs[len(calls) - 1]
        return [text], [torch.tensor([[1]], dtype=torch.long)]

    policy._batch_generate = _generate
    obs = {
        "main_images": np.zeros((1, 32, 32, 3), dtype=np.uint8),
        "extra_view_images": np.zeros((1, 3, 32, 32, 3), dtype=np.uint8),
        "wrist_images": np.full((1, 4, 32, 32, 1), 2.0, dtype=np.float32),
            # The current pose must be away from the anchor: the runtime no
            # longer offers the waypoint occupied by the agent as a stale
            # backtrack candidate.
            "states": np.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        "task_descriptions": ["Walk through the doorway."],
    }

    actions, _ = policy.predict_action_batch(obs)

    assert len(calls) == 2
    assert actions.shape == (1, 1)
    assert int(actions[0, 0]) not in (ACTION_NOOP, ACTION_PARSE_FAIL)
    assert controller.state == NavigationState.NAVIGATING
    assert controller.goal_xz == (2.0, 0.0)
    assert controller.waypoints.get(0).failed_dir
    assert controller.consume_backtrack_replan_waypoint_id() is None


@pytest.mark.parametrize(
    ("detected", "detected_bbox", "expected_bbox", "stop"),
    [
        (
            True,
            [200.0, 200.0, 800.0, 900.0],
            [200.0, 200.0, 800.0, 900.0],
            False,
        ),
        (False, None, [250.0, 250.0, 750.0, 750.0], False),
        (
            True,
            [200.0, 200.0, 800.0, 900.0],
            [200.0, 200.0, 800.0, 900.0],
            True,
        ),
    ],
)
def test_policy_routes_grounded_sam_bbox_only_into_fmm_controller(
    monkeypatch, detected, detected_bbox, expected_bbox, stop
):
    """A NAVIGATE or STOP target must create an FMM waypoint."""
    class _Processor:
        def apply_chat_template(self, messages, **_kwargs):
            return "".join(
                item.get("text", "")
                if item["type"] == "text"
                else "<|image_pad|>"
                for item in messages[1]["content"]
            )

    policy = object.__new__(QwenNavPolicy)
    torch.nn.Module.__init__(policy)
    policy.collect_forward_inputs = False
    policy._padding_initialized = True
    policy.prompt_style = "lavira_waypoint"
    policy.use_4dir = True
    policy.image_size = (32, 32)
    policy._blank_image = Image.new("RGB", policy.image_size, (127, 127, 127))
    policy._waypoint_images_per_record = 7
    policy.history_max_frames = 2
    policy.history_every_k = 1
    policy.history_action_aware = False
    policy.history_forward_interval = 5
    policy.history_turn_interval = 2
    policy.lavira_runtime_enabled = True
    policy.lavira_source_prompt_alignment = False
    policy.lavira_runtime_cfg = SimpleNamespace(
        initial_scan_turns=0,
        max_steps_to_target=15,
        target_reached_threshold_m=0.75,
        hfov_deg=79.0,
        camera_height=0.88,
        map_frame_width=32,
        map_frame_height=32,
        layered_history=True,
        history_wp_max=8,
        layered_backtrack_radius_m=6.0,
    )
    policy.lavira_layered_history = True
    policy.lavira_history_wp_max = 8
    policy.lavira_backtrack_radius_m = 6.0
    policy.lavira_backtrack_second_chance = True
    policy.grounded_sam_enabled = True
    policy._grounded_sam = SimpleNamespace(
        segment_classes=lambda *_args, **_kwargs: {}
    )
    policy.stop_double_check_enabled = False
    policy.json_retry_enabled = False
    policy._parse_fail_debug_printed = 0
    policy._parse_fail_debug_limit = 0
    policy._per_env_cache = {}
    policy._lavira_runtime = {}
    policy._grounded_sam_episode_counts = {}
    policy._lavira_map_episode_counts = {}
    policy.lavira_map_visualize = False
    policy._action_stats_flush_every = 0
    policy._wm_state_collector = None
    policy.processor = _Processor()
    policy._save_lavira_map_snapshot = lambda *_args, **_kwargs: None
    policy._write_lavira_projection_audit = lambda *_args, **_kwargs: None
    policy._save_grounded_sam_diagnostic = lambda *_args, **_kwargs: None
    grounded_targets = []

    def _refine_target(parsed, _views, *, env_i=0):
        assert env_i == 0
        grounded_targets.append(parsed.target)
        return GroundedSAMRefineResult(
            detected=detected,
            point_2d=None,
            bbox_2d=detected_bbox,
            label="doorway" if detected else "",
            confidence=0.9 if detected else 0.0,
            fallback_reason=None if detected else "no_boxes",
        )

    policy._refine_waypoint_with_grounded_sam = _refine_target
    policy._batch_generate = lambda _prompts, _images: (
        [_json(action="navigate to forward", target="doorway", stop=stop)],
        [torch.tensor([[1]], dtype=torch.long)],
    )

    controller = policy._get_lavira_runtime(0)
    controller.state = NavigationState.NEED_DECISION
    monkeypatch.setattr(
        controller, "_project_traversible_target", lambda *_args, **_kwargs: (2.0, 0.0)
    )
    monkeypatch.setattr(
        controller.map, "fmm_action", lambda *_args, **_kwargs: ACTION_FORWARD
    )
    obs = {
        "main_images": np.zeros((1, 32, 32, 3), dtype=np.uint8),
        "extra_view_images": np.zeros((1, 3, 32, 32, 3), dtype=np.uint8),
        "wrist_images": np.full((1, 4, 32, 32, 1), 2.0, dtype=np.float32),
        "states": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "task_descriptions": ["Walk through the doorway."],
    }

    actions, _ = policy.predict_action_batch(obs)

    expected_action = (
        ACTION_FORWARD
        if stop
        else ACTION_PARSE_OK_HAS_BBOX_BASE + ACTION_FORWARD
    )
    assert int(actions[0, 0]) == expected_action
    assert controller.state == NavigationState.NAVIGATING
    assert controller.goal_xz == (2.0, 0.0)
    assert controller._going_to_stop is stop
    assert controller.audit_request["point_2d"] is None
    assert controller.audit_request["bbox_2d"] == expected_bbox
    assert grounded_targets == ["doorway"]
    assert controller.current_waypoint_id == 0
    assert controller.waypoints.get(0) is not None
    assert len(policy._get_cache(0).waypoints) == 1
    assert policy._get_cache(0).waypoints[0].point_2d == [
        500.0,
        expected_bbox[3],
    ]

def test_map_visualization_uses_active_episode_reference_path():
    from rlinf.models.embodiment.qwen_nav.policy_diagnostics import (
        PolicyDiagnosticsMixin,
    )

    diagnostics = object.__new__(PolicyDiagnosticsMixin)
    path_824 = np.asarray([[1.0, 0.0, 2.0], [2.0, 0.0, 3.0]], dtype=np.float32)
    path_1301 = np.asarray([[14.0, 0.0, -13.0], [24.0, 0.0, -13.0]], dtype=np.float32)
    diagnostics._lavira_map_gt_reference_paths = {
        "824": path_824,
        "1301": path_1301,
    }
    diagnostics._lavira_map_default_gt_reference_path = path_824
    diagnostics._lavira_map_episode_ids_by_env = {0: "1301"}

    selected = diagnostics._lavira_map_reference_path_for_env(0)

    assert selected is path_1301

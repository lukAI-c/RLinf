from rlinf.models.embodiment.qwen_nav.action_parser import (
    ACTION_FORWARD,
    ACTION_STOP,
    parse_lavira_waypoint_json,
)
from rlinf.models.embodiment.qwen_nav.prompts import (
    build_waypoint_user_content_text,
)


def _json(action="navigate to forward", stop=False, bbox=None, point=None):
    bbox = [100, 200, 500, 800] if bbox is None else bbox
    point = [300, 700] if point is None else point
    stop_txt = "true" if stop else "false"
    return f"""{{
      "progress_analysis": "near hallway",
      "reasoning_plan_action": "go forward",
      "planning": "continue",
      "action": "{action}",
      "stop": {stop_txt},
      "stair": false,
      "reasoning_bbox_point": "FORWARD view shows target",
      "bbox_2d": {bbox},
      "point_2d": {point},
      "target": "hallway"
    }}"""


def test_waypoint_navigate_bbox_point_ok():
    parsed = parse_lavira_waypoint_json(_json())
    assert parsed.ok
    assert parsed.action_type == "NAVIGATE"
    assert parsed.actions == [ACTION_FORWARD]
    assert parsed.bbox_2d == [100.0, 200.0, 500.0, 800.0]
    assert parsed.point_2d == [300.0, 700.0]


def test_waypoint_point_outside_bbox_invalid():
    parsed = parse_lavira_waypoint_json(_json(point=[900, 900]))
    assert not parsed.ok
    assert parsed.err == "point_outside_bbox"


def test_waypoint_point_out_of_range_invalid():
    parsed = parse_lavira_waypoint_json(_json(point=[1001, 500]))
    assert not parsed.ok
    assert parsed.err == "point_out_of_range"


def test_waypoint_backtrack_ok():
    parsed = parse_lavira_waypoint_json(
        _json(action="backtrack to 7", bbox=[0, 0, 0, 0], point=[0, 0])
    )
    assert parsed.ok
    assert parsed.action_type == "BACKTRACK"
    assert parsed.waypoint_id == 7
    assert parsed.actions == []


def test_waypoint_stop_ok():
    parsed = parse_lavira_waypoint_json(_json(stop=True))
    assert parsed.ok
    assert parsed.action_type == "STOP"
    assert parsed.actions == [ACTION_STOP]


def test_waypoint_missing_field_invalid():
    parsed = parse_lavira_waypoint_json('{"action": "navigate to forward"}')
    assert not parsed.ok
    assert parsed.err.startswith("missing_fields:")


def test_waypoint_behind_invalid():
    parsed = parse_lavira_waypoint_json(_json(action="navigate to behind"))
    assert not parsed.ok
    assert parsed.err == "unknown_direction:navigate to behind"


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
    assert '`point_2d` MUST lie strictly inside bbox_2d' in text
    assert 'reasoning_bbox_point="(unused: backtracking)"' in text
    assert text.count("<image>") == 11  # arrival + after-turn + 5 continuous + 4 current

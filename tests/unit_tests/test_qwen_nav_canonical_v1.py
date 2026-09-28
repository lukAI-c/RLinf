import json

from rlinf.models.embodiment.qwen_nav.action_parser import (
    ACTION_FORWARD,
    ACTION_STOP,
    parse_lavira_canonical_waypoint_json,
)
from rlinf.models.embodiment.qwen_nav.canonical_targets import (
    AREA_CLASSES,
    DINO_TARGET_VOCAB,
    PORTAL_CLASSES,
    map_target_for_grounding,
    normalize_target,
    region_for_box,
    target_geometry_kind,
)
from rlinf.models.embodiment.qwen_nav.grounded_sam import waypoint_bbox_scene_metrics
from rlinf.models.embodiment.qwen_nav.prompts import (
    build_waypoint_user_content_text,
)


def _canonical(**updates):
    obj = {
        "progress_analysis": "near the kitchen",
        "reasoning_plan_action": "the opening is visible",
        "planning": "leave the kitchen",
        "action": "navigate to forward",
        "stop": False,
        "stair": False,
        "target_class": "archway",
        "target_region": "center",
    }
    obj.update(updates)
    return json.dumps(obj, separators=(",", ":"))


def test_canonical_parser_normalizes_exact_alias():
    parsed = parse_lavira_canonical_waypoint_json(
        _canonical(target_class="archway exit", target_region="left")
    )
    assert parsed.ok
    assert parsed.actions == [ACTION_FORWARD]
    assert parsed.target == "archway"
    assert parsed.raw_target == "archway exit"
    assert parsed.target_region == "left"


def test_canonical_parser_rejects_unreviewed_phrase():
    parsed = parse_lavira_canonical_waypoint_json(_canonical(target_class="opening"))
    assert not parsed.ok
    assert parsed.err == "unknown_target:opening"


def test_canonical_stop_requires_empty_target():
    parsed = parse_lavira_canonical_waypoint_json(
        _canonical(stop=True, target_class="", target_region="any")
    )
    assert parsed.ok
    assert parsed.actions == [ACTION_STOP]

    malformed = parse_lavira_canonical_waypoint_json(
        _canonical(stop=True, target_class="rug", target_region="any")
    )
    assert not malformed.ok


def test_canonical_prompt_contains_single_json_braces_and_schema():
    text = build_waypoint_user_content_text("find the doorway", canonical=True)
    assert "target_class" in text
    assert "target_region" in text
    assert "{{" not in text
    assert "}}" not in text


def test_target_alias_and_region_helpers_are_deterministic():
    assert normalize_target("white floor rug") == ("rug", "alias")
    assert "archway" in DINO_TARGET_VOCAB
    assert region_for_box([0, 0, 300, 500]) == "left"
    assert region_for_box([350, 0, 650, 500]) == "center"
    assert region_for_box([700, 0, 1000, 500]) == "right"


def test_only_localizable_openings_use_portal_geometry():
    assert PORTAL_CLASSES == {"archway", "doorway", "door"}
    assert "hallway" not in PORTAL_CLASSES
    assert "corridor" not in PORTAL_CLASSES


def test_free_form_geometry_routing_keeps_portals_ahead_of_area_words():
    assert AREA_CLASSES == {"hallway", "corridor", "floor", "path"}
    assert target_geometry_kind("hallway entrance") == "area"
    assert target_geometry_kind("corridor floor") == "area"
    assert target_geometry_kind("doorway to hallway") == "portal"
    assert target_geometry_kind("archway to hallway") == "portal"
    assert target_geometry_kind("white floor rug") == "object"


def test_free_form_grounding_mapping_uses_visible_nouns_without_fuzzy_matching():
    cases = {
        "archway exit": ("archway", "portal", "alias"),
        "stone archway opening": ("archway", "portal", "head_noun"),
        "kitchen exit doorway": ("doorway", "portal", "head_noun"),
        "hallway floor": ("hallway", "area", "alias"),
        "floor passage between pillars": ("path", "area", "alias"),
        "White FLOOR rug!": ("rug", "object", "alias"),
    }
    for raw_target, expected in cases.items():
        mapping = map_target_for_grounding(raw_target)
        assert (
            mapping.dino_query,
            mapping.geometry_kind,
            mapping.status,
        ) == expected

    unknown = map_target_for_grounding("exercise room entrance")
    assert unknown.canonical_target is None
    assert unknown.dino_query == "exercise room entrance"
    assert unknown.geometry_kind == "unknown"
    assert unknown.status == "unknown"


def test_waypoint_scene_bbox_filter_rejects_large_or_three_edge_boxes():
    local = waypoint_bbox_scene_metrics([200, 100, 700, 800])
    large = waypoint_bbox_scene_metrics([100, 100, 900, 950])
    three_edges = waypoint_bbox_scene_metrics([0, 0, 1000, 600])

    assert not local["scene_region"]
    assert large["area_ratio"] > 0.65
    assert large["scene_region"]
    assert three_edges["edge_count"] == 3
    assert three_edges["scene_region"]

from tools.lavira_world_model_data.run_candidate_target_v5 import (
    localized_candidates, mapper_gate_summary, unique_labels,
)


def test_filters_scene_regions_and_preserves_confidence_order() -> None:
    candidates = [
        {"label": "hallway", "confidence": 0.9, "bbox_2d": [0, 0, 1000, 1000], "scene_region": True},
        {"label": "doorway", "confidence": 0.8, "bbox_2d": [100, 100, 300, 900], "scene_region": False, "area_ratio": 0.16, "edge_count": 0},
        {"label": "doorway", "confidence": 0.7, "bbox_2d": [400, 100, 600, 900], "scene_region": False, "area_ratio": 0.16, "edge_count": 0},
    ]
    values = localized_candidates(candidates)
    assert [value["confidence"] for value in values] == [0.8, 0.7]
    assert unique_labels(values) == ["doorway"]


def test_mapper_gate_summary_separates_backoff_and_fmm_availability() -> None:
    rows = [
        {"source_mapper_gate": {"strict": {
            "direct_endpoint_traversible": True,
            "goal_xz": [1.0, 2.0],
            "accepted_reason": "initial_traversible",
            "goal_distance_m": 2.0,
            "fmm_action": 1,
            "fmm_audit": {"fmm_distance_at_agent": 4.0},
        }}},
        {"source_mapper_gate": {"strict": {
            "direct_endpoint_traversible": False,
            "goal_xz": [1.0, 2.0],
            "accepted_reason": "source_depth_exhausted",
            "goal_distance_m": 0.2,
            "fmm_action": None,
            "fmm_audit": {},
        }}},
    ]
    result = mapper_gate_summary(rows, "strict")
    assert result["candidate_rows"] == 2
    assert result["direct_endpoint_traversible"] == 1
    assert result["source_depth_exhausted"] == 1
    assert result["near_agent_below_0_75m"] == 1
    assert result["fmm_action_available"] == 1

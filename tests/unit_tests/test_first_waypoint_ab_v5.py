import pytest

from tools.lavira_world_model_data.run_first_waypoint_ab_v5 import (
    episode_id_from_decision_id,
    summarize_pairs,
)


def test_episode_id_from_v4b_decision() -> None:
    assert episode_id_from_decision_id("episode_v4b-1122_step_0000") == "v4b-1122"


def test_pair_summary_reports_directional_progress_gain() -> None:
    rows = [{
        "baseline": {"summary": {
            "along_track_delta_m": 0.2, "collision_count": 2,
            "waypoint_reached": False,
        }},
        "wm": {"summary": {
            "along_track_delta_m": 0.8, "collision_count": 0,
            "waypoint_reached": True,
        }},
    }]
    result = summarize_pairs(rows)
    assert result["pairs"] == 1
    assert result["mean_paired_progress_gain_m"] == pytest.approx(0.6)
    assert result["wm_better"] == 1
    assert result["wm_collision_count"] == 0

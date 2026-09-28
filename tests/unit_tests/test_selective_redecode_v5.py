from __future__ import annotations

from PIL import Image

import tools.lavira_world_model_data.run_selective_redecode_v5 as module
from tools.lavira_world_model_data.run_selective_redecode_v5 import conflict_rows, parse_redecode


def test_selects_only_accepted_direction_conflicts() -> None:
    source = {"rows": [
        {"wm_rerank_accepted": True, "baseline": {"sector": "left"}, "recommended_sector": "right"},
        {"wm_rerank_accepted": True, "baseline": {"sector": "right"}, "recommended_sector": "right"},
        {"wm_rerank_accepted": False, "baseline": {"sector": "left"}, "recommended_sector": "right"},
    ]}
    assert len(conflict_rows(source)) == 1


def test_parses_direction_compliant_seven_field_output() -> None:
    decoded = '''{"progress_analysis":"start","reasoning_plan_action":"door visible","planning":"approach door","action":"navigate to right","stop":false,"stair":false,"target":"doorway"}'''
    result = parse_redecode(decoded, "right")
    assert result["parse_ok"] is True
    assert result["direction_compliant"] is True
    assert result["target"] == "doorway"
    assert result["target_generation_eligible"] is True


def test_rejects_valid_json_with_wrong_direction() -> None:
    decoded = '''{"progress_analysis":"start","reasoning_plan_action":"door visible","planning":"approach door","action":"navigate to left","stop":false,"stair":false,"target":"doorway"}'''
    assert parse_redecode(decoded, "right")["direction_compliant"] is False


def test_selected_view_conditioning_duplicates_only_wm_sector_image(monkeypatch) -> None:
    images = [Image.new("RGB", (2, 2), color=(index, 0, 0)) for index in range(4)]
    messages = [
        {"role": "system", "content": []},
        {"role": "user", "content": [{"type": "image", "image": image} for image in images]},
    ]
    monkeypatch.setattr(module, "build_messages", lambda row, guided: (messages, images))
    result_messages, result_images = module.constrained_messages(
        {}, "behind", selected_view_only=True
    )
    selected = images[2]
    assert all(image is selected for image in result_images)
    assert all(
        item["image"] is selected
        for item in result_messages[1]["content"] if item["type"] == "image"
    )

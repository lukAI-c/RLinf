from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from rlinf.models.embodiment.qwen_nav.prompts import (
    build_termination_shadow_text,
)


def _load_analyzer_module():
    script = Path(__file__).parents[2] / "scripts" / "analyze_termination_shadow.py"
    spec = importlib.util.spec_from_file_location("analyze_termination_shadow", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_termination_shadow_prompt_is_neutral() -> None:
    prompt = build_termination_shadow_text("Exit the kitchen and wait in the gym.")

    assert "You have decided to STOP" not in prompt
    assert "Ignore any navigation action" in prompt
    assert prompt.endswith("true or false.")


def test_shadow_analyzer_detects_separable_representation(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(8), 4)
    labels = np.tile(np.array([0, 0, 1, 1]), 8)
    hidden = rng.normal(0.0, 0.1, size=(labels.size, 12)).astype(np.float32)
    hidden[:, 0] += np.where(labels == 1, 2.0, -2.0)
    probability = np.where(labels == 1, 0.9, 0.1).astype(np.float32)
    np.savez_compressed(
        tmp_path / "shadow_rank00_step000000_test.npz",
        hidden=hidden,
        probability=probability,
        prediction=probability >= 0.5,
        oracle_within_radius=labels,
        episode_id=groups,
        dtg=np.where(labels == 1, 1.0, 5.0),
        start_dtg=np.full(labels.size, 8.0),
        success_distance=np.full(labels.size, 3.0),
        eligible_clean_stop=labels,
        trial_id=np.arange(labels.size),
        decision_index=np.tile(np.arange(4), 8),
    )

    analyzer = _load_analyzer_module()
    result = analyzer.analyze(tmp_path, bootstrap_repeats=100, seed=11)

    assert result["shadow_judge"]["auc"] > 0.99
    assert result["linear_probe"]["auc"] > 0.95
    assert result["samples"] == labels.size

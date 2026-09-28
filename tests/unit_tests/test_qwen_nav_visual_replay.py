from types import SimpleNamespace

import pytest
import torch

from rlinf.models.embodiment.qwen_nav.qwen_nav_policy import QwenNavPolicy


def _policy(capacity: int = 8) -> QwenNavPolicy:
    policy = object.__new__(QwenNavPolicy)
    policy.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0)
    )
    policy._prompt_len = 6
    policy.max_new_tokens = 4
    policy._total_patches = capacity
    policy._pixel_value_dim = 3
    policy._n_images_fixed = 2
    policy.forward_input_profile_image_size = (640, 480)
    policy._compute_action_loss_mask = lambda *_args, **_kwargs: torch.ones(
        4, dtype=torch.bool
    )
    return policy


def _inputs(pixel_rows: int, grid) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "pixel_values": torch.ones(pixel_rows, 3),
        "image_grid_thw": torch.tensor(grid, dtype=torch.long),
    }


def test_visual_replay_preserves_all_processor_patches():
    policy = _policy(capacity=8)
    result = policy._build_forward_inputs_for_env(
        _inputs(6, [[1, 2, 2], [1, 1, 2]]),
        torch.tensor([[7, 8]]),
    )

    assert result["pixel_values"].shape == (1, 8, 3)
    assert torch.equal(result["pixel_values"][0, :6], torch.ones(6, 3))
    assert not result["pixel_values"][0, 6:].any()
    assert result["image_grid_thw"][0].prod(dim=-1).sum().item() == 6


def test_visual_replay_rejects_pixel_grid_mismatch():
    policy = _policy(capacity=8)
    with pytest.raises(ValueError, match="pixel/grid mismatch"):
        policy._build_forward_inputs_for_env(
            _inputs(5, [[1, 2, 2], [1, 1, 2]]),
            torch.tensor([[7, 8]]),
        )


def test_visual_replay_rejects_capacity_truncation():
    policy = _policy(capacity=5)
    with pytest.raises(ValueError, match="capacity is smaller"):
        policy._build_forward_inputs_for_env(
            _inputs(6, [[1, 2, 2], [1, 1, 2]]),
            torch.tensor([[7, 8]]),
        )

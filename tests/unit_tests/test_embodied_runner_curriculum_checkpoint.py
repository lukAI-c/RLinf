import torch

from rlinf.runners.embodied_runner import EmbodiedRunner


def test_curriculum_checkpoint_request_detects_any_env_rank():
    results = [
        {"rft_curriculum_advance_pending": torch.tensor([0.0])},
        {"rft_curriculum_advance_pending": torch.tensor([1.0])},
    ]

    assert EmbodiedRunner._curriculum_checkpoint_requested(results)


def test_curriculum_checkpoint_request_ignores_missing_or_zero_values():
    results = [
        {},
        None,
        {"rft_curriculum_advance_pending": torch.tensor([0.0])},
    ]

    assert not EmbodiedRunner._curriculum_checkpoint_requested(results)

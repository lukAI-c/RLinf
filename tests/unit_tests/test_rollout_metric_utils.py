import math

import torch

from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import compute_rollout_metrics


class _CpuPlatform:
    @staticmethod
    def current_device():
        return torch.device("cpu")


def test_advantage_diagnostics_use_valid_finite_decisions(monkeypatch):
    monkeypatch.setattr(Worker, "torch_platform", _CpuPlatform)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None: tensor)

    metrics = compute_rollout_metrics(
        {
            "advantages": torch.tensor([[[1.0], [-1.0], [10.0], [float("nan")]]]),
            "loss_mask": torch.tensor([[[True], [True], [False], [True]]]),
        }
    )

    assert metrics["advantages_std"] == 1.0
    assert metrics["advantages_abs_mean"] == 1.0
    assert metrics["advantages_max_abs_to_mean_abs"] == 1.0
    assert metrics["advantages_effective_sample_ratio"] == 1.0
    assert metrics["advantages_positive_fraction"] == 0.5
    assert metrics["advantages_negative_fraction"] == 0.5
    assert math.isclose(
        metrics["advantages_nonfinite_fraction"], 1.0 / 3.0, rel_tol=1e-6
    )


def test_zero_advantages_produce_finite_concentration_metrics(monkeypatch):
    monkeypatch.setattr(Worker, "torch_platform", _CpuPlatform)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None: tensor)

    metrics = compute_rollout_metrics(
        {
            "advantages": torch.zeros(2, 2, 1),
            "loss_mask": torch.ones(2, 2, 1, dtype=torch.bool),
        }
    )

    assert metrics["advantages_std"] == 0.0
    assert metrics["advantages_max_abs_to_mean_abs"] == 0.0
    assert metrics["advantages_effective_sample_ratio"] == 0.0
    assert metrics["advantages_nonfinite_fraction"] == 0.0

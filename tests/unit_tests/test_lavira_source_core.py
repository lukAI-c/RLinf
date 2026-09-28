"""Contract tests for the pinned LHX mapper/FMM integration path."""

import numpy as np

from rlinf.models.embodiment.qwen_nav.action_parser import ACTION_FORWARD
from rlinf.models.embodiment.qwen_nav.lavira_runtime.navigation_controller import (
    LaviraNavigationController,
)
from rlinf.third_party.lavira_rft.loader import SOURCE_COMMIT
from rlinf.third_party.lavira_rft.source_core import LaviraSourceCore


def test_source_mapper_and_fmm_run_without_sparse_planner(tmp_path):
    core = LaviraSourceCore(
        device="cpu",
        results_dir=tmp_path / "source",
        hfov_deg=79.0,
        camera_height=0.88,
        frame_width=32,
        frame_height=32,
    )
    depth = np.full((32, 32), 2.0, dtype=np.float32)
    masks = {"floor": np.ones((32, 32), dtype=np.float32)}

    core.configure_fmm_output(tmp_path / "run", episode_id=609, trial_id=1)
    core.update(depth, 0.0, 0.0, 0.0, masks)
    traversible = core.rebuild_traversible()
    action = core.fmm_action(0.0, 0.0, 0.0, 1.0, 0.0)

    assert SOURCE_COMMIT == "b3d6c35067ad731a16ec096b698ddfc2f5c91af6"
    assert core.channels.shape == (6, 480, 480)
    assert core.detected_classes.order == ["floor", "unknow"]
    assert traversible.shape == (480, 480)
    assert action == ACTION_FORWARD
    assert core.last_fmm_audit["status"] == "source_policy"
    assert (
        tmp_path / "run" / "fmm_fields" / "eps_609_trial_001" / "step-1.png"
    ).is_file()


def test_controller_reset_reuses_source_mapper_allocation():
    from types import SimpleNamespace

    controller = LaviraNavigationController(
        SimpleNamespace(map_backend="source", map_device="cpu")
    )
    mapper = controller.map.mapper
    controller.reset()
    assert controller.map.mapper is mapper


def test_fmm_audit_clamps_only_its_distance_sample_at_map_boundary(tmp_path):
    core = LaviraSourceCore(
        device="cpu",
        results_dir=tmp_path / "source",
        hfov_deg=79.0,
        camera_height=0.88,
        frame_width=32,
        frame_height=32,
    )
    depth = np.full((32, 32), 2.0, dtype=np.float32)
    masks = {"floor": np.ones((32, 32), dtype=np.float32)}
    core.update(depth, 0.0, 0.0, 0.0, masks)
    core.rebuild_traversible()
    core.full_pose = np.asarray([25.9, 12.0, 0.0], dtype=np.float32)
    received_poses = []

    def _get_action(full_pose, *_args):
        received_poses.append(full_pose.copy())
        core.policy.fmm_dist = np.arange(480 * 480, dtype=np.float32).reshape(480, 480)
        return ACTION_FORWARD

    core.policy._get_action = _get_action
    action = core.fmm_action(0.0, 0.0, 0.0, 1.0, 0.0)

    assert action == ACTION_FORWARD
    np.testing.assert_array_equal(
        received_poses[0], np.asarray([25.9, 12.0, 0.0], dtype=np.float32)
    )
    assert core.last_fmm_audit["agent_map_rc_raw"] == [240, 518]
    assert core.last_fmm_audit["agent_map_rc_sampled"] == [240, 479]
    assert core.last_fmm_audit["agent_map_out_of_bounds"] is True
    assert core.last_fmm_audit["fmm_distance_at_agent"] == float(
        core.policy.fmm_dist[240, 479]
    )


def test_source_fmm_preview_detects_disconnected_goal_without_mutating_policy(tmp_path):
    core = LaviraSourceCore(
        device="cpu", results_dir=tmp_path / "source",
        hfov_deg=79.0, camera_height=0.88,
        map_size_cm=1000, resolution_cm=50,
        frame_width=32, frame_height=32,
    )
    core._traversible = np.ones((20, 20), dtype=bool)
    core._traversible[:, 10] = False
    core.origin_x = core.origin_z = core.origin_yaw = 0.0
    core.center = 10
    core.cells_per_m = 2.0
    # Source full_pose is [x_m, y_m, heading_deg], sampled as [row=y, col=x].
    core.full_pose = np.asarray([2.5, 5.0, 0.0], dtype=np.float32)
    prior_fmm = core.policy.fmm_dist.copy()

    connected = core.preview_fmm_reachability(0.0, 0.0, -2.0, 0.0)
    disconnected = core.preview_fmm_reachability(0.0, 0.0, 2.0, 0.0)

    assert connected["reachable"] is True
    assert connected["status"] == "reachable"
    assert disconnected["reachable"] is False
    assert disconnected["status"] == "disconnected"
    np.testing.assert_array_equal(core.policy.fmm_dist, prior_fmm)

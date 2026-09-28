import math
import json

import numpy as np
import torch

from rlinf.envs.genark import genesis_backend
from rlinf.envs.genark.genesis_backend import (
    GenesisLocalBackend,
    _horizontal_to_vertical_fov_deg,
    _linear_rgb_to_srgb,
    _load_mesh_rgb_calibration,
    _rgb_color_transform,
)


class _DepthCamera:
    def __init__(self, backend):
        self.backend = backend

    def render(self, *, rgb, depth, segmentation, force_render):
        assert not rgb and depth and not segmentation
        value = float(self.backend._render_index)
        frame = torch.full(
            (self.backend._active_count, self.backend._cam_h, self.backend._cam_w),
            value,
            dtype=torch.float32,
        )
        return None, frame, None, None


class _RgbCamera:
    def __init__(self, rgb):
        self.rgb = rgb

    def render(self, *, rgb, depth, segmentation, force_render):
        assert rgb and not depth and not segmentation
        return self.rgb, None, None, None


def test_habitat_hfov_is_converted_to_matching_genesis_vfov():
    vfov = _horizontal_to_vertical_fov_deg(79.0, 640, 480)

    assert math.isclose(vfov, 63.453048374758716, abs_tol=1e-12)
    genesis_focal_length = 480.0 / (2.0 * math.tan(math.radians(vfov) / 2.0))
    habitat_focal_length = 640.0 / (2.0 * math.tan(math.radians(79.0) / 2.0))
    assert math.isclose(genesis_focal_length, habitat_focal_length, abs_tol=1e-12)


def test_hfov_conversion_rejects_invalid_camera_contract():
    for width, height, hfov in ((0, 480, 79.0), (640, 0, 79.0), (640, 480, 180.0)):
        try:
            _horizontal_to_vertical_fov_deg(hfov, width, height)
        except ValueError:
            continue
        raise AssertionError((width, height, hfov))


def test_linear_rgb_to_srgb_uses_standard_transfer_curve():
    linear = torch.tensor([0.0, 0.0031308, 0.18, 1.0], dtype=torch.float32)
    encoded = _linear_rgb_to_srgb(linear)

    torch.testing.assert_close(
        encoded,
        torch.tensor([0.0, 0.04044994, 0.46135613, 1.0]),
        atol=1e-6,
        rtol=0.0,
    )


def test_rgb_color_transform_defaults_and_validates_shape():
    default_cfg = type("Cfg", (), {})()
    matrix, bias = _rgb_color_transform(default_cfg)
    torch.testing.assert_close(matrix, torch.eye(3))
    torch.testing.assert_close(bias, torch.zeros(3))

    calibrated_cfg = type("Cfg", (), {
        "rgb_color_matrix": ((1.1, 0.1, 0.0), (0.0, 0.9, 0.0), (0.0, 0.0, 1.0)),
        "rgb_color_bias": (-0.01, 0.0, 0.02),
    })()
    matrix, bias = _rgb_color_transform(calibrated_cfg)
    torch.testing.assert_close(matrix, torch.tensor(calibrated_cfg.rgb_color_matrix))
    torch.testing.assert_close(bias, torch.tensor(calibrated_cfg.rgb_color_bias))

    invalid_cfg = type("Cfg", (), {"rgb_color_matrix": ((1.0,),), "rgb_color_bias": (0.0,)})()
    try:
        _rgb_color_transform(invalid_cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid RGB transform shape was accepted")


def test_mesh_rgb_calibration_loads_validated_sidecar(tmp_path):
    sidecar = tmp_path / "scene.render.json"
    sidecar.write_text(json.dumps({
        "light_scale": 2.0,
        "rgb_color_matrix": [[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 1.0]],
        "rgb_color_bias": [-0.01, 0.0, 0.02],
    }))

    exposure, matrix, bias = _load_mesh_rgb_calibration(str(sidecar))

    assert exposure == 2.0
    torch.testing.assert_close(
        matrix,
        torch.tensor([[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 1.0]]),
    )
    torch.testing.assert_close(bias, torch.tensor([-0.01, 0.0, 0.02]))


def test_render_color_pipeline_matches_calibration_order():
    backend = object.__new__(GenesisLocalBackend)
    raw = torch.tensor(
        [[[[0.02, 0.04, 0.08], [0.10, 0.10, 0.10]]]], dtype=torch.float32
    )
    backend._cam = _RgbCamera(raw)
    backend._light_scale = 2.0
    backend._rgb_color_matrix = torch.tensor(
        [[1.1, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    backend._rgb_color_bias = torch.tensor([-0.01, 0.0, 0.02], dtype=torch.float32)
    backend._rgb_linear_to_srgb = True

    rendered = backend._render_and_scale(active_slot_count=1)
    linear = torch.einsum(
        "...c,dc->...d", raw * backend._light_scale, backend._rgb_color_matrix
    ) + backend._rgb_color_bias
    expected = torch.round(_linear_rgb_to_srgb(linear.clamp(0.0, 1.0)) * 255.0).byte()
    torch.testing.assert_close(rendered, expected)


def test_atomic_panorama_preserves_pose_and_uses_lhx_heading_order(monkeypatch):
    backend = object.__new__(GenesisLocalBackend)
    backend._num_envs = 2
    backend._cam_h = 2
    backend._cam_w = 3
    backend._step_turn = math.radians(30.0)
    backend._depth_min = 0.1
    backend._depth_max = 5.0
    backend._cam_pos_t = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32
    )
    backend._cam_yaw_t = torch.tensor([0.25, -0.5], dtype=torch.float32)
    backend._active_count = 2
    backend._render_index = 0
    backend._cam = _DepthCamera(backend)
    camera_updates = []

    def fake_update_camera(camera, positions, yaws):
        camera_updates.append((positions.clone(), yaws.clone()))

    def fake_render(active_slot_count):
        backend._render_index += 1
        return torch.full(
            (active_slot_count, backend._cam_h, backend._cam_w, 3),
            backend._render_index,
            dtype=torch.uint8,
        )

    monkeypatch.setattr(genesis_backend, "_update_camera", fake_update_camera)
    monkeypatch.setattr(backend, "_render_and_scale", fake_render)
    original_pos = backend._cam_pos_t.clone()
    original_yaw = backend._cam_yaw_t.clone()

    rgb, depth, yaw = backend.render_panorama_with_depth(2)

    assert rgb.shape == (2, 12, 2, 3, 3)
    assert depth.shape == (2, 12, 2, 3)
    assert yaw.shape == (2, 12)
    assert len(camera_updates) == 13
    expected = original_yaw[:, None] + torch.arange(1, 13) * math.radians(30.0)
    np.testing.assert_allclose(yaw, expected.numpy(), atol=1e-6)
    torch.testing.assert_close(camera_updates[-1][1], original_yaw)
    torch.testing.assert_close(backend._cam_pos_t, original_pos)
    torch.testing.assert_close(backend._cam_yaw_t, original_yaw)
    assert np.all(rgb[:, 0] == 1)
    assert np.all(rgb[:, 11] == 12)
    assert np.all(depth >= 0.1)
    assert np.all(depth <= 5.0)

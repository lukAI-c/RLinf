import json
import math
from types import SimpleNamespace

import numpy as np
import torch

from rlinf.envs.genark.genark_env import GenarkVecEnv


class _PanoramaBackend:
    def __init__(self):
        self.device = torch.device("cpu")
        self.cam_pos = torch.zeros(1, 3)
        self.cam_yaw = torch.zeros(1)
        self.current_tri_idx = torch.zeros(1, dtype=torch.long)
        self.physics_calls = []

    def load_scene(self, scene_id, n_active_envs):
        self.scene_id = scene_id

    def set_agent_poses(self, env_idx, positions, yaws):
        for local, index in enumerate(env_idx):
            self.cam_pos[index] = positions[local]
            self.cam_yaw[index] = yaws[local]

    def cam_pos_hab(self, camera_height):
        return torch.stack(
            (
                self.cam_pos[:, 0],
                self.cam_pos[:, 2] - camera_height,
                -self.cam_pos[:, 1],
            ),
            dim=1,
        )

    def step_physics(self, actions, active_mask, active_slot_count):
        self.physics_calls.append((actions.clone(), active_mask.clone()))

    def render_main_with_depth(self, active_slot_count):
        return (
            np.zeros((1, 2, 3, 3), dtype=np.uint8),
            np.ones((1, 2, 3), dtype=np.float32),
        )

    def render_4dir_with_depth(self, active_slot_count):
        return (
            np.zeros((1, 3, 2, 3, 3), dtype=np.uint8),
            np.ones((1, 3, 2, 3), dtype=np.float32),
        )

    def render_panorama_with_depth(self, active_slot_count):
        rgb = np.zeros((1, 12, 2, 3, 3), dtype=np.uint8)
        depth = np.zeros((1, 12, 2, 3), dtype=np.float32)
        yaw = np.zeros((1, 12), dtype=np.float32)
        front = float(self.cam_yaw[0])
        for index in range(12):
            rgb[:, index] = index + 1
            depth[:, index] = index + 1
            yaw[:, index] = front + (index + 1) * math.radians(30.0)
        return rgb, depth, yaw

    def is_scene_healthy(self):
        return True


def test_action_7_is_atomic_panorama_and_reaches_policy_schema(tmp_path):
    episode = {
        "episode_id": 259,
        "scene_id": "mp3d/TbHJrupSAjP/TbHJrupSAjP.glb",
        "start_position": [6.4071698, 3.6244001, -9.8953104],
        "start_rotation": [0.0, -0.95402475, 0.0, -0.29972782],
        "goals": [{"position": [5.14005, 3.6212056, -4.26937]}],
        "instruction": {"instruction_text": "Exit the bathroom."},
        "reference_path": [
            [6.4071698, 3.6244001, -9.8953104],
            [5.14005, 3.6212056, -4.26937],
        ],
    }
    episodes_path = tmp_path / "episodes.json"
    episodes_path.write_text(json.dumps([episode]))
    cfg = SimpleNamespace(
        seed=0,
        init_params=SimpleNamespace(
            episodes_file=str(episodes_path), scene_datasets=str(tmp_path)
        ),
        video_cfg=SimpleNamespace(video_base_dir=str(tmp_path / "video")),
        genesis_backend="local",
        cam_res=(3, 2),
        camera_height=0.88,
        fov=79,
        enable_depth_obs=True,
        enable_4dir_render=True,
        enable_4dir_depth_obs=True,
        max_episode_steps=300,
        auto_reset=False,
        is_eval=True,
        cyclic_episode_sampling=False,
        reward_profile="nav",
    )
    backend = _PanoramaBackend()
    env = GenarkVecEnv(
        cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        backend=backend,
    )
    env.reset()
    position_before = backend.cam_pos.clone()
    yaw_before = backend.cam_yaw.clone()

    obs, reward, terminated, truncated, _ = env.step(torch.tensor([7]))

    actions, active_mask = backend.physics_calls[-1]
    assert actions.tolist() == [0]
    assert active_mask.tolist() == [False]
    torch.testing.assert_close(backend.cam_pos, position_before)
    torch.testing.assert_close(backend.cam_yaw, yaw_before)
    assert env._elapsed_steps.tolist() == [12]
    assert reward.tolist() == [0.0]
    assert not bool(terminated[0])
    assert not bool(truncated[0])
    assert obs["scan_images"].shape == (1, 12, 2, 3, 3)
    assert obs["scan_depth_images"].shape == (1, 12, 2, 3, 1)
    assert obs["scan_states"].shape == (1, 12, 3)
    assert obs["scan_valid"].tolist() == [True]
    assert obs["episode_active"].tolist() == [True]
    assert obs["episode_ids"] == ["259"]
    assert obs["trial_ids"] == [1]
    assert torch.all(obs["main_images"] == 12)
    assert torch.all(obs["extra_view_images"][:, 0] == 3)
    assert torch.all(obs["extra_view_images"][:, 1] == 9)
    assert torch.all(obs["extra_view_images"][:, 2] == 6)
    assert torch.all(obs["wrist_images"][:, 0] == 12)
    assert torch.all(obs["wrist_images"][:, 1] == 3)
    assert torch.all(obs["wrist_images"][:, 2] == 6)
    assert torch.all(obs["wrist_images"][:, 3] == 9)

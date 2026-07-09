# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
    Trajectory,
)
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.wrappers import RecordVideo
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import (
    copy_dict_tensor,
    split_dict,
    update_nested_cfg,
)
from rlinf.utils.placement import HybridComponentPlacement


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list = []
        self.eval_env_list = []
        self.global_step = 0

        self.last_obs_list = []
        self.last_intervened_info_list = []
        self.rollout_epoch = self.cfg.algorithm.get("rollout_epoch", 1)
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.stage_num = self.cfg.rollout.pipeline_stage_num

        self.reward_mode = self.cfg.get("reward", {}).get("reward_mode", "per_step")
        self.use_reward_model = self.cfg.get("reward", {}).get(
            "use_reward_model", False
        )
        self.use_realworld_reward = self.cfg.get("reward", {}).get(
            "standalone_realworld", False
        )
        self.use_external_reward_model = (
            self.use_reward_model and not self.use_realworld_reward
        )
        if self.use_external_reward_model:
            self.reward_weight = self.cfg.reward.get("reward_weight", 1.0)
            self.env_reward_weight = self.cfg.reward.get("env_reward_weight", 0.0)

        # Env configurations
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        train_env_cfg = self.cfg.env.get("train", None)
        eval_env_cfg = self.cfg.env.eval
        self.enable_offload = (
            train_env_cfg.get("enable_offload", False)
            if train_env_cfg is not None
            else eval_env_cfg.get("enable_offload", False)
        )
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        if not self.only_eval:
            if train_env_cfg is None:
                raise ValueError(
                    "env.train config is required when runner.only_eval=False."
                )
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )
        self.n_train_chunk_steps = 0
        if not self.only_eval:
            self.n_train_chunk_steps = (
                self.cfg.env.train.max_steps_per_rollout_epoch
                // self.cfg.actor.model.num_action_chunks
            )
        self.n_eval_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.actor_split_num = self.get_actor_split_num()

        if not self.only_eval:
            self.train_prev_done: list[torch.Tensor] = [
                torch.zeros(self.train_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
        if self.enable_eval:
            self.eval_prev_done: list[torch.Tensor] = [
                torch.zeros(self.eval_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]

    def init_worker(self):
        self.dst_rank_map = self._setup_dst_rank_map()
        self.src_rank_map = self._setup_src_rank_map()

        self.log_info(f"Env worker initialized with dst_rank_map: {self.dst_rank_map}")
        self.log_info(f"Env worker initialized with src_rank_map: {self.src_rank_map}")

        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        self.update_env_cfg()

        if not self.only_eval:
            train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
        if self.enable_eval:
            eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )

        if not self.only_eval:
            self._init_env()

    def set_global_step(self, global_step: int) -> None:
        self.global_step = int(global_step)
        for env in list(self.env_list) + list(self.eval_env_list):
            target = getattr(env, "env", env)
            if hasattr(target, "set_global_step"):
                target.set_global_step(self.global_step)

    def _ppo_loss_mask_mode_id(self) -> int:
        """Curriculum-aware token loss mask mode for qwen_nav rollout.

        0 = full response (format/schema stage)
        1 = action/stop + bbox_2d/point_2d values (geometry stage)
        2 = action/stop values only (navigation/full stage and default)
        """
        rcfg = getattr(self.cfg.env.train, "reward_curriculum", None)
        enabled = bool(getattr(rcfg, "enabled", False)) if rcfg is not None else False
        if not enabled:
            return 2
        step = int(self.global_step)
        if step < int(getattr(rcfg, "format_steps", 0)):
            return 0
        if step < int(getattr(rcfg, "geometry_steps", 0)):
            return 1
        return 2

    def update_env_cfg(self):
        if not self.only_eval:
            # train env
            train_override_cfgs = self.cfg.env.train.get("override_cfgs", None)
            if train_override_cfgs is not None:
                assert len(train_override_cfgs) > self._rank, (
                    f"{len(train_override_cfgs)=} > {self._rank=}"
                )

                general_train_override_cfg = OmegaConf.to_container(
                    self.cfg.env.train.get("override_cfg", {}), resolve=True
                )
                override_cfg = OmegaConf.to_container(
                    train_override_cfgs[self._rank], resolve=True
                ).copy()

                base_cfg = {}
                base_cfg = update_nested_cfg(base_cfg, general_train_override_cfg)
                base_cfg = update_nested_cfg(base_cfg, override_cfg)
                setattr(self.cfg.env.train, "override_cfg", OmegaConf.create(base_cfg))
        self._inject_realworld_reward_cfg(self.cfg.env.train)
        eval_override_cfgs = self.cfg.env.eval.get("override_cfgs", None)
        if eval_override_cfgs is not None:
            assert len(eval_override_cfgs) > self._rank, (
                f"{len(eval_override_cfgs)=} > {self._rank=}"
            )

            general_eval_override_cfg = OmegaConf.to_container(
                self.cfg.env.eval.get("override_cfg", {}), resolve=True
            )
            eval_override_cfg = OmegaConf.to_container(
                eval_override_cfgs[self._rank], resolve=True
            ).copy()
            base_eval_cfg = {}
            base_eval_cfg = update_nested_cfg(base_eval_cfg, general_eval_override_cfg)
            base_eval_cfg = update_nested_cfg(base_eval_cfg, eval_override_cfg)
            setattr(self.cfg.env.eval, "override_cfg", OmegaConf.create(base_eval_cfg))
        self._inject_realworld_reward_cfg(self.cfg.env.eval)

    def _inject_realworld_reward_cfg(self, env_cfg: DictConfig):
        if not (self.use_reward_model and self.use_realworld_reward):
            return
        if env_cfg.env_type != "realworld":
            return

        reward_placements = self._component_placement.get_strategy(
            "reward"
        ).get_placement(Cluster())
        assert len(reward_placements) > 0, (
            "Reward placement must contain at least one worker."
        )
        reward_placement = reward_placements[0]
        reward_hardware_ranks = self._component_placement.get_hardware_ranks("reward")
        assert len(reward_hardware_ranks) > 0, (
            "Reward placement must contain at least one hardware rank."
        )

        override_cfg = OmegaConf.to_container(
            env_cfg.get("override_cfg", {}), resolve=True
        )
        override_cfg["use_reward_model"] = True
        override_cfg["reward_worker_cfg"] = OmegaConf.to_container(
            self.cfg.reward, resolve=True
        )
        override_cfg["reward_worker_hardware_rank"] = reward_hardware_ranks[0]
        override_cfg["reward_worker_node_rank"] = reward_placement.cluster_node_rank
        override_cfg["reward_worker_node_group"] = reward_placement.node_group_label
        override_cfg["reward_image_key"] = env_cfg.main_image_key
        setattr(env_cfg, "override_cfg", OmegaConf.create(override_cfg))

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []

        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            if env_cfg.get("data_collection", None) and getattr(
                env_cfg.data_collection, "enabled", False
            ):
                from rlinf.envs.wrappers import CollectEpisode

                env = CollectEpisode(
                    env,
                    save_dir=env_cfg.data_collection.save_dir,
                    rank=self._rank,
                    num_envs=num_envs_per_stage,
                    export_format=getattr(
                        env_cfg.data_collection, "export_format", "pickle"
                    ),
                    robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
                    fps=getattr(env_cfg.data_collection, "fps", 10),
                    only_success=getattr(
                        env_cfg.data_collection, "only_success", False
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                )
            env_list.append(env)
        return env_list

    def _setup_dst_rank_map(self) -> dict[str, list[tuple[int, int]]]:
        """Compute destination rank map for this env worker.

        This mapping supports both one-to-many and many-to-one env/rollout/reward layouts.
        The returned ranks are used as communication counterparts for both sending
        env outputs and receiving results from rollout and reward workers.

        Returns:
            Destination rank map for this env worker.
            The key is the channel name (e.g. "rollout_train", "reward_train", "rollout_eval"), and the value is a ordered list of tuples of (dst_rank, batch_size).
        """
        dst_rank_map = {}
        if not self.only_eval:
            dst_rank_map = {
                "rollout_train": CommMapper.get_dst_ranks(
                    batch_size=self.cfg.env.train.total_num_envs // self.stage_num,
                    src_world_size=self._component_placement.get_world_size("env"),
                    dst_world_size=self._component_placement.get_world_size("rollout"),
                    src_rank=self._rank,
                ),
            }
            if self.cfg.get("reward", {}).get("use_reward_model", False):
                dst_rank_map.update(
                    {
                        "reward_train": CommMapper.get_dst_ranks(
                            batch_size=self.cfg.env.train.total_num_envs
                            // self.stage_num,
                            src_world_size=self._component_placement.get_world_size(
                                "env"
                            ),
                            dst_world_size=self._component_placement.get_world_size(
                                "reward"
                            ),
                            src_rank=self._rank,
                        ),
                    }
                )

        if self.enable_eval:
            dst_rank_map.update(
                {
                    "rollout_eval": CommMapper.get_dst_ranks(
                        batch_size=self.cfg.env.eval.total_num_envs // self.stage_num,
                        src_world_size=self._component_placement.get_world_size("env"),
                        dst_world_size=self._component_placement.get_world_size(
                            "rollout"
                        ),
                        src_rank=self._rank,
                    ),
                }
            )
        return dst_rank_map

    def _setup_src_rank_map(self) -> dict[str, list[tuple[int, int]]]:
        """Compute source rank map for this env worker.

        This mapping supports both one-to-many and many-to-one env/rollout/reward layouts.
        The returned ranks are used as communication counterparts for both receiving results from rollout and reward workers and sending action chunks.

        Returns:
            Source rank map for this env worker.
            The key is the channel name (e.g. "rollout_train", "reward_train", "rollout_eval"), and the value is a ordered list of tuples of (src_rank, batch_size).
        """
        src_rank_map = {}
        if not self.only_eval:
            src_rank_map = {
                "rollout_train": CommMapper.get_src_ranks(
                    batch_size=self.cfg.env.train.total_num_envs // self.stage_num,
                    src_world_size=self._component_placement.get_world_size("rollout"),
                    dst_world_size=self._component_placement.get_world_size("env"),
                    dst_rank=self._rank,
                ),
            }
            if self.cfg.get("reward", {}).get("use_reward_model", False):
                src_rank_map.update(
                    {
                        "reward_train": CommMapper.get_src_ranks(
                            batch_size=self.cfg.env.train.total_num_envs
                            // self.stage_num,
                            src_world_size=self._component_placement.get_world_size(
                                "reward"
                            ),
                            dst_world_size=self._component_placement.get_world_size(
                                "env"
                            ),
                            dst_rank=self._rank,
                        ),
                    }
                )
        if self.enable_eval:
            src_rank_map.update(
                {
                    "rollout_eval": CommMapper.get_src_ranks(
                        batch_size=self.cfg.env.eval.total_num_envs // self.stage_num,
                        src_world_size=self._component_placement.get_world_size(
                            "rollout"
                        ),
                        dst_world_size=self._component_placement.get_world_size("env"),
                        dst_rank=self._rank,
                    ),
                }
            )
        return src_rank_map

    def _init_env(self):
        for i in range(self.stage_num):
            if self.cfg.env.train.auto_reset:
                extracted_obs, _ = self.env_list[i].reset()
                self.last_obs_list.append(extracted_obs)
                self.last_intervened_info_list.append((None, None))
            if self.enable_offload and hasattr(self.env_list[i], "offload"):
                self.env_list[i].offload()

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            rewards=chunk_rewards,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )
        return env_output, env_info

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )

        current_dones = chunk_dones[:, -1]  # [num_envs] bool
        prev = self.eval_prev_done[stage_id]
        newly_done = current_dones & ~prev.to(current_dones.device)
        self.eval_prev_done[stage_id] = current_dones.clone()

        if newly_done.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][newly_done].cpu()
            elif "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key][newly_done].cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
        )
        return env_output, env_info

    def _build_chunk_final_obs(self, obs_list, infos_list):
        """Build per-env terminal observations for a whole chunk.

        Matches the old wrapper semantics:
        - default to the last rollout observation for each env
        - if an env terminated earlier in the chunk, replace that env's observation
          with the true `final_observation` captured at that substep
        """
        if not isinstance(obs_list, (list, tuple)) or len(obs_list) == 0:
            return None

        last_obs = obs_list[-1]
        if not isinstance(last_obs, dict):
            return None

        merged_final_obs = copy_dict_tensor(last_obs)

        if not isinstance(infos_list, (list, tuple)):
            return merged_final_obs

        for step_infos in infos_list:
            if not isinstance(step_infos, dict):
                continue
            if (
                "final_observation" not in step_infos
                or "_final_observation" not in step_infos
            ):
                continue

            final_obs = step_infos["final_observation"]
            reset_mask = step_infos["_final_observation"]
            if final_obs is None or reset_mask is None:
                continue
            reset_mask = (
                reset_mask.detach().cpu().numpy()
                if isinstance(reset_mask, torch.Tensor)
                else np.asarray(reset_mask)
            )
            done_mask = (
                reset_mask.any(axis=-1)
                if reset_mask.ndim > 1
                else reset_mask.astype(bool)
            )
            if not done_mask.any():
                continue

            for key, value in merged_final_obs.items():
                if key not in final_obs:
                    continue

                final_value = final_obs[key]
                if isinstance(value, torch.Tensor) and isinstance(
                    final_value, torch.Tensor
                ):
                    dst_mask = torch.as_tensor(done_mask, device=value.device)
                    src_mask = dst_mask.to(device=final_value.device)
                    merged_final_obs[key][dst_mask] = final_value[src_mask]
                elif isinstance(value, np.ndarray) and isinstance(
                    final_value, np.ndarray
                ):
                    merged_final_obs[key][done_mask] = final_value[done_mask]

        return merged_final_obs

    def recv_chunk_actions(self, input_channel: Channel, mode="train") -> np.ndarray:
        """Receive and merge chunked actions for the current env worker.

        The method fetches one action shard from each mapped rollout source rank
        under a deterministic channel key pattern and concatenates them on the
        batch dimension.

        Args:
            input_channel: Channel carrying rollout->env action chunks.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            Concatenated action chunk array with shape ``[num_envs_per_stage, ...]``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_rank_map[f"rollout_{mode}"]
        chunk_action = []
        for src_rank, expected_size in src_ranks_and_sizes:
            action_i = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_actions"
                ),
            )
            if isinstance(action_i, torch.Tensor):
                action_i = action_i.detach().cpu().numpy()
            else:
                action_i = np.asarray(action_i)
            assert action_i.shape[0] == expected_size, (
                f"Expected action shard size {expected_size} from rollout rank {src_rank}, "
                f"got shape {action_i.shape}."
            )
            chunk_action.append(action_i)
        chunk_action = np.concatenate(chunk_action, axis=0)
        expected_total_size = sum(size for _, size in src_ranks_and_sizes)
        assert chunk_action.shape[0] == expected_total_size, (
            f"Expected concatenated action size {expected_total_size}, got {chunk_action.shape[0]}."
        )
        return chunk_action

    @Worker.timer("recv_rollout_results")
    def recv_rollout_results(
        self, input_channel: Channel, mode="train"
    ) -> RolloutResult:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_rank_map[f"rollout_{mode}"]
        rollout_results: list[RolloutResult] = []

        def _infer_rollout_batch_size(rollout_result: RolloutResult) -> int:
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(rollout_result, field_name, None)
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
            if rollout_result.forward_inputs:
                first_tensor = next(iter(rollout_result.forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return first_tensor.shape[0]
            raise ValueError("Cannot infer batch size from rollout result.")

        for src_rank, expected_size in src_ranks_and_sizes:
            rollout_result = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_rollout_results"
                ),
            )

            actual_size = _infer_rollout_batch_size(rollout_result)
            assert actual_size == expected_size, (
                f"Expected rollout result size {expected_size} from rollout rank {src_rank}, "
                f"got batch size {actual_size}."
            )

            rollout_results.append(rollout_result)

        return RolloutResult.merge_rollout_results(rollout_results)

    @Worker.timer("compute_bootstrap_rewards")
    def compute_bootstrap_rewards(
        self,
        env_output: EnvOutput,
        bootstrap_values: torch.Tensor | None,
        reward_model_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rewards = env_output.rewards
        if rewards is None:
            return None

        if reward_model_output is not None:
            reward_model_output = reward_model_output.to(rewards.dtype)
            rewards = (
                self.env_reward_weight * rewards
                + self.reward_weight * reward_model_output
            )

        adjusted_rewards = rewards.clone()
        if (
            bootstrap_values is None
            or not self.cfg.env.train.auto_reset
            or env_output.dones is None
        ):
            return adjusted_rewards

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "standard":
            last_step_truncations = env_output.truncations[:, -1]
        else:
            last_step_truncations = env_output.dones[:, -1]

        if not last_step_truncations.any():
            return adjusted_rewards

        final_values = torch.zeros_like(adjusted_rewards[:, -1], dtype=torch.float32)
        final_values[last_step_truncations] = (
            bootstrap_values[last_step_truncations].reshape(-1).to(torch.float32)
        )
        adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values
        return adjusted_rewards

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            for i in range(self.stage_num):
                if self.cfg.env.train.video_cfg.save_video and isinstance(
                    self.env_list[i], RecordVideo
                ):
                    self.env_list[i].flush_video()
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.cfg.env.eval.video_cfg.save_video and isinstance(
                    self.eval_env_list[i], RecordVideo
                ):
                    self.eval_env_list[i].flush_video()
                if not self.cfg.env.eval.auto_reset:
                    self.eval_env_list[i].update_reset_state_ids()

    def send_env_batch(
        self,
        rollout_channel: Channel,
        env_batch: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> None:
        """Send split env batches to mapped rollout ranks.

        Each destination rank receives one split batch via a stable key built from
        ``src_rank``, ``dst_rank`` and ``mode``.

        Args:
            rollout_channel: Channel carrying env->rollout outputs.
            env_batch: Env output dictionary for one pipeline stage.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_rank_map[f"rollout_{mode}"]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        env_batches = split_dict(env_batch, split_sizes)
        for (rank, _), env_batch_i in zip(dst_ranks_and_sizes, env_batches):
            rollout_channel.put(
                item=env_batch_i,
                key=CommMapper.build_channel_key(self._rank, rank, extra=f"{mode}_obs"),
            )

    def send_reward_input(
        self,
        send_channel: Channel,
        reward_input: dict[str, torch.Tensor],
        mode: Literal["train", "eval"] = "train",
    ):
        dst_ranks_and_sizes = self.dst_rank_map[f"reward_{mode}"]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        reward_input_batches = split_dict(reward_input, split_sizes)
        for (rank, _), reward_input_i in zip(dst_ranks_and_sizes, reward_input_batches):
            send_channel.put(
                item=reward_input_i,
                key=CommMapper.build_channel_key(
                    self._rank, rank, extra=f"{mode}_reward_input"
                ),
                async_op=True,
            )

    @Worker.timer("recv_reward_results")
    def recv_reward_results(self, recv_channel: Channel) -> torch.Tensor:
        reward_results: list[torch.Tensor] = []
        src_ranks_and_sizes = self.src_rank_map["reward_train"]
        for src_rank, expected_size in src_ranks_and_sizes:
            rewards = recv_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra="reward_output"
                ),
            )
            actual_size = rewards.shape[0]
            assert actual_size == expected_size, (
                f"Expected reward result size {expected_size} from reward rank {src_rank}, "
                f"got batch size {actual_size}."
            )
            reward_results.append(rewards)
        return torch.cat(reward_results, dim=0)

    @Worker.timer("get_reward_model_output")
    def get_reward_model_output(
        self,
        env_output: EnvOutput,
        send_channel: Channel,
        recv_channel: Channel,
        last_run: bool = False,
    ):
        if self.reward_mode == "per_step":
            reward_input_obs = (
                env_output.final_obs
                if env_output.final_obs is not None
                else env_output.obs
            )
        elif self.reward_mode == "terminal" and env_output.final_obs is not None:
            reward_input_obs = env_output.final_obs
        else:
            return None

        reward_input = {"images": reward_input_obs["main_images"]}
        if last_run:
            reward_input.update(
                {
                    "last_run": torch.ones(
                        (self.train_num_envs_per_stage, 1), dtype=torch.bool
                    )
                }
            )
        self.send_reward_input(send_channel=send_channel, reward_input=reward_input)
        reward_output = self.recv_reward_results(recv_channel=recv_channel)
        if self.reward_mode != "terminal" or reward_output is None:
            return reward_output
        return self._scatter_terminal_reward_output(
            env_output=env_output, reward_output=reward_output
        )

    def _scatter_terminal_reward_output(
        self,
        env_output: EnvOutput,
        reward_output: torch.Tensor,
    ) -> torch.Tensor:
        if env_output.rewards is None or env_output.dones is None:
            return reward_output

        done_envs = env_output.dones.any(dim=1)
        sparse_rewards = torch.zeros_like(env_output.rewards, dtype=reward_output.dtype)
        if not done_envs.any():
            return sparse_rewards

        done_steps = env_output.dones.to(torch.int64).argmax(dim=1)
        sparse_rewards[done_envs, done_steps[done_envs]] = (
            reward_output[done_envs].reshape(-1).to(sparse_rewards.dtype)
        )
        return sparse_rewards

    def bootstrap_step(self) -> list[EnvOutput]:
        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.cfg.actor.model.num_action_chunks)
            )

        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                dones = get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=(
                        infos["final_observation"]
                        if "final_observation" in infos
                        else None
                    ),
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)

        return env_outputs

    def record_env_metrics(
        self, env_metrics: dict[str, list], env_info: dict[str, Any], epoch: int
    ):
        for key, value in env_info.items():
            if (
                not self.cfg.env.train.auto_reset
                and not self.cfg.env.train.ignore_terminations
            ):
                if key in env_metrics and len(env_metrics[key]) > epoch:
                    env_metrics[key][epoch] = value
                else:
                    env_metrics[key].append(value)
            else:
                env_metrics[key].append(value)

    @staticmethod
    def _tensor_scalar(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        if torch.is_tensor(value):
            if value.numel() == 0:
                return default
            return float(value.detach().reshape(-1)[0].cpu().item())
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            return float(value.reshape(-1)[0])
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            return EnvWorker._tensor_scalar(value[0], default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _policy_diag_from_forward_inputs(self, forward_inputs: dict | None) -> dict:
        if not forward_inputs:
            return {}
        action = int(self._tensor_scalar(forward_inputs.get("action"), -1.0))
        return {
            "is_stop": action == 0,
            "grounded_sam_enabled": bool(
                self._tensor_scalar(forward_inputs.get("grounded_sam_enabled"), 0.0)
            ),
            "grounded_sam_used": bool(
                self._tensor_scalar(forward_inputs.get("grounded_sam_used"), 0.0)
            ),
            "grounded_sam_detected": bool(
                self._tensor_scalar(forward_inputs.get("grounded_sam_detected"), 0.0)
            ),
            "grounded_sam_confidence": self._tensor_scalar(
                forward_inputs.get("grounded_sam_confidence"), 0.0
            ),
        }

    def _grpo_diag_by_env(
        self,
        stage_id: int,
        env_diag_snapshot: list[dict] | None = None,
    ) -> dict[int, dict]:
        diag_by_env = {
            int(d.get("env_id", idx)): dict(d)
            for idx, d in enumerate(env_diag_snapshot or [])
            if isinstance(d, dict)
        }
        env_obj = self.env_list[stage_id]
        target_env = getattr(env_obj, "env", env_obj)
        if hasattr(target_env, "get_grpo_process_diagnostics"):
            try:
                for d in target_env.get_grpo_process_diagnostics():
                    if not isinstance(d, dict):
                        continue
                    env_id = int(d.get("env_id", len(diag_by_env)))
                    # Keep exact one-shot completion diagnostics when present, and
                    # fill any missing slots from the read-only env snapshot. This
                    # is diagnostic-only; rewards/advantages are already in
                    # per_env_results and are not changed here.
                    diag_by_env.setdefault(env_id, dict(d))
            except Exception as exc:
                print(
                    f"[GRPO][group-diag] stage={stage_id} diagnostics unavailable: {exc}",
                    flush=True,
                )
        return diag_by_env

    @staticmethod
    def _trajectory_reward_sum(result: EmbodiedRolloutResult) -> float:
        total = 0.0
        for reward in result.rewards:
            if reward is not None:
                total += float(reward.detach().float().sum().cpu().item())
        return total

    def _candidate_group_selection_cfg(self):
        cfg = getattr(self.cfg.algorithm, "candidate_group_selection", None)
        if cfg is None or not bool(cfg.get("enabled", False)):
            return None
        if bool(cfg.get("only_episode_overfit", True)):
            overfit_cfg = getattr(self.cfg.env.train, "episode_overfit", None)
            if overfit_cfg is None or not bool(overfit_cfg.get("enabled", False)):
                return None
        return cfg

    def _candidate_record(
        self,
        env_i: int,
        result: EmbodiedRolloutResult,
        diag_by_env: dict[int, dict],
    ) -> dict:
        diag = dict(diag_by_env.get(env_i, {}))
        success_type = str(diag.get("success_type", "unknown"))
        success = float(diag.get("success", 0.0) or 0.0) > 0.5
        return {
            "env_i": env_i,
            "episode_id": diag.get("episode_id", "?"),
            "scene_id": diag.get("scene_id", "?"),
            "reward_sum": self._trajectory_reward_sum(result),
            "success": success,
            "success_type": success_type,
            "wrong_stop": success_type == "wrong_stop",
            "no_stop": success_type == "no_stop",
            "recovered": success_type == "recovered",
            "best_dtg_progress": float(diag.get("best_dtg_progress", 0.0) or 0.0),
            "final_dtg": float(diag.get("final_dtg", 0.0) or 0.0),
            "min_dtg": float(diag.get("min_dtg", 0.0) or 0.0),
            "final_regression": float(diag.get("final_regression", 0.0) or 0.0),
            "steps_taken": float(diag.get("steps_taken", diag.get("steps", 0.0)) or 0.0),
            "parse_fail_count": float(diag.get("parse_fail_count", diag.get("parse_fail", 0.0)) or 0.0),
        }

    @staticmethod
    def _select_unique_candidate(
        selected: list[int],
        reasons: dict[int, str],
        candidates: list[dict],
        key_fn,
        reason: str,
        reverse: bool = True,
        predicate=None,
    ) -> None:
        pool = [
            c for c in candidates
            if c["env_i"] not in selected and (predicate is None or predicate(c))
        ]
        if not pool:
            return
        pool = sorted(pool, key=key_fn, reverse=reverse)
        env_i = int(pool[0]["env_i"])
        selected.append(env_i)
        reasons[env_i] = reason

    def _select_candidate_group(
        self,
        env_metrics: dict[str, list],
        stage_id: int,
        per_env_results: list[EmbodiedRolloutResult],
        env_diag_snapshot: list[dict] | None,
    ) -> tuple[list[EmbodiedRolloutResult], list[dict]]:
        cfg = self._candidate_group_selection_cfg()
        if cfg is None:
            return per_env_results, list(env_diag_snapshot or [])

        candidate_k = int(cfg.get("candidate_k", len(per_env_results)))
        train_group_size = int(cfg.get("train_group_size", self.cfg.algorithm.group_size))
        if train_group_size <= 0 or len(per_env_results) <= train_group_size:
            return per_env_results, list(env_diag_snapshot or [])

        limited_results = per_env_results[: min(candidate_k, len(per_env_results))]
        diag_by_env = self._grpo_diag_by_env(stage_id, env_diag_snapshot)
        candidates = [
            self._candidate_record(env_i, result, diag_by_env)
            for env_i, result in enumerate(limited_results)
        ]
        if len(candidates) <= train_group_size:
            return limited_results, [
                dict(diag_by_env.get(i, {"env_id": i})) for i in range(len(limited_results))
            ]

        selected: list[int] = []
        reasons: dict[int, str] = {}

        # 1) Prefer true success/recovered if exploration produced one.
        self._select_unique_candidate(
            selected,
            reasons,
            candidates,
            key_fn=lambda c: (c["success"], c["recovered"], c["reward_sum"], c["best_dtg_progress"]),
            reason="success_or_recovered",
            predicate=lambda c: c["success"] or c["recovered"],
        )
        # 2) Keep the best progress trajectory, even if it failed.
        self._select_unique_candidate(
            selected,
            reasons,
            candidates,
            key_fn=lambda c: (c["best_dtg_progress"], c["reward_sum"]),
            reason="top_progress",
        )
        # 3) Keep a wrong-stop representative; prefer one that got closer.
        self._select_unique_candidate(
            selected,
            reasons,
            candidates,
            key_fn=lambda c: (c["best_dtg_progress"], -c["final_regression"], c["reward_sum"]),
            reason="wrong_stop",
            predicate=lambda c: c["wrong_stop"],
        )
        # 4) Keep a no-stop / strong regression hard negative.
        self._select_unique_candidate(
            selected,
            reasons,
            candidates,
            key_fn=lambda c: (c["no_stop"], c["final_regression"], -c["reward_sum"]),
            reason="no_stop_or_regression",
            predicate=lambda c: c["no_stop"] or c["final_regression"] > 0.0,
        )

        # Fill with reward/progress diversity, not pure top-k.
        remaining = [c for c in candidates if c["env_i"] not in selected]
        if len(selected) < train_group_size and remaining:
            by_reward = sorted(remaining, key=lambda c: c["reward_sum"])
            fill_order: list[tuple[dict, str]] = []
            fill_order.append((by_reward[-1], "reward_high"))
            fill_order.append((by_reward[len(by_reward) // 2], "reward_mid"))
            fill_order.append((by_reward[0], "reward_low"))
            by_progress = sorted(remaining, key=lambda c: c["best_dtg_progress"], reverse=True)
            fill_order.append((by_progress[0], "progress_fill"))
            for cand, reason in fill_order:
                env_i = int(cand["env_i"])
                if env_i in selected:
                    continue
                selected.append(env_i)
                reasons[env_i] = reason
                if len(selected) >= train_group_size:
                    break

        for cand in candidates:
            if len(selected) >= train_group_size:
                break
            env_i = int(cand["env_i"])
            if env_i not in selected:
                selected.append(env_i)
                reasons[env_i] = "fallback"

        selected = selected[:train_group_size]
        selected_results = [limited_results[i] for i in selected]
        selected_diag: list[dict] = []
        for local_i, env_i in enumerate(selected):
            d = dict(diag_by_env.get(env_i, {"env_id": env_i}))
            d["original_env_id"] = env_i
            d["env_id"] = local_i
            selected_diag.append(d)

        def _std(vals: list[float]) -> float:
            if not vals:
                return 0.0
            return float(torch.tensor(vals, dtype=torch.float32).std(unbiased=False).item())

        pool_rewards = [float(c["reward_sum"]) for c in candidates]
        selected_records = [candidates[i] for i in selected]
        selected_rewards = [float(c["reward_sum"]) for c in selected_records]
        pool_progress = [float(c["best_dtg_progress"]) for c in candidates]
        selected_progress = [float(c["best_dtg_progress"]) for c in selected_records]

        def _metric(name: str, value: float) -> None:
            env_metrics[name].append(torch.tensor([float(value)], dtype=torch.float32))

        _metric("grpo/candidate_pool_reward_std", _std(pool_rewards))
        _metric("grpo/candidate_selected_reward_std", _std(selected_rewards))
        _metric("grpo/candidate_pool_best_progress_max", max(pool_progress) if pool_progress else 0.0)
        _metric("grpo/candidate_selected_best_progress_max", max(selected_progress) if selected_progress else 0.0)
        _metric("grpo/candidate_pool_success_rate", sum(c["success"] for c in candidates) / max(len(candidates), 1))
        _metric("grpo/candidate_selected_success_rate", sum(c["success"] for c in selected_records) / max(len(selected_records), 1))

        print(
            "[GRPO][candidate-pool] "
            f"stage={stage_id} candidate_k={len(candidates)} train_group_size={train_group_size} "
            f"selected={selected} "
            f"reward_sum={[round(c['reward_sum'], 3) for c in selected_records]} "
            f"success_type={[c['success_type'] for c in selected_records]} "
            f"best_progress={[round(c['best_dtg_progress'], 3) for c in selected_records]} "
            f"final_regression={[round(c['final_regression'], 3) for c in selected_records]} "
            f"selection_reason={[reasons.get(i, 'unknown') for i in selected]}",
            flush=True,
        )

        return selected_results, selected_diag

    def _record_grpo_group_diagnostics(
        self,
        env_metrics: dict[str, list],
        stage_id: int,
        per_env_results: list[EmbodiedRolloutResult],
        env_diag_snapshot: list[dict] | None = None,
    ) -> None:
        group_size = int(getattr(self.cfg.algorithm, "group_size", 1) or 1)
        if group_size <= 0:
            group_size = 1
        # AVSPO ACR uses a tiny numerical threshold for reward homogeneity:
        # ACR = mean_j I(std(R_group_j) < tau), tau = 1e-6.
        acr_threshold = float(
            getattr(self.cfg.algorithm, "acr_reward_std_threshold", 1e-6)
        )
        low_std_threshold = float(
            getattr(self.cfg.algorithm, "low_reward_std_threshold", 0.05)
        )

        diag_by_env = self._grpo_diag_by_env(stage_id, env_diag_snapshot)

        reward_sums: list[float] = []
        for result in per_env_results:
            reward_sums.append(self._trajectory_reward_sum(result))

        for group_start in range(0, len(per_env_results), group_size):
            members = list(
                range(group_start, min(group_start + group_size, len(per_env_results)))
            )
            if not members:
                continue
            vals = torch.tensor(
                [reward_sums[i] for i in members], dtype=torch.float32
            )
            reward_mean = float(vals.mean().item())
            reward_std = float(vals.std(unbiased=False).item()) if len(members) > 1 else 0.0
            member_diag = [diag_by_env[i] for i in members if i in diag_by_env]
            diag_count = len(member_diag)
            diag_coverage = diag_count / max(len(members), 1)
            full_diag = diag_count == len(members)

            success_count = sum(float(d.get("success", 0.0)) > 0.5 for d in member_diag)
            wrong_stop_count = sum(
                str(d.get("success_type", "")) == "wrong_stop" for d in member_diag
            )
            no_stop_count = sum(
                str(d.get("success_type", "")) == "no_stop" for d in member_diag
            )
            best_progress = [
                float(d.get("best_dtg_progress", 0.0)) for d in member_diag
            ]
            if best_progress:
                best_progress_t = torch.tensor(best_progress, dtype=torch.float32)
                best_progress_mean = float(best_progress_t.mean().item())
                best_progress_max = float(best_progress_t.max().item())
                best_progress_std = (
                    float(best_progress_t.std(unbiased=False).item())
                    if len(best_progress) > 1
                    else 0.0
                )
            else:
                best_progress_mean = 0.0
                best_progress_max = 0.0
                best_progress_std = 0.0
            all_failure = full_diag and success_count == 0
            all_wrong_stop = full_diag and all_failure and wrong_stop_count == len(members)
            ep_id = next(
                (d.get("episode_id", "?") for d in member_diag if d.get("episode_id") not in (None, "")),
                "?",
            )

            def _metric(name: str, value: float) -> None:
                env_metrics[name].append(torch.tensor([float(value)], dtype=torch.float32))

            _metric("grpo/acr", reward_std < acr_threshold)
            _metric("grpo/low_reward_std_rate", reward_std < low_std_threshold)
            _metric("grpo/group_diag_coverage", diag_coverage)
            _metric("grpo/group_reward_std_mean", reward_std)
            _metric("grpo/group_reward_mean", reward_mean)
            if full_diag:
                _metric("grpo/all_failure_group_rate", all_failure)
                _metric("grpo/all_wrong_stop_group_rate", all_wrong_stop)
                _metric("grpo/group_wrong_stop_count", wrong_stop_count)
                _metric("grpo/group_no_stop_count", no_stop_count)
            if diag_count > 0:
                _metric("grpo/group_best_progress_mean", best_progress_mean)
                _metric("grpo/group_best_progress_std", best_progress_std)

            msg = (
                f"[GRPO][group-diag] stage={stage_id} "
                f"group={group_start // group_size} ep={ep_id} "
                f"reward_mean={reward_mean:.3f} reward_std={reward_std:.6f} "
                f"acr={int(reward_std < acr_threshold)} "
                f"low_std={int(reward_std < low_std_threshold)} "
                f"diag={diag_count}/{len(members)}"
            )
            if full_diag:
                msg += (
                    f" success_count={success_count} all_failure={int(all_failure)} "
                    f"wrong_stop={wrong_stop_count} no_stop={no_stop_count}"
                )
            if diag_count > 0:
                msg += (
                    f" best_progress_max={best_progress_max:.2f} "
                    f"best_progress_std={best_progress_std:.2f}"
                )
            print(msg, flush=True)

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    async def send_rollout_trajectories(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ):
        trajectories: Trajectory = rollout_result.to_splited_trajectories(
            self.actor_split_num
        )
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)

    @Worker.timer("run_interact_once")
    async def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        # Determine whether to use decision-level rollout.
        max_dec = self.cfg.env.train.get(
            "max_decisions_per_rollout_epoch",
            None,
        )
        use_decision_rollout = max_dec is not None

        if use_decision_rollout:
            # Per-env EmbodiedRolloutResult: each env accumulates its own decision steps.
            # Indexed as rollout_results_per_env[stage_id][env_i].
            rollout_results_per_env: list[list[EmbodiedRolloutResult]] = [
                [
                    EmbodiedRolloutResult(
                        max_episode_length=self.cfg.env.train.max_episode_steps,
                    )
                    for _ in range(self.train_num_envs_per_stage)
                ]
                for _ in range(self.stage_num)
            ]
            # Per-stage reward/done accumulators (accumulated between decisions).
            # Explicitly on CPU: env_worker may run with a CUDA default device, so
            # torch.zeros() without device= would create CUDA tensors.
            _cpu = torch.device("cpu")
            acc_rewards = [
                torch.zeros(self.train_num_envs_per_stage, 1, device=_cpu)
                for _ in range(self.stage_num)
            ]
            acc_dones = [
                torch.zeros(self.train_num_envs_per_stage, 1, dtype=torch.bool, device=_cpu)
                for _ in range(self.stage_num)
            ]
            acc_terminations = [
                torch.zeros(self.train_num_envs_per_stage, 1, dtype=torch.bool, device=_cpu)
                for _ in range(self.stage_num)
            ]
            acc_truncations = [
                torch.zeros(self.train_num_envs_per_stage, 1, dtype=torch.bool, device=_cpu)
                for _ in range(self.stage_num)
            ]
            decision_counts = [
                torch.zeros(self.train_num_envs_per_stage, dtype=torch.long, device=_cpu)
                for _ in range(self.stage_num)
            ]
            # Per-env pending forward_inputs/logprobs/versions for the current decision
            pending_decision_data: list[list[dict | None]] = [
                [None] * self.train_num_envs_per_stage
                for _ in range(self.stage_num)
            ]
            completed_episode_diag_by_stage: list[dict[int, dict]] = [
                {} for _ in range(self.stage_num)
            ]
        else:
            self.rollout_results: list[EmbodiedRolloutResult] = [
                EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.train.max_episode_steps,
                )
                for _ in range(self.stage_num)
            ]

        env_metrics = defaultdict(list)

        def _flush_pending_decision(stage_id: int, env_i: int) -> bool:
            """Append one pending LLM decision with all rewards accumulated so far."""
            pdata = pending_decision_data[stage_id][env_i]
            if pdata is None:
                return False

            _env = self.env_list[stage_id]
            if hasattr(_env, "compute_decision_ndtw_reward"):
                policy_diag = self._policy_diag_from_forward_inputs(
                    pdata.get("forward_inputs")
                )
                try:
                    _ndtw_r = _env.compute_decision_ndtw_reward(
                        [env_i],
                        policy_diag_by_env={env_i: policy_diag},
                    )
                except TypeError:
                    _ndtw_r = _env.compute_decision_ndtw_reward([env_i])
                if hasattr(_env, "pop_gsam_reward_diagnostics"):
                    for _name, _value in _env.pop_gsam_reward_diagnostics().items():
                        env_metrics[_name].append(
                            torch.tensor([float(_value)], dtype=torch.float32)
                        )
                acc_rewards[stage_id][env_i] += float(_ndtw_r[env_i])
                if hasattr(_env, "pop_completed_episode_diagnostic"):
                    try:
                        _completed_diag = _env.pop_completed_episode_diagnostic(env_i)
                    except Exception as exc:
                        print(
                            f"[GRPO][group-diag] stage={stage_id} env={env_i} "
                            f"completed diagnostic unavailable: {exc}",
                            flush=True,
                        )
                        _completed_diag = None
                    if isinstance(_completed_diag, dict):
                        completed_episode_diag_by_stage[stage_id][env_i] = _completed_diag

            chunk_step_result = ChunkStepResult(
                actions=pdata["actions"],
                prev_logprobs=pdata["prev_logprobs"],
                prev_values=pdata["prev_values"],
                forward_inputs=pdata["forward_inputs"],
                versions=pdata["versions"],
                dones=pdata["dones"],
                truncations=pdata["truncations"],
                terminations=pdata["terminations"],
                rewards=acc_rewards[stage_id][env_i:env_i + 1],
            )
            rollout_results_per_env[stage_id][env_i].append_step_result(
                chunk_step_result
            )
            if pdata.get("save_flags") is not None:
                rollout_results_per_env[stage_id][env_i].mark_last_step_with_flags(
                    pdata["save_flags"]
                )

            acc_rewards[stage_id][env_i] = 0.0
            acc_dones[stage_id][env_i] = False
            acc_terminations[stage_id][env_i] = False
            acc_truncations[stage_id][env_i] = False
            pending_decision_data[stage_id][env_i] = None
            return True

        for epoch in range(self.rollout_epoch):
            env_outputs = self.bootstrap_step()
            for stage_id in range(self.stage_num):
                env_output: EnvOutput = env_outputs[stage_id]
                env_batch = env_output.to_dict()
                # Initial obs of the epoch: nothing decided yet → global termination
                # is always False. Inject it (decision-rollout only) so the rollout's
                # very first step also reads the global flag rather than its local
                # all_done, keeping every step on the synchronized global path.
                if use_decision_rollout:
                    mask_mode_id = self._ppo_loss_mask_mode_id()
                    env_batch["obs"]["should_terminate"] = torch.zeros(
                        self.train_num_envs_per_stage, dtype=torch.bool,
                        device=torch.device("cpu"),
                    )
                    env_batch["obs"]["ppo_loss_mask_mode_id"] = torch.full(
                        (self.train_num_envs_per_stage,),
                        int(mask_mode_id),
                        dtype=torch.long,
                        device=torch.device("cpu"),
                    )
                self.send_env_batch(
                    rollout_channel,
                    {
                        "obs": env_batch["obs"],
                        "final_obs": env_batch["final_obs"],
                    },
                )

            if use_decision_rollout:
                # ── Decision-level rollout loop ──────────────────────────────
                # Safety bound: max_dec decisions × up to 10 env steps each + pipeline priming
                safety_steps = max_dec * 10 + self.stage_num + 1
                step = 0
                received_bootstrap = False
                # Track last seen env dormant state per stage for termination check
                last_env_dormant = [
                    torch.zeros(self.train_num_envs_per_stage, dtype=torch.bool, device=_cpu)
                    for _ in range(self.stage_num)
                ]

                while step < safety_steps:
                    for stage_id in range(self.stage_num):
                        if cooperative_yield:
                            await asyncio.sleep(0)

                        env_output = env_outputs[stage_id]
                        curr_obs = env_output.obs

                        reward_model_output = None
                        if reward_channel is not None and step != 0:
                            reward_model_output = self.get_reward_model_output(
                                env_output,
                                send_channel=reward_channel,
                                recv_channel=input_channel,
                            )
                            if reward_model_output is not None:
                                env_metrics["reward_model_output"].append(
                                    reward_model_output.detach().float().reshape(-1).cpu()
                                )

                        rollout_result = self.recv_rollout_results(
                            input_channel, mode="train"
                        )

                        if rollout_result.is_last_decision:
                            # Bootstrap result received inline — rollout has exited its main loop.
                            # Handle intervene_actions and append bootstrap step, then exit.
                            if env_output.intervene_actions is not None:
                                for env_i in range(self.train_num_envs_per_stage):
                                    if len(rollout_results_per_env[stage_id][env_i].actions) > 0:
                                        rollout_results_per_env[stage_id][env_i].update_last_actions(
                                            env_output.intervene_actions[env_i:env_i + 1],
                                            env_output.intervene_flags[env_i:env_i + 1],
                                        )
                            for env_i in range(self.train_num_envs_per_stage):
                                _flush_pending_decision(stage_id, env_i)
                            for env_i in range(self.train_num_envs_per_stage):
                                bootstrap_step_result = ChunkStepResult(
                                    prev_values=(
                                        rollout_result.prev_values[env_i:env_i + 1]
                                        if self.collect_prev_infos and rollout_result.prev_values is not None
                                        else None
                                    ),
                                    dones=env_output.dones[env_i:env_i + 1] if env_output.dones is not None else None,
                                    truncations=env_output.truncations[env_i:env_i + 1] if env_output.truncations is not None else None,
                                    terminations=env_output.terminations[env_i:env_i + 1] if env_output.terminations is not None else None,
                                )
                                rollout_results_per_env[stage_id][env_i].append_step_result(bootstrap_step_result)
                            received_bootstrap = True
                            break  # break inner for loop; outer while exits below

                        rewards = self.compute_bootstrap_rewards(
                            env_output, rollout_result.bootstrap_values, reward_model_output
                        )

                        # is_decision[i]: True = new LLM inference, False = macro replay
                        is_dec = rollout_result.is_decision
                        if is_dec is None:
                            is_dec = torch.ones(
                                self.train_num_envs_per_stage, dtype=torch.bool, device=_cpu
                            )
                        else:
                            is_dec = is_dec.cpu()

                        # Detect dormant envs from the pre-action observation.
                        # These slots should not flush a decision for the action
                        # generated from this observation.
                        task_descs = env_output.obs.get("task_descriptions", None)
                        if task_descs is not None:
                            env_dormant = torch.tensor(
                                [desc == "" for desc in task_descs], dtype=torch.bool, device=_cpu
                            )
                        else:
                            env_dormant = torch.zeros(
                                self.train_num_envs_per_stage, dtype=torch.bool, device=_cpu
                            )

                        def _pre_action_done_tensor(value):
                            if value is None:
                                return torch.zeros(
                                    self.train_num_envs_per_stage,
                                    1,
                                    dtype=torch.bool,
                                    device=_cpu,
                                )
                            value = value.cpu()
                            return (
                                value.any(dim=-1, keepdim=True)
                                if value.dim() > 1
                                else value.reshape(-1, 1)
                            )

                        pre_step_done = _pre_action_done_tensor(env_output.dones)
                        pre_step_term = _pre_action_done_tensor(env_output.terminations)
                        pre_step_trunc = _pre_action_done_tensor(env_output.truncations)

                        # On decision steps: save current forward_inputs/logprobs for this env
                        for env_i in range(self.train_num_envs_per_stage):
                            if not is_dec[env_i]:
                                continue
                            if pending_decision_data[stage_id][env_i] is not None:
                                _flush_pending_decision(stage_id, env_i)
                            if env_dormant[env_i]:
                                continue
                            if decision_counts[stage_id][env_i] >= max_dec:
                                continue
                            # Extract per-env slices
                            env_fi = {
                                k: v[env_i:env_i + 1]
                                for k, v in rollout_result.forward_inputs.items()
                            } if rollout_result.forward_inputs else {}
                            pending_decision_data[stage_id][env_i] = {
                                "forward_inputs": env_fi,
                                "prev_logprobs": (
                                    rollout_result.prev_logprobs[env_i:env_i + 1]
                                    if self.collect_prev_infos and rollout_result.prev_logprobs is not None
                                    else None
                                ),
                                "prev_values": (
                                    rollout_result.prev_values[env_i:env_i + 1]
                                    if self.collect_prev_infos and rollout_result.prev_values is not None
                                    else None
                                ),
                                "versions": (
                                    rollout_result.versions[env_i:env_i + 1]
                                    if rollout_result.versions is not None
                                    else None
                                ),
                                "actions": (
                                    env_fi["action"]
                                    if "action" in env_fi
                                    else rollout_result.actions[env_i:env_i + 1]
                                    if rollout_result.actions is not None
                                    else None
                                ),
                                # ChunkStepResult.dones is consumed by
                                # compute_loss_mask() as the pre-action done
                                # state.  Storing post-action done here masks
                                # out the terminal action itself and removes the
                                # reward-bearing STOP/no-stop gradient.
                                "dones": pre_step_done[env_i:env_i + 1],
                                "terminations": pre_step_term[env_i:env_i + 1],
                                "truncations": pre_step_trunc[env_i:env_i + 1],
                                "save_flags": (
                                    rollout_result.save_flags[env_i:env_i + 1]
                                    if rollout_result.save_flags is not None
                                    else None
                                ),
                            }
                            decision_counts[stage_id][env_i] += 1

                        # Execute the action before computing decision-level reward.
                        # The reward belongs to the just-saved decision/action above.
                        # Computing it before env_interact_step() binds reward to the
                        # previous observation and can drop the terminal STOP/progress
                        # segment entirely.
                        env_output, env_info = self.env_interact_step(
                            rollout_result.actions, stage_id
                        )

                        rewards = self.compute_bootstrap_rewards(
                            env_output, rollout_result.bootstrap_values, reward_model_output
                        )

                        # Accumulate rewards (with γ=1 across macro steps)
                        if rewards is not None:
                            # rewards shape: [B, num_action_chunks] or [B, 1]
                            # Sum across chunk dim to get [B, 1]
                            step_reward = rewards.sum(dim=-1, keepdim=True) if rewards.dim() > 1 else rewards
                            acc_rewards[stage_id] += step_reward.cpu()
                        # Accumulate done/term/trunc from the action result.
                        if env_output.dones is not None:
                            acc_dones[stage_id] |= _pre_action_done_tensor(env_output.dones)
                        if env_output.terminations is not None:
                            acc_terminations[stage_id] |= _pre_action_done_tensor(env_output.terminations)
                        if env_output.truncations is not None:
                            acc_truncations[stage_id] |= _pre_action_done_tensor(env_output.truncations)

                        # Terminal decisions must be flushed immediately after the
                        # env step that completed the episode. Once GenArk marks a
                        # slot done, the next obs is dormant (empty instruction), so
                        # waiting for the next decision boundary can leave the STOP /
                        # timeout reward detached from the action that caused it.
                        terminal_now = (
                            _pre_action_done_tensor(env_output.dones)
                            | _pre_action_done_tensor(env_output.terminations)
                            | _pre_action_done_tensor(env_output.truncations)
                        )
                        for env_i in range(self.train_num_envs_per_stage):
                            if bool(terminal_now[env_i].item()):
                                _flush_pending_decision(stage_id, env_i)

                        # Non-terminal decisions stay pending while cached low-level
                        # actions replay, accumulating rewards into acc_rewards. They
                        # are flushed at the next LLM decision boundary or bootstrap.

                        env_batch = env_output.to_dict()
                        next_task_descs = env_batch["obs"].get("task_descriptions", None)
                        if next_task_descs is not None:
                            last_env_dormant[stage_id] = torch.tensor(
                                [desc == "" for desc in next_task_descs],
                                dtype=torch.bool,
                                device=_cpu,
                            )
                        else:
                            last_env_dormant[stage_id] = torch.zeros(
                                self.train_num_envs_per_stage,
                                dtype=torch.bool,
                                device=_cpu,
                            )
                        # Global termination signal for synchronized rollout-rank exit.
                        # The env worker is the sole coordinator: it sees every stage's
                        # decision_counts/dormant state (the full global batch), so it
                        # computes the SAME all-done condition the rollout used to derive
                        # locally — but globally — and broadcasts it to all rollout ranks
                        # via the obs. This makes every rank enter bootstrap on the same
                        # step, so prev_logprobs is never None on one shard and a tensor on
                        # another (heterogeneous multi-scene termination otherwise crashes
                        # RolloutResult.merge). Single-scene: global == local, behavior is
                        # byte-identical to before.
                        should_terminate = all(
                            not (
                                (decision_counts[s] < max_dec) & (~last_env_dormant[s])
                            ).any()
                            for s in range(self.stage_num)
                        )
                        env_batch["obs"]["should_terminate"] = torch.full(
                            (self.train_num_envs_per_stage,),
                            bool(should_terminate),
                            dtype=torch.bool,
                            device=_cpu,
                        )
                        env_batch["obs"]["ppo_loss_mask_mode_id"] = torch.full(
                            (self.train_num_envs_per_stage,),
                            int(self._ppo_loss_mask_mode_id()),
                            dtype=torch.long,
                            device=_cpu,
                        )
                        self.send_env_batch(
                            rollout_channel,
                            {
                                "obs": env_batch["obs"],
                                "final_obs": env_batch["final_obs"],
                            },
                        )
                        env_outputs[stage_id] = env_output
                        self.record_env_metrics(env_metrics, env_info, epoch)

                    if received_bootstrap:
                        break
                    step += 1

                # Bootstrap step: consume one more rollout result for value estimation.
                # Skipped (via early break) when is_last_decision was received inline.
                for stage_id in range(self.stage_num):
                    if received_bootstrap:
                        break
                    env_output = env_outputs[stage_id]
                    if env_output.intervene_actions is not None:
                        # Apply any pending interventions to the last recorded action
                        for env_i in range(self.train_num_envs_per_stage):
                            if len(rollout_results_per_env[stage_id][env_i].actions) > 0:
                                rollout_results_per_env[stage_id][env_i].update_last_actions(
                                    env_output.intervene_actions[env_i:env_i + 1],
                                    env_output.intervene_flags[env_i:env_i + 1],
                                )

                    for env_i in range(self.train_num_envs_per_stage):
                        _flush_pending_decision(stage_id, env_i)

                    reward_model_output = None
                    if reward_channel is not None:
                        last_run = epoch == self.rollout_epoch - 1
                        reward_model_output = self.get_reward_model_output(
                            env_output,
                            send_channel=reward_channel,
                            recv_channel=input_channel,
                            last_run=last_run,
                        )
                        if reward_model_output is not None:
                            env_metrics["reward_model_output"].append(
                                reward_model_output.detach().float().reshape(-1).cpu()
                            )
                    rollout_result = self.recv_rollout_results(input_channel, mode="train")
                    rewards = self.compute_bootstrap_rewards(
                        env_output, rollout_result.bootstrap_values, reward_model_output
                    )
                    # Bootstrap step: only dones/truncations/terminations (no rewards).
                    # Mirrors the step-based path where the initial bootstrap obs has
                    # rewards=None (no action yet) → not appended, making dones have
                    # exactly one more entry than rewards for GAE computation.
                    # Result: rewards[0..D-1] (D entries), dones[0..D] (D+1 entries).
                    for env_i in range(self.train_num_envs_per_stage):
                        bootstrap_step_result = ChunkStepResult(
                            prev_values=(
                                rollout_result.prev_values[env_i:env_i + 1]
                                if self.collect_prev_infos and rollout_result.prev_values is not None
                                else None
                            ),
                            dones=env_output.dones[env_i:env_i + 1] if env_output.dones is not None else None,
                            truncations=env_output.truncations[env_i:env_i + 1] if env_output.truncations is not None else None,
                            terminations=env_output.terminations[env_i:env_i + 1] if env_output.terminations is not None else None,
                            # rewards intentionally omitted: bootstrap provides terminal done for GAE only.
                        )
                        rollout_results_per_env[stage_id][env_i].append_step_result(
                            bootstrap_step_result
                        )

                # Pad short trajectories (dormant envs or early-termination envs)
                # to exactly max_dec decision steps so rollout_size = max_dec*B is stable.
                # Blank padding steps:
                #   - dones=True → compute_loss_mask masks them out (loss_mask=False), so
                #     they contribute 0 to PPO loss and are skipped in recompute.
                #   - forward_inputs / actions / prev_logprobs / etc are cloned from the
                #     last real decision so all tensor lists have len = max_dec. This keeps
                #     forward_inputs and rewards dim-aligned (required by
                #     process_nested_dict_for_train's reshape+shuffle_id indexing).
                #   - Cloning (rather than zero-filling pixel_values) avoids ~23GB CPU
                #     memory blowup for blank pixel tensors.
                for stage_id in range(self.stage_num):
                    refs = [r for r in rollout_results_per_env[stage_id] if r.actions]
                    if not refs:
                        print(
                            "[EnvWorker][decision-rollout] no valid decisions for "
                            f"stage {stage_id}; skipping this stage."
                        )
                        continue
                    stage_ref = refs[0]

                    for env_i in range(self.train_num_envs_per_stage):
                        result = rollout_results_per_env[stage_id][env_i]
                        n_decisions = len(result.actions) if result.actions else 0
                        ref_source = result if n_decisions > 0 else stage_ref
                        ref_fi = ref_source.forward_inputs[-1] if ref_source.forward_inputs else None
                        ref_act = ref_source.actions[-1] if ref_source.actions else None
                        ref_iflag = (
                            ref_source.intervene_flags[-1]
                            if ref_source.intervene_flags
                            else None
                        )
                        ref_lp = ref_source.prev_logprobs[-1] if ref_source.prev_logprobs else None
                        ref_pv = ref_source.prev_values[-1] if ref_source.prev_values else None
                        ref_ver = ref_source.versions[-1] if ref_source.versions else None
                        while n_decisions < max_dec:
                            blank = ChunkStepResult(
                                # dones=True → loss_mask=False for this and all later steps.
                                dones=torch.ones(1, 1, dtype=torch.bool),
                                truncations=torch.zeros(1, 1, dtype=torch.bool),
                                terminations=torch.ones(1, 1, dtype=torch.bool),
                                rewards=torch.zeros(1, 1),
                                actions=ref_act.clone() if ref_act is not None else None,
                                prev_logprobs=ref_lp.clone() if ref_lp is not None else None,
                                prev_values=ref_pv.clone() if ref_pv is not None else None,
                                versions=ref_ver.clone() if ref_ver is not None else None,
                                forward_inputs={k: v.clone() for k, v in ref_fi.items()} if ref_fi else None,
                            )
                            result.append_step_result(blank)
                            # append_step_result auto-appends zeros_like(actions) for intervene_flags;
                            # overwrite with the real ref so all lists stay aligned.
                            if ref_iflag is not None and result.intervene_flags:
                                result.intervene_flags[-1] = ref_iflag.clone()
                            n_decisions += 1

            else:
                # ── Original step-based rollout loop (backward compat) ──────
                for chunk_step_idx in range(self.n_train_chunk_steps):
                    for stage_id in range(self.stage_num):
                        if cooperative_yield:
                            await asyncio.sleep(0)

                        env_output = env_outputs[stage_id]
                        curr_obs = env_output.obs
                        if env_output.intervene_actions is not None:
                            self.rollout_results[stage_id].update_last_actions(
                                env_output.intervene_actions,
                                env_output.intervene_flags,
                            )

                        reward_model_output = None
                        if reward_channel is not None and chunk_step_idx != 0:
                            reward_model_output = self.get_reward_model_output(
                                env_output,
                                send_channel=reward_channel,
                                recv_channel=input_channel,
                            )
                            if reward_model_output is not None:
                                env_metrics["reward_model_output"].append(
                                    reward_model_output.detach().float().reshape(-1).cpu()
                                )

                        rollout_result = self.recv_rollout_results(
                            input_channel, mode="train"
                        )
                        rewards = self.compute_bootstrap_rewards(
                            env_output, rollout_result.bootstrap_values, reward_model_output
                        )
                        chunk_step_result = ChunkStepResult(
                            actions=rollout_result.forward_inputs.get("action", None),
                            prev_logprobs=(
                                rollout_result.prev_logprobs
                                if self.collect_prev_infos
                                else None
                            ),
                            prev_values=(
                                rollout_result.prev_values
                                if self.collect_prev_infos
                                else None
                            ),
                            forward_inputs=rollout_result.forward_inputs,
                            versions=rollout_result.versions,
                            dones=env_output.dones,
                            truncations=env_output.truncations,
                            terminations=env_output.terminations,
                            rewards=rewards,
                        )
                        self.rollout_results[stage_id].append_step_result(chunk_step_result)
                        if rollout_result.save_flags is not None:
                            self.rollout_results[stage_id].mark_last_step_with_flags(
                                rollout_result.save_flags
                            )

                        env_output, env_info = self.env_interact_step(
                            rollout_result.actions, stage_id
                        )
                        env_batch = env_output.to_dict()
                        self.send_env_batch(
                            rollout_channel,
                            {
                                "obs": env_batch["obs"],
                                "final_obs": env_batch["final_obs"],
                            },
                        )
                        if self.collect_transitions:
                            next_obs = (
                                env_output.final_obs
                                if env_output.dones.any() and self.cfg.env.train.auto_reset
                                else env_output.obs
                            )
                            self.rollout_results[stage_id].append_transitions(
                                curr_obs, next_obs
                            )

                        env_outputs[stage_id] = env_output
                        self.record_env_metrics(env_metrics, env_info, epoch)

                for stage_id in range(self.stage_num):
                    env_output = env_outputs[stage_id]
                    if env_output.intervene_actions is not None:
                        self.rollout_results[stage_id].update_last_actions(
                            env_output.intervene_actions,
                            env_output.intervene_flags,
                        )

                    reward_model_output = None
                    if reward_channel is not None:
                        last_run = epoch == self.rollout_epoch - 1
                        reward_model_output = self.get_reward_model_output(
                            env_output,
                            send_channel=reward_channel,
                            recv_channel=input_channel,
                            last_run=last_run,
                        )
                        if reward_model_output is not None:
                            env_metrics["reward_model_output"].append(
                                reward_model_output.detach().float().reshape(-1).cpu()
                            )
                    rollout_result = self.recv_rollout_results(input_channel, mode="train")
                    rewards = self.compute_bootstrap_rewards(
                        env_output, rollout_result.bootstrap_values, reward_model_output
                    )
                    chunk_step_result = ChunkStepResult(
                        prev_values=(
                            rollout_result.prev_values if self.collect_prev_infos else None
                        ),
                        dones=env_output.dones,
                        truncations=env_output.truncations,
                        terminations=env_output.terminations,
                        rewards=rewards,
                    )
                    self.rollout_results[stage_id].append_step_result(chunk_step_result)

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        if actor_channel is not None:
            if use_decision_rollout:
                # Merge per-env trajectories into a single EmbodiedRolloutResult per stage,
                # then send as normal.  Each env has max_dec decision steps + 1 bootstrap
                # appended to its EmbodiedRolloutResult list.  We merge by building a
                # combined EmbodiedRolloutResult where each list entry is a batch tensor
                # formed by stacking across envs.
                for stage_id in range(self.stage_num):
                    # Convert each per-env result to a Trajectory, then cat across envs
                    # using to_splited_trajectories with split_num=1 to get a single batch.
                    # Simpler: build a merged EmbodiedRolloutResult by cat-ing per-env lists.
                    per_env_results = rollout_results_per_env[stage_id]
                    n_envs = len(per_env_results)
                    if n_envs == 0:
                        continue
                    if not any(r.actions for r in per_env_results):
                        continue
                    ref_result = next(r for r in per_env_results if r.actions)
                    n_steps = int(max_dec)
                    assert all(len(r.rewards) == n_steps for r in per_env_results), (
                        "decision rollout padding incomplete: rewards length mismatch"
                    )
                    assert all(len(r.actions) == n_steps for r in per_env_results), (
                        "decision rollout padding incomplete: actions length mismatch"
                    )
                    assert all(
                        len(r.forward_inputs) == n_steps for r in per_env_results
                    ), "decision rollout padding incomplete: forward_inputs length mismatch"
                    if ref_result.prev_logprobs:
                        assert all(
                            len(r.prev_logprobs) == n_steps for r in per_env_results
                        ), "decision rollout padding incomplete: prev_logprobs length mismatch"
                    if ref_result.versions:
                        assert all(
                            len(r.versions) == n_steps for r in per_env_results
                        ), "decision rollout padding incomplete: versions length mismatch"

                    selected_diag_snapshot = list(
                        completed_episode_diag_by_stage[stage_id].values()
                    )
                    per_env_results, selected_diag_snapshot = (
                        self._select_candidate_group(
                            env_metrics,
                            stage_id,
                            per_env_results,
                            selected_diag_snapshot,
                        )
                    )
                    n_envs = len(per_env_results)
                    if n_envs == 0:
                        continue
                    ref_result = next(r for r in per_env_results if r.actions)

                    self._record_grpo_group_diagnostics(
                        env_metrics,
                        stage_id,
                        per_env_results,
                        selected_diag_snapshot,
                    )

                    merged = EmbodiedRolloutResult(
                        max_episode_length=self.cfg.env.train.max_episode_steps,
                    )
                    # Stack each time step across envs to form batch tensors.
                    # rewards: n_steps entries. dones: n_steps+1 entries (bootstrap done).
                    # This matches the step-based path invariant required by
                    # compute_loss_mask and preprocess_embodied_advantages_inputs.
                    for t in range(n_steps):
                        rewards_t = [
                            r.rewards[t] if t < len(r.rewards) else torch.zeros(1, 1)
                            for r in per_env_results
                        ]
                        dones_t = [
                            r.dones[t] if t < len(r.dones) else torch.zeros(1, 1, dtype=torch.bool)
                            for r in per_env_results
                        ]
                        terminations_t = [
                            r.terminations[t] if t < len(r.terminations) else torch.zeros(1, 1, dtype=torch.bool)
                            for r in per_env_results
                        ]
                        truncations_t = [
                            r.truncations[t] if t < len(r.truncations) else torch.zeros(1, 1, dtype=torch.bool)
                            for r in per_env_results
                        ]
                        merged.rewards.append(torch.cat(rewards_t, dim=0))
                        merged.dones.append(torch.cat(dones_t, dim=0))
                        merged.terminations.append(torch.cat(terminations_t, dim=0))
                        merged.truncations.append(torch.cat(truncations_t, dim=0))

                        # actions / prev_logprobs / versions / forward_inputs
                        if ref_result.actions and t < len(ref_result.actions):
                            actions_t = [
                                r.actions[t] if t < len(r.actions) else torch.zeros_like(ref_result.actions[0])
                                for r in per_env_results
                            ]
                            merged.actions.append(torch.cat(actions_t, dim=0))
                            merged.intervene_flags.append(
                                torch.cat(
                                    [
                                        r.intervene_flags[t] if t < len(r.intervene_flags) else torch.zeros_like(ref_result.intervene_flags[0])
                                        for r in per_env_results
                                    ],
                                    dim=0,
                                )
                            )

                        if ref_result.prev_logprobs and t < len(ref_result.prev_logprobs):
                            lp_t = [
                                r.prev_logprobs[t] if t < len(r.prev_logprobs) else torch.zeros_like(ref_result.prev_logprobs[0])
                                for r in per_env_results
                            ]
                            merged.prev_logprobs.append(torch.cat(lp_t, dim=0))

                        if ref_result.prev_values and t < len(ref_result.prev_values):
                            pv_t = [
                                r.prev_values[t] if t < len(r.prev_values) else torch.zeros_like(ref_result.prev_values[0])
                                for r in per_env_results
                            ]
                            merged.prev_values.append(torch.cat(pv_t, dim=0))

                        if ref_result.versions and t < len(ref_result.versions):
                            ver_t = [
                                r.versions[t] if t < len(r.versions) else torch.zeros_like(ref_result.versions[0])
                                for r in per_env_results
                            ]
                            merged.versions.append(torch.cat(ver_t, dim=0))

                        if ref_result.forward_inputs and t < len(ref_result.forward_inputs):
                            fi_t = {}
                            ref_fi = ref_result.forward_inputs[0]
                            for k in ref_fi.keys():
                                fi_t[k] = torch.cat(
                                    [
                                        r.forward_inputs[t][k] if t < len(r.forward_inputs) else torch.zeros_like(ref_fi[k])
                                        for r in per_env_results
                                    ],
                                    dim=0,
                                )
                            merged.forward_inputs.append(fi_t)

                    # Append the extra bootstrap done (index n_steps) so dones has n_steps+1
                    # entries — required by compute_loss_mask(dones) which uses shape[0]-1.
                    bootstrap_dones_t = [
                        r.dones[n_steps] if n_steps < len(r.dones) else torch.zeros(1, 1, dtype=torch.bool)
                        for r in per_env_results
                    ]
                    bootstrap_terms_t = [
                        r.terminations[n_steps] if n_steps < len(r.terminations) else torch.zeros(1, 1, dtype=torch.bool)
                        for r in per_env_results
                    ]
                    bootstrap_trunc_t = [
                        r.truncations[n_steps] if n_steps < len(r.truncations) else torch.zeros(1, 1, dtype=torch.bool)
                        for r in per_env_results
                    ]
                    merged.dones.append(torch.cat(bootstrap_dones_t, dim=0))
                    merged.terminations.append(torch.cat(bootstrap_terms_t, dim=0))
                    merged.truncations.append(torch.cat(bootstrap_trunc_t, dim=0))

                    await self.send_rollout_trajectories(merged, actor_channel)
            else:
                for stage_id in range(self.stage_num):
                    await self.send_rollout_trajectories(
                        self.rollout_results[stage_id], actor_channel
                    )

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None = None,
    ):
        env_metrics = await self._run_interact_once(
            input_channel,
            rollout_channel,
            reward_channel,
            actor_channel,
            cooperative_yield=False,
        )

        for env in self.env_list:
            if self.enable_offload and hasattr(env, "offload"):
                env.offload()

        return env_metrics

    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        eval_metrics = defaultdict(list)

        for eval_rollout_epoch in range(self.cfg.algorithm.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    self.eval_prev_done[stage_id] = torch.zeros(
                        self.eval_num_envs_per_stage, dtype=torch.bool
                    )
                    extracted_obs, infos = self.eval_env_list[stage_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=(
                            infos["final_observation"]
                            if "final_observation" in infos
                            else None
                        ),
                    )
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        rollout_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(
                        input_channel, mode="eval"
                    )
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch
                            == self.cfg.algorithm.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        ):
                            continue
                    else:
                        if eval_step == self.n_eval_chunk_steps - 1:
                            continue
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        rollout_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                    )

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            if self.cfg.env.eval.get("enable_offload", False) and hasattr(
                self.eval_env_list[stage_id], "offload"
            ):
                self.eval_env_list[stage_id].offload()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

    def get_actor_split_num(self):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

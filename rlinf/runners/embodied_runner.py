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

import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Union

from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor",
            "EmbodiedNFTFSDPPolicy",
            "EmbodiedSACFSDPPolicy",
            "AsyncEmbodiedSACFSDPPolicy",
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        reward: Union["EmbodiedRewardWorker"] = None,
        critic=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward
        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        if self.reward is not None:
            self.reward_channel = Channel.create("Reward")
        else:
            self.reward_channel = None

        # this timer checks if we should stop training
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )

        # Async logging setup
        self.stop_logging = False
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

    def _log_worker(self):
        """Background thread for processing log messages."""
        while not self.stop_logging:
            try:
                # Wait for log message with timeout
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logging error: {e}")
                continue

    def print_metrics_table_async(
        self,
        step: int,
        total_steps: int,
        start_time: float,
        metrics: dict,
        start_step: int = 0,
    ):
        """Async version that puts table printing in queue."""
        self.log_queue.put(
            (print_metrics_table, (step, total_steps, start_time, metrics, start_step))
        )

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()
        if self.reward is not None:
            self.reward.init_worker().wait()

        rollout_handle.wait()
        env_handle.wait()
        if self.actor is not None:
            self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info(f"Resuming training from checkpoint directory {resume_dir}.")
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        for rank, metrics in enumerate(metrics_list):
            if not metrics:
                continue
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    def _aggregate_numeric_metrics(self, metrics_list: list[dict] | None) -> dict:
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    def _process_ranked_eval_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = (
            compute_evaluate_metrics(metric_list) if metric_list else {}
        )
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = compute_evaluate_metrics(metrics_list)
        return aggregated_metrics, ranked_metrics_list

    def _reinforce_ada_cfg(self):
        cfg = self.cfg.algorithm.get("reinforce_ada", None)
        if cfg is None or not bool(cfg.get("enabled", False)):
            return None
        return cfg

    def _validate_reinforce_ada_setup(self, cfg) -> None:
        """Keep v1's channel protocol intentionally narrow and group-safe."""
        if self.reward is not None:
            raise ValueError("Reinforce-Ada v1 does not support an external reward worker")
        if bool(self.cfg.env.train.get("enable_offload", False)):
            raise ValueError("Reinforce-Ada v1 does not support env.enable_offload")
        if self.cfg.rollout.pipeline_stage_num != 1:
            raise ValueError("Reinforce-Ada v1 requires rollout.pipeline_stage_num=1")
        if len(self.env.worker_info_list) != 1 or len(self.actor.worker_info_list) != 1:
            raise ValueError(
                "Reinforce-Ada v1 requires one EnvWorker and one actor rank so a "
                "success-anchored GRPO group is never split across channels."
            )
        overfit_cfg = self.cfg.env.train.get("episode_overfit", None)
        if bool(cfg.get("only_episode_overfit", True)) and not bool(
            overfit_cfg and overfit_cfg.get("enabled", False)
        ):
            raise ValueError("Reinforce-Ada v1 requires env.train.episode_overfit.enabled=true")
        if not bool(cfg.get("skip_on_exhaustion", True)):
            raise ValueError("Reinforce-Ada v1 currently requires skip_on_exhaustion=true")
        group_size = int(self.cfg.algorithm.group_size)
        train_group_size = int(cfg.get("train_group_size", group_size))
        if group_size != train_group_size or train_group_size != 4:
            raise ValueError(
                "Reinforce-Ada v1 requires algorithm.group_size=train_group_size=4"
            )
        mode = str(cfg.get("mode", "adaptive_pos"))
        total_envs = int(self.cfg.env.train.total_num_envs)
        if mode == "static_pos":
            expected = int(cfg.get("initial_candidates", total_envs))
        else:
            expected = int(cfg.get("initial_candidates", 4))
            retry_batch_size = int(cfg.get("retry_batch_size", expected))
            if retry_batch_size != expected:
                raise ValueError(
                    "Reinforce-Ada v1 uses a fixed physical rollout batch; "
                    "retry_batch_size must equal initial_candidates."
                )
        if total_envs != expected:
            raise ValueError(
                f"Reinforce-Ada {mode} requires env.train.total_num_envs={expected}; "
                f"got {total_envs}."
            )

    @staticmethod
    def _merge_attempt_env_metrics(attempt_results: list[dict]) -> dict:
        metric_dicts = [result for result in attempt_results if result]
        return compute_evaluate_metrics(metric_dicts) if metric_dicts else {}

    @staticmethod
    def _curriculum_checkpoint_requested(env_results: list[dict]) -> bool:
        for result in env_results:
            if not isinstance(result, dict):
                continue
            value = result.get("rft_curriculum_advance_pending")
            if value is None:
                continue
            try:
                if bool((value > 0).any().item()):
                    return True
            except (AttributeError, TypeError):
                if bool(value):
                    return True
        return False

    def _finish_run(self) -> None:
        self.metric_logger.finish()
        self.stop_logging = True
        self.log_queue.join()
        self.log_thread.join(timeout=1.0)

    def _run_reinforce_ada(self, cfg) -> None:
        """Collect success-anchored candidates before each actor update.

        An exhausted pool deliberately consumes rollout compute without advancing
        ``global_step``. This preserves the invariant that one RL step means one
        actual actor update, not one failed attempt to find a positive example.
        """
        self._validate_reinforce_ada_setup(cfg)
        start_time = time.time()
        start_step = self.global_step
        exhausted_groups = 0
        max_exhausted_groups = int(cfg.get("max_exhausted_groups", 10))

        while self.global_step < self.max_steps:
            step = self.global_step
            self.actor.set_global_step(step)
            self.rollout.set_global_step(step)
            self.env.set_global_step(step)

            if step % self.weight_sync_interval == 0:
                with self.timer("sync_weights"):
                    self.update_rollout_weights()

            attempt_results: list[dict] = []
            while True:
                # Retry samples must be fresh draws from the same frozen policy.
                self.rollout.set_sampling_seed_offset(
                    step * 10_000 + len(attempt_results)
                ).wait()
                attempt_idx = len(attempt_results) + 1
                with self.timer(f"generate_rollouts_attempt_{attempt_idx}"):
                    env_handle: Handle = self.env.interact(
                        input_channel=self.env_channel,
                        rollout_channel=self.rollout_channel,
                        reward_channel=None,
                        actor_channel=None,
                        candidate_collection=True,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    payloads = env_handle.wait()
                    rollout_handle.wait()

                payloads = [payload for payload in payloads if payload is not None]
                if len(payloads) != 1:
                    raise RuntimeError(
                        "Reinforce-Ada v1 expected exactly one EnvWorker outcome, "
                        f"got {len(payloads)}"
                    )
                payload = payloads[0]
                if not isinstance(payload, dict) or "candidate_outcome" not in payload:
                    raise RuntimeError("EnvWorker did not return a Reinforce-Ada outcome")
                attempt_results.append(payload.get("metrics", {}))
                outcome = payload["candidate_outcome"]
                status = str(outcome.get("status", ""))
                if status == "retry":
                    self.env.reset_reinforce_ada_candidates().wait()
                    continue
                if status not in {"ready_train", "exhausted"}:
                    raise RuntimeError(f"Unknown Reinforce-Ada candidate status: {status!r}")
                break

            env_metrics_raw = self._merge_attempt_env_metrics(attempt_results)
            env_metrics = {f"env/{key}": value for key, value in env_metrics_raw.items()}

            if status == "exhausted":
                exhausted_groups += 1
                self.env.discard_reinforce_ada_pool().wait()
                skip_durations = self.timer.consume_durations()
                rollout_attempt_keys = [
                    key
                    for key in skip_durations
                    if key.startswith("generate_rollouts_attempt_")
                ]
                if rollout_attempt_keys:
                    skip_durations["generate_rollouts"] = sum(
                        skip_durations.pop(key) for key in rollout_attempt_keys
                    )
                skip_time_metrics = {
                    f"time/{key}": value for key, value in skip_durations.items()
                }
                skip_metrics = {
                    **env_metrics,
                    **skip_time_metrics,
                    "grpo/group_skipped_rate": 1.0,
                    "grpo/hard_prompt_rate": 1.0,
                    "grpo/exhausted_groups": float(exhausted_groups),
                }
                self.metric_logger.log(skip_metrics, step)
                print(
                    "[ReinforceAda][skip] "
                    f"rl_step={step} exhausted_groups={exhausted_groups}/{max_exhausted_groups} "
                    f"attempts={outcome.get('candidate_attempts')} "
                    f"candidates={outcome.get('candidate_count')}",
                    flush=True,
                )
                if exhausted_groups >= max_exhausted_groups:
                    print("[ReinforceAda] exhausted-group budget reached; ending validation run.", flush=True)
                    break
                self.env.reset_reinforce_ada_candidates().wait()
                continue

            exhausted_groups = 0
            actor_recv_handle: Handle = self.actor.recv_rollout_trajectories(
                input_channel=self.actor_channel
            )
            self.env.emit_reinforce_ada_trajectories(self.actor_channel).wait()
            actor_recv_handle.wait()

            with self.timer("cal_adv_and_returns"):
                actor_rollout_metrics = self.actor.compute_advantages_and_returns().wait()
            actor_training_handle: Handle = self.actor.run_training()
            actor_training_metrics = actor_training_handle.wait()
            self.global_step += 1

            rollout_metrics = {
                f"rollout/{key}": value
                for key, value in self._aggregate_numeric_metrics(actor_rollout_metrics).items()
            }
            training_metrics = {
                f"train/{key}": value
                for key, value in self._aggregate_numeric_metrics(actor_training_metrics).items()
            }
            durations = self.timer.consume_durations()
            rollout_attempt_keys = [
                key for key in durations if key.startswith("generate_rollouts_attempt_")
            ]
            if rollout_attempt_keys:
                durations["generate_rollouts"] = sum(
                    durations.pop(key) for key in rollout_attempt_keys
                )
            time_metrics = {f"time/{key}": value for key, value in durations.items()}
            self.metric_logger.log(env_metrics, step)
            self.metric_logger.log(rollout_metrics, step)
            self.metric_logger.log(training_metrics, step)
            self.metric_logger.log(time_metrics, step)
            self.print_metrics_table_async(
                step,
                self.max_steps,
                start_time,
                {**env_metrics, **rollout_metrics, **training_metrics, **time_metrics},
                start_step,
            )

            run_val, save_model, _ = check_progress(
                self.global_step,
                self.max_steps,
                self.cfg.runner.val_check_interval,
                self.cfg.runner.save_interval,
                1.0,
                run_time_exceeded=False,
            )
            if run_val:
                self.update_rollout_weights()
                eval_metrics = {f"eval/{key}": value for key, value in self.evaluate().items()}
                self.metric_logger.log(eval_metrics, step)
            if save_model:
                self._save_checkpoint()

        self._finish_run()

    def run(self):
        if self.cfg.runner.get("only_eval", False):
            if (
                self.cfg.runner.get("resume_dir", None)
                and self.actor is not None
            ):
                # Load FSDP weights into vLLM before the frozen rollout.
                self.update_rollout_weights()
                # Keep eval sampling identity independent of the resume step
                # so checkpoint-1200 and global_step_N share the same seeds.
                self.global_step = 0
                self.rollout.set_global_step(0)
            eval_metrics = self.evaluate()
            eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
            self.metric_logger.log(data=eval_metrics, step=0)

            # Save JSON results (mirrors original GenArk avg_metrics.json)
            log_path = self.cfg.runner.logger.get("log_path", ".")
            os.makedirs(log_path, exist_ok=True)
            json_path = os.path.join(log_path, "avg_metrics.json")
            json_data = {k: float(v) if hasattr(v, "item") else v
                         for k, v in eval_metrics.items()}
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)
            print(f"\n[RLinf] Eval metrics saved → {json_path}", flush=True)
            for k, v in json_data.items():
                print(f"  {k}: {v}", flush=True)

            self.metric_logger.finish()
            self.stop_logging = True
            self.log_queue.join()
            self.log_thread.join(timeout=1.0)
            return

        reinforce_ada_cfg = self._reinforce_ada_cfg()
        if reinforce_ada_cfg is not None:
            return self._run_reinforce_ada(reinforce_ada_cfg)

        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            self.env.set_global_step(self.global_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                with self.timer("generate_rollouts"):
                    env_handle: Handle = self.env.interact(
                        input_channel=self.env_channel,
                        rollout_channel=self.rollout_channel,
                        reward_channel=self.reward_channel,
                        actor_channel=self.actor_channel,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    if self.reward is not None:
                        reward_handle: Handle = self.reward.compute_rewards(
                            input_channel=self.reward_channel,
                            output_channel=self.env_channel,
                        )
                    self.actor.recv_rollout_trajectories(
                        input_channel=self.actor_channel
                    ).wait()
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                # actor training.
                actor_training_handle: Handle = self.actor.run_training()

                actor_training_metrics = actor_training_handle.wait()

                self.global_step += 1
                env_results = env_handle.wait()
                curriculum_checkpoint = self._curriculum_checkpoint_requested(
                    env_results
                )

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                eval_metrics = {}
                if run_val:
                    with self.timer("eval"):
                        self.update_rollout_weights()
                        eval_metrics = self.evaluate()
                        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        self.metric_logger.log(data=eval_metrics, step=_step)
                        # Append per-step eval metrics to training_metrics.json for
                        # easy inspection of training progress over RL iterations.
                        _log_path = self.cfg.runner.logger.get("log_path", ".")
                        _metrics_json = os.path.join(_log_path, "training_metrics.json")
                        _entry = {"step": _step, **{k: float(v) for k, v in eval_metrics.items()}}
                        _history = []
                        if os.path.exists(_metrics_json):
                            with open(_metrics_json) as _f:
                                try:
                                    _history = json.load(_f)
                                except json.JSONDecodeError:
                                    _history = []
                        _history.append(_entry)
                        with open(_metrics_json, "w") as _f:
                            json.dump(_history, _f, indent=2)

                if curriculum_checkpoint and not save_model:
                    self.logger.info(
                        "Saving checkpoint at curriculum advancement boundary."
                    )
                if save_model or curriculum_checkpoint:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            env_time_metrics, env_time_metrics_per_rank = env_handle.consume_durations(
                return_per_rank=True
            )
            rollout_time_metrics, rollout_time_metrics_per_rank = (
                rollout_handle.consume_durations(return_per_rank=True)
            )
            actor_time_metrics, actor_time_metrics_per_rank = (
                actor_training_handle.consume_durations(return_per_rank=True)
            )
            time_metrics.update(
                {f"time/env/{k}": v for k, v in env_time_metrics.items()}
            )
            time_metrics.update(
                {f"time/rollout/{k}": v for k, v in rollout_time_metrics.items()}
            )
            time_metrics.update(
                {f"time/actor/{k}": v for k, v in actor_time_metrics.items()}
            )
            if self.reward is not None:
                reward_time_metrics, reward_time_metrics_per_rank = (
                    reward_handle.consume_durations(return_per_rank=True)
                )
                time_metrics.update(
                    {f"time/reward/{k}": v for k, v in reward_time_metrics.items()}
                )

            env_results_list = [
                results for results in env_results if results is not None
            ]
            env_metrics = compute_evaluate_metrics(env_results_list)
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            ranked_env_results = [
                {"rank": rank, "env": rank_metrics}
                for rank, rank_metrics in enumerate(env_results)
                if rank_metrics is not None
            ]
            _, env_metrics_per_rank = self._process_ranked_eval_results(
                ranked_env_results, metric_field="env"
            )

            rollout_metrics = {
                f"rollout/{k}": v
                for k, v in self._aggregate_numeric_metrics(
                    actor_rollout_metrics
                ).items()
            }
            training_metrics = {
                f"train/{k}": v
                for k, v in self._aggregate_numeric_metrics(
                    actor_training_metrics
                ).items()
            }

            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)
            self._log_ranked_metrics(
                metrics_list=actor_rollout_metrics,
                step=_step,
                prefix="rollout",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=actor_training_metrics,
                step=_step,
                prefix="train",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=actor_time_metrics_per_rank,
                step=_step,
                prefix="time/actor",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=rollout_time_metrics_per_rank,
                step=_step,
                prefix="time/rollout",
                worker_group_name=self.rollout.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_time_metrics_per_rank,
                step=_step,
                prefix="time/env",
                worker_group_name=self.env.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_metrics_per_rank,
                step=_step,
                prefix="env",
                worker_group_name=self.env.worker_group_name,
            )
            if self.reward is not None:
                self._log_ranked_metrics(
                    metrics_list=reward_time_metrics_per_rank,
                    step=_step,
                    prefix="time/reward",
                    worker_group_name=self.reward.worker_group_name,
                )

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)

            self.print_metrics_table_async(
                _step, self.max_steps, start_time, logging_metrics, start_step
            )

        self.metric_logger.finish()

        # Stop logging thread
        self.stop_logging = True
        self.log_queue.join()  # Wait for all queued logs to be processed
        self.log_thread.join(timeout=1.0)

    def _save_checkpoint(self):
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch

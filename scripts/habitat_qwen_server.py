"""Habitat 0.1.7 bridge for the RLinf QwenNavPolicy A/B evaluator.

This file is intentionally standalone.  It is executed by the Python 3.8
environment inside ``lavira:v4`` and must not import RLinf or modern vLLM.
The host-side adapter talks to it over a localhost TCP socket using a small
length-prefixed pickle protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import socket
import struct
import traceback
from pathlib import Path
from typing import Optional

import numpy as np


HEADER = struct.Struct("!I")


def _recv_exact(conn, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("client closed the Habitat bridge socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(conn):
    size = HEADER.unpack(_recv_exact(conn, HEADER.size))[0]
    if size > 512 * 1024 * 1024:
        raise ValueError("request payload is unexpectedly large")
    return pickle.loads(_recv_exact(conn, size))


def send_message(conn, payload) -> None:
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(HEADER.pack(len(raw)) + raw)


def _quat_yaw(rotation) -> float:
    """Habitat agent forward -> RLinf/Genesis yaw convention."""
    import habitat_sim

    forward = habitat_sim.utils.quat_rotate_vector(rotation, habitat_sim.geo.FRONT)
    return float(math.atan2(-float(forward[2]), float(forward[0])))


def _load_lavira_get_config(lavira_root):
    """Load LaViRA's config module without importing its model package.

    ``vlnce_baselines.__init__`` eagerly imports GroundingDINO and the full
    zero-shot evaluator.  The bridge needs only the Habitat config, so load
    that one module through a lightweight package stub instead.
    """
    import importlib.util
    import sys
    import types

    root = Path(lavira_root)
    package_name = "vlnce_baselines"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root / package_name)]
    sys.modules[package_name] = package

    config_name = package_name + ".config"
    config_package = types.ModuleType(config_name)
    config_package.__path__ = [str(root / package_name / "config")]
    sys.modules[config_name] = config_package

    module_name = config_name + ".default"
    spec = importlib.util.spec_from_file_location(
        module_name, str(root / package_name / "config/default.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.get_config


class HabitatQwenBridge:
    def __init__(self, args):
        self._worker_mode = bool(getattr(args, "worker_mode", False))
        self._pool_mode = not self._worker_mode and int(args.num_envs) > 1
        if self._pool_mode:
            # Habitat-Sim 0.1.7 cannot reliably own several EGL contexts in
            # one Python process.  Keep one TCP bridge but isolate each
            # simulator in its own subprocess, matching Habitat's SubprocEnv.
            import multiprocessing as mp

            self._args = args
            self._worker_conns = []
            self._worker_procs = []
            self._done = np.zeros(args.num_envs, dtype=bool)
            self._terminal_observation_emitted = np.zeros(args.num_envs, dtype=bool)
            self._elapsed = np.zeros(args.num_envs, dtype=np.int32)
            self._start_distance = np.full(args.num_envs, np.nan, dtype=np.float32)
            self._min_distance = np.full(args.num_envs, np.inf, dtype=np.float32)
            self._current_ids = [None] * args.num_envs
            self._current_trial_ids = [None] * args.num_envs
            self._metrics_path = args.metrics_path
            self._records = []
            context = mp.get_context("spawn")
            gpu_ids = [
                int(token) for token in str(args.gpu_ids).split(",")
                if token.strip()
            ] or [int(args.gpu_id)]
            for worker_index in range(args.num_envs):
                parent_conn, child_conn = context.Pipe()
                child_args = copy.copy(args)
                child_args.num_envs = 1
                child_args.worker_mode = True
                child_args.metrics_path = ""
                child_args.gpu_id = gpu_ids[worker_index % len(gpu_ids)]
                process = context.Process(
                    target=_bridge_worker_main,
                    args=(child_conn, child_args),
                    daemon=True,
                )
                process.start()
                self._worker_conns.append(parent_conn)
                self._worker_procs.append(process)
            for conn in self._worker_conns:
                status, payload = conn.recv()
                if status != "ready":
                    raise RuntimeError("Habitat worker failed during init: %s" % payload)
            return

        # Registration must happen before get_config/make_dataset.
        import habitat_extensions  # noqa: F401
        import habitat
        get_config = _load_lavira_get_config(args.lavira_root)

        self._habitat = habitat
        self._args = args
        config_path = str(Path(args.lavira_root) / "vlnce_baselines/config/r2r.yaml")
        previous_cwd = os.getcwd()
        os.chdir(args.lavira_root)
        try:
            cfg = get_config(config_path)
        finally:
            os.chdir(previous_cwd)
        task_cfg = cfg.TASK_CONFIG.clone()
        task_cfg.defrost()
        task_cfg.DATASET.DATA_PATH = args.dataset_path
        task_cfg.DATASET.SCENES_DIR = args.scenes_dir
        task_cfg.DATASET.SPLIT = args.split
        # LaViRA's default config keeps a development-only EPISODES_ALLOWED
        # filter (episode 701).  The A/B bridge must use the requested file
        # without silently dropping episode 259 or the full val split.
        if hasattr(task_cfg.DATASET, "EPISODES_ALLOWED"):
            task_cfg.DATASET.EPISODES_ALLOWED = None
        gt_path = args.gt_path or str(
            Path(args.dataset_path).parent / (args.split + "_gt.json.gz")
        )
        task_cfg.TASK.NDTW.GT_PATH = gt_path
        task_cfg.TASK.SDTW.GT_PATH = gt_path
        task_cfg.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_steps
        task_cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
        task_cfg.SIMULATOR.FORWARD_STEP_SIZE = args.forward_step
        task_cfg.SIMULATOR.TURN_ANGLE = args.turn_angle
        task_cfg.SIMULATOR.RGB_SENSOR.WIDTH = args.width
        task_cfg.SIMULATOR.RGB_SENSOR.HEIGHT = args.height
        task_cfg.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
        task_cfg.SIMULATOR.RGB_SENSOR.POSITION = [0.0, args.camera_height, 0.0]
        task_cfg.SIMULATOR.DEPTH_SENSOR.WIDTH = args.width
        task_cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.height
        task_cfg.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
        task_cfg.SIMULATOR.DEPTH_SENSOR.POSITION = [0.0, args.camera_height, 0.0]
        task_cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.1
        task_cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = args.depth_max
        # Habitat 0.1.7 exposes metric depth when normalization is disabled.
        if hasattr(task_cfg.SIMULATOR.DEPTH_SENSOR, "NORMALIZE_DEPTH"):
            task_cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        task_cfg.freeze()

        dataset = habitat.make_dataset(task_cfg.DATASET.TYPE, config=task_cfg.DATASET)
        self._episodes = {str(ep.episode_id): ep for ep in dataset.episodes}
        self._dataset = dataset
        self._task_cfg = task_cfg
        self._envs = []
        self._current_ids = [None] * args.num_envs
        self._current_trial_ids = [None] * args.num_envs
        self._done = np.zeros(args.num_envs, dtype=bool)
        self._terminal_observation_emitted = np.zeros(args.num_envs, dtype=bool)
        self._elapsed = np.zeros(args.num_envs, dtype=np.int32)
        self._start_distance = np.full(args.num_envs, np.nan, dtype=np.float32)
        self._min_distance = np.full(args.num_envs, np.inf, dtype=np.float32)
        self._metrics_path = args.metrics_path
        self._records = []
        self._scan_observations = [None] * args.num_envs
        self._last_terminal = [None] * args.num_envs
        for _ in range(args.num_envs):
            self._envs.append(None)

    def _worker_request(self, index, method, payload=None):
        conn = self._worker_conns[index]
        conn.send((method, payload or {}))
        status, result = conn.recv()
        if status != "ok":
            raise RuntimeError("Habitat worker %d failed: %s" % (index, result))
        return result

    def _worker_request_batch(self, method, payloads=None):
        """Run one bridge operation concurrently for every Habitat slot.

        GenArk's evaluator advances all slots in a scene batch at the same
        logical step.  The Habitat 0.1.7 compatibility layer keeps one EGL
        context per subprocess, so the parent must fan out the request first
        and collect second; a send/recv pair per slot would serialize the
        simulator step and change the batching semantics.
        """
        if payloads is None:
            payloads = [{} for _ in self._worker_conns]
        if len(payloads) != len(self._worker_conns):
            raise ValueError("batch payload count must match Habitat worker count")

        for conn, payload in zip(self._worker_conns, payloads):
            conn.send((method, payload or {}))

        results = []
        for index, conn in enumerate(self._worker_conns):
            status, result = conn.recv()
            if status != "ok":
                raise RuntimeError("Habitat worker %d failed: %s" % (index, result))
            results.append(result)
        return results

    @staticmethod
    def _merge_observations(per_env):
        return {
            # Each isolated worker returns a one-slot batch.  Concatenate that
            # batch dimension; stacking would introduce an erroneous [N, 1, ...]
            # axis and violate the host-side [N, ...] observation contract.
            "main_images": np.concatenate([x["main_images"] for x in per_env], axis=0),
            "extra_view_images": np.concatenate(
                [x["extra_view_images"] for x in per_env], axis=0
            ),
            "wrist_images": np.concatenate(
                [x["wrist_images"] for x in per_env], axis=0
            ),
            "states": np.concatenate([x["states"] for x in per_env], axis=0),
            "scan_images": np.concatenate([x["scan_images"] for x in per_env], axis=0),
            "scan_depth_images": np.concatenate(
                [x["scan_depth_images"] for x in per_env], axis=0
            ),
            "scan_states": np.concatenate([x["scan_states"] for x in per_env], axis=0),
            "scan_valid": np.concatenate([x["scan_valid"] for x in per_env], axis=0),
            "episode_active": np.concatenate(
                [x["episode_active"] for x in per_env], axis=0
            ),
            "task_descriptions": [x["task_descriptions"][0] for x in per_env],
        }

    def _make_env(self, episode_id: str):
        """Create a one-episode Habitat env with deterministic reset semantics."""
        import copy as _copy

        dataset = _copy.deepcopy(self._dataset)
        dataset.episodes = [self._episode(str(episode_id))]
        return self._habitat.Env(config=self._task_cfg, dataset=dataset)

    def _episode(self, episode_id: str):
        try:
            return self._episodes[str(episode_id)]
        except KeyError as exc:
            raise KeyError("unknown Habitat episode_id=%s" % episode_id) from exc

    @staticmethod
    def _instruction(env) -> str:
        episode = env.current_episode
        instruction = getattr(episode, "instruction", None)
        return str(getattr(instruction, "instruction_text", instruction or ""))

    def _pano(self, env):
        """Capture front/left/behind/right without changing the trajectory."""
        import quaternion

        state = env.sim.get_agent_state()
        base_rotation = state.rotation
        views = []
        for delta in (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0):
            rotated = copy.deepcopy(state)
            rotated.rotation = quaternion.from_rotation_vector([0.0, delta, 0.0]) * base_rotation
            # Habitat-Sim 0.1.7 accepts position and rotation separately.
            env.sim.set_agent_state(rotated.position, rotated.rotation)
            raw = env.sim.get_sensor_observations()
            rgb = np.asarray(raw["rgb"])[..., :3].copy()
            depth = np.asarray(raw["depth"], dtype=np.float32).copy()
            if depth.ndim == 3:
                depth = depth[..., 0]
            views.append((rgb, depth))
        restored = copy.deepcopy(state)
        restored.rotation = base_rotation
        env.sim.set_agent_state(restored.position, restored.rotation)
        # Policy order: RGB [front, left, behind, right] internally; the
        # public extra_view_images order is [left, right, behind].
        front, left, behind, right = views
        rgb_extra = np.stack([left[0], right[0], behind[0]], axis=0)
        depth_all = np.stack([front[1], left[1], behind[1], right[1]], axis=0)[..., None]
        return front[0], rgb_extra, depth_all

    def _observation(self, index: int):
        env = self._envs[index]
        scan_observation = self._scan_observations[index]
        if env is None or (
            self._done[index] and self._terminal_observation_emitted[index]
        ):
            front = np.zeros(
                (self._args.height, self._args.width, 3), dtype=np.uint8
            )
            rgb_extra = np.zeros(
                (3, self._args.height, self._args.width, 3), dtype=np.uint8
            )
            depth_all = np.zeros(
                (4, self._args.height, self._args.width, 1), dtype=np.float32
            )
            scan_images = np.zeros(
                (12, self._args.height, self._args.width, 3), dtype=np.uint8
            )
            scan_depth_images = np.zeros(
                (12, self._args.height, self._args.width, 1), dtype=np.float32
            )
            scan_states = np.zeros((12, 3), dtype=np.float32)
            scan_valid = np.asarray(False)
            self._scan_observations[index] = None
        elif scan_observation is not None:
            front, rgb_extra, depth_all = scan_observation["selected"]
            scan_images = scan_observation["images"]
            scan_depth_images = scan_observation["depths"]
            scan_states = scan_observation["states"]
            scan_valid = np.asarray(True)
            self._scan_observations[index] = None
        else:
            front, rgb_extra, depth_all = self._pano(env)
            scan_images = np.zeros((12, self._args.height, self._args.width, 3), dtype=np.uint8)
            scan_depth_images = np.zeros(
                (12, self._args.height, self._args.width, 1), dtype=np.float32
            )
            scan_states = np.zeros((12, 3), dtype=np.float32)
            scan_valid = np.asarray(False)
        if self._done[index]:
            self._terminal_observation_emitted[index] = True
        if env is None:
            position = np.zeros(3, dtype=np.float32)
            yaw = 0.0
        else:
            state = env.sim.get_agent_state()
            position = np.asarray(state.position, dtype=np.float32)
            yaw = _quat_yaw(state.rotation)
        obs = {
            "main_images": front,
            "extra_view_images": rgb_extra,
            "wrist_images": depth_all,
            "states": np.asarray([self._elapsed[index], position[0], position[2], yaw], dtype=np.float32),
            "scan_images": scan_images,
            "scan_depth_images": scan_depth_images,
            "scan_states": scan_states,
            "scan_valid": scan_valid,
            "episode_active": np.asarray(not self._done[index], dtype=bool),
            "task_descriptions": (
                "" if env is None or self._done[index] else self._instruction(env)
            ),
        }
        return obs

    def _physical_panorama(self, index: int):
        """Match ZS_Evaluator_mp.get_panorama with 12 real left turns."""
        env = self._envs[index]
        frames = []
        for _ in range(12):
            observation = env.step({"action": "TURN_LEFT"})
            self._elapsed[index] += 1
            raw = dict(env.get_metrics())
            distance = float(
                raw.get("distance_to_goal", raw.get("geodesic_to_goal", np.inf))
            )
            self._min_distance[index] = min(
                float(self._min_distance[index]), distance
            )
            rgb = np.asarray(observation["rgb"])[..., :3].copy()
            depth = np.asarray(observation["depth"], dtype=np.float32).copy()
            if depth.ndim == 3:
                depth = depth[..., 0]
            frames.append((rgb, depth))
            state = env.sim.get_agent_state()
            if len(frames) == 1:
                scan_states = []
            scan_states.append(
                [float(state.position[0]), float(state.position[2]), _quat_yaw(state.rotation)]
            )
            if env.episode_over or self._elapsed[index] >= self._args.max_episode_steps:
                return None

        # Source rotates [30,...,360] to [360,30,...,330], then takes [::3].
        front, left, behind, right = (frames[11], frames[2], frames[5], frames[8])
        rgb_extra = np.stack([left[0], right[0], behind[0]], axis=0)
        depth_all = np.stack(
            [front[1], left[1], behind[1], right[1]], axis=0
        )[..., None]
        return {
            "selected": (front[0], rgb_extra, depth_all),
            "images": np.stack([frame[0] for frame in frames], axis=0),
            "depths": np.stack([frame[1] for frame in frames], axis=0)[..., None],
            "states": np.asarray(scan_states, dtype=np.float32),
        }

    def _batch_observation(self):
        per_env = [self._observation(i) for i in range(len(self._envs))]
        return {
            "main_images": np.stack([x["main_images"] for x in per_env]),
            "extra_view_images": np.stack([x["extra_view_images"] for x in per_env]),
            "wrist_images": np.stack([x["wrist_images"] for x in per_env]),
            "states": np.stack([x["states"] for x in per_env]),
            "scan_images": np.stack([x["scan_images"] for x in per_env]),
            "scan_depth_images": np.stack([x["scan_depth_images"] for x in per_env]),
            "scan_states": np.stack([x["scan_states"] for x in per_env]),
            "scan_valid": np.stack([x["scan_valid"] for x in per_env]),
            "episode_active": np.stack([x["episode_active"] for x in per_env]),
            "task_descriptions": [x["task_descriptions"] for x in per_env],
        }

    def reset(self, episode_ids, trial_ids=None):
        if trial_ids is None:
            trial_ids = ["trial_%d" % i for i in range(len(episode_ids))]
        if len(trial_ids) != len(episode_ids):
            raise ValueError("reset trial_ids must match episode_ids")
        if self._pool_mode:
            if len(episode_ids) != len(self._worker_conns):
                raise ValueError("reset episode_ids must match num_envs")
            results = self._worker_request_batch(
                "reset",
                [
                    {"episode_id": str(episode_id), "trial_id": str(trial_id)}
                    for episode_id, trial_id in zip(episode_ids, trial_ids)
                ],
            )
            self._current_ids = [str(x) for x in episode_ids]
            self._current_trial_ids = [str(x) for x in trial_ids]
            self._elapsed[:] = 0
            self._done[:] = False
            self._terminal_observation_emitted[:] = False
            return {
                "obs": self._merge_observations([x["obs"] for x in results]),
                "infos": {},
            }

        if len(episode_ids) != len(self._envs):
            raise ValueError("reset episode_ids must match num_envs")
        for i, episode_id in enumerate(episode_ids):
            old_env = self._envs[i]
            if old_env is not None:
                old_env.close()
            if not str(episode_id):
                self._envs[i] = None
                self._current_ids[i] = None
                self._current_trial_ids[i] = None
                self._elapsed[i] = 0
                self._done[i] = True
                self._terminal_observation_emitted[i] = True
                self._scan_observations[i] = None
                self._last_terminal[i] = None
                continue
            env = self._make_env(str(episode_id))
            self._envs[i] = env
            env.reset()
            self._current_ids[i] = str(episode_id)
            self._current_trial_ids[i] = str(trial_ids[i])
            self._elapsed[i] = 0
            self._done[i] = False
            self._terminal_observation_emitted[i] = False
            self._scan_observations[i] = None
            self._last_terminal[i] = None
            raw = dict(env.get_metrics())
            start_distance = float(
                raw.get("distance_to_goal", raw.get("geodesic_to_goal", np.nan))
            )
            self._start_distance[i] = start_distance
            self._min_distance[i] = start_distance
        return {"obs": self._batch_observation(), "infos": {}}

    def _metrics(self, env, index: int, cause: Optional[str] = None):
        raw = dict(env.get_metrics())
        final_distance = float(
            raw.get("distance_to_goal", raw.get("geodesic_to_goal", np.inf))
        )
        habitat_success = float(bool(raw.get("success", False)))
        distance_success = float(final_distance <= self._args.success_distance)
        start_distance = float(self._start_distance[index])
        path_length = float(raw.get("path_length", 0.0))
        spl = 0.0
        if distance_success and np.isfinite(start_distance):
            spl = start_distance / max(start_distance, path_length, 1.0e-8)
        ndtw = float(raw.get("ndtw", 0.0))
        result = {
            "episode_id": self._current_ids[index],
            "trial_id": self._current_trial_ids[index],
            "scene_id": str(getattr(env.current_episode, "scene_id", "")),
            # Match ZS_Evaluator_mp: success is final-distance based and does
            # not require an explicit Habitat STOP action.
            "success": distance_success,
            "habitat_success": habitat_success,
            "distance_success": distance_success,
            "oracle_success": float(self._min_distance[index] <= self._args.success_distance),
            "spl": float(spl),
            "ndtw": ndtw,
            "sdtw": float(ndtw * distance_success),
            "distance_to_goal": final_distance,
            "path_length": path_length,
            "steps_taken": float(raw.get("steps_taken", self._elapsed[index])),
            "termination_cause": cause or ("timeout" if env.episode_over else ""),
        }
        if result["success"]:
            result["success_type"] = "clean"
        elif result["termination_cause"] == "stop":
            result["success_type"] = "wrong_stop"
        else:
            result["success_type"] = "no_stop"
        return result

    def _write_metrics(self):
        if not self._metrics_path:
            return
        output = Path(self._metrics_path)
        output_dir = output.parent if output.suffix else output
        output_dir.mkdir(parents=True, exist_ok=True)
        numeric_keys = (
            "success", "habitat_success", "distance_success", "oracle_success",
            "spl", "ndtw", "sdtw", "distance_to_goal",
            "path_length", "steps_taken",
        )
        avg = {}
        for key in numeric_keys:
            values = [row[key] for row in self._records if row.get(key) is not None]
            if values:
                avg[key] = float(np.mean(values))
        grouped = {}
        for row in self._records:
            key = (str(row.get("scene_id", "")), str(row.get("episode_id", "")))
            grouped.setdefault(key, []).append(row)
        per_episode = []
        for (scene_id, episode_id), rows in grouped.items():
            summary = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "num_trials": len(rows),
            }
            for key in numeric_keys:
                values = [row[key] for row in rows if row.get(key) is not None]
                if values:
                    summary[key] = float(np.mean(values))
            per_episode.append(summary)
        payloads = {
            "all_episode_metrics.json": self._records,
            "per_episode_summary.json": per_episode,
            "avg_metrics.json": avg,
        }
        for name, payload in payloads.items():
            destination = output_dir / name
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            with temporary.open("w") as handle:
                json.dump(payload, handle, indent=2)
            temporary.replace(destination)
            # The server runs as the container's unprivileged user while the
            # host-side RLinf evaluator writes its own aggregate files into
            # the same bind mount after evaluation.
            os.chmod(destination, 0o666)

    def _append_record(self, row):
        """Append one terminal row, ignoring repeated force-timeout reports."""
        if not isinstance(row, dict) or row.get("episode_id") is None:
            return
        key = (
            str(row.get("scene_id", "")),
            str(row["episode_id"]),
            str(row.get("trial_id", "")),
        )
        for index, existing in enumerate(self._records):
            existing_key = (
                str(existing.get("scene_id", "")),
                str(existing.get("episode_id", "")),
                str(existing.get("trial_id", "")),
            )
            if existing_key == key:
                self._records[index] = row
                return
        self._records.append(row)

    def step(self, actions):
        if self._pool_mode:
            if len(actions) != len(self._worker_conns):
                raise ValueError("step actions must match num_envs")
            results = self._worker_request_batch(
                "step",
                [{"action": int(action)} for action in actions],
            )
            terminal = []
            for i, result in enumerate(results):
                done = bool(result["terminations"][0] or result["truncations"][0])
                self._done[i] = done
                self._elapsed[i] = int(result["obs"]["states"][0][0])
                episode_info = result["infos"].get("episode", {})
                row = {
                    key: values[0] for key, values in episode_info.items()
                }
                terminal.append(row if done else None)
                if done and row.get("episode_id") is not None:
                    self._append_record(row)
            self._write_metrics()
            keys = (
                "episode_id", "trial_id", "scene_id", "success", "habitat_success",
                "distance_success", "oracle_success", "spl", "ndtw", "sdtw",
                "distance_to_goal", "path_length", "steps_taken",
                "termination_cause", "success_type",
            )
            info = {"episode": {}}
            for key in keys:
                info["episode"][key] = [
                    None if row is None else row.get(key) for row in terminal
                ]
            return {
                "obs": self._merge_observations([x["obs"] for x in results]),
                "rewards": np.zeros(len(results), dtype=np.float32),
                "terminations": self._done.copy(),
                "truncations": np.zeros_like(self._done),
                "infos": info,
            }

        if len(actions) != len(self._envs):
            raise ValueError("step actions must match num_envs")
        done = self._done.copy()
        terminal = [None] * len(self._envs)
        for i, code in enumerate(actions):
            code = int(code)
            if done[i] or code == 6:
                continue
            if code == 7:
                scan_observation = self._physical_panorama(i)
                self._scan_observations[i] = scan_observation
                if scan_observation is None:
                    self._done[i] = True
                    terminal[i] = self._metrics(self._envs[i], i, "timeout")
                    self._last_terminal[i] = terminal[i]
                    self._append_record(terminal[i])
                continue
            action = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}.get(code, "MOVE_FORWARD")
            self._envs[i].step({"action": action})
            self._elapsed[i] += 1
            raw = dict(self._envs[i].get_metrics())
            distance = float(
                raw.get("distance_to_goal", raw.get("geodesic_to_goal", np.inf))
            )
            self._min_distance[i] = min(float(self._min_distance[i]), distance)
            is_stop = code == 0
            is_timeout = self._elapsed[i] >= self._args.max_episode_steps
            if self._args.max_decisions > 0:
                is_timeout = is_timeout or self._elapsed[i] >= self._args.max_decisions
            self._done[i] = bool(self._envs[i].episode_over or is_stop or is_timeout)
            if self._done[i]:
                terminal[i] = self._metrics(self._envs[i], i, "stop" if is_stop else "timeout")
                self._last_terminal[i] = terminal[i]
                self._append_record(terminal[i])
        self._write_metrics()
        done = self._done.copy()
        info = {"episode": {}}
        keys = ("episode_id", "trial_id", "scene_id", "success", "habitat_success", "distance_success", "oracle_success", "spl", "ndtw", "sdtw", "distance_to_goal", "path_length", "steps_taken", "termination_cause", "success_type")
        for key in keys:
            info["episode"][key] = [None if x is None else x[key] for x in terminal]
        return {"obs": self._batch_observation(), "rewards": np.zeros(len(self._envs), dtype=np.float32), "terminations": done, "truncations": np.zeros_like(done), "infos": info}

    def force_timeout(self):
        """Finalize every still-active episode without another simulator step."""
        if self._pool_mode:
            results = self._worker_request_batch("force_timeout")
            terminal = []
            for i, result in enumerate(results):
                done = bool(
                    result["terminations"][0] or result["truncations"][0]
                )
                self._done[i] = done
                self._elapsed[i] = int(result["obs"]["states"][0][0])
                episode_info = result["infos"].get("episode", {})
                row = {key: values[0] for key, values in episode_info.items()}
                terminal.append(row if row.get("episode_id") is not None else None)
                if terminal[-1] is not None:
                    self._append_record(terminal[-1])
            self._write_metrics()
            keys = (
                "episode_id", "trial_id", "scene_id", "success", "habitat_success",
                "distance_success", "oracle_success", "spl", "ndtw", "sdtw",
                "distance_to_goal", "path_length", "steps_taken",
                "termination_cause", "success_type",
            )
            info = {"episode": {}}
            for key in keys:
                info["episode"][key] = [
                    None if row is None else row.get(key) for row in terminal
                ]
            return {
                "obs": self._merge_observations([x["obs"] for x in results]),
                "rewards": np.zeros(len(results), dtype=np.float32),
                "terminations": self._done.copy(),
                "truncations": np.zeros_like(self._done),
                "infos": info,
            }

        terminal = [None] * len(self._envs)
        for i, env in enumerate(self._envs):
            if env is None:
                self._done[i] = True
                continue
            if self._done[i]:
                terminal[i] = self._last_terminal[i]
                # A slot may have been marked done by the simulator before
                # the terminal row was materialised (for example after a
                # physical panorama).  Never leave that slot unreportable.
                if terminal[i] is None:
                    terminal[i] = self._metrics(env, i, "timeout")
                    self._last_terminal[i] = terminal[i]
                    self._append_record(terminal[i])
                continue
            terminal[i] = self._metrics(env, i, "timeout")
            self._last_terminal[i] = terminal[i]
            self._done[i] = True
            self._append_record(terminal[i])
        self._write_metrics()
        keys = (
            "episode_id", "trial_id", "scene_id", "success", "habitat_success",
            "distance_success", "oracle_success", "spl", "ndtw", "sdtw",
            "distance_to_goal", "path_length", "steps_taken",
            "termination_cause", "success_type",
        )
        info = {"episode": {}}
        for key in keys:
            info["episode"][key] = [
                None if row is None else row[key] for row in terminal
            ]
        return {
            "obs": self._batch_observation(),
            "rewards": np.zeros(len(self._envs), dtype=np.float32),
            "terminations": self._done.copy(),
            "truncations": np.zeros_like(self._done),
            "infos": info,
        }

    def close(self):
        if self._pool_mode:
            for index in range(len(self._worker_conns)):
                try:
                    self._worker_request(index, "close")
                except (EOFError, BrokenPipeError, ConnectionError):
                    pass
            for process in self._worker_procs:
                process.join(timeout=10)
            return
        for env in self._envs:
            if env is not None:
                env.close()


def _bridge_worker_main(conn, args):
    """Own exactly one Habitat simulator/OpenGL context."""
    try:
        bridge = HabitatQwenBridge(args)
        conn.send(("ready", None))
        while True:
            method, payload = conn.recv()
            if method == "reset":
                result = bridge.reset(
                    [payload["episode_id"]], [payload.get("trial_id", "trial_0")]
                )
            elif method == "step":
                result = bridge.step([payload["action"]])
            elif method == "force_timeout":
                result = bridge.force_timeout()
            elif method == "close":
                bridge.close()
                conn.send(("ok", {"closed": True}))
                return
            else:
                raise ValueError("unknown worker method %r" % method)
            conn.send(("ok", result))
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass


def serve(args):
    bridge = HabitatQwenBridge(args)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print("[habitat-qwen-server] ready host=%s port=%d num_envs=%d" % (args.host, args.port, args.num_envs), flush=True)
    while True:
        conn, _ = server.accept()
        with conn:
            while True:
                try:
                    request = recv_message(conn)
                    request_id = request.get("request_id")
                    method = request.get("method")
                    if method == "health":
                        result = {"ready": True}
                    elif method == "reset":
                        result = bridge.reset(
                            request["episode_ids"], request.get("trial_ids")
                        )
                    elif method == "step":
                        result = bridge.step(request["actions"])
                    elif method == "force_timeout":
                        result = bridge.force_timeout()
                    elif method == "metrics":
                        # The host adapter uses this as the authoritative
                        # terminal-record snapshot at the eval lifecycle
                        # boundary.  Returning the bridge-owned records avoids
                        # losing slots when an intermediate EnvWorker info
                        # batch is sparse.
                        trial_ids = request.get("trial_ids")
                        records = list(bridge._records)
                        if trial_ids is not None:
                            wanted = {str(value) for value in trial_ids}
                            records = [
                                row for row in records
                                if str(row.get("trial_id", "")) in wanted
                            ]
                        result = {"records": records}
                    elif method == "close":
                        result = {"closed": True}
                        send_message(conn, {"request_id": request_id, "status": "ok", "result": result, "error": None})
                        bridge.close()
                        return
                    else:
                        raise ValueError("unknown method %r" % method)
                    send_message(conn, {"request_id": request_id, "status": "ok", "result": result, "error": None})
                except (ConnectionError, BrokenPipeError):
                    # RLinf closes the client socket after evaluate() returns;
                    # this is normal lifecycle cleanup, not a bridge failure.
                    break
                except Exception as exc:
                    send_message(conn, {"request_id": request.get("request_id") if "request" in locals() else None, "status": "error", "result": None, "error": "%s: %s" % (type(exc).__name__, exc), "traceback": traceback.format_exc()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lavira-root", default="/root/lavira-code")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--gt-path", default="")
    parser.add_argument("--scenes-dir", default="/root/lavira-code/data/scene_datasets")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18770)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--gpu-ids", default="",
        help="Comma-separated logical GPU ids, assigned round-robin to env workers.",
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=79.0)
    parser.add_argument("--camera-height", type=float, default=0.88)
    parser.add_argument("--depth-max", type=float, default=5.0)
    parser.add_argument("--forward-step", type=float, default=0.25)
    parser.add_argument("--turn-angle", type=float, default=30.0)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--max-decisions", type=int, default=0)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument("--metrics-path", default="")
    serve(parser.parse_args())


if __name__ == "__main__":
    main()

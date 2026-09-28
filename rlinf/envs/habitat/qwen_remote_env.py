"""Host-side Habitat 0.1.7 client with the GenArk Qwen observation contract."""

from __future__ import annotations

import json
import fcntl
import pickle
import socket
import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


_HEADER = struct.Struct("!I")


class _AtomicSceneQueue:
    """Small file-backed queue shared by the host EnvWorker processes.

    RLinf communication ranks are fixed for the lifetime of a run, while
    GenArk replaces a finished scene worker immediately.  Claiming scene
    chunks from one locked state file gives Habitat the same dynamic refill
    behavior without changing CommMapper's fixed batch sizes.
    """

    def __init__(self, manifest_path: Path, state_path: Path):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 2:
            raise ValueError("Habitat scene queue requires schema_version=2")
        self.chunks = list(payload.get("scene_jobs", payload.get("scene_chunks", [])))
        self.episode_ids = [
            str(episode_id)
            for chunk in self.chunks
            for episode_id in chunk.get("episode_ids", [])
        ]
        self.slot_count = int(payload["slots_per_worker"])
        self.state_path = state_path

    def claim(self, worker_rank: int) -> tuple[int, dict] | None:
        with self.state_path.open("r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                state = json.load(handle)
                next_index = int(state.get("next_scene", state.get("next_chunk", 0)))
                if next_index >= len(self.chunks):
                    return None
                state["next_scene"] = next_index + 1
                state.setdefault("claims", []).append({
                    "chunk_index": next_index,
                    "worker_rank": int(worker_rank),
                })
                handle.seek(0)
                json.dump(state, handle, indent=2)
                handle.write("\n")
                handle.truncate()
                return next_index, dict(self.chunks[next_index])
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class HabitatQwenRemoteEnv:
    """Synchronous client used only by frozen QwenNavPolicy evaluation.

    Habitat itself stays in the lavira:v4 container.  Keeping this class
    compatible with EnvWorker's ``chunk_step`` contract lets the rollout and
    policy code remain unchanged.
    """

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info=None):
        del worker_info
        self.cfg = cfg
        # Required by RLinf's RecordVideo wrapper for seed_*/<episode>.mp4
        # output naming. Keep the same per-worker convention as other envs.
        self.seed = int(seed_offset)
        self.num_envs = int(num_envs)
        self.group_size = int(getattr(cfg, "group_size", 1))
        self.auto_reset = bool(getattr(cfg, "auto_reset", False))
        self.max_episode_steps = int(getattr(cfg, "max_episode_steps", 300))
        self._worker_rank = int(seed_offset)
        shared_cfg = getattr(cfg, "shared_rollout", None)
        self._shared_rollout = bool(
            getattr(shared_cfg, "enabled", False) if shared_cfg is not None else False
        )
        queue_cfg = getattr(cfg, "dynamic_scene_queue", None)
        self._dynamic_scene_queue = bool(
            getattr(queue_cfg, "enabled", False) if queue_cfg is not None else False
        )
        self._episode_pool: list[str] = []
        self._episode_cursor = 0
        self._all_assignments: list[tuple[str, str]] = []
        if self._shared_rollout and not self._dynamic_scene_queue:
            shards_path = Path(str(getattr(shared_cfg, "episode_shards_path", "")))
            if not shards_path.is_file():
                raise FileNotFoundError(
                    f"Habitat shared-rollout episode shards not found: {shards_path}"
                )
            payload = json.loads(shards_path.read_text(encoding="utf-8"))
            shards = payload.get("shards", payload) if isinstance(payload, dict) else payload
            if len(shards) != int(total_num_processes):
                raise ValueError(
                    "Habitat shared-rollout shard count must equal EnvWorker world "
                    f"size: shards={len(shards)} workers={total_num_processes}"
                )
            self._episode_pool = [str(value) for value in shards[self._worker_rank]]
            if not self._episode_pool or len(self._episode_pool) % self.num_envs != 0:
                raise ValueError(
                    "Each Habitat shared-rollout shard must be non-empty and divisible "
                    f"by local num_envs={self.num_envs}; rank={self._worker_rank} "
                    f"episodes={len(self._episode_pool)}"
                )
        self._scene_queue = None
        self._queue_exhausted = False
        self._current_chunk_index: int | None = None
        self._current_scene_episodes: list[str] = []
        self._current_scene_offset = 0
        self._current_scene_batch = 0
        if getattr(self, "_dynamic_scene_queue", False):
            if not self._shared_rollout:
                raise ValueError("dynamic_scene_queue requires shared_rollout.enabled=true")
            manifest_path = Path(str(getattr(queue_cfg, "manifest_path", "")))
            state_path = Path(str(getattr(queue_cfg, "state_path", "")))
            if not manifest_path.is_file() or not state_path.is_file():
                raise FileNotFoundError(
                    "Habitat dynamic scene queue manifest/state is missing: "
                    f"manifest={manifest_path} state={state_path}"
                )
            self._scene_queue = _AtomicSceneQueue(manifest_path, state_path)
            if self._scene_queue.slot_count != self.num_envs:
                raise ValueError(
                    "scene queue slot count must match local num_envs: "
                    f"queue={self._scene_queue.slot_count} env={self.num_envs}"
                )
            self._episode_pool = list(self._scene_queue.episode_ids)
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self._is_start = True
        self._request_id = 0
        server_port = int(getattr(cfg, "server_port", 18770))
        if self._shared_rollout:
            server_port = int(getattr(shared_cfg, "server_port_base", server_port)) + self._worker_rank
        self._sock = socket.create_connection(
            (str(getattr(cfg, "server_host", "127.0.0.1")), server_port),
            timeout=float(getattr(cfg, "server_timeout_s", 120.0)),
        )
        self._sock.settimeout(float(getattr(cfg, "server_timeout_s", 120.0)))
        self._episode_ids = self._resolve_episode_ids(cfg, self.num_envs)
        self._reset_count = 0
        self._trial_ids: list[str] = []
        self._episode_records: dict[tuple[str, str], dict[str, Any]] = {}
        self._rpc("health", {})

    @staticmethod
    def _resolve_episode_ids(cfg, num_envs=None):
        explicit = getattr(cfg, "episode_ids", None)
        if explicit:
            ids = [str(x) for x in explicit]
        else:
            ids = [str(getattr(cfg, "episode_id", 259))]
        num_envs = int(num_envs if num_envs is not None else getattr(cfg, "num_envs", len(ids)))
        if len(ids) == 1:
            return ids * num_envs
        if len(ids) != num_envs:
            raise ValueError(
                "episode_ids must contain exactly num_envs IDs when more than "
                "one ID is provided"
            )
        return ids

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = bool(value)

    def update_reset_state_ids(self):
        """Compatibility hook used by EnvWorker after non-auto-reset eval.

        Habitat owns terminal episode state in the remote server, so there is
        no local reset-state queue to update.
        """
        return None

    @property
    def info_logging_keys(self):
        return [
            "success", "spl", "ndtw", "sdtw", "distance_to_goal",
            "path_length", "steps_taken",
        ]

    def _send_all(self, raw: bytes) -> None:
        self._sock.sendall(raw)

    def _recv_exact(self, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Habitat bridge closed the socket")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _rpc(self, method: str, payload: dict):
        self._request_id += 1
        request = {"request_id": self._request_id, "method": method, **payload}
        raw = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        self._send_all(_HEADER.pack(len(raw)) + raw)
        size = _HEADER.unpack(self._recv_exact(_HEADER.size))[0]
        response = pickle.loads(self._recv_exact(size))
        if response.get("status") != "ok":
            raise RuntimeError(
                "Habitat bridge %s failed: %s\n%s"
                % (method, response.get("error"), response.get("traceback", ""))
            )
        if response.get("request_id") != self._request_id:
            raise RuntimeError("Habitat bridge request_id mismatch")
        return response["result"]

    @staticmethod
    def _to_tensor_obs(obs: dict) -> dict:
        task_descriptions = list(obs["task_descriptions"])
        if any(not isinstance(value, str) for value in task_descriptions):
            raise TypeError(
                "Habitat task_descriptions must be a flat list[str], got %r"
                % task_descriptions
            )
        return {
            "main_images": torch.from_numpy(np.asarray(obs["main_images"], dtype=np.uint8)),
            "extra_view_images": torch.from_numpy(np.asarray(obs["extra_view_images"], dtype=np.uint8)),
            "wrist_images": torch.from_numpy(np.asarray(obs["wrist_images"], dtype=np.float32)),
            "states": torch.from_numpy(np.asarray(obs["states"], dtype=np.float32)),
            "scan_images": torch.from_numpy(np.asarray(obs["scan_images"], dtype=np.uint8)),
            "scan_depth_images": torch.from_numpy(
                np.asarray(obs["scan_depth_images"], dtype=np.float32)
            ),
            "scan_states": torch.from_numpy(np.asarray(obs["scan_states"], dtype=np.float32)),
            "scan_valid": torch.from_numpy(np.asarray(obs["scan_valid"], dtype=bool)),
            "episode_active": torch.from_numpy(
                np.asarray(obs["episode_active"], dtype=bool)
            ),
            "task_descriptions": task_descriptions,
        }

    def _with_episode_context(self, obs: dict) -> dict:
        """Attach stable identifiers used by policy-side diagnostics."""
        prepared = self._to_tensor_obs(obs)
        prepared["episode_ids"] = list(self._episode_ids)
        prepared["trial_ids"] = list(self._trial_ids)
        return prepared

    @staticmethod
    def _decode_action(value: int) -> int:
        value = int(value)
        if value in (6, 7):
            return value
        if 10 <= value < 14:
            return value - 10
        if value in (0, 1, 2, 3):
            return value
        # Match GenArk parse/schema fallback: continue with FORWARD.
        return 1

    def reset(self, env_idx=None):
        if env_idx is not None:
            raise NotImplementedError("partial reset is not used by frozen evaluation")
        if getattr(self, "_dynamic_scene_queue", False):
            return self._reset_next_scene_chunk()
        if getattr(self, "_shared_rollout", False):
            end = self._episode_cursor + self.num_envs
            if end > len(self._episode_pool):
                raise RuntimeError(
                    "Habitat shared-rollout episode pool exhausted before eval epochs "
                    f"finished: rank={self._worker_rank} cursor={self._episode_cursor} "
                    f"pool={len(self._episode_pool)}"
                )
            self._episode_ids = self._episode_pool[self._episode_cursor:end]
            self._episode_cursor = end
        self._trial_ids = [
            "worker_%03d_reset_%06d_trial_%d"
            % (getattr(self, "_worker_rank", 0), self._reset_count, index)
            for index in range(self.num_envs)
        ]
        self._reset_count += 1
        result = self._rpc(
            "reset",
            {"episode_ids": self._episode_ids, "trial_ids": self._trial_ids},
        )
        if not getattr(self, "_shared_rollout", False):
            self._episode_records.clear()
        assignments = getattr(self, "_all_assignments", None)
        if assignments is None:
            self._all_assignments = assignments = []
        assignments.extend(zip(self._episode_ids, self._trial_ids))
        self._elapsed_steps[:] = 0
        self._done[:] = False
        self._is_start = False
        return self._with_episode_context(result["obs"]), {}

    def _reset_next_scene_chunk(self):
        if self._current_scene_offset >= len(self._current_scene_episodes):
            claim = self._scene_queue.claim(self._worker_rank)
            if claim is None:
                self._queue_exhausted = True
                self._current_chunk_index = None
                episode_ids = [""] * self.num_envs
            else:
                scene_index, scene_job = claim
                self._current_scene_episodes = [
                    str(value) for value in scene_job.get("episode_ids", [])
                ]
                if not self._current_scene_episodes:
                    raise ValueError(f"invalid empty scene job {scene_index}")
                self._current_scene_offset = 0
                self._current_scene_batch = 0
                self._current_chunk_index = int(scene_index)
        if not self._queue_exhausted:
            start = self._current_scene_offset
            end = min(start + self.num_envs, len(self._current_scene_episodes))
            episode_ids = self._current_scene_episodes[start:end]
            self._current_scene_offset = end
            episode_ids.extend([""] * (self.num_envs - len(episode_ids)))
            self._queue_exhausted = False
        self._episode_ids = episode_ids
        self._trial_ids = [
            (
                ""
                if not episode_id
                else "worker_%03d_scene_%06d_batch_%04d_slot_%d"
                % (
                    self._worker_rank,
                    self._current_chunk_index,
                    self._current_scene_batch,
                    index,
                )
            )
            for index, episode_id in enumerate(episode_ids)
        ]
        result = self._rpc(
            "reset",
            {"episode_ids": self._episode_ids, "trial_ids": self._trial_ids},
        )
        for episode_id, trial_id in zip(self._episode_ids, self._trial_ids):
            if episode_id:
                self._all_assignments.append((episode_id, trial_id))
        self._elapsed_steps[:] = 0
        self._done[:] = np.asarray(
            [not bool(episode_id) for episode_id in self._episode_ids], dtype=bool
        )
        self._is_start = False
        self._current_scene_batch += 1
        return self._with_episode_context(result["obs"]), {}

    def step(self, actions=None):
        raw = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        decoded = [self._decode_action(x) for x in raw.reshape(-1)]
        result = self._rpc("step", {"actions": decoded})
        terminations = torch.from_numpy(np.asarray(result["terminations"], dtype=bool))
        truncations = torch.from_numpy(np.asarray(result["truncations"], dtype=bool))
        self._done = terminations.numpy() | truncations.numpy()
        self._elapsed_steps = np.asarray(result["obs"]["states"], dtype=np.float32)[:, 0].astype(np.int32)
        self._remember_episode_infos(result.get("infos", {}))
        infos = self._normalize_infos(result.get("infos", {}))
        if (
            getattr(self, "_dynamic_scene_queue", False)
            and not self._queue_exhausted
            and bool(self._done.all())
        ):
            next_obs, _ = self._reset_next_scene_chunk()
            # The authoritative bridge metrics already captured the completed
            # chunk.  Return the replacement observations as a continuing
            # fixed-slot stream; this avoids marking newly assigned slots done.
            return (
                next_obs,
                torch.zeros(self.num_envs, dtype=torch.float32),
                torch.zeros(self.num_envs, dtype=torch.bool),
                torch.zeros(self.num_envs, dtype=torch.bool),
                infos,
            )
        return (
            self._with_episode_context(result["obs"]),
            torch.from_numpy(np.asarray(result["rewards"], dtype=np.float32)),
            terminations,
            truncations,
            infos,
        )

    @staticmethod
    def _normalize_infos(infos: dict) -> dict:
        episode = infos.get("episode")
        if episode is None:
            return infos
        normalized = {}
        for key, values in episode.items():
            # EnvWorker's metric reducer expects tensors.  String metadata is
            # persisted by the bridge's JSON writer and intentionally does not
            # enter the distributed numeric metric channel.
            if key in (
                "episode_id", "trial_id", "scene_id", "termination_cause",
                "success_type",
            ):
                continue
            normalized[key] = torch.from_numpy(
                np.asarray(
                    [np.nan if value is None else value for value in values],
                    dtype=np.float32,
                )
            )
        return {"episode": normalized}

    def _remember_episode_infos(self, infos: dict) -> None:
        """Keep the bridge's raw terminal rows for exact eval aggregation."""
        episode = infos.get("episode") if isinstance(infos, dict) else None
        if not isinstance(episode, dict):
            return
        episode_ids = episode.get("episode_id", [])
        trial_ids = episode.get("trial_id", [])
        for index, episode_id in enumerate(episode_ids):
            if episode_id is None:
                continue
            row = {}
            for key, values in episode.items():
                if index < len(values):
                    row[key] = values[index]
            trial_id = (
                trial_ids[index]
                if index < len(trial_ids) and trial_ids[index] is not None
                else self._trial_ids[index]
            )
            self._episode_records[(str(episode_id), str(trial_id))] = row

    def get_episode_metrics(self) -> list[dict[str, Any]]:
        """Return one raw metric row per evaluated (episode, trial) pair."""
        # Ask the bridge for its authoritative snapshot.  EnvWorker receives
        # sparse per-step infos by design, while the bridge owns the complete
        # terminal record set for all simulator slots.
        shared_rollout = getattr(self, "_shared_rollout", False)
        metric_request = {} if shared_rollout else {"trial_ids": list(self._trial_ids)}
        snapshot = self._rpc("metrics", metric_request).get("records", [])
        for row in snapshot:
            if isinstance(row, dict) and row.get("episode_id") is not None:
                key = (str(row["episode_id"]), str(row.get("trial_id", "")))
                self._episode_records[key] = dict(row)
        ordered = []
        assignments = (
            self._all_assignments
            if shared_rollout
            else list(zip(self._episode_ids, self._trial_ids))
        )
        for episode_id, trial_id in assignments:
            row = self._episode_records.get((str(episode_id), str(trial_id)))
            if row is not None:
                ordered.append(dict(row))
        known = {
            (str(row.get("episode_id")), str(row.get("trial_id", "")))
            for row in ordered
        }
        ordered.extend(
            dict(row)
            for key, row in self._episode_records.items()
            if key not in known
        )
        return ordered

    @property
    def expected_episode_metrics_count(self) -> int:
        """Number of terminal rows this adapter must return at eval completion."""
        if getattr(self, "_dynamic_scene_queue", False):
            return len(self._all_assignments)
        if getattr(self, "_shared_rollout", False):
            return len(self._episode_pool)
        return self.num_envs

    def chunk_step(self, chunk_actions):
        actions = chunk_actions.detach().cpu().numpy() if isinstance(chunk_actions, torch.Tensor) else np.asarray(chunk_actions)
        if actions.ndim == 1:
            actions = actions[:, None]
        obs_list, reward_list, term_list, trunc_list, info_list = [], [], [], [], []
        for step_idx in range(actions.shape[1]):
            obs, reward, term, trunc, info = self.step(actions[:, step_idx])
            obs_list.append(obs)
            reward_list.append(reward)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)
        return (
            obs_list,
            torch.stack(reward_list, dim=1),
            torch.stack(term_list, dim=1),
            torch.stack(trunc_list, dim=1),
            info_list,
        )

    def force_eval_timeout(self):
        """Finalize active Habitat episodes at the RLinf eval budget.

        The EnvWorker uses this hook when its evaluation horizon ends.  Without
        it, a policy that emits only NOOPs can leave Habitat episodes nonterminal
        because NOOP intentionally does not advance primitive steps.
        """
        result = self._rpc("force_timeout", {})
        terminations = np.asarray(result["terminations"], dtype=bool)
        truncations = np.asarray(result["truncations"], dtype=bool)
        self._done = terminations | truncations
        self._elapsed_steps = np.asarray(
            result["obs"]["states"], dtype=np.float32
        )[:, 0].astype(np.int32)
        self._remember_episode_infos(result.get("infos", {}))
        return (
            self._with_episode_context(result["obs"]),
            self._normalize_infos(result.get("infos", {})),
            self._done.copy(),
        )

    def close(self):
        try:
            # Always perform an idempotent terminal flush.  A wrapper or an
            # asynchronous rollout boundary can mark a slot done before its
            # terminal info has reached the host-side record cache.  The
            # bridge keeps the last terminal row and de-duplicates it, so
            # doing this unconditionally closes that lifecycle gap without
            # changing simulator semantics.
            self.force_eval_timeout()
            self._rpc("close", {})
        finally:
            self._sock.close()

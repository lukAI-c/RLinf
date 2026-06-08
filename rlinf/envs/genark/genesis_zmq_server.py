"""
GenesisZMQServer — standalone ZMQ IPC server for Genesis simulation.

Architecture
------------
::

    RLinf EnvWorker process
        GenarkVecEnv
            ↓ Python call
        GenesisZMQBackend          ← implements GenesisSimBackend
            ↕ ZMQ PUSH/PULL  (ipc:///tmp/genesis_<base>_{req,rep}.ipc)
    GenesisZMQServer process       ← standalone subprocess, CPU-only
        GenesisServerPool
            ↓ Ray remote (num_gpus=1 per actor)
        GenesisSceneActor × N      ← each in its own GPU process

Benefits over direct Ray usage
-------------------------------
- Server process is independent of EnvWorker: if Ray crashes the EnvWorker
  is unaffected, the ZMQ socket just times out.
- One well-defined IPC boundary: easy to debug, log, and restart.
- Async render (Phase 3) is natural: PUSH request, do other work, PULL later.

Startup
-------
``GenesisZMQBackend.__init__`` auto-spawns the server subprocess via
``launch_zmq_server()``.  A sentinel file signals readiness so the backend
does not need to busy-poll ZMQ sockets during Genesis scene loading (~30s).

Protocol
--------
All messages are pickle-serialised dicts.

Request::

    {"req_id": int, "method": str, **kwargs}

Response::

    {"req_id": int, "status": "ok"|"crash"|"error", "result": any, "error": str|None}

Usage
-----
Launched automatically by ``GenarkVecEnv`` when ``cfg.genesis_backend = "zmq"``.
Can also be run directly::

    python -m rlinf.envs.genark.genesis_zmq_server \\
        --cfg-json '{"cam_res":[640,480],...}' \\
        --scene-id  mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb \\
        --num-envs  12 \\
        --socket-base genesis_worker0
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.genark.genesis_backend import GenesisSimBackend
from rlinf.envs.genark.genesis_server import (
    GenesisServerPool,
    SceneCrashError,
)

try:
    import zmq
except ImportError as exc:
    raise ImportError(
        "GenesisZMQServer requires pyzmq. Install with: pip install pyzmq"
    ) from exc

try:
    import ray
    import ray.exceptions
except ImportError as exc:
    raise ImportError(
        "GenesisZMQServer requires Ray. Install with: pip install ray"
    ) from exc


# ---------------------------------------------------------------------------
# Socket path helpers
# ---------------------------------------------------------------------------

_IPC_DIR = "/tmp"


def _req_addr(socket_base: str) -> str:
    return f"ipc://{_IPC_DIR}/{socket_base}_req.ipc"


def _rep_addr(socket_base: str) -> str:
    return f"ipc://{_IPC_DIR}/{socket_base}_rep.ipc"


def _ready_path(socket_base: str) -> Path:
    return Path(_IPC_DIR) / f"{socket_base}.ready"


# ---------------------------------------------------------------------------
# Standalone server process
# ---------------------------------------------------------------------------

class GenesisZMQServer:
    """ZMQ server that wraps GenesisServerPool and exposes it over IPC sockets.

    Runs as a standalone subprocess (one per EnvWorker / scene).  The server
    process itself is CPU-only; Genesis runs inside Ray actors (one GPU each).
    """

    def __init__(
        self,
        cfg: DictConfig,
        scene_id: str,
        num_envs: int,
        socket_base: str,
    ):
        self._scene_id    = scene_id
        self._num_envs    = num_envs
        self._socket_base = socket_base
        self._pool        = GenesisServerPool(cfg, {scene_id: num_envs})

    # ------------------------------------------------------------------

    def run(self) -> None:
        """Enter the request-dispatch loop (blocks until SIGINT/SIGTERM)."""
        ctx = zmq.Context()

        req_sock = ctx.socket(zmq.PULL)
        req_sock.bind(_req_addr(self._socket_base))

        rep_sock = ctx.socket(zmq.PUSH)
        rep_sock.bind(_rep_addr(self._socket_base))

        # Signal readiness: backend polls this file during startup.
        _ready_path(self._socket_base).write_text("ready")
        print(
            f"[GenesisZMQServer] ready — scene='{self._scene_id}' "
            f"socket_base='{self._socket_base}'",
            flush=True,
        )

        while True:
            try:
                raw = req_sock.recv()
                msg = pickle.loads(raw)
                resp = self._dispatch(msg)
                rep_sock.send(pickle.dumps(resp))
            except KeyboardInterrupt:
                break
            except Exception as exc:
                try:
                    rep_sock.send(pickle.dumps({
                        "req_id": msg.get("req_id", -1) if "msg" in dir() else -1,
                        "status": "error",
                        "result": None,
                        "error": str(exc),
                    }))
                except Exception:
                    pass

        # Cleanup
        _ready_path(self._socket_base).unlink(missing_ok=True)
        req_sock.close(linger=0)
        rep_sock.close(linger=0)
        ctx.term()

    # ------------------------------------------------------------------

    def _dispatch(self, msg: dict) -> dict:
        req_id = msg["req_id"]
        method = msg["method"]

        try:
            if method == "health_check":
                return {"req_id": req_id, "status": "ok", "result": True, "error": None}

            actor = self._pool.get_actor(self._scene_id)

            if method == "get_state":
                result = ray.get(actor.get_state.remote())

            elif method == "set_agent_poses":
                result = ray.get(actor.set_agent_poses.remote(
                    msg["env_idx"],
                    msg["positions_np"],
                    msg["yaws_np"],
                ))

            elif method == "step_physics":
                result = ray.get(actor.step_physics.remote(
                    msg["actions_np"],
                    msg["active_mask_np"],
                    msg["active_slot_count"],
                ))

            elif method == "render_main":
                result = ray.get(actor.render_main.remote(msg["active_slot_count"]))

            elif method == "render_4dir":
                result = ray.get(actor.render_4dir.remote(msg["active_slot_count"]))

            else:
                return {
                    "req_id": req_id, "status": "error",
                    "result": None, "error": f"Unknown method: {method}",
                }

            return {"req_id": req_id, "status": "ok", "result": result, "error": None}

        except (
            ray.exceptions.RayActorError,
            ray.exceptions.RayWorkerError,
            ray.exceptions.RayTaskError,
        ) as exc:
            self._pool.rebuild_actor(self._scene_id)
            return {
                "req_id": req_id, "status": "crash",
                "result": None, "error": str(exc),
            }
        except Exception as exc:
            return {
                "req_id": req_id, "status": "error",
                "result": None, "error": str(exc),
            }


# ---------------------------------------------------------------------------
# ZMQ client backend — used by GenarkVecEnv
# ---------------------------------------------------------------------------

class GenesisZMQBackend(GenesisSimBackend):
    """Implements GenesisSimBackend by talking to GenesisZMQServer over IPC.

    Drop-in replacement for GenesisRemoteBackend; adds async render support
    (Phase 3) and crash isolation (Phase 4) via SceneCrashError.
    """

    def __init__(
        self,
        socket_base: str,
        scene_id: str,
        num_envs: int,
        device: torch.device,
    ):
        self._socket_base = socket_base
        self._scene_id    = scene_id
        self._num_envs    = num_envs
        self._device      = device
        self._req_counter = 0

        ctx = zmq.Context.instance()
        self._req_sock = ctx.socket(zmq.PUSH)
        self._rep_sock = ctx.socket(zmq.PULL)
        self._req_sock.connect(_req_addr(socket_base))
        self._rep_sock.connect(_rep_addr(socket_base))

        # Sync initial state (cam_h, cam_w, agent poses)
        state = self._call("get_state")["result"]
        self._cam_h = state["cam_h"]
        self._cam_w = state["cam_w"]
        self._sync_state(state)

    # --- Properties ----------------------------------------------------------

    @property
    def cam_pos(self) -> torch.Tensor:
        return self._cam_pos_t

    @property
    def cam_yaw(self) -> torch.Tensor:
        return self._cam_yaw_t

    @property
    def current_tri_idx(self) -> torch.Tensor:
        return self._current_tri_idx_t

    @property
    def device(self) -> torch.device:
        return self._device

    # --- Health --------------------------------------------------------------

    def is_scene_healthy(self) -> bool:
        try:
            resp = self._call("health_check", timeout_ms=3_000)
            return resp["status"] == "ok"
        except Exception:
            return False

    # --- Scene lifecycle -----------------------------------------------------

    def load_scene(self, scene_id: str, n_active_envs: int) -> None:
        # Scene is loaded when the server actor was created — no-op here.
        pass

    # --- Agent pose management -----------------------------------------------

    def set_agent_poses(
        self,
        env_idx: list[int],
        positions: torch.Tensor,
        yaws: torch.Tensor,
    ) -> None:
        resp = self._call(
            "set_agent_poses",
            env_idx=env_idx,
            positions_np=positions.cpu().numpy(),
            yaws_np=yaws.cpu().numpy(),
        )
        self._sync_state(resp["result"])

    # --- Physics step --------------------------------------------------------

    def step_physics(
        self,
        actions: torch.Tensor,
        active_mask: torch.Tensor,
        active_slot_count: int,
    ) -> None:
        resp = self._call(
            "step_physics",
            actions_np=actions.cpu().numpy(),
            active_mask_np=active_mask.cpu().numpy(),
            active_slot_count=active_slot_count,
        )
        self._sync_state(resp["result"])

    # --- Synchronous rendering -----------------------------------------------

    def render_main(self, active_slot_count: int) -> np.ndarray:
        resp = self._call("render_main", active_slot_count=active_slot_count)
        return (
            np.frombuffer(resp["result"], dtype=np.uint8)
            .reshape(self._num_envs, self._cam_h, self._cam_w, 3)
            .copy()
        )

    def render_4dir(self, active_slot_count: int) -> Optional[np.ndarray]:
        resp = self._call("render_4dir", active_slot_count=active_slot_count)
        raw = resp["result"]
        if raw is None:
            return None
        return (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape(self._num_envs, 3, self._cam_h, self._cam_w, 3)
            .copy()
        )

    # --- Async rendering (Phase 3) -------------------------------------------

    def render_main_async(self, active_slot_count: int) -> int:
        """Submit render; return req_id (non-blocking)."""
        return self._submit("render_main", active_slot_count=active_slot_count)

    def render_4dir_async(self, active_slot_count: int) -> int:
        """Submit 4-dir render; return req_id (non-blocking)."""
        return self._submit("render_4dir", active_slot_count=active_slot_count)

    def fetch_render_main(self, req_id: int, timeout: float = 30.0) -> np.ndarray:
        resp = self._fetch(req_id, timeout_ms=int(timeout * 1_000))
        return (
            np.frombuffer(resp["result"], dtype=np.uint8)
            .reshape(self._num_envs, self._cam_h, self._cam_w, 3)
            .copy()
        )

    def fetch_render_4dir(self, req_id: int, timeout: float = 30.0) -> Optional[np.ndarray]:
        if req_id is None:
            return None
        resp = self._fetch(req_id, timeout_ms=int(timeout * 1_000))
        raw = resp["result"]
        if raw is None:
            return None
        return (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape(self._num_envs, 3, self._cam_h, self._cam_w, 3)
            .copy()
        )

    # --- Internal helpers ----------------------------------------------------

    def _next_req_id(self) -> int:
        self._req_counter += 1
        return self._req_counter

    def _send(self, msg: dict) -> None:
        self._req_sock.send(pickle.dumps(msg))

    def _recv(self, timeout_ms: int) -> dict:
        if not self._rep_sock.poll(timeout_ms):
            raise TimeoutError(
                f"[GenesisZMQBackend] no response within {timeout_ms}ms "
                f"(scene='{self._scene_id}')"
            )
        resp = pickle.loads(self._rep_sock.recv())
        if resp["status"] == "crash":
            raise SceneCrashError(self._scene_id, RuntimeError(resp["error"]))
        if resp["status"] == "error":
            raise RuntimeError(
                f"[GenesisZMQServer] error in '{resp.get('method', '?')}': {resp['error']}"
            )
        return resp

    def _call(self, method: str, timeout_ms: int = 60_000, **kwargs) -> dict:
        """Synchronous call: PUSH request then PULL response."""
        req_id = self._next_req_id()
        self._send({"req_id": req_id, "method": method, **kwargs})
        return self._recv(timeout_ms)

    def _submit(self, method: str, **kwargs) -> int:
        """Async submit: PUSH request without waiting. Returns req_id."""
        req_id = self._next_req_id()
        self._send({"req_id": req_id, "method": method, **kwargs})
        return req_id

    def _fetch(self, req_id: int, timeout_ms: int = 60_000) -> dict:
        """Block until the response for req_id arrives."""
        return self._recv(timeout_ms)

    def _sync_state(self, state: dict) -> None:
        self._cam_pos_t = torch.from_numpy(
            state["cam_pos"].copy()
        ).to(self._device)
        self._cam_yaw_t = torch.from_numpy(
            state["cam_yaw"].copy()
        ).to(self._device)
        self._current_tri_idx_t = torch.from_numpy(
            state["current_tri_idx"].copy()
        ).to(self._device)


# ---------------------------------------------------------------------------
# Server launcher — called by GenarkVecEnv when genesis_backend = "zmq"
# ---------------------------------------------------------------------------

def launch_zmq_server(
    cfg: DictConfig,
    scene_id: str,
    num_envs: int,
    socket_base: str,
    startup_timeout: float = 180.0,
) -> subprocess.Popen:
    """Spawn GenesisZMQServer as a subprocess; wait until ready.

    The subprocess inherits the current environment (PYTHONPATH, etc.).
    Genesis scene loading typically takes 30–90s; ``startup_timeout`` should
    be generous.
    """
    ready_file = _ready_path(socket_base)
    ready_file.unlink(missing_ok=True)

    cfg_json = json.dumps(OmegaConf.to_container(cfg, resolve=True))
    cmd = [
        sys.executable, "-m", "rlinf.envs.genark.genesis_zmq_server",
        "--cfg-json",    cfg_json,
        "--scene-id",    scene_id,
        "--num-envs",    str(num_envs),
        "--socket-base", socket_base,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if ready_file.exists():
            print(
                f"[GenesisZMQServer] started (pid={proc.pid}) "
                f"socket_base='{socket_base}'",
                flush=True,
            )
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"GenesisZMQServer subprocess exited early "
                f"(returncode={proc.returncode})"
            )
        time.sleep(1.0)

    proc.kill()
    raise TimeoutError(
        f"GenesisZMQServer did not become ready within {startup_timeout}s "
        f"(socket_base='{socket_base}')"
    )


# ---------------------------------------------------------------------------
# __main__ entry point — used by launch_zmq_server subprocess
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GenesisZMQServer subprocess")
    parser.add_argument("--cfg-json",    required=True)
    parser.add_argument("--scene-id",    required=True)
    parser.add_argument("--num-envs",    type=int, required=True)
    parser.add_argument("--socket-base", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.create(json.loads(args.cfg_json))
    server = GenesisZMQServer(cfg, args.scene_id, args.num_envs, args.socket_base)
    server.run()

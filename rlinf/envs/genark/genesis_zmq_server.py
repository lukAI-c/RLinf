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
from rlinf.envs.genark.genesis_server import SceneCrashError

try:
    import zmq
except ImportError as exc:
    raise ImportError(
        "GenesisZMQServer requires pyzmq. Install with: pip install pyzmq"
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
    """ZMQ server that wraps GenesisLocalBackend and exposes it over IPC sockets.

    Runs as a standalone subprocess (one per EnvWorker / scene).  The subprocess
    boundary itself provides crash isolation — if Genesis segfaults the process
    dies, the ZMQ socket times out on the client, and is_scene_healthy() returns
    False.  No Ray actor layer needed here.
    """

    def __init__(
        self,
        cfg: DictConfig,
        scene_id: str,
        num_envs: int,
        socket_base: str,
    ):
        from rlinf.envs.genark.genesis_backend import GenesisLocalBackend
        self._scene_id    = scene_id
        self._num_envs    = num_envs
        self._socket_base = socket_base
        self._backend     = GenesisLocalBackend(cfg, num_envs)
        self._backend.load_scene(scene_id, n_active_envs=num_envs)

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
        dev    = self._backend.device

        try:
            if method == "health_check":
                return {"req_id": req_id, "status": "ok", "result": True, "error": None}

            if method == "get_state":
                result = {
                    "cam_pos":         self._backend.cam_pos.cpu().numpy(),
                    "cam_yaw":         self._backend.cam_yaw.cpu().numpy(),
                    "current_tri_idx": self._backend.current_tri_idx.cpu().numpy(),
                    "num_envs": self._num_envs,
                    "cam_h":    self._backend._cam_h,
                    "cam_w":    self._backend._cam_w,
                }

            elif method == "set_agent_poses":
                self._backend.set_agent_poses(
                    msg["env_idx"],
                    torch.from_numpy(msg["positions_np"]).to(dev),
                    torch.from_numpy(msg["yaws_np"]).to(dev),
                )
                result = {
                    "cam_pos":         self._backend.cam_pos.cpu().numpy(),
                    "cam_yaw":         self._backend.cam_yaw.cpu().numpy(),
                    "current_tri_idx": self._backend.current_tri_idx.cpu().numpy(),
                }

            elif method == "step_physics":
                self._backend.step_physics(
                    torch.from_numpy(msg["actions_np"]).to(dev),
                    torch.from_numpy(msg["active_mask_np"]).to(dev),
                    msg["active_slot_count"],
                )
                result = {
                    "cam_pos":         self._backend.cam_pos.cpu().numpy(),
                    "cam_yaw":         self._backend.cam_yaw.cpu().numpy(),
                    "current_tri_idx": self._backend.current_tri_idx.cpu().numpy(),
                }

            elif method == "render_main":
                result = self._backend.render_main(msg["active_slot_count"]).tobytes()

            elif method == "render_4dir":
                arr = self._backend.render_4dir(msg["active_slot_count"])
                result = arr.tobytes() if arr is not None else None

            elif method == "render_main_with_depth":
                rgb, depth = self._backend.render_main_with_depth(msg["active_slot_count"])
                result = (rgb.tobytes(), depth.tobytes())

            elif method == "render_4dir_with_depth":
                rgb, depth = self._backend.render_4dir_with_depth(msg["active_slot_count"])
                result = (None, None) if rgb is None or depth is None else (rgb.tobytes(), depth.tobytes())

            elif method == "render_panorama_with_depth":
                rgb, depth, yaw = self._backend.render_panorama_with_depth(
                    msg["active_slot_count"]
                )
                result = (rgb.tobytes(), depth.tobytes(), yaw.tobytes())

            else:
                return {
                    "req_id": req_id, "status": "error",
                    "result": None, "error": f"Unknown method: {method}",
                }

            return {"req_id": req_id, "status": "ok", "result": result, "error": None}

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
        self._socket_base       = socket_base
        self._scene_id          = scene_id
        self._num_envs          = num_envs
        self._device            = device
        self._req_counter       = 0
        self._response_buffer: dict[int, dict] = {}  # req_id → buffered response

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

    def render_main_with_depth(self, active_slot_count: int):
        raw_rgb, raw_depth = self._call("render_main_with_depth", active_slot_count=active_slot_count)["result"]
        return (
            np.frombuffer(raw_rgb, dtype=np.uint8).reshape(self._num_envs, self._cam_h, self._cam_w, 3).copy(),
            np.frombuffer(raw_depth, dtype=np.float32).reshape(self._num_envs, self._cam_h, self._cam_w).copy(),
        )

    def render_4dir_with_depth(self, active_slot_count: int):
        raw_rgb, raw_depth = self._call("render_4dir_with_depth", active_slot_count=active_slot_count)["result"]
        if raw_rgb is None or raw_depth is None:
            return None, None
        return (
            np.frombuffer(raw_rgb, dtype=np.uint8).reshape(self._num_envs, 3, self._cam_h, self._cam_w, 3).copy(),
            np.frombuffer(raw_depth, dtype=np.float32).reshape(self._num_envs, 3, self._cam_h, self._cam_w).copy(),
        )

    def render_panorama_with_depth(self, active_slot_count: int):
        raw_rgb, raw_depth, raw_yaw = self._call(
            "render_panorama_with_depth",
            active_slot_count=active_slot_count,
            timeout_ms=120_000,
        )["result"]
        return (
            np.frombuffer(raw_rgb, dtype=np.uint8).reshape(
                self._num_envs, 12, self._cam_h, self._cam_w, 3
            ).copy(),
            np.frombuffer(raw_depth, dtype=np.float32).reshape(
                self._num_envs, 12, self._cam_h, self._cam_w
            ).copy(),
            np.frombuffer(raw_yaw, dtype=np.float32).reshape(
                self._num_envs, 12
            ).copy(),
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

    def _recv_one(self, timeout_ms: int) -> dict:
        """Read exactly one raw response from the socket (no req_id filtering)."""
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

    def _recv(self, expected_req_id: int, timeout_ms: int) -> dict:
        """Return the response matching expected_req_id, buffering others.

        Needed because render_main_async submits a request without waiting,
        so the render response may arrive in the socket before the next
        step_physics response — we must not mistake one for the other.
        """
        if expected_req_id in self._response_buffer:
            return self._response_buffer.pop(expected_req_id)
        deadline = time.time() + timeout_ms / 1000.0
        while True:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                raise TimeoutError(
                    f"[GenesisZMQBackend] no response for req_id={expected_req_id} "
                    f"within {timeout_ms}ms (scene='{self._scene_id}')"
                )
            resp = self._recv_one(remaining_ms)
            if resp["req_id"] == expected_req_id:
                return resp
            self._response_buffer[resp["req_id"]] = resp

    def _call(self, method: str, timeout_ms: int = 60_000, **kwargs) -> dict:
        """Synchronous call: PUSH request then PULL matching response."""
        req_id = self._next_req_id()
        self._send({"req_id": req_id, "method": method, **kwargs})
        return self._recv(req_id, timeout_ms)

    def _submit(self, method: str, **kwargs) -> int:
        """Async submit: PUSH request without waiting. Returns req_id."""
        req_id = self._next_req_id()
        self._send({"req_id": req_id, "method": method, **kwargs})
        return req_id

    def _fetch(self, req_id: int, timeout_ms: int = 60_000) -> dict:
        """Block until the response for req_id arrives."""
        return self._recv(req_id, timeout_ms)

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

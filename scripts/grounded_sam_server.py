#!/usr/bin/env python3
"""Standalone single-GPU GroundedSAM service for shared Habitat evaluation."""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import socket
import threading
import traceback
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dino-config-path", required=True)
    parser.add_argument("--dino-checkpoint-path", required=True)
    parser.add_argument("--repvit-sam-checkpoint-path", required=True)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.95)
    parser.add_argument("--waypoint-scene-box-area-ratio", type=float, default=0.65)
    parser.add_argument("--waypoint-scene-edge-margin-ratio", type=float, default=0.02)
    parser.add_argument(
        "--reject-scene-region-source-waypoint",
        action="store_true",
    )
    parser.add_argument("--min-mask-area-px", type=int, default=16)
    return parser.parse_args()


def dispatch_request(model, method, payload):
    if method == "health":
        return {"ready": True}, False
    if method == "segment_classes_batch":
        return model.segment_classes_batch(payload), False
    if method == "refine_batch":
        if hasattr(model, "refine_batch"):
            return [
                dataclasses.asdict(result)
                for result in model.refine_batch(payload)
            ], False
        refine_parameters = inspect.signature(model.refine).parameters
        supports_region = "target_region" in refine_parameters
        supports_source_lhx = "source_lhx" in refine_parameters
        results = []
        for job in payload:
            kwargs = {}
            if supports_source_lhx:
                kwargs.update(
                    grounding_classes=job.get("grounding_classes"),
                    source_lhx=bool(job.get("source_lhx", False)),
                )
            if supports_region:
                result = model.refine(
                    job["image_rgb"],
                    job["target"],
                    job.get("target_region", "any"),
                    **kwargs,
                )
            else:
                result = model.refine(job["image_rgb"], job["target"], **kwargs)
            results.append(dataclasses.asdict(result))
        return results, False
    if method == "close":
        return {"closed": True}, True
    raise ValueError(f"unknown GroundedSAM method: {method!r}")


def serve_connection(conn, model, model_lock, recv_rpc_message, send_rpc_message):
    """Serve one client; model execution stays serialized across clients."""
    with conn:
        while True:
            try:
                request = recv_rpc_message(conn)
            except (ConnectionError, OSError):
                return
            try:
                method = request.get("method")
                payload = request.get("payload")
                if method in {"health", "close"}:
                    result, close_connection = dispatch_request(
                        model, method, payload
                    )
                else:
                    with model_lock:
                        result, close_connection = dispatch_request(
                            model, method, payload
                        )
                send_rpc_message(
                    conn, {"status": "ok", "result": result, "error": None}
                )
                if close_connection:
                    return
            except Exception:
                error = traceback.format_exc()
                try:
                    send_rpc_message(
                        conn,
                        {"status": "error", "result": None, "error": error},
                    )
                except (BrokenPipeError, ConnectionError, OSError):
                    return


def main():
    args = parse_args()
    from rlinf.models.embodiment.qwen_nav.grounded_sam import (
        GroundedSAMWaypointRefiner,
        recv_rpc_message,
        send_rpc_message,
    )

    cfg = SimpleNamespace(
        dino_config_path=args.dino_config_path,
        dino_checkpoint_path=args.dino_checkpoint_path,
        repvit_sam_checkpoint_path=args.repvit_sam_checkpoint_path,
        device="cuda:0",
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_box_area_ratio=args.max_box_area_ratio,
        waypoint_scene_box_area_ratio=args.waypoint_scene_box_area_ratio,
        waypoint_scene_edge_margin_ratio=args.waypoint_scene_edge_margin_ratio,
        reject_scene_region_source_waypoint=(
            args.reject_scene_region_source_waypoint
        ),
        min_mask_area_px=args.min_mask_area_px,
        fail_fast_on_missing_assets=True,
    )
    model = GroundedSAMWaypointRefiner(cfg)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(4)
    print(
        f"[grounded-sam-server] ready host={args.host} port={args.port}",
        flush=True,
    )

    model_lock = threading.Lock()
    while True:
        conn, _ = server.accept()
        threading.Thread(
            target=serve_connection,
            args=(conn, model, model_lock, recv_rpc_message, send_rpc_message),
            daemon=True,
        ).start()


if __name__ == "__main__":
    main()

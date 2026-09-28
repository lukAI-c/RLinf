#!/usr/bin/env python3
"""Replay identical RGB-D mapper inputs through LHX source and sparse A/B maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rlinf.models.embodiment.qwen_nav.lavira_runtime.semantic_map import (
    LaviraSemanticMap,
)
from rlinf.third_party.lavira_rft.source_core import LaviraSourceCore
from rlinf.third_party.lavira_rft.loader import SOURCE_COMMIT


def _binary_gap(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left_bool = np.asarray(left).astype(bool)
    right_bool = np.asarray(right).astype(bool)
    xor = np.logical_xor(left_bool, right_bool)
    return {
        "left_cells": int(left_bool.sum()),
        "right_cells": int(right_bool.sum()),
        "xor_cells": int(xor.sum()),
        "xor_rate": float(xor.mean()),
    }


def _load_frame(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        labels = [str(label) for label in payload["semantic_labels"].tolist()]
        raw_masks = payload["semantic_masks"]
        masks = {label: raw_masks[i] for i, label in enumerate(labels)}
        return {
            "depth_m": payload["depth_m"],
            "pose_hab": payload["pose_hab"],
            "previous_action": int(payload["previous_action"]),
            "masks": masks,
            "saved_channels": payload["channels"],
        }


def compare(args) -> dict:
    paths = sorted(Path(args.input).glob("map_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no map_*.npz frames under {args.input}")

    first = _load_frame(paths[0])
    origin_x, origin_z, origin_yaw = map(float, first["pose_hab"])
    source = LaviraSourceCore(
        device=args.device,
        results_dir=args.results_dir,
        hfov_deg=args.hfov,
        camera_height=args.camera_height,
        map_size_cm=args.map_size_cm,
        resolution_cm=args.map_resolution_cm,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )
    sparse = LaviraSemanticMap(
        hfov_deg=args.hfov,
        camera_height=args.camera_height,
        map_size_cm=args.map_size_cm,
        resolution_cm=args.map_resolution_cm,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )
    source.reset(origin_x, origin_z, origin_yaw)
    sparse.reset(origin_x, origin_z, origin_yaw)

    frames = []
    for path in paths:
        frame = _load_frame(path)
        x, z, yaw = map(float, frame["pose_hab"])
        previous_action = frame["previous_action"]
        action = None if previous_action < 0 else previous_action
        source.set_last_action(action)
        sparse.set_last_action(action)
        source.update(frame["depth_m"], x, z, yaw, frame["masks"])
        sparse.update(frame["depth_m"], x, z, yaw, frame["masks"])
        source.rebuild_traversible()
        sparse.rebuild_traversible()

        saved_channels = frame["saved_channels"]
        record = {
            "frame": path.name,
            "obstacle": _binary_gap(source.obstacle, sparse.obstacle),
            "explored": _binary_gap(source.explored, sparse.explored),
            "traversible": _binary_gap(source.traversible(), sparse.traversible()),
            "source_replay_vs_saved_obstacle": _binary_gap(
                source.obstacle, saved_channels[0]
            ),
            "source_replay_vs_saved_explored": _binary_gap(
                source.explored, saved_channels[1]
            ),
        }
        frames.append(record)

    return {
        "input": str(Path(args.input).resolve()),
        "frame_count": len(frames),
        "source_commit": SOURCE_COMMIT,
        "frames": frames,
        "final": frames[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="episode directory with map_*.npz")
    parser.add_argument("--output", default="")
    parser.add_argument("--results-dir", default="/tmp/rlinf_lavira_mapper_ab")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hfov", type=float, default=79.0)
    parser.add_argument("--camera-height", type=float, default=0.88)
    parser.add_argument("--map-size-cm", type=int, default=2400)
    parser.add_argument("--map-resolution-cm", type=int, default=5)
    parser.add_argument("--frame-width", type=int, default=160)
    parser.add_argument("--frame-height", type=int, default=120)
    args = parser.parse_args()

    result = compare(args)
    encoded = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()

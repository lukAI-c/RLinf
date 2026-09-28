#!/usr/bin/env python3
"""Compare two independent LaViRA source cores under one fixed input contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from rlinf.models.embodiment.qwen_nav.lavira_runtime.observation import (
    LaviraObservation,
)
from rlinf.third_party.lavira_rft.source_core import LaviraSourceCore


SOURCE_FILES = (
    "vlnce_baselines/map/mapping.py",
    "vlnce_baselines/models/Policy.py",
    "vlnce_baselines/models/fmm_planner.py",
    "vlnce_baselines/utils/map_utils.py",
    "vlnce_baselines/utils/depth_utils.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        labels = [str(value) for value in payload["semantic_labels"].tolist()]
        masks = {
            label: np.asarray(payload["semantic_masks"][index], dtype=np.float32)
            for index, label in enumerate(labels)
        }
        return {
            "depth": np.asarray(payload["depth_m"], dtype=np.float32),
            "pose": np.asarray(payload["pose_hab"], dtype=np.float32),
            "masks": masks,
            "previous_action": int(payload["previous_action"]),
        }


def _run(label: str, frame: dict, bbox: list[float], results_dir: Path) -> dict:
    x, z, yaw = map(float, frame["pose"])
    core = LaviraSourceCore(device="cpu", results_dir=results_dir / label)
    core.reset(x, z, yaw)
    core.set_last_action(None if frame["previous_action"] < 0 else frame["previous_action"])
    core.update(frame["depth"], x, z, yaw, frame["masks"])
    traversible = core.rebuild_traversible().astype(bool)
    blank = Image.new("RGB", (frame["depth"].shape[1], frame["depth"].shape[0]))
    observation = LaviraObservation(
        rgb_by_direction=[blank] * 4,
        depth_by_direction=np.stack([frame["depth"]] * 4),
        yaw_by_direction=np.asarray(
            [yaw, yaw + math.pi / 2, yaw + math.pi, yaw - math.pi / 2],
            dtype=np.float32,
        ),
        hab_x=x,
        hab_z=z,
        hfov_deg=79.0,
    )
    goal = observation.project_target("navigate to forward", bbox_2d=bbox)
    if goal is None:
        raise RuntimeError("fixed bbox/depth input did not produce a goal")
    action = core.fmm_action(x, z, yaw, float(goal[0]), float(goal[1]))
    fmm_dist = np.asarray(core.policy.fmm_dist, dtype=np.float64)
    return {
        "goal_xz": np.asarray(goal, dtype=np.float64),
        "goal_map_rc": np.asarray(core.last_fmm_audit["requested_goal_map_rc"]),
        "traversible": traversible,
        "fmm_dist": fmm_dist,
        "primitive_action": int(action),
        "fmm_audit": core.last_fmm_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--bbox", default="473.6984,261.4648,576.9235,670.8724")
    parser.add_argument("--lhx-root", default="/home/clk/workspace/lavira-rft")
    parser.add_argument("--results-dir", default="/tmp/fixed_lavira_input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = _load(Path(args.input))
    bbox = [float(value) for value in args.bbox.split(",")]
    if len(bbox) != 4:
        raise ValueError("bbox must have four comma-separated values")
    results_dir = Path(args.results_dir)
    left = _run("genesis_adapter", frame, bbox, results_dir)
    right = _run("habitat_adapter", frame, bbox, results_dir)

    finite = np.isfinite(left["fmm_dist"]) & np.isfinite(right["fmm_dist"])
    fmm_gap = np.abs(left["fmm_dist"][finite] - right["fmm_dist"][finite])
    vendor_root = (
        Path(__file__).resolve().parents[1]
        / "rlinf/third_party/lavira_rft/source"
    )
    lhx_root = Path(args.lhx_root)
    hashes = {}
    for relative in SOURCE_FILES:
        vendor_hash = _sha256(vendor_root / relative)
        lhx_hash = _sha256(lhx_root / relative)
        hashes[relative] = {
            "vendor": vendor_hash,
            "lhx": lhx_hash,
            "identical": vendor_hash == lhx_hash,
        }

    result = {
        "fixed_input": {
            "path": str(Path(args.input).resolve()),
            "pose_hab": frame["pose"].tolist(),
            "bbox_2d": bbox,
            "depth_min_m": float(np.nanmin(frame["depth"])),
            "depth_max_m": float(np.nanmax(frame["depth"])),
            "semantic_labels": list(frame["masks"]),
        },
        "source_file_hashes": hashes,
        "all_source_files_identical": all(row["identical"] for row in hashes.values()),
        "goal_gap_m": float(np.linalg.norm(left["goal_xz"] - right["goal_xz"])),
        "goal_map_equal": bool(np.array_equal(left["goal_map_rc"], right["goal_map_rc"])),
        "traversible_xor_cells": int(np.logical_xor(left["traversible"], right["traversible"]).sum()),
        "fmm_finite_mask_equal": bool(
            np.array_equal(np.isfinite(left["fmm_dist"]), np.isfinite(right["fmm_dist"]))
        ),
        "fmm_max_abs_gap": float(fmm_gap.max()) if fmm_gap.size else 0.0,
        "genesis_primitive_action": left["primitive_action"],
        "habitat_primitive_action": right["primitive_action"],
        "primitive_action_equal": left["primitive_action"] == right["primitive_action"],
        "genesis_fmm_audit": left["fmm_audit"],
        "habitat_fmm_audit": right["fmm_audit"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate RGB calibration with the Qwen vision encoder used by the policy.

This consumes the held-out labels from ``calibrate_genesis_rgb.py``.  Genesis
linear renders are transformed with the baseline and fitted parameters before
being encoded, while Habitat PNGs are passed through unchanged.  The score is
therefore independent from the RGB fitting objective and measures the visual
features actually exposed to the navigation model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1.0 / 2.4) - 0.055)


def _render(raw: np.ndarray, exposure: float, matrix: np.ndarray, bias: np.ndarray) -> Image.Image:
    linear = np.einsum("...c,dc->...d", raw * exposure, matrix) + bias
    srgb = _linear_to_srgb(linear)
    return Image.fromarray(np.round(srgb * 255.0).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--habitat-dir", required=True)
    parser.add_argument("--genesis-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-pixels", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=122500)
    args = parser.parse_args()

    calibration = json.loads(Path(args.calibration).read_text())
    habitat_dir, genesis_dir = Path(args.habitat_dir), Path(args.genesis_dir)
    habitat = json.loads((habitat_dir / "manifest.json").read_text())
    genesis = json.loads((genesis_dir / "manifest.json").read_text())
    habitat_by_label = {row["label"]: row for row in habitat["captures"]}
    genesis_by_label = {row["label"]: row for row in genesis["frames"]}
    labels = calibration["holdout_frames"]

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    def features(image: Image.Image) -> torch.Tensor:
        inputs = processor(
            text=["<|vision_start|><|image_pad|><|vision_end|>"],
            images=[image],
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = model.get_image_features(
                pixel_values=inputs["pixel_values"].to(device),
                image_grid_thw=inputs["image_grid_thw"].to(device),
            )
        return output.pooler_output[0].float()

    candidates = {key: calibration[key] for key in ("baseline", "calibrated")}
    rows: dict[str, list[dict[str, float | str]]] = {key: [] for key in candidates}
    for label in labels:
        h_row, g_row = habitat_by_label[label], genesis_by_label[label]
        habitat_rgb = Image.open(habitat_dir / h_row["rgb"]).convert("RGB")
        target_features = features(habitat_rgb)
        raw = np.asarray(np.load(genesis_dir / g_row["linear"]), dtype=np.float32)
        for name, params in candidates.items():
            candidate = _render(
                raw,
                float(params["exposure"]),
                np.asarray(params["matrix"], dtype=np.float32),
                np.asarray(params["bias"], dtype=np.float32),
            )
            value = features(candidate)
            if value.shape != target_features.shape:
                raise ValueError(
                    f"Qwen feature shape mismatch for {label}: {tuple(value.shape)} vs "
                    f"{tuple(target_features.shape)}"
                )
            cosine = torch.nn.functional.cosine_similarity(value, target_features, dim=-1)
            rows[name].append({
                "label": label,
                "feature_mae": float(torch.mean(torch.abs(value - target_features)).cpu()),
                "one_minus_cosine": float((1.0 - cosine.mean()).cpu()),
            })

    result = {
        "schema_version": 1,
        "model_path": args.model_path,
        "holdout_frames": labels,
        "baseline": {
            "feature_mae": float(np.mean([row["feature_mae"] for row in rows["baseline"]])),
            "one_minus_cosine": float(np.mean([row["one_minus_cosine"] for row in rows["baseline"]])),
            "per_frame": rows["baseline"],
        },
        "calibrated": {
            "feature_mae": float(np.mean([row["feature_mae"] for row in rows["calibrated"]])),
            "one_minus_cosine": float(np.mean([row["one_minus_cosine"] for row in rows["calibrated"]])),
            "per_frame": rows["calibrated"],
        },
    }
    result["calibrated_improves_qwen_features"] = (
        result["calibrated"]["feature_mae"] < result["baseline"]["feature_mae"]
        and result["calibrated"]["one_minus_cosine"] < result["baseline"]["one_minus_cosine"]
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

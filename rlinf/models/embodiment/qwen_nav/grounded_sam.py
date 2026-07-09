"""
Optional GroundingDINO + SAM waypoint refiner for QwenNav.

This is intentionally a thin adapter around LaViRA's GroundedSAM idea:
given the target phrase emitted by the VLM and the RGB view selected by the
VLM action, detect/segment the target and return a normalized waypoint point
plus bbox in the existing 0-1000 QwenNav coordinate convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig


@dataclass
class GroundedSAMRefineResult:
    detected: bool
    point_2d: Optional[list[float]] = None
    bbox_2d: Optional[list[float]] = None
    label: str = ""
    confidence: float = 0.0
    fallback_reason: str = ""


class GroundedSAMWaypointRefiner:
    """Lazy optional GroundingDINO/SAM wrapper.

    Dependencies are imported only when this class is instantiated, so the
    default Qwen-only path does not require GroundingDINO or segment-anything.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.box_threshold = float(getattr(cfg, "box_threshold", 0.25))
        self.text_threshold = float(getattr(cfg, "text_threshold", 0.25))
        self.max_box_area_ratio = float(getattr(cfg, "max_box_area_ratio", 0.95))
        self.min_mask_area_px = int(getattr(cfg, "min_mask_area_px", 16))
        self.device = str(getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))

        dino_config = Path(str(getattr(cfg, "dino_config_path", ""))).expanduser()
        dino_ckpt = Path(str(getattr(cfg, "dino_checkpoint_path", ""))).expanduser()
        sam_ckpt = Path(str(getattr(cfg, "sam_checkpoint_path", ""))).expanduser()
        missing = [str(p) for p in (dino_config, dino_ckpt, sam_ckpt) if not p.exists()]
        if missing and bool(getattr(cfg, "fail_fast_on_missing_assets", True)):
            raise FileNotFoundError(
                "GroundedSAM enabled but required asset path(s) are missing: "
                + ", ".join(missing)
            )

        try:
            from groundingdino.util.inference import Model
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:  # pragma: no cover - depends on optional packages
            raise ImportError(
                "GroundedSAM requires GroundingDINO and segment-anything. "
                "Install them or set grounded_sam.enabled=false."
            ) from exc

        self.grounding_dino_model = Model(
            model_config_path=str(dino_config),
            model_checkpoint_path=str(dino_ckpt),
            device=self.device,
        )
        sam_encoder = str(getattr(cfg, "sam_encoder_version", "vit_h"))
        sam = sam_model_registry[sam_encoder](checkpoint=str(sam_ckpt)).to(device=self.device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)
        self.grounding_dino_model.model.eval()

    @torch.no_grad()
    def refine(self, image_rgb: np.ndarray, target: str) -> GroundedSAMRefineResult:
        target = str(target or "").strip()
        if not target:
            return GroundedSAMRefineResult(False, fallback_reason="empty_target")
        if image_rgb is None or image_rgb.ndim != 3:
            return GroundedSAMRefineResult(False, fallback_reason="invalid_image")

        classes = [target]
        try:
            detections = self.grounding_dino_model.predict_with_classes(
                image=image_rgb,
                classes=classes,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
        except Exception as exc:
            return GroundedSAMRefineResult(False, fallback_reason=f"dino_error:{type(exc).__name__}")

        xyxy = getattr(detections, "xyxy", None)
        if xyxy is None or len(xyxy) == 0:
            return GroundedSAMRefineResult(False, fallback_reason="no_detection")

        h, w = image_rgb.shape[:2]
        confidences = getattr(detections, "confidence", None)
        class_ids = getattr(detections, "class_id", None)
        valid_indices: list[int] = []
        for i, box in enumerate(np.asarray(xyxy)):
            x1, y1, x2, y2 = [float(v) for v in box]
            area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, float(w * h))
            if area_ratio >= self.max_box_area_ratio:
                continue
            valid_indices.append(i)
        if not valid_indices:
            return GroundedSAMRefineResult(False, fallback_reason="only_full_image_boxes")

        if confidences is None:
            best_i = valid_indices[0]
            conf = 0.0
        else:
            best_i = max(valid_indices, key=lambda i: float(confidences[i]))
            conf = float(confidences[best_i])

        box = np.asarray(xyxy[best_i], dtype=np.float32)
        masks = self._segment(image_rgb, box[None, :])
        mask = masks[0] if masks is not None and len(masks) > 0 else None
        point_px = self._mask_target_point(mask, box, h, w)
        bbox_2d = self._normalize_box(box, w, h)
        point_2d = [float(point_px[0] / max(1, w) * 1000.0), float(point_px[1] / max(1, h) * 1000.0)]

        label = target
        if class_ids is not None:
            cid = class_ids[best_i]
            if cid is not None and int(cid) < len(classes):
                label = classes[int(cid)]
        return GroundedSAMRefineResult(
            detected=True,
            point_2d=[self._clip1000(point_2d[0]), self._clip1000(point_2d[1])],
            bbox_2d=bbox_2d,
            label=label,
            confidence=conf,
        )

    def _segment(self, image_rgb: np.ndarray, xyxy: np.ndarray) -> Optional[np.ndarray]:
        try:
            self.sam_predictor.set_image(image_rgb)
            result_masks = []
            for box in xyxy:
                masks, scores, _logits = self.sam_predictor.predict(
                    box=box,
                    multimask_output=True,
                )
                result_masks.append(masks[int(np.argmax(scores))])
            return np.asarray(result_masks)
        except Exception:
            return None

    def _mask_target_point(
        self,
        mask: Optional[np.ndarray],
        box: np.ndarray,
        h: int,
        w: int,
    ) -> tuple[float, float]:
        if mask is not None and int(mask.sum()) >= self.min_mask_area_px:
            ys, xs = np.where(mask)
            if len(xs) > 0:
                y_bottom = float(np.percentile(ys, 90))
                band = ys >= max(0.0, y_bottom - 3.0)
                x_center = float(np.median(xs[band])) if np.any(band) else float(np.median(xs))
                return (min(max(x_center, 0.0), w - 1.0), min(max(y_bottom, 0.0), h - 1.0))

        x1, y1, x2, y2 = [float(v) for v in box]
        return (
            min(max((x1 + x2) * 0.5, 0.0), w - 1.0),
            min(max(y2, 0.0), h - 1.0),
        )

    @staticmethod
    def _normalize_box(box: np.ndarray, w: int, h: int) -> list[float]:
        x1, y1, x2, y2 = [float(v) for v in box]
        return [
            GroundedSAMWaypointRefiner._clip1000(x1 / max(1, w) * 1000.0),
            GroundedSAMWaypointRefiner._clip1000(y1 / max(1, h) * 1000.0),
            GroundedSAMWaypointRefiner._clip1000(x2 / max(1, w) * 1000.0),
            GroundedSAMWaypointRefiner._clip1000(y2 / max(1, h) * 1000.0),
        ]

    @staticmethod
    def _clip1000(v: float) -> float:
        return float(min(max(v, 0.0), 1000.0))

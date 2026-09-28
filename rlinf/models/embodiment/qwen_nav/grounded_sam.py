"""
Optional GroundingDINO + SAM waypoint refiner for QwenNav.

This is intentionally a thin adapter around LaViRA's GroundedSAM idea:
given the target phrase emitted by the VLM and the RGB view selected by the
VLM action, detect/segment the target and return a normalized waypoint point
plus bbox in the existing 0-1000 QwenNav coordinate convention.
"""

from __future__ import annotations

import pickle
import socket
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from .canonical_targets import region_for_box


def waypoint_bbox_scene_metrics(
    bbox_2d: list[float],
    *,
    max_area_ratio: float = 0.65,
    edge_margin_ratio: float = 0.02,
) -> dict[str, Any]:
    """Classify a normalized 0-1000 bbox before depth/map validation."""
    x1, y1, x2, y2 = [float(value) for value in bbox_2d]
    area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / 1_000_000.0
    margin = 1000.0 * float(edge_margin_ratio)
    edge_count = sum((x1 <= margin, y1 <= margin, x2 >= 1000.0 - margin, y2 >= 1000.0 - margin))
    return {
        "area_ratio": float(area_ratio),
        "edge_count": int(edge_count),
        "scene_region": bool(area_ratio > float(max_area_ratio) or edge_count >= 3),
    }


@dataclass
class GroundedSAMRefineResult:
    detected: bool
    point_2d: Optional[list[float]] = None
    bbox_2d: Optional[list[float]] = None
    label: str = ""
    confidence: float = 0.0
    fallback_reason: str = ""
    # Kept only for optional evaluation diagnostics. The controller continues
    # to use point_2d/bbox_2d, so normal rollout behavior is unchanged.
    mask: Optional[np.ndarray] = None
    candidates: Optional[list[dict[str, Any]]] = None


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
        self.waypoint_scene_box_area_ratio = float(
            getattr(cfg, "waypoint_scene_box_area_ratio", 0.65)
        )
        self.waypoint_scene_edge_margin_ratio = float(
            getattr(cfg, "waypoint_scene_edge_margin_ratio", 0.02)
        )
        self.reject_scene_region_source_waypoint = bool(
            getattr(cfg, "reject_scene_region_source_waypoint", False)
        )
        self.min_mask_area_px = int(getattr(cfg, "min_mask_area_px", 16))
        self.device = str(getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))

        dino_config = Path(str(getattr(cfg, "dino_config_path", ""))).expanduser()
        dino_ckpt = Path(str(getattr(cfg, "dino_checkpoint_path", ""))).expanduser()
        sam_ckpt = Path(
            str(getattr(cfg, "repvit_sam_checkpoint_path", ""))
        ).expanduser()
        missing = [str(p) for p in (dino_config, dino_ckpt, sam_ckpt) if not p.exists()]
        if missing and bool(getattr(cfg, "fail_fast_on_missing_assets", True)):
            raise FileNotFoundError(
                "GroundedSAM enabled but required asset path(s) are missing: "
                + ", ".join(missing)
            )

        try:
            from groundingdino.util.inference import Model
            from segment_anything import SamPredictor
            from .repvit_sam import build_sam_repvit
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
        # LHX R2R fixes MAP.REPVITSAM=1. Use the vendored source builder rather
        # than keeping a second ViT-H semantic-map mode.
        sam = build_sam_repvit(checkpoint=str(sam_ckpt)).to(device=self.device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)
        self.grounding_dino_model.model.eval()

    @torch.no_grad()
    def refine(
        self,
        image_rgb: np.ndarray,
        target: str,
        target_region: str = "any",
        *,
        grounding_classes: Optional[list[str]] = None,
        source_lhx: bool = False,
    ) -> GroundedSAMRefineResult:
        target = str(target or "").strip()
        classes = [
            str(value).strip()
            for value in (grounding_classes or [target])
            if str(value).strip()
        ]
        if not classes:
            return GroundedSAMRefineResult(False, fallback_reason="empty_target")
        if image_rgb is None or image_rgb.ndim != 3:
            return GroundedSAMRefineResult(False, fallback_reason="invalid_image")
        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        detections = self.grounding_dino_model.predict_with_classes(
            image=image_bgr,
            classes=classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        return self._refine_from_detections(
            image_rgb,
            target,
            target_region,
            detections,
            classes=classes,
            source_lhx=source_lhx,
        )

    @torch.no_grad()
    def refine_batch(self, jobs: list[dict[str, Any]]) -> list[GroundedSAMRefineResult]:
        return [
            self.refine(
                job.get("image_rgb"),
                job.get("target", ""),
                job.get("target_region", "any"),
                grounding_classes=job.get("grounding_classes"),
                source_lhx=bool(job.get("source_lhx", False)),
            )
            for job in jobs
        ]

    def _refine_from_detections(
        self,
        image_rgb: np.ndarray,
        target: str,
        target_region: str,
        detections: Any,
        *,
        classes: Optional[list[str]] = None,
        source_lhx: bool = False,
    ) -> GroundedSAMRefineResult:
        classes = list(classes or [target])

        xyxy = getattr(detections, "xyxy", None)
        if xyxy is None or len(xyxy) == 0:
            return GroundedSAMRefineResult(False, fallback_reason="no_detection")

        h, w = image_rgb.shape[:2]
        confidences = getattr(detections, "confidence", None)
        class_ids = getattr(detections, "class_id", None)
        # LHX waypoint grounding does not discard large detections here. Its
        # _ground_target_with_dino() ranks every returned box and leaves map
        # traversibility/depth checks to the downstream projection boundary.
        valid_indices = list(range(len(xyxy)))

        if confidences is None:
            boxes = np.asarray(xyxy, dtype=np.float32)
            areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
                0.0, boxes[:, 3] - boxes[:, 1]
            )
            base_order = sorted(valid_indices, key=lambda i: (-float(areas[i]), i))
            confidences_values = [0.0] * len(xyxy)
        else:
            confidences_values = [float(v) for v in confidences]
            base_order = sorted(
                valid_indices,
                key=lambda i: (-confidences_values[i], i),
            )

        candidates: list[dict[str, Any]] = []
        for index in base_order:
            box_i = np.asarray(xyxy[index], dtype=np.float32)
            normalized = self._normalize_box(box_i, w, h)
            label_i = target
            if class_ids is not None:
                cid = class_ids[index]
                if cid is not None and int(cid) < len(classes):
                    label_i = classes[int(cid)]
            scene_metrics = waypoint_bbox_scene_metrics(
                normalized,
                max_area_ratio=float(
                    getattr(self, "waypoint_scene_box_area_ratio", 1.0)
                ),
                edge_margin_ratio=float(
                    getattr(self, "waypoint_scene_edge_margin_ratio", 0.0)
                ),
            )
            candidates.append({
                "index": int(index),
                "candidate_type": "dino",
                "bbox_2d": normalized,
                "label": label_i,
                "confidence": float(confidences_values[index]),
                "region": region_for_box(normalized),
                **scene_metrics,
            })

        if source_lhx and not bool(
            getattr(self, "reject_scene_region_source_waypoint", False)
        ):
            # ZS_Evaluator_mp._ground_target_with_dino() ranks all returned
            # boxes globally. It does not reject scene-sized boxes or apply a
            # region preference before selecting the waypoint bbox.
            selected = candidates[0]
            matching = candidates
        else:
            localized = [
                candidate for candidate in candidates
                if not candidate["scene_region"]
            ]
            if not localized:
                return GroundedSAMRefineResult(
                    detected=False,
                    label=target,
                    candidates=candidates,
                    fallback_reason="scene_region_bbox",
                )
            matching = [
                candidate for candidate in localized
                if target_region == "any" or candidate["region"] == target_region
            ]
            selected = matching[0] if matching else localized[0]
        best_i = int(selected["index"])
        conf = float(selected["confidence"])

        box = np.asarray(xyxy[best_i], dtype=np.float32)
        bbox_2d = self._normalize_box(box, w, h)

        label = target
        if class_ids is not None:
            cid = class_ids[best_i]
            if cid is not None and int(cid) < len(classes):
                label = classes[int(cid)]
        return GroundedSAMRefineResult(
            detected=True,
            # Match LaViRA: waypoint projection uses the highest-confidence
            # GroundingDINO bbox bottom-center. SAM remains responsible for
            # semantic-map masks in segment_classes(), not waypoint location.
            point_2d=None,
            bbox_2d=bbox_2d,
            label=label,
            confidence=conf,
            mask=None,
            candidates=candidates,
            fallback_reason=(
                "region_fallback"
                if not source_lhx and not matching and target_region != "any"
                else ""
            ),
        )

    @torch.no_grad()
    def segment_classes(self, image_rgb: np.ndarray, classes: list[str]) -> dict[str, np.ndarray]:
        """LaViRA-RFT semantic-map interface.

        Unlike :meth:`refine`, this keeps every accepted instance mask and
        unions instances by class.  It is invoked on every primitive frame by
        the full runtime, so the result can be accumulated into persistent
        semantic-map channels rather than used only as a current waypoint.
        """
        if image_rgb is None or image_rgb.ndim != 3 or not classes:
            return {}
        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        detections = self.grounding_dino_model.predict_with_classes(
            image=image_bgr,
            classes=classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        return self._segment_classes_from_detections(
            image_rgb, classes, detections
        )

    @torch.no_grad()
    def segment_classes_batch(
        self, jobs: list[dict[str, Any]]
    ) -> list[dict[str, np.ndarray]]:
        return [
            self.segment_classes(
                job.get("image_rgb"), list(job.get("classes") or [])
            )
            for job in jobs
        ]

    def _segment_classes_from_detections(
        self,
        image_rgb: np.ndarray,
        classes: list[str],
        detections: Any,
    ) -> dict[str, np.ndarray]:
        xyxy = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
        class_ids = getattr(detections, "class_id", None)
        if len(xyxy) == 0 or class_ids is None:
            return {}
        h, w = image_rgb.shape[:2]
        valid = []
        for idx, box in enumerate(xyxy):
            x1, y1, x2, y2 = box
            if (
                max(0.0, x2 - x1) * max(0.0, y2 - y1)
                / max(1.0, h * w)
                < float(getattr(self, "max_box_area_ratio", 0.95))
            ):
                valid.append(idx)
        if not valid:
            return {}
        masks = self._segment(image_rgb, xyxy[valid])
        if masks is None:
            return {}
        result: dict[str, np.ndarray] = {}
        for local_idx, original_idx in enumerate(valid):
            class_id = class_ids[original_idx]
            if class_id is None or not (0 <= int(class_id) < len(classes)):
                continue
            label = str(classes[int(class_id)]).strip().lower()
            mask = np.asarray(masks[local_idx], dtype=np.float32)
            result[label] = mask if label not in result else result[label] + mask
        return result

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


_MESSAGE_HEADER = struct.Struct("!Q")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("GroundedSAM service closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_rpc_message(sock: socket.socket, payload: Any) -> None:
    encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_MESSAGE_HEADER.pack(len(encoded)))
    sock.sendall(encoded)


def recv_rpc_message(sock: socket.socket) -> Any:
    size = _MESSAGE_HEADER.unpack(_recv_exact(sock, _MESSAGE_HEADER.size))[0]
    return pickle.loads(_recv_exact(sock, size))


class _GroundedSAMRPCClient:
    def __init__(self, endpoint: str, timeout_s: float):
        host, port = str(endpoint).rsplit(":", 1)
        self.endpoint = str(endpoint)
        self._socket = socket.create_connection(
            (host, int(port)), timeout=float(timeout_s)
        )
        self._socket.settimeout(float(timeout_s))
        self._lock = threading.Lock()
        health = self.request("health", None)
        if not health.get("ready", False):
            raise RuntimeError(f"GroundedSAM service is not ready: {endpoint}")

    def request(self, method: str, payload: Any) -> Any:
        with self._lock:
            send_rpc_message(
                self._socket, {"method": method, "payload": payload}
            )
            response = recv_rpc_message(self._socket)
        if response.get("status") != "ok":
            raise RuntimeError(
                f"GroundedSAM service {self.endpoint} failed: "
                f"{response.get('error', 'unknown error')}"
            )
        return response.get("result")

    def close(self) -> None:
        try:
            self.request("close", None)
        except Exception:
            pass
        self._socket.close()


class GroundedSAMServicePool:
    """Route slot-owned GroundedSAM jobs to independent GPU services."""

    def __init__(self, cfg: DictConfig):
        remote_cfg = getattr(cfg, "remote", None)
        endpoints = list(getattr(remote_cfg, "endpoints", []) or [])
        if not endpoints:
            raise ValueError("grounded_sam.remote.endpoints must not be empty")
        self.slots_per_service = int(
            getattr(remote_cfg, "slots_per_service", 1)
        )
        if self.slots_per_service < 1:
            raise ValueError("grounded_sam.remote.slots_per_service must be positive")
        timeout_s = float(getattr(remote_cfg, "timeout_s", 300.0))
        self._clients = [
            _GroundedSAMRPCClient(endpoint, timeout_s) for endpoint in endpoints
        ]
        self._executor = ThreadPoolExecutor(
            max_workers=len(self._clients), thread_name_prefix="grounded-sam-rpc"
        )

    def _service_index(self, env_i: int) -> int:
        index = int(env_i) // self.slots_per_service
        if not 0 <= index < len(self._clients):
            raise IndexError(
                f"env_i={env_i} maps to GroundedSAM service {index}, "
                f"but only {len(self._clients)} services are configured"
            )
        return index

    def _dispatch(self, method: str, jobs: list[dict]) -> list[Any]:
        if not jobs:
            return []
        grouped: list[list[tuple[int, dict]]] = [
            [] for _ in self._clients
        ]
        for result_index, job in enumerate(jobs):
            service_index = self._service_index(int(job["env_i"]))
            payload = dict(job)
            payload.pop("env_i", None)
            grouped[service_index].append((result_index, payload))

        futures = {}
        for service_index, service_jobs in enumerate(grouped):
            if not service_jobs:
                continue
            payload = [job for _, job in service_jobs]
            futures[service_index] = self._executor.submit(
                self._clients[service_index].request, method, payload
            )

        results: list[Any] = [None] * len(jobs)
        for service_index, future in futures.items():
            service_results = future.result()
            service_jobs = grouped[service_index]
            if len(service_results) != len(service_jobs):
                raise RuntimeError(
                    f"GroundedSAM service {service_index} returned "
                    f"{len(service_results)} results for {len(service_jobs)} jobs"
                )
            for (result_index, _), value in zip(service_jobs, service_results):
                results[result_index] = value
        return results

    def segment_classes_batch(self, jobs: list[dict]) -> list[dict[str, np.ndarray]]:
        return self._dispatch("segment_classes_batch", jobs)

    def refine_batch(self, jobs: list[dict]) -> list[GroundedSAMRefineResult]:
        values = self._dispatch("refine_batch", jobs)
        return [GroundedSAMRefineResult(**value) for value in values]

    def segment_classes(
        self, image_rgb: np.ndarray, classes: list[str], *, env_i: int = 0
    ) -> dict[str, np.ndarray]:
        return self.segment_classes_batch([
            {"env_i": env_i, "image_rgb": image_rgb, "classes": classes}
        ])[0]

    def refine(
        self,
        image_rgb: np.ndarray,
        target: str,
        target_region: str = "any",
        *,
        env_i: int = 0,
        grounding_classes: Optional[list[str]] = None,
        source_lhx: bool = False,
    ) -> GroundedSAMRefineResult:
        return self.refine_batch([
            {
                "env_i": env_i,
                "image_rgb": image_rgb,
                "target": target,
                "target_region": target_region,
                "grounding_classes": grounding_classes,
                "source_lhx": source_lhx,
            }
        ])[0]

    def close(self) -> None:
        for client in self._clients:
            client.close()
        self._executor.shutdown(wait=True)


def build_grounded_sam(cfg: DictConfig):
    remote_cfg = getattr(cfg, "remote", None)
    if bool(getattr(remote_cfg, "enabled", False)):
        return GroundedSAMServicePool(cfg)
    return GroundedSAMWaypointRefiner(cfg)

"""Opt-in profiling and visual diagnostics for Qwen navigation."""

from __future__ import annotations

import gzip
import json
import os
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from .action_parser import ParsedAction
from .lavira_runtime import LaviraNavigationController, LaviraObservation
from .waypoint_runtime import _HistoryCache


class PolicyDiagnosticsMixin:
    """Diagnostics that must not alter policy or controller decisions."""

    def _profile_add(
        self,
        stage: str,
        elapsed_s: float,
        *,
        items: int = 0,
    ) -> None:
        """Accumulate opt-in wall timing without synchronizing CUDA."""
        if not getattr(self, "_rollout_profile_enabled", False):
            return
        stats = self._rollout_profile_stats.setdefault(
            stage, {"seconds": 0.0, "calls": 0, "items": 0}
        )
        stats["seconds"] = float(stats["seconds"]) + float(elapsed_s)
        stats["calls"] = int(stats["calls"]) + 1
        stats["items"] = int(stats["items"]) + int(items)

    def _profile_flush_if_due(self) -> None:
        if not getattr(self, "_rollout_profile_enabled", False):
            return
        self._rollout_profile_batches += 1
        if self._rollout_profile_batches % self._rollout_profile_flush_every:
            return
        stages = {}
        for name, values in sorted(self._rollout_profile_stats.items()):
            calls = max(int(values["calls"]), 1)
            stages[name] = {
                "seconds": round(float(values["seconds"]), 6),
                "calls": int(values["calls"]),
                "items": int(values["items"]),
                "mean_ms": round(
                    1000.0 * float(values["seconds"]) / calls, 3
                ),
            }
        print(
            "[RolloutProfile][policy] "
            + json.dumps(
                {
                    "batches": self._rollout_profile_batches,
                    "stages": stages,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _profiled_batch_generate(
        self,
        prompts: list[str],
        image_lists: list[list[Image.Image]],
    ) -> tuple[list[str], list[torch.Tensor]]:
        started = time.perf_counter()
        try:
            return self._batch_generate(prompts, image_lists)
        finally:
            self._profile_add(
                "qwen_generate",
                time.perf_counter() - started,
                items=len(prompts),
            )

    def _profiled_next_primitive(
        self,
        controller: LaviraNavigationController,
        observation: LaviraObservation,
    ) -> Optional[int]:
        started = time.perf_counter()
        try:
            return controller.next_primitive(observation)
        finally:
            self._profile_add(
                "controller_fmm",
                time.perf_counter() - started,
                items=1,
            )

    def _profiled_observe(
        self,
        controller: LaviraNavigationController,
        observation: LaviraObservation,
        semantic_masks: Optional[dict],
        *,
        publish_planner_state: bool,
    ) -> None:
        started = time.perf_counter()
        try:
            controller.observe(
                observation,
                semantic_masks,
                publish_planner_state=publish_planner_state,
            )
        finally:
            self._profile_add(
                "map_fusion",
                time.perf_counter() - started,
                items=1,
            )

    def _load_lavira_map_reference_paths(self) -> dict[str, np.ndarray]:
        """Load GT paths keyed by episode for map-overlay diagnostics."""
        cfg = self._lavira_map_visualization_cfg
        dataset_path = str(getattr(cfg, "episode_json", "") if cfg is not None else "").strip()
        if not self.lavira_map_visualize or not dataset_path:
            return {}
        try:
            opener = gzip.open if dataset_path.endswith(".gz") else open
            with opener(dataset_path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
            paths: dict[str, np.ndarray] = {}
            for episode in episodes:
                episode_id = str(episode.get("episode_id", "")).strip()
                path = np.asarray(episode.get("reference_path", []), dtype=np.float32)
                if episode_id and path.ndim == 2 and path.shape[1] >= 3 and len(path):
                    paths[episode_id] = path
            return paths
        except Exception as exc:
            print(
                f"[QwenNav][map-visualize] gt_load_failed={type(exc).__name__}",
                flush=True,
            )
        return {}

    def _lavira_map_reference_path_for_env(self, env_i: int) -> np.ndarray | None:
        """Resolve the GT overlay from the episode currently active in a slot."""
        episode_id = self._lavira_map_episode_ids_by_env.get(env_i, "")
        if episode_id:
            path = self._lavira_map_gt_reference_paths.get(episode_id)
            if path is not None:
                return path
        return self._lavira_map_default_gt_reference_path

    def _save_lavira_map_snapshot(self, env_i: int) -> None:
        """Persist one post-observation 2D map snapshot without affecting control."""
        if (
            not self.lavira_map_visualize
            or self._lavira_map_visualization_dir is None
            or not self.lavira_runtime_enabled
        ):
            return
        try:
            cache = self._get_cache(env_i)
            episode_idx = self._lavira_map_episode_counts.get(env_i, 0)
            out_dir = (
                self._lavira_map_visualization_dir
                / f"worker_{os.getpid()}"
                / f"env_{env_i:02d}"
                / f"episode_{episode_idx:03d}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            step = cache.step_count
            npz_path = None
            if self._lavira_map_save_raw_every > 0 and step % self._lavira_map_save_raw_every == 0:
                npz_path = str(out_dir / f"map_{step:04d}.npz")
            controller = self._get_lavira_runtime(env_i)
            if (
                self._lavira_map_save_every > 0
                and step % self._lavira_map_save_every != 0
            ):
                return
            controller.map.save_debug_snapshot(
                str(out_dir / f"map_{step:04d}.png"),
                npz_path=npz_path,
                gt_reference_path=self._lavira_map_reference_path_for_env(env_i),
                goal_xz=controller.goal_xz,
            )
        except Exception as exc:
            print(
                f"[QwenNav][map-visualize] write_failed={type(exc).__name__}",
                flush=True,
            )

    def _write_lavira_projection_audit(
        self,
        env_i: int,
        observation: LaviraObservation,
        primitive: Optional[int],
        *,
        is_new_decision: bool,
        prompt_text: Optional[str] = None,
        model_output: Optional[str] = None,
    ) -> None:
        """Append one controller provenance record beside opt-in map diagnostics."""
        if not self.lavira_map_visualize or self._lavira_map_visualization_dir is None:
            return
        try:
            cache = self._get_cache(env_i)
            episode_idx = self._lavira_map_episode_counts.get(env_i, 0)
            out_dir = (
                self._lavira_map_visualization_dir
                / f"worker_{os.getpid()}"
                / f"env_{env_i:02d}"
                / f"episode_{episode_idx:03d}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = self._get_lavira_runtime(env_i).audit_snapshot(observation, primitive)
            payload.update({
                "sim_step": int(cache.step_count),
                "env_id": int(env_i),
                "is_new_decision": bool(is_new_decision),
            })
            # Keep raw decision provenance only in opt-in map/audit mode. This
            # lets a single-episode Habitat run be compared against LaViRA's
            # DEBUG_LOGGING prompt/response artifacts without affecting policy
            # inputs, batching, or normal evaluation output.
            if prompt_text is not None:
                payload["prompt_text"] = str(prompt_text)
            if model_output is not None:
                payload["model_output"] = str(model_output)
            with (out_dir / "projection_audit.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception as exc:
            print(
                f"[QwenNav][projection-audit] write_failed={type(exc).__name__}",
                flush=True,
            )

    def _save_grounded_sam_diagnostic(
        self,
        *,
        env_i: int,
        cache: _HistoryCache,
        instruction: str,
        parsed: ParsedAction,
        raw_model_output: str,
        current_views: list[Image.Image],
        runtime_observation: Optional[LaviraObservation],
        sam_result,
        diag: Optional[dict],
    ) -> None:
        """Persist one decision's target grounding evidence for eval review.

        The source image is exactly the direction view passed to GroundingDINO.
        `decision.json` preserves the unmodified model output alongside the
        parsed/final action, while `overlay.png` makes any DINO box/SAM mask
        inspectable without changing the navigation path.
        """
        if not self.grounded_sam_visualize or self._grounded_sam_visualization_dir is None:
            return
        if (
            self.grounded_sam_visualize_every > 0
            and cache.step_count % self.grounded_sam_visualize_every != 0
        ):
            return
        try:
            detected = bool(diag and diag.get("detected", False))
            episode_idx = self._grounded_sam_episode_counts.get(env_i, 0)
            decision_dir = (
                self._grounded_sam_visualization_dir
                / f"worker_{os.getpid()}"
                / f"env_{env_i:02d}"
                / f"episode_{episode_idx:03d}"
                / f"decision_{cache.step_count:04d}"
            )
            decision_dir.mkdir(parents=True, exist_ok=True)

            # Prompt views are [front,left,right,behind]; source runtime depth
            # is [front,left,behind,right]. The explicit map prevents a visual
            # audit from pairing a DINO box with another direction's depth.
            view_spec = {
                "navigate to forward": ("front", 0, 0),
                "navigate to left": ("left", 1, 1),
                "navigate to right": ("right", 2, 3),
                "navigate to behind": ("behind", 3, 2),
            }
            direction, view_idx, depth_idx = view_spec.get(
                str(parsed.raw_dir or ""), ("front", 0, 0)
            )
            if view_idx >= len(current_views):
                direction, view_idx, depth_idx = "front", 0, 0
            source = current_views[view_idx].convert("RGB")
            source.save(decision_dir / "source.png")
            source.save(decision_dir / "selected_direction_rgb.png")

            overlay = source.convert("RGBA")
            draw = ImageDraw.Draw(overlay)
            bbox = parsed.bbox_2d or (diag or {}).get("bbox_2d")
            point = (diag or {}).get("point_2d")
            bottom_center_1000 = None
            bottom_center_px = None
            depth_summary = None
            # Also render the source-compatible centre-bbox fallback.  This
            # makes a detector miss distinguishable from a bad depth ray in a
            # single directory, without changing either action or controller.
            if bbox and len(bbox) == 4:
                w, h = overlay.size
                x1, y1, x2, y2 = [float(v) for v in bbox]
                px_box = [x1 * w / 1000.0, y1 * h / 1000.0,
                          x2 * w / 1000.0, y2 * h / 1000.0]
                bottom_center_1000 = [(x1 + x2) / 2.0, y2]
                bottom_center_px = [
                    int(np.clip(round(bottom_center_1000[0] * w / 1000.0), 0, w - 1)),
                    int(np.clip(round(bottom_center_1000[1] * h / 1000.0), 0, h - 1)),
                ]
                mask = getattr(sam_result, "mask", None)
                if mask is not None and np.asarray(mask).shape[:2] == (h, w):
                    rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    rgba[np.asarray(mask, dtype=bool)] = [0, 255, 0, 80]
                    overlay = Image.alpha_composite(overlay, Image.fromarray(rgba, "RGBA"))
                    draw = ImageDraw.Draw(overlay)
                draw.rectangle(px_box, outline=(0, 255, 0, 255), width=4)
                px, py = bottom_center_px
                draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(255, 64, 64, 255))
                if runtime_observation is not None:
                    depth = np.asarray(
                        runtime_observation.depth_by_direction[depth_idx], dtype=np.float32
                    )
                    dh, dw = depth.shape
                    dx = int(np.clip(round(bottom_center_1000[0] * dw / 1000.0), 0, dw - 1))
                    dy = int(np.clip(round(bottom_center_1000[1] * dh / 1000.0), 0, dh - 1))
                    patch = depth[max(0, dy - 3):min(dh, dy + 4),
                                  max(0, dx - 3):min(dw, dx + 4)].copy()
                    np.save(decision_dir / "bbox_bottom_center_depth_patch.npy", patch)
                    valid = patch[np.isfinite(patch) & (patch > 0.0)]
                    depth_summary = {
                        "patch_path": "bbox_bottom_center_depth_patch.npy",
                        "pixel_xy": [dx, dy],
                        "patch_shape": list(patch.shape),
                        "center_depth_m": float(depth[dy, dx]),
                        "finite_positive_count": int(valid.size),
                        "min_m": float(valid.min()) if valid.size else None,
                        "median_m": float(np.median(valid)) if valid.size else None,
                        "max_m": float(valid.max()) if valid.size else None,
                    }
            if point and len(point) == 2:
                w, h = overlay.size
                px = float(point[0]) * w / 1000.0
                py = float(point[1]) * h / 1000.0
                draw.ellipse(
                    [px - 6, py - 6, px + 6, py + 6],
                    fill=(255, 64, 64, 255),
                )
                if runtime_observation is not None and bbox is None:
                    depth = np.asarray(
                        runtime_observation.depth_by_direction[depth_idx],
                        dtype=np.float32,
                    )
                    dh, dw = depth.shape
                    dx = int(np.clip(float(point[0]) * dw / 1000.0, 0, dw - 1))
                    dy = int(np.clip(float(point[1]) * dh / 1000.0, 0, dh - 1))
                    patch = depth[
                        max(0, dy - 3):min(dh, dy + 4),
                        max(0, dx - 3):min(dw, dx + 4),
                    ].copy()
                    np.save(decision_dir / "area_point_depth_patch.npy", patch)
                    valid = patch[np.isfinite(patch) & (patch > 0.0)]
                    depth_summary = {
                        "patch_path": "area_point_depth_patch.npy",
                        "pixel_xy": [dx, dy],
                        "patch_shape": list(patch.shape),
                        "center_depth_m": float(depth[dy, dx]),
                        "finite_positive_count": int(valid.size),
                        "min_m": float(valid.min()) if valid.size else None,
                        "median_m": float(np.median(valid)) if valid.size else None,
                        "max_m": float(valid.max()) if valid.size else None,
                    }
            summary = (
                f"target={parsed.target!r} label={(diag or {}).get('label', '')!r} "
                f"detected={detected} conf={(diag or {}).get('confidence', 0.0):.3f} "
                f"fallback={(diag or {}).get('fallback_reason', '')!s}"
            )
            draw.rectangle([0, 0, overlay.size[0], 22], fill=(0, 0, 0, 180))
            draw.text((4, 4), summary[:180], fill=(255, 255, 255, 255))
            overlay.convert("RGB").save(decision_dir / "overlay.png")

            payload = {
                "env_id": env_i,
                "episode_index": episode_idx,
                "sim_step": cache.step_count,
                "instruction": instruction,
                "view": direction,
                "view_indices": {"prompt_rgb": view_idx, "canonical_depth": depth_idx},
                "target": parsed.target,
                "raw_target": parsed.raw_target,
                "target_mapping": self._grounding_target_diag(parsed),
                "target_region": parsed.target_region,
                "selected_action": parsed.raw_dir,
                "raw_model_output": raw_model_output,
                "parsed": {
                    "ok": bool(parsed.ok), "error": parsed.err,
                    "action_type": parsed.action_type, "actions": list(parsed.actions),
                    "stop": bool(parsed.stop), "target": parsed.target,
                    "raw_target": parsed.raw_target,
                    "target_region": parsed.target_region,
                    "point_2d": parsed.point_2d, "bbox_2d": parsed.bbox_2d,
                    "planning": parsed.planning,
                    "reasoning_plan_action": parsed.reasoning_plan_action,
                    "reasoning_bbox_point": parsed.reasoning_bbox_point,
                },
                "grounded_sam": diag or {
                    "enabled": bool(self.grounded_sam_enabled),
                    "attempted": False,
                    "fallback_reason": "not_navigate_or_no_target",
                },
                "bbox_source": (
                    "dino" if detected else str((diag or {}).get(
                        "fallback_reason", "parser_or_runtime"
                    ))
                ),
                "bbox_bottom_center_1000": bottom_center_1000,
                "bbox_bottom_center_rgb_px": bottom_center_px,
                "depth_patch": depth_summary,
            }
            with (decision_dir / "decision.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[QwenNav][GroundedSAM][visualize] write_failed={type(exc).__name__}", flush=True)

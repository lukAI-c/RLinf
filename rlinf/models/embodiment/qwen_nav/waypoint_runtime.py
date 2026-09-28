"""Waypoint history, prompt, and grounding helpers for Qwen navigation."""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from .action_parser import (
    ACTION_FORWARD,
    ACTION_PARSE_FAIL,
    ACTION_PARSE_OK_HAS_BBOX_BASE,
    ACTION_SCHEMA_OK_PARSE_FAIL,
    ACTION_STRUCTURED_PARSE_FAIL_BASE,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    ParsedAction,
    VALID_DIRECTIONS,
)
from .canonical_targets import map_target_for_grounding, target_geometry_kind
from .grounded_sam import (
    GroundedSAMRefineResult,
    GroundedSAMServicePool,
    GroundedSAMWaypointRefiner,
    build_grounded_sam,
)
from .lavira_runtime import LaviraNavigationController, LaviraObservation
from .prompts import (
    LAVIRA_BACKTRACK_REPLAN_FAILED,
    LAVIRA_BACKTRACK_REPLAN_OUTPUT,
    LAVIRA_CANONICAL_SYSTEM_PROMPT,
    LAVIRA_MERGED_SYSTEM_PROMPT,
    LAVIRA_WAYPOINT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_backtrack_replan_user_content_text,
    build_merged_user_content_text,
    build_user_content_text,
    build_waypoint_user_content_text,
    expected_image_count,
)

class _WaypointRecord:
    """Compact per-env waypoint memory for lavira_waypoint prompts."""

    __slots__ = (
        "id", "pose_hab", "arrival_image", "after_turn_image",
        "continuous_frames", "action", "bbox_2d", "point_2d", "target",
        "progress_analysis", "history_end_index", "anchor_views",
        "anchor_depth_by_direction", "step", "failed", "failed_dir",
    )

    def __init__(
        self,
        waypoint_id: int,
        pose_hab: tuple[float, float, float],
        arrival_image: Image.Image,
        after_turn_image: Image.Image,
        continuous_frames: list[Image.Image],
        action: str,
        bbox_2d: list[float],
        point_2d: list[float],
        target: str,
        progress_analysis: str,
        history_end_index: int = 0,
        anchor_views: Optional[list[Image.Image]] = None,
        anchor_depth_by_direction: Optional[np.ndarray] = None,
        step: int = 0,
    ):
        self.id = waypoint_id
        self.pose_hab = pose_hab
        self.arrival_image = arrival_image
        self.after_turn_image = after_turn_image
        self.continuous_frames = continuous_frames
        self.action = action
        self.bbox_2d = bbox_2d
        self.point_2d = point_2d
        self.target = target
        self.progress_analysis = progress_analysis
        self.history_end_index = int(history_end_index)
        # LaViRA-RFT re-plans from the selected old waypoint's four panorama
        # views, not from the agent's later physical pose.  Keep its RGB-D
        # observation with the waypoint so that source timing remains possible
        # in RLinf's batched environment.
        self.anchor_views = [image.copy() for image in (anchor_views or [])]
        self.anchor_depth_by_direction = (
            None if anchor_depth_by_direction is None
            else np.asarray(anchor_depth_by_direction, dtype=np.float16).copy()
        )
        self.step = int(step)
        self.failed = False
        self.failed_dir = False

class _HistoryCache:
    """
    Per-env state for one episode under lavira single-turn rebuild.

    history_images : full record of past observation images (PIL.Image)
    pending_actions: macro-expanded action queue (eval mode only)
    last_parse_ok  : whether last JSON parsed cleanly (for format_reward)
    last_err       : error code from last parse failure
    last_bbox      : last predicted bbox (for logging)
    """

    __slots__ = (
        "history_images", "history_steps", "pending_actions", "step_count",
        "last_parse_ok", "last_err", "last_bbox",
        "last_point", "last_waypoint_id", "last_reasoning_plan_action",
        "last_reasoning_bbox_point", "last_backtrack_valid", "last_target",
        "waypoints", "next_waypoint_id", "last_waypoint_history_len",
        "stop_failure_count", "stop_rejection_feedback",
        "last_action", "forward_streak", "turn_streak",
        "stop_check_pending", "last_fused_observation_key",
        "last_history_observation_key", "last_history_physical_step",
    )

    def __init__(self):
        self.history_images: list[Image.Image] = []
        self.history_steps: list[int] = []
        self.pending_actions: list[int] = []
        self.step_count: int = 0
        self.last_parse_ok: bool = True
        self.last_err: Optional[str] = None
        self.last_bbox: Optional[list[float]] = None
        self.last_point: Optional[list[float]] = None
        self.last_waypoint_id: Optional[int] = None
        self.last_reasoning_plan_action: str = ""
        self.last_reasoning_bbox_point: str = ""
        self.last_backtrack_valid: bool = False
        self.last_target: str = ""
        self.waypoints: list[_WaypointRecord] = []
        self.next_waypoint_id: int = 0
        self.last_waypoint_history_len: int = 0
        # Stop double-check state (Lavira-style)
        self.stop_failure_count: int = 0        # consecutive rejected STOPs
        self.stop_rejection_feedback: str = ""  # injected into next prompt if non-empty
        self.last_action: Optional[int] = None
        self.forward_streak: int = 0
        self.turn_streak: int = 0
        self.stop_check_pending: bool = False
        # Map/GSAM fusion is keyed by the simulator observation, rather than
        # by policy calls.  This prevents a true NOOP from fusing the same
        # RGB-D frame repeatedly while still allowing every real primitive to
        # produce a new map observation.
        self.last_fused_observation_key = None
        self.last_history_observation_key = None
        self.last_history_physical_step: Optional[int] = None

    def reset(self):
        self.history_images.clear()
        self.history_steps.clear()
        self.pending_actions.clear()
        self.step_count = 0
        self.last_parse_ok = True
        self.last_err = None
        self.last_bbox = None
        self.last_point = None
        self.last_waypoint_id = None
        self.last_reasoning_plan_action = ""
        self.last_reasoning_bbox_point = ""
        self.last_backtrack_valid = False
        self.last_target = ""
        self.waypoints.clear()
        self.next_waypoint_id = 0
        self.last_waypoint_history_len = 0
        self.stop_failure_count = 0
        self.stop_rejection_feedback = ""
        self.last_action = None
        self.forward_streak = 0
        self.turn_streak = 0
        self.stop_check_pending = False
        self.last_fused_observation_key = None
        self.last_history_observation_key = None
        self.last_history_physical_step = None

    def push_image(
        self,
        img: Image.Image,
        *,
        action_aware: bool = False,
        forward_interval: int = 5,
        turn_interval: int = 2,
    ):
        should_append = not action_aware or not self.history_images
        action = self.last_action
        if action is not None and ACTION_PARSE_OK_HAS_BBOX_BASE <= action < ACTION_PARSE_OK_HAS_BBOX_BASE + 4:
            action -= ACTION_PARSE_OK_HAS_BBOX_BASE
        elif action in (ACTION_PARSE_FAIL, ACTION_SCHEMA_OK_PARSE_FAIL) or (
            action is not None and action >= ACTION_STRUCTURED_PARSE_FAIL_BASE
        ):
            # Both Genesis and the Habitat adapter execute parser failures as
            # MOVE_FORWARD, matching ZS_Evaluator_mp's exhausted-retry fallback.
            action = ACTION_FORWARD
        if action_aware and self.history_images:
            if action == ACTION_FORWARD:
                self.forward_streak += 1
                self.turn_streak = 0
                if self.forward_streak >= max(1, forward_interval):
                    should_append = True
                    self.forward_streak = 0
            elif action in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT):
                self.turn_streak += 1
                self.forward_streak = 0
                if self.turn_streak >= max(1, turn_interval):
                    should_append = True
                    self.turn_streak = 0
        if should_append:
            self.history_images.append(img)
        self.step_count += 1

    def mark_failed_branch(self, waypoint_id: Optional[int]) -> bool:
        """Match source replan bookkeeping for the abandoned branch."""
        if waypoint_id is None:
            return False
        for index, record in enumerate(self.waypoints):
            if record.id != waypoint_id:
                continue
            record.failed_dir = True
            for abandoned in self.waypoints[index + 1:]:
                abandoned.failed = True
            return True
        return False

def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):  # CHW
        arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr[..., :3])

def _sample_history(
    images: list[Image.Image],
    every_k: int,
    max_frames: int,
) -> tuple[list[Image.Image], list[int]]:
    """
    Sample history with stride `every_k`, cap at `max_frames` (keep most recent).
    Returns (sampled_images, sampled_step_indices).
    """
    if not images:
        return [], []
    indices = list(range(0, len(images), every_k))
    if not indices or indices[-1] != len(images) - 1:
        indices.append(len(images) - 1)
    if len(indices) > max_frames:
        indices = indices[-max_frames:]
    return [images[i] for i in indices], indices

def _sample_waypoints(
    records: list[_WaypointRecord],
    max_frames: int,
    *,
    current_leg_frames: Optional[list[Image.Image]] = None,
    current_pose: Optional[np.ndarray] = None,
    layered_history: bool = False,
    history_wp_max: int = 8,
    backtrack_radius_m: float = 6.0,
    blank_image: Optional[Image.Image] = None,
) -> tuple[list[Image.Image], list[int], list[dict]]:
    """Build a fixed-image adaptation of source ``_build_layered_history``.

    Zone 2 keeps only a completed waypoint's arrival/after-turn pair.  Zone 1
    is deliberately attached only to the final retained waypoint and contains
    the live frames from that waypoint to the current decision.  RLinf needs a
    static image count per sample, so unused slots are explicit blank padding,
    never mislabeled as trajectory evidence.
    """
    per_record = 7
    blank = blank_image or Image.new("RGB", (32, 32), (127, 127, 127))
    live_frames = list(current_leg_frames or [])[:per_record - 2]
    if not records:
        if not live_frames or max_frames <= 0:
            return [], [], []
        images = live_frames + [blank] * (per_record - len(live_frames))
        return images, [-2], [{
            "id": -2,
            "kind": "initial",
            "continuous_count": len(live_frames),
            "padding_count": per_record - len(live_frames),
        }]
    chosen = records
    zone3_note = ""
    if layered_history and current_pose is not None and records:
        # Direct adaptation of source _build_layered_history's zone-2 lower
        # bound.  RLinf cache records are completed waypoints; current_pose is
        # the source function's current, not-yet-decided waypoint.
        reachable = [
            index for index, record in enumerate(records)
            if np.hypot(float(current_pose[0]) - record.pose_hab[0],
                        float(current_pose[1]) - record.pose_hab[1]) <= backtrack_radius_m
        ]
        lower = min(reachable) if reachable else len(records)
        lower = max(lower, len(records) - max(1, int(history_wp_max)))
        if lower > 0:
            zone3_note = (
                f"(Waypoints 0..{lower - 1} are older than the backtrack range; "
                "their route is summarized in the Progress Analysis text — no images shown for them.)"
            )
        chosen = records[lower:]
    if len(chosen) > max_frames:
        chosen = chosen[-max_frames:]

    images: list[Image.Image] = []
    infos = []
    final_index = len(chosen) - 1
    for record_index, r in enumerate(chosen):
        is_current_leg_anchor = record_index == final_index and not r.failed
        continuous_count = len(live_frames) if is_current_leg_anchor else 0
        infos.append({
            "id": r.id,
            "action": r.action,
            "target": r.target,
            "progress_analysis": r.progress_analysis,
            "continuous_count": continuous_count,
            "padding_count": per_record - 2 - continuous_count,
            "failed": bool(r.failed),
            "failed_dir": bool(r.failed_dir),
            "zone3_note": zone3_note if record_index == 0 else "",
        }
        )
    for record_index, r in enumerate(chosen):
        if r.failed:
            images.extend([blank] * per_record)
            continue
        record_images = [r.arrival_image, r.after_turn_image]
        if record_index == final_index:
            record_images.extend(live_frames)
        record_images.extend([blank] * (per_record - len(record_images)))
        if r.failed_dir:
            # Source suppresses only the branch-start after-turn image.  The
            # current-leg frames remain valid evidence after a later revisit.
            record_images[1] = blank
        images.extend(record_images)
    return images, [r.id for r in chosen], infos

def _draw_waypoint_overlay(
    img: Image.Image,
    bbox_2d: list[float],
    point_2d: list[float],
    waypoint_id: int,
) -> Image.Image:
    """Draw bbox/point/WP label on the chosen-direction view."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    x1, y1, x2, y2 = bbox_2d
    px1, py1 = x1 / 1000.0 * w, y1 / 1000.0 * h
    px2, py2 = x2 / 1000.0 * w, y2 / 1000.0 * h
    px, py = point_2d[0] / 1000.0 * w, point_2d[1] / 1000.0 * h
    draw.rectangle([px1, py1, px2, py2], outline=(0, 255, 0), width=3)
    r = 5
    draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 0, 0))
    draw.text((px1, max(0, py1 - 14)), f"WP{waypoint_id}", fill=(0, 255, 0))
    return out

class WaypointRuntimeMixin:
    """Behavior-preserving helpers mixed into ``QwenNavPolicy``."""

    def _build_prompt(
        self,
        instruction: str,
        history_imgs: list[Image.Image],
        history_step_indices: list[int],
        current_view: Image.Image,
        extra_views: Optional[list[Image.Image]] = None,
        pad_history_to: Optional[int] = None,
        stop_rejection_feedback: str = "",
        history_waypoint_ids: Optional[list[int]] = None,
        history_waypoint_infos: Optional[list[dict]] = None,
        available_backtrack_ids: Optional[list[int]] = None,
        allow_move_behind: Optional[bool] = None,
        blocked_directions: Optional[set[str]] = None,
    ) -> tuple[str, list[Image.Image]]:
        """
        Build Qwen-VL chat-template prompt + ordered image list.

        When `pad_history_to` is set (training mode), history is padded to
        that exact length with blank images so tensor shapes are consistent.

        Image order: history_imgs (chronological) + 4-dir current views (F/L/R/B).
        """
        if pad_history_to is not None:
            if self.is_waypoint_style:
                n_records = len(history_waypoint_infos or [])
                if n_records < pad_history_to:
                    pad_records = pad_history_to - n_records
                    pad_imgs = [self._blank_image] * (
                        pad_records * self._waypoint_images_per_record
                    )
                    history_imgs = pad_imgs + list(history_imgs)
                    pad_ids = [-1] * pad_records
                    history_waypoint_ids = pad_ids + list(history_waypoint_ids or [])
                    pad_infos = [
                        {
                            "id": -1,
                            "action": "",
                            "target": "",
                            "progress_analysis": "",
                            "continuous_count": self._waypoint_images_per_record - 2,
                        }
                        for _ in range(pad_records)
                    ]
                    history_waypoint_infos = pad_infos + list(history_waypoint_infos or [])
            elif len(history_imgs) < pad_history_to:
                pad_n = pad_history_to - len(history_imgs)
                # Prepend blank images and prepend dummy step indices
                history_imgs = [self._blank_image] * pad_n + list(history_imgs)
                history_step_indices = [-1] * pad_n + list(history_step_indices)

        if self.is_waypoint_style:
            user_text = build_waypoint_user_content_text(
                instruction=instruction,
                waypoint_ids=history_waypoint_ids or [],
                waypoint_infos=history_waypoint_infos,
                available_backtrack_ids=available_backtrack_ids,
                stop_rejection_feedback=stop_rejection_feedback,
                allow_move_behind=allow_move_behind,
                blocked_directions=blocked_directions,
                canonical=self.prompt_style == "canonical_v1",
            )
        elif self.prompt_style == "lavira_merged":
            user_text = build_merged_user_content_text(
                instruction=instruction,
                history_step_indices=history_step_indices,
                stop_rejection_feedback=stop_rejection_feedback,
            )
        else:
            user_text = build_user_content_text(
                instruction=instruction,
                history_step_indices=history_step_indices,
            )
            if stop_rejection_feedback:
                user_text = user_text + stop_rejection_feedback

        if self.use_4dir:
            if extra_views is not None and len(extra_views) == 3:
                current_views = [current_view] + list(extra_views)
            else:
                current_views = [current_view, current_view, current_view, current_view]
        else:
            current_views = [current_view]

        if self.is_waypoint_style and len(current_views) == 4:
            # Template order is FORWARD, LEFT, BEHIND, RIGHT.  The simulator
            # extras are stored as LEFT, RIGHT, BEHIND, so only this prompt
            # style reorders images to match the visible labels.
            prompt_current_views = [
                current_views[0],
                current_views[1],
                current_views[3],
                current_views[2],
            ]
        else:
            prompt_current_views = current_views

        ordered_images = list(history_imgs) + prompt_current_views

        n_expected = expected_image_count(len(history_imgs), has_4dir=self.use_4dir)
        if len(ordered_images) != n_expected:
            raise RuntimeError(
                f"image count mismatch: built {len(ordered_images)}, "
                f"prompt expects {n_expected}"
            )

        if self.is_waypoint_style:
            sys_prompt = (
                LAVIRA_CANONICAL_SYSTEM_PROMPT
                if self.prompt_style == "canonical_v1"
                else LAVIRA_WAYPOINT_SYSTEM_PROMPT
            )
        elif self.prompt_style == "lavira_merged":
            sys_prompt = LAVIRA_MERGED_SYSTEM_PROMPT
        else:
            sys_prompt = SYSTEM_PROMPT
        if self.is_waypoint_style:
            # VLM_example_ep1799 interleaves each history/current image at its
            # textual marker.  Feeding every image before one trailing text
            # block changes Qwen's visual grounding context even when counts
            # happen to match.
            text_parts = user_text.split("<image>")
            if len(text_parts) != len(ordered_images) + 1:
                raise RuntimeError(
                    "waypoint prompt image-marker mismatch: "
                    f"markers={len(text_parts) - 1} images={len(ordered_images)}"
                )
            user_content: list[dict] = []
            for part, image in zip(text_parts[:-1], ordered_images):
                if part:
                    user_content.append({"type": "text", "text": part})
                user_content.append({"type": "image", "image": image})
            if text_parts[-1]:
                user_content.append({"type": "text", "text": text_parts[-1]})
        else:
            user_content = (
                [{"type": "image", "image": img} for img in ordered_images]
                + [{"type": "text", "text": user_text}]
            )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        try:
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        # Keep Qwen's textual vision placeholders and the ordered PIL list in
        # lockstep.  The waypoint builder deliberately carries source-style
        # ``<image>`` markers in its text; a mismatch here otherwise reaches
        # vLLM as a malformed multimodal request and can look like a stalled
        # first generation.
        text_image_count = prompt_text.count("<|image_pad|>")
        if text_image_count == 0:
            text_image_count = prompt_text.count("<image>")
        if text_image_count != len(ordered_images):
            raise RuntimeError(
                f"chat-template image mismatch: text={text_image_count} "
                f"images={len(ordered_images)} style={self.prompt_style}"
            )
        # Resize all images to fixed resolution so pixel_values shape is
        # consistent between _init_padding_params and actual rollout/training.
        if not self.is_waypoint_style:
            ordered_images = [img.resize(self.image_size) for img in ordered_images]
        return prompt_text, ordered_images

    def _waypoint_record_images(self, record: _WaypointRecord) -> list[Image.Image]:
        """Return one fixed source-history record: arrival, turn, five frames."""
        blank = self._blank_image
        if record.failed:
            return [blank] * self._waypoint_images_per_record
        images = [record.arrival_image, record.after_turn_image]
        images.extend(record.continuous_frames[:self._waypoint_images_per_record - 2])
        images.extend([blank] * (self._waypoint_images_per_record - len(images)))
        if record.failed_dir:
            images[1] = blank
        return images

    def _build_backtrack_replan_prompt(
        self,
        *,
        instruction: str,
        cache: _HistoryCache,
        waypoint_id: int,
        current_view: Image.Image,
        extra_views: Optional[list[Image.Image]],
        blocked_directions: Optional[set[str]],
    ) -> tuple[str, list[Image.Image]]:
        """Adapt source ``replan_at_backtrack`` to Qwen's fixed image batch."""
        branch_index = next(
            (index for index, record in enumerate(cache.waypoints) if record.id == waypoint_id),
            None,
        )
        if branch_index is None:
            raise RuntimeError(f"backtrack replan waypoint {waypoint_id} is missing from cache")

        backtrack_record = cache.waypoints[branch_index]
        if getattr(self, "lavira_source_prompt_alignment", False):
            return self._build_source_backtrack_replan_prompt(
                instruction=instruction,
                cache=cache,
                branch_index=branch_index,
                backtrack_record=backtrack_record,
                current_view=current_view,
                extra_views=extra_views,
                blocked_directions=blocked_directions,
            )
        # The source replan prompt gets the *actual frames* after the
        # backtrack point, not merely later waypoint summaries.  Recover that
        # same failed path from RLinf's per-episode Genesis image history.
        failed_path = cache.history_images[backtrack_record.history_end_index:]
        failed_budget_records = 1 if failed_path and self.history_max_frames > 1 else 0
        history_budget_records = self.history_max_frames - failed_budget_records
        retained = cache.waypoints[
            max(0, branch_index - history_budget_records + 1):branch_index + 1
        ]
        # Source can emit variable image content.  Preserve its priority under
        # RLinf's fixed image-token budget: layered history first, then one
        # explicit chunk containing real abandoned-branch frames.
        budget_records = self.history_max_frames
        history_entries = [
            {"id": record.id, "image_count": self._waypoint_images_per_record}
            for record in retained
        ]
        failed_entries = []
        failed_images: list[Image.Image] = []
        if failed_budget_records:
            failed_images, _ = _sample_history(
                failed_path,
                every_k=max(1, len(failed_path) // self._waypoint_images_per_record),
                max_frames=self._waypoint_images_per_record,
            )
            failed_images.extend(
                [self._blank_image] * (self._waypoint_images_per_record - len(failed_images))
            )
            failed_entries.append({"id": "path", "image_count": len(failed_images)})
        history_images: list[Image.Image] = []
        for record in retained:
            history_images.extend(self._waypoint_record_images(record))
        history_images.extend(failed_images)
        pad_count = (budget_records - len(retained) - failed_budget_records) * self._waypoint_images_per_record
        if pad_count:
            history_images.extend([self._blank_image] * pad_count)
        previous_action = backtrack_record.action
        user_text = build_backtrack_replan_user_content_text(
            instruction=instruction,
            history_entries=history_entries,
            failed_entries=failed_entries,
            previous_action=previous_action,
            blocked_directions=blocked_directions,
            padding_image_count=pad_count,
        )
        if len(backtrack_record.anchor_views) == 4:
            # Source replan prompt: current views are the saved panorama at
            # the selected backtrack waypoint, ordered F/L/B/R.
            current = backtrack_record.anchor_views
        elif self.use_4dir and extra_views is not None and len(extra_views) == 3:
            # Simulator extras are L/R/B; source prompt is F/L/B/R.
            current = [current_view, extra_views[0], extra_views[2], extra_views[1]]
        else:
            current = [current_view] * 4
        ordered_images = history_images + current
        if user_text.count("<image>") != len(ordered_images):
            raise RuntimeError(
                f"replan image-token mismatch: text={user_text.count('<image>')} images={len(ordered_images)}"
            )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": LAVIRA_WAYPOINT_SYSTEM_PROMPT}]},
            {"role": "user", "content": (
                [{"type": "image", "image": image} for image in ordered_images]
                + [{"type": "text", "text": user_text}]
            )},
        ]
        try:
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
                add_vision_id=getattr(self, "add_vision_id", False),
            )
        except TypeError:
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return prompt_text, [image.resize(self.image_size) for image in ordered_images]

    def _build_source_backtrack_replan_prompt(
        self,
        *,
        instruction: str,
        cache: _HistoryCache,
        branch_index: int,
        backtrack_record: _WaypointRecord,
        current_view: Image.Image,
        extra_views: Optional[list[Image.Image]],
        blocked_directions: Optional[set[str]],
    ) -> tuple[str, list[Image.Image]]:
        """Port LHX replan_at_backtrack continuous-history message ordering."""
        retained = sorted(cache.waypoints[:branch_index + 1], key=lambda record: record.step)
        history_content: list[dict] = []
        failed_content: list[dict] = []
        last_leg_index = -2
        failed_header_added = False
        for index, image in enumerate(cache.history_images):
            image_step = cache.history_steps[index] if index < len(cache.history_steps) else index
            if image_step <= backtrack_record.step:
                leg_index = -1
                for candidate in range(len(retained) - 1):
                    if retained[candidate].step <= image_step < retained[candidate + 1].step:
                        leg_index = candidate
                        break
                if leg_index == -1 and retained and image_step >= retained[-1].step:
                    leg_index = len(retained) - 1
                if leg_index != last_leg_index and leg_index >= 0:
                    if leg_index < len(retained) - 1:
                        label = (
                            f"Waypoint {retained[leg_index].id} -> "
                            f"Waypoint {retained[leg_index + 1].id}: "
                        )
                    else:
                        label = (
                            f"Waypoint {retained[leg_index].id} -> Waypoint "
                            f"{retained[leg_index].id} (Backtrack Point): "
                        )
                    history_content.append({"type": "text", "text": label})
                    last_leg_index = leg_index
                history_content.append({"type": "image", "image": image})
            else:
                if not failed_header_added:
                    failed_content.append({
                        "type": "text",
                        "text": "Trajectory after Backtrack Point (Failed Path):",
                    })
                    failed_header_added = True
                failed_content.append({"type": "image", "image": image})

        if len(backtrack_record.anchor_views) == 4:
            current = backtrack_record.anchor_views
        elif self.use_4dir and extra_views is not None and len(extra_views) == 3:
            current = [current_view, extra_views[0], extra_views[2], extra_views[1]]
        else:
            current = [current_view] * 4

        blocked = set(blocked_directions or ())
        directions = [direction for direction in ("forward", "left", "right", "behind")
                      if direction not in blocked]
        if not directions:
            directions = ["behind"]
        descriptions = {
            "forward": "navigate to forward - continue straight ahead",
            "left": "navigate to left - turn left and go forward",
            "right": "navigate to right - turn right and go forward",
            "behind": "navigate to behind - turn around and go forward",
        }
        previous_action = backtrack_record.action.replace("navigate to ", "") or "unknown"
        tail = LAVIRA_BACKTRACK_REPLAN_FAILED.format(previous_action=previous_action)
        tail += LAVIRA_BACKTRACK_REPLAN_OUTPUT.format(
            available_actions="\n".join(
                f"   - {descriptions[direction]}" for direction in directions
            )
        )
        content: list[dict] = [{"type": "text", "text": "Navigation History:"}]
        content.extend(history_content)
        content.append({
            "type": "text",
            "text": "\nPrevious Trajectory (Path taken from here):",
        })
        content.extend(failed_content or [{"type": "text", "text": "None (Immediate failure)"}])
        content.append({
            "type": "text",
            "text": "\nCurrent 4-directional views at Backtrack Waypoint:",
        })
        for label, image in zip((
            "Current FORWARD view:", "View after turning LEFT:",
            "View after turning BEHIND:", "View after turning RIGHT:",
        ), current):
            content.append({"type": "text", "text": label})
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": f'Instruction: "{instruction}"\n{tail}'})
        messages = [{"role": "user", "content": content}]
        try:
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                add_vision_id=getattr(self, "add_vision_id", False),
            )
        except TypeError:
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        ordered_images = [item["image"] for item in content if item["type"] == "image"]
        return prompt_text, ordered_images

    def _waypoint_anchor_observation(
        self, record: _WaypointRecord
    ) -> Optional[LaviraObservation]:
        """Recover the source replan pose and four RGB-D views for one waypoint."""
        if len(record.anchor_views) != 4 or record.anchor_depth_by_direction is None:
            return None
        depth = np.asarray(record.anchor_depth_by_direction, dtype=np.float32)
        if depth.ndim != 3 or depth.shape[0] != 4:
            return None
        yaw = float(record.pose_hab[2])
        return LaviraObservation(
            rgb_by_direction=record.anchor_views,
            depth_by_direction=depth,
            yaw_by_direction=np.asarray(
                [yaw, yaw + math.pi / 2, yaw + math.pi, yaw - math.pi / 2],
                dtype=np.float32,
            ),
            hab_x=float(record.pose_hab[0]),
            hab_z=float(record.pose_hab[1]),
            hfov_deg=float(getattr(self.lavira_runtime_cfg, "hfov_deg", 79.0)),
        )

    def bbox_geom_valid(self, parsed) -> bool:
        """Depth-independent subset of the bbox_valid predicate (P1, hybrid-gated C).

        Checks: JSON parse ok, bbox_2d well-formed and ordered within 0-1000, area
        in a sane (non-degenerate, non-full-frame) range, and a recognised navigate
        direction. Depth-dependent checks (valid depth crop, projected goal in map,
        goal traversable) are deferred to P2/P3. Returns False → caller falls back to
        the discrete direction macro.
        """
        if not parsed.ok or parsed.bbox_2d is None:
            return False
        b = parsed.bbox_2d
        if len(b) != 4:
            return False
        x1, y1, x2, y2 = b
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            return False
        area = (x2 - x1) * (y2 - y1)
        if not (self._BBOX_AREA_MIN <= area <= self._BBOX_AREA_MAX):
            return False
        if parsed.raw_dir not in VALID_DIRECTIONS:
            return False
        return True

    def point_geom_valid(self, parsed) -> bool:
        """Return True when parsed point_2d is valid and lies inside bbox_2d."""
        if not self.bbox_geom_valid(parsed) or parsed.point_2d is None:
            return False
        if parsed.raw_dir not in ("navigate to forward", "navigate to left", "navigate to right"):
            return False
        if len(parsed.point_2d) != 2:
            return False
        x, y = parsed.point_2d
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            return False
        x1, y1, x2, y2 = parsed.bbox_2d
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _chosen_direction_view(
        raw_dir: Optional[str],
        current_views: list[Image.Image],
    ) -> Image.Image:
        """Return the current-view image matching a navigate action."""
        if raw_dir == "navigate to left" and len(current_views) > 1:
            return current_views[1]
        if raw_dir == "navigate to right" and len(current_views) > 2:
            return current_views[2]
        if raw_dir == "navigate to behind" and len(current_views) > 3:
            return current_views[3]
        return current_views[0]

    def _get_grounded_sam(
        self,
    ) -> GroundedSAMWaypointRefiner | GroundedSAMServicePool:
        if self._grounded_sam is None:
            if self.grounded_sam_cfg is None:
                raise RuntimeError("grounded_sam.enabled=true but no grounded_sam config was provided")
            self._grounded_sam = build_grounded_sam(self.grounded_sam_cfg)
        return self._grounded_sam

    def _segment_grounded_sam_jobs(self, jobs: list[dict]) -> list[dict]:
        if not jobs:
            return []
        started = time.perf_counter()
        try:
            refiner = self._get_grounded_sam()
            batch_fn = getattr(refiner, "segment_classes_batch", None)
            if callable(batch_fn):
                return batch_fn(jobs)
            # Preserve compatibility with legacy/custom single-job refiners.
            return [refiner.segment_classes(job) for job in jobs]
        finally:
            self._profile_add(
                "gsam_semantic_rpc",
                time.perf_counter() - started,
                items=len(jobs),
            )

    def _refine_grounded_sam_jobs(self, jobs: list[dict]):
        if not jobs:
            return []
        started = time.perf_counter()
        try:
            grounding_jobs = []
            for job in jobs:
                target, classes, source_lhx = self._grounding_request(
                    job.get("target"), bool(job.get("stair", False))
                )
                grounding_jobs.append({
                    **job,
                    "target": target,
                    "target_region": (
                        "any" if source_lhx else job.get("target_region", "any")
                    ),
                    "grounding_classes": classes,
                    "source_lhx": source_lhx,
                })
            refiner = self._get_grounded_sam()
            batch_fn = getattr(refiner, "refine_batch", None)
            if callable(batch_fn):
                return batch_fn(grounding_jobs)
            # Preserve compatibility with legacy/custom single-job refiners.
            return [
                refiner.refine(
                    job.get("image_rgb"),
                    job.get("target", ""),
                    job.get("target_region", "any"),
                    grounding_classes=job.get("grounding_classes"),
                    source_lhx=bool(job.get("source_lhx", False)),
                )
                for job in grounding_jobs
            ]
        finally:
            self._profile_add(
                "gsam_waypoint_rpc",
                time.perf_counter() - started,
                items=len(jobs),
            )

    @staticmethod
    def _source_lhx_grounding_classes(
        target_value: object, stair: bool
    ) -> list[str]:
        """Mirror ZS_Evaluator_mp._grounding_classes_from_target()."""
        target = str(target_value or "").strip().lower()
        if target in ("", "none", "null", "unknown", "unknown target"):
            return ["stairs"] if stair else []
        classes = [target]
        if stair or any(word in target for word in ("stair", "step")):
            classes.extend(["stairs", "stairway", "staircase", "steps"])
        return list(dict.fromkeys(value for value in classes if value))

    def _grounding_request(
        self, target_value: object, stair: bool
    ) -> tuple[str, list[str], bool]:
        if str(getattr(self, "prompt_style", "default")) == "lavira_waypoint":
            query_value = target_value
            if bool(
                getattr(self, "canonicalize_lavira_waypoint_query", False)
            ):
                mapping = map_target_for_grounding(target_value)
                query_value = mapping.dino_query
            classes = self._source_lhx_grounding_classes(query_value, stair)
            return (classes[0] if classes else "", classes, True)
        mapping = map_target_for_grounding(target_value)
        classes = [mapping.dino_query] if mapping.dino_query else []
        return mapping.dino_query, classes, False

    def _grounding_target_diag(self, parsed: ParsedAction) -> dict:
        query, classes, source_lhx = self._grounding_request(
            parsed.target, parsed.stair
        )
        if source_lhx:
            canonicalize_query = bool(
                getattr(self, "canonicalize_lavira_waypoint_query", False)
            )
            mapping = (
                map_target_for_grounding(parsed.target)
                if canonicalize_query else None
            )
            return {
                "target": parsed.target,
                "raw_target": parsed.raw_target,
                "mapped_target": (
                    mapping.canonical_target if mapping is not None else query
                ),
                "dino_query": query,
                "grounding_classes": classes,
                "target_mapping_status": (
                    mapping.status if mapping is not None else "source_raw"
                ),
                "target_geometry_kind": "source_lhx_query_only",
            }
        mapping = map_target_for_grounding(parsed.target)
        return {
            "target": parsed.target,
            "raw_target": parsed.raw_target,
            "mapped_target": mapping.canonical_target,
            "dino_query": mapping.dino_query,
            "grounding_classes": classes,
            "target_mapping_status": mapping.status,
            "target_geometry_kind": mapping.geometry_kind,
        }

    def _refine_waypoint_with_grounded_sam(
        self,
        parsed: ParsedAction,
        current_views: list[Image.Image],
        *,
        env_i: int = 0,
    ):
        """Use GroundingDINO/SAM to refine the parsed waypoint target if possible."""
        if not self.grounded_sam_enabled:
            return None
        if parsed.action_type != "NAVIGATE" or not str(parsed.target or "").strip():
            return None
        view = self._chosen_direction_view(parsed.raw_dir, current_views)
        image_rgb = np.asarray(view.convert("RGB"))
        target, classes, source_lhx = self._grounding_request(
            parsed.target, parsed.stair
        )
        if not classes:
            return GroundedSAMRefineResult(False, fallback_reason="empty_dino_query")
        refiner = self._get_grounded_sam()
        if isinstance(refiner, GroundedSAMServicePool):
            return refiner.refine(
                image_rgb,
                target,
                "any" if source_lhx else parsed.target_region,
                env_i=env_i,
                grounding_classes=classes,
                source_lhx=source_lhx,
            )
        return refiner.refine(
            image_rgb,
            target,
            "any" if source_lhx else parsed.target_region,
            grounding_classes=classes,
            source_lhx=source_lhx,
        )

    @staticmethod
    def _apply_grounding_geometry_filter(
        controller,
        observation: LaviraObservation,
        parsed: ParsedAction,
        sam_result,
    ):
        if target_geometry_kind(parsed.target) == "area":
            selected_area, area_candidates = controller.select_area_target(
                observation,
                parsed.raw_dir or "navigate to forward",
                stair=parsed.stair,
            )
            if selected_area is not None:
                return GroundedSAMRefineResult(
                    detected=True,
                    point_2d=list(selected_area["point_2d"]),
                    bbox_2d=None,
                    label=str(parsed.target),
                    confidence=0.0,
                    fallback_reason="area_geometry",
                    candidates=area_candidates,
                )
            # Area geometry is an improvement layer, not a new failure mode.
            # If it cannot find a strong free-space point, retain the original
            # DINO -> LHX bbox/backoff path and attach the area audit evidence.
            if sam_result is not None:
                prior_reason = str(sam_result.fallback_reason or "")
                sam_result.fallback_reason = (
                    f"area_no_valid_goal|{prior_reason}"
                    if prior_reason else "area_no_valid_goal"
                )
                sam_result.candidates = (
                    area_candidates + list(sam_result.candidates or [])
                )
            else:
                # Keep the source centre-bbox fallback reachable even when
                # DINO itself raised or returned no result for an area query.
                return GroundedSAMRefineResult(
                    detected=False,
                    label=str(parsed.target),
                    fallback_reason="area_no_valid_goal",
                    candidates=area_candidates,
                )

        if (
            sam_result is None
            or not sam_result.detected
            or not sam_result.candidates
        ):
            return sam_result
        selected, evaluated = controller.select_grounding_candidate(
            observation,
            parsed.raw_dir or "navigate to forward",
            sam_result.candidates,
            stair=parsed.stair,
            target_region=parsed.target_region,
        )
        sam_result.candidates = evaluated
        if selected is None:
            prior_reason = str(sam_result.fallback_reason or "")
            sam_result.detected = False
            sam_result.point_2d = None
            sam_result.bbox_2d = None
            sam_result.confidence = 0.0
            sam_result.fallback_reason = (
                f"{prior_reason}|no_geometrically_valid_bbox"
                if prior_reason else "no_geometrically_valid_bbox"
            )
            return sam_result
        sam_result.point_2d = None
        sam_result.bbox_2d = list(selected["bbox_2d"])
        sam_result.label = str(selected.get("label", sam_result.label))
        sam_result.confidence = float(selected.get("confidence", 0.0))
        if selected.get("selection_mode") == "lhx_passthrough":
            prior_reason = str(sam_result.fallback_reason or "")
            sam_result.fallback_reason = (
                f"{prior_reason}|lhx_dino_passthrough"
                if prior_reason else "lhx_dino_passthrough"
            )
        return sam_result

    def _save_waypoint_record(
        self,
        cache: _HistoryCache,
        parsed: ParsedAction,
        pose_row: np.ndarray,
        current_views: list[Image.Image],
        runtime_observation: Optional[LaviraObservation] = None,
        waypoint_id: Optional[int] = None,
    ) -> None:
        """Update the panorama-created waypoint with the accepted decision."""
        if parsed.bbox_2d is None:
            return
        # Source LaViRA localises a navigation target at the DINO bbox bottom
        # centre. Keep that derived point only in the history record/overlay;
        # projection still receives the original bbox-only detection.
        record_point = (
            list(parsed.point_2d)
            if parsed.point_2d is not None
            else [
                (float(parsed.bbox_2d[0]) + float(parsed.bbox_2d[2])) / 2.0,
                float(parsed.bbox_2d[3]),
            ]
        )
        existing = self._find_waypoint(cache, waypoint_id)
        if waypoint_id is None:
            waypoint_id = cache.next_waypoint_id
        wid = int(waypoint_id)
        cache.next_waypoint_id = max(cache.next_waypoint_id, wid + 1)
        arrival_img = current_views[0]
        chosen_img = self._chosen_direction_view(parsed.raw_dir, current_views)
        overlay = _draw_waypoint_overlay(chosen_img, parsed.bbox_2d, record_point, wid)
        segment = cache.history_images[cache.last_waypoint_history_len:]
        if not segment:
            segment = [arrival_img]
        cont_imgs, _cont_idx = _sample_history(
            segment,
            every_k=max(1, len(segment) // 5),
            max_frames=5,
        )
        if existing is None:
            existing = _WaypointRecord(
                waypoint_id=wid,
                pose_hab=(float(pose_row[0]), float(pose_row[1]), float(pose_row[2])),
                arrival_image=arrival_img.copy(),
                after_turn_image=overlay,
                continuous_frames=[img.copy() for img in cont_imgs],
                action=parsed.raw_dir or "",
                bbox_2d=list(parsed.bbox_2d),
                point_2d=record_point,
                target=parsed.target,
                progress_analysis=parsed.progress,
                history_end_index=len(cache.history_images),
                step=cache.step_count,
                anchor_views=(
                    runtime_observation.rgb_by_direction
                    if runtime_observation is not None else None
                ),
                anchor_depth_by_direction=(
                    runtime_observation.depth_by_direction
                    if runtime_observation is not None else None
                ),
            )
            cache.waypoints.append(existing)
        else:
            existing.after_turn_image = overlay
            existing.continuous_frames = [img.copy() for img in cont_imgs]
            existing.action = parsed.raw_dir or ""
            existing.bbox_2d = list(parsed.bbox_2d)
            existing.point_2d = record_point
            existing.target = parsed.target
            existing.progress_analysis = parsed.progress
            existing.history_end_index = len(cache.history_images)
        cache.last_waypoint_history_len = len(cache.history_images)

    def _begin_decision_waypoint_record(
        self,
        cache: _HistoryCache,
        controller: LaviraNavigationController,
        observation: LaviraObservation,
        pose_row: np.ndarray,
        current_views: list[Image.Image],
    ) -> int:
        """Mirror ZS_Evaluator_mp: append the waypoint after panorama, before VLM."""
        waypoint_id = controller.begin_decision_waypoint(
            observation, cache.next_waypoint_id
        )
        if self._find_waypoint(cache, waypoint_id) is not None:
            return waypoint_id
        segment = cache.history_images[cache.last_waypoint_history_len:]
        if not segment:
            segment = [current_views[0]]
        cont_imgs, _ = _sample_history(
            segment,
            every_k=max(1, len(segment) // 5),
            max_frames=5,
        )
        record = _WaypointRecord(
            waypoint_id=waypoint_id,
            pose_hab=(float(pose_row[0]), float(pose_row[1]), float(pose_row[2])),
            arrival_image=current_views[0].copy(),
            after_turn_image=current_views[0].copy(),
            continuous_frames=[image.copy() for image in cont_imgs],
            action="",
            bbox_2d=[],
            point_2d=[],
            target="",
            progress_analysis="",
            history_end_index=len(cache.history_images),
            step=cache.step_count,
            anchor_views=observation.rgb_by_direction,
            anchor_depth_by_direction=observation.depth_by_direction,
        )
        cache.waypoints.append(record)
        cache.next_waypoint_id = max(cache.next_waypoint_id, waypoint_id + 1)
        cache.last_waypoint_history_len = len(cache.history_images)
        return waypoint_id

    @staticmethod
    def _drop_waypoint_records(cache: _HistoryCache, waypoint_ids: list[int]) -> None:
        if not waypoint_ids:
            return
        removed = set(waypoint_ids)
        cache.waypoints[:] = [record for record in cache.waypoints if record.id not in removed]

    def _find_waypoint(self, cache: _HistoryCache, waypoint_id: Optional[int]) -> Optional[_WaypointRecord]:
        if waypoint_id is None:
            return None
        for rec in cache.waypoints:
            if rec.id == waypoint_id:
                return rec
        return None

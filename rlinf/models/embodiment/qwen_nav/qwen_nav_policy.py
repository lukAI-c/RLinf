"""
QwenNavPolicy — Qwen-VL-based navigation policy for GenArk RFT (Plan B).

Design: lavira-style single-turn prompt rebuild.
  - Per-env history image buffer (sampled every K steps, capped at M frames)
  - Each step: rebuild full prompt (system + history + 4-dir + instruction + JSON schema)
  - Generate JSON → parse → map to GenArk discrete actions [0=stop, 1=fwd, 2=L, 3=R]
  - Multi-step macro actions (e.g. "navigate to left" → [turn_left, forward])
    buffered into pending_actions (eval mode only).

Training support (Step 3b):
  - predict_action_batch returns forward_inputs + prev_logprobs when
    cfg.collect_forward_inputs=True (set in genark_grpo_qwen.yaml).
  - History is padded to history_max_frames blank images for fixed tensor shapes.
  - default_forward() runs teacher-forcing, returns {"logprobs", "entropy"}.

See docs/genark_rft_caveats.md for known design caveats (C1-C11).
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .prompts import (
    SYSTEM_PROMPT, STOP_CHECK_SYSTEM_PROMPT, TERMINATION_SHADOW_SYSTEM_PROMPT,
    LAVIRA_MERGED_SYSTEM_PROMPT, LAVIRA_WAYPOINT_SYSTEM_PROMPT,
    LAVIRA_CANONICAL_SYSTEM_PROMPT,
    build_user_content_text, build_stop_check_text, build_source_stop_check_parts,
    build_termination_shadow_text,
    build_merged_user_content_text, build_waypoint_user_content_text,
    USER_STOP_REJECTED,
)
from .action_parser import (
    ParsedAction, ParsedStopCheck,
    parse_lavira_json, parse_lavira_merged_json, parse_lavira_waypoint_json,
    parse_lavira_canonical_waypoint_json,
    parse_stop_check_json,
    ACTION_STOP, ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
    ACTION_PARSE_FAIL, ACTION_SCHEMA_OK_PARSE_FAIL, ACTION_NOOP,
    ACTION_PANORAMA_SCAN,
    ACTION_PARSE_OK_HAS_BBOX_BASE,
    ACTION_STRUCTURED_PARSE_FAIL_BASE,
    REWARD_JSON_VALID, REWARD_REQUIRED_FIELDS, REWARD_FIELD_FORMAT,
    REWARD_GEOMETRY_VALID,
    REWARD_LENGTH_OK,
)
from .grounded_sam import (
    GroundedSAMServicePool,
    GroundedSAMWaypointRefiner,
)
from .canonical_targets import target_geometry_kind
from .lavira_runtime import LaviraNavigationController, LaviraObservation
from .policy_diagnostics import PolicyDiagnosticsMixin
from .waypoint_runtime import (
    WaypointRuntimeMixin,
    _HistoryCache,
    _WaypointRecord,
    _sample_history,
    _sample_waypoints,
    _to_pil,
)


# ---------------------------------------------------------------------------
# QwenNavPolicy
# ---------------------------------------------------------------------------

class QwenNavPolicy(
    WaypointRuntimeMixin, PolicyDiagnosticsMixin, nn.Module, BasePolicy
):
    @property
    def is_waypoint_style(self) -> bool:
        return str(getattr(self, "prompt_style", "default")) in {
            "lavira_waypoint", "canonical_v1"
        }

    """
    Lavira-style single-turn policy wrapping a Qwen-VL HuggingFace model.

    Required cfg fields:
      model_path           : str       — local path to Qwen-VL checkpoint
      max_new_tokens       : int       — generation cap (default 512)
      temperature          : float
      do_sample            : bool
      history_every_k      : int       — sample one history frame every K steps
      history_max_frames   : int       — cap history image count (default 8)
      use_4dir             : bool      — pass 4 view labels in prompt
      precision            : "bf16"|"fp16"|"fp32"
      trust_remote_code    : bool
      attn_implementation  : "eager"|"sdpa"|"flash_attention_2"
      collect_forward_inputs: bool     — True in training mode (default False)
    """

    _per_env_cache: dict[int, _HistoryCache]

    def __init__(self, cfg: DictConfig):
        nn.Module.__init__(self)
        BasePolicy.__init__(self)

        self.cfg = cfg
        self.model_path = str(cfg.model_path)
        self.max_new_tokens = int(getattr(cfg, "max_new_tokens", 512))
        self.temperature = float(getattr(cfg, "temperature", 0.5))
        self.do_sample = bool(getattr(cfg, "do_sample", True))
        self.history_every_k = int(getattr(cfg, "history_every_k", 4))
        self.history_max_frames = int(getattr(cfg, "history_max_frames", 8))
        # Eval-only LHX lifecycle compatibility for STOP verification, JSON
        # retry, backtrack replanning, and DINO fallback. The main prompt keeps
        # the shared seven-field VLM_example_ep1799 contract.
        self.lavira_source_prompt_alignment = bool(
            getattr(cfg, "lavira_source_prompt_alignment", False)
        )
        self.use_4dir = bool(getattr(cfg, "use_4dir", True))
        self.add_vision_id = bool(getattr(cfg, "add_vision_id", False))
        self.json_retry_enabled = bool(getattr(cfg, "json_retry_enabled", False))
        self.json_retry_max_retries = int(getattr(cfg, "json_retry_max_retries", 5))
        self.stop_double_check_enabled = bool(
            getattr(cfg, "stop_double_check_enabled", True)
        )
        self.stop_double_check_max_failures = int(
            getattr(cfg, "stop_double_check_max_failures", 3)
        )
        self.stop_double_check_distance_threshold = float(
            getattr(cfg, "stop_double_check_distance_threshold", 1.0)
        )
        shadow_cfg = getattr(cfg, "termination_shadow", None)
        self.termination_shadow_enabled = bool(
            getattr(shadow_cfg, "enabled", False)
            if shadow_cfg is not None else False
        )
        self.history_action_aware = bool(getattr(cfg, "history_action_aware", False))
        self.history_forward_interval = int(getattr(cfg, "history_forward_interval", 5))
        self.history_turn_interval = int(getattr(cfg, "history_turn_interval", 2))
        self.chunk_size = int(getattr(cfg, "chunk_size", 4))
        # Training-mode flag: when True, compute prev_logprobs + forward_inputs
        # and disable macro-action buffering (pending_actions).
        self.collect_forward_inputs = bool(getattr(cfg, "collect_forward_inputs", False))
        # Exact variable-image source prompts remain eval-only. Training keeps
        # fixed image shapes, while the two backtrack responses occupy separate
        # policy ticks so each retains its own on-policy token/logprob record.
        if self.collect_forward_inputs:
            self.lavira_source_prompt_alignment = False
        # Per-call action distribution logger. Flushes a one-line summary every
        # `_action_stats_flush_every` policy-inference batches that contained at
        # least one decision env. Set env QWEN_NAV_ACTION_STATS_EVERY=0 to disable.
        self._action_stats_flush_every = int(os.environ.get("QWEN_NAV_ACTION_STATS_EVERY", "10"))
        self._action_stats_counter = 0
        self._action_stats = {
            "forward": 0, "left": 0, "right": 0, "behind": 0,
            "backtrack": 0, "stop": 0, "parse_fail": 0, "total": 0,
            # P1 hybrid-gated controller: per-decision bbox gate outcomes.
            "bbox_valid": 0, "bbox_fallback": 0,
            "point_valid": 0, "backtrack_valid": 0,
            "json_valid": 0, "struct_ok": 0, "field_format_ok": 0,
            "geometry_ok": 0, "length_ok": 0,
            "grounded_sam_used": 0, "grounded_sam_detected": 0,
        }
        profile_cfg = getattr(cfg, "rollout_profile", None)
        self._rollout_profile_enabled = bool(
            getattr(profile_cfg, "enabled", False)
            if profile_cfg is not None else False
        )
        self._rollout_profile_flush_every = max(
            1,
            int(
                getattr(profile_cfg, "flush_every", 25)
                if profile_cfg is not None else 25
            ),
        )
        self._rollout_profile_batches = 0
        self._rollout_profile_stats: dict[str, dict[str, float | int]] = {}
        self._parse_fail_debug_limit = int(os.environ.get("QWEN_NAV_PARSE_FAIL_DEBUG", "3"))
        self._parse_fail_debug_printed = 0
        # When True, skip teacher-forcing logprobs at rollout time.
        # Safe only when actor.recompute_prev_logprobs=True; actor will overwrite
        # prev_logprobs with its own eval forward before PPO training.
        self.skip_rollout_logprobs = bool(getattr(cfg, "skip_rollout_logprobs", False))
        # Prompt/schema style: "default" uses 6-field schema; "lavira_merged" uses
        # 9-field merged LA+VA schema. Gated so running baseline is unaffected.
        self.prompt_style = str(getattr(cfg, "prompt_style", "default"))
        # Counterfactual foresight is an offline/shadow research path. Keep
        # the config visible for explicit callers, but do not import its
        # modules or alter action selection unless a future gated adapter
        # invokes it directly.
        self.cf_foresight_cfg = getattr(cfg, "cf_foresight", None)
        self._wm_state_collector = None
        _collection_cfg = getattr(cfg, "data_collection", None)
        if _collection_cfg is not None and bool(
            getattr(_collection_cfg, "enabled", False)
        ):
            from .wm_state_collector import OnPolicyStateCollector
            self._wm_state_collector = OnPolicyStateCollector(_collection_cfg)
        self.lavira_runtime_cfg = getattr(cfg, "lavira_runtime", None)
        self.lavira_runtime_enabled = bool(
            getattr(self.lavira_runtime_cfg, "enabled", False)
            if self.lavira_runtime_cfg is not None else False
        )
        self.lavira_layered_history = bool(
            getattr(self.lavira_runtime_cfg, "layered_history", True)
            if self.lavira_runtime_cfg is not None else False
        )
        self.lavira_history_wp_max = int(
            getattr(self.lavira_runtime_cfg, "history_wp_max", 8)
            if self.lavira_runtime_cfg is not None else 8
        )
        self.lavira_backtrack_radius_m = float(
            getattr(self.lavira_runtime_cfg, "layered_backtrack_radius_m", 6.0)
            if self.lavira_runtime_cfg is not None else 6.0
        )
        self.lavira_backtrack_second_chance = bool(
            getattr(self.lavira_runtime_cfg, "backtrack_second_chance", True)
            if self.lavira_runtime_cfg is not None else False
        )
        self.grounded_sam_cfg = getattr(cfg, "grounded_sam", None)
        self.grounded_sam_enabled = bool(
            getattr(self.grounded_sam_cfg, "enabled", False)
            if self.grounded_sam_cfg is not None
            else False
        )
        self.canonicalize_lavira_waypoint_query = bool(
            getattr(
                self.grounded_sam_cfg,
                "canonicalize_lavira_waypoint_query",
                False,
            )
            if self.grounded_sam_cfg is not None
            else False
        )
        self._grounded_sam: Optional[
            GroundedSAMWaypointRefiner | GroundedSAMServicePool
        ] = None
        # Opt-in rollout diagnostics. This is deliberately independent from
        # the controller and never changes a policy action.
        self.grounded_sam_visualize = bool(
            getattr(self.grounded_sam_cfg, "visualize", False)
            if self.grounded_sam_cfg is not None
            else False
        )
        self.grounded_sam_visualize_every = int(
            getattr(self.grounded_sam_cfg, "visualize_every", 0)
            if self.grounded_sam_cfg is not None else 0
        )
        debug_dir = str(
            getattr(self.grounded_sam_cfg, "visualization_dir", "")
            if self.grounded_sam_cfg is not None
            else ""
        ).strip()
        self._grounded_sam_visualization_dir = Path(debug_dir) if debug_dir else None
        self._grounded_sam_episode_counts: dict[int, int] = {}
        # Opt-in top-down map diagnostics. This reads GT only to draw an
        # overlay; it is intentionally outside the policy/planner/reward path.
        self._lavira_map_visualization_cfg = getattr(
            self.lavira_runtime_cfg, "map_visualization", None
        )
        self.lavira_map_visualize = bool(
            getattr(self._lavira_map_visualization_cfg, "enabled", False)
            if self._lavira_map_visualization_cfg is not None else False
        )
        map_debug_dir = str(
            getattr(self._lavira_map_visualization_cfg, "output_dir", "")
            if self._lavira_map_visualization_cfg is not None else ""
        ).strip()
        self._lavira_map_visualization_dir = Path(map_debug_dir) if map_debug_dir else None
        fmm_debug_dir = str(
            getattr(self._lavira_map_visualization_cfg, "fmm_fields_output_dir", "")
            if self._lavira_map_visualization_cfg is not None else ""
        ).strip()
        self._lavira_fmm_results_dir = (
            Path(fmm_debug_dir)
            if fmm_debug_dir
            else (
                self._lavira_map_visualization_dir.parent
                if self._lavira_map_visualization_dir is not None else None
            )
        )
        self._lavira_map_save_raw_every = int(
            getattr(self._lavira_map_visualization_cfg, "save_raw_every", 0)
            if self._lavira_map_visualization_cfg is not None else 0
        )
        self._lavira_map_save_every = int(
            getattr(self._lavira_map_visualization_cfg, "save_every", 0)
            if self._lavira_map_visualization_cfg is not None else 0
        )
        self._lavira_map_gt_reference_paths = self._load_lavira_map_reference_paths()
        configured_map_episode_id = str(
            getattr(self._lavira_map_visualization_cfg, "episode_id", "")
            if self._lavira_map_visualization_cfg is not None else ""
        ).strip()
        self._lavira_map_default_gt_reference_path = (
            self._lavira_map_gt_reference_paths.get(configured_map_episode_id)
            if configured_map_episode_id and configured_map_episode_id != "null"
            else None
        )
        self._lavira_map_episode_ids_by_env: dict[int, str] = {}
        self._lavira_map_episode_counts: dict[int, int] = {}
        # Fixed image resolution fed to the VLM processor.
        # All images (env renders + history + blank pads) are resized to (W, H)
        # before tokenisation so that pixel_values shape is consistent across
        # init/rollout/training.  Must match blank images used in _init_padding_params.
        _img_sz = getattr(cfg, "image_size", [448, 448])
        self.image_size: tuple[int, int] = (int(_img_sz[0]), int(_img_sz[1]))  # (W, H)
        _profile_img_sz = getattr(
            cfg, "forward_input_profile_image_size", [640, 480]
        )
        self.forward_input_profile_image_size: tuple[int, int] = (
            int(_profile_img_sz[0]),
            int(_profile_img_sz[1]),
        )
        self.min_pixels = getattr(cfg, "min_pixels", None)
        self.max_pixels = getattr(cfg, "max_pixels", None)
        self.min_pixels = None if self.min_pixels is None else int(self.min_pixels)
        self.max_pixels = None if self.max_pixels is None else int(self.max_pixels)
        # ``predict_action_batch`` builds fixed LaViRA history before the
        # processor-derived padding parameters exist.  Eval-only workers do
        # not take the actor's initialization path, so provide the same neutral
        # image up front and let ``_init_padding_params`` replace it later.
        self._blank_image = Image.new("RGB", self.image_size, (127, 127, 127))

        precision = str(getattr(cfg, "precision", "bf16")).lower()
        self._dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(precision, torch.bfloat16)

        self._trust_remote = bool(getattr(cfg, "trust_remote_code", True))
        self._attn_impl = str(getattr(cfg, "attn_implementation", "sdpa"))

        self.model = None
        self.processor = None
        self._per_env_cache = {}
        self._lavira_runtime: dict[int, LaviraNavigationController] = {}

        # _no_split_modules: populated after model loads so FSDP wraps each
        # transformer block individually instead of flattening the whole model.
        # Without this, inner modules lose their parameters and self.device fails.
        self._no_split_modules = []

        # Padding params — auto-detected on first call to predict_action_batch
        self._padding_initialized = False
        self._prompt_len: int = 0
        self._max_seq_len: int = 0
        self._total_patches: int = 0
        self._pixel_value_dim: int = 0
        # Strict LaViRA waypoint history uses 7 images per waypoint:
        # arrival + after-turn + 5 continuous frames.
        self._waypoint_images_per_record: int = 7
        self._n_images_fixed: int = 0  # fixed prompt image count
        self._patches_per_image: int = 0

        # Pluggable generation backend.  When None, uses HF model.generate (default).
        # Set to a callable by VLLMMultiStepEmbodiedWorker before rollout starts:
        #   fn(prompts: list[str], image_lists: list[list[Image]]) -> (list[str], list[Tensor])
        # where the returned tensors are (1, resp_len) int64 response token ids.
        self._vllm_generate_fn = None

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        import transformers
        from transformers import AutoProcessor
        AutoModelVL = getattr(transformers, "AutoModelForImageTextToText", None) \
                      or getattr(transformers, "AutoModelForVision2Seq", None)
        if AutoModelVL is None:
            raise ImportError(
                "Neither AutoModelForImageTextToText nor AutoModelForVision2Seq "
                "is available in this transformers version."
            )

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"QwenNavPolicy: model_path does not exist: {self.model_path}"
            )

        print(f"[QwenNavPolicy] loading {self.model_path} (dtype={self._dtype})", flush=True)
        processor_kwargs = {"trust_remote_code": self._trust_remote}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, **processor_kwargs
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
            true_ids = self.processor.tokenizer.encode(
                " true", add_special_tokens=False
            )
            false_ids = self.processor.tokenizer.encode(
                " false", add_special_tokens=False
            )
            if self.termination_shadow_enabled and (
                len(true_ids) != 1 or len(false_ids) != 1
            ):
                raise ValueError(
                    "termination shadow requires single-token ' true'/' false' "
                    f"literals, got {true_ids=} {false_ids=}"
                )
            self._termination_shadow_true_token_id = (
                int(true_ids[0]) if len(true_ids) == 1 else -1
            )
            self._termination_shadow_false_token_id = (
                int(false_ids[0]) if len(false_ids) == 1 else -1
            )
        self.model = AutoModelVL.from_pretrained(
            self.model_path,
            torch_dtype=self._dtype,
            trust_remote_code=self._trust_remote,
            attn_implementation=self._attn_impl,
        )
        self.model.eval()
        # Parameters stay trainable (requires_grad=True) by default so the actor
        # optimizer can update them.  Inference contexts (@torch.inference_mode)
        # handle no-gradient during rollout.

        # Tell FSDP which layers to wrap individually so each transformer block
        # keeps its own parameters and self.device works inside the forward pass.
        if hasattr(self.model, "_no_split_modules"):
            self._no_split_modules = list(self.model._no_split_modules)

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @property
    def _model_device(self) -> torch.device:
        """Return the device holding the model weights.

        Works in both normal (self.model.parameters()) and FSDP-wrapped
        (self.parameters() via the FSDP FlatParameter) contexts.
        """
        # Try inner HF model first (works for rollout worker, non-FSDP)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            pass
        # Fallback: top-level parameters through FSDP wrapper
        try:
            return next(self.parameters()).device
        except StopIteration:
            pass
        # Last resort: LOCAL_RANK env var (always valid in Ray workers)
        import os
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))

    # ------------------------------------------------------------------
    # Padding parameter auto-detection
    # ------------------------------------------------------------------

    def _init_padding_params(self):
        """
        Run a dummy forward through the processor to measure fixed tensor shapes.
        Called once on first predict_action_batch call.
        Uses history_max_frames blank images + n_current_views current images,
        so all steps have identical image counts → stable pixel_values shapes.
        """
        n_current = 4 if self.use_4dir else 1
        if self.is_waypoint_style:
            self._n_images_fixed = (
                self.history_max_frames * self._waypoint_images_per_record
                + n_current
            )
        else:
            self._n_images_fixed = self.history_max_frames + n_current

        blank = Image.new("RGB", self.image_size, (127, 127, 127))
        # Source-aligned waypoint prompts keep the camera aspect ratio.  Size
        # replay tensors against that real input, while retaining the existing
        # square grey image for neutral history padding.
        profile_blank = Image.new(
            "RGB", self.forward_input_profile_image_size, (127, 127, 127)
        )
        dummy_images = [profile_blank] * self._n_images_fixed

        # Build fake step indices for history
        dummy_hist_idx = list(range(self.history_max_frames))
        if self.is_waypoint_style:
            # Use long ids so dynamic backtrack labels do not exceed the
            # detected prompt length once real waypoint ids grow.
            dummy_hist_idx = [999_000 + i for i in range(self.history_max_frames)]
            dummy_waypoint_infos = [
                {
                    "id": wid,
                    "action": "navigate to forward",
                    "target": "hallway entrance",
                    "progress_analysis": "Moved through previous waypoints toward the goal.",
                    "continuous_count": self._waypoint_images_per_record - 2,
                }
                for wid in dummy_hist_idx
            ]
            user_text = build_waypoint_user_content_text(
                instruction="go to the elevator",
                waypoint_ids=dummy_hist_idx,
                waypoint_infos=dummy_waypoint_infos,
                canonical=self.prompt_style == "canonical_v1",
            )
            sys_prompt = (
                LAVIRA_CANONICAL_SYSTEM_PROMPT
                if self.prompt_style == "canonical_v1"
                else LAVIRA_WAYPOINT_SYSTEM_PROMPT
            )
        elif self.prompt_style == "lavira_merged":
            user_text = build_merged_user_content_text(
                instruction="go to the elevator",
                history_step_indices=dummy_hist_idx,
            )
            sys_prompt = LAVIRA_MERGED_SYSTEM_PROMPT
        else:
            user_text = build_user_content_text(
                instruction="go to the elevator",
                history_step_indices=dummy_hist_idx,
            )
            sys_prompt = SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in dummy_images]
                    + [{"type": "text", "text": user_text}]
                ),
            },
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

        inputs = self.processor(
            text=[prompt_text],
            images=[dummy_images],
            padding=True,
            return_tensors="pt",
        )

        self._prompt_len = inputs["input_ids"].shape[1]
        self._max_seq_len = self._prompt_len + self.max_new_tokens

        pv = inputs["pixel_values"]
        self._total_patches = pv.shape[0]        # total patches for max images
        self._pixel_value_dim = pv.shape[1] if pv.ndim == 2 else pv.shape[-1]

        if "image_grid_thw" in inputs:
            grid = inputs["image_grid_thw"]      # (n_images, 3)
            self._patches_per_image = (
                grid[0][0] * grid[0][1] * grid[0][2]
            ).item()
        else:
            self._patches_per_image = self._total_patches // self._n_images_fixed

        self._blank_image = blank  # image_size resolution blank used for history padding
        self._padding_initialized = True

        print(
            f"[QwenNavPolicy] padding params: prompt_len={self._prompt_len}, "
            f"max_seq_len={self._max_seq_len}, "
            f"n_images={self._n_images_fixed}, "
            f"patches_per_image={self._patches_per_image}, "
            f"pixel_value_dim={self._pixel_value_dim}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Per-env state mgmt
    # ------------------------------------------------------------------

    def reset_env_cache(self, env_indices: list[int]):
        for i in env_indices:
            cache = self._per_env_cache.setdefault(i, _HistoryCache())
            self._grounded_sam_episode_counts[i] = (
                self._grounded_sam_episode_counts.get(i, 0) + 1
            )
            self._lavira_map_episode_counts[i] = (
                self._lavira_map_episode_counts.get(i, 0) + 1
            )
            cache.reset()
            if self.lavira_runtime_enabled:
                controller = self._lavira_runtime.setdefault(
                    i, LaviraNavigationController(self.lavira_runtime_cfg)
                )
                controller.reset()

    def export_cf_policy_snapshot(self, env_i: int) -> dict:
        """Export the research-only policy cache for counterfactual branches.

        This is intentionally a lazy import and a read-only helper.  The
        normal rollout path never calls it, so enabling this method cannot
        change prompt construction, action selection, or RLOO behaviour.
        """
        from rlinf.research.cf_foresight.policy_snapshot import (
            export_policy_episode_snapshot,
        )

        return export_policy_episode_snapshot(self, int(env_i))

    def restore_cf_policy_snapshot(self, env_i: int, snapshot: dict) -> None:
        """Restore a research snapshot into one policy cache explicitly."""
        from rlinf.research.cf_foresight.policy_snapshot import (
            restore_policy_episode_snapshot,
        )

        restore_policy_episode_snapshot(self, snapshot, int(env_i))

    def _get_cache(self, env_id: int) -> _HistoryCache:
        return self._per_env_cache.setdefault(env_id, _HistoryCache())

    def _get_lavira_runtime(self, env_id: int) -> LaviraNavigationController:
        controller = self._lavira_runtime.setdefault(
            env_id, LaviraNavigationController(self.lavira_runtime_cfg)
        )
        return controller









    # ------------------------------------------------------------------
    # Prompt building (per env)
    # ------------------------------------------------------------------






    # ------------------------------------------------------------------
    # Batched generate
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _batch_generate(
        self,
        prompts: list[str],
        image_lists: list[list[Image.Image]],
    ) -> tuple[list[str], list[torch.Tensor]]:
        """
        One processor call → one model.generate per chunk.
        Returns (decoded_text_list, generated_ids_list).
        generated_ids_list[i]: (1, resp_len) int64 — response token ids per sample.

        When _vllm_generate_fn is set (injected by VLLMMultiStepEmbodiedWorker),
        delegates to vLLM and skips the HF model.generate path entirely.
        """
        if self._vllm_generate_fn is not None:
            return self._vllm_generate_fn(prompts, image_lists)

        decoded: list[str] = []
        gen_ids_list: list[torch.Tensor] = []

        for start in range(0, len(prompts), self.chunk_size):
            sub_prompts = prompts[start: start + self.chunk_size]
            sub_images = image_lists[start: start + self.chunk_size]

            inputs = self.processor(
                text=sub_prompts,
                images=sub_images,
                padding=True,
                return_tensors="pt",
            ).to(self._model_device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)

            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
            )

            input_len = inputs["input_ids"].shape[1]
            new_tok = gen_out[:, input_len:]        # (chunk, resp_len)

            batch_decoded = self.processor.batch_decode(
                new_tok, skip_special_tokens=True,
            )
            decoded.extend(batch_decoded)
            for i in range(new_tok.shape[0]):
                gen_ids_list.append(new_tok[i: i + 1].cpu())  # (1, resp_len)

        return decoded, gen_ids_list

    def _run_stop_check_batch(
        self,
        env_ids: list[int],
        instructions: list[str],
        current_views_per_env: dict[int, list[Image.Image]],
    ) -> dict[int, ParsedStopCheck]:
        prompts: list[str] = []
        image_lists: list[list[Image.Image]] = []
        for env_i in env_ids:
            cache = self._get_cache(env_i)
            main_views = current_views_per_env[env_i]  # F/L/R/B
            source_views = [main_views[0], main_views[1], main_views[3], main_views[2]]
            resized = source_views
            if (
                self.lavira_source_prompt_alignment
                or (self.lavira_runtime_enabled and self.is_waypoint_style)
            ):
                prefix, suffix = build_source_stop_check_parts(
                    instructions[env_i],
                    target=cache.last_target,
                    distance_threshold=self.stop_double_check_distance_threshold,
                )
                labels = (
                    "Current FORWARD view",
                    "View after turning LEFT",
                    "View after turning BEHIND",
                    "View after turning RIGHT",
                )
                content = [{"type": "text", "text": prefix}]
                for image, label in zip(resized, labels):
                    # Exact LHX double_check_stop ordering: image, then label.
                    content.append({"type": "image", "image": image})
                    content.append({"type": "text", "text": label})
                content.append({"type": "text", "text": suffix})
                messages = [{"role": "user", "content": content}]
            else:
                text = build_stop_check_text(
                    instructions[env_i],
                    distance_threshold=self.stop_double_check_distance_threshold,
                    target=cache.last_target,
                )
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": STOP_CHECK_SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": (
                            [{"type": "image", "image": image} for image in resized]
                            + [{"type": "text", "text": text}]
                        ),
                    },
                ]
            try:
                prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    add_vision_id=getattr(self, "add_vision_id", False),
                )
            except TypeError:
                prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            prompts.append(prompt)
            image_lists.append(resized)
        decoded, _ = self._profiled_batch_generate(prompts, image_lists)
        return {
            env_i: parse_stop_check_json(text)
            for env_i, text in zip(env_ids, decoded)
        }

    def _build_termination_shadow_inputs(
        self,
        instruction: str,
        current_views: list[Image.Image],
    ) -> dict[str, torch.Tensor]:
        """Build fixed-shape inputs for the read-only DITA shadow judge.

        The tensors travel with the rollout to the actor, where the current
        checkpoint evaluates them without changing the navigation response or
        consuming another vLLM sampling request.
        """
        source_views = [
            image.resize(self.image_size, Image.Resampling.BILINEAR)
            for image in (
                current_views[0],
                current_views[1],
                current_views[3],
                current_views[2],
            )
        ]
        labels = (
            "Current FORWARD view",
            "View after turning LEFT",
            "View after turning BEHIND",
            "View after turning RIGHT",
        )
        content = [
            {
                "type": "text",
                "text": build_termination_shadow_text(instruction),
            }
        ]
        for image, label in zip(source_views, labels):
            content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": label})
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": TERMINATION_SHADOW_SYSTEM_PROMPT}
                ],
            },
            {"role": "user", "content": content},
        ]
        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                add_vision_id=getattr(self, "add_vision_id", False),
            )
        except TypeError:
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.processor(
            text=[prompt],
            images=[source_views],
            padding=False,
            return_tensors="pt",
        )

        pad_id = (getattr(self.processor.tokenizer, "pad_token_id", None) or 0)
        input_ids = self._pad_1d(
            inputs["input_ids"][0], self._prompt_len, pad_value=pad_id
        )
        attention_mask = self._pad_1d(
            inputs["attention_mask"][0], self._prompt_len, pad_value=0
        )
        if "mm_token_type_ids" in inputs:
            mm_token_type_ids = self._pad_1d(
                inputs["mm_token_type_ids"][0], self._prompt_len, pad_value=0
            )
        else:
            mm_token_type_ids = torch.zeros(self._prompt_len, dtype=torch.long)

        pixel_values = inputs["pixel_values"]
        if pixel_values.shape[0] < self._total_patches:
            pixel_values = F.pad(
                pixel_values,
                (0, 0, 0, self._total_patches - pixel_values.shape[0]),
            )
        else:
            pixel_values = pixel_values[:self._total_patches]

        grid = inputs.get("image_grid_thw")
        if grid is None:
            grid = torch.zeros(self._n_images_fixed, 3, dtype=torch.long)
        elif grid.shape[0] < self._n_images_fixed:
            grid = torch.cat(
                [
                    grid,
                    torch.zeros(
                        self._n_images_fixed - grid.shape[0],
                        3,
                        dtype=grid.dtype,
                    ),
                ],
                dim=0,
            )
        else:
            grid = grid[:self._n_images_fixed]

        return {
            "termination_shadow_input_ids": input_ids.unsqueeze(0),
            "termination_shadow_attention_mask": attention_mask.unsqueeze(0),
            "termination_shadow_mm_token_type_ids": mm_token_type_ids.unsqueeze(0),
            "termination_shadow_pixel_values": pixel_values.unsqueeze(0),
            "termination_shadow_image_grid_thw": grid.unsqueeze(0),
            "termination_shadow_valid": torch.ones(1, 1, dtype=torch.bool),
        }

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _pad_1d(
        self, t: torch.Tensor, target_len: int, pad_value: int = 0
    ) -> torch.Tensor:
        """Pad/truncate a 1-D token tensor to target_len."""
        cur = t.shape[0]
        if cur >= target_len:
            return t[:target_len]
        return F.pad(t, (target_len - cur, 0), value=pad_value)

    @torch.inference_mode()
    def _compute_teacher_forcing_logprobs(
        self,
        prompt_input_ids: torch.Tensor,         # (1, prompt_len)
        prompt_attn_mask: torch.Tensor,          # (1, prompt_len)
        prompt_pixel_values: torch.Tensor,       # (total_patches, C)
        prompt_image_grid_thw: Optional[torch.Tensor],     # (n_images, 3)
        response_ids: torch.Tensor,              # (1, resp_len)
        prompt_mm_token_type_ids: Optional[torch.Tensor] = None,  # (1, prompt_len)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forcing forward for one sample.
        Returns:
          logprobs: (resp_len,)  — per-token log_prob of each response token
          entropy:  (resp_len,)  — per-token entropy
        """
        resp_len = response_ids.shape[1]

        # RIGHT-pad response to max_new_tokens so total seq_len = _prompt_len + max_new_tokens,
        # matching default_forward exactly. Qwen3.5's hybrid linear-attention layers produce
        # NaN logits for variable-length sequences; using a fixed length avoids this.
        # We still extract only the first resp_len logits after the forward pass.
        if self._padding_initialized and resp_len < self.max_new_tokens:
            pad_id_resp = int(self.processor.tokenizer.pad_token_id
                              if hasattr(self.processor, "tokenizer") else 0) or 0
            pad_size_resp = self.max_new_tokens - resp_len
            response_ids = F.pad(response_ids, (0, pad_size_resp), value=pad_id_resp)
            # attention mask: 0 for padded response positions (mirrors response_mask in default_forward)
            _resp_pad_attn = torch.zeros(1, pad_size_resp, dtype=torch.long,
                                         device=response_ids.device)
        else:
            _resp_pad_attn = None

        # LEFT-pad prompt to _prompt_len so that M-RoPE absolute positions of response
        # tokens match default_forward (which always uses _prompt_len-length padded input).
        # Without this, early steps (few history frames → shorter prompt) have a pad_len
        # position shift that changes RoPE embeddings and causes ratio explosion.
        actual_prompt_len = prompt_input_ids.shape[1]
        pad_id = int(self.processor.tokenizer.pad_token_id
                     if hasattr(self.processor, "tokenizer") else 0) or 0
        if self._padding_initialized and actual_prompt_len < self._prompt_len:
            pad_size = self._prompt_len - actual_prompt_len
            prompt_input_ids = F.pad(prompt_input_ids, (pad_size, 0), value=pad_id)
            prompt_attn_mask = F.pad(prompt_attn_mask, (pad_size, 0), value=0)
            if prompt_mm_token_type_ids is not None:
                prompt_mm_token_type_ids = F.pad(
                    prompt_mm_token_type_ids, (pad_size, 0), value=0
                )
        elif self._padding_initialized and actual_prompt_len > self._prompt_len:
            # Truncate to match _pad_1d behaviour in _build_forward_inputs_for_env
            # (keeps first _prompt_len tokens, same as t[:target_len])
            prompt_input_ids = prompt_input_ids[:, :self._prompt_len]
            prompt_attn_mask = prompt_attn_mask[:, :self._prompt_len]
            if prompt_mm_token_type_ids is not None:
                prompt_mm_token_type_ids = prompt_mm_token_type_ids[:, :self._prompt_len]

        full_ids = torch.cat([prompt_input_ids, response_ids], dim=1)  # (1, full_len)
        resp_attn_valid = torch.ones(1, resp_len, dtype=torch.long, device=prompt_attn_mask.device)
        if _resp_pad_attn is not None:
            resp_attn = torch.cat([resp_attn_valid, _resp_pad_attn.to(prompt_attn_mask.device)], dim=1)
        else:
            resp_attn = resp_attn_valid
        full_attn = torch.cat([prompt_attn_mask, resp_attn], dim=1)  # (1, full_len)

        dev = self._model_device
        model_kwargs = {
            "input_ids":      full_ids.to(dev),
            "attention_mask": full_attn.to(dev),
            "use_cache":      False,
        }
        if prompt_pixel_values is not None:
            model_kwargs["pixel_values"] = prompt_pixel_values.to(dev).to(self._dtype)
        if prompt_image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = prompt_image_grid_thw.to(dev)
        # mm_token_type_ids: required by Qwen2.5-VL for multimodal RoPE (M-RoPE).
        # Response tokens are all text (type 0).
        if prompt_mm_token_type_ids is not None:
            # Use response_ids.shape[1] (possibly padded to max_new_tokens)
            _full_resp_len = response_ids.shape[1]
            resp_type_ids = torch.zeros(1, _full_resp_len, dtype=prompt_mm_token_type_ids.dtype,
                                        device=dev)
            full_mm_type_ids = torch.cat(
                [prompt_mm_token_type_ids.to(dev), resp_type_ids], dim=1
            )
            model_kwargs["mm_token_type_ids"] = full_mm_type_ids

        # Use same autocast as default_forward for numerical consistency.
        with torch.amp.autocast("cuda", dtype=self._dtype):
            outputs = self.model(**model_kwargs)
        logits = outputs.logits[0]          # (full_len, vocab_size)

        # Response logits: positions prompt_len-1 .. prompt_len+resp_len-2
        # After padding, prompt_len == _prompt_len, matching default_forward.
        prompt_len = prompt_input_ids.shape[1]  # = _prompt_len after padding
        resp_logits = logits[prompt_len - 1: prompt_len - 1 + resp_len, :]  # (resp_len, V)
        resp_targets = response_ids[0, :resp_len]   # (resp_len,) — slice original length

        log_probs = F.log_softmax(resp_logits.float(), dim=-1)
        token_logprobs = log_probs[range(resp_len), resp_targets]  # (resp_len,)
        token_logprobs = torch.nan_to_num(token_logprobs, nan=0.0, posinf=0.0, neginf=-100.0)

        probs = log_probs.exp()
        token_entropy = -(probs * log_probs).sum(dim=-1)            # (resp_len,)

        return token_logprobs.cpu(), token_entropy.cpu()

    # Regex spans for narrow PPO loss mask.  We train on the JSON *value*
    # tokens for action / stop / stair by default. Long free-text fields remain
    # excluded so the PPO ratio doesn't accumulate drift over ~200 tokens.
    _ACTION_VALUE_RE = re.compile(
        r'"action"\s*:\s*"((?:navigate to (?:forward|left|right|behind))|(?:backtrack to \d+))"'
    )
    _STOP_VALUE_RE = re.compile(r'"stop"\s*:\s*(true|false)')
    _STAIR_VALUE_RE = re.compile(r'"stair"\s*:\s*("up"|"down"|true|false)')
    _TARGET_VALUE_RE = re.compile(
        r'"target"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
    )
    _BBOX_VALUE_RE = re.compile(r'"bbox_2d"\s*:\s*(\[[^\]]*\])')
    _POINT_VALUE_RE = re.compile(r'"point_2d"\s*:\s*(\[[^\]]*\])')

    # P1 bbox_geom_valid: depth-independent area bounds in 0-1000 normalized space
    # (full frame area = 1000*1000 = 1e6). Reject degenerate points and near-full-frame.
    _BBOX_AREA_MIN: float = 400.0       # ~20x20 px-equiv: reject point/sliver boxes
    _BBOX_AREA_MAX: float = 810_000.0   # 0.81 * 1e6: reject boxes covering whole view

















    def _compute_action_loss_mask(
        self,
        decoded_text: str,
        resp_len: int,
        mask_mode: str = "action",
    ) -> torch.Tensor:
        """
        Build a (max_new_tokens,) bool mask that is True only on tokens
        belonging to action / stop / stair values inside the JSON response.
        Falls back to all-True over the response length on any failure so
        behavior degrades gracefully.

        Implementation: re-tokenize prefix text up to each char span boundary;
        the prefix token count gives the boundary token index. Assumes the
        underlying BPE tokenizer is mostly prefix-stable (Qwen / GPT-2 family
        satisfy this for our JSON payloads).
        """
        full_mask = torch.zeros(self.max_new_tokens, dtype=torch.bool)
        if resp_len <= 0 or not decoded_text:
            return full_mask
        if mask_mode == "full":
            full_mask[:resp_len] = True
            return full_mask

        try:
            tok = self.processor.tokenizer
        except Exception:
            full_mask[:resp_len] = True
            return full_mask

        include_stop = os.environ.get(
            "QWEN_NAV_LOSS_MASK_INCLUDE_STOP", "1"
        ) not in ("0", "false", "False")
        include_stair = os.environ.get(
            "QWEN_NAV_LOSS_MASK_INCLUDE_STAIR", "1"
        ) not in ("0", "false", "False")

        spans: list[tuple[int, int]] = []
        m = self._ACTION_VALUE_RE.search(decoded_text)
        if m:
            spans.append(m.span(1))   # group 1 = "navigate to X" (without quotes)
        if include_stop:
            m2 = self._STOP_VALUE_RE.search(decoded_text)
            if m2:
                spans.append(m2.span(1))   # "true" / "false" literal
        if include_stair:
            m3 = self._STAIR_VALUE_RE.search(decoded_text)
            if m3:
                spans.append(m3.span(1))   # "up" / "down" / true / false literal
        if mask_mode == "action_target":
            target_match = self._TARGET_VALUE_RE.search(decoded_text)
            if target_match:
                spans.append(target_match.span(1))
        if mask_mode == "geometry":
            for regex in (self._BBOX_VALUE_RE, self._POINT_VALUE_RE):
                mg = regex.search(decoded_text)
                if mg:
                    spans.append(mg.span(1))

        if not spans:
            # No recognizable action span — fall back to full response so
            # the sample still contributes some signal rather than zeroing out.
            full_mask[:resp_len] = True
            return full_mask

        # Fast tokenizer offset_mapping gives a stable char→token map.
        try:
            enc = tok(
                decoded_text, add_special_tokens=False, return_offsets_mapping=True
            )
            offsets = enc["offset_mapping"]
        except Exception:
            full_mask[:resp_len] = True
            return full_mask

        any_set = False
        for ch_start, ch_end in spans:
            # First token whose end > ch_start, first token whose start >= ch_end.
            t_start = next((i for i, (a, b) in enumerate(offsets) if b > ch_start), len(offsets))
            t_end = next((i for i, (a, b) in enumerate(offsets) if a >= ch_end), len(offsets))
            t_start = max(0, min(t_start, resp_len))
            t_end = max(0, min(t_end, resp_len))
            if t_end > t_start:
                full_mask[t_start:t_end] = True
                any_set = True

        if not any_set:
            full_mask[:resp_len] = True
        return full_mask

    def _build_forward_inputs_for_env(
        self,
        inputs_single: "BatchEncoding",   # processor output for 1 sample
        response_ids: torch.Tensor,        # (1, resp_len)
        decoded_text: Optional[str] = None,
        loss_mask_mode: str = "action",
    ) -> dict[str, torch.Tensor]:
        """
        Build a fixed-shape forward_inputs dict for one env step.
        All tensors have batch dim = 1 so they can be stacked over steps.
        """
        pad_id = (
            self.processor.tokenizer.pad_token_id
            if hasattr(self.processor, "tokenizer")
            else 0
        ) or 0

        # Pad prompt ids to max_seq_len - max_new_tokens = _prompt_len
        raw_ids = inputs_single["input_ids"][0]        # (actual_prompt_len,)
        padded_ids = self._pad_1d(raw_ids, self._prompt_len, pad_value=pad_id)
        padded_attn = self._pad_1d(
            inputs_single["attention_mask"][0], self._prompt_len, pad_value=0
        )

        # Pad response ids to max_new_tokens — RIGHT pad so real tokens stay at
        # the beginning (positions 0..resp_len-1), matching resp_mask[:resp_len]=True.
        # _pad_1d uses LEFT pad (matches prompt convention) so we use F.pad directly.
        raw_resp = response_ids[0]  # (actual_resp_len,)
        resp_len = raw_resp.shape[0]
        if resp_len < self.max_new_tokens:
            padded_resp = F.pad(raw_resp, (0, self.max_new_tokens - resp_len),
                                value=pad_id)
        else:
            padded_resp = raw_resp[:self.max_new_tokens]
        resp_mask = torch.zeros(self.max_new_tokens, dtype=torch.bool)
        resp_mask[:resp_len] = True

        # pixel_values: pad to (max_total_patches, pixel_value_dim).  Never
        # truncate a live image: doing so leaves image_grid_thw describing
        # patches that are no longer present and silently corrupts actor replay.
        pv = inputs_single["pixel_values"]             # (actual_patches, C)
        actual_patches = int(pv.shape[0])
        max_pv = self._total_patches
        if actual_patches > max_pv:
            raise ValueError(
                "actor visual replay capacity is smaller than processor output: "
                f"actual_patches={actual_patches}, capacity={max_pv}, "
                f"profile_image_size={self.forward_input_profile_image_size}"
            )
        if actual_patches < max_pv:
            pv = F.pad(pv, (0, 0, 0, max_pv - actual_patches))

        # image_grid_thw: pad to (max_images, 3)
        if "image_grid_thw" in inputs_single:
            grid = inputs_single["image_grid_thw"]     # (actual_n_images, 3)
            grid_patches = int(grid.prod(dim=-1).sum().item())
            if grid_patches != actual_patches:
                raise ValueError(
                    "processor pixel/grid mismatch before actor replay: "
                    f"pixel_rows={actual_patches}, grid_patches={grid_patches}"
                )
            max_img = self._n_images_fixed
            if grid.shape[0] < max_img:
                grid = torch.cat(
                    [grid, torch.zeros(max_img - grid.shape[0], 3, dtype=grid.dtype)],
                    dim=0,
                )
            else:
                grid = grid[:max_img]
        else:
            grid = torch.zeros(self._n_images_fixed, 3, dtype=torch.long)

        # mm_token_type_ids: required by Qwen2.5-VL multimodal RoPE (M-RoPE).
        # Pad to prompt_len with 0 (text type) where tokens were left-padded.
        if "mm_token_type_ids" in inputs_single:
            raw_ttids = inputs_single["mm_token_type_ids"][0]  # (actual_prompt_len,)
            padded_ttids = self._pad_1d(raw_ttids, self._prompt_len, pad_value=0)
        else:
            padded_ttids = torch.zeros(self._prompt_len, dtype=torch.long)

        # Narrow PPO loss mask: True only on action (+stop) value tokens.
        # Falls back to full resp_mask when decoded_text not provided.
        if decoded_text is not None:
            loss_mask = self._compute_action_loss_mask(
                decoded_text, resp_len, mask_mode=loss_mask_mode
            )
        else:
            loss_mask = resp_mask.clone()

        return {
            "input_ids":          padded_ids.unsqueeze(0),     # (1, prompt_len)
            "attention_mask":     padded_attn.unsqueeze(0),     # (1, prompt_len)
            "mm_token_type_ids":  padded_ttids.unsqueeze(0),    # (1, prompt_len)
            "pixel_values":       pv.unsqueeze(0),              # (1, max_patches, C)
            "image_grid_thw":     grid.unsqueeze(0),            # (1, max_images, 3)
            "response_ids":       padded_resp.unsqueeze(0),     # (1, max_new_tokens)
            "response_mask":      resp_mask.unsqueeze(0),       # (1, max_new_tokens)
            "ppo_token_loss_mask": loss_mask.unsqueeze(0),      # (1, max_new_tokens)
        }

    # ------------------------------------------------------------------
    # BasePolicy interface — inference
    # ------------------------------------------------------------------

    def predict_action_batch(
        self,
        env_obs: Optional[dict] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Inference for a batch of envs.

        Returns (action_t, result) where result contains:
          parse_ok, bboxes, actions, info
          + (when collect_forward_inputs=True):
              prev_logprobs: (N, max_new_tokens) — per-token response log_probs
              forward_inputs: dict of padded tensors for teacher-forcing replay
        """
        if env_obs is None:
            raise ValueError("env_obs must be provided")
        profile_started = time.perf_counter()
        if self.collect_forward_inputs and not self._padding_initialized:
            self._init_padding_params()

        main_images = env_obs.get("main_images")
        instructions = env_obs.get("task_descriptions") or []
        episode_active = env_obs.get("episode_active")
        episode_ids = env_obs.get("episode_ids")
        trial_ids = env_obs.get("trial_ids")
        scene_ids = env_obs.get("scene_ids")
        states = env_obs.get("states")
        extra_views_t = env_obs.get("extra_view_images")
        runtime_depth_t = env_obs.get("wrist_images")
        scan_images_t = env_obs.get("scan_images")
        scan_depth_t = env_obs.get("scan_depth_images")
        scan_states_t = env_obs.get("scan_states")
        scan_valid_t = env_obs.get("scan_valid")
        simulator_positions_t = env_obs.get("simulator_positions")

        if main_images is None:
            raise ValueError("env_obs missing 'main_images'")

        rgb_np = (
            main_images.cpu().numpy()
            if isinstance(main_images, torch.Tensor)
            else np.asarray(main_images)
        )
        num_envs = rgb_np.shape[0]
        episode_ids = list(episode_ids) if episode_ids is not None else [""] * num_envs
        trial_ids = list(trial_ids) if trial_ids is not None else [0] * num_envs
        scene_ids = list(scene_ids) if scene_ids is not None else [""] * num_envs
        if (
            len(episode_ids) != num_envs
            or len(trial_ids) != num_envs
            or len(scene_ids) != num_envs
        ):
            raise ValueError(
                "episode_ids/trial_ids/scene_ids must contain one value per env"
            )
        self._lavira_map_episode_ids_by_env = {
            env_i: str(episode_id).strip()
            for env_i, episode_id in enumerate(episode_ids)
        }
        if len(instructions) != num_envs or any(
            not isinstance(value, str) for value in instructions
        ):
            raise TypeError(
                "task_descriptions must be a flat list[str] with one entry per env"
            )
        if episode_active is not None:
            active_np = (
                episode_active.detach().cpu().numpy()
                if isinstance(episode_active, torch.Tensor)
                else np.asarray(episode_active)
            ).astype(bool, copy=False).reshape(-1)
            if active_np.shape[0] != num_envs:
                raise ValueError("episode_active must contain one value per env")
            instructions = [
                instruction if bool(active_np[index]) else ""
                for index, instruction in enumerate(instructions)
            ]
        mask_mode_ids = env_obs.get("ppo_loss_mask_mode_id")
        if isinstance(mask_mode_ids, torch.Tensor):
            mask_mode_ids_np = mask_mode_ids.detach().cpu().numpy().astype(np.int64)
        elif mask_mode_ids is None:
            mask_mode_ids_np = np.full(num_envs, 2, dtype=np.int64)
        else:
            mask_mode_ids_np = np.asarray(mask_mode_ids, dtype=np.int64)
        if mask_mode_ids_np.shape[0] < num_envs:
            mask_mode_ids_np = np.pad(
                mask_mode_ids_np,
                (0, num_envs - mask_mode_ids_np.shape[0]),
                constant_values=2,
            )
        _mask_mode_name = {
            0: "full",
            1: "geometry",
            2: "action",
            3: "action_target",
        }

        extras_np  = None
        runtime_depth_np = None  # (N,4,H,W), canonical F/L/B/R depth for LaViRA runtime
        scan_images_np = None
        scan_depth_np = None
        scan_states_np = None
        scan_valid_np = None
        simulator_positions_np = None
        if extra_views_t is not None:
            arr = (
                extra_views_t.cpu().numpy()
                if isinstance(extra_views_t, torch.Tensor)
                else np.asarray(extra_views_t)
            )
            if arr.shape[0] == num_envs and arr.shape[1] == 3 and arr.dtype == np.uint8:
                # 4-dir RGB extras: (N, 3, H, W, 3) uint8
                extras_np = arr
        if runtime_depth_t is not None:
            arr = (
                runtime_depth_t.cpu().numpy()
                if isinstance(runtime_depth_t, torch.Tensor)
                else np.asarray(runtime_depth_t)
            )
            if arr.ndim == 5 and arr.shape[0] == num_envs and arr.shape[1] in (1, 4):
                runtime_depth_np = arr[:, :, :, :, 0].astype(np.float32, copy=False)
        if scan_images_t is not None:
            scan_images_np = (
                scan_images_t.cpu().numpy()
                if isinstance(scan_images_t, torch.Tensor)
                else np.asarray(scan_images_t)
            )
        if scan_depth_t is not None:
            arr = (
                scan_depth_t.cpu().numpy()
                if isinstance(scan_depth_t, torch.Tensor)
                else np.asarray(scan_depth_t)
            )
            scan_depth_np = arr[..., 0].astype(np.float32, copy=False)
        if scan_states_t is not None:
            scan_states_np = (
                scan_states_t.cpu().numpy()
                if isinstance(scan_states_t, torch.Tensor)
                else np.asarray(scan_states_t)
            ).astype(np.float32, copy=False)
        if scan_valid_t is not None:
            scan_valid_np = (
                scan_valid_t.cpu().numpy()
                if isinstance(scan_valid_t, torch.Tensor)
                else np.asarray(scan_valid_t)
            ).astype(bool, copy=False)
        if simulator_positions_t is not None:
            simulator_positions_np = (
                simulator_positions_t.detach().cpu().numpy()
                if isinstance(simulator_positions_t, torch.Tensor)
                else np.asarray(simulator_positions_t)
            ).astype(np.float32, copy=False)

        # P2: extract pose from extended states (N, 4) = [elapsed, hab_x, hab_z, yaw]
        pose_np = None  # (N, 3) float32: [hab_x, hab_z, gen_yaw_rad]
        if not instructions:
            instructions = [""] * num_envs

        # Episode reset detection (elapsed_steps == 0)
        if states is not None:
            elapsed_arr = (
                states.cpu().numpy()
                if isinstance(states, torch.Tensor)
                else np.asarray(states)
            )
            for i in range(num_envs):
                if instructions[i] and float(elapsed_arr[i, 0]) == 0.0:
                    self.reset_env_cache([i])
                    if self._wm_state_collector is not None:
                        self._wm_state_collector.reset_trial(
                            scene_ids[i], episode_ids[i], str(trial_ids[i])
                        )
            if elapsed_arr.shape[1] >= 4:
                pose_np = elapsed_arr[:, 1:4].astype(np.float32)

        # Keep the physical trajectory frames used by the layered-history
        # prompt. Repeated policy ticks at an unchanged simulator pose are not
        # new visual history (for example STOP verification and runtime NOOP).
        current_pils: list[Optional[Image.Image]] = [None] * num_envs
        for i in range(num_envs):
            if not instructions[i]:
                continue
            current_pils[i] = _to_pil(rgb_np[i])
            cache = self._get_cache(i)
            history_key = (
                int(round(float(elapsed_arr[i, 0]))) if elapsed_arr is not None else None,
                tuple(np.round(np.asarray(pose_np[i], dtype=np.float32), 3).tolist())
                if pose_np is not None else None,
            )
            if cache.last_history_observation_key != history_key:
                history_len_before = len(cache.history_images)
                cache.push_image(
                    current_pils[i],
                    action_aware=self.history_action_aware,
                    forward_interval=self.history_forward_interval,
                    turn_interval=self.history_turn_interval,
                )
                physical_step = (
                    int(round(float(elapsed_arr[i, 0])))
                    if elapsed_arr is not None else cache.step_count
                )
                if len(cache.history_images) > history_len_before:
                    cache.history_steps.append(physical_step)
                    cache.last_history_physical_step = physical_step
                cache.last_history_observation_key = history_key

        current_views_per_env: dict[int, list[Image.Image]] = {}
        for i in range(num_envs):
            if current_pils[i] is None:
                continue
            if self.use_4dir and extras_np is not None:
                # Main decision order remains F/L/R/B.
                current_views_per_env[i] = [
                    current_pils[i],
                    _to_pil(extras_np[i, 0]),
                    _to_pil(extras_np[i, 1]),
                    _to_pil(extras_np[i, 2]),
                ]
            else:
                current_views_per_env[i] = [current_pils[i]] * 4

        runtime_obs_by_env: dict[int, LaviraObservation] = {}
        semantic_jobs: list[dict] = []
        fusion_plans: list[
            tuple[
                int,
                LaviraNavigationController,
                tuple,
                list[tuple[LaviraObservation, Optional[int]]],
            ]
        ] = []
        if self.lavira_runtime_enabled and pose_np is not None and runtime_depth_np is not None:
            for i in range(num_envs):
                if not instructions[i] or extras_np is None:
                    continue
                # Simulator extras are [left,right,behind]; canonical LaViRA
                # order is [front,left,behind,right].
                rgb_views = [
                    current_pils[i],
                    _to_pil(extras_np[i, 0]),
                    _to_pil(extras_np[i, 2]),
                    _to_pil(extras_np[i, 1]),
                ]
                yaw = float(pose_np[i, 2])
                observation = LaviraObservation(
                    rgb_by_direction=rgb_views,
                    # GenArk provides actual canonical F/L/B/R depth.  The map
                    # still fuses only front depth after physical turns, while
                    # source-equivalent blocked-direction filtering consumes all
                    # four views before a VLM decision.
                    depth_by_direction=(
                        runtime_depth_np[i]
                        if runtime_depth_np.shape[1] == 4
                        else np.repeat(runtime_depth_np[i, 0][None, ...], 4, axis=0)
                    ),
                    yaw_by_direction=np.asarray([
                        yaw, yaw + math.pi / 2, yaw + math.pi, yaw - math.pi / 2
                    ], dtype=np.float32),
                    hab_x=float(pose_np[i, 0]),
                    hab_z=float(pose_np[i, 1]),
                    hfov_deg=float(getattr(self.lavira_runtime_cfg, "hfov_deg", 79.0)),
                )
                runtime_obs_by_env[i] = observation
                controller = self._get_lavira_runtime(i)
                if (
                    getattr(self, "_lavira_fmm_results_dir", None) is not None
                    and str(episode_ids[i]).strip()
                    and str(trial_ids[i]).strip() not in ("", "0")
                ):
                    controller.configure_fmm_output(
                        self._lavira_fmm_results_dir,
                        episode_ids[i],
                        trial_ids[i],
                    )
                scan_is_valid = bool(
                    scan_valid_np is not None and scan_valid_np[i]
                )
                cache = self._get_cache(i)
                observation_key = (
                    int(round(float(elapsed_arr[i, 0]))) if elapsed_arr is not None else None,
                    tuple(np.round(np.asarray(pose_np[i], dtype=np.float32), 3).tolist()),
                )
                # ACTION_NOOP leaves the simulator unchanged.  The next call
                # may still be needed for a stop re-check, but it must not run
                # GroundedSAM or mutate the semantic map a second time.
                if cache.last_fused_observation_key != observation_key:
                    frames: list[
                        tuple[LaviraObservation, Optional[int], bool]
                    ] = []
                    if (
                        scan_is_valid
                        and scan_images_np is not None
                        and scan_depth_np is not None
                        and scan_states_np is not None
                    ):
                        for scan_index in range(12):
                            scan_yaw = float(scan_states_np[i, scan_index, 2])
                            scan_image = _to_pil(scan_images_np[i, scan_index])
                            scan_depth = scan_depth_np[i, scan_index]
                            job_index = None
                            if self.grounded_sam_enabled:
                                # Match lavira-rft: _batch_obs() runs semantic
                                # prediction for every real panorama turn before
                                # that frame is fused into the persistent map.
                                job_index = len(semantic_jobs)
                                semantic_jobs.append(
                                    {
                                        "env_i": i,
                                        "image_rgb": np.asarray(scan_image),
                                        "classes": controller.semantic_query_classes,
                                    }
                                )
                            frames.append((
                                LaviraObservation(
                                    rgb_by_direction=[scan_image] * 4,
                                    depth_by_direction=np.repeat(
                                        scan_depth[None, ...], 4, axis=0
                                    ),
                                    yaw_by_direction=np.full(
                                        4, scan_yaw, dtype=np.float32
                                    ),
                                    hab_x=float(scan_states_np[i, scan_index, 0]),
                                    hab_z=float(scan_states_np[i, scan_index, 1]),
                                    hfov_deg=float(
                                        getattr(
                                            self.lavira_runtime_cfg,
                                            "hfov_deg",
                                            79.0,
                                        )
                                    ),
                                ),
                                job_index,
                                False,
                            ))
                    else:
                        job_index = None
                        # Match lavira-rft's normal post-action path: every
                        # real observation is segmented before map fusion.
                        if self.grounded_sam_enabled:
                            job_index = len(semantic_jobs)
                            semantic_jobs.append(
                                {
                                    "env_i": i,
                                    "image_rgb": np.asarray(current_pils[i]),
                                    "classes": controller.semantic_query_classes,
                                }
                            )
                        frames.append((observation, job_index, True))
                    fusion_plans.append(
                        (i, controller, observation_key, frames)
                    )

            # DINO/SAM results are independent across frames. Generate all
            # masks first so remote services run concurrently, then preserve
            # LHX's exact per-slot panorama order while updating each map.
            semantic_results = (
                self._segment_grounded_sam_jobs(semantic_jobs)
                if semantic_jobs else []
            )
            for env_i, controller, observation_key, frames in fusion_plans:
                for frame_observation, job_index, publish_planner_state in frames:
                    semantic_masks = (
                        semantic_results[job_index]
                        if job_index is not None
                        else None
                    )
                    self._profiled_observe(
                        controller,
                        frame_observation,
                        semantic_masks,
                        publish_planner_state=publish_planner_state,
                    )
                self._get_cache(env_i).last_fused_observation_key = (
                    observation_key
                )

        # Phase 1: dispatch buffered actions
        actions: list[Optional[int]] = [None] * num_envs
        need_infer: list[int] = []
        pending_stop_checks: list[int] = []
        # Track which envs are making a new LLM decision vs replaying cached actions.
        # Dormant envs (no instruction) count as non-decision steps.
        is_decision_per_env: list[bool] = [False] * num_envs
        for i in range(num_envs):
            if not instructions[i]:
                actions[i] = ACTION_NOOP
                is_decision_per_env[i] = False
                continue
            cache = self._get_cache(i)
            if cache.stop_check_pending:
                pending_stop_checks.append(i)
                is_decision_per_env[i] = False
                continue
            if self.lavira_runtime_enabled and i in runtime_obs_by_env:
                controller = self._get_lavira_runtime(i)
                primitive = self._profiled_next_primitive(
                    controller, runtime_obs_by_env[i]
                )
                self._drop_waypoint_records(
                    cache, controller.consume_removed_waypoint_ids()
                )
                if primitive is not None:
                    actions[i] = primitive
                    is_decision_per_env[i] = False
                    continue
                if not controller.needs_decision():
                    raise RuntimeError(
                        "LaViRA controller returned no primitive outside NEED_DECISION: "
                        f"env={i} state={controller.state}"
                    )
                if controller.consume_stop_check_ready():
                    pending_stop_checks.append(i)
                    is_decision_per_env[i] = False
                    continue
                navigation_feedback = controller.consume_navigation_feedback()
                if navigation_feedback:
                    cache.stop_rejection_feedback = navigation_feedback
                if (
                    self.is_waypoint_style
                    and pose_np is not None
                    and i in current_views_per_env
                ):
                    self._begin_decision_waypoint_record(
                        cache=cache,
                        controller=controller,
                        observation=runtime_obs_by_env[i],
                        pose_row=pose_np[i],
                        current_views=current_views_per_env[i],
                    )
            # Dispatch cached macro actions if available (both train and eval).
            # Episode resets automatically clear pending_actions via reset_env_cache
            # (triggered when elapsed_steps == 0 at the top of this method).
            if cache.pending_actions:
                actions[i] = cache.pending_actions.pop(0)
                is_decision_per_env[i] = False  # replaying cached action, not a new decision
            else:
                need_infer.append(i)
                is_decision_per_env[i] = True  # new LLM inference required

        if pending_stop_checks:
            stop_results = self._run_stop_check_batch(
                pending_stop_checks, instructions, current_views_per_env
            )
            for env_i in pending_stop_checks:
                cache = self._get_cache(env_i)
                cache.stop_check_pending = False
                if stop_results[env_i].decision == "STOP":
                    actions[env_i] = ACTION_STOP
                    cache.stop_failure_count = 0
                    cache.stop_rejection_feedback = ""
                else:
                    cache.stop_failure_count += 1
                    if cache.stop_failure_count >= self.stop_double_check_max_failures:
                        actions[env_i] = ACTION_STOP
                        cache.stop_failure_count = 0
                        cache.stop_rejection_feedback = ""
                    else:
                        controller = self._get_lavira_runtime(env_i)
                        controller.resume_after_stop_rejection()
                        actions[env_i] = self._profiled_next_primitive(
                            controller, runtime_obs_by_env[env_i]
                        )
                        cache.stop_rejection_feedback = USER_STOP_REJECTED.format(
                            count=cache.stop_failure_count
                        )

        # Phase 2: build prompts + run batched generate
        per_env_forward_inputs: dict[int, dict] = {}
        per_env_logprobs: dict[int, torch.Tensor] = {}
        per_env_entropy: dict[int, torch.Tensor] = {}
        parsed_actions: dict[int, ParsedAction] = {}
        grounded_sam_diag: dict[int, dict] = {}
        runtime_ids_by_env: dict[int, Optional[list[int]]] = {}
        source_replan_sequences: dict[int, list[int]] = {}
        deferred_replan_waypoint_by_env: dict[int, int] = {}
        prompt_by_env: dict[int, str] = {}
        raw_decoded_by_env: dict[int, str] = {}
        termination_shadow_inputs_by_env: dict[int, dict[str, torch.Tensor]] = {}

        if need_infer:
            prompts: list[str] = []
            img_lists: list[list[Image.Image]] = []
            pad_h = self.history_max_frames if self.collect_forward_inputs else None

            for env_i in need_infer:
                cache = self._get_cache(env_i)
                sampled_waypoint_ids = None
                sampled_waypoint_infos = None
                if self.is_waypoint_style:
                    sampled, sampled_idx, sampled_infos = _sample_waypoints(
                        cache.waypoints,
                        max_frames=self.history_max_frames,
                        current_leg_frames=_sample_history(
                            cache.history_images[cache.last_waypoint_history_len:],
                            every_k=max(1, len(cache.history_images[cache.last_waypoint_history_len:]) // 5),
                            max_frames=self._waypoint_images_per_record - 2,
                        )[0],
                        current_pose=pose_np[env_i] if pose_np is not None else None,
                        layered_history=self.lavira_layered_history,
                        history_wp_max=self.lavira_history_wp_max,
                        backtrack_radius_m=self.lavira_backtrack_radius_m,
                        blank_image=self._blank_image,
                    )
                    sampled_waypoint_ids = sampled_idx
                    sampled_waypoint_infos = sampled_infos
                else:
                    past_imgs = cache.history_images[:-1]
                    sampled, sampled_idx = _sample_history(
                        past_imgs,
                        every_k=self.history_every_k,
                        max_frames=self.history_max_frames,
                    )
                extra_views = None
                if extras_np is not None:
                    extra_views = [_to_pil(extras_np[env_i, k]) for k in range(3)]
                # Store current 4-dir views for potential stop double-check
                if self.use_4dir and extra_views is not None:
                    current_views_per_env[env_i] = [current_pils[env_i]] + extra_views
                else:
                    current_views_per_env[env_i] = [current_pils[env_i]] * 4

                cache_i = self._get_cache(env_i)
                controller = (
                    self._get_lavira_runtime(env_i)
                    if self.lavira_runtime_enabled and env_i in runtime_obs_by_env else None
                )
                runtime_ids = (
                    controller.available_waypoint_ids(runtime_obs_by_env[env_i])
                    if controller is not None else None
                )
                runtime_ids_by_env[env_i] = runtime_ids
                blocked_directions = (
                    controller.blocked_directions(runtime_obs_by_env[env_i])
                    if controller is not None else None
                )
                replan_waypoint_id = (
                    controller.consume_backtrack_replan_waypoint_id()
                    if controller is not None and self.lavira_backtrack_second_chance else None
                )
                if replan_waypoint_id is not None:
                    deferred_replan_waypoint_by_env[env_i] = int(replan_waypoint_id)
                    prompt_text, ordered_imgs = self._build_backtrack_replan_prompt(
                        instruction=instructions[env_i],
                        cache=cache_i,
                        waypoint_id=replan_waypoint_id,
                        current_view=current_pils[env_i],
                        extra_views=extra_views,
                        blocked_directions=blocked_directions,
                    )
                    print(
                        f"[LaViRA][backtrack-replan] env={env_i} waypoint={replan_waypoint_id}",
                        flush=True,
                    )
                else:
                    prompt_text, ordered_imgs = self._build_prompt(
                        instruction=instructions[env_i],
                        history_imgs=sampled,
                        history_step_indices=sampled_idx,
                        current_view=current_pils[env_i],
                        extra_views=extra_views,
                        pad_history_to=pad_h,
                        stop_rejection_feedback=cache_i.stop_rejection_feedback,
                        history_waypoint_ids=sampled_waypoint_ids,
                        history_waypoint_infos=sampled_waypoint_infos,
                        available_backtrack_ids=runtime_ids,
                        allow_move_behind=(
                            controller is None
                            or controller.current_waypoint_id in (None, 0)
                        ),
                        blocked_directions=blocked_directions,
                    )
                prompts.append(prompt_text)
                img_lists.append(ordered_imgs)

            decoded, gen_ids_list = self._profiled_batch_generate(
                prompts, img_lists
            )
            prompt_by_env = dict(zip(need_infer, prompts))

            # Native ZS_Evaluator_mp retries only responses without a JSON
            # object. Keep this eval-only so RFT samples remain on-policy and
            # preserve their original response/logprob pairing.
            if self.json_retry_enabled and not self.collect_forward_inputs:
                invalid = [
                    idx for idx, text in enumerate(decoded)
                    if re.search(r"\{.*\}", str(text), re.DOTALL) is None
                ]
                retry_count = 0
                while invalid and retry_count < self.json_retry_max_retries:
                    retry_prompts = [prompts[idx] for idx in invalid]
                    retry_images = [img_lists[idx] for idx in invalid]
                    retry_decoded, retry_ids = self._profiled_batch_generate(
                        retry_prompts, retry_images
                    )
                    for local_idx, original_idx in enumerate(invalid):
                        decoded[original_idx] = retry_decoded[local_idx]
                        gen_ids_list[original_idx] = retry_ids[local_idx]
                    invalid = [
                        idx for idx in invalid
                        if re.search(r"\{.*\}", str(decoded[idx]), re.DOTALL) is None
                    ]
                    retry_count += 1
            raw_decoded_by_env = dict(zip(need_infer, decoded))

            if self.collect_forward_inputs and self.termination_shadow_enabled:
                for env_i in need_infer:
                    termination_shadow_inputs_by_env[env_i] = (
                        self._build_termination_shadow_inputs(
                            instructions[env_i], current_views_per_env[env_i]
                        )
                    )

            # If training mode: compute teacher-forcing logprobs + store forward_inputs
            if self.collect_forward_inputs:
                # Re-tokenize per sample to get padded tensors
                for idx, env_i in enumerate(need_infer):
                    single_inputs = self.processor(
                        text=[prompts[idx]],
                        images=[img_lists[idx]],
                        padding=False,
                        return_tensors="pt",
                    )
                    resp_ids = gen_ids_list[idx]  # (1, resp_len)

                    fi = self._build_forward_inputs_for_env(
                        single_inputs,
                        resp_ids,
                        decoded_text=decoded[idx],
                        loss_mask_mode=_mask_mode_name.get(
                            int(mask_mode_ids_np[env_i]), "action"
                        ),
                    )
                    fi.update(termination_shadow_inputs_by_env.get(env_i, {}))
                    per_env_forward_inputs[env_i] = fi

                    if self.skip_rollout_logprobs:
                        # Actor will recompute prev_logprobs via eval forward before
                        # PPO training (recompute_prev_logprobs=True).  Fill zeros as
                        # placeholder so tensor shapes remain consistent downstream.
                        per_env_logprobs[env_i] = torch.zeros(
                            self.max_new_tokens, dtype=torch.float32
                        )
                        per_env_entropy[env_i] = torch.zeros(
                            self.max_new_tokens, dtype=torch.float32
                        )
                    else:
                        # Teacher-forcing logprobs
                        lp, ent = self._compute_teacher_forcing_logprobs(
                            prompt_input_ids=single_inputs["input_ids"],
                            prompt_attn_mask=single_inputs["attention_mask"],
                            prompt_pixel_values=single_inputs.get("pixel_values"),
                            prompt_image_grid_thw=single_inputs.get("image_grid_thw"),
                            response_ids=resp_ids,
                            prompt_mm_token_type_ids=single_inputs.get("mm_token_type_ids"),
                        )

                        # Pad to max_new_tokens
                        resp_len = lp.shape[0]
                        if resp_len < self.max_new_tokens:
                            lp = F.pad(lp, (0, self.max_new_tokens - resp_len))
                            ent = F.pad(ent, (0, self.max_new_tokens - resp_len))
                        else:
                            lp = lp[:self.max_new_tokens]
                            ent = ent[:self.max_new_tokens]
                        per_env_logprobs[env_i] = lp    # (max_new_tokens,)
                        per_env_entropy[env_i] = ent

            # Phase 3a: parse JSON responses
            use_merged = self.prompt_style == "lavira_merged"
            use_waypoint = self.is_waypoint_style
            for env_i, text in zip(need_infer, decoded):
                if use_waypoint:
                    parsed = (
                        parse_lavira_canonical_waypoint_json(text)
                        if self.prompt_style == "canonical_v1"
                        else parse_lavira_waypoint_json(text)
                    )
                elif use_merged:
                    parsed = parse_lavira_merged_json(text)
                else:
                    parsed = parse_lavira_json(text)
                # RL-Struct style length guardrail: reward useful structured
                # responses, not empty JSON fragments or runaway reasoning.
                text_len = len(str(text).strip())
                if 80 <= text_len <= 3500:
                    parsed.reward_bits |= REWARD_LENGTH_OK
                cache = self._get_cache(env_i)
                cache.last_parse_ok = parsed.ok
                cache.last_err = parsed.err
                # For merged schema: bbox_2d is the navigation target bbox;
                # store in last_bbox so existing bboxes diagnostics still works.
                cache.last_bbox = parsed.bbox_2d if (use_merged or use_waypoint) else parsed.bbox
                cache.last_point = parsed.point_2d
                cache.last_waypoint_id = parsed.waypoint_id
                cache.last_reasoning_plan_action = parsed.reasoning_plan_action
                cache.last_reasoning_bbox_point = parsed.reasoning_bbox_point
                cache.last_target = parsed.target
                cache.last_backtrack_valid = False
                if (
                    not parsed.ok
                    and self._parse_fail_debug_printed < self._parse_fail_debug_limit
                ):
                    preview = str(text).replace("\n", "\\n")[:1200]
                    print(
                        f"[QwenNav][parse-fail-debug] env={env_i} "
                        f"style={self.prompt_style} err={parsed.err} "
                        f"text={preview!r}",
                        flush=True,
                    )
                    self._parse_fail_debug_printed += 1
                if use_merged and parsed.action_type == "BACKTRACK":
                    import logging as _log
                    _log.getLogger(__name__).debug(
                        "merged:backtrack env=%d wp=%s degraded→forward",
                        env_i, parsed.waypoint_id,
                    )
                parsed_actions[env_i] = parsed

            # LHX/LaViRA second chance is immediate: after a valid BACKTRACK
            # decision, query the VLM again on the chosen old waypoint's
            # stored panorama and project the new target from that old pose.
            # Eval can execute both responses in this call. Training schedules
            # the replan on the next policy tick so both responses retain
            # independent on-policy token/logprob records.
            if (
                self.lavira_runtime_enabled
                and self.lavira_backtrack_second_chance
                and not self.collect_forward_inputs
            ):
                replan_envs: list[int] = []
                replan_prompts: list[str] = []
                replan_images: list[list[Image.Image]] = []
                replan_context: dict[int, tuple[_WaypointRecord, LaviraObservation]] = {}
                for env_i in need_infer:
                    parsed = parsed_actions[env_i]
                    if parsed.action_type != "BACKTRACK" or not parsed.ok:
                        continue
                    controller = self._get_lavira_runtime(env_i)
                    current_observation = runtime_obs_by_env.get(env_i)
                    cache = self._get_cache(env_i)
                    record = self._find_waypoint(cache, parsed.waypoint_id)
                    runtime_ids = runtime_ids_by_env.get(env_i)
                    if (
                        current_observation is None
                        or record is None
                        or runtime_ids is None
                        or parsed.waypoint_id not in runtime_ids
                    ):
                        source_replan_sequences[env_i] = [ACTION_PARSE_FAIL]
                        parsed.backtrack_valid = False
                        cache.last_backtrack_valid = False
                        continue
                    anchor_observation = self._waypoint_anchor_observation(record)
                    if anchor_observation is None:
                        source_replan_sequences[env_i] = [ACTION_PARSE_FAIL]
                        parsed.backtrack_valid = False
                        cache.last_backtrack_valid = False
                        continue

                    # Source marks the abandoned branch before composing the
                    # replan prompt, so the old turn and descendants cannot be
                    # selected again.
                    cache.mark_failed_branch(parsed.waypoint_id)
                    controller.waypoints.mark_failed_branch(parsed.waypoint_id)
                    anchor_views_flrb = [
                        anchor_observation.rgb_by_direction[0],
                        anchor_observation.rgb_by_direction[1],
                        anchor_observation.rgb_by_direction[3],
                        anchor_observation.rgb_by_direction[2],
                    ]
                    prompt_text, ordered_imgs = self._build_backtrack_replan_prompt(
                        instruction=instructions[env_i],
                        cache=cache,
                        waypoint_id=int(parsed.waypoint_id),
                        current_view=anchor_views_flrb[0],
                        extra_views=anchor_views_flrb[1:],
                        blocked_directions=controller.blocked_directions(anchor_observation),
                    )
                    replan_envs.append(env_i)
                    replan_prompts.append(prompt_text)
                    replan_images.append(ordered_imgs)
                    replan_context[env_i] = (record, anchor_observation)

                if replan_envs:
                    replan_decoded, _ = self._profiled_batch_generate(
                        replan_prompts, replan_images
                    )
                    for env_i, text in zip(replan_envs, replan_decoded):
                        record, anchor_observation = replan_context[env_i]
                        controller = self._get_lavira_runtime(env_i)
                        cache = self._get_cache(env_i)
                        replan = (
                            parse_lavira_canonical_waypoint_json(text)
                            if self.prompt_style == "canonical_v1"
                            else parse_lavira_waypoint_json(text)
                        )
                        anchor_views_flrb = [
                            anchor_observation.rgb_by_direction[0],
                            anchor_observation.rgb_by_direction[1],
                            anchor_observation.rgb_by_direction[3],
                            anchor_observation.rgb_by_direction[2],
                        ]
                        if replan.ok and replan.action_type == "NAVIGATE" and not replan.stop:
                            sam_result = self._refine_waypoint_with_grounded_sam(
                                replan, anchor_views_flrb, env_i=env_i
                            )
                            if self.prompt_style == "canonical_v1":
                                sam_result = self._apply_grounding_geometry_filter(
                                    controller,
                                    anchor_observation,
                                    replan,
                                    sam_result,
                                )
                            if sam_result is not None and sam_result.detected:
                                replan.point_2d = sam_result.point_2d
                                replan.bbox_2d = sam_result.bbox_2d
                            # LHX falls back to a central bbox when the VLM did
                            # not return usable geometry.  In RLinf GroundedSAM
                            # owns geometry, so retain that same projection
                            # fallback only when detection is absent.
                            if replan.bbox_2d is None:
                                replan.bbox_2d = [250.0, 250.0, 750.0, 750.0]
                                replan.point_2d = None

                            replan_waypoint_id = cache.next_waypoint_id
                            accepted = controller.accept_backtrack_replan(
                                record.id,
                                anchor_observation,
                                replan.raw_dir or "navigate to forward",
                                replan.point_2d,
                                replan.bbox_2d,
                                replan.target,
                                replan.progress,
                                replan.stair,
                                replan_waypoint_id=replan_waypoint_id,
                                canonical_mode=self.prompt_style == "canonical_v1",
                            )
                            if accepted:
                                pose_row = np.asarray(record.pose_hab, dtype=np.float32)
                                self._save_waypoint_record(
                                    cache=cache,
                                    parsed=replan,
                                    pose_row=pose_row,
                                    current_views=anchor_views_flrb,
                                    runtime_observation=anchor_observation,
                                    waypoint_id=replan_waypoint_id,
                                )
                                primitive = self._profiled_next_primitive(
                                    controller, runtime_obs_by_env[env_i]
                                )
                                source_replan_sequences[env_i] = [
                                    primitive if primitive is not None else ACTION_NOOP
                                ]
                                replan.backtrack_valid = True
                                # The executed response is the source-style
                                # replan, not the initial BACKTRACK request.
                                # Keep candidate diagnostics aligned with the
                                # action which actually reaches the simulator.
                                cache.last_parse_ok = replan.ok
                                cache.last_err = replan.err
                                cache.last_bbox = replan.bbox_2d
                                cache.last_point = replan.point_2d
                                cache.last_waypoint_id = replan.waypoint_id
                                cache.last_reasoning_plan_action = replan.reasoning_plan_action
                                cache.last_reasoning_bbox_point = replan.reasoning_bbox_point
                                cache.last_target = replan.target
                                cache.last_backtrack_valid = True
                                parsed_actions[env_i] = replan
                                raw_decoded_by_env[env_i] = text
                                print(
                                    f"[LaViRA][backtrack-replan] mode=source_immediate env={env_i} "
                                    f"waypoint={record.id} action={replan.raw_dir} "
                                    f"target={replan.target!r}",
                                    flush=True,
                                )
                                continue
                        source_replan_sequences[env_i] = [ACTION_PARSE_FAIL]
                        cache.last_backtrack_valid = False
                        print(
                            f"[LaViRA][backtrack-replan-failed] env={env_i} "
                            f"waypoint={record.id} fallback=parse_fail",
                            flush=True,
                        )

            # Non-runtime compatibility path. Source waypoint mode instead
            # grounds and approaches the STOP target; the controller requests
            # this panorama only after that waypoint ends.
            if (
                self.stop_double_check_enabled
                and not self.collect_forward_inputs
                and not (self.lavira_runtime_enabled and self.is_waypoint_style)
            ):
                for env_i in need_infer:
                    parsed = parsed_actions[env_i]
                    if not (parsed.ok and parsed.stop):
                        continue
                    cache = self._get_cache(env_i)
                    cache.stop_check_pending = True
                    parsed_actions[env_i] = ParsedAction(
                        actions=[ACTION_PANORAMA_SCAN],
                        bbox=None,
                        stop=False,
                        stair=False,
                        progress=parsed.progress,
                        reasoning="[physical panorama before STOP verification]",
                        raw_dir=None,
                        ok=True,
                        err=None,
                    )

            # Most NAVIGATE refinements use the current panorama. Dispatch
            # those targets together so the GroundedSAM services work in
            # parallel. Backtrack-anchor refinements retain their dedicated
            # source control flow below.
            prefetched_sam_results: dict[int, object] = {}
            if self.grounded_sam_enabled:
                # Only batch-capable refiners opt into the prefetch path. A
                # legacy/custom test adapter continues through its original
                # per-target call below, preserving the old lifecycle.
                refiner = self._get_grounded_sam()
                batch_opt_in = isinstance(refiner, GroundedSAMServicePool) or bool(
                    getattr(refiner, "local_pipeline_enabled", False)
                )
                if not batch_opt_in or not callable(
                    getattr(refiner, "refine_batch", None)
                ):
                    refiner = None
            else:
                refiner = None
            if refiner is not None:
                refine_envs: list[int] = []
                refine_jobs: list[dict] = []
                for env_i in need_infer:
                    if env_i in prefetched_sam_results:
                        continue
                    parsed = parsed_actions[env_i]
                    source_stop_navigation = bool(
                        self.lavira_runtime_enabled
                        and self.is_waypoint_style
                        and parsed.ok
                        and parsed.stop
                    )
                    if (
                        env_i in source_replan_sequences
                        or env_i in deferred_replan_waypoint_by_env
                        or runtime_obs_by_env.get(env_i) is None
                        or not parsed.ok
                        or (parsed.stop and not source_stop_navigation)
                        or (
                            parsed.action_type != "NAVIGATE"
                            and not source_stop_navigation
                        )
                        or not str(parsed.target or "").strip()
                    ):
                        continue
                    view = self._chosen_direction_view(
                        parsed.raw_dir, current_views_per_env[env_i]
                    )
                    refine_envs.append(env_i)
                    refine_jobs.append(
                        {
                            "env_i": env_i,
                            "image_rgb": np.asarray(view.convert("RGB")),
                            "target": parsed.target,
                            "target_region": parsed.target_region,
                            "stair": parsed.stair,
                        }
                    )
                if refine_jobs:
                    prefetched_sam_results = dict(zip(
                        refine_envs,
                        self._refine_grounded_sam_jobs(refine_jobs),
                    ))

            # Phase 3d: apply final actions
            for env_i in need_infer:
                parsed = parsed_actions[env_i]
                act_seq = parsed.actions if parsed.actions else (
                    [ACTION_PARSE_FAIL] if not parsed.ok else [ACTION_STOP]
                )
                cache = self._get_cache(env_i)
                sam_result = None
                runtime_ids = runtime_ids_by_env.get(env_i)

                if env_i in source_replan_sequences:
                    act_seq = source_replan_sequences[env_i]
                elif self.lavira_runtime_enabled and act_seq[0] != ACTION_PANORAMA_SCAN:
                    controller = self._get_lavira_runtime(env_i)
                    observation = runtime_obs_by_env.get(env_i)
                    accepted = False
                    source_stop_navigation = bool(
                        self.is_waypoint_style and parsed.ok and parsed.stop
                    )
                    if (
                        observation is not None
                        and parsed.ok
                        and (not parsed.stop or source_stop_navigation)
                    ):
                        if (
                            parsed.action_type == "NAVIGATE"
                            or source_stop_navigation
                        ):
                            replan_source_id = deferred_replan_waypoint_by_env.get(env_i)
                            navigation_observation = observation
                            navigation_views = current_views_per_env[env_i]
                            navigation_pose = pose_np[env_i] if pose_np is not None else None
                            waypoint_id = (
                                controller.current_waypoint_id
                            )
                            if replan_source_id is not None:
                                source_record = self._find_waypoint(cache, replan_source_id)
                                anchor_observation = (
                                    self._waypoint_anchor_observation(source_record)
                                    if source_record is not None else None
                                )
                                if source_record is None or anchor_observation is None:
                                    parsed.ok = False
                                    parsed.err = "missing_backtrack_anchor"
                                else:
                                    navigation_observation = anchor_observation
                                    navigation_views = [
                                        anchor_observation.rgb_by_direction[0],
                                        anchor_observation.rgb_by_direction[1],
                                        anchor_observation.rgb_by_direction[3],
                                        anchor_observation.rgb_by_direction[2],
                                    ]
                                    navigation_pose = np.asarray(
                                        source_record.pose_hab, dtype=np.float32
                                    )
                                    waypoint_id = cache.next_waypoint_id
                            # The universal model schema emits a target phrase,
                            # while GroundedSAM owns visual localization.
                            try:
                                sam_result = prefetched_sam_results.get(env_i)
                                if sam_result is None:
                                    sam_result = self._refine_waypoint_with_grounded_sam(
                                        parsed, navigation_views, env_i=env_i
                                    )
                            except Exception as exc:
                                sam_result = None
                                grounded_sam_diag[env_i] = {
                                    "enabled": bool(self.grounded_sam_enabled),
                                    "used": False,
                                    "detected": False,
                                    **self._grounding_target_diag(parsed),
                                    "label": "",
                                    "confidence": 0.0,
                                    "fallback_reason": f"exception:{type(exc).__name__}",
                                    "point_2d": None,
                                    "bbox_2d": None,
                                }
                            if self.prompt_style == "canonical_v1" and (
                                sam_result is not None
                                or target_geometry_kind(parsed.target) == "area"
                            ):
                                sam_result = self._apply_grounding_geometry_filter(
                                    controller,
                                    navigation_observation,
                                    parsed,
                                    sam_result,
                                )
                            if sam_result is not None:
                                detected = bool(sam_result.detected)
                                grounded_sam_diag[env_i] = {
                                    "enabled": bool(self.grounded_sam_enabled),
                                    "used": detected,
                                    "detected": detected,
                                    **self._grounding_target_diag(parsed),
                                    "label": sam_result.label,
                                    "confidence": float(sam_result.confidence),
                                    "fallback_reason": sam_result.fallback_reason,
                                    "point_2d": sam_result.point_2d,
                                    "bbox_2d": sam_result.bbox_2d,
                                    "candidates": sam_result.candidates,
                                }
                                if detected:
                                    parsed.point_2d = sam_result.point_2d
                                    parsed.bbox_2d = sam_result.bbox_2d
                                    cache.last_point = parsed.point_2d
                                    cache.last_bbox = parsed.bbox_2d
                            if (
                                parsed.bbox_2d is None
                                and getattr(controller, "map_backend", None) == "source"
                                and not (
                                    bool(
                                        getattr(
                                            self,
                                            "canonicalize_lavira_waypoint_query",
                                            False,
                                        )
                                    )
                                    and sam_result is not None
                                    and sam_result.fallback_reason
                                    == "scene_region_bbox"
                                )
                            ):
                                # Exact LHX grounding fallback: when DINO
                                # returns no box (or grounding raises), project
                                # the central quarter and let source depth /
                                # traversibility backoff decide executability.
                                # This is valid in training too: it introduces
                                # no extra LLM response and therefore preserves
                                # the on-policy token/logprob trajectory.
                                parsed.point_2d = None
                                parsed.bbox_2d = [250.0, 250.0, 750.0, 750.0]
                                cache.last_point = None
                                cache.last_bbox = parsed.bbox_2d
                                diag = grounded_sam_diag.setdefault(env_i, {})
                                prior_reason = str(diag.get("fallback_reason", ""))
                                diag["fallback_reason"] = (
                                    f"{prior_reason}|lhx_center_bbox"
                                    if prior_reason else "lhx_center_bbox"
                                )
                                diag["bbox_2d"] = list(parsed.bbox_2d)
                            # P0 research capture happens at the exact
                            # decision boundary, after target refinement but
                            # before accept_navigation mutates the source
                            # controller. It is fully opt-in and requires a
                            # real 12-frame panorama plus simulator pose.
                            if (
                                self._wm_state_collector is not None
                                and parsed.action_type == "NAVIGATE"
                                and not parsed.stop
                                and scan_valid_np is not None
                                and bool(scan_valid_np[env_i])
                                and scan_images_np is not None
                                and scan_depth_np is not None
                                and scan_states_np is not None
                                and pose_np is not None
                                and simulator_positions_np is not None
                            ):
                                collection_decision_index = (
                                    self._wm_state_collector.next_decision_index(
                                        scene_ids[env_i], episode_ids[env_i],
                                        str(trial_ids[env_i]),
                                    )
                                )
                                self._wm_state_collector.collect(
                                    episode_id=episode_ids[env_i],
                                    trial_id=str(trial_ids[env_i]),
                                    scene_id=scene_ids[env_i],
                                    instruction=instructions[env_i],
                                    decision_index=collection_decision_index,
                                    sim_step=int(round(float(elapsed_arr[env_i, 0]))),
                                    rgb=np.asarray(scan_images_np[env_i], dtype=np.uint8),
                                    depth_m=np.asarray(scan_depth_np[env_i], dtype=np.float32),
                                    yaw_rad=np.asarray(scan_states_np[env_i, :, 2], dtype=np.float32),
                                    pose_hab=np.asarray(pose_np[env_i], dtype=np.float32),
                                    position_genesis=np.asarray(
                                        simulator_positions_np[env_i], dtype=np.float32
                                    ),
                                    parsed=parsed,
                                    raw_response=raw_decoded_by_env.get(env_i, ""),
                                    cache=cache,
                                    controller=controller,
                                    observation=observation,
                                    policy_snapshot=(
                                        self.export_cf_policy_snapshot(env_i)
                                    ),
                                )
                            # LHX projects the highest-confidence DINO bbox at its
                            # bottom centre.  A point is optional; requiring both
                            # silently bypasses the controller for bbox-only
                            # GroundedSAM detections.
                            if parsed.point_2d is not None or parsed.bbox_2d is not None:
                                if replan_source_id is not None:
                                    accepted = controller.accept_backtrack_replan(
                                        replan_source_id,
                                        navigation_observation,
                                        parsed.raw_dir or "navigate to forward",
                                        parsed.point_2d,
                                        parsed.bbox_2d,
                                        parsed.target,
                                        parsed.progress,
                                        parsed.stair,
                                        replan_waypoint_id=waypoint_id,
                                        canonical_mode=self.prompt_style == "canonical_v1",
                                    )
                                else:
                                    navigation_action = (
                                        parsed.raw_dir
                                        if parsed.raw_dir in {
                                            "navigate to forward",
                                            "navigate to left",
                                            "navigate to right",
                                            "navigate to behind",
                                        }
                                        else "navigate to forward"
                                    )
                                    accepted = controller.accept_navigation(
                                        navigation_action, parsed.point_2d,
                                        parsed.bbox_2d, parsed.target, parsed.progress, observation,
                                        parsed.stair, waypoint_id=waypoint_id,
                                        canonical_mode=self.prompt_style == "canonical_v1",
                                        stop_after_reach=source_stop_navigation,
                                    )
                            if accepted and navigation_pose is not None:
                                self._save_waypoint_record(
                                    cache=cache,
                                    parsed=parsed,
                                    pose_row=navigation_pose,
                                    current_views=navigation_views,
                                    runtime_observation=navigation_observation,
                                    waypoint_id=waypoint_id,
                                )
                        elif parsed.action_type == "BACKTRACK":
                            # The controller's FMM graph and the prompt-image
                            # cache must agree on this node.  A navigation
                            # response without saveable geometry cannot become
                            # a source-style replan anchor.
                            cache_has_waypoint = any(
                                record.id == parsed.waypoint_id
                                for record in cache.waypoints
                            )
                            # The prompt's enum is generated from runtime_ids.
                            # Require the same controller-owned set here so a
                            # stale history record cannot be executed after the
                            # FMM graph has retired it.
                            runtime_has_waypoint = (
                                parsed.waypoint_id in runtime_ids
                                if runtime_ids is not None else False
                            )
                            if (
                                runtime_has_waypoint
                                and cache_has_waypoint
                                and self.lavira_backtrack_second_chance
                                and self.collect_forward_inputs
                            ):
                                cache.mark_failed_branch(parsed.waypoint_id)
                                accepted = controller.schedule_backtrack_replan(
                                    parsed.waypoint_id
                                )
                            else:
                                accepted = (
                                    runtime_has_waypoint
                                    and cache_has_waypoint
                                    and controller.accept_backtrack(
                                        parsed.waypoint_id, observation
                                    )
                                )
                            parsed.backtrack_valid = accepted
                            cache.last_backtrack_valid = accepted
                        if accepted:
                            primitive = self._profiled_next_primitive(
                                controller, observation
                            )
                            # Source LaViRA does not call env.step when no
                            # executable primitive exists. RLinf's batched
                            # interface carries that branch as a true no-op.
                            act_seq = [primitive] if primitive is not None else [ACTION_NOOP]
                        else:
                            if parsed.action_type == "BACKTRACK":
                                # An unavailable waypoint is an invalid action,
                                # not an executable no-op.  NOOP would leave
                                # Habitat at the same state and let the model
                                # repeat the stale backtrack forever.  Reuse
                                # the existing parser-failure fallback, which
                                # maps to one forward primitive in both
                                # Genesis and Habitat.  This branch is distinct
                                # from projection failure for a valid NAVIGATE
                                # action, whose existing behavior is unchanged.
                                print(
                                    f"[LaViRA][backtrack-invalid] env={env_i} "
                                    f"waypoint={parsed.waypoint_id} fallback=parse_fail",
                                    flush=True,
                                )
                                parsed.backtrack_valid = False
                                act_seq = [ACTION_PARSE_FAIL]
                            else:
                                # A detector miss is not a schema failure.
                                # Preserve the model's requested direction.
                                act_seq = parsed.actions

                # B1: encode has_bbox sentinel when merged schema + valid bbox_2d.
                has_bbox_sentinel = (
                    self.prompt_style == "lavira_merged"
                    and parsed.ok
                    and parsed.bbox_2d is not None
                    and act_seq[0] in (
                        ACTION_STOP, ACTION_FORWARD,
                        ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
                    )
                ) or (
                    self.is_waypoint_style
                    and parsed.ok
                    and parsed.action_type == "NAVIGATE"
                    and self.bbox_geom_valid(parsed)
                    and act_seq[0] in (
                        ACTION_STOP, ACTION_FORWARD,
                        ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
                    )
                )
                if has_bbox_sentinel:
                    actions[env_i] = ACTION_PARSE_OK_HAS_BBOX_BASE + act_seq[0]
                else:
                    if parsed.ok:
                        actions[env_i] = act_seq[0]
                    elif int(getattr(parsed, "reward_bits", 0)) > 0:
                        actions[env_i] = (
                            ACTION_STRUCTURED_PARSE_FAIL_BASE
                            + int(getattr(parsed, "reward_bits", 0))
                        )
                    elif getattr(parsed, "schema_ok", False):
                        actions[env_i] = ACTION_SCHEMA_OK_PARSE_FAIL
                    else:
                        actions[env_i] = ACTION_PARSE_FAIL

                if not self.lavira_runtime_enabled and len(act_seq) > 1:
                    cache.pending_actions.extend(act_seq[1:])
                # Clear rejection feedback once a non-stop action is taken
                if not parsed.stop and not parsed.reasoning.startswith("[stop rejected #"):
                    cache.stop_rejection_feedback = ""
                if env_i in current_views_per_env:
                    self._save_grounded_sam_diagnostic(
                        env_i=env_i,
                        cache=cache,
                        instruction=instructions[env_i],
                        parsed=parsed,
                        raw_model_output=raw_decoded_by_env.get(env_i, ""),
                        current_views=current_views_per_env[env_i],
                        runtime_observation=runtime_obs_by_env.get(env_i),
                        sam_result=sam_result,
                        diag=grounded_sam_diag.get(env_i),
                    )

        # ---- Action distribution logger (per inference batch) ----
        if self._action_stats_flush_every > 0 and need_infer:
            for env_i in need_infer:
                p = parsed_actions.get(env_i)
                if p is None:
                    continue
                self._action_stats["total"] += 1
                bits = int(getattr(p, "reward_bits", 0))
                if bits & REWARD_JSON_VALID:
                    self._action_stats["json_valid"] += 1
                if bits & REWARD_REQUIRED_FIELDS:
                    self._action_stats["struct_ok"] += 1
                if bits & REWARD_FIELD_FORMAT:
                    self._action_stats["field_format_ok"] += 1
                if bits & REWARD_GEOMETRY_VALID:
                    self._action_stats["geometry_ok"] += 1
                if bits & REWARD_LENGTH_OK:
                    self._action_stats["length_ok"] += 1
                if not p.ok:
                    self._action_stats["parse_fail"] += 1
                    continue
                gs_diag = grounded_sam_diag.get(i, {})
                if gs_diag.get("used", False):
                    self._action_stats["grounded_sam_used"] += 1
                if gs_diag.get("detected", False):
                    self._action_stats["grounded_sam_detected"] += 1
                if p.stop:
                    self._action_stats["stop"] += 1
                    continue
                if p.action_type == "BACKTRACK":
                    self._action_stats["backtrack"] += 1
                    if p.backtrack_valid:
                        self._action_stats["backtrack_valid"] += 1
                    continue
                rd = (p.raw_dir or "").replace("navigate to ", "").strip()
                if rd in ("forward", "left", "right", "behind"):
                    self._action_stats[rd] += 1
                # P1 hybrid-gated controller: record bbox gate outcome per decision.
                if self.lavira_runtime_enabled and self.prompt_style in ("lavira_merged", "lavira_waypoint", "canonical_v1"):
                    if self.bbox_geom_valid(p):
                        self._action_stats["bbox_valid"] += 1
                    else:
                        self._action_stats["bbox_fallback"] += 1
                    if self.is_waypoint_style and self.point_geom_valid(p):
                        self._action_stats["point_valid"] += 1
            self._action_stats_counter += 1
            if self._action_stats_counter >= self._action_stats_flush_every:
                tot = max(self._action_stats["total"], 1)
                mode = "train" if self.collect_forward_inputs else "eval"
                pct = lambda k: 100.0 * self._action_stats[k] / tot
                gate_str = ""
                if self.lavira_runtime_enabled and self.prompt_style in ("lavira_merged", "lavira_waypoint", "canonical_v1"):
                    gate_str = (
                        f" || bbox_valid={self._action_stats['bbox_valid']}({pct('bbox_valid'):.1f}%) "
                        f"fallback={self._action_stats['bbox_fallback']}({pct('bbox_fallback'):.1f}%)"
                    )
                    if self.is_waypoint_style:
                        gate_str += (
                            f" point_valid={self._action_stats['point_valid']}({pct('point_valid'):.1f}%) "
                            f"backtrack_valid={self._action_stats['backtrack_valid']}({pct('backtrack_valid'):.1f}%)"
                        )
                        if self.grounded_sam_enabled:
                            gate_str += (
                                f" gsam_used={self._action_stats['grounded_sam_used']}({pct('grounded_sam_used'):.1f}%) "
                                f"gsam_detected={self._action_stats['grounded_sam_detected']}({pct('grounded_sam_detected'):.1f}%)"
                            )
                struct_str = (
                    f" || json={self._action_stats['json_valid']}({pct('json_valid'):.1f}%) "
                    f"struct={self._action_stats['struct_ok']}({pct('struct_ok'):.1f}%) "
                    f"field={self._action_stats['field_format_ok']}({pct('field_format_ok'):.1f}%) "
                    f"geom={self._action_stats['geometry_ok']}({pct('geometry_ok'):.1f}%) "
                    f"len={self._action_stats['length_ok']}({pct('length_ok'):.1f}%)"
                )
                print(
                    f"[QwenNav][action-dist][{mode}] batches={self._action_stats_counter} "
                    f"decisions={self._action_stats['total']} | "
                    f"fwd={self._action_stats['forward']}({pct('forward'):.1f}%) "
                    f"left={self._action_stats['left']}({pct('left'):.1f}%) "
                    f"right={self._action_stats['right']}({pct('right'):.1f}%) "
                    f"behind={self._action_stats['behind']}({pct('behind'):.1f}%) "
                    f"backtrack={self._action_stats['backtrack']}({pct('backtrack'):.1f}%) "
                    f"stop={self._action_stats['stop']}({pct('stop'):.1f}%) "
                    f"parse_fail={self._action_stats['parse_fail']}({pct('parse_fail'):.1f}%)"
                    f"{struct_str}{gate_str}",
                    flush=True,
                )
                self._action_stats_counter = 0
                for k in self._action_stats:
                    self._action_stats[k] = 0

        # The map has already consumed this primitive's RGB-D observation.
        visualization_started = time.perf_counter()
        try:
            for env_i in runtime_obs_by_env:
                self._save_lavira_map_snapshot(env_i)
                self._write_lavira_projection_audit(
                    env_i,
                    runtime_obs_by_env[env_i],
                    actions[env_i],
                    is_new_decision=is_decision_per_env[env_i],
                    prompt_text=prompt_by_env.get(env_i) if is_decision_per_env[env_i] else None,
                    model_output=raw_decoded_by_env.get(env_i) if is_decision_per_env[env_i] else None,
                )
        finally:
            self._profile_add(
                "visualization_audit",
                time.perf_counter() - visualization_started,
                items=len(runtime_obs_by_env),
            )

        # The next observation is the post-action frame; action-aware history
        # sampling associates it with the primitive returned in this call.
        for env_i, action in enumerate(actions):
            self._get_cache(env_i).last_action = int(action)

        # Build output tensor + diagnostics
        action_t = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)  # (N, 1)

        parse_ok = np.array(
            [self._get_cache(i).last_parse_ok for i in range(num_envs)],
            dtype=np.float32,
        )
        bboxes = [self._get_cache(i).last_bbox for i in range(num_envs)]
        points = [self._get_cache(i).last_point for i in range(num_envs)]

        diagnostics = {
            "parse_ok":     parse_ok,
            "bboxes":       bboxes,
            "actions":      action_t,
            "info": {
                "policy": "qwen_nav",
                "errors": [self._get_cache(i).last_err for i in range(num_envs)],
            },
            "prev_values":  None,
        }
        # Merged-schema extra diagnostics (logged when prompt_style=lavira_merged)
        if self.prompt_style in ("lavira_merged", "lavira_waypoint", "canonical_v1"):
            diagnostics["action_types"] = [
                (parsed_actions[i].action_type if i in parsed_actions else "")
                for i in range(num_envs)
            ]
        if self.is_waypoint_style:
            def _gs(i: int, key: str, default):
                return grounded_sam_diag.get(i, {}).get(key, default)

            def _projection(i: int) -> dict:
                controller = self._get_lavira_runtime(i)
                audit = getattr(controller, "audit_projection", None)
                return audit if isinstance(audit, dict) else {}

            diagnostics.update({
                "point_2d": points,
                "waypoint_id": [
                    self._get_cache(i).last_waypoint_id for i in range(num_envs)
                ],
                "reasoning_plan_action": [
                    self._get_cache(i).last_reasoning_plan_action for i in range(num_envs)
                ],
                "reasoning_bbox_point": [
                    self._get_cache(i).last_reasoning_bbox_point for i in range(num_envs)
                ],
                "backtrack_valid": np.array(
                    [self._get_cache(i).last_backtrack_valid for i in range(num_envs)],
                    dtype=np.float32,
                ),
                "target": [self._get_cache(i).last_target for i in range(num_envs)],
                "grounded_sam_enabled": np.array(
                    [float(_gs(i, "enabled", self.grounded_sam_enabled)) for i in range(num_envs)],
                    dtype=np.float32,
                ),
                "grounded_sam_used": np.array(
                    [float(_gs(i, "used", False)) for i in range(num_envs)],
                    dtype=np.float32,
                ),
                "grounded_sam_detected": np.array(
                    [float(_gs(i, "detected", False)) for i in range(num_envs)],
                    dtype=np.float32,
                ),
                "grounded_sam_target": [
                    _gs(i, "target", self._get_cache(i).last_target) for i in range(num_envs)
                ],
                "grounded_sam_label": [
                    _gs(i, "label", "") for i in range(num_envs)
                ],
                "grounded_sam_confidence": np.array(
                    [float(_gs(i, "confidence", 0.0)) for i in range(num_envs)],
                    dtype=np.float32,
                ),
                "grounded_sam_fallback_reason": [
                    _gs(i, "fallback_reason", "") for i in range(num_envs)
                ],
                "grounded_sam_point_2d": [
                    _gs(i, "point_2d", None) for i in range(num_envs)
                ],
                "grounded_sam_bbox_2d": [
                    _gs(i, "bbox_2d", None) for i in range(num_envs)
                ],
                "projection_depth_exhausted": np.array(
                    [
                        float(
                            _projection(i).get("accepted_reason")
                            == "source_depth_exhausted"
                        )
                        for i in range(num_envs)
                    ],
                    dtype=np.float32,
                ),
                "projection_backoff_count": np.array(
                    [
                        float(_projection(i).get("backoff_count", 0) or 0)
                        for i in range(num_envs)
                    ],
                    dtype=np.float32,
                ),
            })

        if self.collect_forward_inputs:
            # Aggregate per_env → batch tensors: (N, ...)
            zero_lp = torch.zeros(self.max_new_tokens)
            batch_lp = torch.stack(
                [per_env_logprobs.get(i, zero_lp) for i in range(num_envs)], dim=0
            )  # (N, max_new_tokens)

            # Aggregate forward_inputs: merge dict-of-(1,*) → dict-of-(N,*)
            # IMPORTANT: even when all envs are dormant (per_env_forward_inputs
            # is empty), we MUST emit a fully-populated merged_fi with N blank
            # entries so the trajectory's stacked forward_inputs keeps the same
            # T dimension as prev_logprobs.  Otherwise RLinf's
            # process_nested_dict_for_train hits a shape mismatch (forward_inputs
            # tensors have fewer steps than prev_logprobs).
            blank_fi = self._make_blank_forward_inputs()
            keys = list(blank_fi.keys()) if not per_env_forward_inputs else \
                   list(next(iter(per_env_forward_inputs.values())).keys())
            merged_fi = {}
            for k in keys:
                tensors = [
                    per_env_forward_inputs.get(i, blank_fi)[k] for i in range(num_envs)
                ]
                merged_fi[k] = torch.cat(tensors, dim=0)  # (N, *)

            diagnostics["prev_logprobs"] = batch_lp     # (N, max_new_tokens)
            diagnostics["forward_inputs"] = merged_fi
        else:
            diagnostics["prev_logprobs"] = torch.zeros(num_envs, 1)
            diagnostics["forward_inputs"] = {}

        # Signal which envs made a new LLM inference decision this step.
        # Non-decision envs (macro replay or dormant) contribute blank forward_inputs
        # and zero logprobs above, so downstream GRPO training safely ignores them.
        diagnostics["is_decision"] = torch.tensor(is_decision_per_env, dtype=torch.bool)  # (N,)

        self._profile_add(
            "policy_total",
            time.perf_counter() - profile_started,
            items=num_envs,
        )
        self._profile_flush_if_due()
        return action_t, diagnostics

    def _make_blank_forward_inputs(self) -> dict[str, torch.Tensor]:
        """Return a zero-filled forward_inputs dict (shape 1, *) for inactive envs."""
        pad_id = getattr(self.processor, "tokenizer", None)
        pad_id = (pad_id.pad_token_id if pad_id else None) or 0
        result = {
            "input_ids":         torch.full((1, self._prompt_len), pad_id, dtype=torch.long),
            "attention_mask":    torch.zeros(1, self._prompt_len, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(1, self._prompt_len, dtype=torch.long),
            "pixel_values":      torch.zeros(1, self._total_patches, self._pixel_value_dim),
            "image_grid_thw":    torch.zeros(1, self._n_images_fixed, 3, dtype=torch.long),
            "response_ids":      torch.full((1, self.max_new_tokens), pad_id, dtype=torch.long),
            "response_mask":     torch.zeros(1, self.max_new_tokens, dtype=torch.bool),
            "ppo_token_loss_mask": torch.zeros(1, self.max_new_tokens, dtype=torch.bool),
        }
        if self.termination_shadow_enabled:
            result.update({
                "termination_shadow_input_ids": torch.full(
                    (1, self._prompt_len), pad_id, dtype=torch.long
                ),
                "termination_shadow_attention_mask": torch.zeros(
                    1, self._prompt_len, dtype=torch.long
                ),
                "termination_shadow_mm_token_type_ids": torch.zeros(
                    1, self._prompt_len, dtype=torch.long
                ),
                "termination_shadow_pixel_values": torch.zeros(
                    1, self._total_patches, self._pixel_value_dim
                ),
                "termination_shadow_image_grid_thw": torch.zeros(
                    1, self._n_images_fixed, 3, dtype=torch.long
                ),
                "termination_shadow_valid": torch.zeros(1, 1, dtype=torch.bool),
            })
        return result

    # ------------------------------------------------------------------
    # BasePolicy interface — training
    # ------------------------------------------------------------------

    def _termination_shadow_forward(
        self, forward_inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Evaluate the factorized termination prompt without affecting policy.

        Returns the binary judge probability and the exact final hidden vector
        presented to the LM head. The caller owns persistence and oracle labels.
        """
        dev = self._model_device
        input_ids = forward_inputs["termination_shadow_input_ids"].to(dev)
        attention_mask = forward_inputs[
            "termination_shadow_attention_mask"
        ].to(dev)
        pixel_values = forward_inputs[
            "termination_shadow_pixel_values"
        ].to(dev)
        image_grid_thw = forward_inputs[
            "termination_shadow_image_grid_thw"
        ].to(dev)
        mm_token_type_ids = forward_inputs.get(
            "termination_shadow_mm_token_type_ids"
        )
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(dev)

        # Decision trajectories can retain the per-env singleton inserted by
        # ChunkStepResult.  The ordinary teacher-forcing path consumes [B, ...];
        # normalize the shadow copy to the same contract before slicing images.
        def _remove_env_singleton(tensor: torch.Tensor, expected_dim: int) -> torch.Tensor:
            while tensor.dim() > expected_dim and tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)
            return tensor

        input_ids = _remove_env_singleton(input_ids, 2)
        attention_mask = _remove_env_singleton(attention_mask, 2)
        pixel_values = _remove_env_singleton(pixel_values, 3)
        image_grid_thw = _remove_env_singleton(image_grid_thw, 3)
        if mm_token_type_ids is not None:
            mm_token_type_ids = _remove_env_singleton(mm_token_type_ids, 2)
        shapes = {
            "input_ids": tuple(input_ids.shape),
            "attention_mask": tuple(attention_mask.shape),
            "pixel_values": tuple(pixel_values.shape),
            "image_grid_thw": tuple(image_grid_thw.shape),
        }
        if (
            input_ids.dim() != 2
            or attention_mask.dim() != 2
            or pixel_values.dim() != 3
            or image_grid_thw.dim() != 3
        ):
            raise RuntimeError(f"invalid termination shadow tensor ranks: {shapes}")
        if not (
            input_ids.shape[0]
            == attention_mask.shape[0]
            == pixel_values.shape[0]
            == image_grid_thw.shape[0]
        ):
            raise RuntimeError(f"termination shadow batch mismatch: {shapes}")

        pv_list = []
        grid_list = []
        for batch_idx in range(input_ids.shape[0]):
            grid_b = image_grid_thw[batch_idx]
            valid_rows = grid_b.prod(dim=-1) > 0
            n_valid_images = int(valid_rows.sum().item())
            if n_valid_images:
                n_valid_patches = int(
                    grid_b[valid_rows].prod(dim=-1).sum().item()
                )
                pixels_b = pixel_values[batch_idx, :n_valid_patches, :]
                if pixels_b.shape[0] != n_valid_patches:
                    raise RuntimeError(
                        "termination shadow pixel/grid mismatch: "
                        f"sample={batch_idx} expected_patches={n_valid_patches} "
                        f"available_patches={pixel_values.shape[1]} shapes={shapes}"
                    )
                pv_list.append(pixels_b)
                grid_list.append(grid_b[valid_rows])

        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "output_hidden_states": True,
        }
        if pv_list:
            model_kwargs["pixel_values"] = (
                torch.cat(pv_list, dim=0).to(self._dtype).contiguous().clone()
            )
            model_kwargs["image_grid_thw"] = (
                torch.cat(grid_list, dim=0).contiguous().clone()
            )
        if mm_token_type_ids is not None:
            model_kwargs["mm_token_type_ids"] = mm_token_type_ids

        flat_pixels_pre_shape = tuple(model_kwargs["pixel_values"].shape)
        flat_pixels_pre_numel = model_kwargs["pixel_values"].numel()
        try:
            with torch.amp.autocast("cuda", dtype=self._dtype):
                outputs = self.model(**model_kwargs)
        except RuntimeError as exc:
            flat_pixels = model_kwargs.get("pixel_values")
            flat_grid = model_kwargs.get("image_grid_thw")
            raise RuntimeError(
                "termination shadow model forward failed: "
                f"input_shapes={shapes} "
                f"flat_pixels_pre_shape={flat_pixels_pre_shape} "
                f"flat_pixels_pre_numel={flat_pixels_pre_numel} "
                f"flat_pixels_shape={tuple(flat_pixels.shape) if flat_pixels is not None else None} "
                f"flat_pixels_numel={flat_pixels.numel() if flat_pixels is not None else 0} "
                f"flat_grid={flat_grid.detach().cpu().tolist() if flat_grid is not None else None}"
            ) from exc
        if not outputs.hidden_states:
            raise RuntimeError("termination shadow model returned no hidden states")
        hidden = outputs.hidden_states[-1]

        final_logits = outputs.logits[:, -1, :].float()
        pair_logits = torch.stack(
            [
                final_logits[:, self._termination_shadow_false_token_id],
                final_logits[:, self._termination_shadow_true_token_id],
            ],
            dim=-1,
        )
        probability = pair_logits.softmax(dim=-1)[:, 1]
        return {
            "termination_shadow_probability": probability,
            "termination_shadow_prediction": probability >= 0.5,
            "termination_shadow_hidden": hidden[:, -1, :].float(),
        }

    def default_forward(
        self,
        forward_inputs: Optional[dict] = None,
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> dict:
        """
        Teacher-forcing forward for GRPO training.

        forward_inputs keys (all shape (B, *)):
          input_ids       (B, prompt_len)
          attention_mask  (B, prompt_len)
          pixel_values    (B, max_patches, C)
          image_grid_thw  (B, max_images, 3)
          response_ids    (B, max_new_tokens)
          response_mask   (B, max_new_tokens)   — True where response is valid

        Returns:
          logprobs (B, max_new_tokens)  — per-token log_probs (0 for padded positions)
          entropy  (B, max_new_tokens)  — per-token entropy (0 for padded positions)

        Note on train/eval mode:
          Qwen3.5's hybrid architecture uses SDPA for standard attention layers.
          PyTorch's SDPA selects different backends (flash / efficient / math) based on
          tensor properties AND training mode, which can cause numerically different
          float accumulation in BF16. Since (a) all dropout is 0.0 and (b) the
          teacher-forcing path is the same regardless of mode, we force train() here so
          that recompute (called under model.eval() by fsdp_actor_worker) and the PPO
          forward pass both use the same SDPA backend → identical logprobs → ratio=1.0.
        """
        if forward_inputs is None:
            raise ValueError("forward_inputs is required for default_forward")

        # Force train mode for numerical consistency: SDPA backend is mode-sensitive
        # even when all dropout rates are 0. Since lora_dropout=0 and attention_dropout=0,
        # train mode is fully deterministic and matches the PPO forward path exactly.
        _was_training = self.model.training
        if not _was_training:
            self.model.train()

        if kwargs.get("termination_shadow_only", False):
            try:
                return self._termination_shadow_forward(forward_inputs)
            finally:
                if not _was_training:
                    self.model.eval()

        dev = self._model_device
        bsz = forward_inputs["input_ids"].shape[0]

        input_ids      = forward_inputs["input_ids"].to(dev)       # (B, prompt_len)
        attn_mask      = forward_inputs["attention_mask"].to(dev)  # (B, prompt_len)
        pixel_values   = forward_inputs["pixel_values"].to(dev)    # (B, max_patches, C)
        image_grid_thw = forward_inputs["image_grid_thw"].to(dev)  # (B, max_images, 3)
        response_ids   = forward_inputs["response_ids"].to(dev)    # (B, max_new_tokens)
        response_mask  = forward_inputs["response_mask"].to(dev)   # (B, max_new_tokens)
        mm_token_type_ids = forward_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(dev)           # (B, prompt_len)

        max_resp = response_ids.shape[1]
        prompt_len = input_ids.shape[1]

        # Concat prompt + response for teacher-forcing
        full_ids = torch.cat([input_ids, response_ids], dim=1)     # (B, full_len)
        resp_attn = response_mask.long()
        full_attn = torch.cat([attn_mask, resp_attn], dim=1)       # (B, full_len)

        # mm_token_type_ids: extend with zeros (text) for the response tokens
        if mm_token_type_ids is not None:
            resp_type_ids = torch.zeros(bsz, max_resp, dtype=mm_token_type_ids.dtype, device=dev)
            full_mm_type_ids = torch.cat([mm_token_type_ids, resp_type_ids], dim=1)

        # Flatten pixel_values per batch: Qwen2.5-VL expects (total_patches, C).
        # For each sample, only contribute pv rows that correspond to VALID
        # image_grid_thw entries.  blank/dormant samples have all-zero grids,
        # so they contribute 0 patches AND 0 grids — keeping pv ↔ grid ↔
        # input_ids' image-token count consistent.  Otherwise an extra batch of
        # patches with no matching image tokens triggers NaN in attention.
        pv_list = []
        grid_list = []
        for b in range(bsz):
            grid_b = image_grid_thw[b]                          # (max_images, 3)
            valid_rows = (grid_b.prod(dim=-1) > 0)
            n_valid_images = int(valid_rows.sum().item())
            if n_valid_images > 0:
                # Derive the raw patch count from the processor-provided grid.
                # Actor workers never call predict_action_batch(), so their
                # rollout-only _patches_per_image cache is intentionally unset.
                n_valid_patches = int(
                    grid_b[valid_rows].prod(dim=-1).sum().item()
                )
                if n_valid_patches > pixel_values.shape[1]:
                    raise ValueError(
                        "actor visual replay grid exceeds stored pixel capacity: "
                        f"sample={b}, required={n_valid_patches}, "
                        f"capacity={pixel_values.shape[1]}"
                    )
                pv_list.append(pixel_values[b, :n_valid_patches, :])
                grid_list.append(grid_b[valid_rows])
            # else: blank sample — skip pv and grid entirely

        if pv_list:
            flat_pv = torch.cat(pv_list, dim=0)        # (sum_valid_patches, C)
            flat_grid = torch.cat(grid_list, dim=0)    # (sum_valid_images, 3)
        else:
            flat_pv = pixel_values.new_zeros((0, pixel_values.shape[-1]))
            flat_grid = image_grid_thw.new_zeros((0, 3))

        model_kwargs = {
            "input_ids":      full_ids,
            "attention_mask": full_attn,
            "use_cache":      False,
        }
        if flat_pv.shape[0] > 0:
            model_kwargs["pixel_values"]    = flat_pv.to(self._dtype)
            model_kwargs["image_grid_thw"]  = flat_grid
        if mm_token_type_ids is not None:
            model_kwargs["mm_token_type_ids"] = full_mm_type_ids

        with torch.amp.autocast("cuda", dtype=self._dtype):
            outputs = self.model(**model_kwargs)

        logits = outputs.logits                          # (B, full_len, V)

        # Response logits: positions prompt_len-1 .. prompt_len+max_resp-2
        resp_logits = logits[:, prompt_len - 1: prompt_len - 1 + max_resp, :]
        # (B, max_resp, V)

        log_probs = F.log_softmax(resp_logits.float(), dim=-1)     # (B, max_resp, V)
        token_logprobs = log_probs.gather(
            -1, response_ids.unsqueeze(-1)
        ).squeeze(-1)                                               # (B, max_resp)

        # Zero out padded response positions; guard against NaN×0=NaN (IEEE 754)
        token_logprobs = token_logprobs * response_mask.float()
        token_logprobs = torch.nan_to_num(token_logprobs, nan=0.0, posinf=0.0, neginf=-100.0)

        out = {"logprobs": token_logprobs}

        if compute_entropy:
            probs = log_probs.exp()
            token_entropy = -(probs * log_probs).sum(dim=-1)       # (B, max_resp)
            token_entropy = token_entropy * response_mask.float()
            out["entropy"] = token_entropy

        if compute_values:
            out["values"] = None  # value head not implemented (add_value_head=False)

        if kwargs.get("compute_termination_shadow", False):
            required_key = "termination_shadow_input_ids"
            if required_key not in forward_inputs:
                raise ValueError(
                    "compute_termination_shadow=True requires shadow inputs"
                )
            out.update(self._termination_shadow_forward(forward_inputs))

        if not _was_training:
            self.model.eval()

        return out

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Forward type {forward_type} not supported")

    # ------------------------------------------------------------------
    # FSDP / gradient-checkpointing hooks (delegate to inner HF model)
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, **kwargs):
        if self.model is not None and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if self.model is not None and hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        if self.model is not None and hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(cfg: DictConfig, torch_dtype=None) -> QwenNavPolicy:
    return QwenNavPolicy(cfg)

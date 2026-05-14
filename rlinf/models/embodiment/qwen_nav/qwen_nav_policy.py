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

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .prompts import SYSTEM_PROMPT, build_user_content_text, expected_image_count
from .action_parser import ParsedAction, parse_lavira_json, ACTION_STOP, ACTION_PARSE_FAIL


# ---------------------------------------------------------------------------
# Per-env episode cache
# ---------------------------------------------------------------------------

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
        "history_images", "pending_actions", "step_count",
        "last_parse_ok", "last_err", "last_bbox",
    )

    def __init__(self):
        self.history_images: list[Image.Image] = []
        self.pending_actions: list[int] = []
        self.step_count: int = 0
        self.last_parse_ok: bool = True
        self.last_err: Optional[str] = None
        self.last_bbox: Optional[list[float]] = None

    def reset(self):
        self.history_images.clear()
        self.pending_actions.clear()
        self.step_count = 0
        self.last_parse_ok = True
        self.last_err = None
        self.last_bbox = None

    def push_image(self, img: Image.Image):
        self.history_images.append(img)
        self.step_count += 1


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


# ---------------------------------------------------------------------------
# QwenNavPolicy
# ---------------------------------------------------------------------------

class QwenNavPolicy(nn.Module, BasePolicy):
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
        self.use_4dir = bool(getattr(cfg, "use_4dir", True))
        self.chunk_size = int(getattr(cfg, "chunk_size", 4))
        # Training-mode flag: when True, compute prev_logprobs + forward_inputs
        # and disable macro-action buffering (pending_actions).
        self.collect_forward_inputs = bool(getattr(cfg, "collect_forward_inputs", False))
        # When True, skip teacher-forcing logprobs at rollout time.
        # Safe only when actor.recompute_prev_logprobs=True; actor will overwrite
        # prev_logprobs with its own eval forward before PPO training.
        self.skip_rollout_logprobs = bool(getattr(cfg, "skip_rollout_logprobs", False))
        # Fixed image resolution fed to the VLM processor.
        # All images (env renders + history + blank pads) are resized to (W, H)
        # before tokenisation so that pixel_values shape is consistent across
        # init/rollout/training.  Must match blank images used in _init_padding_params.
        _img_sz = getattr(cfg, "image_size", [448, 448])
        self.image_size: tuple[int, int] = (int(_img_sz[0]), int(_img_sz[1]))  # (W, H)

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
        self._n_images_fixed: int = 0  # history_max_frames + n_current_views
        self._patches_per_image: int = 0
        self._grid_h: int = 0
        self._grid_w: int = 0

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
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self._trust_remote,
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
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
        self._n_images_fixed = self.history_max_frames + n_current

        blank = Image.new("RGB", self.image_size, (127, 127, 127))
        dummy_images = [blank] * self._n_images_fixed

        # Build fake step indices for history
        dummy_hist_idx = list(range(self.history_max_frames))
        user_text = build_user_content_text(
            instruction="go to the elevator",
            history_step_indices=dummy_hist_idx,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
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
            self._grid_h = grid[0][1].item()
            self._grid_w = grid[0][2].item()
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
            cache.reset()

    def _get_cache(self, env_id: int) -> _HistoryCache:
        return self._per_env_cache.setdefault(env_id, _HistoryCache())

    # ------------------------------------------------------------------
    # Prompt building (per env)
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        instruction: str,
        history_imgs: list[Image.Image],
        history_step_indices: list[int],
        current_view: Image.Image,
        extra_views: Optional[list[Image.Image]] = None,
        pad_history_to: Optional[int] = None,
    ) -> tuple[str, list[Image.Image]]:
        """
        Build Qwen-VL chat-template prompt + ordered image list.

        When `pad_history_to` is set (training mode), history is padded to
        that exact length with blank images so tensor shapes are consistent.

        Image order: history_imgs (chronological) + 4-dir current views (F/L/R/B).
        """
        if pad_history_to is not None and len(history_imgs) < pad_history_to:
            pad_n = pad_history_to - len(history_imgs)
            # Prepend blank images and prepend dummy step indices
            history_imgs = [self._blank_image] * pad_n + list(history_imgs)
            history_step_indices = [-1] * pad_n + list(history_step_indices)

        user_text = build_user_content_text(
            instruction=instruction,
            history_step_indices=history_step_indices,
        )

        if self.use_4dir:
            if extra_views is not None and len(extra_views) == 3:
                current_views = [current_view] + list(extra_views)
            else:
                current_views = [current_view, current_view, current_view, current_view]
        else:
            current_views = [current_view]

        ordered_images = list(history_imgs) + current_views

        n_expected = expected_image_count(len(history_imgs), has_4dir=self.use_4dir)
        if len(ordered_images) != n_expected:
            raise RuntimeError(
                f"image count mismatch: built {len(ordered_images)}, "
                f"prompt expects {n_expected}"
            )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in ordered_images]
                    + [{"type": "text", "text": user_text}]
                ),
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
        # Resize all images to fixed resolution so pixel_values shape is
        # consistent between _init_padding_params and actual rollout/training.
        ordered_images = [img.resize(self.image_size) for img in ordered_images]
        return prompt_text, ordered_images

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
        """
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

    def _build_forward_inputs_for_env(
        self,
        inputs_single: "BatchEncoding",   # processor output for 1 sample
        response_ids: torch.Tensor,        # (1, resp_len)
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

        # pixel_values: pad to (max_total_patches, pixel_value_dim)
        pv = inputs_single["pixel_values"]             # (actual_patches, C)
        max_pv = self._total_patches
        if pv.shape[0] < max_pv:
            pv = F.pad(pv, (0, 0, 0, max_pv - pv.shape[0]))
        else:
            pv = pv[:max_pv]

        # image_grid_thw: pad to (max_images, 3)
        if "image_grid_thw" in inputs_single:
            grid = inputs_single["image_grid_thw"]     # (actual_n_images, 3)
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

        return {
            "input_ids":          padded_ids.unsqueeze(0),     # (1, prompt_len)
            "attention_mask":     padded_attn.unsqueeze(0),     # (1, prompt_len)
            "mm_token_type_ids":  padded_ttids.unsqueeze(0),    # (1, prompt_len)
            "pixel_values":       pv.unsqueeze(0),              # (1, max_patches, C)
            "image_grid_thw":     grid.unsqueeze(0),            # (1, max_images, 3)
            "response_ids":       padded_resp.unsqueeze(0),     # (1, max_new_tokens)
            "response_mask":      resp_mask.unsqueeze(0),       # (1, max_new_tokens)
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

        if self.collect_forward_inputs and not self._padding_initialized:
            self._init_padding_params()

        main_images = env_obs.get("main_images")
        instructions = env_obs.get("task_descriptions") or []
        states = env_obs.get("states")
        extra_views_t = env_obs.get("extra_view_images")

        if main_images is None:
            raise ValueError("env_obs missing 'main_images'")

        rgb_np = (
            main_images.cpu().numpy()
            if isinstance(main_images, torch.Tensor)
            else np.asarray(main_images)
        )
        num_envs = rgb_np.shape[0]

        extras_np = None
        if extra_views_t is not None:
            extras_np = (
                extra_views_t.cpu().numpy()
                if isinstance(extra_views_t, torch.Tensor)
                else np.asarray(extra_views_t)
            )
            if extras_np.shape[0] != num_envs or extras_np.shape[1] != 3:
                extras_np = None

        if not instructions:
            instructions = [""] * num_envs

        # Episode reset detection (elapsed_steps == 0)
        if states is not None:
            elapsed = (
                states.cpu().numpy()
                if isinstance(states, torch.Tensor)
                else np.asarray(states)
            )
            for i in range(num_envs):
                if instructions[i] and float(elapsed[i, 0]) == 0.0:
                    self.reset_env_cache([i])

        # Push current image into per-env history
        current_pils: list[Optional[Image.Image]] = [None] * num_envs
        for i in range(num_envs):
            if not instructions[i]:
                continue
            current_pils[i] = _to_pil(rgb_np[i])
            self._get_cache(i).push_image(current_pils[i])

        # Phase 1: dispatch buffered actions (eval mode only)
        actions: list[Optional[int]] = [None] * num_envs
        need_infer: list[int] = []
        for i in range(num_envs):
            if not instructions[i]:
                actions[i] = ACTION_PARSE_FAIL  # dormant env, no instruction
                continue
            cache = self._get_cache(i)
            # In training mode, always run inference (no macro buffering)
            if not self.collect_forward_inputs and cache.pending_actions:
                actions[i] = cache.pending_actions.pop(0)
            else:
                need_infer.append(i)

        # Phase 2: build prompts + run batched generate
        per_env_forward_inputs: dict[int, dict] = {}
        per_env_logprobs: dict[int, torch.Tensor] = {}
        per_env_entropy: dict[int, torch.Tensor] = {}

        if need_infer:
            prompts: list[str] = []
            img_lists: list[list[Image.Image]] = []
            pad_h = self.history_max_frames if self.collect_forward_inputs else None

            for env_i in need_infer:
                cache = self._get_cache(env_i)
                past_imgs = cache.history_images[:-1]
                sampled, sampled_idx = _sample_history(
                    past_imgs,
                    every_k=self.history_every_k,
                    max_frames=self.history_max_frames,
                )
                extra_views = None
                if extras_np is not None:
                    extra_views = [_to_pil(extras_np[env_i, k]) for k in range(3)]

                prompt_text, ordered_imgs = self._build_prompt(
                    instruction=instructions[env_i],
                    history_imgs=sampled,
                    history_step_indices=sampled_idx,
                    current_view=current_pils[env_i],
                    extra_views=extra_views,
                    pad_history_to=pad_h,
                )
                prompts.append(prompt_text)
                img_lists.append(ordered_imgs)

            decoded, gen_ids_list = self._batch_generate(prompts, img_lists)

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

                    fi = self._build_forward_inputs_for_env(single_inputs, resp_ids)
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

            for env_i, text in zip(need_infer, decoded):
                parsed: ParsedAction = parse_lavira_json(text)
                cache = self._get_cache(env_i)
                cache.last_parse_ok = parsed.ok
                cache.last_err = parsed.err
                cache.last_bbox = parsed.bbox

                act_seq = parsed.actions if parsed.actions else [ACTION_STOP]
                # Use ACTION_PARSE_FAIL (4) as sentinel so genark_env can
                # distinguish intentional stop from parse-failure stop, enabling
                # format_reward computation without any side-channel.
                actions[env_i] = ACTION_PARSE_FAIL if not parsed.ok else act_seq[0]
                # Macro buffering only in eval mode
                if not self.collect_forward_inputs and len(act_seq) > 1:
                    cache.pending_actions.extend(act_seq[1:])

        # Build output tensor + diagnostics
        action_t = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)  # (N, 1)

        parse_ok = np.array(
            [self._get_cache(i).last_parse_ok for i in range(num_envs)],
            dtype=np.float32,
        )
        bboxes = [self._get_cache(i).last_bbox for i in range(num_envs)]

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

        return action_t, diagnostics

    def _make_blank_forward_inputs(self) -> dict[str, torch.Tensor]:
        """Return a zero-filled forward_inputs dict (shape 1, *) for inactive envs."""
        pad_id = getattr(self.processor, "tokenizer", None)
        pad_id = (pad_id.pad_token_id if pad_id else None) or 0
        return {
            "input_ids":         torch.full((1, self._prompt_len), pad_id, dtype=torch.long),
            "attention_mask":    torch.zeros(1, self._prompt_len, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(1, self._prompt_len, dtype=torch.long),
            "pixel_values":      torch.zeros(1, self._total_patches, self._pixel_value_dim),
            "image_grid_thw":    torch.zeros(1, self._n_images_fixed, 3, dtype=torch.long),
            "response_ids":      torch.full((1, self.max_new_tokens), pad_id, dtype=torch.long),
            "response_mask":     torch.zeros(1, self.max_new_tokens, dtype=torch.bool),
        }

    # ------------------------------------------------------------------
    # BasePolicy interface — training
    # ------------------------------------------------------------------

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
        """
        if forward_inputs is None:
            raise ValueError("forward_inputs is required for default_forward")

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
                # All real images same resolution → patches_per_image each.
                n_valid_patches = n_valid_images * self._patches_per_image
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

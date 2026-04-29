# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
UniNaVid policy for RLinf — wraps LlavaLlamaAttForCausalLM as a BasePolicy.

Batched inference (4-phase, ported from genark/model_server.py):
  Phase 2: one vision tower forward for ALL envs in the batch
  Phase 3: per-env feat_cache update + manual input_embeds construction
  Phase 4: one LLM generate() call per chunk (CHUNK_SIZE bound, OOM guard)
           uses _bypass_fwd to skip UniNaVid's multimodal path and go directly
           to model.model(inputs_embeds=...) + lm_head

chunk_size (config: model.chunk_size, default 8) controls the LLM batch size
per generate() call. Set lower to trade speed for memory.
"""

from __future__ import annotations

import re
import sys
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

_ACTION_WORDS = {"stop": 0, "forward": 1, "left": 2, "right": 3}
_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the "
    "current observation <image>. Your assigned task is: '{}'. "
    "Analyze this series of images to determine your next four actions. "
    "The predicted action should be one of the following: forward, left, right, or stop."
)


def _parse_action_list(text: str, max_actions: int = 2) -> list[int]:
    actions = []
    for word in text.lower().split():
        clean = re.sub(r"[^a-z]", "", word)
        if clean in _ACTION_WORDS:
            actions.append(_ACTION_WORDS[clean])
            if len(actions) >= max_actions:
                break
    return actions or [0]


# ---------------------------------------------------------------------------
# Per-env episode cache
# ---------------------------------------------------------------------------

class _EpisodeCache:
    def __init__(self):
        self.rgb_list: list[np.ndarray] = []
        self.new_rgb: list[np.ndarray] = []
        self.pending_actions: list[int] = []
        self.feat_cache = None
        self.long_feat_cache = None

    def reset(self):
        self.rgb_list.clear()
        self.new_rgb.clear()
        self.pending_actions.clear()
        self.feat_cache = None
        self.long_feat_cache = None

    def push_rgb(self, rgb: np.ndarray):
        self.rgb_list.append(rgb)
        self.new_rgb.append(rgb)


# ---------------------------------------------------------------------------
# UniNaVidPolicy
# ---------------------------------------------------------------------------

class UniNaVidPolicy(nn.Module, BasePolicy):

    _per_env_cache: dict[int, _EpisodeCache]

    def __init__(self, cfg: DictConfig):
        nn.Module.__init__(self)
        self.cfg = cfg

        model_path    = cfg.model_path
        uninavid_path = str(getattr(cfg, "uninavid_src_path", "/home/nvme03/lck/genark"))

        if uninavid_path not in sys.path:
            sys.path.insert(0, uninavid_path)

        from uninavid.mm_utils import get_model_name_from_path
        from uninavid.model.builder import load_pretrained_model

        model_name = get_model_name_from_path(model_path)
        _prev_dir = os.getcwd()
        os.chdir(uninavid_path)
        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = (
                load_pretrained_model(model_path, None, model_name)
            )
        finally:
            os.chdir(_prev_dir)

        self.model.eval()
        self.conv_mode     = str(getattr(cfg, "conv_mode",      "vicuna_v1"))
        self.max_new_tokens = int(getattr(cfg, "max_new_tokens", 1024))
        self.temperature    = float(getattr(cfg, "temperature",  0.5))
        self.do_sample      = bool(getattr(cfg, "do_sample",     True))
        # Max envs per LLM generate() call — lower to trade speed for VRAM
        self.chunk_size     = int(getattr(cfg, "chunk_size",     8))

        self._per_env_cache: dict[int, _EpisodeCache] = {}

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def reset_env_cache(self, env_indices: list[int]):
        for i in env_indices:
            if i not in self._per_env_cache:
                self._per_env_cache[i] = _EpisodeCache()
            else:
                self._per_env_cache[i].reset()
            if hasattr(self.model.get_model(), "initialize_online_inference_nav_feat_cache"):
                self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0
        self.model.config.run_type = "eval"

    def push_rgb(self, env_idx: int, rgb: np.ndarray):
        self._per_env_cache.setdefault(env_idx, _EpisodeCache()).push_rgb(rgb)

    # ------------------------------------------------------------------
    # Vision helpers
    # ------------------------------------------------------------------

    def _preprocess_frames(self, rgb_list: list[np.ndarray]) -> Optional[torch.Tensor]:
        """Stack RGB list and run image_processor → (T, C, H, W) on CUDA."""
        if not rgb_list:
            return None
        try:
            batch = np.stack(rgb_list)
        except ValueError:
            return None
        return self.image_processor.preprocess(
            batch, return_tensors="pt"
        )["pixel_values"].to(dtype=self.model.dtype, device="cuda")

    @staticmethod
    def _process_grid(vis_embed: torch.Tensor, grid_size: int) -> torch.Tensor:
        """Spatial pool vision features to grid_size × grid_size tokens."""
        T, P, C = vis_embed.shape
        H       = int(P ** 0.5)
        stride  = H // grid_size
        x = vis_embed.reshape(T, H, H, C).permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, kernel_size=stride, stride=stride, padding=0)
        return x.permute(0, 2, 3, 1).flatten(1, 2)  # (T, grid_size², C)

    def _get_nav_size(self) -> int:
        return {"grid:2": 4, "grid:4": 16, "mean": 1}.get(
            self.model.config.compress_type, 4
        )

    def _get_grid_size(self) -> int:
        ct = self.model.config.compress_type
        return int(ct.split("grid:")[-1]) if "grid:" in ct else 2

    # ------------------------------------------------------------------
    # Token / embed helpers
    # ------------------------------------------------------------------

    def _build_input_ids(self, instruction: str) -> torch.LongTensor:
        """Tokenise the navigation prompt with video/image special tokens."""
        from uninavid.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
        )
        from uninavid.conversation import conv_templates
        from uninavid.mm_utils import tokenizer_image_token

        prompt_q = _PROMPT_TEMPLATE.format(instruction)
        if self.model.config.mm_use_im_start_end:
            qs = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN
                  + DEFAULT_IM_END_TOKEN + "\n"
                  + prompt_q.replace("<image>", ""))
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt_q.replace("<image>", "")

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).cuda()

        special = {}
        for name in ["<video_special>", "</video_special>",
                     "<image_special>", "</image_special>",
                     "[Navigation]", "<image_sep>"]:
            tid = self.tokenizer.convert_tokens_to_ids(name)
            special[name] = torch.tensor([tid], device="cuda")

        indices = torch.where(token_prompt == -200)[0]
        parts = []
        while indices.numel() > 0:
            idx = indices[0]
            parts += [
                token_prompt[:idx],
                special["<video_special>"],
                special["<image_sep>"],
                token_prompt[idx: idx + 1],
                special["</video_special>"],
                special["<image_special>"],
                special["</image_special>"],
                special["[Navigation]"],
            ]
            token_prompt = token_prompt[idx + 1:]
            indices = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            parts.append(token_prompt)
        return torch.cat(parts, dim=0).unsqueeze(0)

    def _build_input_embeds_batch(
        self,
        instruction: str,
        compressed_hist: torch.Tensor,
        lengths_list: list[int],
        vis_cur_proj: torch.Tensor,
        model_inner,
    ) -> torch.Tensor:
        """Build a single env's input_embeds tensor from vision + text tokens.

        Mirrors model_server.py:277-309: text tokens are embedded, image token
        position is replaced with compressed history + current-frame tokens.
        """
        from uninavid.constants import IMAGE_TOKEN_INDEX

        cur_ids = self._build_input_ids(instruction)[0]   # (seq_len,)
        img_pos = torch.where(cur_ids == IMAGE_TOKEN_INDEX)[0]
        if img_pos.numel() == 0:
            return model_inner.embed_tokens(cur_ids)

        its     = img_pos[0].item()
        sep_emb = model_inner.embed_tokens(cur_ids[its - 1: its])

        parts = [model_inner.embed_tokens(cur_ids[:its])]
        vid_idx = 0
        for k, n_tok in enumerate(lengths_list):
            parts.append(compressed_hist[vid_idx: vid_idx + n_tok])
            if k < len(lengths_list) - 1:
                parts.append(sep_emb)
            vid_idx += n_tok
        parts.append(model_inner.embed_tokens(cur_ids[its + 1: its + 3]))
        parts.append(vis_cur_proj.squeeze(0))
        parts.append(model_inner.embed_tokens(cur_ids[its + 3:]))

        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    # Batched LLM generate (Phase 4)
    # ------------------------------------------------------------------

    def _batch_generate(self, input_embeds_list: list[torch.Tensor]) -> list[str]:
        """Left-pad embeds of different lengths, run generate() in chunks.

        Uses _bypass_fwd monkey-patch to skip UniNaVid's multimodal forward
        (which doesn't accept pre-built batched inputs_embeds) and go directly
        to the underlying language model. Mirrors model_server.py:311-402.
        """
        N = len(input_embeds_list)
        if N == 0:
            return []

        from uninavid.conversation import conv_templates, SeparatorStyle
        from uninavid.mm_utils import KeywordsStoppingCriteria

        conv     = conv_templates[self.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        _orig_fwd  = self.model.forward
        _model_ref = self.model

        def _bypass_fwd(
            input_ids=None, attention_mask=None, past_key_values=None,
            inputs_embeds=None, labels=None, use_cache=None,
            output_attentions=None, output_hidden_states=None,
            images=None, prompts=None, return_dict=None, **kw,
        ):
            rd  = return_dict          if return_dict          is not None else _model_ref.config.use_return_dict
            oa  = output_attentions    if output_attentions    is not None else _model_ref.config.output_attentions
            ohs = output_hidden_states if output_hidden_states is not None else _model_ref.config.output_hidden_states
            outputs = _model_ref.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=oa,
                output_hidden_states=ohs,
                return_dict=rd,
            )
            logits = _model_ref.lm_head(outputs[0])
            if not rd:
                return (None, logits) + outputs[1:]
            return CausalLMOutputWithPast(
                loss=None, logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        self.model.forward = _bypass_fwd
        all_texts: list[str] = []
        try:
            for chunk_start in range(0, N, self.chunk_size):
                chunk_embeds = input_embeds_list[chunk_start: chunk_start + self.chunk_size]
                chunk_N = len(chunk_embeds)

                max_len   = max(e.shape[0] for e in chunk_embeds)
                hidden    = chunk_embeds[0].shape[-1]
                batched   = torch.zeros(chunk_N, max_len, hidden,
                                        dtype=self.model.dtype, device="cuda")
                attn_mask = torch.zeros(chunk_N, max_len,
                                        dtype=torch.long, device="cuda")
                for i, emb in enumerate(chunk_embeds):
                    L = emb.shape[0]
                    batched[i, max_len - L:]   = emb
                    attn_mask[i, max_len - L:] = 1

                dummy_ids = torch.zeros(chunk_N, max_len,
                                        dtype=torch.long, device="cuda")
                stopping = KeywordsStoppingCriteria(
                    [stop_str], self.tokenizer, dummy_ids
                )

                print(f"[UniNaVid] _batch_generate: chunk_size={chunk_N} embeds_len={max_len}", flush=True)
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        inputs_embeds=batched,
                        attention_mask=attn_mask,
                        do_sample=self.do_sample,
                        temperature=self.temperature if self.do_sample else None,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=True,
                        stopping_criteria=[stopping],
                    )

                for i in range(chunk_N):
                    raw = self.tokenizer.decode(
                        output_ids[i], skip_special_tokens=True
                    ).strip()
                    if raw.endswith(stop_str):
                        raw = raw[: -len(stop_str)].strip()
                    all_texts.append(raw)

                torch.cuda.empty_cache()
        finally:
            self.model.forward = _orig_fwd

        return all_texts

    # ------------------------------------------------------------------
    # 4-phase batched inference (Phases 2-4)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _batch_infer(
        self, env_ids: list[int], all_instructions: list[str]
    ) -> list[list[int]]:
        """Run batched 4-phase inference for the given env indices.

        all_instructions: full list indexed 0..total_envs.
        Returns list[list[int]] of parsed action lists, indexed same as env_ids.
        """
        # Phase 2: gather new video frames, one vision tower forward for all envs
        video_tensors: list[torch.Tensor] = []
        frame_counts:  list[int]          = []
        valid_env_ids: list[int]          = []

        for i in env_ids:
            cache = self._per_env_cache.get(i)
            if cache and cache.new_rgb:
                vt = self._preprocess_frames(cache.new_rgb)
                if vt is not None:
                    video_tensors.append(vt)
                    frame_counts.append(vt.shape[0])
                    valid_env_ids.append(i)
                    cache.new_rgb.clear()

        if not video_tensors:
            return [[0]] * len(env_ids)

        inner     = self.model.get_model()
        nav_size  = self._get_nav_size()
        grid_size = self._get_grid_size()

        print(f"[UniNaVid] _batch_infer: n_envs={len(valid_env_ids)} vision_frames={sum(frame_counts)}", flush=True)
        all_frames = torch.cat(video_tensors, dim=0)          # (ΣTi, C, H, W)
        all_raw    = inner.get_vision_tower()(all_frames)     # (ΣTi, P, Dv)
        if (getattr(self.model.config, "mm_vision_select_feature", "") == "patch"
                and all_raw.shape[1] % 2 == 1):
            all_raw = all_raw[:, 1:]                           # strip CLS token

        # Phase 3: per-env feat_cache swap + input_embeds construction
        input_embeds_list: list[torch.Tensor] = []
        offset = 0
        for env_i, Ti in zip(valid_env_ids, frame_counts):
            cache     = self._per_env_cache[env_i]
            agent_raw = all_raw[offset: offset + Ti]

            vis_hist      = self._process_grid(agent_raw,     grid_size)
            vis_cur       = self._process_grid(agent_raw[-1:], 8)
            vis_hist_proj = inner.mm_projector(vis_hist)
            vis_cur_proj  = inner.mm_projector(vis_cur)

            # Swap in this env's feat_cache
            inner.feat_cache      = cache.feat_cache
            inner.long_feat_cache = cache.long_feat_cache
            inner.new_frames      = Ti

            old = inner.feat_cache
            inner.feat_cache = (
                torch.cat([old, vis_hist_proj], dim=0)
                if old is not None else vis_hist_proj
            )

            compressed, lengths = self.model.online_process_tensor(nav_size)

            embeds_i = self._build_input_embeds_batch(
                all_instructions[env_i], compressed, lengths, vis_cur_proj, inner
            )
            input_embeds_list.append(embeds_i)

            # Swap out
            cache.feat_cache      = inner.feat_cache
            cache.long_feat_cache = inner.long_feat_cache
            inner.feat_cache      = None
            inner.long_feat_cache = None
            inner.new_frames      = 0
            offset += Ti

        # Phase 4: batched LLM generate
        output_texts  = self._batch_generate(input_embeds_list)
        text_by_env   = dict(zip(valid_env_ids, output_texts))

        return [_parse_action_list(text_by_env.get(i, ""), 2) for i in env_ids]

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    def predict_action_batch(
        self,
        env_obs: dict[str, torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        if env_obs is None:
            raise ValueError("env_obs must be provided")

        main_images  = env_obs.get("main_images")
        instructions = env_obs.get("task_descriptions", None)

        if main_images is None:
            raise ValueError("env_obs missing 'main_images'")

        rgb_np   = main_images.cpu().numpy() if isinstance(main_images, torch.Tensor) \
                   else np.array(main_images)
        num_envs = rgb_np.shape[0]

        if instructions is None:
            instructions = ["navigate to the goal"] * num_envs

        # Reset cache on episode start (elapsed_steps == 0)
        # Skip dormant envs (empty instruction) — they have no real episode
        states = env_obs.get("states", None)
        if states is not None:
            elapsed_np = states.cpu().numpy() if isinstance(states, torch.Tensor) \
                         else np.array(states)
            for i in range(num_envs):
                if instructions[i] and float(elapsed_np[i, 0]) == 0.0:
                    self.reset_env_cache([i])

        # Push current RGB only for active envs (dormant have zero-frame obs)
        for i in range(num_envs):
            if instructions[i]:
                self.push_rgb(i, rgb_np[i])

        # Phase 1: separate envs with buffered actions from those needing inference.
        # Dormant envs (empty instruction) return STOP immediately, no inference.
        actions: list[Optional[int]] = [None] * num_envs
        need_infer: list[int] = []
        for i in range(num_envs):
            if not instructions[i]:
                actions[i] = 0
                continue
            cache = self._per_env_cache.get(i, _EpisodeCache())
            if cache.pending_actions:
                actions[i] = cache.pending_actions.pop(0)
            else:
                need_infer.append(i)

        # Phases 2-4: batched inference for envs without buffered actions
        if need_infer:
            batched_results = self._batch_infer(need_infer, instructions)
            for env_i, acts in zip(need_infer, batched_results):
                actions[env_i] = acts.pop(0)
                if acts:
                    self._per_env_cache[env_i].pending_actions.extend(acts)

        action_t = torch.tensor(actions, dtype=torch.long)

        result = {
            "prev_logprobs":  None,
            "prev_values":    None,
            "forward_inputs": {
                "main_images":      env_obs.get("main_images"),
                "instruction_ids":  env_obs.get("instruction_ids"),
                "instruction_mask": env_obs.get("instruction_mask"),
                "actions":          action_t,
            },
        }
        return action_t, result

    def default_forward(
        self,
        forward_inputs: Optional[dict] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        **kwargs,
    ):
        raise NotImplementedError(
            "UniNaVidPolicy.default_forward() not yet implemented (Step 3b)."
        )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Forward type {forward_type} not supported")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(cfg: DictConfig, torch_dtype=None) -> UniNaVidPolicy:
    return UniNaVidPolicy(cfg)

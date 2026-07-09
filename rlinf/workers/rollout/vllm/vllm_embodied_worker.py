# Copyright 2026 The RLinf Authors.
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
VLLMMultiStepEmbodiedWorker — vLLM rollout backend for embodied navigation.

Extends MultiStepRolloutWorker so that generate_one_epoch / predict / forward_inputs
are all reused unchanged.  Only _batch_generate inside QwenNavPolicy is replaced
by a vLLM call via the _vllm_generate_fn injection point (V1).

Architecture:
  - vLLM AsyncLLM engine runs in a *dedicated thread* (self._vllm_thread) with its
    own event loop (self._vllm_loop).  This avoids the "cannot run nested event loop"
    problem that would arise if we tried to await vLLM from inside generate_one_epoch's
    async context.
  - _vllm_generate_fn wraps the async vLLM call with asyncio.run_coroutine_threadsafe,
    making it synchronously callable from _batch_generate.
  - V2 (this file): vLLM loads weights from disk (same path as HF model).
  - V3 (future): override sync_model_from_actor to also push fresh weights into vLLM
    after each actor update.

Usage:
  Set rollout.backend: vllm_embodied in the config.  train_embodied_agent.py selects
  this worker class when that backend is configured.
"""

import asyncio
import threading
from typing import Optional

import torch
from omegaconf import DictConfig
from PIL import Image
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs.llm import TextPrompt
from vllm.sampling_params import SamplingParams
from vllm.utils.counter import Counter
from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMEngine

from rlinf.config import torch_dtype_from_precision
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class VLLMMultiStepEmbodiedWorker(MultiStepRolloutWorker):
    """
    Drop-in replacement for MultiStepRolloutWorker that generates tokens via vLLM.

    The HF model (self.hf_model) is still loaded for:
      - forward_inputs construction (HF processor → input_ids / pixel_values)
      - actor logprob recomputation (recompute_prev_logprobs=True path)
    Only model.generate() is replaced by the vLLM engine.
    """

    def init_worker(self):
        # Step 1: Load HF model + set up rank/channel infra (parent handles this).
        super().init_worker()

        # Step 2: Start dedicated event loop thread for vLLM async engine.
        self._vllm_loop = asyncio.new_event_loop()
        self._vllm_thread = threading.Thread(
            target=self._vllm_loop.run_forever, daemon=True, name="vllm-engine-loop"
        )
        self._vllm_thread.start()
        self._request_counter = Counter()

        # Step 3: Initialize vLLM engine in that thread (blocks until ready).
        init_future = asyncio.run_coroutine_threadsafe(
            self._init_vllm_engine(), self._vllm_loop
        )
        init_future.result(timeout=600)
        self.log_info("[VLLMEmbodied] vLLM engine initialized.")

        # Step 4: Inject sync generate fn into the *inner* QwenNavPolicy.
        # CRITICAL: self.hf_model is a PeftModel (is_lora=true), which does NOT
        # define __setattr__, so `self.hf_model._vllm_generate_fn = fn` lands in
        # the PeftModel.__dict__ — but _batch_generate reads self._vllm_generate_fn
        # on the *inner* QwenNavPolicy (a different object), so it would never see
        # it and would silently fall back to HF generation. Unwrap PEFT first.
        self._policy = self._unwrap_policy()
        self._policy._vllm_generate_fn = self._make_vllm_generate_fn()
        injected_ok = getattr(self._policy, "_vllm_generate_fn", None) is not None
        self.log_info(
            f"[VLLMEmbodied] _vllm_generate_fn injected into inner policy "
            f"({type(self._policy).__name__}); routed={injected_ok}."
        )

        # Force-skip the rollout-side teacher-forcing logprob forward. This is the
        # ONLY HF-model forward in the rollout path; with recompute_prev_logprobs=True
        # the actor recomputes prev_logprobs anyway, so the rollout-side logprob is
        # redundant work that gets overwritten. The yaml sets skip_rollout_logprobs=true
        # but it does NOT reach the policy (arrives False at runtime), so the model was
        # being forward'd every decision. Forcing it here both (a) makes opt1 (HF on CPU)
        # valid — that forward would otherwise run a 4B multimodal pass on CPU and hang —
        # and (b) speeds up rollout by removing the redundant forward.
        _prev_skip = getattr(self._policy, "skip_rollout_logprobs", None)
        self._policy.skip_rollout_logprobs = True
        self.log_info(
            f"[VLLMEmbodied] skip_rollout_logprobs forced True (was {_prev_skip}); "
            f"rollout HF-forward eliminated."
        )

        # Step 5 (opt1): the HF model is now REDUNDANT on GPU — vLLM does all
        # generation, and the rollout path never runs an HF forward
        # (skip_rollout_logprobs=True fills zeros; forward_inputs use the processor
        # only; logprobs are recomputed on the *actor's* separate model). The HF
        # model is only needed for the weight-sync merge, which runs fine on CPU.
        # Park it on CPU to free ~14 GiB on the collocated GPU 6/7 for the actor
        # train step + a higher vLLM gpu_memory_utilization. _model_device reads
        # next(self.model.parameters()).device dynamically, so CPU is transparent.
        self._vllm_asleep = False
        # opt2 sleep/offload is OFF by default — it corrupts vLLM's multimodal cache
        # across sleep/wake (mm_hash assertion). Memory is handled by opt1 (HF on CPU)
        # + a low gpu_memory_utilization instead. Gate kept for future re-enable.
        self._sleep_offload_enabled = bool(
            self.cfg.rollout.get("vllm_sleep_offload", False)
        )
        self.hf_model.to("cpu")
        self.torch_platform.empty_cache()
        self.log_info("[VLLMEmbodied] HF model parked on CPU (opt1: ~14 GiB freed).")

    def _vllm_run(self, coro, timeout: float = 600.0):
        """Run a vLLM-engine coroutine on the dedicated loop thread, block for result."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._vllm_loop)
        return fut.result(timeout=timeout)

    def _ensure_vllm_awake(self):
        """Best-effort wake of the vLLM engine (restore weights + KV after sleep).

        Fail-safe: if wake errors, log and continue — opt1 (HF on CPU) +
        gradient_checkpointing keep the actor train step within memory even if the
        engine never slept, so a sleep/wake hiccup must not kill the run.
        """
        if not getattr(self, "_vllm_asleep", False):
            return
        try:
            self._vllm_run(self._async_engine.wake_up())
            self._vllm_asleep = False
            self.log_info("[VLLMEmbodied] vLLM woke up (weights + KV restored).")
        except Exception as e:  # noqa: BLE001
            self.log_info(f"[VLLMEmbodied] wake_up failed (continuing awake-assumed): {e!r}")
            self._vllm_asleep = False

    def _vllm_sleep(self):
        """Best-effort sleep(level=1): offload vLLM weights + drop KV during actor train.

        DISABLED by default (rollout.vllm_sleep_offload): vLLM 0.19.1 sleep(level=1) +
        wake_up corrupts the multimodal encoder cache — after waking, generation asserts
        `Expected a cached item for mm_hash=...` (core.py:1502) and the epoch dies. We
        instead keep vLLM resident at a low gpu_memory_utilization (0.18) so the actor
        train step still fits alongside it (opt1 parks the HF model on CPU, freeing
        ~14G). Re-enable only if the vLLM mm-cache/sleep interaction is fixed.
        """
        if not getattr(self, "_sleep_offload_enabled", False):
            return
        if getattr(self, "_vllm_asleep", False):
            return
        try:
            self._vllm_run(self._async_engine.sleep(level=1))
            self._vllm_asleep = True
            self.log_info("[VLLMEmbodied] vLLM asleep (opt2: weights+KV released for actor train).")
        except Exception as e:  # noqa: BLE001
            self.log_info(f"[VLLMEmbodied] sleep failed (staying awake): {e!r}")

    async def generate_one_epoch(self, *args, **kwargs):
        # vLLM was woken in sync_model_from_actor (before reload_weights). Generate,
        # then release vLLM memory while the actor trains on the same GPU.
        self._ensure_vllm_awake()
        result = await super().generate_one_epoch(*args, **kwargs)
        self._vllm_sleep()
        return result

    async def evaluate(self, *args, **kwargs):
        # Eval reuses current synced weights; ensure the engine is awake, then sleep.
        self._ensure_vllm_awake()
        result = await super().evaluate(*args, **kwargs)
        self._vllm_sleep()
        return result

    def _unwrap_policy(self):
        """Return the inner QwenNavPolicy, unwrapping the PEFT/LoRA wrapper.

        get_model() returns a PeftModel when is_lora=true; PeftModel.get_base_model()
        returns the wrapped QwenNavPolicy (LoRA is not prompt-learning, so this is
        self.base_model.model). When is_lora=false, hf_model already is the policy.
        """
        m = self.hf_model
        if hasattr(m, "get_base_model"):
            return m.get_base_model()
        return m

    async def _init_vllm_engine(self) -> None:
        """Initialize vLLM AsyncLLM engine (runs inside _vllm_loop thread)."""
        rollout_cfg = self.cfg.rollout

        engine_args = AsyncEngineArgs(
            model=rollout_cfg.model.model_path,
            tensor_parallel_size=int(rollout_cfg.get("tensor_parallel_size", 1)),
            dtype=torch_dtype_from_precision(rollout_cfg.model.precision),
            gpu_memory_utilization=float(rollout_cfg.get("gpu_memory_utilization", 0.4)),
            enforce_eager=bool(rollout_cfg.get("enforce_eager", True)),
            load_format="auto",
            trust_remote_code=bool(self.cfg.actor.model.get("trust_remote_code", True)),
            max_model_len=int(rollout_cfg.get("max_model_len", 4096)),
            max_num_seqs=int(rollout_cfg.get("max_num_seqs", 32)),
            enable_sleep_mode=bool(rollout_cfg.get("enable_sleep_mode", True)),
        )
        self._async_engine = AsyncLLMEngine.from_engine_args(
            engine_args=engine_args,
            start_engine_loop=True,
        )

    def _make_vllm_generate_fn(self):
        """
        Return a *synchronous* callable matching the _batch_generate contract:
          fn(prompts: list[str], image_lists: list[list[Image]])
            -> (list[str], list[torch.Tensor])
        where each tensor is (1, resp_len) int64 response token ids.
        """
        policy = self._policy

        async def _async_generate(
            prompts: list[str],
            image_lists: list[list[Image.Image]],
        ):
            sampling_params = SamplingParams(
                temperature=policy.temperature if policy.do_sample else 0.0,
                max_tokens=policy.max_new_tokens,
            )

            tasks = []
            request_ids = []
            for prompt, images in zip(prompts, image_lists):
                request_id = str(next(self._request_counter))
                request_ids.append(request_id)
                inp = TextPrompt(
                    prompt=prompt,
                    multi_modal_data={"image": images} if images else None,
                )
                tasks.append(
                    self._async_engine.generate(
                        prompt=inp,
                        sampling_params=sampling_params,
                        request_id=request_id,
                    )
                )

            # Collect final output from each async generator
            async def collect(gen):
                last = None
                async for out in gen:
                    last = out
                return last

            outputs = await asyncio.gather(*(collect(t) for t in tasks))

            decoded: list[str] = []
            gen_ids_list: list[torch.Tensor] = []
            for out in outputs:
                text = out.outputs[0].text if out is not None else ""
                token_ids = list(out.outputs[0].token_ids) if out is not None else []
                decoded.append(text)
                gen_ids_list.append(
                    torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # (1, resp_len)
                )

            return decoded, gen_ids_list

        def vllm_generate_fn(
            prompts: list[str],
            image_lists: list[list[Image.Image]],
        ):
            future = asyncio.run_coroutine_threadsafe(
                _async_generate(prompts, image_lists), self._vllm_loop
            )
            return future.result()

        return vllm_generate_fn

    async def sync_model_from_actor(self):
        """
        V3: parent loads fresh LoRA weights into self.hf_model (PeftModel), then we
        merge the adapter, recover HF-*checkpoint* parameter names, and push them
        into the vLLM engine via reload_weights(is_checkpoint_format=True).

        Why this naming works (root cause of the earlier failures):
          - vLLM's Qwen3_5ForConditionalGeneration.load_weights uses a WeightsMapper
            (hf_to_vllm_mapper) that expects ORIGINAL HF-checkpoint names:
                "model.visual."        -> "visual."
                "model.language_model."-> "language_model.model."
                "lm_head."             -> "language_model.lm_head."
            Feeding it pre-stripped names (e.g. "visual.x"/"language_model.x") breaks
            every rule, so nothing maps and it errors with "no module named 'model'".
          - The actor's FSDP state_dict is PEFT-wrapped
            (base_model.model.<policy>.model.<hf>.<...>.base_layer/lora_A/lora_B),
            which is NOT checkpoint format at all. We therefore extract names from
            the *inner HF model* (policy.model), which are exactly the checkpoint
            names the mapper wants, after folding ".base_layer." -> ".".
        """
        # vLLM may be asleep from the previous generate_one_epoch (opt2). reload_weights
        # needs the engine's weights resident on GPU, so wake it first.
        self._ensure_vllm_awake()

        await super().sync_model_from_actor()

        # Write merged weights to a local safetensors dir, then have the vLLM
        # engine reload from that path. We go through disk (not collective_rpc
        # tensor kwargs) on purpose: collective_rpc decodes kwargs *untyped*, so a
        # torch.Tensor passed in kwargs arrives at the spawned EngineCore as a plain
        # Python list (AttributeError: 'list' object has no attribute 'size').
        # weights_path is a str — it round-trips cleanly, and vLLM's own loader
        # reconstructs tensors + applies hf_to_vllm_mapper correctly.
        weights = self._extract_merged_hf_weights()
        n_w = len(weights)
        path = self._write_weights_to_disk(weights)
        del weights
        fut = asyncio.run_coroutine_threadsafe(
            self._reload_vllm_weights(path), self._vllm_loop
        )
        fut.result(timeout=900)
        self.log_info(
            f"[VLLMEmbodied] V3 weight sync done: {n_w} tensors merged + "
            f"reloaded into vLLM from {path}."
        )

    def _write_weights_to_disk(self, weights):
        """Write [(hf_name, tensor)] to a per-process safetensors dir on NVMe.

        Produces model.safetensors + model.safetensors.index.json so vLLM's
        DefaultModelLoader.get_all_weights can read it (architecture/config come
        from the already-loaded model_config, so no config.json is needed here).
        Overwrites the same dir each sync to avoid filling disk.
        """
        import json
        import os

        from safetensors.torch import save_file

        out_dir = f"/home/nvme03/lck/tmp/vllm_wsync_pid{os.getpid()}"
        os.makedirs(out_dir, exist_ok=True)
        tensors = {name: t.contiguous() for name, t in weights}
        st_path = os.path.join(out_dir, "model.safetensors")
        save_file(tensors, st_path)

        total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        index = {
            "metadata": {"total_size": int(total_bytes)},
            "weight_map": {name: "model.safetensors" for name in tensors},
        }
        with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
        return out_dir

    def _extract_merged_hf_weights(self):
        """Merge LoRA into base and return [(hf_checkpoint_name, cpu_tensor), ...].

        Operates on the rollout's PeftModel (plain, single-GPU — not FSDP). Uses
        merge_adapter()/unmerge_adapter() so the in-memory adapter structure is
        preserved for any later use; the merged value lives in base_layer.weight
        only between the two calls.
        """
        peft = self.hf_model
        is_lora = hasattr(peft, "merge_adapter")

        def _collect(inner_hf):
            out = []
            for name, p in inner_hf.named_parameters():
                if (
                    ".lora_A" in name
                    or ".lora_B" in name
                    or "lora_magnitude_vector" in name
                    or "lora_embedding_" in name
                ):
                    continue
                clean = name.replace(".base_layer.", ".")
                out.append((clean, p.detach().to("cpu")))
            return out

        if not is_lora:
            return _collect(self._policy.model)

        peft.merge_adapter()
        try:
            weights = _collect(self._policy.model)
        finally:
            peft.unmerge_adapter()
        return weights

    async def _reload_vllm_weights(self, path):
        """Reload vLLM engine weights from a local safetensors dir.

        reload_weights(weights_path=..., is_checkpoint_format=True) routes through
        the model's load_weights + hf_to_vllm_mapper, which handles all
        fused-kernel / prefix remapping internally. Runs inside the dedicated vLLM
        event loop thread.
        """
        await self._async_engine.collective_rpc(
            "reload_weights",
            timeout=600,
            kwargs={"weights_path": path, "is_checkpoint_format": True},
        )

    def shutdown(self):
        """Clean up vLLM engine thread on exit."""
        if hasattr(self, "_vllm_loop") and self._vllm_loop.is_running():
            self._vllm_loop.call_soon_threadsafe(self._vllm_loop.stop)
        super().shutdown() if hasattr(super(), "shutdown") else None

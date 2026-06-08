#!/usr/bin/env python3
"""Standalone QwenNav HF vs vLLM generation benchmark.

This script is intentionally outside the RLinf training entrypoints.  It does
not modify rollout workers, env workers, checkpoints, or running training jobs.

Example:
    CUDA_VISIBLE_DEVICES=0 python tools/bench_qwen_nav_vllm.py --backend hf
    CUDA_VISIBLE_DEVICES=0 python tools/bench_qwen_nav_vllm.py --backend vllm
    CUDA_VISIBLE_DEVICES=0 python tools/bench_qwen_nav_vllm.py --backend both --num-samples 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN_NAV_DIR = REPO_ROOT / "rlinf" / "models" / "embodiment" / "qwen_nav"


def _load_module_from_file(module_name: str, path: Path):
    """Load qwen_nav helper files without importing the whole rlinf package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_prompts = _load_module_from_file("qwen_nav_prompts", QWEN_NAV_DIR / "prompts.py")
_parser = _load_module_from_file("qwen_nav_action_parser", QWEN_NAV_DIR / "action_parser.py")

SYSTEM_PROMPT = _prompts.SYSTEM_PROMPT
build_user_content_text = _prompts.build_user_content_text
parse_lavira_json = _parser.parse_lavira_json


DEFAULT_MODEL_PATH = "/home/clk/workspace/model_zoo/Qwen/Qwen3.5-4B"
DEFAULT_INSTRUCTION = (
    "Walk forward into the room, turn toward the target object, and stop when "
    "you are close to the destination."
)


@dataclass
class BenchOutput:
    backend: str
    latency_s: float
    decoded: list[str]
    token_ids: list[list[int]]


def _now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _make_synthetic_view(label: str, size: tuple[int, int], seed: int) -> Image.Image:
    """Create a simple RGB scene-like image for multimodal plumbing tests."""
    rng = np.random.default_rng(seed)
    width, height = size
    base = np.zeros((height, width, 3), dtype=np.uint8)
    sky = np.array([180, 190, 200], dtype=np.uint8)
    floor = np.array([120, 105, 90], dtype=np.uint8)
    wall = np.array([155, 145, 130], dtype=np.uint8)
    base[: height // 2] = wall
    base[height // 2 :] = floor

    # Add a few colored rectangles so the image is not blank.
    for _ in range(8):
        x0 = int(rng.integers(0, max(1, width - 80)))
        y0 = int(rng.integers(0, max(1, height - 80)))
        x1 = min(width, x0 + int(rng.integers(30, 120)))
        y1 = min(height, y0 + int(rng.integers(30, 120)))
        color = rng.integers(40, 230, size=3, dtype=np.uint8)
        base[y0:y1, x0:x1] = color

    # Add a horizon highlight and label for human inspection.
    base[height // 2 - 2 : height // 2 + 2] = sky
    img = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((10, 10, 220, 46), fill=(0, 0, 0))
    draw.text((18, 20), label, fill=(255, 255, 255))
    return img


def build_sample_prompts(
    *,
    num_samples: int,
    image_size: tuple[int, int],
    history_frames: int,
    instruction: str,
) -> tuple[list[list[dict[str, Any]]], list[list[Image.Image]]]:
    messages_list: list[list[dict[str, Any]]] = []
    image_lists: list[list[Image.Image]] = []

    for sample_idx in range(num_samples):
        hist = [
            _make_synthetic_view(f"history step {i}", image_size, seed=1000 + sample_idx * 17 + i)
            for i in range(history_frames)
        ]
        current = [
            _make_synthetic_view("front", image_size, seed=2000 + sample_idx * 17),
            _make_synthetic_view("left", image_size, seed=3000 + sample_idx * 17),
            _make_synthetic_view("right", image_size, seed=4000 + sample_idx * 17),
            _make_synthetic_view("behind", image_size, seed=5000 + sample_idx * 17),
        ]
        user_text = build_user_content_text(
            instruction=instruction,
            history_step_indices=list(range(len(hist))),
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in hist + current]
                    + [{"type": "text", "text": user_text}]
                ),
            },
        ]
        # We build the exact final chat template with the HF processor later.
        # Keep PIL images in memory; they are not JSON-serializable.
        messages_list.append(messages)
        image_lists.append(hist + current)

    return messages_list, image_lists


def _apply_chat_template(processor: Any, messages: list[dict[str, Any]]) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def load_processor(model_path: str, trust_remote_code: bool):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    return processor


def run_hf(
    *,
    model_path: str,
    messages_list: list[list[dict[str, Any]]],
    image_lists: list[list[Image.Image]],
    precision: str,
    attn_implementation: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    trust_remote_code: bool,
    chunk_size: int,
) -> BenchOutput:
    import transformers

    AutoModelVL = (
        getattr(transformers, "AutoModelForImageTextToText", None)
        or getattr(transformers, "AutoModelForVision2Seq", None)
    )
    if AutoModelVL is None:
        raise RuntimeError(
            "Current transformers lacks AutoModelForImageTextToText and "
            "AutoModelForVision2Seq."
        )

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(precision, torch.bfloat16)

    processor = load_processor(model_path, trust_remote_code)
    model = AutoModelVL.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    prompt_texts = [_apply_chat_template(processor, msg) for msg in messages_list]
    decoded: list[str] = []
    token_ids: list[list[int]] = []

    t0 = _now()
    with torch.inference_mode():
        for start in range(0, len(prompt_texts), chunk_size):
            sub_prompts = prompt_texts[start : start + chunk_size]
            sub_images = image_lists[start : start + chunk_size]
            inputs = processor(
                text=sub_prompts,
                images=sub_images,
                padding=True,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

            gen_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
            input_len = inputs["input_ids"].shape[1]
            new_tok = gen_out[:, input_len:]
            decoded.extend(processor.batch_decode(new_tok, skip_special_tokens=True))
            token_ids.extend(new_tok.detach().cpu().tolist())
    latency = _now() - t0
    return BenchOutput("hf", latency, decoded, token_ids)


def run_vllm(
    *,
    model_path: str,
    messages_list: list[list[dict[str, Any]]],
    image_lists: list[list[Image.Image]],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    trust_remote_code: bool,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
) -> BenchOutput:
    if importlib.util.find_spec("vllm") is None:
        raise RuntimeError("vLLM is not installed in this Python environment.")

    from vllm import LLM, SamplingParams

    processor = load_processor(model_path, trust_remote_code)
    prompt_texts = [_apply_chat_template(processor, msg) for msg in messages_list]

    sampling_params = SamplingParams(
        temperature=temperature if do_sample else 0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )
    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        }
        for prompt, images in zip(prompt_texts, image_lists, strict=True)
    ]

    llm_kwargs = {
        "model": model_path,
        "trust_remote_code": trust_remote_code,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    try:
        llm = LLM(**llm_kwargs, task="generate")
    except TypeError as exc:
        # Older vLLM versions do not expose the `task` EngineArgs keyword.
        if "task" not in str(exc):
            raise
        llm = LLM(**llm_kwargs)
    t0 = _now()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    latency = _now() - t0

    decoded: list[str] = []
    token_ids: list[list[int]] = []
    for out in outputs:
        item = out.outputs[0]
        decoded.append(item.text)
        token_ids.append(list(item.token_ids))
    return BenchOutput("vllm", latency, decoded, token_ids)


def summarize(output: BenchOutput) -> dict[str, Any]:
    parse_results = [parse_lavira_json(text) for text in output.decoded]
    token_lens = [len(ids) for ids in output.token_ids]
    ok_count = sum(1 for p in parse_results if p.ok)
    actions = []
    for p in parse_results:
        if not p.ok:
            actions.append("parse_fail")
        elif p.stop:
            actions.append("stop")
        else:
            actions.append(p.raw_dir or "unknown")
    return {
        "backend": output.backend,
        "num_samples": len(output.decoded),
        "latency_s": round(output.latency_s, 4),
        "samples_per_s": round(len(output.decoded) / max(output.latency_s, 1e-9), 4),
        "parse_ok": ok_count,
        "parse_fail": len(output.decoded) - ok_count,
        "parse_ok_rate": round(ok_count / max(len(output.decoded), 1), 4),
        "tokens_mean": round(statistics.mean(token_lens), 2) if token_lens else 0,
        "tokens_max": max(token_lens) if token_lens else 0,
        "actions": actions,
    }


def print_examples(output: BenchOutput, max_examples: int) -> None:
    print(f"\n[{output.backend}] decoded examples")
    for i, text in enumerate(output.decoded[:max_examples]):
        parsed = parse_lavira_json(text)
        print("-" * 88)
        print(f"sample={i} parse_ok={parsed.ok} err={parsed.err} action={parsed.raw_dir} stop={parsed.stop}")
        print(text.strip()[:2000])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone QwenNav HF/vLLM multimodal generation benchmark."
    )
    parser.add_argument("--backend", choices=["hf", "vllm", "both"], default="both")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--history-frames", type=int, default=2)
    parser.add_argument("--image-size", type=int, nargs=2, default=[448, 448], metavar=("W", "H"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--print-examples", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.model_path).exists():
        print(f"[bench] model path does not exist: {args.model_path}", file=sys.stderr)
        return 2

    print("[bench] QwenNav vLLM POC benchmark")
    print(f"[bench] repo={REPO_ROOT}")
    print(f"[bench] backend={args.backend} model={args.model_path}")
    print(f"[bench] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[bench] torch.cuda.is_available={torch.cuda.is_available()}")

    messages_list, image_lists = build_sample_prompts(
        num_samples=args.num_samples,
        image_size=tuple(args.image_size),
        history_frames=args.history_frames,
        instruction=args.instruction,
    )

    outputs: list[BenchOutput] = []
    if args.backend in ("hf", "both"):
        outputs.append(
            run_hf(
                model_path=args.model_path,
                messages_list=messages_list,
                image_lists=image_lists,
                precision=args.precision,
                attn_implementation=args.attn_implementation,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                trust_remote_code=args.trust_remote_code,
                chunk_size=args.chunk_size,
            )
        )

    if args.backend in ("vllm", "both"):
        try:
            outputs.append(
                run_vllm(
                    model_path=args.model_path,
                    messages_list=messages_list,
                    image_lists=image_lists,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.do_sample,
                    trust_remote_code=args.trust_remote_code,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    tensor_parallel_size=args.tensor_parallel_size,
                )
            )
        except Exception as exc:
            print(f"[bench][vllm] skipped/failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.backend == "vllm":
                return 1

    print("\n[bench] summary")
    print(json.dumps([summarize(out) for out in outputs], indent=2, ensure_ascii=False))
    for out in outputs:
        print_examples(out, max_examples=args.print_examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

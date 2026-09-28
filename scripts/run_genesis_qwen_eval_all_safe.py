#!/usr/bin/env python3
"""Run the existing RLinf Genesis pure-eval mode over OpenNav100.

This is deliberately an orchestration-only entry point.  It uses the existing
``genark_eval_qwen_zeroshot`` config and never kills global Ray processes.
Each pass owns one or more physical env GPUs and evaluates one scene per
EnvWorker; the rollout process stays on a separate physical GPU.

By default, every EnvWorker is sized to the largest scene in the dataset.  A
worker therefore loads all episodes for its pinned scene into one Genesis
BatchRenderer build and advances them as a vectorized slot batch, matching
``/home/clk/workspace/genark/genesis_uninavid_batch_parallel.py``.  Passing an
explicit smaller ``--envs-per-worker`` retains the rolling-window mode for
lower-memory runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/clk/miniconda3/envs/genesis-vllm/bin/python"
CONFIG_DIR = ROOT / "examples" / "embodiment" / "config"
TRAIN = ROOT / "examples" / "embodiment" / "train_embodied_agent.py"
DATASET = Path(
    "/home/clk/workspace/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json"
)
AGGREGATOR = ROOT / "scripts" / "aggregate_aligned_nav_metrics.py"
DEFAULT_MODEL = Path(
    "/home/lhx/workspace/test_model/output/"
    "qwen3.5_v2_unfreeze_vit_nav_target/"
    "v0-20260709-081633/checkpoint-1200"
)


def parse_gpu_list(spec: str) -> list[int]:
    if re.fullmatch(r"\d+-\d+", spec):
        start, end = map(int, spec.split("-"))
        return list(range(start, end + 1))
    return [int(item) for item in spec.split(",") if item]


def placement(gpus: list[int]) -> str:
    if len(gpus) == 1:
        return str(gpus[0])
    if gpus == list(range(gpus[0], gpus[-1] + 1)):
        return "-".join(map(str, gpus))
    # Hydra treats an unquoted comma as an ambiguous sweep/list separator.
    # Keep non-contiguous physical GPU placement as one string value.
    return "'" + ",".join(map(str, gpus)) + "'"


def dataset_scene_counts() -> dict[str, int]:
    data = json.loads(DATASET.read_text())
    episodes = data["episodes"] if isinstance(data, dict) else data
    counts: dict[str, int] = {}
    for episode in episodes:
        scene_id = str(episode["scene_id"])
        counts[scene_id] = counts.get(scene_id, 0) + 1
    return counts


def run_pass(
    pass_index: int,
    scene_offset: int,
    rollout_gpu: int,
    env_gpus: list[int],
    envs_per_worker: int,
    roll_through_episode_pool: bool,
    eval_repeats_per_episode: int,
    max_episode_steps: int,
    model_path: Path,
    temperature: float,
    tmp_root: Path,
    output_dir: Path,
) -> int:
    pass_dir = output_dir / f"pass_{pass_index:02d}"
    pass_dir.mkdir(parents=True, exist_ok=True)
    total_envs = len(env_gpus) * envs_per_worker
    env_placement = placement(env_gpus)

    overrides = [
        f"runner.logger.log_path={pass_dir}",
        f"cluster.component_placement.rollout.placement={rollout_gpu}",
        f"cluster.component_placement.actor.placement={rollout_gpu}",
        f"cluster.component_placement.env.placement={env_placement}",
        f"env.eval.total_num_envs={total_envs}",
        f"env.eval.scene_offset={scene_offset}",
        "env.eval.genesis_backend=local",
        "env.eval.eval_roll_through_episode_pool="
        + str(roll_through_episode_pool).lower(),
        f"env.eval.eval_repeats_per_episode={eval_repeats_per_episode}",
        "env.eval.group_size=1",
        f"env.eval.max_episode_steps={max_episode_steps}",
        "env.eval.max_steps_per_rollout_epoch=5000",
        "env.eval.video_cfg.save_video=false",
        # Frozen evaluation must be deterministic on both the runner and model
        # paths.  These overrides also keep Genesis/Habitat A/B comparable.
        "algorithm.sampling_params.do_sample="
        + str(temperature > 0.0).lower(),
        f"algorithm.sampling_params.temperature_eval={temperature}",
        f"actor.model.model_path={model_path}",
        "actor.model.prompt_style=lavira_waypoint",
        "actor.model.do_sample=" + str(temperature > 0.0).lower(),
        f"actor.model.temperature={temperature}",
        "actor.model.lavira_runtime.enabled=true",
        "actor.model.grounded_sam.enabled=true",
        "actor.model.grounded_sam.dino_config_path=/home/clk/workspace/RLinf/assets/grounded_sam/GroundingDINO_SwinT_OGC.py",
        "actor.model.grounded_sam.dino_checkpoint_path=/home/clk/workspace/RLinf/assets/grounded_sam/groundingdino_swint_ogc.pth",
        "actor.model.grounded_sam.repvit_sam_checkpoint_path=/home/nvme01/uni-lavira/data/grounded_sam/repvit_sam.pt",
        "actor.model.history_max_frames=1",
        "actor.model.max_new_tokens=256",
        "rollout.gpu_memory_utilization=0.25",
        "rollout.max_num_seqs=8",
    ]
    command = [
        PYTHON,
        str(TRAIN),
        "--config-path",
        str(CONFIG_DIR),
        "--config-name",
        "genark_eval_qwen_zeroshot",
        *overrides,
    ]

    # Keep Ray's Unix-domain socket path below Linux's 107-byte limit.  The
    # experiment log path is intentionally descriptive, so it is unsuitable
    # as RAY_TMPDIR.
    ray_tmp = tmp_root / f"gq{pass_index}_{int(time.time())}"
    process_tmp = tmp_root / f"gqt{pass_index}_{int(time.time())}"

    env = os.environ.copy()
    env.update(
        {
            "EMBODIED_PATH": str(ROOT / "examples" / "embodiment"),
            "REPO_PATH": str(ROOT),
            "PYTHONPATH": f"{ROOT}:{env.get('PYTHONPATH', '')}",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "GENESIS_HEADLESS": "1",
            # These are physical GPU ids.  Placement above is interpreted in
            # the full Ray hardware namespace because all GPUs stay visible.
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "TORCHDYNAMO_DISABLE": "1",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "3600",
            "NCCL_TIMEOUT_MS": "3600000",
            "RLINF_RAY_LOCAL": "1",
            "RAY_ADDRESS": "",
            "RAY_TMPDIR": str(ray_tmp),
            "TMPDIR": str(process_tmp),
        }
    )
    ray_tmp.mkdir(parents=True, exist_ok=True)
    process_tmp.mkdir(parents=True, exist_ok=True)

    print(
        f"[pure-eval] pass={pass_index} scene_offset={scene_offset} "
        f"rollout=physical GPU {rollout_gpu} env=physical GPUs {env_gpus} "
        f"envs={total_envs} slots_per_scene={envs_per_worker} "
        f"roll_through={roll_through_episode_pool} log={pass_dir}",
        flush=True,
    )
    with (pass_dir / "eval.log").open("w") as log:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        return_code = proc.wait()
    if return_code:
        print(f"[pure-eval] pass={pass_index} failed with exit={return_code}", flush=True)
    else:
        print(f"[pure-eval] pass={pass_index} completed", flush=True)
    return return_code


def aggregate(output_dir: Path, expected_episodes: int, max_episode_steps: int) -> None:
    files = sorted(output_dir.glob("pass_*/per_scene_*.json"))
    if not files:
        raise RuntimeError(f"no per-scene metric files found under {output_dir}")
    command = [
        sys.executable,
        str(AGGREGATOR),
        "--input-files",
        *[str(path) for path in files],
        "--output-dir",
        str(output_dir),
        "--expected-episodes",
        str(expected_episodes),
        "--source",
        "genesis_rlinf_qwen_pure_eval",
        "--max-episode-steps",
        str(max_episode_steps),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-gpu", type=int, default=4)
    parser.add_argument(
        "--env-gpus",
        default="6",
        help=(
            "Physical Genesis GPU(s). The default single GPU processes scenes "
            "serially like the reference GenArk evaluator; a range/list runs "
            "independent scene workers concurrently."
        ),
    )
    parser.add_argument(
        "--envs-per-worker",
        type=int,
        default=None,
        help=(
            "Genesis slots per scene. Default: maximum episode count of any "
            "scene, so every scene is evaluated in one static BatchRenderer "
            "batch. A smaller explicit value enables rolling-window mode."
        ),
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=300,
        help="shared episode horizon",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--eval-repeats-per-episode", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--start-pass",
        type=int,
        default=0,
        help="Skip already completed scene passes when resuming an evaluation.",
    )
    parser.add_argument(
        "--start-scene-offset",
        type=int,
        default=None,
        help=(
            "Dataset scene offset for a resumed run. This is independent of "
            "--start-pass so the number of environment GPUs can change safely. "
            "Default: start_pass * number of environment GPUs."
        ),
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=Path(os.environ.get("RLINF_EVAL_TMP_ROOT", "/tmp")),
        help="Parent directory for per-pass Ray and process temporary files.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=None,
        help="Run at most this many passes (used by failure recovery).",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Do not aggregate after a partial recovery run.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    env_gpus = parse_gpu_list(args.env_gpus)
    scene_counts = dataset_scene_counts()
    n_scenes = len(scene_counts)
    max_scene_episodes = max(scene_counts.values())
    envs_per_worker = args.envs_per_worker or max_scene_episodes
    if envs_per_worker <= 0:
        raise ValueError("--envs-per-worker must be positive")
    if args.eval_repeats_per_episode <= 0:
        raise ValueError("--eval-repeats-per-episode must be positive")
    if args.start_pass < 0:
        raise ValueError("--start-pass cannot be negative")
    if args.max_passes is not None and args.max_passes <= 0:
        raise ValueError("--max-passes must be positive")
    start_scene_offset = (
        args.start_scene_offset
        if args.start_scene_offset is not None
        else args.start_pass * len(env_gpus)
    )
    if not 0 <= start_scene_offset <= n_scenes:
        raise ValueError(
            f"--start-scene-offset must be between 0 and {n_scenes}, "
            f"got {start_scene_offset}"
        )
    if not args.model_path.is_dir():
        raise ValueError(f"--model-path is not a directory: {args.model_path}")
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    roll_through_episode_pool = envs_per_worker < max_scene_episodes
    expected_episodes = len(json.loads(DATASET.read_text())["episodes"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "launcher_config.json").write_text(
        json.dumps(
            {
                "config_name": "genark_eval_qwen_zeroshot",
                "dataset": str(DATASET),
                "expected_episodes": expected_episodes,
                "expected_scenes": n_scenes,
                "rollout_gpu_physical": args.rollout_gpu,
                "env_gpus_physical": env_gpus,
                "envs_per_worker": envs_per_worker,
                "max_scene_episodes": max_scene_episodes,
                "scene_episode_counts": scene_counts,
                "scene_batch_mode": (
                    "rolling_window"
                    if roll_through_episode_pool
                    else "all_episodes_static_batch"
                ),
                "max_episode_steps": args.max_episode_steps,
                "model_path": str(args.model_path),
                "eval_repeats_per_episode": args.eval_repeats_per_episode,
                "temperature": args.temperature,
                "start_pass": args.start_pass,
                "start_scene_offset": start_scene_offset,
                "tmp_root": str(args.tmp_root),
            },
            indent=2,
        )
    )

    pass_count = 0
    for pass_index, offset in enumerate(
        range(start_scene_offset, n_scenes, len(env_gpus)),
        start=args.start_pass,
    ):
        if args.max_passes is not None and pass_count >= args.max_passes:
            break
        code = run_pass(
            pass_index,
            offset,
            args.rollout_gpu,
            env_gpus[: min(len(env_gpus), n_scenes - offset)],
            envs_per_worker,
            roll_through_episode_pool,
            args.eval_repeats_per_episode,
            args.max_episode_steps,
            args.model_path,
            args.temperature,
            args.tmp_root,
            args.output_dir,
        )
        if code:
            raise SystemExit(code)
        pass_count += 1
    if not args.skip_aggregate:
        aggregate(args.output_dir, expected_episodes, args.max_episode_steps)


if __name__ == "__main__":
    main()

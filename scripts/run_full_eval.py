"""
GenArk full-dataset evaluation — Python multi-pass orchestrator.

Automatically splits N scenes across M env workers in ceil(N/M) passes.
Each pass runs a fresh RLinf eval process with the correct scene_offset.
Results from all passes are aggregated into combined_metrics.json.

Usage:
    python scripts/run_full_eval.py [OPTIONS]

    python scripts/run_full_eval.py                         # defaults
    python scripts/run_full_eval.py --rollout-gpu 3 --env-gpus 4-7 --envs-per-worker 20
    python scripts/run_full_eval.py --env-gpus 4,5 --envs-per-worker 20
    python scripts/run_full_eval.py --env-gpus 4 --envs-per-worker 20   # single worker
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


RLINF_ROOT   = Path(__file__).parent.parent.resolve()
PYTHON       = "/home/clk/miniconda3/envs/genesis/bin/python"
EMBODIED_CFG = str(RLINF_ROOT / "examples/embodiment")
EPISODES_FILE = "/home/nvme03/lck/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json"
RESULTS_DIR  = Path("/home/clk/workspace/results/genark_eval")


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_gpu_list(spec: str) -> list[int]:
    """'4-7' → [4,5,6,7]   '4,5,7' → [4,5,7]   '4' → [4]"""
    if re.fullmatch(r"\d+-\d+", spec):
        a, b = map(int, spec.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in spec.split(",")]


def placement_str(gpus: list[int]) -> str:
    """[4,5,6,7] → '4-7'   [4,6] → '4,6'   [4] → '4'"""
    if len(gpus) == 1:
        return str(gpus[0])
    if gpus == list(range(gpus[0], gpus[-1] + 1)):
        return f"{gpus[0]}-{gpus[-1]}"
    return ",".join(map(str, gpus))


def count_scenes() -> int:
    with open(EPISODES_FILE) as f:
        data = json.load(f)
    eps = data.get("episodes", data) if isinstance(data, dict) else data
    return len({e["scene_id"] for e in eps})


def kill_ray():
    subprocess.run(["pkill", "-9", "-f", "ray::"], stderr=subprocess.DEVNULL)
    time.sleep(2)


def clear_pyc():
    for p in (RLINF_ROOT / "rlinf").rglob("*.pyc"):
        p.unlink(missing_ok=True)


def run_pass(
    pass_idx: int,
    rollout_gpu: int,
    env_gpus: list[int],
    envs_per_worker: int,
    scene_offset: int,
    log_path: Path,
) -> int:
    placement = placement_str(env_gpus)
    total_envs = len(env_gpus) * envs_per_worker

    print(f"\n{'─'*60}", flush=True)
    print(f"  Pass {pass_idx+1}: scene_offset={scene_offset}  "
          f"workers={len(env_gpus)} gpus={placement}  "
          f"total_envs={total_envs}", flush=True)
    print(f"  Log → {log_path}", flush=True)
    print(f"{'─'*60}", flush=True)

    cmd = [
        PYTHON,
        str(RLINF_ROOT / "examples/embodiment/train_embodied_agent.py"),
        "--config-name", "genark_eval_only",
        f"cluster.component_placement.rollout.placement={rollout_gpu}",
        f"cluster.component_placement.actor.placement={rollout_gpu}",
        f"cluster.component_placement.env.placement={placement}",
        f"env.eval.total_num_envs={total_envs}",
        f"env.eval.scene_offset={scene_offset}",  # declared in genark_r2r.yaml
    ]

    env = os.environ.copy()
    env["EMBODIED_PATH"]  = EMBODIED_CFG
    env["TORCHDYNAMO_DISABLE"] = "1"

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        proc.wait()
    return proc.returncode


def aggregate(results_dir: Path, num_scenes: int):
    files = sorted(results_dir.glob("per_scene_*.json"))
    if not files:
        print("WARNING: no per_scene_*.json found — nothing to aggregate.", flush=True)
        return

    all_eps = []
    scene_rows = []
    for f in files:
        d = json.loads(f.read_text())
        scene_rows.append({
            "scene_id":   d["scene_id"],
            "scan_name":  d["scan_name"],
            "n_episodes": d["n_episodes"],
            "summary":    d["summary"],
        })
        all_eps.extend(d["episodes"])

    n = len(all_eps)
    keys = ["success", "spl", "ndtw", "sdtw",
            "distance_to_goal", "path_length", "steps_taken"]
    combined = {k: sum(e.get(k, 0) for e in all_eps) / n for k in keys}
    combined["num_episodes"] = n
    combined["num_scenes"]   = len(scene_rows)

    out = {"combined": combined, "per_scene": scene_rows}
    out_path = results_dir / "combined_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\n{'='*60}", flush=True)
    print(f"  FULL DATASET RESULTS  ({n} eps / {len(scene_rows)}/{num_scenes} scenes)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  SR   : {combined['success']:.3f}  ({combined['success']*100:.1f}%)", flush=True)
    print(f"  SPL  : {combined['spl']:.3f}", flush=True)
    print(f"  nDTW : {combined['ndtw']:.3f}", flush=True)
    print(f"  DTG  : {combined['distance_to_goal']:.2f}m", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\nPer-scene breakdown:", flush=True)
    for s in sorted(scene_rows, key=lambda x: -x["summary"]["success"]):
        sm = s["summary"]
        print(f"  {s['scan_name']:20s}  n={sm['num_episodes']:2d}  "
              f"SR={sm['success']:.3f}  SPL={sm['spl']:.3f}", flush=True)
    print(f"\nSaved → {out_path}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GenArk full-dataset eval (multi-pass)")
    parser.add_argument("--rollout-gpu",      type=int,   default=3)
    parser.add_argument("--env-gpus",         type=str,   default="4-7")
    parser.add_argument("--envs-per-worker",  type=int,   default=20,
                        help="Must be >= max episodes per scene (20)")
    parser.add_argument("--scene-offset",     type=int,   default=None,
                        help="Start from this scene offset (skip earlier scenes)")
    parser.add_argument("--dry-run",          action="store_true",
                        help="Print plan without running")
    args = parser.parse_args()

    all_env_gpus  = parse_gpu_list(args.env_gpus)
    max_workers   = len(all_env_gpus)
    num_scenes    = count_scenes()
    start_offset  = args.scene_offset if args.scene_offset is not None else 0
    num_passes    = (num_scenes - start_offset + max_workers - 1) // max_workers

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  GenArk Full Eval Plan", flush=True)
    print(f"  rollout GPU    : {args.rollout_gpu}", flush=True)
    print(f"  env GPUs       : {all_env_gpus}  (max {max_workers} workers)", flush=True)
    print(f"  envs/worker    : {args.envs_per_worker}", flush=True)
    print(f"  total scenes   : {num_scenes}  start_offset={start_offset}", flush=True)
    print(f"  passes needed  : {num_passes}", flush=True)
    print(f"{'='*60}", flush=True)

    if args.dry_run:
        for p in range(num_passes):
            offset        = start_offset + p * max_workers
            remaining     = num_scenes - offset
            active_n      = min(remaining, max_workers)
            active_gpus   = all_env_gpus[:active_n]
            print(f"  Pass {p+1}: scene_offset={offset}  "
                  f"workers={active_n} gpus={active_gpus}  "
                  f"total_envs={active_n * args.envs_per_worker}")
        return

    # Clean previous results only when starting from scratch (no --scene-offset)
    if args.scene_offset is None:
        for f in RESULTS_DIR.glob("per_scene_*.json"):
            f.unlink()
        for f in RESULTS_DIR.glob("combined_metrics.json"):
            f.unlink()

    log_dir = Path("/tmp")
    for pass_idx in range(num_passes):
        scene_offset  = start_offset + pass_idx * max_workers
        remaining     = num_scenes - scene_offset
        active_n      = min(remaining, max_workers)
        active_gpus   = all_env_gpus[:active_n]
        log_path      = log_dir / f"genark_eval_pass{pass_idx}_{int(time.time())}.log"

        kill_ray()
        clear_pyc()

        rc = run_pass(
            pass_idx     = pass_idx,
            rollout_gpu  = args.rollout_gpu,
            env_gpus     = active_gpus,
            envs_per_worker = args.envs_per_worker,
            scene_offset = scene_offset,
            log_path     = log_path,
        )
        if rc != 0:
            print(f"\nERROR: Pass {pass_idx+1} failed (exit {rc}). "
                  f"Check {log_path}", flush=True)
            sys.exit(rc)

        print(f"\nPass {pass_idx+1}/{num_passes} complete.", flush=True)

    aggregate(RESULTS_DIR, num_scenes)


if __name__ == "__main__":
    main()

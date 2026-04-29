#!/bin/bash
# GenArk full-dataset evaluation — auto multi-pass scene traversal.
#
# Automatically splits N scenes across M env workers in ceil(N/M) passes.
# Each pass adjusts scene_offset, active worker count, and GPU placement.
# Final per_scene_*.json files are aggregated into combined_metrics.json.
#
# Usage:
#   bash scripts/run_genark_eval_all.sh [ROLLOUT_GPU] [ENV_GPUS] [ENVS_PER_WORKER]
#
# Examples:
#   bash scripts/run_genark_eval_all.sh 3 4-7 20    # 4 workers × 20 envs (default)
#   bash scripts/run_genark_eval_all.sh 3 4,5 20    # 2 workers
#   bash scripts/run_genark_eval_all.sh 3 4 20      # 1 worker (serial)

set -e

ROLLOUT_GPU=${1:-3}
ENV_GPUS=${2:-4-7}
ENVS_PER_WORKER=${3:-20}   # must be >= max(ep_count per scene) = 20

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_PYTHON="/home/clk/miniconda3/envs/genesis/bin/python"
EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
RESULTS_DIR="/home/clk/workspace/results/genark_eval"
EPISODES_FILE="/home/nvme03/lck/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json"

mkdir -p "$RESULTS_DIR"

# ── Parse ENV_GPUS into an ordered list ────────────────────────────────────
if [[ "$ENV_GPUS" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    GPU_LIST=($(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"))
else
    IFS=',' read -ra GPU_LIST <<< "$ENV_GPUS"
fi
MAX_WORKERS=${#GPU_LIST[@]}

# ── Count unique scenes from episode file ──────────────────────────────────
NUM_SCENES=$($CONDA_PYTHON -c "
import json
with open('$EPISODES_FILE') as f:
    data = json.load(f)
eps = data.get('episodes', data) if isinstance(data, dict) else data
print(len(sorted({e['scene_id'] for e in eps})))
")

NUM_PASSES=$(( (NUM_SCENES + MAX_WORKERS - 1) / MAX_WORKERS ))

echo "=================================================="
echo "  GenArk Full Dataset Eval"
echo "  Rollout GPU    : $ROLLOUT_GPU"
echo "  Env GPUs       : ${GPU_LIST[*]}  ($MAX_WORKERS max workers)"
echo "  Envs/worker    : $ENVS_PER_WORKER"
echo "  Total scenes   : $NUM_SCENES"
echo "  Passes needed  : $NUM_PASSES"
echo "=================================================="

# ── Clean previous results ─────────────────────────────────────────────────
rm -f "$RESULTS_DIR"/per_scene_*.json "$RESULTS_DIR"/avg_metrics.json \
      "$RESULTS_DIR"/combined_metrics.json

# ── Run each pass ─────────────────────────────────────────────────────────
PASS_LOGS=()
for (( PASS=0; PASS<NUM_PASSES; PASS++ )); do
    SCENE_OFFSET=$(( PASS * MAX_WORKERS ))
    SCENES_REMAINING=$(( NUM_SCENES - SCENE_OFFSET ))
    ACTIVE_WORKERS=$(( SCENES_REMAINING < MAX_WORKERS ? SCENES_REMAINING : MAX_WORKERS ))
    TOTAL_ENVS=$(( ACTIVE_WORKERS * ENVS_PER_WORKER ))

    # Build placement string for active workers
    if [ $ACTIVE_WORKERS -eq 1 ]; then
        PLACEMENT="${GPU_LIST[0]}"
    else
        FIRST_GPU="${GPU_LIST[0]}"
        LAST_GPU="${GPU_LIST[$((ACTIVE_WORKERS - 1))]}"
        PLACEMENT="${FIRST_GPU}-${LAST_GPU}"
    fi

    LOG_FILE="/tmp/genark_eval_pass${PASS}_$(date +%Y%m%d_%H%M%S).log"
    PASS_LOGS+=("$LOG_FILE")

    echo ""
    echo "── Pass $((PASS+1))/$NUM_PASSES ──────────────────────────────────────────"
    echo "  scene_offset=${SCENE_OFFSET}  active_workers=${ACTIVE_WORKERS}"
    echo "  env.placement=${PLACEMENT}  total_envs=${TOTAL_ENVS}"
    echo "  Log: $LOG_FILE"
    echo ""

    # Kill any stale Ray processes
    pkill -9 -f "ray::" 2>/dev/null || true
    sleep 3

    # Clear pyc to ensure latest code is used
    find "$RLINF_ROOT/rlinf" -name "*.pyc" -delete 2>/dev/null || true

    # Run eval
    cd "$RLINF_ROOT"
    EMBODIED_PATH="$EMBODIED_PATH" \
    TORCHDYNAMO_DISABLE=1 \
    $CONDA_PYTHON examples/embodiment/train_embodied_agent.py \
        --config-name genark_eval_only \
        "cluster.component_placement.rollout.placement=$ROLLOUT_GPU" \
        "cluster.component_placement.actor.placement=$ROLLOUT_GPU" \
        "cluster.component_placement.env.placement=$PLACEMENT" \
        "env.eval.total_num_envs=$TOTAL_ENVS" \
        "env.eval.scene_offset=$SCENE_OFFSET" \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Pass $((PASS+1)) failed (exit $EXIT_CODE). Last 20 lines:"
        tail -20 "$LOG_FILE"
        exit $EXIT_CODE
    fi

    echo "Pass $((PASS+1)) complete."
done

# ── Aggregate all per_scene_*.json ────────────────────────────────────────
echo ""
echo "── Aggregating results ───────────────────────────────────────────────"
$CONDA_PYTHON << 'PYEOF'
import json, glob, os

results_dir = "/home/clk/workspace/results/genark_eval"
files = sorted(glob.glob(os.path.join(results_dir, "per_scene_*.json")))

all_eps = []
scene_summaries = []

for f in files:
    with open(f) as fp:
        d = json.load(fp)
    scene_summaries.append({
        "scene_id": d["scene_id"],
        "scan_name": d["scan_name"],
        "n_episodes": d["n_episodes"],
        "summary": d["summary"],
    })
    all_eps.extend(d["episodes"])

if not all_eps:
    print("No episode data found!")
    exit(1)

keys = ["success", "spl", "ndtw", "sdtw", "distance_to_goal", "path_length", "steps_taken"]
n = len(all_eps)
combined = {k: sum(e.get(k, 0) for e in all_eps) / n for k in keys}
combined["num_episodes"] = n
combined["num_scenes"]   = len(scene_summaries)

out = {
    "combined": combined,
    "per_scene": scene_summaries,
}
out_path = os.path.join(results_dir, "combined_metrics.json")
with open(out_path, "w") as fp:
    json.dump(out, fp, indent=2)

print(f"\n{'='*56}")
print(f"  FULL DATASET RESULTS  ({n} episodes, {len(scene_summaries)} scenes)")
print(f"{'='*56}")
print(f"  SR   : {combined['success']:.3f}  ({combined['success']*100:.1f}%)")
print(f"  SPL  : {combined['spl']:.3f}")
print(f"  nDTW : {combined['ndtw']:.3f}")
print(f"  DTG  : {combined['distance_to_goal']:.2f}m")
print(f"{'='*56}")
print(f"\nPer-scene breakdown:")
for s in sorted(scene_summaries, key=lambda x: -x["summary"]["success"]):
    sm = s["summary"]
    print(f"  {s['scan_name']:20s}  n={sm['num_episodes']:2d}  SR={sm['success']:.3f}  SPL={sm['spl']:.3f}")
print(f"\nSaved → {out_path}")
PYEOF

echo ""
echo "=================================================="
echo "  All passes complete. Results in $RESULTS_DIR"
echo "=================================================="

#!/bin/bash
# GenArk eval script — supports multi-scene parallel evaluation.
# Each Env Worker pins to a distinct scene (round-robin via unique_scenes[rank]).
#
# Usage:
#   bash scripts/run_genark_eval.sh [ROLLOUT_GPU] [ENV_GPUS] [NUM_ENVS]
#
# ENV_GPUS may be either:
#   - A single GPU index            (e.g. 3)            → 1 worker, 1 scene
#   - A range "start-end" (incl.)   (e.g. 3-7)          → 5 workers, 5 scenes
#   - A comma-separated list        (e.g. 3,5,7)        → 3 workers, 3 scenes
#
# Examples:
#   bash scripts/run_genark_eval.sh                     # defaults: rollout=2, env=3-7, 40 envs
#   bash scripts/run_genark_eval.sh 2 3-7 40            # 5 scenes parallel
#   bash scripts/run_genark_eval.sh 5 6 8               # 1 scene only
#   bash scripts/run_genark_eval.sh 2 3,5,7 24          # 3 specific GPUs

set -e

# ── Arguments (with defaults) ──────────────────────────────────────────────
ROLLOUT_GPU=${1:-2}        # GPU for UniNaVid inference (~14 GB)
ENV_GPUS=${2:-3-7}         # GPU(s) for Genesis rendering  (~5 GB each)
NUM_ENVS=${3:-40}          # total_num_envs (split evenly across env workers)

# ── Parse ENV_GPUS into a list ─────────────────────────────────────────────
if [[ "$ENV_GPUS" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    ENV_GPU_LIST=$(seq -s, "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")
else
    ENV_GPU_LIST="$ENV_GPUS"   # already comma-list or single value
fi
NUM_ENV_WORKERS=$(echo "$ENV_GPU_LIST" | tr ',' '\n' | wc -l)

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_PYTHON="/home/clk/miniconda3/envs/genesis/bin/python"
EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
LOG_FILE="/tmp/genark_eval_$(date +%Y%m%d_%H%M%S).log"

# ── Pre-flight ─────────────────────────────────────────────────────────────
echo "=================================================="
echo "  GenArk Multi-Scene Eval"
echo "  Rollout GPU      : $ROLLOUT_GPU  (~14 GB, UniNaVid)"
echo "  Env GPUs         : $ENV_GPU_LIST  ($NUM_ENV_WORKERS workers, ~5 GB each)"
echo "  Total envs       : $NUM_ENVS  ($((NUM_ENVS / NUM_ENV_WORKERS)) per worker)"
echo "  Scenes evaluated : $NUM_ENV_WORKERS  (one per worker)"
echo "  Log file         : $LOG_FILE"
echo "=================================================="

# Sanity: NUM_ENVS divisible by NUM_ENV_WORKERS
if [ $((NUM_ENVS % NUM_ENV_WORKERS)) -ne 0 ]; then
    echo "ERROR: NUM_ENVS ($NUM_ENVS) must be divisible by NUM_ENV_WORKERS ($NUM_ENV_WORKERS)"
    exit 1
fi

# Check all GPUs free enough
ALL_GPUS="$ROLLOUT_GPU,$ENV_GPU_LIST"
for GPU in $(echo "$ALL_GPUS" | tr ',' ' '); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null || echo "0")
    USED=$(echo "$USED" | tr -d ' ')
    if [ "$USED" -gt 10000 ]; then
        echo "WARNING: GPU $GPU already has ${USED} MiB used."
        nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits \
            | awk -F',' '{printf "  GPU %s: used=%s MiB  free=%s MiB\n", $1, $2, $3}'
        if [ -t 0 ]; then
            read -p "Continue anyway? [y/N] " CONFIRM
            [[ "$CONFIRM" =~ ^[Yy]$ ]] || exit 1
        else
            echo "(non-interactive mode: continuing)"
        fi
    fi
done

# Stop stale Ray cluster
echo "Cleaning up any stale Ray processes..."
$CONDA_PYTHON -m ray stop --force 2>/dev/null || true
sleep 2

# Translate env GPU list into Hydra placement override.
# Hydra/RLinf accepts either:
#   - A range string  : "3-7"
#   - A comma list    : "[3,5,7]"
# Pass the raw input through (genark_eval_only.yaml uses range form by default).
ENV_PLACEMENT="$ENV_GPUS"

# ── Launch ─────────────────────────────────────────────────────────────────
cd "$RLINF_ROOT"

EMBODIED_PATH="$EMBODIED_PATH" \
TORCHDYNAMO_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$CONDA_PYTHON examples/embodiment/train_embodied_agent.py \
    --config-name genark_eval_only \
    "cluster.component_placement.rollout.placement=$ROLLOUT_GPU" \
    "cluster.component_placement.actor.placement=$ROLLOUT_GPU" \
    "cluster.component_placement.env.placement=$ENV_PLACEMENT" \
    "env.eval.total_num_envs=$NUM_ENVS" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Done. Full log: $LOG_FILE"
    echo "  Results:  $RLINF_ROOT/../results/genark_eval/"
    echo "  Metrics:  /home/clk/workspace/results/genark_eval/avg_metrics.json"
else
    echo "  FAILED (exit $EXIT_CODE). Full log: $LOG_FILE"
    echo "  Last 30 lines:"
    tail -30 "$LOG_FILE"
fi
echo "=================================================="
exit $EXIT_CODE

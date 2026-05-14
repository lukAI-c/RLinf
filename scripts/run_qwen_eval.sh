#!/bin/bash
# QwenNavPolicy zero-shot eval (lavira-style 4-dir + JSON)
#
# Usage:
#   bash scripts/run_qwen_eval.sh [ROLLOUT_GPU] [ENV_GPUS] [TOTAL_ENVS] [SCENE_OFFSET]
#
# Examples:
#   bash scripts/run_qwen_eval.sh                     # defaults: rollout=4, env=5, 4 envs, scene 0
#   bash scripts/run_qwen_eval.sh 4 5 4 0             # 1 worker on 1 scene
#   bash scripts/run_qwen_eval.sh 2 3,4,5 30 0        # 3 workers, scenes 0-2 in pass 0
#
# All placement values are PHYSICAL GPU indices.

set -e

ROLLOUT_GPU=${1:-4}
ENV_GPUS=${2:-5}
TOTAL_ENVS=${3:-4}
SCENE_OFFSET=${4:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_PYTHON="/home/clk/miniconda3/envs/genesis/bin/python"
EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"

# Pre-flight cleanup
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "train_embodied_agent" 2>/dev/null || true
sleep 2
find "$RLINF_ROOT/rlinf" -name "*.pyc" -delete 2>/dev/null || true

LOG_FILE="/tmp/qwen_eval_$(date +%Y%m%d_%H%M%S).log"
echo "============================================================"
echo "  QwenNavPolicy Zero-Shot Eval"
echo "  Rollout GPU    : $ROLLOUT_GPU"
echo "  Env GPUs       : $ENV_GPUS"
echo "  Total envs     : $TOTAL_ENVS"
echo "  Scene offset   : $SCENE_OFFSET"
echo "  Log            : $LOG_FILE"
echo "============================================================"

cd "$RLINF_ROOT"
EMBODIED_PATH="$EMBODIED_PATH" \
TORCHDYNAMO_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$CONDA_PYTHON examples/embodiment/train_embodied_agent.py \
    --config-name genark_eval_qwen \
    "cluster.component_placement.rollout.placement=$ROLLOUT_GPU" \
    "cluster.component_placement.actor.placement=$ROLLOUT_GPU" \
    "cluster.component_placement.env.placement=$ENV_GPUS" \
    "env.eval.total_num_envs=$TOTAL_ENVS" \
    "env.eval.scene_offset=$SCENE_OFFSET" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Eval finished. Results in /home/clk/workspace/results/genark_eval_qwen/"
else
    echo "  Eval FAILED (exit $EXIT_CODE). Check log: $LOG_FILE"
fi
echo "============================================================"
exit $EXIT_CODE

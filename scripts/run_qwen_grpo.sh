#!/bin/bash
# QwenNavPolicy GRPO training (Path A: RLinf embodied GRPO)
#
# Usage:
#   bash scripts/run_qwen_grpo.sh [ROLLOUT_GPU] [ENV_GPUS]
#
# Examples:
#   bash scripts/run_qwen_grpo.sh          # defaults: rollout+actor=2, env=3-5
#   bash scripts/run_qwen_grpo.sh 2 3,4,5  # explicit GPU assignment

set -e

ROLLOUT_GPU=${1:-2}
ENV_GPUS=${2:-3-5}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_PYTHON="/home/clk/miniconda3/envs/genesis/bin/python"
EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"

pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "train_embodied_agent" 2>/dev/null || true
sleep 2
find "$RLINF_ROOT/rlinf" -name "*.pyc" -delete 2>/dev/null || true

LOG_FILE="/tmp/qwen_grpo_$(date +%Y%m%d_%H%M%S).log"
echo "============================================================"
echo "  QwenNavPolicy GRPO Training"
echo "  Rollout+Actor GPU : $ROLLOUT_GPU"
echo "  Env GPUs          : $ENV_GPUS"
echo "  Log               : $LOG_FILE"
echo "============================================================"

cd "$RLINF_ROOT"
EMBODIED_PATH="$EMBODIED_PATH" \
TORCHDYNAMO_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$CONDA_PYTHON examples/embodiment/train_embodied_agent.py \
    --config-name genark_grpo_qwen \
    "cluster.component_placement.rollout,actor.placement=$ROLLOUT_GPU" \
    "cluster.component_placement.actor.placement=$ROLLOUT_GPU" \
    "cluster.component_placement.env.placement=$ENV_GPUS" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Training finished. Results in /home/clk/workspace/results/genark_grpo_qwen/"
else
    echo "  Training FAILED (exit $EXIT_CODE). Check log: $LOG_FILE"
fi
echo "============================================================"
exit $EXIT_CODE

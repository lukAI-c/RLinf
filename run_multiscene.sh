#!/bin/bash
# Launch multiscene GRPO training on GPUs 4,5,6,7.
#
# GPU assignment:
#   4,5  →  actor + rollout (Qwen3.5-4B FSDP 2-way)
#   6,7  →  env worker + 2 GenesisSceneActors (1 GPU each via Ray)
#
# Usage:
#   bash run_multiscene.sh               # start fresh
#   bash run_multiscene.sh --resume      # resume from latest checkpoint

set -e

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
EMBODIED_PATH="${SCRIPT_DIR}/examples/embodiment"
SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export EMBODIED_PATH
export REPO_PATH="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"

# Genesis / EGL / simulator env
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export GENESIS_HEADLESS=1

# Restrict CUDA to GPUs 4,5,6,7 (our own free cards) so Genesis EGL never
# time-slices with other users' jobs. env worker packs K=8 scene actors onto
# GPUs 4,5 (scenes_per_gpu=4); actor/rollout use 6,7.
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Memory / stability env (mirrors run_qwen_rft.sh) — expandable_segments avoids
# fragmentation OOM when actor/rollout share a GPU with other jobs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT_MS=3600000

CONFIG_NAME="genark_grpo_qwen_multiscene"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/train.log"

RESUME_ARGS=""
if [[ "$1" == "--resume" ]]; then
    LATEST=$(ls -td /home/clk/workspace/results/genark_grpo_qwen_multiscene/checkpoints/epoch_* 2>/dev/null | head -1)
    if [[ -n "$LATEST" ]]; then
        echo "[resume] latest checkpoint: ${LATEST}"
        RESUME_ARGS="runner.resume_dir=${LATEST}"
    else
        echo "[resume] no checkpoint found, starting fresh"
    fi
fi

PYTHON=${PYTHON:-/home/clk/miniconda3/envs/genesis/bin/python}
CMD="${PYTHON} ${SRC_FILE} \
    --config-path ${EMBODIED_PATH}/config/ \
    --config-name ${CONFIG_NAME} \
    runner.logger.log_path=${LOG_DIR} \
    ${RESUME_ARGS}"

echo "============================================================"
echo "  Multi-scene GRPO training"
echo "  Config:  ${CONFIG_NAME}"
echo "  GPUs:    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  Log:     ${LOGFILE}"
echo "============================================================"
echo "${CMD}" | tee "${LOGFILE}"
echo "------------------------------------------------------------"

# Run and tee to both terminal and log file
exec ${CMD} 2>&1 | tee -a "${LOGFILE}"

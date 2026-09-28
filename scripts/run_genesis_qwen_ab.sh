#!/usr/bin/env bash
set -euo pipefail

# Strict Genesis side of the frozen QwenNavPolicy A/B evaluation.
# Ray sees all physical GPUs in the existing node resource view, so these are
# physical hardware ranks, not indices into CUDA_VISIBLE_DEVICES.
# Physical GPU 3: QwenNavPolicy/vLLM; physical GPU 4: Genesis scene renderer.

REPO=/home/clk/workspace/RLinf
PYTHON=/home/clk/miniconda3/envs/genesis-vllm/bin/python
LOG_DIR="${REPO}/logs/$(date +%Y%m%d-%H%M%S)-genesis-qwennav-frozen"
mkdir -p "${LOG_DIR}"
echo "${LOG_DIR}" > "${REPO}/logs/latest_genesis_qwen_frozen.txt"

export EMBODIED_PATH="${REPO}/examples/embodiment"
export REPO_PATH="${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export GENESIS_HEADLESS=1
# Keep the full Ray hardware-rank namespace visible. The YAML placement above
# is what pins actors to physical ranks 3 and 4.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT_MS=3600000
export RLINF_RAY_LOCAL=1
export RAY_TMPDIR="${RAY_TMPDIR:-/home/clk/workspace/ray_tmp}"
export TMPDIR="${TMPDIR:-/home/clk/workspace/tmp}"
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"

exec "${PYTHON}" "${REPO}/examples/embodiment/train_embodied_agent.py" \
  --config-path "${REPO}/examples/embodiment/config/" \
  --config-name genesis_eval_qwen_frozen \
  runner.logger.log_path="${LOG_DIR}" \
  2>&1 | tee "${LOG_DIR}/eval.log"

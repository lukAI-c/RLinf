#!/usr/bin/env bash
set -euo pipefail

# One-episode, source-vs-RLinf Habitat parity trace.  This intentionally uses
# lhx's untouched evaluator and RLinf's normal frozen-policy launcher; the
# only additions are isolated output directories and audit capture.

REPO=/home/clk/workspace/RLinf
LHX_REPO=/home/lhx/workspace/lavira-code
VLLM=/home/clk/miniconda3/envs/genesis-vllm/bin/vllm
CHECKPOINT=/home/lhx/workspace/test_model/output/qwen3.5_v2_unfreeze_vit_nav_target/v0-20260709-081633/checkpoint-1200
RUN_ROOT="${RUN_ROOT:-${REPO}/logs/$(date +%Y%m%d-%H%M%S)-lhx-rlinf-decision-alignment-ep259}"

VLLM_GPU="${VLLM_GPU:-0}"
LHX_SIM_GPU="${LHX_SIM_GPU:-1}"
RLINF_GPU="${RLINF_GPU:-3}"
RLINF_HABITAT_GPU="${RLINF_HABITAT_GPU:-4}"
VLLM_PORT="${VLLM_PORT:-8889}"
RLINF_HABITAT_PORT="${RLINF_HABITAT_PORT:-18780}"
EPISODE_ID="${EPISODE_ID:-259}"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse existing alignment root: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p \
  "${RUN_ROOT}/lhx/data/logs" \
  "${RUN_ROOT}/lhx/data/checkpoints" \
  "${RUN_ROOT}/lhx/logs" \
  "${RUN_ROOT}/lhx/saved_rgb" \
  "${RUN_ROOT}/rlinf"

VLLM_LOG="${RUN_ROOT}/vllm.log"
LHX_LOG="${RUN_ROOT}/lhx/eval.log"
RLINF_LOG="${RUN_ROOT}/rlinf/eval.log"
LHX_CONTAINER="lhx-decision-align-${$}"

cleanup() {
  docker rm -f "${LHX_CONTAINER}" >/dev/null 2>&1 || true
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${VLLM}" serve "${CHECKPOINT}" \
  --served-model-name "${CHECKPOINT}" \
  --port "${VLLM_PORT}" \
  --trust-remote-code \
  --max-model-len 32768 \
  --mm-processor-kwargs '{"max_pixels":122500,"min_pixels":4096}' \
  --limit-mm-per-prompt '{"image":100}' \
  --mm-encoder-tp-mode data \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --default-chat-template-kwargs '{"enable_thinking":false,"add_vision_id":true}' \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

until rg -q "Application startup complete|Uvicorn running" "${VLLM_LOG}"; do
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    tail -100 "${VLLM_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done

# Preserve lhx source and its complete data tree as read-only.  Overlay only
# paths the evaluator writes, otherwise an empty data/ mount hides its
# GroundedSAM assets and other source-side resources.
docker run --rm -d \
  --name "${LHX_CONTAINER}" \
  --network host \
  --workdir /root/lhx-lavira \
  --gpus "\"device=${LHX_SIM_GPU}\"" \
  -e PYTHONPATH=/root/lhx-lavira:/root/GroundingDINO \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  -v "${LHX_REPO}:/root/lhx-lavira:ro" \
  -v /home/clk/workspace/third_party/GroundingDINO_py38:/root/GroundingDINO:ro \
  -v /home/clk/.cache/huggingface:/root/.cache/huggingface:ro \
  -v "${RUN_ROOT}/lhx/data/logs:/root/lhx-lavira/data/logs" \
  -v "${RUN_ROOT}/lhx/data/checkpoints:/root/lhx-lavira/data/checkpoints" \
  -v /home/nvme01/uni-lavira/data/grounded_sam:/root/lhx-lavira/data/grounded_sam:ro \
  -v "${RUN_ROOT}/lhx/logs:/root/lhx-lavira/logs" \
  -v "${RUN_ROOT}/lhx/saved_rgb:/root/lhx-lavira/saved_rgb_images_navtarget_ckpt1200_failure_analysis_0712" \
  -v /home/nvme01/uni-lavira/data/scene_datasets:/root/data/scene_datasets:ro \
  -v /home/lhx/workspace/opennav_r2r_100_bertidx.json.gz:/root/data/opennav_r2r_100_bertidx.json.gz:ro \
  -v /home/lhx/workspace/opennav_r2r_100_gt.json.gz:/root/data/opennav_r2r_100_gt.json.gz:ro \
  lavira:v4 \
  /root/miniconda/envs/lavira/bin/python3.8 run_mp.py \
  --exp_name lhx_alignment \
  --run-type eval \
  --exp-config vlnce_baselines/config/r2r.yaml \
  --nprocesses 1 \
  --debug-episodes "${EPISODE_ID}" \
  NUM_ENVIRONMENTS 1 \
  TRAINER_NAME ZS-Evaluator-mp \
  TORCH_GPU_IDS '[0]' \
  SIMULATOR_GPU_IDS '[0]' \
  TASK_CONFIG.DATASET.DATA_PATH /root/data/opennav_r2r_100_bertidx.json.gz \
  TASK_CONFIG.DATASET.SCENES_DIR /root/data/scene_datasets/ \
  TASK_CONFIG.TASK.NDTW.GT_PATH /root/data/opennav_r2r_100_gt.json.gz \
  TASK_CONFIG.TASK.SDTW.GT_PATH /root/data/opennav_r2r_100_gt.json.gz \
  TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 300 \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  TASK_CONFIG.SIMULATOR.AGENT_0.RADIUS 0.1 \
  EVAL.EPISODE_COUNT -1 \
  >"${LHX_LOG}" 2>&1 &

docker logs -f "${LHX_CONTAINER}" >>"${LHX_LOG}" 2>&1 &
LHX_LOG_PID=$!

EPISODE_ID="${EPISODE_ID}" \
RLINF_GPU="${RLINF_GPU}" \
HABITAT_GPUS="${RLINF_HABITAT_GPU}" \
NUM_ENVS=1 \
PORT="${RLINF_HABITAT_PORT}" \
RUN_TAG="rlinf-lhx-alignment-ep${EPISODE_ID}" \
LOG_DIR="${RUN_ROOT}/rlinf" \
VIDEO_SAVE=true \
bash "${REPO}/scripts/run_habitat_qwen_ab.sh" >"${RLINF_LOG}" 2>&1 &
RLINF_PID=$!

wait "${RLINF_PID}"
wait "${LHX_LOG_PID}" || true

echo "Alignment run root: ${RUN_ROOT}"

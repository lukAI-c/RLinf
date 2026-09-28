#!/usr/bin/env bash
set -euo pipefail

# Run the unmodified LHX OpenNav100 evaluator as an external oracle.
# LHX source is mounted read-only; all generated files use an isolated run dir.

LHX_REPO="${LHX_REPO:-/home/lhx/workspace/lavira-code}"
MODEL="${MODEL:-/home/lhx/workspace/test_model/output/qwen3.5_v2_unfreeze_vit_nav_target/v0-20260709-081633/checkpoint-1200}"
VLLM_BIN="${VLLM_BIN:-/home/lhx/workspace/miniconda3/envs/qwen3.5/bin/vllm}"
IMAGE="${IMAGE:-lavira:v4}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPUS="${EVAL_GPUS:-0,3,5,7}"
NPROCESSES="${NPROCESSES:-16}"
WAIT_FOR_PID="${WAIT_FOR_PID:-4079715}"
WAIT_CONTAINER_PREFIX="${WAIT_CONTAINER_PREFIX:-rlinf-habitat-shared-4079715-}"
RUN_ROOT="${RUN_ROOT:-/home/clk/workspace/RLinf/logs/$(date +%Y%m%d-%H%M%S)-lhx-original-open100-ckpt1200}"
EXP_NAME="${EXP_NAME:-lhx_original_$(date +%m%d-%H%M%S)}"
CONTAINER="lhx-original-open100-${$}"
VLLM_PID=""

mkdir -p \
  "${RUN_ROOT}/runtime_data/checkpoints" \
  "${RUN_ROOT}/runtime_data/datasets" \
  "${RUN_ROOT}/runtime_data/scene_datasets" \
  "${RUN_ROOT}/runtime_data/grounded_sam" \
  "${RUN_ROOT}/runtime_data/logs/running_log" \
  "${RUN_ROOT}/runtime_data/logs/eval_results" \
  "${RUN_ROOT}/runtime_data/tensorboard_dirs" \
  "${RUN_ROOT}/video" \
  "${RUN_ROOT}/saved_rgb"

cleanup() {
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  if [[ -n "${VLLM_PID}" ]]; then
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "[lhx-oracle] result=${RUN_ROOT}"
echo "[lhx-oracle] waiting for RLinf pid=${WAIT_FOR_PID} and containers=${WAIT_CONTAINER_PREFIX}*"
while kill -0 "${WAIT_FOR_PID}" >/dev/null 2>&1 || \
      docker ps --format '{{.Names}}' | grep -q "^${WAIT_CONTAINER_PREFIX}"; do
  sleep 30
done

# The requested topology needs one empty policy GPU and four evaluator GPUs.
# Wait instead of stealing resources from a newly-started job.
while true; do
  policy_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${POLICY_GPU}" | tr -d ' ')
  eval_busy=0
  IFS=',' read -r -a eval_gpu_list <<< "${EVAL_GPUS}"
  for gpu in "${eval_gpu_list[@]}"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
    if (( used > 12000 )); then
      eval_busy=1
    fi
  done
  if (( policy_used < 12000 && eval_busy == 0 )); then
    break
  fi
  echo "[lhx-oracle] waiting for GPU availability policy_used=${policy_used}MiB eval_busy=${eval_busy}"
  sleep 60
done

if ss -ltn | grep -q ':8889 '; then
  echo "[lhx-oracle] refusing to reuse occupied port 8889" >&2
  exit 2
fi

echo "[lhx-oracle] starting original checkpoint-1200 vLLM on physical GPU ${POLICY_GPU}"
CUDA_VISIBLE_DEVICES="${POLICY_GPU}" \
VLLM_USE_V1=0 \
VLLM_IMAGE_MAX_PIXELS=122500 \
TRANSFORMERS_OFFLINE=1 \
"${VLLM_BIN}" serve "${MODEL}" \
  --served-model-name "${MODEL}" \
  --port 8889 \
  --trust-remote-code \
  --max-model-len 32768 \
  --mm-processor-kwargs '{"max_pixels": 122500, "min_pixels": 4096}' \
  --limit-mm-per-prompt '{"image": 100}' \
  --mm-encoder-tp-mode data \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --default-chat-template-kwargs '{"enable_thinking": false, "add_vision_id": true}' \
  >"${RUN_ROOT}/vllm.log" 2>&1 &
VLLM_PID=$!
echo "${VLLM_PID}" > "${RUN_ROOT}/vllm.pid"

# Qwen3.5 multimodal profiling can take well over six minutes on a cold
# torch.compile cache. This wait only gates evaluator startup; it does not
# change the model or LHX evaluation semantics.
for _ in $(seq 1 600); do
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "[lhx-oracle] vLLM exited before readiness" >&2
    tail -100 "${RUN_ROOT}/vllm.log" >&2 || true
    exit 1
  fi
  if curl -fsS http://127.0.0.1:8889/v1/models > "${RUN_ROOT}/vllm_models.json"; then
    break
  fi
  sleep 2
done
if [[ ! -s "${RUN_ROOT}/vllm_models.json" ]]; then
  echo "[lhx-oracle] vLLM readiness timeout" >&2
  tail -100 "${RUN_ROOT}/vllm.log" >&2 || true
  exit 1
fi

echo "[lhx-oracle] starting unmodified LHX evaluator: GPUs=${EVAL_GPUS}, workers=${NPROCESSES}"
docker run --rm \
  --name "${CONTAINER}" \
  --network host \
  --gpus "\"device=${EVAL_GPUS}\"" \
  --workdir /root/lavira-code \
  -e GLOG_minloglevel=0 \
  -e MAGNUM_LOG=verbose \
  -e EGL_PLATFORM=surfaceless \
  -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH=/root/lavira-code:/root/lavira-code/GroundingDINO \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  -v "${LHX_REPO}:/root/lavira-code:ro" \
  -v /home/nvme01/uni-lavira/model/GroundingDINO:/root/lavira-code/GroundingDINO:ro \
  -v "${RUN_ROOT}/runtime_data:/root/lavira-code/data" \
  -v /home/nvme01/uni-lavira/data/datasets:/root/lavira-code/data/datasets:ro \
  -v /home/nvme01/uni-lavira/data/scene_datasets:/root/lavira-code/data/scene_datasets:ro \
  -v /home/nvme01/uni-lavira/data/grounded_sam:/root/lavira-code/data/grounded_sam:ro \
  -v /home/lhx/workspace/opennav_r2r_100_bertidx.json.gz:/home/lhx/workspace/opennav_r2r_100_bertidx.json.gz:ro \
  -v /home/lhx/workspace/opennav_r2r_100_gt.json.gz:/home/lhx/workspace/opennav_r2r_100_gt.json.gz:ro \
  -v "${RUN_ROOT}/video:/root/lavira-code/data/logs/video" \
  -v "${RUN_ROOT}/saved_rgb:/root/lavira-code/saved_rgb_images_navtarget_ckpt1200_failure_analysis_0712" \
  "${IMAGE}" \
  /root/miniconda/envs/lavira/bin/python run_mp.py \
    --exp_name "${EXP_NAME}" \
    --run-type eval \
    --exp-config vlnce_baselines/config/r2r.yaml \
    --nprocesses "${NPROCESSES}" \
    NUM_ENVIRONMENTS 1 \
    TRAINER_NAME ZS-Evaluator-mp \
    TORCH_GPU_IDS '[0,1,2,3]' \
    SIMULATOR_GPU_IDS '[0,1,2,3]' \
    TASK_CONFIG.DATASET.DATA_PATH /home/lhx/workspace/opennav_r2r_100_bertidx.json.gz \
    TASK_CONFIG.TASK.NDTW.GT_PATH /home/lhx/workspace/opennav_r2r_100_gt.json.gz \
    TASK_CONFIG.TASK.SDTW.GT_PATH /home/lhx/workspace/opennav_r2r_100_gt.json.gz \
  2>&1 | tee "${RUN_ROOT}/lhx_eval.log"

echo "[lhx-oracle] completed result=${RUN_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

# GenArk-style Habitat evaluation topology
#
#   Habitat server 0 -- EnvWorker 0 --\
#   Habitat server 1 -- EnvWorker 1 ----> one shared rollout rank / one vLLM
#   Habitat server 2 -- EnvWorker 2 ----> model (batched observations)
#   Habitat server 3 -- EnvWorker 3 --/
#
# Habitat simulators remain isolated in lavira:v4. EnvWorkers are lightweight
# RPC clients; RLinf's existing CommMapper performs the many-to-one merge. This
# avoids the old batch launcher, which created one complete RLinf/vLLM process
# per independent evaluator.

REPO=/home/clk/workspace/RLinf
PYTHON=/home/clk/miniconda3/envs/genesis-vllm/bin/python
IMAGE=lavira:v4
EPISODES_FILE="${EPISODES_FILE:-/home/nvme03/lck/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json}"
CHECKPOINT="${CHECKPOINT:-/home/lhx/workspace/test_model/output/qwen3.5_v2_unfreeze_vit_nav_target/v0-20260709-081633/checkpoint-1200}"

POLICY_GPU="${POLICY_GPU:-0}"
HABITAT_GPUS="${HABITAT_GPUS:-1,2,3,4}"
PORT_BASE="${PORT_BASE:-18970}"
EPISODE_LIMIT="${EPISODE_LIMIT:-0}"
SLOTS_PER_WORKER="${SLOTS_PER_WORKER:-4}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS:-}"
VIDEO_SAVE="${VIDEO_SAVE:-false}"
MAP_VISUALIZATION="${MAP_VISUALIZATION:-false}"
SAVE_RAW_EVERY="${SAVE_RAW_EVERY:-0}"
GSAM_REMOTE_ENABLED="${GSAM_REMOTE_ENABLED:-true}"
GSAM_PORT_BASE="${GSAM_PORT_BASE:-19070}"
RUN_TAG="${RUN_TAG:-habitat-qwennav-shared-open100}"
RUN_ROOT="${RUN_ROOT:-${REPO}/logs/$(date +%Y%m%d-%H%M%S)-${RUN_TAG}}"

IFS=',' read -r -a HABITAT_GPU_LIST <<< "${HABITAT_GPUS}"
NUM_WORKERS="${#HABITAT_GPU_LIST[@]}"
if [[ "${NUM_WORKERS}" -lt 1 ]]; then
  echo "HABITAT_GPUS must contain at least one physical GPU" >&2
  exit 2
fi
for gpu in "${HABITAT_GPU_LIST[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid physical GPU id in HABITAT_GPUS: ${gpu}" >&2
    exit 2
  fi
done
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse result directory: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

# Build GenArk's unit of scheduling: one job per scene.  EnvWorkers claim jobs
# dynamically from the shared state file, then run that scene's episodes in
# fixed-size batches.  This preserves one scene owner at a time while keeping
# RLinf's communication batch dimensions stable.
readarray -t QUEUE_META < <(
  "${PYTHON}" - "${EPISODES_FILE}" "${RUN_ROOT}/scene_queue.json" \
    "${RUN_ROOT}/scene_queue_state.json" "${SLOTS_PER_WORKER}" \
    "${EPISODE_LIMIT}" <<'PY'
import gzip
import json
import sys
from collections import OrderedDict
from pathlib import Path

source, destination, state_path, slots, limit = sys.argv[1:]
slots, limit = int(slots), int(limit)
path = Path(source)
opener = gzip.open if path.suffix == ".gz" else open
with opener(path, "rt", encoding="utf-8") as handle:
    payload = json.load(handle)
episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
if limit:
    episodes = episodes[:limit]
ids = [str(row["episode_id"]) for row in episodes]
if not ids:
    raise SystemExit("episode dataset is empty")
if len(ids) != len(set(ids)):
    raise SystemExit("shared evaluation requires globally unique episode ids")
scenes = OrderedDict()
for row in episodes:
    scenes.setdefault(str(row["scene_id"]), []).append(str(row["episode_id"]))
jobs = [
    {"scene_id": scene_id, "episode_ids": episode_ids}
    for scene_id, episode_ids in scenes.items()
]
Path(destination).write_text(json.dumps({
    "schema_version": 2,
    "source": str(Path(source).resolve()),
    "slots_per_worker": slots,
    "episode_count": len(ids),
    "scene_count": len(jobs),
    "scene_jobs": jobs,
}, indent=2) + "\n")
Path(state_path).write_text(json.dumps({
    "schema_version": 1,
    "next_scene": 0,
    "claims": [],
}, indent=2) + "\n")
total_batches = sum((len(job["episode_ids"]) + slots - 1) // slots for job in jobs)
print(len(ids))
print(len(jobs))
print(max(len(job["episode_ids"]) for job in jobs))
print(total_batches)
PY
)
EPISODE_COUNT="${QUEUE_META[0]}"
SCENE_COUNT="${QUEUE_META[1]}"
MAX_SCENE_EPISODES="${QUEUE_META[2]}"
TOTAL_SCENE_BATCHES="${QUEUE_META[3]}"
# Conservative only; the eval termination sentinel exits as soon as the
# shared queue and every active scene batch are empty.
MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS:-$((MAX_EPISODE_STEPS * TOTAL_SCENE_BATCHES))}"

# EnvWorker processes do not render or run GroundedSAM. They share one Ray GPU
# resource only to obtain four process ranks; the external Docker servers use
# the physical devices listed in HABITAT_GPUS.
ENV_PROCESS_GPU="${ENV_PROCESS_GPU:-${HABITAT_GPU_LIST[0]}}"
ENV_PLACEMENT="${ENV_PROCESS_GPU}:0-$((NUM_WORKERS - 1))"

export EMBODIED_PATH="${REPO}/examples/embodiment"
export REPO_PATH="${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export GENESIS_HEADLESS=1
# Placement strings use physical GPU ids, so expose the full physical namespace.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT_MS=3600000
export RLINF_RAY_LOCAL=1
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rlh_shared_${$}}"
export TMPDIR="${TMPDIR:-/tmp/rlh_shared_tmp_${$}}"
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"

CONTAINERS=()
LOG_PIDS=()
GSAM_PIDS=()
cleanup() {
  for pid in "${LOG_PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  for container in "${CONTAINERS[@]:-}"; do
    docker rm -f "${container}" >/dev/null 2>&1 || true
  done
  for pid in "${GSAM_PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

echo "Shared Habitat evaluation"
echo "  policy/vLLM physical GPU: ${POLICY_GPU} (one model copy)"
echo "  Habitat physical GPUs:    ${HABITAT_GPUS} (${NUM_WORKERS} servers)"
echo "  EnvWorker placement:      ${ENV_PLACEMENT} (${NUM_WORKERS} RPC clients)"
echo "  scene jobs:               ${SCENE_COUNT} (dynamic refill)"
echo "  episodes:                 ${EPISODE_COUNT} (max scene=${MAX_SCENE_EPISODES})"
echo "  slots per scene worker:   ${SLOTS_PER_WORKER}"
echo "  result:                   ${RUN_ROOT}"

for ((rank=0; rank<NUM_WORKERS; rank++)); do
  gpu="${HABITAT_GPU_LIST[$rank]}"
  port=$((PORT_BASE + rank))
  container="rlinf-habitat-shared-${$}-${rank}"
  shard_dir="${RUN_ROOT}/shard_${rank}"
  mkdir -p "${shard_dir}"
  : > "${shard_dir}/habitat-server.log"
  CONTAINERS+=("${container}")

  docker run --rm -d \
    --name "${container}" \
    --network host \
    --workdir /root \
    --gpus "\"device=${gpu}\"" \
    -e PYTHONPATH=/root/lavira-source \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    -e http_proxy=http://127.0.0.1:7892 \
    -e https_proxy=http://127.0.0.1:7892 \
    -v "${REPO}/scripts/habitat_qwen_server.py:/root/habitat_qwen_server.py:ro" \
    -v /home/clk/workspace/lavira-rft:/root/lavira-source:ro \
    -v /home/nvme01/uni-lavira/data/datasets:/root/data/datasets:ro \
    -v /home/nvme01/uni-lavira/data/scene_datasets:/root/data/scene_datasets:ro \
    -v /home/nvme01/uni-lavira/data/grounded_sam:/root/data/grounded_sam:ro \
    -v /home/lhx/workspace/opennav_r2r_100_bertidx.json.gz:/root/data/opennav_r2r_100_bertidx.json.gz:ro \
    -v /home/lhx/workspace/opennav_r2r_100_gt.json.gz:/root/data/opennav_r2r_100_gt.json.gz:ro \
    -v "${shard_dir}:/root/habitat_output" \
    "${IMAGE}" \
    /root/miniconda/envs/lavira/bin/python3.8 /root/habitat_qwen_server.py \
    --lavira-root /root/lavira-source \
    --dataset-path /root/data/opennav_r2r_100_bertidx.json.gz \
    --gt-path /root/data/opennav_r2r_100_gt.json.gz \
    --scenes-dir /root/data/scene_datasets \
    --num-envs "${SLOTS_PER_WORKER}" \
    --gpu-id 0 \
    --port "${port}" \
    --width 640 --height 480 \
    --hfov 79.0 --camera-height 0.88 --depth-max 5.0 \
    --forward-step 0.25 --turn-angle 30 \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --max-decisions 0 --success-distance 3.0 \
    --metrics-path /root/habitat_output/habitat_metrics.json \
    >/dev/null

  docker logs -f "${container}" > "${shard_dir}/habitat-server.log" 2>&1 &
  LOG_PIDS+=("$!")
done

GSAM_ENDPOINTS="[]"
if [[ "${GSAM_REMOTE_ENABLED}" == "true" ]]; then
  endpoint_values=()
  for ((rank=0; rank<NUM_WORKERS; rank++)); do
    gpu="${HABITAT_GPU_LIST[$rank]}"
    port=$((GSAM_PORT_BASE + rank))
    service_log="${RUN_ROOT}/shard_${rank}/grounded-sam-server.log"
    endpoint_values+=("\"127.0.0.1:${port}\"")
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
      "${REPO}/scripts/grounded_sam_server.py" \
      --port "${port}" \
      --dino-config-path "${REPO}/assets/grounded_sam/GroundingDINO_SwinT_OGC.py" \
      --dino-checkpoint-path "${REPO}/assets/grounded_sam/groundingdino_swint_ogc.pth" \
      --repvit-sam-checkpoint-path /home/nvme01/uni-lavira/data/grounded_sam/repvit_sam.pt \
      >"${service_log}" 2>&1 &
    GSAM_PIDS+=("$!")
  done
  GSAM_ENDPOINTS="[$(IFS=,; echo "${endpoint_values[*]}")]"

  for ((rank=0; rank<NUM_WORKERS; rank++)); do
    service_log="${RUN_ROOT}/shard_${rank}/grounded-sam-server.log"
    until rg -q "\[grounded-sam-server\] ready" "${service_log}"; do
      if ! kill -0 "${GSAM_PIDS[$rank]}" >/dev/null 2>&1; then
        echo "GroundedSAM service ${rank} exited before readiness" >&2
        cat "${service_log}" >&2 || true
        exit 1
      fi
      sleep 2
    done
  done
  echo "  GroundedSAM services:     ${GSAM_ENDPOINTS}"
fi

for ((rank=0; rank<NUM_WORKERS; rank++)); do
  log_file="${RUN_ROOT}/shard_${rank}/habitat-server.log"
  container="${CONTAINERS[$rank]}"
  until rg -q "\[habitat-qwen-server\] ready" "${log_file}"; do
    if ! docker ps --format '{{.Names}}' | rg -q "^${container}$"; then
      echo "Habitat server ${rank} exited before readiness" >&2
      cat "${log_file}" >&2 || true
      exit 1
    fi
    sleep 2
  done
done

# map_backend is already source in the frozen config. Do not override a key
# below rollout.model.lavira_runtime: that node is an interpolation of the
# complete actor runtime, and a nested Hydra override would replace it with a
# partial dictionary.
"${PYTHON}" "${REPO}/examples/embodiment/train_embodied_agent.py" \
  --config-path "${REPO}/examples/embodiment/config/" \
  --config-name habitat_eval_qwen_frozen \
  runner.logger.log_path="${RUN_ROOT}" \
  "cluster.component_placement.rollout.placement=${POLICY_GPU}" \
  "cluster.component_placement.actor.placement=${POLICY_GPU}" \
  "cluster.component_placement.env.placement=${ENV_PLACEMENT}" \
  actor.model.model_path="${CHECKPOINT}" \
  algorithm.eval_rollout_epoch=1 \
  rollout.max_num_seqs="${MAX_NUM_SEQS:-4}" \
  env.eval.num_envs="${SLOTS_PER_WORKER}" \
  env.eval.total_num_envs="$((NUM_WORKERS * SLOTS_PER_WORKER))" \
  env.eval.max_steps_per_rollout_epoch="${MAX_ROLLOUT_STEPS}" \
  env.eval.max_episode_steps="${MAX_EPISODE_STEPS}" \
  env.eval.video_cfg.save_video="${VIDEO_SAVE}" \
  env.eval.shared_rollout.enabled=true \
  env.eval.shared_rollout.server_port_base="${PORT_BASE}" \
  env.eval.dynamic_scene_queue.enabled=true \
  env.eval.dynamic_scene_queue.manifest_path="${RUN_ROOT}/scene_queue.json" \
  env.eval.dynamic_scene_queue.state_path="${RUN_ROOT}/scene_queue_state.json" \
  ++actor.model.grounded_sam.remote.enabled="${GSAM_REMOTE_ENABLED}" \
  ++actor.model.grounded_sam.remote.endpoints="${GSAM_ENDPOINTS}" \
  ++actor.model.grounded_sam.remote.slots_per_service="${SLOTS_PER_WORKER}" \
  ++actor.model.grounded_sam.remote.timeout_s=300.0 \
  ++actor.model.lavira_runtime.map_visualization.enabled="${MAP_VISUALIZATION}" \
  ++actor.model.lavira_runtime.map_visualization.output_dir="${RUN_ROOT}/lavira_alignment" \
  ++actor.model.lavira_runtime.map_visualization.save_raw_every="${SAVE_RAW_EVERY}" \
  2>&1 | tee "${RUN_ROOT}/eval.log"
status=${PIPESTATUS[0]}
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

INPUT_FILES=()
for ((rank=0; rank<NUM_WORKERS; rank++)); do
  metrics="${RUN_ROOT}/shard_${rank}/all_episode_metrics.json"
  if [[ ! -s "${metrics}" ]]; then
    echo "Missing metrics from Habitat shard ${rank}: ${metrics}" >&2
    exit 1
  fi
  INPUT_FILES+=("${metrics}")
done

"${PYTHON}" "${REPO}/scripts/aggregate_aligned_nav_metrics.py" \
  --input-files "${INPUT_FILES[@]}" \
  --output-dir "${RUN_ROOT}" \
  --expected-episodes "${EPISODE_COUNT}" \
  --source habitat-qwennav-shared \
  --max-episode-steps "${MAX_EPISODE_STEPS}"

echo "Shared Habitat evaluation complete: ${RUN_ROOT}/avg_metrics.json"

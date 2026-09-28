#!/usr/bin/env bash
set -euo pipefail

# Strict environment A/B baseline:
#   host physical GPU 3                   -> RLinf QwenNavPolicy/vLLM
#   container physical GPUs 2,6           -> 4 Habitat evaluator processes
# The existing LaViRA containers are never reused or stopped.

REPO=/home/clk/workspace/RLinf
IMAGE=lavira:v4
CONFIG_FILE="${REPO}/examples/embodiment/config/habitat_eval_qwen_frozen.yaml"
CHECKPOINT="/home/lhx/workspace/test_model/output/qwen3.5_v2_unfreeze_vit_nav_target/v0-20260709-081633/checkpoint-1200"
HFOV_DEG="79.0"
CAMERA_HEIGHT="0.88"
DEPTH_MAX="5.0"
RLINF_GPU="${RLINF_GPU:-3}"
HABITAT_GPUS="${HABITAT_GPUS:-2,6}"
EPISODE_ID="${EPISODE_ID:-259}"
EPISODE_IDS_OVERRIDE="${EPISODE_IDS_OVERRIDE:-}"
RUN_TAG="${RUN_TAG:-habitat-qwennav-frozen}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
# Match LaViRA's Habitat R2R horizon.  This counts real primitive simulator
# actions, including the 12 physical turns used by each panorama scan.
MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS:-${MAX_EPISODE_STEPS}}"
PORT="${PORT:-18770}"
VIDEO_SAVE="${VIDEO_SAVE:-true}"
MAP_BACKEND="${MAP_BACKEND:-source}"
SAVE_RAW_EVERY="${SAVE_RAW_EVERY:-0}"
# This launcher has one behavior contract: strict ZS_Evaluator_mp
# prompt/history/controller alignment. Legacy comparison modes are intentionally
# not exposed because their metrics cannot enter the Habitat A/B aggregate.
# A run may coexist with another Habitat evaluation.  Never remove a fixed
# global container name; callers can still provide CONTAINER explicitly.
_safe_run_tag="${RUN_TAG//[^a-zA-Z0-9_.-]/-}"
CONTAINER="${CONTAINER:-rlinf-qwennav-habitat-eval-${_safe_run_tag}-$$}"
# Docker renumbers the selected physical GPUs from zero.  Permit a one-GPU
# smoke/regression run by deriving the bridge's visible ids instead of always
# assuming the default two-device mapping ``0,1``.
if [[ -n "${HABITAT_INTERNAL_GPU_IDS:-}" ]]; then
  HABITAT_INTERNAL_GPU_IDS="${HABITAT_INTERNAL_GPU_IDS}"
else
  IFS=',' read -r -a _habitat_gpu_list <<< "${HABITAT_GPUS}"
  HABITAT_INTERNAL_GPU_IDS=""
  for _index in "${!_habitat_gpu_list[@]}"; do
    HABITAT_INTERNAL_GPU_IDS+="${HABITAT_INTERNAL_GPU_IDS:+,}${_index}"
  done
fi
PYTHON=/home/clk/miniconda3/envs/genesis-vllm/bin/python
# Same-scene batches at the end of a scene may contain fewer than four
# episodes.  The remote adapter requires episode_ids and num_envs to agree;
# derive the slot count for explicit batch runs instead of padding with an
# unrelated episode.
NUM_ENVS="${NUM_ENVS:-4}"
if [[ -n "${EPISODE_IDS_OVERRIDE}" ]]; then
  NUM_ENVS="$(${PYTHON} - "${EPISODE_IDS_OVERRIDE}" <<'PY'
import json
import sys
print(len(json.loads(sys.argv[1])))
PY
)"
fi
LOG_DIR="${LOG_DIR:-${REPO}/logs/$(date +%Y%m%d-%H%M%S)-${RUN_TAG}}"
mkdir -p "${LOG_DIR}"
echo "${LOG_DIR}" > "${REPO}/logs/latest_habitat_qwen_frozen.txt"
: > "${LOG_DIR}/habitat-server.log"

# Fail closed if the launcher and frozen-policy config drift apart. The batch
# driver compares this runtime record with its immutable root fingerprint
# before accepting metrics into the final OpenNav100 aggregate.
"${PYTHON}" - "${CONFIG_FILE}" "${LOG_DIR}/evaluation_runtime_contract.json" \
  "${CHECKPOINT}" "${HFOV_DEG}" "${CAMERA_HEIGHT}" "${DEPTH_MAX}" \
  "${MAX_EPISODE_STEPS}" "${MAX_ROLLOUT_STEPS}" "$0" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

config_path, output_path, checkpoint, hfov, camera_height, depth_max, max_steps, max_rollout, launcher = sys.argv[1:]
config = Path(config_path)
text = config.read_text()
required = (
    f"model_path: {checkpoint}",
    f"hfov_deg: {hfov}",
    f"camera_height: {camera_height}",
    "max_steps_per_rollout_epoch: 300",
    "max_episode_steps: 300",
)
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"frozen Habitat config violates strict contract: {missing}")
payload = {
    "schema_version": 1,
    "checkpoint": checkpoint,
    "hfov_deg": float(hfov),
    "camera_height_m": float(camera_height),
    "depth_max_m": float(depth_max),
    "max_episode_steps": int(max_steps),
    "max_rollout_steps": int(max_rollout),
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "launcher_sha256": hashlib.sha256(Path(launcher).read_bytes()).hexdigest(),
}
Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")
PY

export EMBODIED_PATH="${REPO}/examples/embodiment"
export REPO_PATH="${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export GENESIS_HEADLESS=1
# RLinf placement values are physical GPU IDs, not indices remapped through
# CUDA_VISIBLE_DEVICES. Keep the complete hardware-rank namespace visible and
# select RLINF_GPU explicitly in the placement overrides below.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT_MS=3600000
export RLINF_RAY_LOCAL=1
export RAY_TMPDIR="${RAY_TMPDIR:-/home/clk/workspace/ray_tmp}"
export TMPDIR="${TMPDIR:-/home/clk/workspace/tmp}"
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"

cleanup() {
  if [[ -n "${LOG_FOLLOW_PID:-}" ]]; then
    kill "${LOG_FOLLOW_PID}" >/dev/null 2>&1 || true
  fi
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
docker run --rm -d \
  --name "${CONTAINER}" \
  --network host \
  --workdir /root \
  --gpus "\"device=${HABITAT_GPUS}\"" \
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
  -v "${LOG_DIR}:/root/habitat_output" \
  "${IMAGE}" \
  /root/miniconda/envs/lavira/bin/python3.8 /root/habitat_qwen_server.py \
  --lavira-root /root/lavira-source \
  --dataset-path /root/data/opennav_r2r_100_bertidx.json.gz \
  --gt-path /root/data/opennav_r2r_100_gt.json.gz \
  --scenes-dir /root/data/scene_datasets \
  --num-envs "${NUM_ENVS}" \
  --gpu-ids "${HABITAT_INTERNAL_GPU_IDS}" \
  --port "${PORT}" \
    --width 640 \
    --height 480 \
  --hfov "${HFOV_DEG}" \
  --camera-height "${CAMERA_HEIGHT}" \
  --depth-max "${DEPTH_MAX}" \
  --forward-step 0.25 \
  --turn-angle 30 \
  --max-episode-steps "${MAX_EPISODE_STEPS}" \
  --max-decisions 0 \
  --success-distance 3.0 \
  --metrics-path /root/habitat_output/habitat_metrics.json \
  >/dev/null

docker logs -f "${CONTAINER}" >"${LOG_DIR}/habitat-server.log" 2>&1 &
LOG_FOLLOW_PID=$!

until rg -q "\[habitat-qwen-server\] ready" "${LOG_DIR}/habitat-server.log"; do
  if ! docker ps --format '{{.Names}}' | rg -q "^${CONTAINER}$"; then
    echo "Habitat server exited before becoming ready" >&2
    cat "${LOG_DIR}/habitat-server.log" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "Habitat server ready; starting RLinf frozen policy eval"
if [[ -n "${EPISODE_IDS_OVERRIDE}" ]]; then
  EPISODE_OVERRIDE=("env.eval.episode_ids=${EPISODE_IDS_OVERRIDE}")
else
  EPISODE_OVERRIDE=("env.eval.episode_id=${EPISODE_ID}")
fi
ALIGNMENT_OVERRIDES=(
  actor.model.json_retry_enabled=true
  rollout.model.json_retry_enabled=true
  ++actor.model.lavira_source_prompt_alignment=true
  ++rollout.model.lavira_source_prompt_alignment=true
  actor.model.lavira_runtime.max_steps_to_target=15
  "++actor.model.lavira_runtime.map_backend=${MAP_BACKEND}"
  "++rollout.model.lavira_runtime.map_backend=${MAP_BACKEND}"
  ++actor.model.lavira_runtime.map_visualization.enabled=true
  "++actor.model.lavira_runtime.map_visualization.output_dir=${LOG_DIR}/lavira_alignment"
  "++actor.model.lavira_runtime.map_visualization.save_raw_every=${SAVE_RAW_EVERY}"
)
"${PYTHON}" "${REPO}/examples/embodiment/train_embodied_agent.py" \
  --config-path "${REPO}/examples/embodiment/config/" \
  --config-name habitat_eval_qwen_frozen \
  runner.logger.log_path="${LOG_DIR}" \
  "cluster.component_placement.rollout.placement=${RLINF_GPU}" \
  "cluster.component_placement.actor.placement=${RLINF_GPU}" \
  "cluster.component_placement.env.placement=${RLINF_GPU}:0" \
  env.eval.server_port="${PORT}" \
  "${EPISODE_OVERRIDE[@]}" \
  env.eval.num_envs="${NUM_ENVS}" \
  env.eval.total_num_envs="${NUM_ENVS}" \
  env.eval.max_steps_per_rollout_epoch="${MAX_ROLLOUT_STEPS}" \
  env.eval.max_episode_steps="${MAX_EPISODE_STEPS}" \
  "env.eval.video_cfg.save_video=${VIDEO_SAVE}" \
  "${ALIGNMENT_OVERRIDES[@]}" \
  2>&1 | tee "${LOG_DIR}/eval.log"

status=${PIPESTATUS[0]}
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

# Direct single-batch runs must fail closed too.  The OpenNav batch driver
# performs the same check before copying a batch into its resume manifest.
if [[ -n "${EPISODE_IDS_OVERRIDE}" ]]; then
  "${PYTHON}" - "${LOG_DIR}/all_episode_metrics.json" "${EPISODE_IDS_OVERRIDE}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

metrics_path, expected_json = sys.argv[1:]
expected = [str(x) for x in json.loads(expected_json)]
path = Path(metrics_path)
if not path.is_file():
    raise SystemExit(f"missing Habitat metrics file: {path}")
payload = json.loads(path.read_text())
rows = payload.get("episodes", payload.get("trials", [])) if isinstance(payload, dict) else payload
actual = [str(row.get("episode_id")) for row in rows if isinstance(row, dict)]
trial_ids = [str(row.get("trial_id", "")) for row in rows if isinstance(row, dict)]
if Counter(actual) != Counter(expected) or len(set(trial_ids)) != len(rows):
    raise SystemExit(
        f"incomplete Habitat batch: expected={expected} actual={actual} trials={trial_ids}"
    )
print(f"Habitat batch complete: {len(rows)} episodes")
PY
fi

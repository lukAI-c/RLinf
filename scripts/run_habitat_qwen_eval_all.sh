#!/usr/bin/env bash
set -euo pipefail

# Run the frozen RLinf QwenNavPolicy on every episode in the OpenNav100 file.
# Each pass evaluates four distinct episodes in parallel.  The existing
# Habitat bridge and policy/evaluation code are reused unchanged; this script
# only supplies episode_ids and aggregates the per-pass JSON outputs.

REPO=/home/clk/workspace/RLinf
PYTHON=/home/clk/miniconda3/envs/genesis-vllm/bin/python
EPISODES_FILE=/home/nvme03/lck/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json
RUN_SCRIPT="${REPO}/scripts/run_habitat_qwen_ab.sh"
RUN_ROOT="${RUN_ROOT:-${REPO}/logs/$(date +%Y%m%d-%H%M%S)-habitat-qwen-open100-strict}"

RLINF_GPU="${RLINF_GPU:-3}"
HABITAT_GPUS="${HABITAT_GPUS:-0,4}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS:-${MAX_EPISODE_STEPS}}"
HABITAT_PORT_BASE="${HABITAT_PORT_BASE:-18870}"
GENESIS_TRIALS_FILE="${GENESIS_TRIALS_FILE:-${REPO}/logs/20260716-205812-genesis-pure-eval-opennav100/all_episode_trials.json}"
# Match GenArk's batch evaluator: episodes are grouped by scene first, then
# split into fixed-size vector-env batches.  Do not mix scenes in one batch;
# the Habitat bridge owns one scene state per slot and the reference evaluator
# keeps all parallel slots in the same scene.
SCENE_BATCH_SIZE="${SCENE_BATCH_SIZE:-4}"
CONFIG_FILE="${REPO}/examples/embodiment/config/habitat_eval_qwen_frozen.yaml"
CHECKPOINT="/home/lhx/workspace/test_model/output/qwen3.5_v2_unfreeze_vit_nav_target/v0-20260709-081633/checkpoint-1200"
HFOV_DEG="79.0"
CAMERA_HEIGHT="0.88"
DEPTH_MAX="5.0"
# One worker is ``policy_gpu:habitat_gpus``.  A worker owns its GPUs for its
# complete queue of batches, while different workers run independently.
# The single-GPU Habitat worker is supported by the bridge and lets the strict
# evaluation use every otherwise-idle A800.
WORKER_SPECS="${WORKER_SPECS:-3:0,4|1:2,5|6:7}"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse existing run root: ${RUN_ROOT}" >&2
  echo "Strict evaluation must use a new empty directory so no old batch can be mixed in." >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

"${PYTHON}" - "${RUN_ROOT}/evaluation_fingerprint.json" \
  "${CHECKPOINT}" "${HFOV_DEG}" "${CAMERA_HEIGHT}" "${DEPTH_MAX}" \
  "${MAX_EPISODE_STEPS}" "${MAX_ROLLOUT_STEPS}" "${CONFIG_FILE}" \
  "${RUN_SCRIPT}" "$0" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out, checkpoint, hfov, camera_height, depth_max, max_steps, max_rollout, *paths = sys.argv[1:]
payload = {
    "schema_version": 1,
    "checkpoint": checkpoint,
    "hfov_deg": float(hfov),
    "camera_height_m": float(camera_height),
    "depth_max_m": float(depth_max),
    "max_episode_steps": int(max_steps),
    "max_rollout_steps": int(max_rollout),
    "files": {
        str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in paths
    },
}
payload["fingerprint"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
Path(out).write_text(json.dumps(payload, indent=2) + "\n")
PY
FINGERPRINT_FILE="${RUN_ROOT}/evaluation_fingerprint.json"

mapfile -t BATCHES < <(
  "${PYTHON}" - "${EPISODES_FILE}" <<'PY'
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with open(path) as handle:
    payload = json.load(handle)
episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
scene_to_ids = defaultdict(list)
for item in episodes:
    scene_to_ids[str(item["scene_id"])].append(str(item["episode_id"]))
ids = [episode_id for scene_ids in scene_to_ids.values() for episode_id in scene_ids]
if len(ids) != 100 or len(set(ids)) != 100:
    raise SystemExit(f"expected 100 unique episodes, got {len(ids)} / {len(set(ids))}")
batch_size = int(__import__("os").environ.get("SCENE_BATCH_SIZE", "4"))
for scene_id, scene_ids in scene_to_ids.items():
    for start in range(0, len(scene_ids), batch_size):
        print(json.dumps(scene_ids[start:start + batch_size], separators=(",", ":")))
PY
)

printf '%s\n' "${BATCHES[@]}" > "${RUN_ROOT}/episode_batches.txt"
EPISODE_COUNT=0
for batch in "${BATCHES[@]}"; do
  batch_count="$(${PYTHON} -c 'import json,sys; print(len(json.loads(sys.argv[1])))' "$batch")"
  EPISODE_COUNT=$((EPISODE_COUNT + batch_count))
done
echo "OpenNav100 episodes: ${EPISODE_COUNT}"
echo "Batches: ${#BATCHES[@]}"
echo "Policy GPU: ${RLINF_GPU}; Habitat GPUs: ${HABITAT_GPUS}"
echo "Max episode steps: ${MAX_EPISODE_STEPS}"
echo "Max rollout steps: ${MAX_ROLLOUT_STEPS}"
echo "Scene batch size: ${SCENE_BATCH_SIZE} (same-scene batches)"
echo "Parallel workers: ${WORKER_SPECS}"
echo "Run root: ${RUN_ROOT}"
echo "Fingerprint: $("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "${FINGERPRINT_FILE}")"

batch_is_complete() {
  local metrics_file="$1"
  local expected_batch="$2"
  [[ -s "${metrics_file}" ]] || return 1
  "${PYTHON}" - "${metrics_file}" "${expected_batch}" <<'PY'
import json
import sys

metrics_path, expected_json = sys.argv[1:]
expected_ids = {str(x) for x in json.loads(expected_json)}
with open(metrics_path) as handle:
    payload = json.load(handle)
rows = payload.get("episodes", payload.get("trials", [])) if isinstance(payload, dict) else payload
if not isinstance(rows, list):
    raise SystemExit(1)
actual_ids = {str(row.get("episode_id")) for row in rows if isinstance(row, dict)}
if actual_ids != expected_ids or len(rows) < len(expected_ids):
    raise SystemExit(1)
PY
}

batch_contract_matches() {
  local expected_file="$1"
  local runtime_file="$2"
  [[ -s "${runtime_file}" ]] || return 1
  "${PYTHON}" - "${expected_file}" "${runtime_file}" <<'PY'
import json
import sys

expected, runtime = (json.load(open(path)) for path in sys.argv[1:])
config_hash = expected["files"].get(
    "/home/clk/workspace/RLinf/examples/embodiment/config/habitat_eval_qwen_frozen.yaml"
)
launcher_hash = expected["files"].get(
    "/home/clk/workspace/RLinf/scripts/run_habitat_qwen_ab.sh"
)
for key in ("checkpoint", "hfov_deg", "camera_height_m", "depth_max_m", "max_episode_steps", "max_rollout_steps"):
    if runtime.get(key) != expected.get(key):
        raise SystemExit(f"runtime contract mismatch for {key}: {runtime.get(key)!r} != {expected.get(key)!r}")
if runtime.get("config_sha256") != config_hash:
    raise SystemExit("runtime contract mismatch for Habitat config hash")
if runtime.get("launcher_sha256") != launcher_hash:
    raise SystemExit("runtime contract mismatch for Habitat launcher hash")
PY
}

IFS='|' read -r -a WORKERS <<< "${WORKER_SPECS}"
if [[ "${#WORKERS[@]}" -eq 0 ]]; then
  echo "No parallel worker specifications were provided" >&2
  exit 2
fi

run_batch() {
  local index="$1"
  local worker_id="$2"
  local worker_policy_gpu="$3"
  local worker_habitat_gpus="$4"
  batch="${BATCHES[$index]}"
  batch_number=$((index + 1))
  batch_file="${RUN_ROOT}/batch_${batch_number}_all_episode_metrics.json"

  echo ""
  echo "===== Habitat batch ${batch_number}/${#BATCHES[@]} on worker ${worker_id} (${worker_policy_gpu} <- ${worker_habitat_gpus}): ${batch} ====="
  batch_log_dir="${RUN_ROOT}/batch_${batch_number}"
  batch_port=$((HABITAT_PORT_BASE + index))
  # Ray places Unix-domain sockets below RAY_TMPDIR. Keep this path short
  # (AF_UNIX caps it at 107 bytes) while retaining worker-level isolation.
  EPISODE_IDS_OVERRIDE="${batch}" \
  RUN_TAG="habitat-qwen-open100-batch-${batch_number}" \
  LOG_DIR="${batch_log_dir}" \
  PORT="${batch_port}" \
  CONTAINER="rlinf-qwennav-habitat-eval-open100-${batch_number}-$$" \
  RLINF_GPU="${worker_policy_gpu}" \
  HABITAT_GPUS="${worker_habitat_gpus}" \
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
  MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS}" \
  RAY_TMPDIR="/tmp/rlh_${$}_w${worker_id}" \
  TMPDIR="/tmp/rlh_${$}_tmp_w${worker_id}" \
  bash "${RUN_SCRIPT}"

  log_dir="${batch_log_dir}"
  candidate_file="${log_dir}/all_episode_metrics.json"
  if ! batch_is_complete "${candidate_file}" "${batch}"; then
    echo "Batch ${batch_number} is incomplete: expected episode ids ${batch}, file=${candidate_file}" >&2
    exit 1
  fi
  if ! batch_contract_matches "${FINGERPRINT_FILE}" "${batch_log_dir}/evaluation_runtime_contract.json"; then
    echo "Batch ${batch_number} runtime configuration does not match strict contract" >&2
    exit 1
  fi
  cp "${candidate_file}" "${batch_file}"
  printf '%s\t%s\tworker=%s\n' "${batch}" "${log_dir}" "${worker_id}" \
    > "${batch_log_dir}/completed_batch.tsv"
}

run_worker() {
  local worker_id="$1"
  local worker_spec="$2"
  local worker_policy_gpu worker_habitat_gpus
  IFS=':' read -r worker_policy_gpu worker_habitat_gpus <<< "${worker_spec}"
  if [[ -z "${worker_policy_gpu}" || -z "${worker_habitat_gpus}" ]]; then
    echo "Invalid worker specification: ${worker_spec}" >&2
    return 2
  fi
  for ((index=worker_id; index<${#BATCHES[@]}; index+=${#WORKERS[@]})); do
    run_batch "${index}" "${worker_id}" "${worker_policy_gpu}" "${worker_habitat_gpus}"
  done
}

worker_pids=()
for worker_id in "${!WORKERS[@]}"; do
  run_worker "${worker_id}" "${WORKERS[$worker_id]}" &
  worker_pids+=("$!")
done

worker_failed=0
for worker_id in "${!worker_pids[@]}"; do
  if ! wait "${worker_pids[$worker_id]}"; then
    echo "Habitat worker ${worker_id} failed" >&2
    worker_failed=1
  fi
done
if [[ "${worker_failed}" -ne 0 ]]; then
  echo "Refusing to aggregate after a parallel worker failure" >&2
  exit 1
fi

mapfile -t BATCH_FILES < <(
  for index in "${!BATCHES[@]}"; do
    batch_file="${RUN_ROOT}/batch_$((index + 1))_all_episode_metrics.json"
    batch_fingerprint="${RUN_ROOT}/batch_$((index + 1))/evaluation_runtime_contract.json"
    if batch_is_complete "${batch_file}" "${BATCHES[$index]}" \
      && batch_contract_matches "${FINGERPRINT_FILE}" "${batch_fingerprint}"; then
      printf '%s\n' "${batch_file}"
    fi
  done
)
if [[ "${#BATCH_FILES[@]}" -ne "${#BATCHES[@]}" ]]; then
  echo "Refusing to aggregate: only ${#BATCH_FILES[@]}/${#BATCHES[@]} complete batches" >&2
  exit 1
fi
"${PYTHON}" "${REPO}/scripts/aggregate_aligned_nav_metrics.py" \
  --input-files "${BATCH_FILES[@]}" \
  --output-dir "${RUN_ROOT}" \
  --expected-episodes 100 \
  --source habitat_rlinf_qwen_pure_eval \
  --max-episode-steps "${MAX_EPISODE_STEPS}"

if [[ -f "${GENESIS_TRIALS_FILE}" ]]; then
  "${PYTHON}" "${REPO}/scripts/compare_habitat_genesis_metrics.py" \
    --genesis "${GENESIS_TRIALS_FILE}" \
    --habitat "${RUN_ROOT}/all_episode_metrics.json" \
    --output "${RUN_ROOT}/genesis_habitat_comparison.json"
else
  echo "Genesis reference not found; skipped paired comparison: ${GENESIS_TRIALS_FILE}" >&2
fi

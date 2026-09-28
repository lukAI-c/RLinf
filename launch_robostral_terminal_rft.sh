#!/usr/bin/env bash
set -euo pipefail

REPO=/home/clk/workspace/RLinf
CONFIG_DIR="${REPO}/examples/embodiment/config"
ENTRY="${REPO}/examples/embodiment/train_embodied_agent.py"
PYTHON=/home/clk/miniconda3/envs/genesis-vllm/bin/python
BASE_MODEL=/home/lhx/workspace/test_model/output/qwen3.5_v2_unfreeze_vit_nav_target/v0-20260709-081633/checkpoint-1200

# Previous cache paths, retained for a quick rollback:
# env.train.init_params.glb_cache_dir=/home/clk/workspace/genark/glb_cache
# env.eval.init_params.glb_cache_dir=/home/clk/workspace/genark/glb_cache
# Previous Z6MFQCViBuw optimized cache:
# SCENE_GLB_CACHE=/home/clk/workspace/genark/glb_cache_genesis_binary_v2
# QUCTc6BB5sX has 96 textures and renders natively without atlas conversion.
SCENE_GLB_CACHE=/home/clk/workspace/genark/mp3d_original_glb_cache
GSAM_DIR="${REPO}/assets/grounded_sam"
# The measured pre-split baseline loads GroundedSAM in the rollout policy
# process. Set this to true to retain the remote, separately placed backend.
GSAM_REMOTE_ENABLED="${GSAM_REMOTE_ENABLED:-false}"
GSAM_VISUALIZE="${GSAM_VISUALIZE:-false}"
MAP_SAVE_EVERY="${MAP_SAVE_EVERY:-50}"
MAP_VISUALIZE="${MAP_VISUALIZE:-false}"
# Optional GT path used only for map overlays. Keep it explicit because an
# episode curriculum must not silently draw episode 586 over another episode.
MAP_EPISODE_ID="${MAP_EPISODE_ID:-null}"
GSAM_SERVICE_GPU="${GSAM_SERVICE_GPU:-7}"
# If the optional remote backend is enabled, keep one model process per GPU.
# Four services on one physical GPU duplicate weights and oversubscribe CPU;
# callers with multiple GSAM GPUs can explicitly raise SERVICE_COUNT.
GSAM_SERVICE_COUNT="${GSAM_SERVICE_COUNT:-1}"
GSAM_SLOTS_PER_SERVICE="${GSAM_SLOTS_PER_SERVICE:-8}"
GSAM_PORT_BASE="${GSAM_PORT_BASE:-29640}"
LAVIRA_MAP_DEVICE="${LAVIRA_MAP_DEVICE:-cuda}"
RLINF_VLLM_WSYNC_ROOT="${RLINF_VLLM_WSYNC_ROOT:-}"
ENV_PLACEMENT="${ENV_PLACEMENT:-3:0}"
# Two independent rollout ranks split the eight environments 4+4. Each rank
# keeps the original single-image GroundedSAM and LaViRA runtime pipeline.
ROLLOUT_PLACEMENT="${ROLLOUT_PLACEMENT:-6-7}"
ACTOR_PLACEMENT="${ACTOR_PLACEMENT:-5}"
ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-32}"
ACTOR_GRADIENT_CHECKPOINTING="${ACTOR_GRADIENT_CHECKPOINTING:-false}"
RFT_GROUP_SIZE="${RFT_GROUP_SIZE:-8}"
RFT_TOTAL_ENVS="${RFT_TOTAL_ENVS:-${RFT_GROUP_SIZE}}"
RFT_MAX_STEPS="${RFT_MAX_STEPS:-6}"
RFT_SCENE_COUNT="${RFT_SCENE_COUNT:-1}"
RFT_SCENES_PER_GPU="${RFT_SCENES_PER_GPU:-1}"
RFT_ENV_GPU_BUDGET="${RFT_ENV_GPU_BUDGET:-1}"
RFT_SCENES="${RFT_SCENES:-${EPISODE_CURRICULUM_SCENE:-mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb}}"
RFT_EPISODES_FILE="${RFT_EPISODES_FILE:-/home/clk/workspace/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json}"
RFT_ASSIGNMENT_ENABLED="${RFT_ASSIGNMENT_ENABLED:-true}"
RFT_EPISODE_BLOCKLIST="${RFT_EPISODE_BLOCKLIST:-[586,207,432,550,559,705]}"
RFT_BALANCED_SAMPLING="${RFT_BALANCED_SAMPLING:-false}"
RFT_SEED="${RFT_SEED:-42}"
RFT_RESUME_DIR="${RFT_RESUME_DIR:-}"
RFT_INITIAL_GLOBAL_STEP="${RFT_INITIAL_GLOBAL_STEP:-0}"
RFT_ROLLOUT_TEMPERATURE="${RFT_ROLLOUT_TEMPERATURE:-1.0}"
RFT_REFERENCE_PATH_COEF="${RFT_REFERENCE_PATH_COEF:-0.0}"
RFT_AUX_ADVANTAGE_COEF="${RFT_AUX_ADVANTAGE_COEF:-0.0}"
TERMINATION_SHADOW_ENABLED="${TERMINATION_SHADOW_ENABLED:-false}"
TERMINATION_SHADOW_COLLECT_ONLY="${TERMINATION_SHADOW_COLLECT_ONLY:-false}"
RFT_MAX_DECISIONS="${RFT_MAX_DECISIONS:-20}"
RFT_MAX_ROLLOUT_STEPS="${RFT_MAX_ROLLOUT_STEPS:-200}"
RFT_MAX_NUM_SEQS="${RFT_MAX_NUM_SEQS:-8}"
RFT_DISTANCE_FLOOR_M="${RFT_DISTANCE_FLOOR_M:-2.0}"
RFT_CLEAN_STOP_BONUS="${RFT_CLEAN_STOP_BONUS:-0.0}"
HARD_POOL_ENABLED="${HARD_POOL_ENABLED:-false}"
HARD_POOL_MANIFEST="${HARD_POOL_MANIFEST:-}"
EPISODE_CURRICULUM_ENABLED="${EPISODE_CURRICULUM_ENABLED:-false}"
LOG_ROOT="${LOG_ROOT:-${REPO}/logs}"
EPISODE_ASSIGNMENT_FILE="${EPISODE_ASSIGNMENT_FILE:-${REPO}/examples/embodiment/config/robostral_endpoint_train_no586.json}"
EPISODE_CURRICULUM_SCENE="${EPISODE_CURRICULUM_SCENE:-mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb}"
LOG_SUFFIX="${LOG_SUFFIX:-}"
LOG_DIR="${LOG_ROOT}/$(date +'%Y%m%d-%H%M%S')-sft1200-robostral-terminal-rft${LOG_SUFFIX}"

mkdir -p "${LOG_DIR}"
echo "${LOG_DIR}" > "${LOG_ROOT}/latest_robostral_terminal_rft.txt"
echo "$$" > "${LOG_DIR}/launcher.pid"

cd "${REPO}"

export EMBODIED_PATH="${REPO}/examples/embodiment"
export REPO_PATH="${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export GENESIS_HEADLESS=1
# Allow the one-time Genesis kernel compilation to finish without changing any
# simulator or navigation behavior.
export GENESIS_STARTUP_TIMEOUT_S="${GENESIS_STARTUP_TIMEOUT_S:-600}"
# IMPORTANT: RLinf FlexiblePlacementStrategy uses GLOBAL hardware ranks here.
# These placement values are physical GPU IDs, not process-local CUDA IDs:
#   placement ${ROLLOUT_PLACEMENT} -> two data-parallel rollout ranks
#   placement ${ACTOR_PLACEMENT}   -> single-rank FSDP actor
#   placement ${ENV_PLACEMENT} -> Genesis env, process 0
# Each isolated worker later sees its assigned device as local CUDA GPU 0.
# Keep all physical IDs visible to the cluster so global rank resolution works.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Endpoint-only default: STOP value tokens off the PPO loss.
# STOP stage overrides this to 1 so the stop action itself gets credit.
# The model can still emit stop=true/false either way; an explicit STOP
# still ends the episode.
export QWEN_NAV_LOSS_MASK_INCLUDE_STOP="${QWEN_NAV_LOSS_MASK_INCLUDE_STOP:-0}"
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT_MS=3600000
export RLINF_RAY_LOCAL="${RLINF_RAY_LOCAL:-1}"
# vLLM weight-sync dumps a full safetensors copy per rollout rank. Default
# /tmp and the workspace tmpfs/partition have already hit ENOSPC. Prefer shm.
export RLINF_VLLM_WSYNC_ROOT="${RLINF_VLLM_WSYNC_ROOT:-/dev/shm/rlinf_vllm_wsync}"
mkdir -p "${RLINF_VLLM_WSYNC_ROOT}"
# Suppress only the high-volume, already-audited warnings emitted once per
# source-map/GroundedSAM frame. Other warnings and all exceptions remain
# visible in train.log.
RLINF_WARNING_FILTERS="ignore::UserWarning:vlnce_baselines.utils.map_utils,ignore::FutureWarning:rlinf.third_party.lavira_rft.source_core,ignore::FutureWarning:groundingdino.models.GroundingDINO.transformer"
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${RLINF_WARNING_FILTERS}"
# Defaults retain the shared project locations.  Callers can override these
# for an isolated smoke run without creating a second Ray session under an
# active training run's temporary directory.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_robostral_terminal_rft}"
export TMPDIR="${TMPDIR:-/home/clk/workspace/tmp}"

mkdir -p "$RAY_TMPDIR" "$TMPDIR"

exec > "${LOG_DIR}/train.log" 2>&1

echo "============================================================"
echo "SFT-1200 Robostral Terminal-Score RFT (Phase 1)"
echo "LOG_DIR=${LOG_DIR}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "ADV_TYPE=decision_terminal_grpo"
echo "RFT_SEED=${RFT_SEED}"
echo "EPISODE_ASSIGNMENT_FILE=${EPISODE_ASSIGNMENT_FILE}"
echo "DISTANCE_FLOOR_M=${RFT_DISTANCE_FLOOR_M}"
echo "CLEAN_STOP_BONUS=${RFT_CLEAN_STOP_BONUS} (eval metric only; not in advantage)"
echo "STOP_TOKEN_PPO_LOSS=off"
echo "TERMINATE_ON_MISSED_STOP=false"
echo "EPISODE_ASSIGNMENT_FILE=${EPISODE_ASSIGNMENT_FILE}"
echo "EPISODE_CURRICULUM_ENABLED=${EPISODE_CURRICULUM_ENABLED}"
echo "HARD_POOL_ENABLED=${HARD_POOL_ENABLED}"
echo "EPISODE_CURRICULUM_SCENE=${EPISODE_CURRICULUM_SCENE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "GSAM_DIR=${GSAM_DIR}"
echo "GSAM_REMOTE_ENABLED=${GSAM_REMOTE_ENABLED}"
if [[ "${GSAM_REMOTE_ENABLED}" == "true" ]]; then
  echo "GSAM_SERVICE_GPU=${GSAM_SERVICE_GPU}"
  echo "GSAM_SERVICE_COUNT=${GSAM_SERVICE_COUNT}"
  echo "GSAM_SLOTS_PER_SERVICE=${GSAM_SLOTS_PER_SERVICE}"
else
  echo "GSAM_SERVICE_GPU=<inactive; GroundedSAM uses rollout placement>"
fi
echo "LAVIRA_MAP_DEVICE=${LAVIRA_MAP_DEVICE}"
echo "RLINF_VLLM_WSYNC_ROOT=${RLINF_VLLM_WSYNC_ROOT:-<framework-default>}"
echo "ENV_PLACEMENT=${ENV_PLACEMENT}"
echo "ROLLOUT_PLACEMENT=${ROLLOUT_PLACEMENT}"
echo "ACTOR_PLACEMENT=${ACTOR_PLACEMENT}"
echo "ACTOR_GLOBAL_BATCH_SIZE=${ACTOR_GLOBAL_BATCH_SIZE}"
echo "ACTOR_GRADIENT_CHECKPOINTING=${ACTOR_GRADIENT_CHECKPOINTING}"
echo "RFT_GROUP_SIZE=${RFT_GROUP_SIZE}"
echo "RFT_MAX_STEPS=${RFT_MAX_STEPS}"
echo "RFT_RESUME_DIR=${RFT_RESUME_DIR:-<none>}"
echo "RFT_INITIAL_GLOBAL_STEP=${RFT_INITIAL_GLOBAL_STEP}"
echo "RFT_ROLLOUT_TEMPERATURE=${RFT_ROLLOUT_TEMPERATURE}"
echo "RFT_REFERENCE_PATH_COEF=${RFT_REFERENCE_PATH_COEF}"
echo "RFT_AUX_ADVANTAGE_COEF=${RFT_AUX_ADVANTAGE_COEF}"
# Placement values below are physical/global GPU IDs (see mapping above).
# Keep these comments outside the backslash-continued Hydra command below.
echo "============================================================"

if [[ "${EPISODE_CURRICULUM_ENABLED}" == "true" && "${HARD_POOL_ENABLED}" == "true" ]]; then
  echo "hard_pool and episode_curriculum cannot both be enabled" >&2
  exit 1
fi
if [[ "${EPISODE_CURRICULUM_ENABLED}" == "true" && -n "${EPISODE_ASSIGNMENT_FILE}" ]]; then
  echo "episode_assignment and episode_curriculum cannot both be enabled" >&2
  exit 1
fi
if [[ "${HARD_POOL_ENABLED}" == "true" && -n "${EPISODE_ASSIGNMENT_FILE}" ]]; then
  echo "episode_assignment and hard_pool cannot both be enabled" >&2
  exit 1
fi
if (( RFT_GROUP_SIZE < 2 )); then
  echo "group_size must be >= 2 for same-episode terminal GRPO" >&2
  exit 1
fi
if [[ "${RFT_REFERENCE_PATH_COEF}" != "0" && "${RFT_REFERENCE_PATH_COEF}" != "0.0" ]]; then
  echo "decision_terminal_grpo forbids a nonzero reference-path coefficient" >&2
  exit 1
fi
if [[ "${RFT_AUX_ADVANTAGE_COEF}" != "0" && "${RFT_AUX_ADVANTAGE_COEF}" != "0.0" ]]; then
  echo "decision_terminal_grpo forbids a nonzero rloo_aux_coef" >&2
  exit 1
fi
if [[ "${RFT_ASSIGNMENT_ENABLED}" == "true" && ! -f "${EPISODE_ASSIGNMENT_FILE}" ]]; then
  echo "episode assignment file is missing: ${EPISODE_ASSIGNMENT_FILE}" >&2
  exit 1
fi
if [[ "${GSAM_REMOTE_ENABLED}" != "true" && "${GSAM_REMOTE_ENABLED}" != "false" ]]; then
  echo "GSAM_REMOTE_ENABLED must be true or false" >&2
  exit 1
fi
if [[ "${GSAM_REMOTE_ENABLED}" == "true" ]] &&
   (( GSAM_SERVICE_COUNT * GSAM_SLOTS_PER_SERVICE < 8 )); then
  echo "GroundedSAM services cover fewer than 8 training env slots" >&2
  exit 1
fi

GSAM_PIDS=()
GSAM_ENDPOINT_VALUES=()
cleanup_grounded_sam_services() {
  local pid
  for pid in "${GSAM_PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  for pid in "${GSAM_PIDS[@]:-}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup_grounded_sam_services EXIT INT TERM HUP

GSAM_ENDPOINTS="[]"
if [[ "${GSAM_REMOTE_ENABLED}" == "true" ]]; then
  for ((service_index=0; service_index<GSAM_SERVICE_COUNT; service_index++)); do
    port=$((GSAM_PORT_BASE + service_index))
    service_log="${LOG_DIR}/grounded-sam-service-${service_index}.log"
    GSAM_ENDPOINT_VALUES+=("\"127.0.0.1:${port}\"")
    CUDA_VISIBLE_DEVICES="${GSAM_SERVICE_GPU}" "${PYTHON}" \
      "${REPO}/scripts/grounded_sam_server.py" \
      --port "${port}" \
      --dino-config-path "${GSAM_DIR}/GroundingDINO_SwinT_OGC.py" \
      --dino-checkpoint-path "${GSAM_DIR}/groundingdino_swint_ogc.pth" \
      --repvit-sam-checkpoint-path /home/nvme01/uni-lavira/data/grounded_sam/repvit_sam.pt \
      --box-threshold 0.25 \
      --text-threshold 0.25 \
      --waypoint-scene-box-area-ratio 0.65 \
      --waypoint-scene-edge-margin-ratio 0.02 \
      --reject-scene-region-source-waypoint \
      >"${service_log}" 2>&1 &
    GSAM_PIDS+=("$!")
  done

  for ((service_index=0; service_index<GSAM_SERVICE_COUNT; service_index++)); do
    service_log="${LOG_DIR}/grounded-sam-service-${service_index}.log"
    service_pid="${GSAM_PIDS[service_index]}"
    ready=false
    for _ in $(seq 1 120); do
      if rg -q "\[grounded-sam-server\] ready" "${service_log}"; then
        ready=true
        break
      fi
      if ! kill -0 "${service_pid}" >/dev/null 2>&1; then
        echo "GroundedSAM service ${service_index} exited before readiness" >&2
        cat "${service_log}" >&2 || true
        exit 1
      fi
      sleep 1
    done
    if [[ "${ready}" != true ]]; then
      echo "GroundedSAM service ${service_index} readiness timed out" >&2
      exit 1
    fi
  done
  GSAM_ENDPOINTS="[$(IFS=,; echo "${GSAM_ENDPOINT_VALUES[*]}")]"
  echo "GroundedSAM services ready: ${GSAM_ENDPOINTS}"
else
  echo "GroundedSAM backend: local rollout process on ${ROLLOUT_PLACEMENT}"
fi

# Decision-RLOO uses one fixed on-policy group of eight trajectories.
# Candidate selection and Reinforce-Ada retries remain disabled so all samples
# enter the same group estimator. Periodic eval is disabled to avoid a second
# Genesis scene on the environment GPU.
"${PYTHON}" "${ENTRY}" \
  --config-path "${CONFIG_DIR}" \
  --config-name genark_grpo_qwen_multiscene \
  runner.logger.log_path="${LOG_DIR}" \
  runner.resume_dir="${RFT_RESUME_DIR:-null}" \
  actor.model.model_path="${BASE_MODEL}" \
  rollout.model.model_path="${BASE_MODEL}" \
  runner.max_epochs="${RFT_MAX_STEPS}" \
  runner.max_steps="${RFT_MAX_STEPS}" \
  runner.val_check_interval=-1 \
  runner.save_interval=2 \
  cluster.component_placement.rollout.placement="${ROLLOUT_PLACEMENT}" \
  cluster.component_placement.actor.placement="${ACTOR_PLACEMENT}" \
  cluster.component_placement.env.placement="${ENV_PLACEMENT}" \
  env.train.total_num_envs="${RFT_TOTAL_ENVS}" \
  env.train.episodes_file="${RFT_EPISODES_FILE}" \
  env.train.init_params.episodes_file="${RFT_EPISODES_FILE}" \
  env.train.genesis_backend=multiscene \
  env.train.multi_scene.scene_count="${RFT_SCENE_COUNT}" \
  env.train.multi_scene.scenes_per_gpu="${RFT_SCENES_PER_GPU}" \
  env.train.multi_scene.gpu_budget="${RFT_ENV_GPU_BUDGET}" \
  "env.train.multi_scene.scenes=[${RFT_SCENES}]" \
  env.train.init_params.glb_cache_dir="${SCENE_GLB_CACHE}" \
  ++env.train.episode_overfit.enabled=false \
  env.train.episode_balanced_sampling="${RFT_BALANCED_SAMPLING}" \
  env.train.cyclic_episode_sampling=true \
  env.train.curriculum_dtg_start=null \
  ++env.train.episode_curriculum.enabled=false \
  ++env.train.episode_assignment.enabled="${RFT_ASSIGNMENT_ENABLED}" \
  ++env.train.episode_assignment.file="${EPISODE_ASSIGNMENT_FILE}" \
  ++env.train.episode_assignment.initial_global_step="${RFT_INITIAL_GLOBAL_STEP}" \
  ++env.train.episode_blocklist="${RFT_EPISODE_BLOCKLIST}" \
  ++env.train.hard_pool.enabled="${HARD_POOL_ENABLED}" \
  ++env.train.terminal_navigation_score.enabled=true \
  ++env.train.terminal_navigation_score.distance_floor_m="${RFT_DISTANCE_FLOOR_M}" \
  ++env.train.terminal_navigation_score.clean_stop_bonus="${RFT_CLEAN_STOP_BONUS}" \
  env.train.group_size="${RFT_GROUP_SIZE}" \
  env.train.max_episode_steps=300 \
  env.train.max_steps_per_rollout_epoch="${RFT_MAX_ROLLOUT_STEPS}" \
  env.train.max_decisions_per_rollout_epoch="${RFT_MAX_DECISIONS}" \
  env.train.camera_height=0.88 \
  env.train.fov=79 \
  env.train.cam_res='[640,480]' \
  env.train.enable_depth_obs=true \
  env.train.enable_4dir_render=true \
  env.train.enable_4dir_depth_obs=true \
  env.train.auto_reset=false \
  env.train.ignore_terminations=false \
  ++env.train.reward_profile=nav \
  env.train.reward_mode=decision_nav \
  env.train.geo_coef=0.0 \
  env.train.geo_step_clip=0.5 \
  env.train.nav_terminal_reward_enabled=false \
  env.train.nav_path_coef=0.0 \
  env.train.nav_endpoint_coef=0.0 \
  env.train.nav_endpoint_decay=0.2 \
  env.train.gsam_reward_enabled=false \
  env.train.ndtw_coef=0.0 \
  env.train.format_reward_coef=0.0 \
  env.train.json_reward_coef=0.0 \
  env.train.struct_reward_coef=0.0 \
  env.train.field_format_reward_coef=0.0 \
  env.train.length_reward_coef=0.0 \
  env.train.bbox_reward_coef=0.0 \
  env.train.sr_coef=0.0 \
  env.train.wrong_stop_penalty=0.0 \
  env.train.conditional_wrong_stop_penalty=false \
  env.train.conditional_wrong_stop_ratio_min=0.5 \
  env.train.conditional_wrong_stop_ratio_max=2.0 \
  env.train.parse_fail_penalty=0.0 \
  env.train.no_stop_penalty=0.0 \
  env.train.process_reward_enabled=false \
  env.train.process_progress_coef=0.0 \
  env.train.process_progress_cap=6.0 \
  ++env.train.reference_path_reward_enabled=false \
  ++env.train.reference_path_progress_coef=0.0 \
  ++env.train.reference_path_lateral_penalty=1.0 \
  ++env.train.reference_path_delta_clip=0.25 \
  ++env.train.reference_path_normalized_potential=true \
  ++env.train.missed_stop_aux_reward_enabled=false \
  ++env.train.missed_stop_aux_penalty=0.25 \
  ++env.train.terminate_on_missed_stop=false \
  algorithm.group_size="${RFT_GROUP_SIZE}" \
  algorithm.sampling_params.temperature_train="${RFT_ROLLOUT_TEMPERATURE}" \
  ++algorithm.early_stop_approx_kl=0.03 \
  algorithm.adv_type=decision_terminal_grpo \
  algorithm.normalize_advantages=false \
  ++algorithm.rloo_aux_coef=0.0 \
  algorithm.acr_reward_std_threshold=1e-6 \
  ++algorithm.low_reward_std_threshold=0.05 \
  ++algorithm.candidate_group_selection.enabled=false \
  ++algorithm.reinforce_ada.enabled=false \
  env.eval.total_num_envs=4 \
  env.eval.genesis_backend=multiscene \
  env.eval.multi_scene.scene_count=1 \
  env.eval.multi_scene.scenes_per_gpu=1 \
  env.eval.multi_scene.gpu_budget=1 \
  "env.eval.multi_scene.scenes=[${EPISODE_CURRICULUM_SCENE}]" \
  env.eval.init_params.glb_cache_dir="${SCENE_GLB_CACHE}" \
  ++env.eval.episode_overfit.enabled=true \
  ++env.eval.episode_overfit.scene_id="${EPISODE_CURRICULUM_SCENE}" \
  ++env.eval.episode_overfit.episode_id=586 \
  env.eval.group_size=1 \
  env.eval.max_episode_steps=300 \
  env.eval.max_steps_per_rollout_epoch=200 \
  env.eval.camera_height=0.88 \
  env.eval.fov=79 \
  env.eval.cam_res='[640,480]' \
  env.eval.enable_depth_obs=true \
  env.eval.enable_4dir_render=true \
  env.eval.enable_4dir_depth_obs=true \
  env.eval.auto_reset=false \
  env.eval.ignore_terminations=false \
  actor.global_batch_size="${ACTOR_GLOBAL_BATCH_SIZE}" \
  actor.seed="${RFT_SEED}" \
  actor.micro_batch_size=1 \
  actor.optim.lr=2.5e-7 \
  rollout.max_model_len=8192 \
  rollout.gpu_memory_utilization=0.25 \
  rollout.max_num_seqs="${RFT_MAX_NUM_SEQS:-8}" \
  actor.model.history_max_frames=2 \
  actor.model.min_pixels=4096 \
  actor.model.max_pixels=122500 \
  actor.model.temperature="${RFT_ROLLOUT_TEMPERATURE}" \
  ++rollout.mm_processor_kwargs.min_pixels=4096 \
  ++rollout.mm_processor_kwargs.max_pixels=122500 \
  ++rollout.limit_mm_per_prompt.image=100 \
  ++rollout.model.min_pixels=4096 \
  ++rollout.model.max_pixels=122500 \
  rollout.model.temperature="${RFT_ROLLOUT_TEMPERATURE}" \
  actor.fsdp_config.gradient_checkpointing="${ACTOR_GRADIENT_CHECKPOINTING}" \
  rollout.model.history_max_frames=2 \
  ++actor.model.prompt_style=lavira_waypoint \
  ++actor.model.termination_shadow.enabled="${TERMINATION_SHADOW_ENABLED}" \
  ++actor.model.termination_shadow.collect_only="${TERMINATION_SHADOW_COLLECT_ONLY}" \
  ++actor.model.termination_shadow.output_dir="${LOG_DIR}/termination_shadow" \
  ++actor.model.lavira_runtime.enabled=true \
  ++actor.model.lavira_runtime.map_backend=source \
  ++actor.model.lavira_runtime.map_device="${LAVIRA_MAP_DEVICE}" \
  ++actor.model.lavira_runtime.hfov_deg=79.0 \
  ++actor.model.lavira_runtime.camera_height=0.88 \
  ++actor.model.lavira_runtime.initial_scan_turns=12 \
  ++actor.model.lavira_runtime.target_reached_threshold_m=0.75 \
  ++actor.model.lavira_runtime.max_steps_to_target=15 \
  ++actor.model.lavira_runtime.layered_history=true \
  ++actor.model.lavira_runtime.history_wp_max=8 \
  ++actor.model.lavira_runtime.layered_backtrack_radius_m=6.0 \
  ++actor.model.lavira_runtime.backtrack_second_chance=true \
  ++actor.model.lavira_runtime.map_visualization.enabled="${MAP_VISUALIZE}" \
  ++actor.model.lavira_runtime.map_visualization.output_dir="${LOG_DIR}/lavira_maps" \
  ++actor.model.lavira_runtime.map_visualization.episode_json=/home/clk/workspace/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json \
  ++actor.model.lavira_runtime.map_visualization.episode_id="${MAP_EPISODE_ID}" \
  ++actor.model.lavira_runtime.map_visualization.save_raw_every=0 \
  ++actor.model.lavira_runtime.map_visualization.save_every="${MAP_SAVE_EVERY}" \
  ++actor.model.grounded_sam.enabled=true \
  ++actor.model.grounded_sam.canonicalize_lavira_waypoint_query=true \
  ++actor.model.grounded_sam.visualize="${GSAM_VISUALIZE}" \
  ++actor.model.grounded_sam.visualization_dir="${LOG_DIR}/grounded_sam" \
  ++actor.model.grounded_sam.dino_config_path="${GSAM_DIR}/GroundingDINO_SwinT_OGC.py" \
  ++actor.model.grounded_sam.dino_checkpoint_path="${GSAM_DIR}/groundingdino_swint_ogc.pth" \
  ++actor.model.grounded_sam.repvit_sam_checkpoint_path=/home/nvme01/uni-lavira/data/grounded_sam/repvit_sam.pt \
  ++actor.model.grounded_sam.device=cuda \
  ++actor.model.grounded_sam.box_threshold=0.25 \
  ++actor.model.grounded_sam.text_threshold=0.25 \
  ++actor.model.grounded_sam.waypoint_scene_box_area_ratio=0.65 \
  ++actor.model.grounded_sam.waypoint_scene_edge_margin_ratio=0.02 \
  ++actor.model.grounded_sam.reject_scene_region_source_waypoint=true \
  ++actor.model.grounded_sam.fail_fast_on_missing_assets=true \
  ++actor.model.grounded_sam.remote.enabled="${GSAM_REMOTE_ENABLED}" \
  ++actor.model.grounded_sam.remote.endpoints="${GSAM_ENDPOINTS}" \
  ++actor.model.grounded_sam.remote.slots_per_service="${GSAM_SLOTS_PER_SERVICE}" \
  ++actor.model.grounded_sam.remote.timeout_s=300.0 \
  ++rollout.model.prompt_style=lavira_waypoint \
  ++rollout.model.termination_shadow.enabled="${TERMINATION_SHADOW_ENABLED}" \
  ++rollout.model.termination_shadow.collect_only="${TERMINATION_SHADOW_COLLECT_ONLY}" \
  ++rollout.model.termination_shadow.output_dir="${LOG_DIR}/termination_shadow" \
  ++rollout.model.lavira_runtime.enabled=true \
  ++rollout.model.lavira_runtime.map_backend=source \
  ++rollout.model.lavira_runtime.map_device="${LAVIRA_MAP_DEVICE}" \
  ++rollout.model.lavira_runtime.hfov_deg=79.0 \
  ++rollout.model.lavira_runtime.camera_height=0.88 \
  ++rollout.model.lavira_runtime.initial_scan_turns=12 \
  ++rollout.model.lavira_runtime.target_reached_threshold_m=0.75 \
  ++rollout.model.lavira_runtime.max_steps_to_target=15 \
  ++rollout.model.lavira_runtime.layered_history=true \
  ++rollout.model.lavira_runtime.history_wp_max=8 \
  ++rollout.model.lavira_runtime.layered_backtrack_radius_m=6.0 \
  ++rollout.model.lavira_runtime.backtrack_second_chance=true \
  ++rollout.model.lavira_runtime.map_visualization.enabled="${MAP_VISUALIZE}" \
  ++rollout.model.lavira_runtime.map_visualization.output_dir="${LOG_DIR}/lavira_maps" \
  ++rollout.model.lavira_runtime.map_visualization.episode_json=/home/clk/workspace/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json \
  ++rollout.model.lavira_runtime.map_visualization.episode_id="${MAP_EPISODE_ID}" \
  ++rollout.model.lavira_runtime.map_visualization.save_raw_every=0 \
  ++rollout.model.lavira_runtime.map_visualization.save_every="${MAP_SAVE_EVERY}" \
  ++rollout.model.grounded_sam.enabled=true \
  ++rollout.model.grounded_sam.canonicalize_lavira_waypoint_query=true \
  ++rollout.model.grounded_sam.visualize="${GSAM_VISUALIZE}" \
  ++rollout.model.grounded_sam.visualization_dir="${LOG_DIR}/grounded_sam" \
  ++rollout.model.grounded_sam.dino_config_path="${GSAM_DIR}/GroundingDINO_SwinT_OGC.py" \
  ++rollout.model.grounded_sam.dino_checkpoint_path="${GSAM_DIR}/groundingdino_swint_ogc.pth" \
  ++rollout.model.grounded_sam.repvit_sam_checkpoint_path=/home/nvme01/uni-lavira/data/grounded_sam/repvit_sam.pt \
  ++rollout.model.grounded_sam.device=cuda \
  ++rollout.model.grounded_sam.box_threshold=0.25 \
  ++rollout.model.grounded_sam.text_threshold=0.25 \
  ++rollout.model.grounded_sam.waypoint_scene_box_area_ratio=0.65 \
  ++rollout.model.grounded_sam.waypoint_scene_edge_margin_ratio=0.02 \
  ++rollout.model.grounded_sam.reject_scene_region_source_waypoint=true \
  ++rollout.model.grounded_sam.fail_fast_on_missing_assets=true \
  ++rollout.model.grounded_sam.remote.enabled="${GSAM_REMOTE_ENABLED}" \
  ++rollout.model.grounded_sam.remote.endpoints="${GSAM_ENDPOINTS}" \
  ++rollout.model.grounded_sam.remote.slots_per_service="${GSAM_SLOTS_PER_SERVICE}" \
  ++rollout.model.grounded_sam.remote.timeout_s=300.0 \
  "$@"

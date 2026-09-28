#!/usr/bin/env bash
set -euo pipefail

REPO=/home/clk/workspace/RLinf
CONFIG_DIR="${REPO}/examples/embodiment/config"
ENTRY="${REPO}/examples/embodiment/train_embodied_agent.py"
PYTHON=/home/clk/miniconda3/envs/genesis-vllm/bin/python
BASE_MODEL=/home/lhx/workspace/test_model/output/qwen3.5_v2_unfreeze_vit_nav_target/v0-20260709-081633/checkpoint-1200
# Start from the frozen SFT base unless a caller explicitly requests an RFT
# checkpoint with RESUME_DIR=/path/to/global_step_N.
RESUME_DIR="${RESUME_DIR:-}"
if [[ -n "${RESUME_DIR}" ]]; then
  RESUME_GLOBAL_STEP="${RESUME_GLOBAL_STEP:-${RESUME_DIR##*global_step_}}"
  RESUME_OVERRIDE="${RESUME_DIR}"
else
  RESUME_GLOBAL_STEP="${RESUME_GLOBAL_STEP:-0}"
  RESUME_OVERRIDE=null
fi

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
RFT_GROUP_SIZE="${RFT_GROUP_SIZE:-8}"
RFT_MAX_STEPS="${RFT_MAX_STEPS:-24}"
RFT_ROLLOUT_TEMPERATURE="${RFT_ROLLOUT_TEMPERATURE:-1.0}"
RFT_REFERENCE_PATH_COEF="${RFT_REFERENCE_PATH_COEF:-1.0}"
RFT_AUX_ADVANTAGE_COEF="${RFT_AUX_ADVANTAGE_COEF:-0.5}"
RFT_GOAL_APPROACH_PATH_INDEX="${RFT_GOAL_APPROACH_PATH_INDEX:-3}"
RFT_HALL_TRANSITION_NEAR_M="${RFT_HALL_TRANSITION_NEAR_M:-4.9}"
RFT_HALL_TRANSITION_FAR_M="${RFT_HALL_TRANSITION_FAR_M:-5.4}"
RFT_HALL_ENTRY_PATH_INDEX="${RFT_HALL_ENTRY_PATH_INDEX:-2}"
RFT_KITCHEN_EXITED_PATH_INDEX="${RFT_KITCHEN_EXITED_PATH_INDEX:-1}"
RFT_GOAL_APPROACH_MIN_CLEAN_STOPS="${RFT_GOAL_APPROACH_MIN_CLEAN_STOPS:-3}"
RFT_HALL_TRANSITION_MIN_CLEAN_STOPS="${RFT_HALL_TRANSITION_MIN_CLEAN_STOPS:-2}"
RFT_HALL_ENTRY_MIN_CLEAN_STOPS="${RFT_HALL_ENTRY_MIN_CLEAN_STOPS:-2}"
RFT_KITCHEN_EXITED_MIN_CLEAN_STOPS="${RFT_KITCHEN_EXITED_MIN_CLEAN_STOPS:-2}"
RFT_GOAL_APPROACH_WINDOW_CLEAN_STOPS="${RFT_GOAL_APPROACH_WINDOW_CLEAN_STOPS:-9}"
RFT_HALL_TRANSITION_WINDOW_CLEAN_STOPS="${RFT_HALL_TRANSITION_WINDOW_CLEAN_STOPS:-6}"
RFT_HALL_ENTRY_WINDOW_CLEAN_STOPS="${RFT_HALL_ENTRY_WINDOW_CLEAN_STOPS:-6}"
RFT_KITCHEN_EXITED_WINDOW_CLEAN_STOPS="${RFT_KITCHEN_EXITED_WINDOW_CLEAN_STOPS:-6}"
RFT_MIN_START_GOAL_MARGIN_M="${RFT_MIN_START_GOAL_MARGIN_M:-0.5}"
RFT_CURRICULUM_CONSECUTIVE_GROUPS="${RFT_CURRICULUM_CONSECUTIVE_GROUPS:-2}"
RFT_CURRICULUM_PROMOTION_WINDOW_GROUPS="${RFT_CURRICULUM_PROMOTION_WINDOW_GROUPS:-3}"
RFT_CURRICULUM_MIN_SUCCESSFUL_GROUPS="${RFT_CURRICULUM_MIN_SUCCESSFUL_GROUPS:-2}"
RFT_CURRICULUM_REPLAY_PREVIOUS_EVERY="${RFT_CURRICULUM_REPLAY_PREVIOUS_EVERY:-3}"
RFT_INITIAL_STAGE="${RFT_INITIAL_STAGE:-hall_transition_near}"
LOG_ROOT="${LOG_ROOT:-${REPO}/logs}"
LOG_DIR="${LOG_ROOT}/$(date +'%Y%m%d-%H%M%S')-episode-overfit-609-quct-clean-rloo-curriculum-rft"

mkdir -p "${LOG_DIR}"
echo "${LOG_DIR}" > "${LOG_ROOT}/latest_episode_609_overfit_rft.txt"
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT_MS=3600000
export RLINF_RAY_LOCAL=1
if [[ -n "${RLINF_VLLM_WSYNC_ROOT}" ]]; then
  export RLINF_VLLM_WSYNC_ROOT
else
  unset RLINF_VLLM_WSYNC_ROOT
fi
# Suppress only the high-volume, already-audited warnings emitted once per
# source-map/GroundedSAM frame. Other warnings and all exceptions remain
# visible in train.log.
RLINF_WARNING_FILTERS="ignore::UserWarning:vlnce_baselines.utils.map_utils,ignore::FutureWarning:rlinf.third_party.lavira_rft.source_core,ignore::FutureWarning:groundingdino.models.GroundingDINO.transformer"
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${RLINF_WARNING_FILTERS}"
# Defaults retain the shared project locations.  Callers can override these
# for an isolated smoke run without creating a second Ray session under an
# active training run's temporary directory.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray609overfitrft}"
export TMPDIR="${TMPDIR:-/home/clk/workspace/tmp}"

mkdir -p "$RAY_TMPDIR" "$TMPDIR"

exec > "${LOG_DIR}/train.log" 2>&1

echo "============================================================"
echo "Episode 609 QUCTc6BB5sX Clean-RLOO Curriculum RFT"
echo "LOG_DIR=${LOG_DIR}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "RESUME_DIR=${RESUME_DIR}"
echo "RESUME_GLOBAL_STEP=${RESUME_GLOBAL_STEP}"
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
echo "RFT_GROUP_SIZE=${RFT_GROUP_SIZE}"
echo "RFT_MAX_STEPS=${RFT_MAX_STEPS}"
echo "RFT_ROLLOUT_TEMPERATURE=${RFT_ROLLOUT_TEMPERATURE}"
echo "RFT_REFERENCE_PATH_COEF=${RFT_REFERENCE_PATH_COEF}"
echo "RFT_AUX_ADVANTAGE_COEF=${RFT_AUX_ADVANTAGE_COEF}"
echo "RFT_GOAL_APPROACH_PATH_INDEX=${RFT_GOAL_APPROACH_PATH_INDEX}"
echo "RFT_HALL_TRANSITION_NEAR_M=${RFT_HALL_TRANSITION_NEAR_M}"
echo "RFT_HALL_TRANSITION_FAR_M=${RFT_HALL_TRANSITION_FAR_M}"
echo "RFT_HALL_ENTRY_PATH_INDEX=${RFT_HALL_ENTRY_PATH_INDEX}"
echo "RFT_KITCHEN_EXITED_PATH_INDEX=${RFT_KITCHEN_EXITED_PATH_INDEX}"
echo "RFT_GOAL_APPROACH_WINDOW_CLEAN_STOPS=${RFT_GOAL_APPROACH_WINDOW_CLEAN_STOPS}"
echo "RFT_HALL_TRANSITION_WINDOW_CLEAN_STOPS=${RFT_HALL_TRANSITION_WINDOW_CLEAN_STOPS}"
echo "RFT_HALL_ENTRY_WINDOW_CLEAN_STOPS=${RFT_HALL_ENTRY_WINDOW_CLEAN_STOPS}"
echo "RFT_KITCHEN_EXITED_WINDOW_CLEAN_STOPS=${RFT_KITCHEN_EXITED_WINDOW_CLEAN_STOPS}"
echo "RFT_MIN_START_GOAL_MARGIN_M=${RFT_MIN_START_GOAL_MARGIN_M}"
echo "RFT_CURRICULUM_CONSECUTIVE_GROUPS=${RFT_CURRICULUM_CONSECUTIVE_GROUPS}"
echo "RFT_CURRICULUM_PROMOTION_WINDOW_GROUPS=${RFT_CURRICULUM_PROMOTION_WINDOW_GROUPS}"
echo "RFT_CURRICULUM_MIN_SUCCESSFUL_GROUPS=${RFT_CURRICULUM_MIN_SUCCESSFUL_GROUPS}"
echo "RFT_CURRICULUM_REPLAY_PREVIOUS_EVERY=${RFT_CURRICULUM_REPLAY_PREVIOUS_EVERY}"
echo "RFT_INITIAL_STAGE=${RFT_INITIAL_STAGE}"
# Placement values below are physical/global GPU IDs (see mapping above).
# Keep these comments outside the backslash-continued Hydra command below.
echo "============================================================"

if [[ "${GSAM_REMOTE_ENABLED}" != "true" && "${GSAM_REMOTE_ENABLED}" != "false" ]]; then
  echo "GSAM_REMOTE_ENABLED must be true or false" >&2
  exit 1
fi
if [[ -n "${RESUME_DIR}" && ! -d "${RESUME_DIR}/actor" ]]; then
  echo "Resume actor checkpoint not found: ${RESUME_DIR}/actor" >&2
  exit 1
fi
if [[ ! "${RESUME_GLOBAL_STEP}" =~ ^[0-9]+$ ]]; then
  echo "RESUME_GLOBAL_STEP must be a non-negative integer" >&2
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
  runner.resume_dir="${RESUME_OVERRIDE}" \
  actor.model.model_path="${BASE_MODEL}" \
  rollout.model.model_path="${BASE_MODEL}" \
  runner.max_epochs="${RFT_MAX_STEPS}" \
  runner.max_steps="${RFT_MAX_STEPS}" \
  runner.val_check_interval=-1 \
  runner.save_interval=2 \
  cluster.component_placement.rollout.placement="${ROLLOUT_PLACEMENT}" \
  cluster.component_placement.actor.placement="${ACTOR_PLACEMENT}" \
  cluster.component_placement.env.placement="${ENV_PLACEMENT}" \
  env.train.total_num_envs=8 \
  env.train.genesis_backend=multiscene \
  env.train.multi_scene.scene_count=1 \
  env.train.multi_scene.scenes_per_gpu=1 \
  env.train.multi_scene.gpu_budget=1 \
  'env.train.multi_scene.scenes=[mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb]' \
  env.train.init_params.glb_cache_dir="${SCENE_GLB_CACHE}" \
  ++env.train.episode_overfit.enabled=true \
  ++env.train.episode_overfit.scene_id=mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb \
  ++env.train.episode_overfit.episode_id=609 \
  env.train.group_size="${RFT_GROUP_SIZE}" \
  env.train.max_episode_steps=300 \
  env.train.max_steps_per_rollout_epoch=200 \
  env.train.max_decisions_per_rollout_epoch=20 \
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
  ++env.train.reference_path_reward_enabled=true \
  ++env.train.reference_path_progress_coef="${RFT_REFERENCE_PATH_COEF}" \
  ++env.train.reference_path_lateral_penalty=1.0 \
  ++env.train.reference_path_delta_clip=0.25 \
  ++env.train.reference_path_normalized_potential=true \
  ++env.train.missed_stop_aux_reward_enabled=true \
  ++env.train.missed_stop_aux_penalty=0.25 \
  algorithm.group_size="${RFT_GROUP_SIZE}" \
  algorithm.sampling_params.temperature_train="${RFT_ROLLOUT_TEMPERATURE}" \
  ++algorithm.early_stop_approx_kl=0.03 \
  ++env.train.rft_start_curriculum.enabled=true \
  env.train.rft_start_curriculum.consecutive_groups="${RFT_CURRICULUM_CONSECUTIVE_GROUPS}" \
  ++env.train.rft_start_curriculum.promotion_window_groups="${RFT_CURRICULUM_PROMOTION_WINDOW_GROUPS}" \
  ++env.train.rft_start_curriculum.promotion_min_successful_groups="${RFT_CURRICULUM_MIN_SUCCESSFUL_GROUPS}" \
  ++env.train.rft_start_curriculum.replay_previous_every="${RFT_CURRICULUM_REPLAY_PREVIOUS_EVERY}" \
  ++env.train.rft_start_curriculum.initial_stage_name="${RFT_INITIAL_STAGE}" \
  ++env.train.rft_start_curriculum.initial_global_step="${RESUME_GLOBAL_STEP}" \
  env.train.rft_start_curriculum.min_start_goal_margin_m="${RFT_MIN_START_GOAL_MARGIN_M}" \
  "env.train.rft_start_curriculum.stages=[{name: goal_approach, reference_path_index: ${RFT_GOAL_APPROACH_PATH_INDEX}, min_clean_stops: ${RFT_GOAL_APPROACH_MIN_CLEAN_STOPS}, min_clean_stops_in_window: ${RFT_GOAL_APPROACH_WINDOW_CLEAN_STOPS}},{name: hall_transition_near, remaining_path_m: ${RFT_HALL_TRANSITION_NEAR_M}, min_clean_stops: ${RFT_HALL_TRANSITION_MIN_CLEAN_STOPS}, min_clean_stops_in_window: ${RFT_HALL_TRANSITION_WINDOW_CLEAN_STOPS}},{name: hall_transition_far, remaining_path_m: ${RFT_HALL_TRANSITION_FAR_M}, min_clean_stops: ${RFT_HALL_TRANSITION_MIN_CLEAN_STOPS}, min_clean_stops_in_window: ${RFT_HALL_TRANSITION_WINDOW_CLEAN_STOPS}},{name: hall_entry, reference_path_index: ${RFT_HALL_ENTRY_PATH_INDEX}, min_clean_stops: ${RFT_HALL_ENTRY_MIN_CLEAN_STOPS}, min_clean_stops_in_window: ${RFT_HALL_ENTRY_WINDOW_CLEAN_STOPS}},{name: kitchen_exited, reference_path_index: ${RFT_KITCHEN_EXITED_PATH_INDEX}, min_clean_stops: ${RFT_KITCHEN_EXITED_MIN_CLEAN_STOPS}, min_clean_stops_in_window: ${RFT_KITCHEN_EXITED_WINDOW_CLEAN_STOPS}},{name: original, original_start: true}]" \
  algorithm.adv_type=decision_rloo_aux_rloo \
  algorithm.normalize_advantages=false \
  ++algorithm.rloo_aux_coef="${RFT_AUX_ADVANTAGE_COEF}" \
  algorithm.acr_reward_std_threshold=1e-6 \
  ++algorithm.low_reward_std_threshold=0.05 \
  ++algorithm.candidate_group_selection.enabled=false \
  ++algorithm.reinforce_ada.enabled=false \
  env.eval.total_num_envs=4 \
  env.eval.genesis_backend=multiscene \
  env.eval.multi_scene.scene_count=1 \
  env.eval.multi_scene.scenes_per_gpu=1 \
  env.eval.multi_scene.gpu_budget=1 \
  'env.eval.multi_scene.scenes=[mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb]' \
  env.eval.init_params.glb_cache_dir="${SCENE_GLB_CACHE}" \
  ++env.eval.episode_overfit.enabled=true \
  ++env.eval.episode_overfit.scene_id=mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb \
  ++env.eval.episode_overfit.episode_id=609 \
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
  actor.global_batch_size=32 \
  actor.micro_batch_size=1 \
  actor.optim.lr=2.5e-7 \
  rollout.max_model_len=8192 \
  rollout.gpu_memory_utilization=0.25 \
  rollout.max_num_seqs=8 \
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
  actor.fsdp_config.gradient_checkpointing=false \
  rollout.model.history_max_frames=2 \
  ++actor.model.prompt_style=lavira_waypoint \
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
  ++actor.model.lavira_runtime.map_visualization.enabled=true \
  ++actor.model.lavira_runtime.map_visualization.output_dir="${LOG_DIR}/lavira_maps" \
  ++actor.model.lavira_runtime.map_visualization.episode_json=/home/clk/workspace/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json \
  ++actor.model.lavira_runtime.map_visualization.episode_id=609 \
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
  ++rollout.model.lavira_runtime.map_visualization.enabled=true \
  ++rollout.model.lavira_runtime.map_visualization.output_dir="${LOG_DIR}/lavira_maps" \
  ++rollout.model.lavira_runtime.map_visualization.episode_json=/home/clk/workspace/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json \
  ++rollout.model.lavira_runtime.map_visualization.episode_id=609 \
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

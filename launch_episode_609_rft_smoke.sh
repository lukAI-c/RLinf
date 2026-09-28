#!/usr/bin/env bash
set -euo pipefail

REPO=/home/clk/workspace/RLinf
CONFIG_DIR="${REPO}/examples/embodiment/config"
ENTRY="${REPO}/examples/embodiment/train_embodied_agent.py"
PYTHON=/home/clk/miniconda3/envs/genesis-vllm/bin/python
# Start fresh from the requested SFT/AWR warm-start model. Do not set
# runner.resume_dir: an RFT checkpoint would overwrite these weights.
BASE_MODEL=/home/lhx/workspace/test_model/output/qwen3.5_v2_unfreeze_vit_nav_target/v0-20260709-081633/checkpoint-1200

# Previous cache paths, retained for a quick rollback:
# env.train.init_params.glb_cache_dir=/home/clk/workspace/genark/glb_cache
# env.eval.init_params.glb_cache_dir=/home/clk/workspace/genark/glb_cache
# Previous Z6MFQCViBuw optimized cache:
# SCENE_GLB_CACHE=/home/clk/workspace/genark/glb_cache_genesis_binary_v2
# QUCTc6BB5sX has 96 textures and renders natively without atlas conversion.
SCENE_GLB_CACHE=/home/clk/workspace/genark/glb_cache
GSAM_DIR="${REPO}/assets/grounded_sam"
LOG_ROOT="${LOG_ROOT:-${REPO}/logs}"
LOG_DIR="${LOG_ROOT}/$(date +'%Y%m%d-%H%M%S')-episode-overfit-609-quct-decision-maxrl-n4-srgb-rft-smoke"

mkdir -p "${LOG_DIR}"
echo "${LOG_DIR}" > "${LOG_ROOT}/latest_episode_609_rft_smoke.txt"
echo "$$" > "${LOG_DIR}/launcher.pid"

cd "${REPO}"

export EMBODIED_PATH="${REPO}/examples/embodiment"
export REPO_PATH="${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export GENESIS_HEADLESS=1
# IMPORTANT: RLinf FlexiblePlacementStrategy uses GLOBAL hardware ranks here.
# These placement values are physical GPU IDs, not process-local CUDA IDs:
#   placement 2,4,5,7 -> physical GPUs 2,4,5,7 (rollout)
#   placement 3       -> physical GPU 3 (actor)
#   placement 6:0     -> physical GPU 6 (Genesis env, process 0)
# Each isolated worker later sees its assigned device as local CUDA GPU 0.
# Keep all physical IDs visible to the cluster so global rank resolution works.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT_MS=3600000
export RLINF_RAY_LOCAL=1
# Defaults retain the shared project locations.  Callers can override these
# for an isolated smoke run without creating a second Ray session under an
# active training run's temporary directory.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray609rftsmoke}"
export TMPDIR="${TMPDIR:-/home/clk/workspace/tmp}"

mkdir -p "$RAY_TMPDIR" "$TMPDIR"

exec > "${LOG_DIR}/train.log" 2>&1

echo "============================================================"
echo "Episode 609 QUCTc6BB5sX Decision-MaxRL sRGB RFT smoke"
echo "LOG_DIR=${LOG_DIR}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "RESUME_DIR=<none; fresh RFT from qwen3.5_awr>"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "GSAM_DIR=${GSAM_DIR}"
# Placement values below are physical/global GPU IDs (see mapping above).
# Keep these comments outside the backslash-continued Hydra command below.
echo "============================================================"

# Decision-MaxRL uses one fixed on-policy group of four trajectories. Candidate
# selection and Reinforce-Ada retries remain disabled so all N samples enter the
# MaxRL estimator directly. Periodic eval is disabled to avoid a second Genesis
# scene on the environment GPU.
exec "${PYTHON}" "${ENTRY}" \
  --config-path "${CONFIG_DIR}" \
  --config-name genark_grpo_qwen_multiscene \
  runner.logger.log_path="${LOG_DIR}" \
  actor.model.model_path="${BASE_MODEL}" \
  rollout.model.model_path="${BASE_MODEL}" \
  runner.max_epochs=1 \
  runner.max_steps=-1 \
  runner.val_check_interval=-1 \
  runner.save_interval=1 \
  'cluster.component_placement.rollout.placement=2' \
  cluster.component_placement.actor.placement=0 \
  cluster.component_placement.env.placement=7:0 \
  env.train.total_num_envs=4 \
  env.train.genesis_backend=multiscene \
  env.train.multi_scene.scene_count=1 \
  env.train.multi_scene.scenes_per_gpu=1 \
  env.train.multi_scene.gpu_budget=1 \
  'env.train.multi_scene.scenes=[mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb]' \
  env.train.init_params.glb_cache_dir="${SCENE_GLB_CACHE}" \
  ++env.train.episode_overfit.enabled=true \
  ++env.train.episode_overfit.scene_id=mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb \
  ++env.train.episode_overfit.episode_id=609 \
  env.train.group_size=4 \
  env.train.max_episode_steps=60 \
  env.train.max_steps_per_rollout_epoch=100 \
  env.train.max_decisions_per_rollout_epoch=4 \
  env.train.camera_height=0.88 \
  env.train.fov=79 \
  env.train.cam_res='[640,480]' \
  env.train.enable_depth_obs=true \
  env.train.enable_4dir_render=true \
  env.train.enable_4dir_depth_obs=true \
  env.train.auto_reset=true \
  env.train.ignore_terminations=true \
  ++env.train.reward_profile=nav \
  env.train.reward_mode=decision_nav \
  env.train.geo_coef=0.1 \
  env.train.geo_step_clip=0.5 \
  env.train.nav_terminal_reward_enabled=true \
  env.train.nav_path_coef=1.0 \
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
  env.train.wrong_stop_penalty=-1.0 \
  env.train.conditional_wrong_stop_penalty=true \
  env.train.conditional_wrong_stop_ratio_min=0.5 \
  env.train.conditional_wrong_stop_ratio_max=2.0 \
  env.train.parse_fail_penalty=-0.1 \
  env.train.no_stop_penalty=-0.5 \
  env.train.process_reward_enabled=true \
  env.train.process_progress_coef=0.5 \
  env.train.process_progress_cap=6.0 \
  algorithm.group_size=4 \
  algorithm.adv_type=decision_maxrl \
  algorithm.normalize_advantages=false \
  ++algorithm.maxrl_process_coef=1.0 \
  ++algorithm.maxrl_epsilon=1.0e-6 \
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
  env.eval.max_episode_steps=60 \
  env.eval.max_steps_per_rollout_epoch=100 \
  env.eval.camera_height=0.88 \
  env.eval.fov=79 \
  env.eval.cam_res='[640,480]' \
  env.eval.enable_depth_obs=true \
  env.eval.enable_4dir_render=true \
  env.eval.enable_4dir_depth_obs=true \
  env.eval.auto_reset=true \
  env.eval.ignore_terminations=true \
  actor.global_batch_size=16 \
  actor.micro_batch_size=1 \
  actor.optim.lr=5.0e-7 \
  rollout.max_model_len=8192 \
  rollout.gpu_memory_utilization=0.25 \
  rollout.max_num_seqs=8 \
  actor.model.history_max_frames=2 \
  actor.model.min_pixels=4096 \
  actor.model.max_pixels=122500 \
  ++rollout.mm_processor_kwargs.min_pixels=4096 \
  ++rollout.mm_processor_kwargs.max_pixels=122500 \
  ++rollout.limit_mm_per_prompt.image=100 \
  ++rollout.model.min_pixels=4096 \
  ++rollout.model.max_pixels=122500 \
  actor.fsdp_config.gradient_checkpointing=true \
  rollout.model.history_max_frames=2 \
  ++actor.model.prompt_style=lavira_waypoint \
  ++actor.model.lavira_runtime.enabled=true \
  ++actor.model.lavira_runtime.map_backend=source \
  ++actor.model.lavira_runtime.hfov_deg=79.0 \
  ++actor.model.lavira_runtime.camera_height=0.88 \
  ++actor.model.lavira_runtime.initial_scan_turns=12 \
  ++actor.model.lavira_runtime.layered_history=true \
  ++actor.model.lavira_runtime.history_wp_max=8 \
  ++actor.model.lavira_runtime.layered_backtrack_radius_m=6.0 \
  ++actor.model.lavira_runtime.backtrack_second_chance=true \
  ++actor.model.lavira_runtime.map_visualization.enabled=true \
  ++actor.model.lavira_runtime.map_visualization.output_dir="${LOG_DIR}/lavira_maps" \
  ++actor.model.lavira_runtime.map_visualization.episode_json=/home/clk/workspace/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json \
  ++actor.model.lavira_runtime.map_visualization.episode_id=609 \
  ++actor.model.lavira_runtime.map_visualization.save_raw_every=12 \
  ++actor.model.grounded_sam.enabled=true \
  ++actor.model.grounded_sam.visualize=true \
  ++actor.model.grounded_sam.visualization_dir="${LOG_DIR}/grounded_sam" \
  ++actor.model.grounded_sam.dino_config_path="${GSAM_DIR}/GroundingDINO_SwinT_OGC.py" \
  ++actor.model.grounded_sam.dino_checkpoint_path="${GSAM_DIR}/groundingdino_swint_ogc.pth" \
  ++actor.model.grounded_sam.repvit_sam_checkpoint_path=/home/nvme01/uni-lavira/data/grounded_sam/repvit_sam.pt \
  ++actor.model.grounded_sam.device=cuda \
  ++actor.model.grounded_sam.box_threshold=0.25 \
  ++actor.model.grounded_sam.text_threshold=0.25 \
  ++actor.model.grounded_sam.fail_fast_on_missing_assets=true \
  ++rollout.model.prompt_style=lavira_waypoint \
  ++rollout.model.lavira_runtime.enabled=true \
  ++rollout.model.lavira_runtime.map_backend=source \
  ++rollout.model.lavira_runtime.hfov_deg=79.0 \
  ++rollout.model.lavira_runtime.camera_height=0.88 \
  ++rollout.model.lavira_runtime.initial_scan_turns=12 \
  ++rollout.model.lavira_runtime.layered_history=true \
  ++rollout.model.lavira_runtime.history_wp_max=8 \
  ++rollout.model.lavira_runtime.layered_backtrack_radius_m=6.0 \
  ++rollout.model.lavira_runtime.backtrack_second_chance=true \
  ++rollout.model.lavira_runtime.map_visualization.enabled=true \
  ++rollout.model.lavira_runtime.map_visualization.output_dir="${LOG_DIR}/lavira_maps" \
  ++rollout.model.lavira_runtime.map_visualization.episode_json=/home/clk/workspace/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json \
  ++rollout.model.lavira_runtime.map_visualization.episode_id=609 \
  ++rollout.model.lavira_runtime.map_visualization.save_raw_every=12 \
  ++rollout.model.grounded_sam.enabled=true \
  ++rollout.model.grounded_sam.visualize=true \
  ++rollout.model.grounded_sam.visualization_dir="${LOG_DIR}/grounded_sam" \
  ++rollout.model.grounded_sam.dino_config_path="${GSAM_DIR}/GroundingDINO_SwinT_OGC.py" \
  ++rollout.model.grounded_sam.dino_checkpoint_path="${GSAM_DIR}/groundingdino_swint_ogc.pth" \
  ++rollout.model.grounded_sam.repvit_sam_checkpoint_path=/home/nvme01/uni-lavira/data/grounded_sam/repvit_sam.pt \
  ++rollout.model.grounded_sam.device=cuda \
  ++rollout.model.grounded_sam.box_threshold=0.25 \
  ++rollout.model.grounded_sam.text_threshold=0.25 \
  ++rollout.model.grounded_sam.fail_fast_on_missing_assets=true \
  "$@"

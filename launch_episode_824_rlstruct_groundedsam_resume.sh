#!/usr/bin/env bash
set -euo pipefail

REPO=/home/clk/workspace/RLinf
CONFIG_DIR="${REPO}/examples/embodiment/config"
ENTRY="${REPO}/examples/embodiment/train_embodied_agent.py"
PYTHON=/home/clk/miniconda3/envs/genesis-vllm/bin/python
# BASE_MODEL=/home/lhx/workspace/models/qwen3.5_awr
BASE_MODEL=/home/lhx/workspace/models/qwen3.5_awr
RESUME_DIR=/home/clk/workspace/RLinf/logs/20260708-124345-episode-overfit-824-candidate8-rlstruct-groundedsam-awr-gpu357-actor5-depth/genark_grpo_qwen_multiscene/checkpoints/global_step_25
GSAM_DIR="${REPO}/assets/grounded_sam"
LOG_DIR="${REPO}/logs/$(date +'%Y%m%d-%H%M%S')-episode-overfit-824-candidate16-condstop-rlstruct-groundedsam-awr-rollout3457-actor2-env6-depth"

mkdir -p "${LOG_DIR}"
echo "${LOG_DIR}" > "${REPO}/logs/latest_episode_824_groundedsam_resume.txt"
echo "$$" > "${LOG_DIR}/launcher.pid"

cd "${REPO}"

export EMBODIED_PATH="${REPO}/examples/embodiment"
export REPO_PATH="${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export GENESIS_HEADLESS=1
export CUDA_VISIBLE_DEVICES=1,3,4,5,6,7
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT_MS=3600000
export RLINF_RAY_LOCAL=1
export RAY_TMPDIR=/home/clk/workspace/ray_tmp
export TMPDIR=/home/clk/workspace/tmp

mkdir -p "$RAY_TMPDIR" "$TMPDIR"

exec > "${LOG_DIR}/train.log" 2>&1

echo "============================================================"
echo "Episode 824 RL-Struct reward + GroundedSAM AWR base"
echo "LOG_DIR=${LOG_DIR}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "RESUME_DIR=${RESUME_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "GSAM_DIR=${GSAM_DIR}"
echo "============================================================"

exec "${PYTHON}" "${ENTRY}" \
  --config-path "${CONFIG_DIR}" \
  --config-name genark_grpo_qwen_multiscene \
  runner.logger.log_path="${LOG_DIR}" \
  runner.resume_dir="${RESUME_DIR}" \
  actor.model.model_path="${BASE_MODEL}" \
  rollout.model.model_path="${BASE_MODEL}" \
  runner.max_epochs=500 \
  runner.max_steps=-1 \
  runner.val_check_interval=25 \
  runner.save_interval=25 \
  'cluster.component_placement.rollout.placement="1,2,3,5"' \
  cluster.component_placement.actor.placement=0 \
  cluster.component_placement.env.placement=4:0 \
  env.train.total_num_envs=16 \
  env.train.genesis_backend=multiscene \
  env.train.multi_scene.scene_count=1 \
  env.train.multi_scene.scenes_per_gpu=1 \
  env.train.multi_scene.gpu_budget=1 \
  'env.train.multi_scene.scenes=[mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb]' \
  ++env.train.episode_overfit.enabled=true \
  ++env.train.episode_overfit.scene_id=mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb \
  ++env.train.episode_overfit.episode_id=824 \
  env.train.group_size=4 \
  env.train.max_episode_steps=80 \
  env.train.max_steps_per_rollout_epoch=120 \
  env.train.auto_reset=true \
  env.train.ignore_terminations=true \
  ++env.train.reward_profile=nav \
  env.train.reward_mode=decision_nav \
  env.train.geo_coef=0.1 \
  env.train.geo_step_clip=0.5 \
  env.train.nav_terminal_reward_enabled=true \
  env.train.nav_path_coef=1.0 \
  env.train.nav_endpoint_coef=1.0 \
  env.train.nav_endpoint_decay=0.2 \
  env.train.gsam_reward_enabled=false \
  env.train.ndtw_coef=0.0 \
  env.train.format_reward_coef=0.0 \
  env.train.json_reward_coef=0.0 \
  env.train.struct_reward_coef=0.0 \
  env.train.field_format_reward_coef=0.0 \
  env.train.length_reward_coef=0.0 \
  env.train.bbox_reward_coef=0.0 \
  env.train.sr_coef=5.0 \
  env.train.wrong_stop_penalty=-1.0 \
  env.train.conditional_wrong_stop_penalty=true \
  env.train.conditional_wrong_stop_ratio_min=0.5 \
  env.train.conditional_wrong_stop_ratio_max=2.0 \
  env.train.parse_fail_penalty=-0.1 \
  env.train.no_stop_penalty=-0.5 \
  env.train.process_reward_enabled=true \
  env.train.process_progress_coef=0.5 \
  env.train.process_progress_cap=6.0 \
  algorithm.acr_reward_std_threshold=1e-6 \
  ++algorithm.low_reward_std_threshold=0.05 \
  ++algorithm.candidate_group_selection.enabled=true \
  ++algorithm.candidate_group_selection.candidate_k=16 \
  ++algorithm.candidate_group_selection.train_group_size=4 \
  ++algorithm.candidate_group_selection.mode=diverse_by_reward_and_behavior \
  ++algorithm.candidate_group_selection.only_episode_overfit=true \
  env.eval.total_num_envs=4 \
  env.eval.genesis_backend=multiscene \
  env.eval.multi_scene.scene_count=1 \
  env.eval.multi_scene.scenes_per_gpu=1 \
  env.eval.multi_scene.gpu_budget=1 \
  'env.eval.multi_scene.scenes=[mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb]' \
  ++env.eval.episode_overfit.enabled=true \
  ++env.eval.episode_overfit.scene_id=mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb \
  ++env.eval.episode_overfit.episode_id=824 \
  env.eval.group_size=1 \
  env.eval.max_episode_steps=80 \
  env.eval.max_steps_per_rollout_epoch=120 \
  env.eval.auto_reset=true \
  env.eval.ignore_terminations=true \
  actor.global_batch_size=16 \
  actor.micro_batch_size=1 \
  actor.optim.lr=1.0e-6 \
  rollout.max_model_len=8192 \
  rollout.gpu_memory_utilization=0.20 \
  rollout.max_num_seqs=8 \
  actor.model.history_max_frames=1 \
  rollout.model.history_max_frames=1 \
  ++actor.model.prompt_style=lavira_waypoint \
  ++actor.model.lavira_controller=true \
  ++actor.model.grounded_sam.enabled=true \
  ++actor.model.grounded_sam.dino_config_path="${GSAM_DIR}/GroundingDINO_SwinT_OGC.py" \
  ++actor.model.grounded_sam.dino_checkpoint_path="${GSAM_DIR}/groundingdino_swint_ogc.pth" \
  ++actor.model.grounded_sam.sam_checkpoint_path="${GSAM_DIR}/sam_vit_h_4b8939.pth" \
  ++actor.model.grounded_sam.sam_encoder_version=vit_h \
  ++actor.model.grounded_sam.device=cuda \
  ++actor.model.grounded_sam.box_threshold=0.25 \
  ++actor.model.grounded_sam.text_threshold=0.25 \
  ++actor.model.grounded_sam.fail_fast_on_missing_assets=true \
  ++rollout.model.prompt_style=lavira_waypoint \
  ++rollout.model.lavira_controller=true \
  "$@"

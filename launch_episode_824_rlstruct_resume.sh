#!/usr/bin/env bash
set -euo pipefail

REPO=/home/clk/workspace/RLinf
CONFIG_DIR="${REPO}/examples/embodiment/config"
ENTRY="${REPO}/examples/embodiment/train_embodied_agent.py"
PYTHON=/home/clk/miniconda3/envs/genesis-vllm/bin/python
CKPT=/home/clk/workspace/RLinf/logs/20260703-193328-episode-overfit-824-longcurriculum-40-70-100-gpu235-depth/genark_grpo_qwen_multiscene/checkpoints/global_step_75
LOG_DIR="${REPO}/logs/$(date +'%Y%m%d-%H%M%S')-episode-overfit-824-rlstruct-reward-resume-gpu235-depth"

mkdir -p "${LOG_DIR}"
echo "${LOG_DIR}" > "${REPO}/logs/latest_episode_824_rlstruct_resume.txt"
echo "$$" > "${LOG_DIR}/launcher.pid"

cd "${REPO}"

export EMBODIED_PATH="${REPO}/examples/embodiment"
export REPO_PATH="${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export GENESIS_HEADLESS=1
export CUDA_VISIBLE_DEVICES=2,3,5
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT_MS=3600000

exec > "${LOG_DIR}/train.log" 2>&1

echo "============================================================"
echo "Episode 824 RL-Struct reward resume"
echo "LOG_DIR=${LOG_DIR}"
echo "CKPT=${CKPT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

exec "${PYTHON}" "${ENTRY}" \
  --config-path "${CONFIG_DIR}" \
  --config-name genark_grpo_qwen_multiscene \
  runner.logger.log_path="${LOG_DIR}" \
  runner.resume_dir="${CKPT}" \
  runner.max_epochs=500 \
  runner.max_steps=-1 \
  runner.val_check_interval=25 \
  runner.save_interval=25 \
  cluster.component_placement.rollout.placement=5 \
  cluster.component_placement.actor.placement=3 \
  cluster.component_placement.env.placement=2:0 \
  env.train.total_num_envs=4 \
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
  env.eval.total_num_envs=1 \
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
  actor.global_batch_size=32 \
  actor.micro_batch_size=1 \
  actor.optim.lr=1.0e-6 \
  rollout.max_model_len=8192 \
  rollout.gpu_memory_utilization=0.25 \
  rollout.max_num_seqs=8 \
  actor.model.history_max_frames=1 \
  rollout.model.history_max_frames=1 \
  ++actor.model.prompt_style=lavira_waypoint \
  ++actor.model.lavira_controller=true \
  ++rollout.model.prompt_style=lavira_waypoint \
  ++rollout.model.lavira_controller=true

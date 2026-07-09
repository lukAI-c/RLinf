#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/clk/workspace/RLinf
PY=/home/clk/miniconda3/envs/genesis-vllm/bin/python
CONFIG_DIR="$ROOT/examples/embodiment/config"
TRAIN="$ROOT/examples/embodiment/train_embodied_agent.py"

export EMBODIED_PATH="$ROOT/examples/embodiment"
export REPO_PATH="$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export GENESIS_HEADLESS=1
ROLLOUT_GPU="${ROLLOUT_GPU:-4}"
ENV_GPUS="${ENV_GPUS:-6-7}"
CUDA_DEVICES="${CUDA_DEVICES:-4,6,7}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT_MS=3600000

BASE_LOG="$ROOT/logs/nav100-zeroshot-remaining-$(date +'%Y%m%d-%H%M%S')"
mkdir -p "$BASE_LOG"

run_batch() {
  local name="$1"
  local scene_a="$2"
  local scene_b="$3"
  local log_dir="$BASE_LOG/$name"
  mkdir -p "$log_dir"
  echo "============================================================" | tee "$log_dir/eval.log"
  echo "[nav100 remaining] batch=$name" | tee -a "$log_dir/eval.log"
  echo "[nav100 remaining] scenes=$scene_a,$scene_b" | tee -a "$log_dir/eval.log"
  echo "[nav100 remaining] rollout_gpu=$ROLLOUT_GPU env_gpus=$ENV_GPUS cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$log_dir/eval.log"
  echo "[nav100 remaining] log_dir=$log_dir" | tee -a "$log_dir/eval.log"
  echo "============================================================" | tee -a "$log_dir/eval.log"

  "$PY" "$TRAIN" \
    --config-path "$CONFIG_DIR" \
    --config-name genark_eval_qwen_zeroshot \
    runner.logger.log_path="$log_dir" \
    cluster.component_placement.rollout.placement="$ROLLOUT_GPU" \
    cluster.component_placement.actor.placement="$ROLLOUT_GPU" \
    cluster.component_placement.env.placement="$ENV_GPUS:0" \
    env.eval.total_num_envs=2 \
    env.eval.multi_scene.scene_count=2 \
    env.eval.multi_scene.scenes_per_gpu=1 \
    env.eval.multi_scene.gpu_budget=2 \
    "env.eval.multi_scene.scenes=[\"$scene_a\",\"$scene_b\"]" \
    2>&1 | tee -a "$log_dir/eval.log"
}

run_batch batch01_2azQ_QUCT \
  mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb \
  mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb

run_batch batch02_Z6MF_EU6 \
  mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb \
  mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb

run_batch batch03_X7Hy_x8F5 \
  mp3d/X7HyMhZNoso/X7HyMhZNoso.glb \
  mp3d/x8F5xyUWy9e/x8F5xyUWy9e.glb

run_batch batch04_oLBM_8194 \
  mp3d/oLBMNvg9in8/oLBMNvg9in8.glb \
  mp3d/8194nk5LbLH/8194nk5LbLH.glb

echo "[nav100 remaining] all batches completed. base_log=$BASE_LOG"

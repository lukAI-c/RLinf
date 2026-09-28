#!/usr/bin/env bash
set -euo pipefail

# Frozen-policy diagnostic: separate intrinsic waypoint/FMM behaviour from
# any effect of Decision-MaxRL updates. This command launches no actor worker.
REPO=/home/clk/workspace/RLinf
CONFIG_DIR="${REPO}/examples/embodiment/config"
ENTRY="${REPO}/examples/embodiment/train_embodied_agent.py"
PYTHON=/home/clk/miniconda3/envs/genesis-vllm/bin/python
BASE_MODEL=/home/lhx/workspace/models/qwen3.5_awr
GSAM_DIR="${REPO}/assets/grounded_sam"
SCENE_GLB_CACHE=/home/clk/workspace/genark/glb_cache
LOG_DIR="${REPO}/logs/$(date +'%Y%m%d-%H%M%S')-episode-259-awr-frozen-eval4-rollout3-env6"

mkdir -p "${LOG_DIR}"
echo "${LOG_DIR}" > "${REPO}/logs/latest_episode_259_awr_frozen_eval.txt"

export EMBODIED_PATH="${REPO}/examples/embodiment"
export REPO_PATH="${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export GENESIS_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT_MS=3600000
export RLINF_RAY_LOCAL=1
export RAY_TMPDIR="${RAY_TMPDIR:-/home/clk/workspace/ray_tmp}"
export TMPDIR="${TMPDIR:-/home/clk/workspace/tmp}"
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"

cd "${REPO}"
exec > "${LOG_DIR}/eval.log" 2>&1

echo "============================================================"
echo "Frozen qwen3.5_awr diagnostic: episode 259, TbHJrupSAjP"
echo "TRIALS=4 parallel samples; only_eval=true; actor update disabled"
echo "ROLL_OUT_GPU=3 ENV_GPU=6"
echo "LOG_DIR=${LOG_DIR}"
echo "============================================================"

exec "${PYTHON}" "${ENTRY}" \
  --config-path "${CONFIG_DIR}" \
  --config-name genark_eval_qwen_zeroshot \
  runner.logger.log_path="${LOG_DIR}" \
  runner.only_eval=true \
  runner.val_check_interval=-1 \
  runner.max_epochs=1 \
  runner.max_steps=-1 \
  'cluster.component_placement.rollout.placement="3"' \
  cluster.component_placement.actor.placement=3 \
  cluster.component_placement.env.placement=6:0 \
  actor.model.model_path="${BASE_MODEL}" \
  rollout.model.model_path="${BASE_MODEL}" \
  algorithm.eval_rollout_epoch=1 \
  algorithm.sampling_params.do_sample=true \
  algorithm.sampling_params.temperature_eval=1.0 \
  env.eval.total_num_envs=4 \
  env.eval.genesis_backend=multiscene \
  env.eval.multi_scene.scene_count=1 \
  env.eval.multi_scene.scenes_per_gpu=1 \
  env.eval.multi_scene.gpu_budget=1 \
  'env.eval.multi_scene.scenes=[mp3d/TbHJrupSAjP/TbHJrupSAjP.glb]' \
  env.eval.init_params.glb_cache_dir="${SCENE_GLB_CACHE}" \
  ++env.eval.episode_overfit.enabled=true \
  ++env.eval.episode_overfit.scene_id=mp3d/TbHJrupSAjP/TbHJrupSAjP.glb \
  ++env.eval.episode_overfit.episode_id=259 \
  env.eval.group_size=1 \
  ++env.eval.is_eval=true \
  env.eval.auto_reset=false \
  env.eval.ignore_terminations=false \
  ++env.eval.cyclic_episode_sampling=false \
  ++env.eval.eval_repeats_per_episode=1 \
  env.eval.max_episode_steps=92 \
  env.eval.max_steps_per_rollout_epoch=200 \
  env.eval.enable_depth_obs=true \
  env.eval.enable_4dir_render=true \
  env.eval.enable_4dir_depth_obs=false \
  env.eval.success_distance=3.0 \
  rollout.max_model_len=8192 \
  rollout.max_num_seqs=8 \
  rollout.gpu_memory_utilization=0.25 \
  actor.model.history_max_frames=1 \
  rollout.model.history_max_frames=1 \
  ++actor.model.prompt_style=lavira_waypoint \
  ++actor.model.lavira_runtime.enabled=true \
  ++actor.model.lavira_runtime.initial_scan_turns=12 \
  ++actor.model.lavira_runtime.map_visualization.enabled=true \
  ++actor.model.lavira_runtime.map_visualization.output_dir="${LOG_DIR}/lavira_maps" \
  ++actor.model.lavira_runtime.map_visualization.episode_json=/home/clk/workspace/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json \
  ++actor.model.lavira_runtime.map_visualization.episode_id=259 \
  ++actor.model.lavira_runtime.map_visualization.save_raw_every=12 \
  ++actor.model.grounded_sam.enabled=true \
  ++actor.model.grounded_sam.dino_config_path="${GSAM_DIR}/GroundingDINO_SwinT_OGC.py" \
  ++actor.model.grounded_sam.dino_checkpoint_path="${GSAM_DIR}/groundingdino_swint_ogc.pth" \
  ++actor.model.grounded_sam.repvit_sam_checkpoint_path=/home/nvme01/uni-lavira/data/grounded_sam/repvit_sam.pt \
  ++actor.model.grounded_sam.device=cuda \
  ++actor.model.grounded_sam.box_threshold=0.25 \
  ++actor.model.grounded_sam.text_threshold=0.25 \
  ++actor.model.grounded_sam.fail_fast_on_missing_assets=true \
  ++rollout.model.prompt_style=lavira_waypoint \
  ++rollout.model.lavira_runtime.enabled=true \
  ++rollout.model.lavira_runtime.initial_scan_turns=12 \
  ++rollout.model.lavira_runtime.map_visualization.enabled=true \
  ++rollout.model.lavira_runtime.map_visualization.output_dir="${LOG_DIR}/lavira_maps" \
  ++rollout.model.lavira_runtime.map_visualization.episode_json=/home/clk/workspace/genark/data/datasets/OpenNav_R2R-CE_100_bertidx.json \
  ++rollout.model.lavira_runtime.map_visualization.episode_id=259 \
  ++rollout.model.lavira_runtime.map_visualization.save_raw_every=12 \
  ++rollout.model.grounded_sam.enabled=true \
  ++rollout.model.grounded_sam.dino_config_path="${GSAM_DIR}/GroundingDINO_SwinT_OGC.py" \
  ++rollout.model.grounded_sam.dino_checkpoint_path="${GSAM_DIR}/groundingdino_swint_ogc.pth" \
  ++rollout.model.grounded_sam.repvit_sam_checkpoint_path=/home/nvme01/uni-lavira/data/grounded_sam/repvit_sam.pt \
  ++rollout.model.grounded_sam.device=cuda \
  ++rollout.model.grounded_sam.box_threshold=0.25 \
  ++rollout.model.grounded_sam.text_threshold=0.25 \
  ++rollout.model.grounded_sam.fail_fast_on_missing_assets=true

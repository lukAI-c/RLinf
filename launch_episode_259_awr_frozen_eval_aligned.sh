#!/usr/bin/env bash
set -euo pipefail

# Frozen episode-259 diagnostic with the Genesis camera/agent contract aligned
# to LHX LaViRA's Habitat R2R setup. No actor worker or optimizer is created.
# Keep prompt, history and sampling equal to the preceding RFT run so that the
# simulator/camera contract is the only intentional experimental change.
REPO=/home/clk/workspace/RLinf
CONFIG_DIR="${REPO}/examples/embodiment/config"
ENTRY="${REPO}/examples/embodiment/train_embodied_agent.py"
PYTHON=/home/clk/miniconda3/envs/genesis-vllm/bin/python
BASE_MODEL=/home/lhx/workspace/models/qwen3.5_awr
GSAM_DIR="${REPO}/assets/grounded_sam"
LOG_DIR="${REPO}/logs/$(date +'%Y%m%d-%H%M%S')-episode259-awr-frozen-genesis-aligned79"

mkdir -p "${LOG_DIR}"
echo "${LOG_DIR}" > "${REPO}/logs/latest_episode_259_awr_frozen_aligned.txt"

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
# Short paths avoid Ray's UNIX-domain socket path limit.
export RAY_TMPDIR=/home/clk/workspace/r
export TMPDIR=/home/clk/workspace/t
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"

cd "${REPO}"
exec > "${LOG_DIR}/eval.log" 2>&1

echo "============================================================"
echo "Frozen qwen3.5_awr Genesis alignment diagnostic"
echo "episode=259 scene=TbHJrupSAjP trials=4"
echo "camera: 640x480 HFOV=79 height=0.88m"
echo "motion: forward=0.25m turn=30deg radius=0.10m sliding=true"
echo "budget: 20 decisions / 300 primitive steps"
echo "rollout_gpu=7 env_gpu=5 actor=disabled"
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
  'cluster.component_placement.rollout.placement="7"' \
  cluster.component_placement.actor.placement=7 \
  cluster.component_placement.env.placement=5:0 \
  actor.model.model_path="${BASE_MODEL}" \
  rollout.model.model_path="${BASE_MODEL}" \
  algorithm.eval_rollout_epoch=1 \
  algorithm.sampling_params.do_sample=true \
  algorithm.sampling_params.temperature_eval=0.6 \
  env.eval.total_num_envs=4 \
  env.eval.genesis_backend=multiscene \
  env.eval.multi_scene.scene_count=1 \
  env.eval.multi_scene.scenes_per_gpu=1 \
  env.eval.multi_scene.gpu_budget=1 \
  'env.eval.multi_scene.scenes=[mp3d/TbHJrupSAjP/TbHJrupSAjP.glb]' \
  env.eval.init_params.glb_cache_dir=/home/clk/workspace/genark/glb_cache \
  ++env.eval.episode_overfit.enabled=true \
  ++env.eval.episode_overfit.scene_id=mp3d/TbHJrupSAjP/TbHJrupSAjP.glb \
  ++env.eval.episode_overfit.episode_id=259 \
  env.eval.group_size=1 \
  ++env.eval.is_eval=true \
  env.eval.auto_reset=false \
  env.eval.ignore_terminations=false \
  ++env.eval.cyclic_episode_sampling=false \
  ++env.eval.eval_roll_through_episode_pool=false \
  ++env.eval.eval_repeats_per_episode=1 \
  ++env.eval.max_decisions_per_rollout_epoch=20 \
  env.eval.max_episode_steps=300 \
  env.eval.max_steps_per_rollout_epoch=300 \
  env.eval.camera_height=0.88 \
  ++env.eval.depth_min=0.1 \
  ++env.eval.depth_max=5.0 \
  env.eval.fov=79 \
  env.eval.agent_radius=0.10 \
  env.eval.step_move=0.25 \
  env.eval.step_turn=0.5236 \
  env.eval.allow_sliding=true \
  env.eval.cam_res='[640,480]' \
  env.eval.enable_depth_obs=true \
  env.eval.enable_4dir_render=true \
  env.eval.enable_4dir_depth_obs=true \
  env.eval.success_distance=3.0 \
  env.eval.video_cfg.save_video=true \
  env.eval.video_cfg.video_base_dir="${LOG_DIR}/video/eval" \
  rollout.max_model_len=8192 \
  rollout.max_num_seqs=8 \
  rollout.gpu_memory_utilization=0.25 \
  actor.model.history_max_frames=2 \
  rollout.model.history_max_frames=2 \
  actor.model.min_pixels=4096 \
  actor.model.max_pixels=122500 \
  ++rollout.model.min_pixels=4096 \
  ++rollout.model.max_pixels=122500 \
  ++actor.model.prompt_style=lavira_waypoint \
  ++actor.model.lavira_runtime.enabled=true \
  ++actor.model.lavira_runtime.map_backend=source \
  ++actor.model.lavira_runtime.hfov_deg=79.0 \
  ++actor.model.lavira_runtime.camera_height=0.88 \
  ++actor.model.lavira_runtime.initial_scan_turns=12 \
  ++actor.model.lavira_runtime.fmm_goal_threshold_m=1.0 \
  ++actor.model.lavira_runtime.max_steps_to_target=30 \
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
  ++rollout.model.lavira_runtime.map_backend=source \
  ++rollout.model.lavira_runtime.hfov_deg=79.0 \
  ++rollout.model.lavira_runtime.camera_height=0.88 \
  ++rollout.model.lavira_runtime.initial_scan_turns=12 \
  ++rollout.model.lavira_runtime.fmm_goal_threshold_m=1.0 \
  ++rollout.model.lavira_runtime.max_steps_to_target=30 \
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

#!/usr/bin/env bash
# STOP second stage. Separate from endpoint-only P0.
# Init: 6-ep endpoint global_step_6 (bonus=0 freeze: DTG 9.56 -> 7.31 m).
# Change vs P0: clean_stop_bonus=1 and STOP tokens on the PPO mask.
# Train the same 5 unsaturated Z6 episodes that already get closer.
# Do not treat held-out DTG as the keep gate for this run.
set -euo pipefail

REPO=/home/clk/workspace/RLinf
export LOG_SUFFIX="${LOG_SUFFIX:--stop-stage}"
export QWEN_NAV_LOSS_MASK_INCLUDE_STOP="${QWEN_NAV_LOSS_MASK_INCLUDE_STOP:-1}"
export RFT_CLEAN_STOP_BONUS="${RFT_CLEAN_STOP_BONUS:-1.0}"
export RFT_MAX_STEPS="${RFT_MAX_STEPS:-6}"
export RFT_RESUME_DIR="${RFT_RESUME_DIR:-${REPO}/logs/20260823-132620-sft1200-robostral-terminal-rft/genark_grpo_qwen_multiscene/checkpoints/global_step_6}"
export EPISODE_ASSIGNMENT_FILE="${EPISODE_ASSIGNMENT_FILE:-${REPO}/examples/embodiment/config/robostral_endpoint_train_no586.json}"
export RFT_EPISODE_BLOCKLIST="${RFT_EPISODE_BLOCKLIST:-[586,207,432,550,559,705]}"
export RFT_GROUP_SIZE="${RFT_GROUP_SIZE:-8}"
export RFT_TOTAL_ENVS="${RFT_TOTAL_ENVS:-8}"

exec bash "${REPO}/launch_robostral_terminal_rft.sh"

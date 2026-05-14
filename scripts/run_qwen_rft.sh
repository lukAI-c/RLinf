#!/bin/bash
# GenArk GRPO RFT 启动脚本
#
# 用法:
#   ./scripts/run_qwen_rft.sh                    # 默认 full 模式（全参微调）
#   ./scripts/run_qwen_rft.sh smoke              # 验证 pipeline（~5 min）
#   ./scripts/run_qwen_rft.sh medium             # 调试 reward 信号
#   ./scripts/run_qwen_rft.sh lora               # LoRA 微调（~10 min/step）
#   RESUME_DIR=/path/ckpt ./scripts/run_qwen_rft.sh  # 从 checkpoint 继续
#
# GPU 分配（支持 3 卡或 4 卡）：
#   GPUS=2,3,4,5 ./scripts/run_qwen_rft.sh      # 4 卡：env 独占第 4 张
#   GPUS=3,4,5   ./scripts/run_qwen_rft.sh      # 3 卡：rollout+env 共享
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMBODIED_PATH="$(dirname "$SCRIPT_DIR")/examples/embodiment"
PYTHON=/home/clk/miniconda3/envs/genesis/bin/python

GPUS=${GPUS:-2,3,4,5}    # 默认 4 卡
MODE=${1:-full}
RESUME_DIR=${RESUME_DIR:-}

# GPU 分配
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_GPUS=${#GPU_ARR[@]}
if [ "$NUM_GPUS" -ge 3 ]; then
  ACTOR_GPUS="${GPU_ARR[0]}-${GPU_ARR[1]}"   # 3/4 卡：前两张给 actor (FSDP 2-rank)
  ROLLOUT_GPU="${GPU_ARR[2]}"                 # 第三张给 rollout
else
  ACTOR_GPUS="${GPU_ARR[0]}"                  # 2 卡：actor 单卡 (FSDP 1-rank)
  ROLLOUT_GPU="${GPU_ARR[1]}"                 # 第二张给 rollout
fi
if [ "$NUM_GPUS" -ge 4 ]; then
  ENV_GPU="${GPU_ARR[3]}"                   # 4 卡：env 独占第四张
  ENV_INFO="dedicated(${ENV_GPU})"
else
  ENV_GPU="$ROLLOUT_GPU"                    # 2/3 卡：env 和 rollout 共享
  ENV_INFO="shared with rollout(${ROLLOUT_GPU})"
fi

LOG_DIR=/home/clk/workspace/results/genark_grpo_qwen
LOG_FILE="$LOG_DIR/rft_$(date +%Y%m%d_%H%M%S)_${MODE}.log"
mkdir -p "$LOG_DIR"

echo "Mode      : $MODE"
echo "GPUs      : $GPUS"
echo "  Actor   : ${ACTOR_GPUS} (FSDP 2-rank)"
echo "  Rollout : ${ROLLOUT_GPU}"
echo "  Env     : ${ENV_INFO}"
echo "Log       : $LOG_FILE"
echo "Resume    : ${RESUME_DIR:-none}"
echo ""

# ─── 模式参数 ────────────────────────────────────────────────────────────────
case "$MODE" in
smoke)
  # ~5 min，验证 pipeline 通畅（format_reward 是否正确计算）
  EXTRA_ARGS=(
    "env.train.total_num_envs=4"
    "env.eval.total_num_envs=2"
    "env.train.max_episode_steps=5"
    "env.eval.max_episode_steps=5"
    "env.train.max_steps_per_rollout_epoch=4"
    "env.eval.max_steps_per_rollout_epoch=4"
    "algorithm.group_size=2"
    "algorithm.rollout_epoch=1"
    "actor.micro_batch_size=2"
    "actor.global_batch_size=4"
    "actor.model.max_new_tokens=32"
    "actor.model.action_dim=32"
    "runner.max_epochs=2"
    "runner.val_check_interval=1"
    "runner.save_interval=999"
  )
  ;;
medium)
  # 调试 reward 信号：去掉格式分、拉高温度，看 geo_progress 是否有真实波动
  # format_reward_coef=0：模型已能输出合法 JSON，格式分只会淹没导航信号
  # temperature=1.2：组内 rollout 多样性↑，GRPO advantage 区分度↑
  # max_new_tokens=128：JSON 不截断（64 时 bbox 字段被截）
  if [ "$NUM_GPUS" -ge 4 ]; then
    TRAIN_ENVS=12; EVAL_ENVS=6; GLOBAL_BS=24
  else
    TRAIN_ENVS=6; EVAL_ENVS=3; GLOBAL_BS=12
  fi
  EXTRA_ARGS=(
    "env.train.total_num_envs=${TRAIN_ENVS}"
    "env.eval.total_num_envs=${EVAL_ENVS}"
    "env.train.max_episode_steps=40"
    "env.eval.max_episode_steps=40"
    "env.train.max_steps_per_rollout_epoch=40"
    "env.eval.max_steps_per_rollout_epoch=40"
    "++env.train.format_reward_coef=0.0"   # 关掉格式分，只看 geo_progress
    "++env.eval.format_reward_coef=0.0"
    "algorithm.group_size=3"
    "algorithm.rollout_epoch=1"
    "actor.model.is_lora=true"
    "actor.model.lora_alpha=64"
    "actor.model.chunk_size=16"
    "actor.micro_batch_size=1"
    "actor.global_batch_size=${GLOBAL_BS}"
    "actor.model.max_new_tokens=128"
    "actor.model.action_dim=128"
    "actor.model.temperature=1.2"          # 略高温度，组内多样性↑
    "algorithm.sampling_params.temperature_train=1.2"
    "actor.optim.lr=5e-7"                  # 低 lr，减少 approx_kl 震荡
    "runner.max_epochs=10"
    "runner.val_check_interval=1"
    "runner.save_interval=999"
  )
  ;;
lora)
  # LoRA 微调：稳定版 GRPO 配置
  # rollout_epoch=1：无 dormant 数据
  # group_size=3：48 envs / 3 = 16 groups; 48%3=0 ✓
  # global_batch_size=96：48×40/96=20 次 gradient step
  # max_new_tokens=128：JSON action 完整输出（64 时截断 bbox 字段）
  # lora_alpha=64：LoRA scaling=1.0
  EXTRA_ARGS=(
    "actor.model.is_lora=true"
    "actor.micro_batch_size=1"          # gradient_checkpointing=False 后 activations 大，降 batch
    "actor.global_batch_size=96"
    "actor.optim.lr=1e-6"
	    "actor.model.chunk_size=16"
	    "actor.model.lora_alpha=64"
	    "algorithm.kl_beta=0.0"
    "algorithm.group_size=3"
    "algorithm.rollout_epoch=1"
    "env.train.total_num_envs=12"       # 取决于场景对应的回合数
    "env.eval.total_num_envs=6"
    "env.train.max_episode_steps=40" # 80
    "env.eval.max_episode_steps=40"  # 80
    "env.train.max_steps_per_rollout_epoch=40"  # 80
    "env.eval.max_steps_per_rollout_epoch=40"   # 80
    "actor.model.max_new_tokens=128"         # 模型实际 JSON 动作token数 100
    "actor.model.action_dim=128"             # 100
    "runner.max_epochs=20"
    "runner.val_check_interval=2"
    "runner.save_interval=4"
  )
  ;;
full)
  # 完整训练配置（全参微调）
  # 4 卡：env 独占，total_num_envs=20；3 卡：使用 yaml 默认 total_num_envs=12
  if [ "$NUM_GPUS" -ge 4 ]; then
    EXTRA_ARGS=(
      "env.train.total_num_envs=20"
      "env.eval.total_num_envs=8"
    )
  else
    EXTRA_ARGS=()
  fi
  ;;
	*)
	  echo "未知 mode: $MODE（支持 smoke / medium / lora / full）"
	  exit 1
	  ;;
	esac

# Resume checkpoint
if [ -n "$RESUME_DIR" ]; then
  EXTRA_ARGS+=("runner.resume_dir=$RESUME_DIR")
fi

# ─── 启动 ─────────────────────────────────────────────────────────────────────
cd "$EMBODIED_PATH"

CUDA_VISIBLE_DEVICES="$GPUS" \
TORCHDYNAMO_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
EMBODIED_PATH="$EMBODIED_PATH" \
  "$PYTHON" train_embodied_agent.py \
    --config-name genark_grpo_qwen \
    "cluster.component_placement.actor.placement=${ACTOR_GPUS}" \
    "cluster.component_placement.rollout.placement=${ROLLOUT_GPU}" \
    "cluster.component_placement.env.placement=${ENV_GPU}" \
    "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

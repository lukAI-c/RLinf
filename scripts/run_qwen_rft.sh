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
# GPU 分配（支持 3/4/6/7/8 卡）：
#   GPUS=4,5,6,7,0,1,2,3 SCENE_OFFSET=7 ./scripts/run_qwen_rft.sh lora  # 8 卡：quad-scene + 2-rank rollout
#     → actor=4-5(2-rank), rollout=6-7(2-rank), env=0-3(4 workers)
#   GPUS=4,5,6,0,1,2,3 SCENE_OFFSET=7 ./scripts/run_qwen_rft.sh lora    # 7 卡：quad-scene，GPU7 idle
#     → actor=4-5(2-rank), rollout=6, env=0-3(4 workers)
#   GPUS=4,5,6,0,1,7 SCENE_OFFSET=3 ./scripts/run_qwen_rft.sh lora       # 6 卡：dual-scene
#     → actor=4-5(2-rank), rollout=6, env=0-1(dual), GPU7 idle
#     ⚠️  注意：GPUS 顺序必须保证 [0-1]=actor(连续), [2]=rollout或rollout起始, env(连续)
#         env 区间 end>=start，否则 assert end_rank>=start_rank 崩溃
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
SCENE_OFFSET=${SCENE_OFFSET:-0}   # env.train/eval.scene_offset — 控制 worker→scene 分配
GENESIS_BACKEND=${GENESIS_BACKEND:-local}   # "local" | "remote" (Ray direct) | "zmq" (Ray + ZMQ IPC server)

# GPU 分配
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_GPUS=${#GPU_ARR[@]}
if [ "$NUM_GPUS" -ge 8 ]; then
  # 8 卡 quad-scene + 2-rank rollout：actor=2-rank, rollout=2-rank, env=4 workers
  # rollout_ws=2 加速：48 envs → 每 rank 处理 24 envs → 2 chunks/decision（并行）→ ~1.5x 加速
  ACTOR_GPUS="${GPU_ARR[0]}-${GPU_ARR[1]}"   # 前两张给 actor (FSDP 2-rank)
  ROLLOUT_GPU="${GPU_ARR[2]}-${GPU_ARR[3]}"  # 第三/四张给 rollout (2-rank)
  ENV_GPU="${GPU_ARR[4]}-${GPU_ARR[7]}"       # 第五至第八张给 env（4 worker，4 scene）
  ENV_INFO="quad(${ENV_GPU})"
  ACTOR_RANK_INFO="FSDP 2-rank"
elif [ "$NUM_GPUS" -ge 7 ]; then
  # 7 卡 quad-scene：2-rank actor + 1 rollout + 4 env workers（第8张闲置）
  ACTOR_GPUS="${GPU_ARR[0]}-${GPU_ARR[1]}"   # 7 卡：前两张给 actor (FSDP 2-rank)
  ROLLOUT_GPU="${GPU_ARR[2]}"                 # 第三张给 rollout
  ENV_GPU="${GPU_ARR[3]}-${GPU_ARR[6]}"       # 第四至第七张给 env（4 worker，4 scene）
  ENV_INFO="quad(${ENV_GPU})"
  ACTOR_RANK_INFO="FSDP 2-rank"
elif [ "$NUM_GPUS" -ge 6 ]; then
  # 6 卡 dual-scene：2-rank actor + 1 rollout + 2 env workers（第6张闲置）
  # 注意：3-rank actor + 2 env workers 的 compute_split_num(2,3)=2 与 compute_split_num(3,2)=3
  # 不对称导致 ActorGroup rank-1 UnpicklingError；改回 2-rank 确保 lcm(2,2)//2=1 对称。
  ACTOR_GPUS="${GPU_ARR[0]}-${GPU_ARR[1]}"   # 6 卡：前两张给 actor (FSDP 2-rank)
  ROLLOUT_GPU="${GPU_ARR[2]}"                 # 第三张给 rollout
  ENV_GPU="${GPU_ARR[3]}-${GPU_ARR[4]}"       # 第四/五张给 env（双 worker，双 scene）
  ENV_INFO="dual(${ENV_GPU}) [GPU${GPU_ARR[5]} idle]"
  ACTOR_RANK_INFO="FSDP 2-rank"
elif [ "$NUM_GPUS" -ge 5 ]; then
  ACTOR_GPUS="${GPU_ARR[0]}-${GPU_ARR[2]}"   # 5 卡：前三张给 actor (FSDP 3-rank)
  ROLLOUT_GPU="${GPU_ARR[3]}"                 # 第四张给 rollout
  ENV_GPU="${GPU_ARR[4]}"                     # 第五张给 env（独占）
  ENV_INFO="dedicated(${ENV_GPU})"
  ACTOR_RANK_INFO="FSDP 3-rank"
elif [ "$NUM_GPUS" -ge 3 ]; then
  ACTOR_GPUS="${GPU_ARR[0]}-${GPU_ARR[1]}"   # 3/4 卡：前两张给 actor (FSDP 2-rank)
  ROLLOUT_GPU="${GPU_ARR[2]}"                 # 第三张给 rollout
  if [ "$NUM_GPUS" -ge 4 ]; then
    ENV_GPU="${GPU_ARR[3]}"                   # 4 卡：env 独占第四张
    ENV_INFO="dedicated(${ENV_GPU})"
  else
    ENV_GPU="$ROLLOUT_GPU"                    # 3 卡：env 和 rollout 共享
    ENV_INFO="shared with rollout(${ROLLOUT_GPU})"
  fi
  ACTOR_RANK_INFO="FSDP 2-rank"
else
  ACTOR_GPUS="${GPU_ARR[0]}"                  # 2 卡：actor 单卡 (FSDP 1-rank)
  ROLLOUT_GPU="${GPU_ARR[1]}"                 # 第二张给 rollout
  ENV_GPU="$ROLLOUT_GPU"                      # 2 卡：env 和 rollout 共享
  ENV_INFO="shared with rollout(${ROLLOUT_GPU})"
  ACTOR_RANK_INFO="FSDP 1-rank"
fi

LOG_DIR=/home/clk/workspace/results/genark_grpo_qwen
LOG_FILE="$LOG_DIR/rft_$(date +%Y%m%d_%H%M%S)_${MODE}.log"
mkdir -p "$LOG_DIR"

echo "Mode      : $MODE"
echo "GPUs      : $GPUS"
echo "  Actor   : ${ACTOR_GPUS} (${ACTOR_RANK_INFO})"
echo "  Rollout : ${ROLLOUT_GPU}"
echo "  Env     : ${ENV_INFO}"
echo "Log       : $LOG_FILE"
echo "Resume    : ${RESUME_DIR:-none}"
echo "SceneOff  : ${SCENE_OFFSET}"
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
  # group_size=3：envs / 3 必须整除
  # 6 卡（双 scene）：total_num_envs=24（2 worker × 12 envs），global_batch_size=96
  # 4 卡（单 scene）：total_num_envs=12（1 worker × 12 envs），global_batch_size=48
  # max_new_tokens=256：含 progress_analysis+reasoning 需要更多 token
  # lora_alpha=64：LoRA scaling=1.0
  if [ "$NUM_GPUS" -ge 7 ]; then
    TRAIN_ENVS=48; EVAL_ENVS=24; GLOBAL_BS=192
    # 四 scene：4 env workers × 12 envs，total=48; 48%(4×3)=0 ✓; 192%(rollout_ws×2)=0 ✓
  elif [ "$NUM_GPUS" -ge 6 ]; then
    TRAIN_ENVS=24; EVAL_ENVS=12; GLOBAL_BS=96
    # 双 scene：scene_offset 通过 placement 分配，env worker 0 用 scene 0，worker 1 用 scene 1
  else
    TRAIN_ENVS=12; EVAL_ENVS=6; GLOBAL_BS=48
  fi
  EXTRA_ARGS=(
    "actor.model.is_lora=true"
    "actor.micro_batch_size=1"          # mbs=2 OOM on 4-card (max_new_tokens=256 → longer seqs)
    "actor.global_batch_size=${GLOBAL_BS}"
    "actor.optim.lr=1e-7"              # 1e-6→1e-7：grad_norm=154/clipped=1.0 说明需要降一个数量级
    "actor.model.chunk_size=16"
    "actor.model.lora_alpha=64"
    "actor.model.lora_dropout=0.0"
    "algorithm.kl_beta=0.0"
    "algorithm.group_size=3"
    "algorithm.rollout_epoch=1"
    "env.train.total_num_envs=${TRAIN_ENVS}"
    "env.eval.total_num_envs=${EVAL_ENVS}"
    "env.train.max_episode_steps=140"
    "env.eval.max_episode_steps=140"
    "env.train.max_decisions_per_rollout_epoch=20"
    "env.train.max_steps_per_rollout_epoch=200"
    "env.eval.max_steps_per_rollout_epoch=140"
    "env.train.reward_mode=geo_ndtw"      # geo_progress + decision-level nDTW + SR
    # 切换到 decision_nav 模式（推荐下一步尝试）：
    # "env.train.reward_mode=decision_nav"  # decision-level DTG + nDTW + SR，无 step-level geo
    # "env.train.decision_dtg_coef=1.0"
    # "env.train.decision_dtg_clip=2.0"
    # "env.train.geo_coef=0.0"
    "env.train.geo_coef=0.0"             # 关闭 step-level geo_progress（宏动作偏差+early-stop下无效）
    "env.train.ndtw_coef=1.0"            # decision-level nDTW 权重
    "env.train.sr_coef=2.0"             # 10.0→2.0：与 nDTW/geo 同量级，降低 advantage 方差
    "++env.train.wrong_stop_penalty=-2.0" # -0.5→-2.0：加强 early-stop 惩罚，迫使模型多探索
    "env.train.format_reward_coef=0.0"   # 已关闭：模型已会输出合法JSON，format分掩盖导航信号
    "env.train.success_distance=5.0"     # P0 实验：放宽成功阈值 3→5m，让模型有正样本可拿
    "env.eval.success_distance=5.0"      # eval 同步放宽，保持口径一致
    "actor.model.max_new_tokens=256"
    "actor.model.action_dim=256"
    "actor.model.history_max_frames=4"   # 减少视觉 token：8→4，降低 OOM 风险
    "env.train.scene_offset=${SCENE_OFFSET}"
    "env.eval.scene_offset=${SCENE_OFFSET}"
    "++env.train.genesis_backend=${GENESIS_BACKEND}"
    "++env.eval.genesis_backend=${GENESIS_BACKEND}"
    "runner.max_epochs=500"
    "runner.val_check_interval=10"
    "runner.save_interval=20"
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
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
NCCL_TIMEOUT_MS=3600000 \
EMBODIED_PATH="$EMBODIED_PATH" \
  "$PYTHON" train_embodied_agent.py \
    --config-name genark_grpo_qwen \
    "cluster.component_placement.actor.placement=${ACTOR_GPUS}" \
    "cluster.component_placement.rollout.placement=${ROLLOUT_GPU}" \
    "cluster.component_placement.env.placement=${ENV_GPU}" \
    "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

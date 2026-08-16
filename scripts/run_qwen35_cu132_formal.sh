#!/usr/bin/env bash
#
# 正式实验：qwen3.6-27B → Qwen3.5-4B/9B 蒸馏（cu132 全栈，V0 trainer 与旧 baseline 同架构）
#   torch 2.13.0+cu132 / vLLM 0.27.1 / flashinfer 0.6.16.post3 / verl v0.9.0 (V0)
#   baseline 对照：va-opd-qwen35-cu128（torch 2.11+cu129 / vLLM 0.23 / verl 334d9f8b V0）
#   原则：一次一个变量。默认走「跑稳」配置，改一个变量用对应 env 覆盖。
#
# 常用变量开关（与 scripts/run_qwen35_formal.sh 对齐）：
#   STUDENT=Qwen3.5-9B / TEACHER=qwen3.6-35B-A3B / LOSS_MODE=k3|forward_kl_topk
#   USE_TASK_REWARDS=True / FORMAL_GPUS=0,1,2,3 / TRAIN_BATCH_SIZE=56|112
#   ROLLOUT_N / TOTAL_TRAINING_STEPS / VAL_BEFORE_TRAIN / DRY_RUN=1
#   ATTENTION_IMPL=sdpa|flash_attention_2（AB 对照必须与 baseline 侧相同）
#   USE_V1=1：切换 verl V1 trainer（注意：V1 强制 transfer_queue，本共享节点会卡死，见 docs）
#
# 用法（GPU 节点）：
#   bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/run_qwen35_cu132_formal.sh
set -euo pipefail

DTOPD_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
PROJECT_ROOT="${PROJECT_ROOT:-${DTOPD_ROOT}/projects/Dual-Track-OPD}"
BACKEND_ROOT="${BACKEND_ROOT:-${DTOPD_ROOT}/fc-opd-storage/backends/verl-qwen35-v090-cu132}"
BACKEND_RUN_DIR="${BACKEND_RUN_DIR:-${BACKEND_ROOT}/examples/on_policy_distillation_trainer}"
ENV_PREFIX="${ENV_PREFIX:-${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1}"
CUDA_HOME="${CUDA_HOME:-${DTOPD_ROOT}/envs/cuda132-toolchain}"
CONDA_SH="${CONDA_SH:-${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh}"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${ENV_PREFIX}/lib/python3.12/site-packages/torch/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="${DTOPD_ROOT}/.cache/flashinfer"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_LOGGING_STREAM=ext://sys.stderr
export RAY_LOG_TO_STDERR=1 RAY_DEDUP_LOGS=0
# 本实例 NVLS multicast 不可用（CUDA error 401），默认关闭；如需开启 NCCL_NVLS_ENABLE=1
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
ulimit -c 0

# shellcheck disable=SC1091
source "${CONDA_SH}"
conda activate "${ENV_PREFIX}"
ray stop --force 2>/dev/null || true
sleep 2

STUDENT_MODEL="${STUDENT_MODEL:-${DTOPD_ROOT}/models/Qwen3.5-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-${DTOPD_ROOT}/models/qwen3.6-27B}"
TRAIN_FILE="${TRAIN_FILE:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/geometry3k_gkd/train_text_only.parquet}"
VAL_FILE="${VAL_FILE:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/geometry3k_gkd/val_text_only.parquet}"

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-7}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
TEACHER_TP=${TEACHER_TP:-1}
TEACHER_EP=${TEACHER_EP:-1}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.55}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
ROLLOUT_NUM_WORKERS=${ROLLOUT_NUM_WORKERS:-8}
FORMAL_GPUS="${FORMAL_GPUS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${FORMAL_GPUS}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-56}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-5}
ACTOR_LR=${ACTOR_LR:-1e-6}

DISTILLATION_LOSS_MODE=${LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
USE_TASK_REWARDS=${USE_TASK_REWARDS:-False}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-64}
ATTENTION_IMPL=${ATTENTION_IMPL:-flash_attention_2}
PROJECT_NAME=${PROJECT_NAME:-verl_distill_qwen35}
RESUME_MODE=${RESUME_MODE:-disable}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
ROLLOUT_N=${ROLLOUT_N:-1}
USE_V1=${USE_V1:-0}
CKPT_ROOT=${CKPT_ROOT:-${BACKEND_RUN_DIR}/checkpoints}
RUN_METADATA_DIR="${RUN_METADATA_DIR:-${DTOPD_ROOT}/fc-opd-storage/logs/qwen35_runs/${PROJECT_NAME}}"
DRY_RUN=${DRY_RUN:-0}

normalize_bool() {
  case "$1" in
    1|true|True|TRUE) echo True ;;
    0|false|False|FALSE) echo False ;;
    *) echo "FATAL: $2 必须为 True/False/1/0（当前值: '$1'）" >&2; return 1 ;;
  esac
}
USE_POLICY_GRADIENT=$(normalize_bool "${USE_POLICY_GRADIENT}" USE_POLICY_GRADIENT)
USE_TASK_REWARDS=$(normalize_bool "${USE_TASK_REWARDS}" USE_TASK_REWARDS)
VAL_BEFORE_TRAIN=$(normalize_bool "${VAL_BEFORE_TRAIN}" VAL_BEFORE_TRAIN)
if [[ "${USE_V1}" == "1" ]]; then USE_V1_FLAG=True; else USE_V1_FLAG=False; fi

for v in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE NGPUS_PER_NODE ROLLOUT_NUM_WORKERS MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH TOTAL_EPOCHS; do
  val="${!v}"
  case "${val}" in
    ''|*[!0-9]*) echo "FATAL: ${v} 必须为正整数（当前值: '${val}'）"; exit 1 ;;
  esac
  [ "${val}" -gt 0 ] || { echo "FATAL: ${v} 必须大于 0"; exit 1; }
done
[ $(( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE )) -eq 0 ] || { echo "FATAL: TRAIN_BATCH_SIZE 必须被 PPO_MINI_BATCH_SIZE 整除"; exit 1; }
[ $(( TRAIN_BATCH_SIZE * ROLLOUT_N % NGPUS_PER_NODE )) -eq 0 ] || { echo "FATAL: TRAIN_BATCH_SIZE*ROLLOUT_N 必须被 NGPUS_PER_NODE 整除"; exit 1; }
[ $(( TRAIN_BATCH_SIZE * ROLLOUT_N % ROLLOUT_NUM_WORKERS )) -eq 0 ] || { echo "FATAL: TRAIN_BATCH_SIZE*ROLLOUT_N 必须被 ROLLOUT_NUM_WORKERS 整除"; exit 1; }

TEACHER_BASE=$(basename "${TEACHER_MODEL}" | tr 'A-Z.' 'a-z_' | tr '-' '_')
STUDENT_BASE=$(basename "${STUDENT_MODEL}" | tr 'A-Z.' 'a-z_' | tr '-' '_')
TASK_TAG=$(echo "${USE_TASK_REWARDS}" | tr 'A-Z' 'a-z')
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${TEACHER_BASE}_to_${STUDENT_BASE}_${DISTILLATION_LOSS_MODE}_task${TASK_TAG}_n${ROLLOUT_N}_v0_cu132}"
HYDRA_RUN_DIR="${RUN_METADATA_DIR}/${EXPERIMENT_NAME}/hydra"

MAX_NUM_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))

echo "== stack: torch 2.13.0+cu132 / vllm 0.27.1 / verl v0.9.0 (V0) / NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE} =="
echo "== student=${STUDENT_MODEL} =="
echo "== teacher=${TEACHER_MODEL} (TP=${TEACHER_TP}) =="
echo "== GPUs=${FORMAL_GPUS} (n=${NGPUS_PER_NODE}+${TEACHER_WORLD_SIZE}) batch=${TRAIN_BATCH_SIZE} =="
echo "== loss=${DISTILLATION_LOSS_MODE} use_task_rewards=${USE_TASK_REWARDS} use_v1=${USE_V1_FLAG} =="
echo "== experiment=${PROJECT_NAME}/${EXPERIMENT_NAME} =="
echo "== checkpoint_dir=${CKPT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/ =="
echo "== run_metadata_dir=${RUN_METADATA_DIR}/${EXPERIMENT_NAME}/ =="

CKPT_DIR="${CKPT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
if [ "${RESUME_MODE}" = "disable" ] && [ -d "${CKPT_DIR}" ] && [ -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ]; then
  echo "FATAL: checkpoint 目录已有内容且 RESUME_MODE=disable（冷启动 guard）：${CKPT_DIR}" >&2
  echo "      换 EXPERIMENT_NAME，或 RESUME_MODE=auto 续跑，或确认后手动清理。" >&2
  exit 1
fi

EXTRA_ARGS=(+actor_rollout_ref.model.override_config.attn_implementation=${ATTENTION_IMPL})
EXTRA_ARGS+=(trainer.use_v1="${USE_V1_FLAG}")
EXTRA_ARGS+=(trainer.resume_mode="${RESUME_MODE}")
EXTRA_ARGS+=(trainer.val_before_train="${VAL_BEFORE_TRAIN}")
EXTRA_ARGS+=(actor_rollout_ref.rollout.n="${ROLLOUT_N}")
EXTRA_ARGS+=(hydra.run.dir="${HYDRA_RUN_DIR}")
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then EXTRA_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"); fi

cd "${BACKEND_RUN_DIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "== DRY_RUN: 组合命令如下（不执行）=="
  echo "STUDENT_MODEL=${STUDENT_MODEL} TEACHER_MODEL=${TEACHER_MODEL} NNODES=${NNODES} \\
  NGPUS_PER_NODE=${NGPUS_PER_NODE} TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE} TEACHER_TP=${TEACHER_TP} TEACHER_EP=${TEACHER_EP} \\
  ROLLOUT_TP=${ROLLOUT_TP} ROLLOUT_NUM_WORKERS=${ROLLOUT_NUM_WORKERS} TRAIN_FILE=${TRAIN_FILE} VAL_FILE=${VAL_FILE} \\
  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH} \\
  TOTAL_EPOCHS=${TOTAL_EPOCHS} SAVE_FREQ=${SAVE_FREQ} TEST_FREQ=${TEST_FREQ} ACTOR_LR=${ACTOR_LR} \\
  DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE} USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT} USE_TASK_REWARDS=${USE_TASK_REWARDS} \\
  DISTILLATION_TOPK=${DISTILLATION_TOPK} PROJECT_NAME=${PROJECT_NAME} EXPERIMENT_NAME=${EXPERIMENT_NAME} \\
  bash run_qwen3_5_4b_fsdp.sh ${EXTRA_ARGS[*]}"
  exit 0
fi

STUDENT_MODEL="${STUDENT_MODEL}" \
TEACHER_MODEL="${TEACHER_MODEL}" \
NNODES="${NNODES}" \
NGPUS_PER_NODE="${NGPUS_PER_NODE}" \
TEACHER_WORLD_SIZE="${TEACHER_WORLD_SIZE}" \
TEACHER_TP="${TEACHER_TP}" \
TEACHER_EP="${TEACHER_EP}" \
TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL}" \
ROLLOUT_TP="${ROLLOUT_TP}" \
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL}" \
ROLLOUT_NUM_WORKERS="${ROLLOUT_NUM_WORKERS}" \
TRAIN_FILE="${TRAIN_FILE}" \
VAL_FILE="${VAL_FILE}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
SAVE_FREQ="${SAVE_FREQ}" \
TEST_FREQ="${TEST_FREQ}" \
ACTOR_LR="${ACTOR_LR}" \
DISTILLATION_LOSS_MODE="${DISTILLATION_LOSS_MODE}" \
USE_POLICY_GRADIENT="${USE_POLICY_GRADIENT}" \
USE_TASK_REWARDS="${USE_TASK_REWARDS}" \
DISTILLATION_TOPK="${DISTILLATION_TOPK}" \
PROJECT_NAME="${PROJECT_NAME}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
bash run_qwen3_5_4b_fsdp.sh "${EXTRA_ARGS[@]}"

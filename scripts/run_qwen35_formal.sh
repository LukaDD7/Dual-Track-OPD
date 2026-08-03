#!/usr/bin/env bash
#
# 正式实验：qwen3.6-27B → Qwen3.5-4B/9B 蒸馏（cu129 全栈，8 卡：7 actor + 1 teacher）
# 原则：一次一个变量。默认走「跑稳」配置，改一个变量用对应 env 覆盖即可。
#
# 常用变量开关：
#   STUDENT=Qwen3.5-9B                 切换学生（默认 Qwen3.5-4B）
#   TEACHER=qwen3.6-35B-A3B            切换老师（默认 qwen3.6-27B；MoE 35B 需另配 TP/卡数，见下）
#   LOSS_MODE=k3|forward_kl_topk       切换蒸馏 loss（默认 k1）
#   USE_TASK_REWARDS=True              开启任务奖励项（默认 False，先跑稳）
#   USE_FCOP_DATASET=1                 用 FCOPDDataset 注入 "\boxed{} 输出指令" prompt
#                                      （默认 0 用 RLHFDataset = 实验1 原始 prompt；奖励可达性见 handoff）
#   VALIDATION_DATA_DIR=<dir>          每个 test_freq 步把 val 生成/得分 dump 到该目录（默认不 dump）
#   RESUME_MODE=disable|auto           disable=冷启动（默认；防止实验名复用续跑旧 checkpoint）
#   VAL_BEFORE_TRAIN=True              训练前先做一次验证并 dump（默认 False）
#   TOTAL_TRAINING_STEPS=20            固定总训练步数（默认空 = 按 TOTAL_EPOCHS 推导）
#   ROLLOUT_N=1                        每个 prompt 的 rollout 样本数（默认 1，与后端脚本一致）
#   DRY_RUN=1                          只打印组合后的命令与有效配置，不碰 GPU（无卡测试用）
#   ALLOW_EXISTING_RUN_DIR=1           resume=disable 时目标 checkpoint 目录已有内容也放行
#   TOTAL_EPOCHS=2                     多跑几个 epoch（默认 1）
#   TRAIN_BATCH_SIZE=112               放大 batch（默认 56；必须同时被 7 和 8 整除）
#   CLEAN_START=1                      启动前 ray stop --force（默认 0；重复跑失败时建议开启）
#   FORMAL_GPUS=0,1,2,3                显存白名单（默认全部 8 卡；节点上另有实验占用 4-7 时务必设为空闲卡）
#   4 卡跑正式实验示例（3 actor + 1 teacher，batch 需被 3 和 8 整除）：
#     FORMAL_GPUS=0,1,2,3 NGPUS_PER_NODE=3 TRAIN_BATCH_SIZE=24 PPO_MINI_BATCH_SIZE=24 \
#       bash scripts/run_qwen35_formal.sh
#
# MoE 35B teacher（qwen3.6-35B-A3B）单卡 TP=1 显存不够：
#   需要 TEACHER_WORLD_SIZE=2 TEACHER_TP=2 NGPUS_PER_NODE=6（6 actor + 2 teacher），
#   此时 TRAIN_BATCH_SIZE 必须被 6 和 ROLLOUT_NUM_WORKERS 整除（如 48/72/96/144）。
#
# 用法（GPU 节点）：
#   bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/run_qwen35_formal.sh
set -euo pipefail

DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
ENV_PREFIX="${DTOPD_ROOT}/envs/va-opd-qwen35-cu128"
CUDA_HOME="${DTOPD_ROOT}/envs/cuda128-toolchain"
SCRIPTS="${DTOPD_ROOT}/projects/Dual-Track-OPD/scripts"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="${DTOPD_ROOT}/.cache/flashinfer"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# vLLM 默认把日志写 stdout（Ray 分流到 worker-*.out）；统一到 stderr 便于诊断捕获
export VLLM_LOGGING_STREAM=ext://sys.stderr
ulimit -c 0

# shellcheck disable=SC1091
source "${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

# ---- 参数（env 可覆盖）----
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
PROJECT_NAME=${PROJECT_NAME:-verl_distill_qwen35}
USE_FCOP_DATASET=${USE_FCOP_DATASET:-0}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-}
RESUME_MODE=${RESUME_MODE:-disable}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
ROLLOUT_N=${ROLLOUT_N:-1}
CKPT_ROOT=${CKPT_ROOT:-${DTOPD_ROOT}/repos/verl-cu130-vllm/examples/on_policy_distillation_trainer/checkpoints}
ALLOW_EXISTING_RUN_DIR=${ALLOW_EXISTING_RUN_DIR:-0}
DRY_RUN=${DRY_RUN:-0}

TEACHER_BASE=$(basename "${TEACHER_MODEL}" | tr 'A-Z.' 'a-z_' | tr '-' '_')
STUDENT_BASE=$(basename "${STUDENT_MODEL}" | tr 'A-Z.' 'a-z_' | tr '-' '_')
DATASET_TAG=$([ "${USE_FCOP_DATASET}" = "1" ] && echo fcop || echo raw)
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${TEACHER_BASE}_to_${STUDENT_BASE}_${DISTILLATION_LOSS_MODE}_task$(echo ${USE_TASK_REWARDS} | tr 'A-Z' 'a-z')_${DATASET_TAG}_n${ROLLOUT_N}}"

MAX_NUM_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))

# ---- 数值参数校验（防多个 env 被挤成一坨导致静默坏配置，如 TRAIN_BATCH_SIZE=24PPO_MINI_BATCH_SIZE=24）----
for v in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE NGPUS_PER_NODE ROLLOUT_NUM_WORKERS ROLLOUT_N SAVE_FREQ TEST_FREQ; do
  val="${!v}"
  case "${val}" in
    ''|*[!0-9-]*)
      echo "FATAL: ${v} 必须为整数（当前值: '${val}'）。检查各 env 之间是否有空格。"
      exit 1 ;;
  esac
done
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
  case "${TOTAL_TRAINING_STEPS}" in
    *[!0-9]*)
      echo "FATAL: TOTAL_TRAINING_STEPS 必须为非负整数（当前值: '${TOTAL_TRAINING_STEPS}'）。"
      exit 1 ;;
  esac
fi

# ---- 校验 ----
for p in \
  "${STUDENT_MODEL}/config.json" \
  "${TEACHER_MODEL}/config.json" \
  "${TRAIN_FILE}" \
  "${VAL_FILE}"; do
  [ -f "${p}" ] || { echo "FATAL: missing ${p}"; exit 1; }
done
[ $(( TRAIN_BATCH_SIZE % NGPUS_PER_NODE )) -eq 0 ] || {
  echo "FATAL: TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} 必须被 NGPUS_PER_NODE=${NGPUS_PER_NODE} 整除"; exit 1; }
[ $(( TRAIN_BATCH_SIZE % ROLLOUT_NUM_WORKERS )) -eq 0 ] || {
  echo "FATAL: TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} 必须被 ROLLOUT_NUM_WORKERS=${ROLLOUT_NUM_WORKERS} 整除"; exit 1; }
case "${TEACHER_MODEL}" in
  *35B-A3B*)
    if [ "${TEACHER_TP}" = "1" ] && [ "${FORCE_TEACHER_TP1:-0}" != "1" ]; then
      echo "FATAL: MoE 35B teacher 建议 TEACHER_WORLD_SIZE=2 TEACHER_TP=2 NGPUS_PER_NODE=6"
      echo "      （batch 改 48/72/96/144）。确要 TP=1 请加 FORCE_TEACHER_TP1=1。"; exit 1
    fi ;;
esac

echo "== student=${STUDENT_MODEL} =="
echo "== teacher=${TEACHER_MODEL} (TP=${TEACHER_TP} mem=${TEACHER_GPU_MEM_UTIL}) =="
echo "== train=${TRAIN_FILE} =="
echo "== val=${VAL_FILE} =="
echo "== batch=${TRAIN_BATCH_SIZE} workers=${ROLLOUT_NUM_WORKERS} n_gpus=${NGPUS_PER_NODE}+${TEACHER_WORLD_SIZE} =="
echo "== loss=${DISTILLATION_LOSS_MODE} use_task_rewards=${USE_TASK_REWARDS} =="
echo "== max_len=${MAX_NUM_TOKENS} (prompt ${MAX_PROMPT_LENGTH} + response ${MAX_RESPONSE_LENGTH}) =="
echo "== experiment=${PROJECT_NAME}/${EXPERIMENT_NAME} =="
echo "== dataset_class=$( [ "${USE_FCOP_DATASET}" = "1" ] && echo FCOPDDataset || echo RLHFDataset ) =="
echo "== resume_mode=${RESUME_MODE} val_before_train=${VAL_BEFORE_TRAIN} rollout_n=${ROLLOUT_N} =="
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then echo "== total_training_steps=${TOTAL_TRAINING_STEPS} =="; fi
if [ -n "${VALIDATION_DATA_DIR}" ]; then echo "== val dump → ${VALIDATION_DATA_DIR} =="; fi
echo "== checkpoint_dir=${CKPT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/ =="

# ---- 参数组合：EXTRA_ARGS（追加在 backend 固定参数之后，Hydra 后覆盖前）----
EXTRA_ARGS=(+actor_rollout_ref.model.override_config.attn_implementation=sdpa)
if [ "${USE_FCOP_DATASET}" = "1" ]; then
  # 让 Ray worker 能 import dual_track_opd（FCOPDDataset 所在包）
  export PYTHONPATH="${DTOPD_ROOT}/projects/Dual-Track-OPD/src:${PYTHONPATH:-}"
  EXTRA_ARGS+=(data.custom_cls.path=pkg://dual_track_opd.fc_opd.verl_dataset)
  EXTRA_ARGS+=(data.custom_cls.name=FCOPDDataset)
fi
if [ -n "${VALIDATION_DATA_DIR}" ]; then
  if [ "${DRY_RUN}" != "1" ]; then
    mkdir -p "${VALIDATION_DATA_DIR}"
  fi
  EXTRA_ARGS+=(trainer.validation_data_dir="${VALIDATION_DATA_DIR}")
fi
EXTRA_ARGS+=(trainer.resume_mode="${RESUME_MODE}")
EXTRA_ARGS+=(trainer.val_before_train="${VAL_BEFORE_TRAIN}")
EXTRA_ARGS+=(actor_rollout_ref.rollout.n="${ROLLOUT_N}")
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
  EXTRA_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

# ---- resume=disable 冷启动 guard：目标 checkpoint 目录已存在则拒绝（除非显式放行）----
CKPT_DIR="${CKPT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
if [ "${RESUME_MODE}" = "disable" ] && [ -d "${CKPT_DIR}" ] && [ -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ] && [ "${ALLOW_EXISTING_RUN_DIR}" != "1" ]; then
  echo "FATAL: resume_mode=disable 但 checkpoint 目录已有内容: ${CKPT_DIR}"
  echo "      这通常是实验名复用（会续跑旧实验）。确认是全新实验名，或显式设 ALLOW_EXISTING_RUN_DIR=1 放行（不会删除旧文件）。"
  exit 1
fi

# ---- DRY_RUN：无卡测试，打印组合后的 backend 启动命令后退出 ----
if [ "${DRY_RUN}" = "1" ]; then
  echo "== [DRY_RUN] composed backend command =="
  echo "cd ${DTOPD_ROOT}/repos/verl-cu130-vllm/examples/on_policy_distillation_trainer"
  echo "bash run_qwen3_5_4b_fsdp.sh ${EXTRA_ARGS[*]} $*"
  exit 0
fi

# ---- 0. GPU 状态检查：残留显存是 vLLM init 失败的常见原因 ----
echo "== [gpu check] =="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
OCCUPIED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 > 4096 {n++} END {print n+0}')
if [ "${OCCUPIED}" -gt 0 ]; then
  echo "WARNING: ${OCCUPIED} 张卡显存占用 >4GiB（可能来自上次运行残留）。"
  echo "        确认无其他任务后，可用 CLEAN_START=1 或手动清理："
  echo "          ray stop --force; pkill -9 -f 'vllm'; pkill -9 -f 'EngineCore'"
fi

# ---- 0.5 可选：清理旧 Ray 集群，避免新运行 join 残留 worker ----
if [ "${CLEAN_START:-0}" = "1" ]; then
  echo "== [clean start] ray stop --force =="
  ray stop --force 2>/dev/null || true
  sleep 3
fi

# ---- teacher 预检（27B 首次加载含 flashinfer JIT，约 3-6 分钟；JIT 缓存后变快）----
echo "== [preflight] teacher engine on GPU 0 =="
cd "${DTOPD_ROOT}"
PREFLIGHT_MODEL="${TEACHER_MODEL}" \
PREFLIGHT_TP="${TEACHER_TP}" \
PREFLIGHT_GPU_MEM="${TEACHER_GPU_MEM_UTIL}" \
PREFLIGHT_MAX_MODEL_LEN="${MAX_NUM_TOKENS}" \
CUDA_VISIBLE_DEVICES=0 \
python3 "${SCRIPTS}/qwen35_vllm_preflight.py"

# ---- verl 启动 ----
cd "${DTOPD_ROOT}/repos/verl-cu130-vllm/examples/on_policy_distillation_trainer"

STUDENT_MODEL="${STUDENT_MODEL}" \
TEACHER_MODEL="${TEACHER_MODEL}" \
TRAIN_FILE="${TRAIN_FILE}" \
VAL_FILE="${VAL_FILE}" \
NNODES="${NNODES}" \
NGPUS_PER_NODE="${NGPUS_PER_NODE}" \
TEACHER_WORLD_SIZE="${TEACHER_WORLD_SIZE}" \
TEACHER_TP="${TEACHER_TP}" \
TEACHER_EP="${TEACHER_EP}" \
TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL}" \
ROLLOUT_TP="${ROLLOUT_TP}" \
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL}" \
ROLLOUT_NUM_WORKERS="${ROLLOUT_NUM_WORKERS}" \
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
bash run_qwen3_5_4b_fsdp.sh "${EXTRA_ARGS[@]}" "$@"

echo "== DONE. checkpoint: ${CKPT_DIR}/ =="

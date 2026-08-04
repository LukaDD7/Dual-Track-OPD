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
#   PROMPT_VERSION=v1|boxed_only|answer_only
#                                      版本化收尾指令消融；answer_only 只用于
#                                      validation-only gate，未经结果批准不作训练口径
#   VALIDATION_DATA_DIR=<dir>          每个 test_freq 步把 val 生成/得分 dump 到该目录（默认不 dump）
#   RESUME_MODE=disable|auto           disable=冷启动（默认；防止实验名复用续跑旧 checkpoint）
#   VAL_BEFORE_TRAIN=True              训练前先做一次验证并 dump（默认 False）
#   TOTAL_TRAINING_STEPS=20            固定总训练步数（默认空 = 按 TOTAL_EPOCHS 推导）
#   ROLLOUT_N=1                        每个 prompt 的 rollout 样本数（默认 1，与后端脚本一致）
#   TRAINER_USE_V1=True                显式使用 v1 trainer（默认 True；实验名含 v1/v0）
#   RUN_METADATA_DIR=<dir>             manifest/Hydra config/train.log 目录（必须在 Git 外）
#   DRY_RUN=1                          只打印组合后的命令与有效配置，不碰 GPU（无卡测试用）
#   ALLOW_EXISTING_RUN_DIR=1           resume=disable 时目标 checkpoint 目录已有内容也放行
#   ALLOW_EXISTING_OUTPUT_DIR=1        validation/metadata 目录非空时显式放行（默认拒绝）
#   TOTAL_EPOCHS=2                     多跑几个 epoch（默认 1）
#   TRAIN_BATCH_SIZE=112               prompt batch（默认 56）；有效序列 batch =
#                                      TRAIN_BATCH_SIZE * ROLLOUT_N
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

DTOPD_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
PROJECT_ROOT="${PROJECT_ROOT:-${DTOPD_ROOT}/projects/Dual-Track-OPD}"
BACKEND_ROOT="${BACKEND_ROOT:-${DTOPD_ROOT}/repos/verl-cu130-vllm}"
BACKEND_RUN_DIR="${BACKEND_RUN_DIR:-${BACKEND_ROOT}/examples/on_policy_distillation_trainer}"
ENV_PREFIX="${ENV_PREFIX:-${DTOPD_ROOT}/envs/va-opd-qwen35-cu128}"
CUDA_HOME="${CUDA_HOME:-${DTOPD_ROOT}/envs/cuda128-toolchain}"
CONDA_SH="${CONDA_SH:-${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh}"
SCRIPTS="${SCRIPTS:-${PROJECT_ROOT}/scripts}"

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
PROMPT_VERSION=${PROMPT_VERSION:-v1}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-}
RESUME_MODE=${RESUME_MODE:-disable}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
ROLLOUT_N=${ROLLOUT_N:-1}
TRAINER_USE_V1=${TRAINER_USE_V1:-True}
CKPT_ROOT=${CKPT_ROOT:-${BACKEND_RUN_DIR}/checkpoints}
ALLOW_EXISTING_RUN_DIR=${ALLOW_EXISTING_RUN_DIR:-0}
ALLOW_EXISTING_OUTPUT_DIR=${ALLOW_EXISTING_OUTPUT_DIR:-0}
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
TRAINER_USE_V1=$(normalize_bool "${TRAINER_USE_V1}" TRAINER_USE_V1)

case "${USE_FCOP_DATASET}" in
  0|1) ;;
  *) echo "FATAL: USE_FCOP_DATASET 必须为 0 或 1（当前值: '${USE_FCOP_DATASET}'）"; exit 1 ;;
esac
case "${PROMPT_VERSION}" in
  v1|boxed_only|answer_only) ;;
  *) echo "FATAL: PROMPT_VERSION 必须为 v1、boxed_only 或 answer_only（当前值: '${PROMPT_VERSION}'）"; exit 1 ;;
esac
if [ "${USE_FCOP_DATASET}" != "1" ] && [ "${PROMPT_VERSION}" != "v1" ]; then
  echo "FATAL: PROMPT_VERSION=${PROMPT_VERSION} 只有在 USE_FCOP_DATASET=1 时才会生效。"
  exit 1
fi

# 这些字段参与实验命名、隔离和 manifest，禁止用尾部 Hydra 参数静默覆盖。
VAL_ONLY_REQUESTED=False
for override in "$@"; do
  case "${override}" in
    trainer.val_only=True|trainer.val_only=true) VAL_ONLY_REQUESTED=True ;;
    trainer.use_v1=*|trainer.resume_mode=*|trainer.val_before_train=*|trainer.total_training_steps=*|\
    trainer.validation_data_dir=*|actor_rollout_ref.rollout.n=*|hydra.run.dir=*|\
    distillation.distillation_loss.use_task_rewards=*|data.prompt_version=*|+data.prompt_version=*)
      echo "FATAL: '${override}' 由 wrapper 管理；请使用对应环境变量，确保实验名与 manifest 一致。"
      exit 1 ;;
  esac
done
if [ "${PROMPT_VERSION}" = "answer_only" ] && [ "${VAL_ONLY_REQUESTED}" != "True" ]; then
  echo "FATAL: answer_only 当前只批准 validation-only gate；请传 trainer.val_only=True。"
  exit 1
fi

TEACHER_BASE=$(basename "${TEACHER_MODEL}" | tr 'A-Z.' 'a-z_' | tr '-' '_')
STUDENT_BASE=$(basename "${STUDENT_MODEL}" | tr 'A-Z.' 'a-z_' | tr '-' '_')
DATASET_TAG=$([ "${USE_FCOP_DATASET}" = "1" ] && echo fcop || echo raw)
PROMPT_TAG=$([ "${USE_FCOP_DATASET}" = "1" ] && [ "${PROMPT_VERSION}" != "v1" ] && echo "_pv${PROMPT_VERSION}" || echo "")
TRAINER_TAG=$([ "${TRAINER_USE_V1}" = "True" ] && echo v1 || echo v0)
TASK_TAG=$(echo "${USE_TASK_REWARDS}" | tr 'A-Z' 'a-z')
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${TEACHER_BASE}_to_${STUDENT_BASE}_${DISTILLATION_LOSS_MODE}_task${TASK_TAG}_${DATASET_TAG}${PROMPT_TAG}_n${ROLLOUT_N}_${TRAINER_TAG}}"
RUN_METADATA_DIR="${RUN_METADATA_DIR:-${DTOPD_ROOT}/fc-opd-storage/logs/qwen35_runs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${RUN_METADATA_DIR}/hydra}"

# ---- 数值参数校验（防多个 env 被挤成一坨导致静默坏配置，如 TRAIN_BATCH_SIZE=24PPO_MINI_BATCH_SIZE=24）----
for v in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE NGPUS_PER_NODE ROLLOUT_NUM_WORKERS ROLLOUT_N MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH PPO_MAX_TOKEN_LEN_PER_GPU TOTAL_EPOCHS; do
  val="${!v}"
  case "${val}" in
    ''|*[!0-9]*)
      echo "FATAL: ${v} 必须为正整数（当前值: '${val}'）。检查各 env 之间是否有空格。"
      exit 1 ;;
  esac
  [ "${val}" -gt 0 ] || { echo "FATAL: ${v} 必须大于 0（当前值: '${val}'）。"; exit 1; }
done
for v in SAVE_FREQ TEST_FREQ; do
  val="${!v}"
  case "${val}" in
    -1|0) ;;
    ''|*[!0-9]*) echo "FATAL: ${v} 必须为 -1、0 或正整数（当前值: '${val}'）。"; exit 1 ;;
    *) [ "${val}" -gt 0 ] || { echo "FATAL: ${v} 必须为 -1、0 或正整数（当前值: '${val}'）。"; exit 1; } ;;
  esac
done
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
  case "${TOTAL_TRAINING_STEPS}" in
    *[!0-9]*)
      echo "FATAL: TOTAL_TRAINING_STEPS 必须为正整数（当前值: '${TOTAL_TRAINING_STEPS}'）。"
      exit 1 ;;
  esac
  [ "${TOTAL_TRAINING_STEPS}" -gt 0 ] || { echo "FATAL: TOTAL_TRAINING_STEPS 必须大于 0。"; exit 1; }
fi

MAX_NUM_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))
EFFECTIVE_SEQUENCE_BATCH=$(( TRAIN_BATCH_SIZE * ROLLOUT_N ))
EFFECTIVE_PPO_MINI_BATCH=$(( PPO_MINI_BATCH_SIZE * ROLLOUT_N ))

[ "${PPO_MINI_BATCH_SIZE}" -le "${TRAIN_BATCH_SIZE}" ] || {
  echo "FATAL: PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} 是 prompt 口径，不能大于 TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"; exit 1; }
[ $(( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE )) -eq 0 ] || {
  echo "FATAL: TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} 必须被 PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} 整除"; exit 1; }
[ $(( EFFECTIVE_SEQUENCE_BATCH % NGPUS_PER_NODE )) -eq 0 ] || {
  echo "FATAL: 有效序列 batch=${EFFECTIVE_SEQUENCE_BATCH} (= TRAIN_BATCH_SIZE ${TRAIN_BATCH_SIZE} * ROLLOUT_N ${ROLLOUT_N}) 必须被 NGPUS_PER_NODE=${NGPUS_PER_NODE} 整除"; exit 1; }
[ $(( EFFECTIVE_SEQUENCE_BATCH % ROLLOUT_NUM_WORKERS )) -eq 0 ] || {
  echo "FATAL: 有效序列 batch=${EFFECTIVE_SEQUENCE_BATCH} (= TRAIN_BATCH_SIZE ${TRAIN_BATCH_SIZE} * ROLLOUT_N ${ROLLOUT_N}) 必须被 ROLLOUT_NUM_WORKERS=${ROLLOUT_NUM_WORKERS} 整除"; exit 1; }
[ $(( EFFECTIVE_PPO_MINI_BATCH % NGPUS_PER_NODE )) -eq 0 ] || {
  echo "FATAL: 有效 PPO mini-batch=${EFFECTIVE_PPO_MINI_BATCH} (= PPO_MINI_BATCH_SIZE ${PPO_MINI_BATCH_SIZE} * ROLLOUT_N ${ROLLOUT_N}) 必须被 NGPUS_PER_NODE=${NGPUS_PER_NODE} 整除"; exit 1; }
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
echo "== prompt_batch=${TRAIN_BATCH_SIZE} rollout_n=${ROLLOUT_N} effective_sequences=${EFFECTIVE_SEQUENCE_BATCH} =="
echo "== ppo_mini_prompt_batch=${PPO_MINI_BATCH_SIZE} effective_ppo_mini=${EFFECTIVE_PPO_MINI_BATCH} workers=${ROLLOUT_NUM_WORKERS} =="
echo "== n_gpus=${NGPUS_PER_NODE}+${TEACHER_WORLD_SIZE} =="
echo "== loss=${DISTILLATION_LOSS_MODE} use_task_rewards=${USE_TASK_REWARDS} =="
echo "== max_len=${MAX_NUM_TOKENS} (prompt ${MAX_PROMPT_LENGTH} + response ${MAX_RESPONSE_LENGTH}) =="
echo "== experiment=${PROJECT_NAME}/${EXPERIMENT_NAME} =="
echo "== dataset_class=$( [ "${USE_FCOP_DATASET}" = "1" ] && echo FCOPDDataset || echo RLHFDataset ) =="
echo "== prompt_version=${PROMPT_VERSION} =="
echo "== trainer=${TRAINER_TAG} resume_mode=${RESUME_MODE} val_before_train=${VAL_BEFORE_TRAIN} rollout_n=${ROLLOUT_N} =="
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then echo "== total_training_steps=${TOTAL_TRAINING_STEPS} =="; fi
if [ -n "${VALIDATION_DATA_DIR}" ]; then echo "== val dump → ${VALIDATION_DATA_DIR} =="; fi
echo "== checkpoint_dir=${CKPT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/ =="
echo "== run_metadata_dir=${RUN_METADATA_DIR} =="

# ---- 参数组合：EXTRA_ARGS（追加在 backend 固定参数之后，Hydra 后覆盖前）----
EXTRA_ARGS=(+actor_rollout_ref.model.override_config.attn_implementation=sdpa)
if [ "${USE_FCOP_DATASET}" = "1" ]; then
  # 让 Ray worker 能 import dual_track_opd（FCOPDDataset 所在包）
  export PYTHONPATH="${DTOPD_ROOT}/projects/Dual-Track-OPD/src:${PYTHONPATH:-}"
  EXTRA_ARGS+=(data.custom_cls.path=pkg://dual_track_opd.fc_opd.verl_dataset)
  EXTRA_ARGS+=(data.custom_cls.name=FCOPDDataset)
  # 新增键必须用 '+' 追加（Hydra struct）：data.custom_cls 已存在于 schema，
  # prompt_version 是新键，直接赋值会被结构化配置拒绝。
  EXTRA_ARGS+=("+data.prompt_version=${PROMPT_VERSION}")
fi
if [ -n "${VALIDATION_DATA_DIR}" ]; then
  EXTRA_ARGS+=(trainer.validation_data_dir="${VALIDATION_DATA_DIR}")
fi
EXTRA_ARGS+=(trainer.resume_mode="${RESUME_MODE}")
EXTRA_ARGS+=(trainer.val_before_train="${VAL_BEFORE_TRAIN}")
EXTRA_ARGS+=(trainer.use_v1="${TRAINER_USE_V1}")
EXTRA_ARGS+=(actor_rollout_ref.rollout.n="${ROLLOUT_N}")
EXTRA_ARGS+=(hydra.run.dir="${HYDRA_RUN_DIR}")
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
  EXTRA_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

# ---- 冷启动 guard：checkpoint、validation 与 metadata 任一非空都拒绝静默复用 ----
CKPT_DIR="${CKPT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
if [ "${RESUME_MODE}" = "disable" ] && [ -d "${CKPT_DIR}" ] && [ -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ] && [ "${ALLOW_EXISTING_RUN_DIR}" != "1" ]; then
  echo "FATAL: resume_mode=disable 但 checkpoint 目录已有内容: ${CKPT_DIR}"
  echo "      这通常是实验名复用（会续跑旧实验）。确认是全新实验名，或显式设 ALLOW_EXISTING_RUN_DIR=1 放行（不会删除旧文件）。"
  exit 1
fi
if [ "${RESUME_MODE}" = "disable" ] && [ -n "${VALIDATION_DATA_DIR}" ] && [ -d "${VALIDATION_DATA_DIR}" ] && [ -n "$(ls -A "${VALIDATION_DATA_DIR}" 2>/dev/null)" ] && [ "${ALLOW_EXISTING_OUTPUT_DIR}" != "1" ]; then
  echo "FATAL: resume_mode=disable 但 validation 目录已有内容: ${VALIDATION_DATA_DIR}"
  echo "      请使用新的目录；确需复用时显式设 ALLOW_EXISTING_OUTPUT_DIR=1（不会删除旧文件）。"
  exit 1
fi
if [ "${RESUME_MODE}" = "disable" ] && [ -d "${RUN_METADATA_DIR}" ] && [ -n "$(ls -A "${RUN_METADATA_DIR}" 2>/dev/null)" ] && [ "${ALLOW_EXISTING_OUTPUT_DIR}" != "1" ]; then
  echo "FATAL: resume_mode=disable 但 metadata 目录已有内容: ${RUN_METADATA_DIR}"
  echo "      请使用新的实验名/目录；确需复用时显式设 ALLOW_EXISTING_OUTPUT_DIR=1。"
  exit 1
fi

# ---- DRY_RUN：无卡测试，打印组合后的 backend 启动命令后退出 ----
if [ "${DRY_RUN}" = "1" ]; then
  echo "== [DRY_RUN] composed backend command =="
  echo "cd ${BACKEND_RUN_DIR}"
  echo "bash run_qwen3_5_4b_fsdp.sh ${EXTRA_ARGS[*]} $*"
  exit 0
fi

# ---- 从这里开始才依赖服务器文件布局/conda/GPU；DRY_RUN 在任意机器均可执行 ----
for p in \
  "${STUDENT_MODEL}/config.json" \
  "${TEACHER_MODEL}/config.json" \
  "${TRAIN_FILE}" \
  "${VAL_FILE}" \
  "${CONDA_SH}" \
  "${SCRIPTS}/qwen35_vllm_preflight.py" \
  "${SCRIPTS}/qwen35_tokenizer_alignment.py" \
  "${SCRIPTS}/qwen35_run_manifest.py" \
  "${BACKEND_RUN_DIR}/run_qwen3_5_4b_fsdp.sh"; do
  [ -f "${p}" ] || { echo "FATAL: missing ${p}"; exit 1; }
done

for output_dir in "${VALIDATION_DATA_DIR}" "${RUN_METADATA_DIR}"; do
  [ -z "${output_dir}" ] && continue
  case "${output_dir}/" in
    "${PROJECT_ROOT}/"*)
      echo "FATAL: raw run output must live outside Git: ${output_dir}"
      exit 1 ;;
  esac
done

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${ENV_PREFIX}"

mkdir -p "${RUN_METADATA_DIR}"
if [ -n "${VALIDATION_DATA_DIR}" ]; then mkdir -p "${VALIDATION_DATA_DIR}"; fi

# The pinned teacher receives student-produced token IDs directly.  A shared
# ID-to-token mapping is therefore a semantic requirement, not just metadata.
python3 "${SCRIPTS}/qwen35_tokenizer_alignment.py" \
  --student-model "${STUDENT_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --output "${RUN_METADATA_DIR}/tokenizer_alignment.json"

LAUNCHER_ARGS_JSON=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@")
MANIFEST_CONFIG=(
  --config "launcher_args_json=${LAUNCHER_ARGS_JSON}"
  --config "project_name=${PROJECT_NAME}"
  --config "experiment_name=${EXPERIMENT_NAME}"
  --config "trainer_use_v1=${TRAINER_USE_V1}"
  --config "resume_mode=${RESUME_MODE}"
  --config "dataset_tag=${DATASET_TAG}"
  --config "prompt_version=${PROMPT_VERSION}"
  --config "use_task_rewards=${USE_TASK_REWARDS}"
  --config "use_policy_gradient=${USE_POLICY_GRADIENT}"
  --config "loss_mode=${DISTILLATION_LOSS_MODE}"
  --config "distillation_topk=${DISTILLATION_TOPK}"
  --config "rollout_n=${ROLLOUT_N}"
  --config "rollout_num_workers=${ROLLOUT_NUM_WORKERS}"
  --config "train_batch_size=${TRAIN_BATCH_SIZE}"
  --config "ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  --config "effective_sequence_batch=${EFFECTIVE_SEQUENCE_BATCH}"
  --config "effective_ppo_mini_batch=${EFFECTIVE_PPO_MINI_BATCH}"
  --config "max_prompt_length=${MAX_PROMPT_LENGTH}"
  --config "max_response_length=${MAX_RESPONSE_LENGTH}"
  --config "ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  --config "total_epochs=${TOTAL_EPOCHS}"
  --config "total_training_steps=${TOTAL_TRAINING_STEPS}"
  --config "save_freq=${SAVE_FREQ}"
  --config "test_freq=${TEST_FREQ}"
  --config "val_before_train=${VAL_BEFORE_TRAIN}"
  --config "actor_lr=${ACTOR_LR}"
  --config "formal_gpus=${FORMAL_GPUS}"
  --config "n_gpus_per_node=${NGPUS_PER_NODE}"
  --config "teacher_world_size=${TEACHER_WORLD_SIZE}"
  --config "teacher_tp=${TEACHER_TP}"
  --config "teacher_ep=${TEACHER_EP}"
  --config "teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL}"
  --config "rollout_tp=${ROLLOUT_TP}"
  --config "rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL}"
  --config "validation_data_dir=${VALIDATION_DATA_DIR}"
  --config "checkpoint_dir=${CKPT_DIR}"
  --config "hydra_run_dir=${HYDRA_RUN_DIR}"
)

python3 "${SCRIPTS}/qwen35_run_manifest.py" start \
  --metadata-dir "${RUN_METADATA_DIR}" \
  --repo-root "${PROJECT_ROOT}" \
  --backend-root "${BACKEND_ROOT}" \
  --env-prefix "${ENV_PREFIX}" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --student-model "${STUDENT_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --launcher "$0" \
  "${MANIFEST_CONFIG[@]}" \
  -- "${EXTRA_ARGS[@]}" "$@"

finalize_run_manifest() {
  rc=$?
  trap - EXIT
  python3 "${SCRIPTS}/qwen35_run_manifest.py" finish \
    --metadata-dir "${RUN_METADATA_DIR}" \
    --returncode "${rc}" || true
  exit "${rc}"
}
trap finalize_run_manifest EXIT

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
cd "${BACKEND_RUN_DIR}"

set +e
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
bash run_qwen3_5_4b_fsdp.sh "${EXTRA_ARGS[@]}" "$@" 2>&1 | tee "${RUN_METADATA_DIR}/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e

if [ "${TRAIN_RC}" -ne 0 ]; then
  echo "FATAL: verl training exited with ${TRAIN_RC}; see ${RUN_METADATA_DIR}/train.log"
  exit "${TRAIN_RC}"
fi

echo "== DONE. checkpoint: ${CKPT_DIR}/ =="

#!/usr/bin/env bash
# Native verl OPD/VA-OPD comparison for Qwen3-VL on Geometry3K.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HPC_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
ENV_PREFIX="${VA_OPD_ENV_PREFIX:-${HPC_ROOT}/fc-opd-storage/envs/va-opd-verl-e003-cu128}"
VERL_DIR="${VERL_VA_OPD_DIR:-${HPC_ROOT}/fc-opd-storage/backends/verl-va-opd-e0031631}"
STUDENT_MODEL="${VA_OPD_STUDENT_MODEL:-${HPC_ROOT}/models/Qwen3-VL-4B-Instruct}"
TEACHER_MODEL="${VA_OPD_TEACHER_MODEL:-${HPC_ROOT}/models/Qwen3-VL-32B-Instruct}"
SOURCE_DATA="${GEOMETRY3K_SOURCE:-${HPC_ROOT}/dataset/geometry3k/data/train-00000-of-00001.parquet}"
DATA_DIR="${GEOMETRY3K_VA_OPD_DATA_DIR:-${HPC_ROOT}/fc-opd-storage/outputs/fc_opd/geometry3k_gkd}"
TRAIN_DATA="${DATA_DIR}/train.parquet"
VAL_DATA="${DATA_DIR}/val.parquet"
CONFIG_REFERENCE="${REPO_ROOT}/configs/experiment/qwen3vl_geometry3k_va_opd_native.yaml"

OBJECTIVE="va_opd"
PROFILE="smoke"
VISIBLE_GPUS="0,1,2,3,4,5"
ACTOR_GPUS=4
TEACHER_GPUS=2
TEACHER_TP=2
PROMPT_BATCH_SIZE=4
ROLLOUT_N=4
TOTAL_STEPS=3
TOTAL_EPOCHS=100
SAVE_FREQ=3
TEST_FREQ=3
MAX_PROMPT_LENGTH=6144
MAX_RESPONSE_LENGTH=2048
LEARNING_RATE=1e-6
ROLLOUT_GPU_MEMORY_UTILIZATION=0.55
TEACHER_GPU_MEMORY_UTILIZATION=0.80
PREPARE_DATA=true
PREFLIGHT_ONLY=false
AUDIT_ALL_IMAGES=false
ALLOW_BUSY_GPUS=false
ALLOW_SYSTEM_NVCC=false
KEEPALIVE_AFTER_SUCCESS=false
NAME=""
STEPS_EXPLICIT=false
BATCH_EXPLICIT=false

usage() {
    cat <<'EOF'
Usage: bash scripts/hpc/run_va_opd_native.sh [options]

  --objective opd|va_opd       Fair native reverse-KL baseline or VA-OPD
  --profile smoke|train        smoke=3 updates/batch 4; train=5 epochs/batch 16
  --visible-gpus LIST          Physical GPU list (default: 0,1,2,3,4,5)
  --actor-gpus N               FSDP/rollout pool size (default: 4)
  --teacher-gpus N             Native teacher pool size (default: 2)
  --teacher-tp N               GPUs per teacher replica (default: 2)
  --student-model PATH
  --teacher-model PATH
  --source-data PATH
  --train-data PATH            Also disables data preparation
  --val-data PATH              Also disables data preparation
  --steps N                    Override optimizer steps; 0 means epoch-driven
  --batch-size N               Prompt batch before K=4 expansion
  --name TAG
  --preflight-only             CPU-safe validation; do not start Ray/GPU work
  --audit-all-images           Validate every prepared full/degraded image pair
  --allow-busy-gpus            Deliberate override after manual PID inspection
  --allow-system-nvcc          Deliberate override; normally /usr nvcc is forbidden
  --keepalive-after-success    Start the project keepalive only after successful cleanup
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --objective) OBJECTIVE="${2:?missing objective}"; shift 2 ;;
        --profile) PROFILE="${2:?missing profile}"; shift 2 ;;
        --visible-gpus) VISIBLE_GPUS="${2:?missing GPU list}"; shift 2 ;;
        --actor-gpus) ACTOR_GPUS="${2:?missing actor GPU count}"; shift 2 ;;
        --teacher-gpus) TEACHER_GPUS="${2:?missing teacher GPU count}"; shift 2 ;;
        --teacher-tp) TEACHER_TP="${2:?missing teacher TP}"; shift 2 ;;
        --student-model) STUDENT_MODEL="${2:?missing student model}"; shift 2 ;;
        --teacher-model) TEACHER_MODEL="${2:?missing teacher model}"; shift 2 ;;
        --source-data) SOURCE_DATA="${2:?missing source data}"; shift 2 ;;
        --train-data) TRAIN_DATA="${2:?missing train data}"; PREPARE_DATA=false; shift 2 ;;
        --val-data) VAL_DATA="${2:?missing val data}"; PREPARE_DATA=false; shift 2 ;;
        --steps) TOTAL_STEPS="${2:?missing step count}"; STEPS_EXPLICIT=true; shift 2 ;;
        --batch-size) PROMPT_BATCH_SIZE="${2:?missing batch size}"; BATCH_EXPLICIT=true; shift 2 ;;
        --name) NAME="_${2:?missing name}"; shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=true; shift ;;
        --audit-all-images) AUDIT_ALL_IMAGES=true; shift ;;
        --allow-busy-gpus) ALLOW_BUSY_GPUS=true; shift ;;
        --allow-system-nvcc) ALLOW_SYSTEM_NVCC=true; shift ;;
        --keepalive-after-success) KEEPALIVE_AFTER_SUCCESS=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${OBJECTIVE}" in
    opd) LOSS_MODE="k1"; LOSS_AGG_MODE="seq-mean-token-mean"; VA_ENABLED=false ;;
    va_opd) LOSS_MODE="va_opd_k1"; LOSS_AGG_MODE="seq-mean-token-sum"; VA_ENABLED=true ;;
    *) echo "FATAL: objective must be opd or va_opd" >&2; exit 2 ;;
esac
case "${PROFILE}" in
    smoke) ;;
    train)
        if ! ${BATCH_EXPLICIT}; then PROMPT_BATCH_SIZE=16; fi
        if ! ${STEPS_EXPLICIT}; then TOTAL_STEPS=0; fi
        TOTAL_EPOCHS=5
        SAVE_FREQ=50
        TEST_FREQ=25
        AUDIT_ALL_IMAGES=true
        ;;
    *) echo "FATAL: profile must be smoke or train" >&2; exit 2 ;;
esac

PYTHON="${ENV_PREFIX}/bin/python"
RAY="${ENV_PREFIX}/bin/ray"
if [[ ! -x "${PYTHON}" || ! -x "${RAY}" ]]; then
    echo "FATAL: native VA-OPD environment is incomplete: ${ENV_PREFIX}" >&2
    echo "Run scripts/hpc/setup_va_opd_native_env.sh on the CPU instance first." >&2
    exit 1
fi

export VERL_VA_OPD_DIR="${VERL_DIR}"
bash "${REPO_ROOT}/scripts/setup/prepare_va_opd_native_verl.sh"
export PYTHONPATH="${REPO_ROOT}/src:${VERL_DIR}:${PYTHONPATH:-}"

if ${PREPARE_DATA} && [[ ! -f "${TRAIN_DATA}" || ! -f "${VAL_DATA}" ]]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/gen_geometry3k_parquet.py" \
        --source "${SOURCE_DATA}" \
        --output "${TRAIN_DATA}" \
        --val-output "${VAL_DATA}" \
        --asset-dir "${DATA_DIR}/assets" \
        --val-size 200 \
        --holdout-val
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="qwen3vl_geometry3k_native_${OBJECTIVE}_${PROFILE}${NAME}_${TIMESTAMP}"
RUN_ROOT="${VA_OPD_RUN_ROOT:-${HPC_ROOT}/fc-opd-storage/runs/va_opd_native}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
CHECKPOINT_DIR="${HPC_ROOT}/fc-opd-storage/checkpoints/va_opd_native/${RUN_ID}"
TRAIN_LOG="${RUN_DIR}/train.log"
mkdir -p "${RUN_DIR}/rollouts" "${RUN_DIR}/validation" "${CHECKPOINT_DIR}"

PREFLIGHT_EXTRA=()
if ${AUDIT_ALL_IMAGES}; then PREFLIGHT_EXTRA+=(--audit-all-images); fi
if ${ALLOW_SYSTEM_NVCC}; then PREFLIGHT_EXTRA+=(--allow-system-nvcc); fi
"${PYTHON}" "${REPO_ROOT}/scripts/hpc/preflight_va_opd_native.py" \
    --repo-root "${REPO_ROOT}" \
    --backend-dir "${VERL_DIR}" \
    --env-prefix "${ENV_PREFIX}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --student-model "${STUDENT_MODEL}" \
    --teacher-model "${TEACHER_MODEL}" \
    --objective "${OBJECTIVE}" \
    --visible-gpus "${VISIBLE_GPUS}" \
    --actor-gpus "${ACTOR_GPUS}" \
    --teacher-gpus "${TEACHER_GPUS}" \
    --teacher-tp "${TEACHER_TP}" \
    --prompt-batch-size "${PROMPT_BATCH_SIZE}" \
    --rollout-n "${ROLLOUT_N}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --max-response-length "${MAX_RESPONSE_LENGTH}" \
    --config-reference "${CONFIG_REFERENCE}" \
    --run-dir "${RUN_DIR}" \
    "${PREFLIGHT_EXTRA[@]}" | tee "${RUN_DIR}/preflight.log"

if ${PREFLIGHT_ONLY}; then
    echo "NATIVE VA-OPD PREFLIGHT PASSED"
    exit 0
fi

IFS=',' read -r -a GPU_ARRAY <<< "${VISIBLE_GPUS}"
if ! ${ALLOW_BUSY_GPUS}; then
    for gpu in "${GPU_ARRAY[@]}"; do
        PIDS="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
        if [[ -n "${PIDS}" ]]; then
            echo "FATAL: GPU ${gpu} has compute PID(s): ${PIDS//$'\n'/, }" >&2
            exit 1
        fi
    done
fi

NVCC_PATH="$(command -v nvcc 2>/dev/null || true)"
if [[ -n "${NVCC_PATH}" && "$(realpath "${NVCC_PATH}")" == /usr/* ]] && ! ${ALLOW_SYSTEM_NVCC}; then
    echo "FATAL: refusing system nvcc: $(realpath "${NVCC_PATH}")" >&2
    exit 1
fi

cleanup() {
    set +e
    "${RAY}" stop -f >/dev/null 2>&1
}
trap cleanup EXIT INT TERM
"${RAY}" stop -f >/dev/null 2>&1 || true

export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
export RAY_memory_usage_threshold=0.95
unset VLLM_ATTENTION_BACKEND

if (( TOTAL_STEPS > 0 )); then
    TRAINING_LIMITS=("trainer.total_training_steps=${TOTAL_STEPS}" "trainer.total_epochs=100")
else
    TRAINING_LIMITS=("trainer.total_training_steps=null" "trainer.total_epochs=${TOTAL_EPOCHS}")
fi

MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
REWARD_FN="file://${REPO_ROOT}/src/dual_track_opd/fc_opd/smoke_reward.py"
set +e
CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" "${PYTHON}" -m verl.trainer.main_ppo \
    "data.train_files=['${TRAIN_DATA}']" \
    "data.val_files=['${VAL_DATA}']" \
    "data.train_batch_size=${PROMPT_BATCH_SIZE}" \
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}" \
    "data.max_response_length=${MAX_RESPONSE_LENGTH}" \
    "data.filter_overlong_prompts=true" \
    "data.truncation=error" \
    "data.image_key=images" \
    "data.dataloader_num_workers=2" \
    "data.custom_cls.path=file://${REPO_ROOT}/src/dual_track_opd/fc_opd/verl_dataset.py" \
    "data.custom_cls.name=FCOPDDataset" \
    "actor_rollout_ref.model.path=${STUDENT_MODEL}" \
    "actor_rollout_ref.model.use_remove_padding=true" \
    "actor_rollout_ref.model.enable_gradient_checkpointing=true" \
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PROMPT_BATCH_SIZE}" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.actor.ppo_epochs=1" \
    "actor_rollout_ref.actor.strategy=fsdp2" \
    "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}" \
    "actor_rollout_ref.actor.calculate_entropy=true" \
    "actor_rollout_ref.actor.use_dynamic_bsz=true" \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240" \
    "actor_rollout_ref.actor.fsdp_config.param_offload=false" \
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false" \
    "actor_rollout_ref.rollout.name=vllm" \
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}" \
    "actor_rollout_ref.rollout.temperature=1.0" \
    "actor_rollout_ref.rollout.top_p=0.99" \
    "actor_rollout_ref.rollout.top_k=-1" \
    "actor_rollout_ref.rollout.agent.num_workers=${ACTOR_GPUS}" \
    "reward_model.enable=false" \
    "custom_reward_function.path=${REWARD_FN}" \
    "custom_reward_function.name=compute_score" \
    "algorithm.adv_estimator=grpo" \
    "algorithm.use_kl_in_reward=false" \
    "+va_opd.enabled=${VA_ENABLED}" \
    "+va_opd.top_fraction=0.20" \
    "+va_opd.lambda_high=0.50" \
    "+va_opd.tau=1.0" \
    "distillation.enabled=true" \
    "distillation.n_gpus_per_node=${TEACHER_GPUS}" \
    "distillation.nnodes=1" \
    "distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL}" \
    "distillation.teacher_models.teacher_model.inference.name=vllm" \
    "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}" \
    "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION}" \
    "distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN}" \
    "distillation.teacher_models.teacher_model.inference.max_num_seqs=4" \
    "distillation.distillation_loss.loss_mode=${LOSS_MODE}" \
    "distillation.distillation_loss.use_policy_gradient=true" \
    "distillation.distillation_loss.use_task_rewards=false" \
    "distillation.distillation_loss.loss_max_clamp=10.0" \
    "distillation.distillation_loss.log_prob_min_clamp=-10.0" \
    "trainer.n_gpus_per_node=${ACTOR_GPUS}" \
    "trainer.nnodes=1" \
    "trainer.logger=['console']" \
    "trainer.project_name=qwen3vl_geometry3k_native_opd" \
    "trainer.experiment_name=${RUN_ID}" \
    "trainer.val_before_train=true" \
    "trainer.test_freq=${TEST_FREQ}" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.default_local_dir=${CHECKPOINT_DIR}" \
    "trainer.rollout_data_dir=${RUN_DIR}/rollouts" \
    "trainer.validation_data_dir=${RUN_DIR}/validation" \
    "trainer.resume_mode=disable" \
    "${TRAINING_LIMITS[@]}" \
    2>&1 | tee "${TRAIN_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

cleanup
trap - EXIT INT TERM

FINALIZE_ARGS=()
if (( TOTAL_STEPS > 0 )); then FINALIZE_ARGS+=(--requested-steps "${TOTAL_STEPS}"); fi
if ! "${PYTHON}" "${REPO_ROOT}/scripts/hpc/finalize_va_opd_native_run.py" \
    --manifest "${RUN_DIR}/run_manifest.json" \
    --train-log "${TRAIN_LOG}" \
    --objective "${OBJECTIVE}" \
    --exit-code "${TRAIN_EXIT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    "${FINALIZE_ARGS[@]}"; then
    echo "NATIVE ${OBJECTIVE} RUN FAILED" >&2
    tail -n 120 "${TRAIN_LOG}" >&2 || true
    exit 1
fi

if ${KEEPALIVE_AFTER_SUCCESS}; then
    CUDA_VISIBLE_DEVICES=3,4 KEEPALIVE_TARGET_UTIL=0.45 KEEPALIVE_WORK_ITERS=32 \
        nohup "${PYTHON}" -u "${HPC_ROOT}/scripts/busy_keepalive.py" \
        >"${RUN_DIR}/keepalive.log" 2>&1 &
    echo "$!" >"${RUN_DIR}/keepalive.pid"
fi

echo "NATIVE ${OBJECTIVE} PASSED"
echo "Run directory: ${RUN_DIR}"
echo "Checkpoint:    ${CHECKPOINT_DIR}"

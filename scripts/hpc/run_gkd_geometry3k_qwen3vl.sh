#!/usr/bin/env bash
# Canonical online Geometry3K distillation launcher for Qwen3-VL.
#
# Default objective is vanilla GKD: the current 4B rollout is immediately
# forced-scored by the 32B teacher under the full image, and forward KL is the
# entire actor objective.  The same launcher can later switch to VA-OPD without
# changing the data, rollout, teacher protocol, or actor transport path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIGINAL_ARGS=("$@")

ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
ENV_PATH="${FC_OPD_ENV:-${ROOT}/fc-opd-storage/envs/fc-opd-verl071-cu128}"
CUDA_TOOLCHAIN="${CUDA_TOOLCHAIN:-${ROOT}/envs/cuda128-toolchain}"
STUDENT_MODEL="${FC_OPD_STUDENT_MODEL:-${ROOT}/models/Qwen3-VL-4B-Instruct}"
TEACHER_MODEL="${FC_OPD_TEACHER_MODEL:-${ROOT}/models/Qwen3-VL-32B-Instruct}"
SOURCE_DATA="${GEOMETRY3K_SOURCE:-${ROOT}/dataset/geometry3k/data/train-00000-of-00001.parquet}"
DATA_DIR="${GEOMETRY3K_GKD_DATA_DIR:-${ROOT}/fc-opd-storage/outputs/fc_opd/geometry3k_gkd}"
TRAIN_DATA="${DATA_DIR}/train.parquet"
VAL_DATA="${DATA_DIR}/val.parquet"
BACKEND_DIR="${VERL_DIR:-${REPO_ROOT}/third_party/verl}"
CONFIG_REFERENCE="${REPO_ROOT}/configs/experiment/qwen3vl_geometry3k_gkd.yaml"

OBJECTIVE="gkd"
TEACHER_GPU=0
TRAIN_GPU_LIST="1,2,3,4"
STEPS=10
TRAIN_BATCH_SIZE=4
ROLLOUT_N=1
ROLLOUT_N_EXPLICIT=false
TOP_K=256
MAX_PROMPT_LENGTH=6144
MAX_RESPONSE_LENGTH=1024
GPU_MEMORY_UTILIZATION=0.45
LEARNING_RATE=1e-6
TEMPERATURE=1.0
TOP_P=0.99
TEACHER_PORT=18080
SAVE_FREQ=0
NAME=""
PREPARE_DATA=true
PREFLIGHT_ONLY=false
BACKGROUND=false
ALLOW_BUSY_GPUS=false
ALLOW_HIGH_TEACHER_EOS=false
IGNORE_EOS=false

usage() {
    cat <<'EOF'
Usage: bash scripts/hpc/run_gkd_geometry3k_qwen3vl.sh [options]

  --objective gkd|va_opd|va_opd_jsd  Pure actor objective (default: gkd)
  --steps N                           Optimizer updates (default: 10)
  --teacher-gpu N                     Teacher physical GPU (default: 0)
  --train-gpus LIST                   Training physical GPUs (default: 1,2,3,4)
  --student-model PATH                Qwen3-VL student checkpoint
  --teacher-model PATH                Qwen3-VL teacher checkpoint
  --source-data PATH                  Raw Geometry3K parquet
  --train-data PATH                   Prepared train parquet
  --val-data PATH                     Prepared validation parquet
  --batch-size N                      Prompt batch size (default: 4)
  --rollout-n N                       Current-policy siblings per prompt (default: 1)
  --top-k N                           Teacher sparse support (default: 256, official parity)
  --diagnostic-ignore-eos             Force max-length rollouts to test EOS-collapse causality
  --save-freq N                       Checkpoint interval; 0 means final step
  --name TAG                          Run-name suffix
  --skip-data-prepare                 Require prepared parquets to exist
  --preflight-only                    Validate data/models/backend without GPUs
  --allow-busy-gpus                   Override the stale-process safety check
  --allow-high-teacher-eos            Override first-token EOS probability gate (known-risk)
  --background                        Relaunch under nohup
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --objective) OBJECTIVE="${2:?missing objective}"; shift 2 ;;
        --steps) STEPS="${2:?missing steps}"; shift 2 ;;
        --teacher-gpu) TEACHER_GPU="${2:?missing teacher GPU}"; shift 2 ;;
        --train-gpus) TRAIN_GPU_LIST="${2:?missing train GPUs}"; shift 2 ;;
        --student-model) STUDENT_MODEL="${2:?missing student model}"; shift 2 ;;
        --teacher-model) TEACHER_MODEL="${2:?missing teacher model}"; shift 2 ;;
        --source-data) SOURCE_DATA="${2:?missing source data}"; shift 2 ;;
        --train-data) TRAIN_DATA="${2:?missing train data}"; PREPARE_DATA=false; shift 2 ;;
        --val-data) VAL_DATA="${2:?missing val data}"; PREPARE_DATA=false; shift 2 ;;
        --batch-size) TRAIN_BATCH_SIZE="${2:?missing batch size}"; shift 2 ;;
        --rollout-n) ROLLOUT_N="${2:?missing rollout n}"; ROLLOUT_N_EXPLICIT=true; shift 2 ;;
        --top-k) TOP_K="${2:?missing top-k}"; shift 2 ;;
        --save-freq) SAVE_FREQ="${2:?missing save frequency}"; shift 2 ;;
        --name) NAME="_${2:?missing name}"; shift 2 ;;
        --skip-data-prepare) PREPARE_DATA=false; shift ;;
        --preflight-only) PREFLIGHT_ONLY=true; shift ;;
        --allow-busy-gpus) ALLOW_BUSY_GPUS=true; shift ;;
        --allow-high-teacher-eos) ALLOW_HIGH_TEACHER_EOS=true; shift ;;
        --diagnostic-ignore-eos) IGNORE_EOS=true; shift ;;
        --background) BACKGROUND=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${OBJECTIVE}" in
    gkd) CONDITIONS="[full]" ;;
    va_opd|va_opd_jsd) CONDITIONS="[full,degraded]"
        if ! ${ROLLOUT_N_EXPLICIT}; then ROLLOUT_N=4; fi
        ;;
    *) echo "FATAL: unsupported objective ${OBJECTIVE}" >&2; exit 2 ;;
esac

if ${BACKGROUND}; then
    RELAUNCH=()
    for arg in "${ORIGINAL_ARGS[@]}"; do
        [[ "${arg}" == "--background" ]] || RELAUNCH+=("${arg}")
    done
    BG_DIR="${REPO_ROOT}/runs/gkd_geometry3k"
    mkdir -p "${BG_DIR}"
    BG_LOG="${BG_DIR}/nohup_$(date +%Y%m%d_%H%M%S).log"
    nohup bash "$0" "${RELAUNCH[@]}" >"${BG_LOG}" 2>&1 &
    disown
    echo "Background PID: $!"
    echo "Monitor: tail -f ${BG_LOG}"
    exit 0
fi

PYTHON="${ENV_PATH}/bin/python"
RAY="${ENV_PATH}/bin/ray"
if [[ ! -x "${PYTHON}" || ! -x "${RAY}" ]]; then
    echo "FATAL: incomplete verl environment: ${ENV_PATH}" >&2
    exit 1
fi
export PYTHONPATH="${REPO_ROOT}/src:${BACKEND_DIR}:${PYTHONPATH:-}"
if [[ -x "${CUDA_TOOLCHAIN}/bin/nvcc" ]]; then
    export CUDA_HOME="${CUDA_TOOLCHAIN}"
    export CUDA_PATH="${CUDA_TOOLCHAIN}"
    export PATH="${CUDA_TOOLCHAIN}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi

IFS=',' read -r -a TRAIN_GPUS <<< "${TRAIN_GPU_LIST}"
TRAIN_GPU_COUNT=${#TRAIN_GPUS[@]}
if (( TRAIN_GPU_COUNT < 1 || STEPS < 1 || TRAIN_BATCH_SIZE < 1 || ROLLOUT_N < 1 )); then
    echo "FATAL: GPU count, steps, batch size, and rollout n must be positive" >&2
    exit 1
fi
if (( TRAIN_BATCH_SIZE % TRAIN_GPU_COUNT != 0 )); then
    echo "FATAL: batch size ${TRAIN_BATCH_SIZE} must be divisible by ${TRAIN_GPU_COUNT} training GPUs" >&2
    exit 1
fi
for gpu in "${TRAIN_GPUS[@]}"; do
    if [[ "${gpu}" == "${TEACHER_GPU}" ]]; then
        echo "FATAL: teacher GPU overlaps training GPU ${gpu}" >&2
        exit 1
    fi
done
if (( SAVE_FREQ <= 0 )); then SAVE_FREQ=${STEPS}; fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="qwen3vl_geometry3k_${OBJECTIVE}${NAME}_${TIMESTAMP}"
RUN_DIR="${REPO_ROOT}/runs/gkd_geometry3k/${RUN_ID}"
TRAIN_LOG="${RUN_DIR}/train.log"
TEACHER_LOG="${RUN_DIR}/teacher.log"
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints/gkd_geometry3k/${RUN_ID}"
ROLLOUT_DIR="${RUN_DIR}/rollouts"
VALIDATION_DIR="${RUN_DIR}/validation"
mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${ROLLOUT_DIR}" "${VALIDATION_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  Qwen3-VL Geometry3K Online Distillation"
echo "  Run ID:          ${RUN_ID}"
echo "  Objective:       ${OBJECTIVE} ${CONDITIONS}"
echo "  Steps:           ${STEPS}"
echo "  Student:         ${STUDENT_MODEL}"
echo "  Teacher:         ${TEACHER_MODEL}"
echo "  Teacher GPU:     ${TEACHER_GPU}"
echo "  Train GPUs:      ${TRAIN_GPU_LIST}"
echo "  Batch × rollout: ${TRAIN_BATCH_SIZE} × ${ROLLOUT_N}"
echo "  Ignore EOS:      ${IGNORE_EOS} (diagnostic only)"
echo "  Train data:      ${TRAIN_DATA}"
echo "  Run dir:         ${RUN_DIR}"
echo "══════════════════════════════════════════════════════════════"

if ${PREPARE_DATA} && [[ ! -f "${TRAIN_DATA}" || ! -f "${VAL_DATA}" ]]; then
    echo "=== Preparing Geometry3K train/validation parquet ==="
    "${PYTHON}" "${REPO_ROOT}/scripts/gen_geometry3k_parquet.py" \
        --source "${SOURCE_DATA}" \
        --output "${TRAIN_DATA}" \
        --val-output "${VAL_DATA}" \
        --asset-dir "${DATA_DIR}/assets" \
        --val-size 200 \
        --holdout-val
fi

echo "=== Preparing verl online-distillation overlays ==="
PYTHON_BIN="${PYTHON}" VERL_DIR="${BACKEND_DIR}" \
    bash "${REPO_ROOT}/scripts/setup/prepare_fc_opd_verl.sh"

echo "=== CPU-safe config/data/model preflight ==="
"${PYTHON}" "${REPO_ROOT}/scripts/hpc/preflight_gkd_geometry3k.py" \
    --repo-root "${REPO_ROOT}" \
    --backend-dir "${BACKEND_DIR}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --student-model "${STUDENT_MODEL}" \
    --teacher-model "${TEACHER_MODEL}" \
    --objective "${OBJECTIVE}" \
    --teacher-gpu "${TEACHER_GPU}" \
    --train-gpus "${TRAIN_GPU_LIST}" \
    --steps "${STEPS}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --rollout-n "${ROLLOUT_N}" \
    --top-k "${TOP_K}" \
    --env-path "${ENV_PATH}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --max-response-length "${MAX_RESPONSE_LENGTH}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --learning-rate "${LEARNING_RATE}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --ignore-eos "${IGNORE_EOS}" \
    --save-freq "${SAVE_FREQ}" \
    --teacher-port "${TEACHER_PORT}" \
    --config-reference "${CONFIG_REFERENCE}" \
    --run-dir "${RUN_DIR}" | tee "${RUN_DIR}/preflight.log"

if ${PREFLIGHT_ONLY}; then
    echo "PREFLIGHT PASSED ✓"
    exit 0
fi

TEACHER_PID=""
cleanup() {
    set +e
    "${RAY}" stop -f >/dev/null 2>&1
    if [[ -n "${TEACHER_PID}" ]]; then
        kill -TERM -- "-${TEACHER_PID}" 2>/dev/null || kill -TERM "${TEACHER_PID}" 2>/dev/null
        sleep 2
        kill -KILL -- "-${TEACHER_PID}" 2>/dev/null || kill -KILL "${TEACHER_PID}" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

echo "=== Cleanup stale Ray and owned teacher port ==="
"${RAY}" stop -f >/dev/null 2>&1 || true
if command -v lsof >/dev/null 2>&1; then
    EXISTING=$(lsof -ti:"${TEACHER_PORT}" 2>/dev/null || true)
    [[ -z "${EXISTING}" ]] || kill -TERM ${EXISTING} 2>/dev/null || true
fi
sleep 2

if ! ${ALLOW_BUSY_GPUS}; then
    for gpu in "${TEACHER_GPU}" "${TRAIN_GPUS[@]}"; do
        PIDS=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
        if [[ -n "${PIDS}" ]]; then
            echo "FATAL: GPU ${gpu} still has compute PID(s): ${PIDS//$'\n'/, }" >&2
            echo "Inspect them or pass --allow-busy-gpus deliberately." >&2
            exit 1
        fi
    done
fi

echo "=== Starting Qwen3-VL teacher (GPU ${TEACHER_GPU}) ==="
setsid env CUDA_VISIBLE_DEVICES="${TEACHER_GPU}" PYTHONPATH="${PYTHONPATH}" \
    "${PYTHON}" -m dual_track_opd.fc_opd.teacher_service \
    --backend transformers --model "${TEACHER_MODEL}" \
    --port "${TEACHER_PORT}" --top-k "${TOP_K}" --dtype bfloat16 --device cuda:0 \
    >"${TEACHER_LOG}" 2>&1 &
TEACHER_PID=$!
READY=false
for _ in $(seq 1 1200); do
    if curl -fsS "http://127.0.0.1:${TEACHER_PORT}/health" >/dev/null 2>&1; then READY=true; break; fi
    if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
        echo "FATAL: teacher exited during startup" >&2
        tail -n 80 "${TEACHER_LOG}" >&2
        exit 1
    fi
    sleep 1
done
if ! ${READY}; then
    echo "FATAL: teacher health timeout" >&2
    tail -n 80 "${TEACHER_LOG}" >&2
    exit 1
fi

echo "=== Teacher end-to-end image warmup ==="
WARMUP_EXTRA=()
if ${ALLOW_HIGH_TEACHER_EOS}; then WARMUP_EXTRA+=(--allow-high-teacher-eos); fi
"${PYTHON}" "${REPO_ROOT}/scripts/hpc/warmup_gkd_geometry3k_teacher.py" \
    --data "${TRAIN_DATA}" --student-model "${STUDENT_MODEL}" \
    --teacher-url "http://127.0.0.1:${TEACHER_PORT}" --objective "${OBJECTIVE}" \
    --diagnostic-output "${RUN_DIR}/teacher_alignment_diagnostic.json" \
    "${WARMUP_EXTRA[@]}"

echo "=== Starting Ray on GPUs ${TRAIN_GPU_LIST} ==="
CUDA_VISIBLE_DEVICES="${TRAIN_GPU_LIST}" "${RAY}" start --head \
    --num-gpus="${TRAIN_GPU_COUNT}" --disable-usage-stats

PPO_MINI_BATCH_SIZE=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
REWARD_FN="file://${REPO_ROOT}/src/dual_track_opd/fc_opd/smoke_reward.py"
VERL_CONFIG_DIR="${BACKEND_DIR}/verl/trainer/config"
export RAY_memory_usage_threshold=0.95
export NCCL_NVLS_ENABLE=1
export NCCL_IBEXT_DISABLE=1
export NCCL_BUFFSIZE=4194304
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200

echo "=== Running ${OBJECTIVE} (${STEPS} optimizer steps) ==="
set +e
CUDA_VISIBLE_DEVICES="${TRAIN_GPU_LIST}" "${PYTHON}" -m verl.trainer.main_ppo \
    --config-path="${VERL_CONFIG_DIR}" --config-name=ppo_trainer \
    "data.train_files=${TRAIN_DATA}" \
    "data.val_files=${VAL_DATA}" \
    "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}" \
    "data.max_response_length=${MAX_RESPONSE_LENGTH}" \
    "data.filter_overlong_prompts=false" \
    "data.truncation=error" \
    "data.image_key=images" \
    "data.dataloader_num_workers=2" \
    "data.custom_cls.path=file://${REPO_ROOT}/src/dual_track_opd/fc_opd/verl_dataset.py" \
    "data.custom_cls.name=FCOPDDataset" \
    "actor_rollout_ref.model.path=${STUDENT_MODEL}" \
    "actor_rollout_ref.model.use_remove_padding=false" \
    "actor_rollout_ref.model.use_fused_kernels=false" \
    "actor_rollout_ref.model.enable_gradient_checkpointing=true" \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.actor.use_dynamic_bsz=true" \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192" \
    "actor_rollout_ref.actor.use_kl_loss=false" \
    "actor_rollout_ref.actor.fsdp_config.param_offload=false" \
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false" \
    "actor_rollout_ref.actor.strategy=fsdp2" \
    "actor_rollout_ref.ref.strategy=fsdp2" \
    "actor_rollout_ref.rollout.name=vllm" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
    "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
    "actor_rollout_ref.rollout.temperature=${TEMPERATURE}" \
    "actor_rollout_ref.rollout.top_p=${TOP_P}" \
    "actor_rollout_ref.rollout.top_k=-1" \
    "actor_rollout_ref.rollout.ignore_eos=${IGNORE_EOS}" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4" \
    "actor_rollout_ref.rollout.agent.num_workers=${TRAIN_GPU_COUNT}" \
    "actor_rollout_ref.ref.fsdp_config.param_offload=false" \
    "reward_model.enable=false" \
    "reward_model.num_workers=null" \
    "reward_model.reward_manager=null" \
    "reward_model.reward_loop_source=null" \
    "reward_model.reward_loop_module_path=null" \
    "reward_model.reward_loop_class_name=null" \
    "reward_model.model.path=null" \
    "custom_reward_function.path=${REWARD_FN}" \
    "custom_reward_function.name=compute_score" \
    "algorithm.adv_estimator=grpo" \
    "algorithm.use_kl_in_reward=false" \
    "+algorithm.fc_opd.post_rollout_hook=dual_track_opd.fc_opd.verl_post_rollout_hook.fc_opd_post_rollout_hook" \
    "+algorithm.fc_opd.compute_hook_loss=false" \
    "+algorithm.fc_opd.teacher_url=http://127.0.0.1:${TEACHER_PORT}" \
    "+algorithm.fc_opd.conditions=${CONDITIONS}" \
    "+algorithm.fc_opd.loss_coef=1.0" \
    "+algorithm.fc_opd.loss_mode=${OBJECTIVE}" \
    "+algorithm.fc_opd.expected_rollouts=${ROLLOUT_N}" \
    "+algorithm.fc_opd.renormalize_topk=true" \
    "+algorithm.fc_opd.include_tail=false" \
    "trainer.total_epochs=100" \
    "trainer.total_training_steps=${STEPS}" \
    "trainer.n_gpus_per_node=${TRAIN_GPU_COUNT}" \
    "trainer.nnodes=1" \
    "trainer.critic_warmup=0" \
    "trainer.logger=['console']" \
    "trainer.project_name=qwen3vl_geometry3k_distill" \
    "trainer.experiment_name=${RUN_ID}" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.test_freq=${SAVE_FREQ}" \
    "trainer.val_before_train=true" \
    "trainer.rollout_data_dir=${ROLLOUT_DIR}" \
    "trainer.validation_data_dir=${VALIDATION_DIR}" \
    "trainer.default_local_dir=${CHECKPOINT_DIR}" \
    "trainer.resume_mode=disable" \
    2>&1 | tee "${TRAIN_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

# PPO trainer emits one consolidated metrics line per completed actor update.
UPDATE_COUNT=$(grep -c "training/global_step:" "${TRAIN_LOG}" 2>/dev/null || true)
LOSS_COUNT=$(grep -c "actor/fc_opd_loss" "${TRAIN_LOG}" 2>/dev/null || true)
FINITE_GRAD_COUNT=$(grep -cE "actor/grad_norm:[[:space:]]*[0-9]" "${TRAIN_LOG}" 2>/dev/null || true)

"${PYTHON}" "${REPO_ROOT}/scripts/hpc/finalize_gkd_geometry3k_run.py" \
    --manifest "${RUN_DIR}/run_manifest.json" \
    --exit-code "${TRAIN_EXIT}" \
    --updates "${UPDATE_COUNT}" \
    --requested-steps "${STEPS}" \
    --loss-lines "${LOSS_COUNT}" \
    --finite-grad-lines "${FINITE_GRAD_COUNT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}"

echo "=== Run validation ==="
echo "  Exit code:          ${TRAIN_EXIT}"
echo "  Actor updates:      ${UPDATE_COUNT}/${STEPS}"
echo "  Distillation loss:  ${LOSS_COUNT} log occurrence(s)"
echo "  Finite grad lines:  ${FINITE_GRAD_COUNT}"
if (( TRAIN_EXIT != 0 || UPDATE_COUNT < STEPS || LOSS_COUNT < 1 || FINITE_GRAD_COUNT < 1 )); then
    echo "GKD/VA-OPD RUN FAILED ✗"
    tail -n 80 "${TRAIN_LOG}" || true
    exit 1
fi

echo "══════════════════════════════════════════════════════════════"
echo "  Run ID:      ${RUN_ID}"
echo "  Objective:   ${OBJECTIVE}"
echo "  Updates:     ${UPDATE_COUNT}"
echo "  Train log:   ${TRAIN_LOG}"
echo "  Teacher log: ${TEACHER_LOG}"
echo "  Checkpoint:  ${CHECKPOINT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo "ONLINE DISTILLATION PASSED ✓"

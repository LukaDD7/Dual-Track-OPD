#!/usr/bin/env bash
# VA-OPD overnight multi-step training.
#
# Scaled-up config referencing Vision-OPD (VA-OPD) paper settings:
#   - Rollout n=8.
#   - LR 2e-6 (VA-OPD: 2e-6 for 4B).
#   - GPU memory 0.45 for vLLM (VA-OPD: 0.7; we're conservative with FSDP).
#   - JSD loss default (bounded, avoids reverse-KL mode collapse).
#   - Checkpoint every 100 steps, keep last 5.
#   - Log to file + console.
#
# Recommended usage (8×H200, 4-rank power-of-2 training):
#   bash scripts/hpc/run_verl_fc_opd_overnight.sh \
#       --teacher-gpus 0 --train-gpus 1,2,3,4 \
#       --name t1_train4 --background
#
# Legacy usage (still works, deprecated):
#   bash scripts/hpc/run_verl_fc_opd_overnight.sh --gpus 4 --background
#
# Flags:
#   --teacher-gpus LIST     GPU indices for teacher(s), e.g. 0 or 0,1 (recommended)
#   --train-gpus LIST       GPU indices for verl training, e.g. 1,2,3,4 (recommended)
#   --allow-nonpower2       Allow non-power-of-2 training world size (risky)
#   --gpus N                [legacy] Total GPU count; teacher auto-derived
#   --steps S               PPO steps (default: 0 = auto-compute 5 epochs)
#   --batch-size N          Override train batch size (default: auto)
#   --data PATH             Override parquet path
#   --name TAG              Run name suffix
#   --test-fix MODE         NCCL workaround: ring|noreshard|hsdp3|replicate
#   --loss-mode MODE        va_opd (reverse KL) | va_opd_jsd (JSD, default)
#   --resume PATH           Resume from checkpoint dir
#   --keepalive             Start post-success GPU keepalive
#   --background            Detach via nohup — safe to close terminal
#
# Background mode:
#   When --background is passed, the script re-launches itself under nohup and
#   exits immediately.  The training runs in the background and survives
#   terminal / code-server disconnects.  Check progress with:
#     tail -f artifacts/fc_opd/train_fc_opd_overnight_*.log

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ── defaults (new explicit layout) ──────────────────────────────────────────
TEACHER_GPU_LIST="0"             # --teacher-gpus
TRAIN_GPU_LIST="1,2,3,4"        # --train-gpus
ALLOW_NONPOWER2=false            # --allow-nonpower2
USE_EXPLICIT_LAYOUT=false        # true when --teacher-gpus or --train-gpus given

# ── legacy fallback ─────────────────────────────────────────────────────────
GPU_COUNT=0                      # 0 = not using legacy mode

# ── training params ─────────────────────────────────────────────────────────
NUM_STEPS=0  # 0 = auto: 5 epochs × 2101 prompts / TRAIN_BATCH_SIZE
TOP_K=32
TEACHER_PORT=18080
TEACHER_PORT_2=18081
RUN_BACKGROUND=false
PARQUET_OVERRIDE=""
KEEPALIVE_SEC=0
RESUME_CKPT=""
NUM_EPOCHS=5
DATASET_SIZE=2101
NAME_TAG=""
TEST_FIX=""  # ring | noreshard | hsdp3 | replicate
BATCH_OVERRIDE=""  # empty = auto: floor(8 / TRAIN_GPUS) * TRAIN_GPUS
LOSS_MODE="va_opd_jsd"  # va_opd | va_opd_jsd

# ── parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --teacher-gpus)    TEACHER_GPU_LIST="${2:?--teacher-gpus needs a list}"; USE_EXPLICIT_LAYOUT=true; shift 2 ;;
        --train-gpus)      TRAIN_GPU_LIST="${2:?--train-gpus needs a list}";   USE_EXPLICIT_LAYOUT=true; shift 2 ;;
        --allow-nonpower2) ALLOW_NONPOWER2=true; shift ;;
        --gpus)            GPU_COUNT="${2:?--gpus needs a value}"; shift 2 ;;
        --steps)           NUM_STEPS="${2:?--steps needs a value}"; shift 2 ;;
        --batch-size)      BATCH_OVERRIDE="${2:?--batch-size needs a value}"; shift 2 ;;
        --data)            PARQUET_OVERRIDE="${2:?--data needs a path}"; shift 2 ;;
        --name)            NAME_TAG="_${2:?--name needs a value}"; shift 2 ;;
        --test-fix)        TEST_FIX="${2:?--test-fix needs ring|noreshard|hsdp3|replicate}"; NAME_TAG="${NAME_TAG}_${2}"; shift 2 ;;
        --loss-mode)       LOSS_MODE="${2:?--loss-mode needs va_opd|va_opd_jsd}"; shift 2 ;;
        --resume)          RESUME_CKPT="${2:?--resume needs a checkpoint dir}"; shift 2 ;;
        --keepalive)       KEEPALIVE_SEC=86400; shift ;;
        --background)      RUN_BACKGROUND=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── background re-launch ────────────────────────────────────────────────────
if ${RUN_BACKGROUND}; then
    # Re-invoke this script without --background, under nohup.
    # Build args without --background
    RELAUNCH_ARGS=()
    for arg in "$@"; do
        [[ "$arg" != "--background" ]] || continue
        RELAUNCH_ARGS+=("$arg")
    done
    # Carry all optional overrides explicitly (safer than re-parsing changed state)
    if [[ -n "${PARQUET_OVERRIDE}" ]]; then
        RELAUNCH_ARGS+=(--data "${PARQUET_OVERRIDE}")
    fi
    if [[ -n "${RESUME_CKPT}" ]]; then
        RELAUNCH_ARGS+=(--resume "${RESUME_CKPT}")
    fi
    if [[ -n "${NAME_TAG}" ]]; then
        RELAUNCH_ARGS+=(--name "${NAME_TAG#_}")
    fi
    if [[ -n "${TEST_FIX}" ]]; then
        RELAUNCH_ARGS+=(--test-fix "${TEST_FIX}")
    fi
    if [[ -n "${BATCH_OVERRIDE}" ]]; then
        RELAUNCH_ARGS+=(--batch-size "${BATCH_OVERRIDE}")
    fi
    if [[ -n "${LOSS_MODE}" ]]; then
        RELAUNCH_ARGS+=(--loss-mode "${LOSS_MODE}")
    fi
    if ${ALLOW_NONPOWER2}; then
        RELAUNCH_ARGS+=(--allow-nonpower2)
    fi
    if ${USE_EXPLICIT_LAYOUT}; then
        RELAUNCH_ARGS+=(--teacher-gpus "${TEACHER_GPU_LIST}")
        RELAUNCH_ARGS+=(--train-gpus "${TRAIN_GPU_LIST}")
    fi
    if [[ -n "${KEEPALIVE_SEC}" ]] && (( KEEPALIVE_SEC > 0 )); then
        RELAUNCH_ARGS+=(--keepalive)
    fi
    NOHUP_LOG="${REPO_ROOT}/artifacts/fc_opd/nohup_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "${NOHUP_LOG}")"
    echo "Launching background training (PID will be printed, then exits)."
    echo "Monitor:  tail -f ${NOHUP_LOG}"
    # Re-invoke with explicit layout args, not legacy --gpus
    nohup bash "$0" --teacher-gpus "${TEACHER_GPU_LIST}" --train-gpus "${TRAIN_GPU_LIST}" \
        --steps "${NUM_STEPS}" "${RELAUNCH_ARGS[@]}" \
        > "${NOHUP_LOG}" 2>&1 &
    disown
    echo "Background PID: $!"
    exit 0
fi

# ── resolve GPU layout ──────────────────────────────────────────────────────
# Helper: convert comma-separated list to bash array and count elements.
_parse_gpu_list() {
    # Prints elements one per line; count with wc -l.
    local _list="$1"
    echo "${_list}" | tr ',' '\n' | sed '/^[[:space:]]*$/d'
}

if ${USE_EXPLICIT_LAYOUT}; then
    # ── explicit layout (recommended) ───────────────────────────────────────
    TEACHER_GPUS=($(_parse_gpu_list "${TEACHER_GPU_LIST}"))
    TRAIN_GPUS_ARR=($(_parse_gpu_list "${TRAIN_GPU_LIST}"))
    NUM_TEACHERS=${#TEACHER_GPUS[@]}
    TRAIN_GPUS=${#TRAIN_GPUS_ARR[@]}
    VERL_GPU_LIST="${TRAIN_GPU_LIST}"
    VERL_GPUS=${TRAIN_GPUS}
else
    # ── legacy layout (--gpus N) ────────────────────────────────────────────
    if (( GPU_COUNT <= 0 )); then
        GPU_COUNT=4
    fi
    echo "=== WARNING: --gpus is legacy.  Prefer --teacher-gpus / --train-gpus. ==="
    if (( GPU_COUNT < 2 )); then
        echo "FATAL: need at least 2 GPUs (1=teacher, >=1=verl train)"
        exit 1
    fi
    if (( GPU_COUNT >= 5 )); then
        NUM_TEACHERS=2
        TEACHER_GPUS=(0 1)
        TRAIN_GPUS=$(( GPU_COUNT - 2 ))
        VERL_GPU_LIST=$(seq -s, 2 $(( GPU_COUNT - 1 )))
    else
        NUM_TEACHERS=1
        TEACHER_GPUS=(0)
        TRAIN_GPUS=$(( GPU_COUNT - 1 ))
        VERL_GPU_LIST=$(seq -s, 1 $(( GPU_COUNT - 1 )))
    fi
    TEACHER_GPU_LIST=$(IFS=,; echo "${TEACHER_GPUS[*]}")
    TRAIN_GPU_LIST="${VERL_GPU_LIST}"
    TRAIN_GPUS_ARR=($(_parse_gpu_list "${TRAIN_GPU_LIST}"))
    VERL_GPUS=${TRAIN_GPUS}
fi

# ── power-of-two guard ──────────────────────────────────────────────────────
_is_power_of_two() {
    local _n="$1"
    (( _n > 0 )) && (( (_n & (_n - 1)) == 0 ))
}
if ! _is_power_of_two "${TRAIN_GPUS}"; then
    if ${ALLOW_NONPOWER2}; then
        echo "=== WARNING: TRAIN_GPUS=${TRAIN_GPUS} is NOT a power of 2. ==="
        echo "=== This topology is known risky for FSDP2 / NCCL allgather deadlock. ==="
        echo "=== Proceeding because --allow-nonpower2 was given. ==="
    else
        echo "FATAL: TRAIN_GPUS=${TRAIN_GPUS} is non-power-of-two and known risky"
        echo "for FSDP2 / NCCL allgather deadlock on this node."
        echo "Use --allow-nonpower2 to override, or set --train-gpus to 1,2,4,8."
        echo "Recommended: --teacher-gpus 0 --train-gpus 1,2,3,4"
        exit 1
    fi
fi

# ── teacher URLs ────────────────────────────────────────────────────────────
if (( NUM_TEACHERS == 1 )); then
    TEACHER_URLS="http://127.0.0.1:${TEACHER_PORT}"
elif (( NUM_TEACHERS == 2 )); then
    TEACHER_URLS="http://127.0.0.1:${TEACHER_PORT},http://127.0.0.1:${TEACHER_PORT_2}"
else
    echo "FATAL: NUM_TEACHERS=${NUM_TEACHERS} unsupported (max 2)"
    exit 1
fi

# ── fixed paths ─────────────────────────────────────────────────────────────
MODEL_PATH="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-4B-Instruct"
TEACHER_MODEL="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct"
PARQUET="${PARQUET_OVERRIDE:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/geometry3k_full/train.parquet}"
CONDA_ENV="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/envs/fc-opd-verl071-cu128"
REPO_ROOT_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="fc_opd_overnight${NAME_TAG}_$(date +%Y%m%d_%H%M%S)"
TEACHER_LOG="${REPO_ROOT_ABS}/artifacts/fc_opd/teacher_${RUN_ID}.log"
TRAIN_LOG="${REPO_ROOT_ABS}/artifacts/fc_opd/train_${RUN_ID}.log"
VERL_CONFIG_DIR="${REPO_ROOT_ABS}/third_party/verl/verl/trainer/config"
REWARD_FN="file://${REPO_ROOT_ABS}/src/dual_track_opd/fc_opd/smoke_reward.py"
mkdir -p "$(dirname "${TEACHER_LOG}")"

# ── validation split ─────────────────────────────────────────────────────────
VAL_PARQUET="${PARQUET%.parquet}_val200.parquet"
ROLLOUT_DIR="${REPO_ROOT_ABS}/outputs/${RUN_ID}/rollouts"
VAL_DIR="${REPO_ROOT_ABS}/outputs/${RUN_ID}/validation"
if [ ! -f "${VAL_PARQUET}" ]; then
    echo "=== Creating validation split: ${VAL_PARQUET} ==="
    ${CONDA_ENV}/bin/python -c "
import pandas as pd
df = pd.read_parquet('${PARQUET}')
n_val = min(200, len(df) // 10)
val = df.tail(n_val)
val.to_parquet('${VAL_PARQUET}', index=False)
print(f'Val split created: {len(val)} rows (last {n_val} of {len(df)})')
"
fi
mkdir -p "${ROLLOUT_DIR}" "${VAL_DIR}"

# ── batch / steps math ──────────────────────────────────────────────────────
SAVE_FREQ=100
ROLLOUT_N=8

if [[ -n "${BATCH_OVERRIDE}" ]]; then
    TRAIN_BATCH_SIZE="${BATCH_OVERRIDE}"
else
    # Choose the largest multiple of TRAIN_GPUS that is ≤ 8.
    TRAIN_BATCH_SIZE=$(( (8 / TRAIN_GPUS) * TRAIN_GPUS ))
fi
if (( TRAIN_BATCH_SIZE < 1 )); then TRAIN_BATCH_SIZE=${TRAIN_GPUS}; fi
PPO_MINI_BATCH_SIZE=$(( TRAIN_BATCH_SIZE * ROLLOUT_N ))
MICRO_BATCH_PER_GPU=1

if (( NUM_STEPS <= 0 )); then
    NUM_STEPS=$(( (NUM_EPOCHS * DATASET_SIZE + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))
fi

# ── resume logic ────────────────────────────────────────────────────────────
if [[ -n "${RESUME_CKPT}" ]]; then
    RESUME_PARENT=$(dirname "${RESUME_CKPT}")
    if [[ "$(basename "${RESUME_CKPT}")" == global_step_* ]]; then
        CHECKPOINT_DIR="${RESUME_PARENT}"
    else
        CHECKPOINT_DIR="${RESUME_CKPT}"
    fi
    RESUME_MODE="auto"
else
    CHECKPOINT_DIR="${REPO_ROOT_ABS}/checkpoints/verl_fc_opd_overnight/${RUN_ID}"
    RESUME_MODE="disable"
fi

# ── layout summary ──────────────────────────────────────────────────────────
_p2_label() { _is_power_of_two "$1" && echo "✓ power-of-2" || echo "✗ NON-POWER-OF-2 (risky)"; }
TEACHER_LOG_2=""
if (( NUM_TEACHERS >= 2 )); then
    TEACHER_LOG_2="${REPO_ROOT_ABS}/artifacts/fc_opd/teacher2_${RUN_ID}.log"
fi

echo "══════════════════════════════════════════════════════════════"
echo "  FC-OPD Training — VA-OPD aligned"
echo "  Run ID:             ${RUN_ID}"
echo "  Steps:              ${NUM_STEPS} (${NUM_EPOCHS} epochs × ${DATASET_SIZE}/${TRAIN_BATCH_SIZE} batch)"
echo "  Save freq:          ${SAVE_FREQ}"
echo "  Loss mode:          ${LOSS_MODE}"
echo "  Top-K:              ${TOP_K}"
echo "  Rollout n:          ${ROLLOUT_N}"
echo "  LR:                 2e-6 (VA-OPD: 2e-6)"
echo "──────────────────────────────────────────────────────────────"
echo "  Teacher GPU list:   ${TEACHER_GPU_LIST}"
echo "  Train GPU list:     ${TRAIN_GPU_LIST}"
echo "  Num teachers:       ${NUM_TEACHERS}"
echo "  Train world size:   ${TRAIN_GPUS}  $(_p2_label ${TRAIN_GPUS})"
echo "  Train batch size:   ${TRAIN_BATCH_SIZE}"
echo "  PPO mini-batch:     ${PPO_MINI_BATCH_SIZE}"
echo "  Non-power-2 ok:     ${ALLOW_NONPOWER2}"
echo "──────────────────────────────────────────────────────────────"
echo "  Data:               ${PARQUET}"
echo "  Val  data:          ${VAL_PARQUET}"
echo "  Eval freq:          every ${SAVE_FREQ} steps"
echo "  Rollout dir:        ${ROLLOUT_DIR}"
echo "  Val dir:            ${VAL_DIR}"
echo "  Train log:          ${TRAIN_LOG}"
echo "  Checkpoint:         ${CHECKPOINT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── verify ──────────────────────────────────────────────────────────────────
if ! grep -q "compute_verl_fc_opd_actor_loss" third_party/verl/verl/workers/actor/dp_actor.py; then
    echo "FATAL: actor patch not applied"; exit 1
fi
if [ -z "${CC:-}" ] || ! command -v "${CC}" >/dev/null 2>&1; then
    export CC=/usr/bin/gcc
fi
echo "[OK] CC=${CC}"

# ── cleanup ─────────────────────────────────────────────────────────────────
echo "=== Cleanup ==="
ray stop -f 2>/dev/null || true
EXISTING_TEACHER=$(lsof -ti:${TEACHER_PORT} 2>/dev/null || true)
if [ -n "${EXISTING_TEACHER}" ]; then
    kill -9 ${EXISTING_TEACHER} 2>/dev/null || true; sleep 2
fi
if (( NUM_TEACHERS >= 2 )); then
    EXISTING_TEACHER2=$(lsof -ti:${TEACHER_PORT_2} 2>/dev/null || true)
    if [ -n "${EXISTING_TEACHER2}" ]; then
        kill -9 ${EXISTING_TEACHER2} 2>/dev/null || true; sleep 2
    fi
fi
rm -rf /dev/shm/*vllm* /dev/shm/*psm_* 2>/dev/null || true
sleep 2

# ── 1) Teacher(s) ───────────────────────────────────────────────────────────
# Each teacher gets its own single visible GPU, and always uses --device cuda:0
# inside the container GPU.

echo "=== Teacher #1 (GPU ${TEACHER_GPUS[0]}, port ${TEACHER_PORT}) ==="
CUDA_VISIBLE_DEVICES=${TEACHER_GPUS[0]} \
    ${CONDA_ENV}/bin/python -m dual_track_opd.fc_opd.teacher_service \
    --backend transformers --model "${TEACHER_MODEL}" \
    --port "${TEACHER_PORT}" --top-k "${TOP_K}" --dtype bfloat16 --device "cuda:0" \
    > "${TEACHER_LOG}" 2>&1 &
TEACHER_PID=$!
echo -n "  Waiting ."
for i in $(seq 1 300); do
    if curl -s "http://127.0.0.1:${TEACHER_PORT}/health" >/dev/null 2>&1; then echo " OK"; break; fi
    if ! kill -0 ${TEACHER_PID} 2>/dev/null; then echo " DIED"; tail -20 "${TEACHER_LOG}"; exit 1; fi
    echo -n "."; sleep 1
done

if (( NUM_TEACHERS >= 2 )); then
    echo "=== Teacher #2 (GPU ${TEACHER_GPUS[1]}, port ${TEACHER_PORT_2}) ==="
    CUDA_VISIBLE_DEVICES=${TEACHER_GPUS[1]} \
        ${CONDA_ENV}/bin/python -m dual_track_opd.fc_opd.teacher_service \
        --backend transformers --model "${TEACHER_MODEL}" \
        --port "${TEACHER_PORT_2}" --top-k "${TOP_K}" --dtype bfloat16 --device "cuda:0" \
        > "${TEACHER_LOG_2}" 2>&1 &
    TEACHER_PID_2=$!
    echo -n "  Waiting ."
    for i in $(seq 1 300); do
        if curl -s "http://127.0.0.1:${TEACHER_PORT_2}/health" >/dev/null 2>&1; then echo " OK"; break; fi
        if ! kill -0 ${TEACHER_PID_2} 2>/dev/null; then echo " DIED"; tail -20 "${TEACHER_LOG_2}"; exit 1; fi
        echo -n "."; sleep 1
    done
fi

# ── 2) Ray ──────────────────────────────────────────────────────────────────
echo "=== Ray (${VERL_GPUS} GPUs, devices ${VERL_GPU_LIST}) ==="
export RAY_memory_usage_threshold=0.95
CUDA_VISIBLE_DEVICES=${VERL_GPU_LIST} ray start --head --num-gpus=${VERL_GPUS} --disable-usage-stats
sleep 3

# ── 3) PPO ──────────────────────────────────────────────────────────────────
echo "=== Training (${NUM_STEPS} steps, VA-OPD, rollout n=${ROLLOUT_N}) ==="
set +e
# NCCL stability: NVLink support, disable IB extensions, larger buffers.
export NCCL_NVLS_ENABLE=1
export NCCL_IBEXT_DISABLE=1
export NCCL_BUFFSIZE=4194304
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
# ── test-fix overrides ──────────────────────────────────────────────────────
VERL_EXTRA_ARGS=()
case "${TEST_FIX}" in
    ring)
        echo "=== Test fix: NCCL_ALGO=Ring ==="
        export NCCL_ALGO=Ring
        ;;
    noreshard)
        echo "=== Test fix: reshard_after_forward=false ==="
        VERL_EXTRA_ARGS+=(
            "actor_rollout_ref.actor.fsdp_config.reshard_after_forward=false"
            "actor_rollout_ref.ref.fsdp_config.reshard_after_forward=false"
        )
        ;;
    hsdp3)
        echo "=== Test fix: HSDP with fsdp_size=3 (3-rank FSDP groups) ==="
        if (( VERL_GPUS % 3 != 0 )); then
            echo "FATAL: --test-fix hsdp3 requires the number of training GPUs to be divisible by 3; got ${VERL_GPUS}"
            exit 1
        fi
        VERL_EXTRA_ARGS+=(
            "actor_rollout_ref.actor.fsdp_config.fsdp_size=3"
            "actor_rollout_ref.ref.fsdp_config.fsdp_size=3"
        )
        ;;
    replicate)
        echo "=== Test fix: FSDP2 fsdp_size=1, avoid FSDP parameter allgather ==="
        VERL_EXTRA_ARGS+=(
            "actor_rollout_ref.actor.fsdp_config.fsdp_size=1"
            "actor_rollout_ref.ref.fsdp_config.fsdp_size=1"
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.25"
        )
        ;;
esac
CUDA_VISIBLE_DEVICES=${VERL_GPU_LIST} \
${CONDA_ENV}/bin/python -m verl.trainer.main_ppo \
    --config-path="${VERL_CONFIG_DIR}" \
    --config-name=ppo_trainer \
    "data.train_files=${PARQUET}" \
    "data.val_files=${VAL_PARQUET}" \
    "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "data.max_prompt_length=8192" \
    "data.max_response_length=2048" \
    "data.filter_overlong_prompts=false" \
    "data.truncation=error" \
    "data.image_key=images" \
    "data.dataloader_num_workers=8" \
    "data.custom_cls.path=file://${REPO_ROOT_ABS}/src/dual_track_opd/fc_opd/verl_dataset.py" \
    "data.custom_cls.name=FCOPDDataset" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.nccl_timeout=1800" \
    "actor_rollout_ref.model.use_remove_padding=false" \
    "actor_rollout_ref.model.use_fused_kernels=false" \
    "actor_rollout_ref.model.enable_gradient_checkpointing=true" \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    "actor_rollout_ref.actor.optim.lr=2e-6" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU}" \
    "actor_rollout_ref.actor.use_dynamic_bsz=true" \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192" \
    "actor_rollout_ref.actor.use_kl_loss=false" \
    "actor_rollout_ref.actor.fsdp_config.param_offload=false" \
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false" \
    "actor_rollout_ref.actor.fsdp_config.forward_prefetch=true" \
    "actor_rollout_ref.actor.strategy=fsdp2" \
    "actor_rollout_ref.ref.strategy=fsdp2" \
    "actor_rollout_ref.rollout.name=vllm" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.45" \
    "actor_rollout_ref.rollout.max_model_len=10240" \
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8" \
    "actor_rollout_ref.rollout.agent.num_workers=4" \
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
    "+algorithm.fc_opd.teacher_urls='${TEACHER_URLS}'" \
    "+algorithm.fc_opd.conditions=[full,degraded]" \
    "+algorithm.fc_opd.loss_coef=1.0" \
    "+algorithm.fc_opd.loss_mode=${LOSS_MODE}" \
    "+algorithm.fc_opd.renormalize_topk=true" \
    "+algorithm.fc_opd.include_tail=true" \
    "trainer.total_training_steps=${NUM_STEPS}" \
    "trainer.n_gpus_per_node=${TRAIN_GPUS}" \
    "trainer.nnodes=1" \
    "trainer.critic_warmup=0" \
    "trainer.logger=['console']" \
    "trainer.project_name=fc_opd_overnight" \
    "trainer.experiment_name=${RUN_ID}" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.max_actor_ckpt_to_keep=5" \
    "trainer.test_freq=${SAVE_FREQ}" \
    "trainer.val_before_train=true" \
    "trainer.log_val_generations=10" \
    "trainer.rollout_data_dir=${ROLLOUT_DIR}" \
    "trainer.validation_data_dir=${VAL_DIR}" \
    "trainer.default_local_dir=${CHECKPOINT_DIR}" \
    "trainer.resume_mode=${RESUME_MODE}" \
    "++actor_rollout_ref.rollout.val_kwargs.n=1" \
    "++actor_rollout_ref.rollout.val_kwargs.do_sample=false" \
    "++actor_rollout_ref.rollout.val_kwargs.temperature=0" \
    "${VERL_EXTRA_ARGS[@]}" \
    2>&1 | tee "${TRAIN_LOG}"
VERL_EXIT=$?

# ── cleanup ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Cleanup ==="
ray stop -f 2>/dev/null || true
kill ${TEACHER_PID} 2>/dev/null || true
if (( NUM_TEACHERS >= 2 )); then
    kill ${TEACHER_PID_2} 2>/dev/null || true
fi
sleep 2

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Run:    ${RUN_ID}"
echo "  Steps:  ${NUM_STEPS}"
echo "  Log:    ${TRAIN_LOG}"
echo "  CKPT:   ${CHECKPOINT_DIR}"
echo "  Exit:   ${VERL_EXIT}"
echo "══════════════════════════════════════════════════════════════"

# ── keepalive ────────────────────────────────────────────────────────────────
_KEEPALIVE_SEC=${KEEPALIVE_SEC:-0}
KEEPALIVE_GPUS="${KEEPALIVE_GPUS:-2,3}"
_KEEPALIVE_SCRIPT="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/scripts/busy_keepalive.py"
echo ""
if (( VERL_EXIT == 0 && _KEEPALIVE_SEC > 0 )); then
    echo "=== Keepalive (${_KEEPALIVE_SEC}s, GPUs ${KEEPALIVE_GPUS}) — kill this process when done ==="
    if [ -f "${_KEEPALIVE_SCRIPT}" ]; then
        CUDA_VISIBLE_DEVICES="${KEEPALIVE_GPUS}" \
            timeout "${_KEEPALIVE_SEC}" \
            ${CONDA_ENV}/bin/python -u "${_KEEPALIVE_SCRIPT}"
    else
        echo "[keepalive] busy_keepalive.py not found, falling back to sleep"
        for ((_i = 0; _i < _KEEPALIVE_SEC; _i += 300)); do
            sleep 300
            echo "[keepalive] $(date '+%Y-%m-%d %H:%M:%S') — PID $$ alive (${_i}s elapsed)"
        done
    fi
fi
exit ${VERL_EXIT}

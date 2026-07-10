#!/usr/bin/env bash
# run_gkd_text_smoke.sh — GKD text-only smoke test (Gate 2 / Gate 3)
#
# Runs the GKD recipe with synthetic text-only data. Qwen3-0.6B default.
# No image data, no VA-OPD patch, no Qwen3.5.
# Pure environment validation: GKD/Megatron/Ray/vLLM pipeline stability.
#
# GPU isolation: teacher on TEACHER_GPU, Ray + training on TRAIN_GPUS.
#
# Usage:
#   bash scripts/hpc/run_gkd_text_smoke.sh                        # 10 steps, synthetic data
#   bash scripts/hpc/run_gkd_text_smoke.sh --steps 200            # Gate 3 (200 steps)
#   bash scripts/hpc/run_gkd_text_smoke.sh --teacher-gpu 0 --train-gpus 1,2,3,4
#   bash scripts/hpc/run_gkd_text_smoke.sh --background

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="gkd_smoke_${TIMESTAMP}"

# ── env var overrides ─────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/miniconda3}"
MODEL_ROOT="${MODEL_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models}"
GKD_ENV="${GKD_ENV:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vaopd-gkd-cu128}"
PYTHON="${GKD_ENV}/bin/python"
RAY="${GKD_ENV}/bin/ray"
VERL_GKD_DIR="${REPO_ROOT}/external/verl_gkd/verl"
GKD_RECIPE_DIR="${VERL_GKD_DIR}/recipe/gkd/megatron"

# ── defaults ──────────────────────────────────────────────────────────────
TEACHER_GPU=0
TRAIN_GPU_LIST="1,2,3,4"
NUM_STEPS=10
MODEL_PATH="${MODEL_ROOT}/Qwen3-0.6B"
SYNTHETIC_DATA=true
RUN_BACKGROUND=false

# ── fixed paths ───────────────────────────────────────────────────────────
OUTPUT_DIR="${REPO_ROOT}/runs/gkd_smoke/${RUN_ID}"
DATA_DIR="${OUTPUT_DIR}/data"
TEACHER_PORT=15555
TEACHER_PROXY_PORT=15556

while [[ $# -gt 0 ]]; do
    case "$1" in
        --teacher-gpu)    TEACHER_GPU="${2:?--teacher-gpu needs a value}"; shift 2 ;;
        --train-gpus)     TRAIN_GPU_LIST="${2:?--train-gpus needs a value}"; shift 2 ;;
        --steps)          NUM_STEPS="${2:?--steps needs a value}"; shift 2 ;;
        --model-path)     MODEL_PATH="${2:?--model-path needs a value}"; shift 2 ;;
        --synthetic-data) SYNTHETIC_DATA=true; shift ;;
        --background)     RUN_BACKGROUND=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}" "${DATA_DIR}"

# ── background re-launch ──────────────────────────────────────────────────
if ${RUN_BACKGROUND}; then
    RELAUNCH_ARGS=()
    for arg in "$@"; do
        [[ "$arg" != "--background" ]] || continue
        RELAUNCH_ARGS+=("$arg")
    done
    NOHUP_LOG="${OUTPUT_DIR}/nohup.log"
    echo "Launching background smoke test → ${NOHUP_LOG}"
    nohup bash "$0" \
        --teacher-gpu "${TEACHER_GPU}" \
        --train-gpus "${TRAIN_GPU_LIST}" \
        --steps "${NUM_STEPS}" \
        --model-path "${MODEL_PATH}" \
        "${RELAUNCH_ARGS[@]}" \
        > "${NOHUP_LOG}" 2>&1 &
    disown
    echo "Background PID: $!"
    exit 0
fi

# ── preamble ──────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  GKD Text Smoke Test — Gate 2/3"
echo "  Run ID:         ${RUN_ID}"
echo "  Steps:          ${NUM_STEPS}"
echo "  Model:          ${MODEL_PATH}"
echo "  Teacher GPU:    ${TEACHER_GPU}"
echo "  Train GPUs:     ${TRAIN_GPU_LIST}"
echo "  Output dir:     ${OUTPUT_DIR}"
echo "  GKD recipe:     ${GKD_RECIPE_DIR}"
echo "  Synthetic data: ${SYNTHETIC_DATA}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── verify env ────────────────────────────────────────────────────────────
if [[ ! -x "${PYTHON}" ]]; then
    echo "FATAL: Python not found at ${PYTHON}"
    echo "Run: bash scripts/setup/setup_vaopd_gkd_cu128.sh --execute"
    exit 1
fi

if [[ ! -d "${VERL_GKD_DIR}" ]]; then
    echo "FATAL: verl GKD checkout not found at ${VERL_GKD_DIR}"
    echo "Run: bash scripts/setup/setup_vaopd_gkd_cu128.sh --execute"
    exit 1
fi

if [[ ! -d "${GKD_RECIPE_DIR}" ]]; then
    echo "FATAL: GKD recipe not found at ${GKD_RECIPE_DIR}"
    echo "Did 'git submodule update --init --recursive recipe' succeed?"
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "FATAL: Model not found at ${MODEL_PATH}"
    echo "Download: huggingface-cli download Qwen/Qwen3-0.6B --local-dir ${MODEL_PATH}"
    exit 1
fi

echo "[OK] Environment checks passed"

# B15 compatibility: the server-side B14 bridge calls async
# ServerAdapter.update_weights() from a synchronous Ray worker.  Python 3.12
# does not create an implicit event loop in that thread.  Apply the tracked,
# strict, idempotent compatibility edit before launching expensive GPU work.
"${PYTHON}" "${REPO_ROOT}/scripts/hpc/patch_gkd_b15_event_loop.py" \
    "${GKD_RECIPE_DIR}/megatron_workers.py"

# Ensure recipe.gkd symlinks exist (files were refactored to megatron/ subdir
# but imports still reference recipe.gkd.* — upstream bug at recipe commit ba24641)
_RECIPE_GKD="${VERL_GKD_DIR}/recipe/gkd"
for _link_target in ray_trainer.py teacher_utils.py teacher; do
    _link_path="${_RECIPE_GKD}/${_link_target}"
    if [[ ! -e "${_link_path}" ]]; then
        ln -sf "megatron/${_link_target}" "${_link_path}"
    fi
done

echo ""

# ── quick version check ───────────────────────────────────────────────────
echo "=== Version check ==="
"${PYTHON}" -c "
import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')
import vllm; print(f'vllm={vllm.__version__}')
import ray; print(f'ray={ray.__version__}')
"
echo ""

# ── generate synthetic data (text-only, no images) ────────────────────────
TRAIN_PARQUET="${DATA_DIR}/train.parquet"
VAL_PARQUET="${DATA_DIR}/val.parquet"

if ${SYNTHETIC_DATA}; then
    echo "=== Generating synthetic text-only data ==="
    "${PYTHON}" -c "
import pandas as pd

prompts = [
    'What is 2 + 2?',
    'What is the capital of France?',
    'If a train travels 60 miles in 2 hours, what is its average speed?',
    'Solve: 3x + 5 = 20. What is x?',
    'What is the square root of 144?',
    'How many sides does a hexagon have?',
    'What is 15% of 200?',
    'If a pizza is cut into 8 slices and you eat 3, what fraction remains?',
    'What is the chemical symbol for water?',
    'How many minutes are in 2.5 hours?',
    'What is the area of a square with side length 5?',
    'Solve: 2^3 + 4^2 = ?',
    'What planet is closest to the Sun?',
    'If John has 5 apples and gives 2 to Mary, how many does he have left?',
    'What is the boiling point of water in Celsius?',
    'Convert 1/4 to a decimal.',
    'What is 7 * 8?',
    'How many grams are in a kilogram?',
    'What is the next prime number after 7?',
    'If a book costs \$12 and is on 25% discount, what is the sale price?',
]
data_sources = ['synthetic_math'] * 10 + ['synthetic_trivia'] * 10
N = 128  # ensure enough for 10 steps at batch_size=4
train_rows = []
for i in range(N):
    train_rows.append({
        'prompt': prompts[i % len(prompts)],
        'data_source': data_sources[i % len(data_sources)],
    })
train_df = pd.DataFrame(train_rows)
train_df.to_parquet('${TRAIN_PARQUET}', index=False)
print(f'Train data: {len(train_df)} rows → ${TRAIN_PARQUET}')

# Validation: 16 samples
val_rows = []
for i in range(16):
    val_rows.append({
        'prompt': prompts[i % len(prompts)],
        'data_source': 'synthetic_val',
    })
val_df = pd.DataFrame(val_rows)
val_df.to_parquet('${VAL_PARQUET}', index=False)
print(f'Val data:   {len(val_df)} rows  → ${VAL_PARQUET}')
"
    echo "[OK] Synthetic data generated"
else
    echo "Using existing data at ${DATA_DIR}"
    if [[ ! -f "${TRAIN_PARQUET}" ]] || [[ ! -f "${VAL_PARQUET}" ]]; then
        echo "FATAL: Data not found. Use --synthetic-data or provide existing parquet files."
        exit 1
    fi
fi
echo ""

# ── cleanup ───────────────────────────────────────────────────────────────
echo "=== Cleanup ==="
"${RAY}" stop -f 2>/dev/null || true
ps -ef | grep "python.*proxy.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
ps -ef | grep "python.*worker.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
lsof -ti:${TEACHER_PORT} 2>/dev/null | xargs -r kill -9 2>/dev/null || true
lsof -ti:${TEACHER_PROXY_PORT} 2>/dev/null | xargs -r kill -9 2>/dev/null || true
rm -rf /dev/shm/*vllm* /dev/shm/*psm_* 2>/dev/null || true
sleep 2

# ── 1. Teacher server (isolated GPU) ──────────────────────────────────────
echo "=== Starting GKD teacher server (GPU ${TEACHER_GPU}) ==="

export PROXY_FRONTEND_PORT=${TEACHER_PORT}
export PROXY_BACKEND_PORT=${TEACHER_PROXY_PORT}

cd "${GKD_RECIPE_DIR}/teacher"

# Start proxy (uses CPU, no GPU needed)
# -u = unbuffered stdout so log is visible immediately
CUDA_VISIBLE_DEVICES="" nohup "${PYTHON}" -u proxy.py > "${OUTPUT_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!

# Wait for proxy backend — fatal on timeout
PROXY_READY=false
echo -n "  Waiting for proxy backend..."
for i in $(seq 1 60); do
    if ss -Hltn "sport = :${TEACHER_PROXY_PORT}" 2>/dev/null | grep -q .; then
        echo " OK"
        PROXY_READY=true
        break
    fi
    if ! kill -0 ${PROXY_PID} 2>/dev/null; then
        echo " DIED"
        echo "=== proxy.log (last 30 lines) ==="
        tail -30 "${OUTPUT_DIR}/proxy.log" 2>/dev/null || true
        echo "FATAL: Teacher proxy died during startup"
        exit 1
    fi
    echo -n "."
    sleep 1
done
if ! ${PROXY_READY}; then
    echo " TIMEOUT"
    echo "=== proxy.log (last 30 lines) ==="
    tail -30 "${OUTPUT_DIR}/proxy.log" 2>/dev/null || true
    echo "FATAL: Teacher proxy not ready after 60s"
    exit 1
fi

# Start worker (isolated to TEACHER_GPU)
# -u = unbuffered stdout so log is visible immediately
CUDA_VISIBLE_DEVICES="${TEACHER_GPU}" nohup "${PYTHON}" -u worker.py \
    --backend vllm \
    --tp-size 1 \
    --n-logprobs 32 \
    --ckpt-path "${MODEL_PATH}" \
    > "${OUTPUT_DIR}/worker.log" 2>&1 &
WORKER_PID=$!

# Wait for frontend — fatal on timeout
WORKER_READY=false
echo -n "  Waiting for teacher frontend..."
for i in $(seq 1 180); do
    if ss -Hltn "sport = :${TEACHER_PORT}" 2>/dev/null | grep -q .; then
        echo " OK"
        WORKER_READY=true
        break
    fi
    if ! kill -0 ${WORKER_PID} 2>/dev/null; then
        echo " DIED"
        echo "=== worker.log (last 30 lines) ==="
        tail -30 "${OUTPUT_DIR}/worker.log" 2>/dev/null || true
        echo "FATAL: Teacher worker died during startup"
        exit 1
    fi
    echo -n "."
    sleep 1
done
if ! ${WORKER_READY}; then
    echo " TIMEOUT"
    echo "=== worker.log (last 30 lines) ==="
    tail -30 "${OUTPUT_DIR}/worker.log" 2>/dev/null || true
    echo "FATAL: Teacher worker not ready after 180s"
    exit 1
fi

cd "${REPO_ROOT}"
echo "[OK] Teacher server ready on port ${TEACHER_PORT}"
echo ""

# ── 2. Ray (isolated GPUs) ────────────────────────────────────────────────
echo "=== Starting Ray (GPUs ${TRAIN_GPU_LIST}) ==="
CUDA_VISIBLE_DEVICES="${TRAIN_GPU_LIST}" "${RAY}" start --head --num-gpus="$(echo "${TRAIN_GPU_LIST}" | tr ',' '\n' | wc -l)" --disable-usage-stats
sleep 3
echo "[OK] Ray started"
echo ""

# ── 3. Run GKD text smoke (direct, no ray job submit) ────────────────────
# GKD splits GPUs: actor_pool + rollout_pool = separate Ray resource pools.
# Each pool gets floor(n_gpus / 2), minimum 1.  With 2 train GPUs: 1+1=2 total.
_TRAIN_GPU_COUNT=$(echo "${TRAIN_GPU_LIST}" | tr ',' '\n' | wc -l)
_POOL_GPUS=$(( _TRAIN_GPU_COUNT / 2 ))
if [[ ${_POOL_GPUS} -lt 1 ]]; then
    _POOL_GPUS=1
fi

echo "=== Running GKD text smoke (${NUM_STEPS} steps) ==="
echo "    Train data: ${TRAIN_PARQUET}"
echo "    Val data:   ${VAL_PARQUET}"
echo ""

# Export env vars from runtime_env.yaml (replicated to avoid ray dashboard dependency)
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export CUDA_LAUNCH_BLOCKING="0"
export NVTE_DEBUG="1"
export NVTE_DEBUG_LEVEL="2"
export NVTE_FLASH_ATTN="1"
export NVTE_FUSED_ATTN="0"
export NVTE_UNFUSED_ATTN="0"
export RAY_DEBUG="legacy"
export NCCL_DEBUG="WARN"
export NCCL_DEBUG_FILE="${OUTPUT_DIR}/nccl_debug.log"
export VLLM_USE_V1="1"
export VERL_VLLM_DISTRIBUTED_BACKEND="ray"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPU_LIST}"

TRAIN_LOG="${OUTPUT_DIR}/train.log"
cd "${GKD_RECIPE_DIR}"
export PYTHONPATH="${VERL_GKD_DIR}:${PYTHONPATH:-}"
set +e
"${PYTHON}" -m recipe.gkd.megatron.main_gkd \
    --config-path="${GKD_RECIPE_DIR}/config" \
    --config-name=on_policy_distill_trainer \
    "data.train_files=${TRAIN_PARQUET}" \
    "data.val_files=${VAL_PARQUET}" \
    "data.prompt_key=prompt" \
    "data.train_batch_size=4" \
    "data.max_prompt_length=512" \
    "data.max_response_length=512" \
    "data.filter_overlong_prompts=True" \
    "data.truncation=error" \
    "data.trust_remote_code=True" \
    "+teacher.server_ip=127.0.0.1" \
    "+teacher.server_port=${TEACHER_PORT}" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.model.trust_remote_code=True" \
    "actor_rollout_ref.actor.megatron.sequence_parallel=False" \
    "+actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=False" \
    "actor_rollout_ref.actor.optim.lr=1e-6" \
    "+actor_rollout_ref.actor.ppo_mini_batch_size=4" \
    "+actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
    "+actor_rollout_ref.actor.use_kl_loss=False" \
    "actor_rollout_ref.actor.use_torch_compile=False" \
    "actor_rollout_ref.rollout.mode=async" \
    "actor_rollout_ref.rollout.name=vllm" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.45" \
    "actor_rollout_ref.rollout.temperature=1.0" \
    "actor_rollout_ref.rollout.top_k=32" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
    "actor_rollout_ref.rollout.load_format=auto" \
    "+algorithm.use_kl_in_reward=False" \
    "trainer.logger=['console']" \
    "trainer.project_name=gkd_smoke" \
    "trainer.experiment_name=${RUN_ID}" \
    "trainer.n_gpus_per_node=${_POOL_GPUS}" \
    "trainer.nnodes=1" \
    "rollout.n_gpus_per_node=${_POOL_GPUS}" \
    "rollout.nnodes=1" \
    "trainer.save_freq=-1" \
    "trainer.test_freq=5" \
    "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1" \
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1" \
    "actor_rollout_ref.actor.megatron.expert_model_parallel_size=1" \
    "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1" \
    "trainer.val_before_train=False" \
    "trainer.total_training_steps=${NUM_STEPS}" \
    "trainer.total_epochs=1" \
    > "${TRAIN_LOG}" 2>&1
VERL_EXIT=$?
set -e
cd "${REPO_ROOT}"


# ── cleanup ───────────────────────────────────────────────────────────────
echo ""
echo "=== Cleanup ==="
"${RAY}" stop -f 2>/dev/null || true
kill ${PROXY_PID} 2>/dev/null || true
kill ${WORKER_PID} 2>/dev/null || true
sleep 2

# ── validate smoke result ─────────────────────────────────────────────────
echo ""
echo "=== Smoke Validation ==="

EXIT_OK=false
STEPS_OK=false

if [[ ${VERL_EXIT} -eq 0 ]]; then
    echo "  Exit code: 0 ✓"
    EXIT_OK=true
else
    echo "  Exit code: ${VERL_EXIT} ✗"
fi

# Check for training steps in log
_STEP_COUNT=$(grep -c 'global_step\|step.*/' "${TRAIN_LOG}" 2>/dev/null) || _STEP_COUNT=0
if [[ "${_STEP_COUNT}" -ge $(( NUM_STEPS / 2 )) ]]; then
    echo "  Steps found in log: ${_STEP_COUNT} (≥ ${NUM_STEPS}/2) ✓"
    STEPS_OK=true
else
    echo "  Steps found in log: ${_STEP_COUNT} (need ≥ $(( NUM_STEPS / 2 ))) ✗"
fi

# Check for loss
if grep -q 'loss\|kl_loss\|distill_loss' "${TRAIN_LOG}" 2>/dev/null; then
    echo "  Loss/kl_loss found in log ✓"
else
    echo "  WARNING: No loss/kl_loss found in log"
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Run ID:     ${RUN_ID}"
echo "  Steps:      ${NUM_STEPS}"
echo "  Ray exit:   ${VERL_EXIT}"
echo "  Log dir:    ${OUTPUT_DIR}"
echo "  Train log:  ${TRAIN_LOG}"
echo "══════════════════════════════════════════════════════════════"

if ${EXIT_OK} && ${STEPS_OK}; then
    echo "SMOKE PASSED ✓"
    exit 0
else
    echo "SMOKE FAILED ✗"
    echo "=== Last 50 lines of train log ==="
    tail -50 "${TRAIN_LOG}" 2>/dev/null || true
    exit 1
fi

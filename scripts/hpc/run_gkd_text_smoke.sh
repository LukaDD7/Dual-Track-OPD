#!/usr/bin/env bash
# run_gkd_text_smoke.sh — GKD text-only smoke test (Gate 2)
#
# Runs the GKD recipe's test_qwen.sh pattern with Qwen3-0.6B on text-only
# GSM8K-style data.  10 steps.  No image data, no VA-OPD patch, no Qwen3.5.
# Pure environment validation: GKD/Megatron/Ray/vLLM pipeline stability.
#
# Usage:
#   bash scripts/hpc/run_gkd_text_smoke.sh              # 10 steps default
#   bash scripts/hpc/run_gkd_text_smoke.sh --steps 200  # Gate 3 (200 steps)
#   bash scripts/hpc/run_gkd_text_smoke.sh --background  # nohup

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="gkd_smoke_${TIMESTAMP}"

CONDA_BASE="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/miniconda3"
GKD_ENV="${CONDA_BASE}/envs/vaopd-gkd-cu128"
PYTHON="${GKD_ENV}/bin/python"
VERL_GKD_DIR="${REPO_ROOT}/external/verl_gkd/verl"
GKD_RECIPE_DIR="${VERL_GKD_DIR}/recipe/gkd/megatron"

# ── fixed paths ───────────────────────────────────────────────────────────
MODEL_PATH="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-0.6B"
TRAIN_DATA="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/geometry3k_full/train.parquet"
VAL_DATA="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/geometry3k_full/val200.parquet"  # FIXME: use text-only data
OUTPUT_DIR="${REPO_ROOT}/runs/gkd_smoke/${RUN_ID}"
TEACHER_PORT=15555
TEACHER_PROXY_PORT=15556

# ── training params ───────────────────────────────────────────────────────
NUM_STEPS=10
RUN_BACKGROUND=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)       NUM_STEPS="${2:?--steps needs a value}"; shift 2 ;;
        --background)  RUN_BACKGROUND=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

# ── background re-launch ──────────────────────────────────────────────────
if ${RUN_BACKGROUND}; then
    RELAUNCH_ARGS=()
    for arg in "$@"; do
        [[ "$arg" != "--background" ]] || continue
        RELAUNCH_ARGS+=("$arg")
    done
    NOHUP_LOG="${OUTPUT_DIR}/nohup.log"
    echo "Launching background smoke test → ${NOHUP_LOG}"
    nohup bash "$0" --steps "${NUM_STEPS}" "${RELAUNCH_ARGS[@]}" > "${NOHUP_LOG}" 2>&1 &
    disown
    echo "Background PID: $!"
    exit 0
fi

# ── preamble ──────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  GKD Text Smoke Test — Gate 2"
echo "  Run ID:       ${RUN_ID}"
echo "  Steps:        ${NUM_STEPS}"
echo "  Model:        ${MODEL_PATH}"
echo "  Train data:   ${TRAIN_DATA}"
echo "  Val data:     ${VAL_DATA}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  GKD recipe:   ${GKD_RECIPE_DIR}"
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
echo ""

# ── quick version check ───────────────────────────────────────────────────
echo "=== Version check ==="
"${PYTHON}" -c "
import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')
import vllm; print(f'vllm={vllm.__version__}')
import ray; print(f'ray={ray.__version__}')
"
echo ""

# ── cleanup ───────────────────────────────────────────────────────────────
echo "=== Cleanup ==="
ray stop -f 2>/dev/null || true
ps -ef | grep "python.*proxy.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
ps -ef | grep "python.*worker.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
lsof -ti:${TEACHER_PORT} 2>/dev/null | xargs -r kill -9 2>/dev/null || true
lsof -ti:${TEACHER_PROXY_PORT} 2>/dev/null | xargs -r kill -9 2>/dev/null || true
rm -rf /dev/shm/*vllm* /dev/shm/*psm_* 2>/dev/null || true
sleep 2

# ── 1. Teacher server ─────────────────────────────────────────────────────
echo "=== Starting GKD teacher server ==="
TEACHER_LOG="${OUTPUT_DIR}/teacher.log"

export PROXY_FRONTEND_PORT=${TEACHER_PORT}
export PROXY_BACKEND_PORT=${TEACHER_PROXY_PORT}

cd "${GKD_RECIPE_DIR}/teacher"

# Start proxy
nohup "${PYTHON}" proxy.py > "${OUTPUT_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!

# Wait for proxy backend to be ready
echo -n "  Waiting for proxy backend..."
for i in $(seq 1 60); do
    if "${PYTHON}" -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1)
try:
    s.connect(('127.0.0.1', ${TEACHER_PROXY_PORT}))
    s.close()
    exit(0)
except:
    exit(1)
" 2>/dev/null; then
        echo " OK"
        break
    fi
    if ! kill -0 ${PROXY_PID} 2>/dev/null; then
        echo " DIED"
        cat "${OUTPUT_DIR}/proxy.log"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# Start worker (vLLM backend)
nohup "${PYTHON}" worker.py \
    --backend vllm \
    --tp-size 1 \
    --n-logprobs 32 \
    --ckpt-path "${MODEL_PATH}" \
    > "${OUTPUT_DIR}/worker.log" 2>&1 &
WORKER_PID=$!

# Wait for frontend to be ready
echo -n "  Waiting for teacher frontend..."
for i in $(seq 1 120); do
    if "${PYTHON}" -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1)
try:
    s.connect(('127.0.0.1', ${TEACHER_PORT}))
    s.close()
    exit(0)
except:
    exit(1)
" 2>/dev/null; then
        echo " OK"
        break
    fi
    if ! kill -0 ${WORKER_PID} 2>/dev/null; then
        echo " DIED"
        tail -30 "${OUTPUT_DIR}/worker.log"
        exit 1
    fi
    echo -n "."
    sleep 1
done

cd "${REPO_ROOT}"
echo "[OK] Teacher server ready on port ${TEACHER_PORT}"
echo ""

# ── 2. Ray ────────────────────────────────────────────────────────────────
echo "=== Starting Ray ==="
export RAY_memory_usage_threshold=0.95
ray start --head --num-gpus=4 --disable-usage-stats
sleep 3
echo "[OK] Ray started"
echo ""

# ── 3. Run GKD text smoke ─────────────────────────────────────────────────
echo "=== Running GKD text smoke (${NUM_STEPS} steps) ==="

# Use ray job submit pattern from test_qwen.sh
RUNTIME_ENV="${GKD_RECIPE_DIR}/config/runtime_env.yaml"

# Override NCCL_DEBUG_FILE to our output dir
export NCCL_DEBUG_FILE="${OUTPUT_DIR}/nccl_debug.log"
export NCCL_DEBUG="WARN"

set +e
ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${GKD_RECIPE_DIR}" \
    -- "${PYTHON}" -m recipe.gkd.megatron.main_gkd \
    --config-path="${GKD_RECIPE_DIR}/config" \
    --config-name=on_policy_distill_trainer \
    "data.train_files=${TRAIN_DATA}" \
    "data.val_files=${VAL_DATA}" \
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
    "trainer.n_gpus_per_node=4" \
    "trainer.nnodes=1" \
    "rollout.n_gpus_per_node=4" \
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
    > "${OUTPUT_DIR}/train.log" 2>&1
VERL_EXIT=$?
set -e

# ── cleanup ───────────────────────────────────────────────────────────────
echo ""
echo "=== Cleanup ==="
ray stop -f 2>/dev/null || true
kill ${PROXY_PID} 2>/dev/null || true
kill ${WORKER_PID} 2>/dev/null || true
sleep 2

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Run ID:   ${RUN_ID}"
echo "  Steps:    ${NUM_STEPS}"
echo "  Exit:     ${VERL_EXIT}"
echo "  Log dir:  ${OUTPUT_DIR}"
echo "  Train log: ${OUTPUT_DIR}/train.log"
echo "══════════════════════════════════════════════════════════════"

exit ${VERL_EXIT}

#!/usr/bin/env bash
# ===========================================================================
# Step 1 — Frozen-Policy Support Diagnostic
#
# Runs the support-aware diagnostic on a single GPU with an already-running
# teacher service on another GPU.
#
# Usage:
#   # Terminal 1 — Start teacher
#   FC_OPD_TEACHER_MODEL="${DTOPD_MODEL_ROOT}/Qwen3-VL-32B-Instruct" \
#   CUDA_VISIBLE_DEVICES=0 bash scripts/hpc/start_fc_teacher.sh
#
#   # Terminal 2 — Run diagnostic
#   CUDA_VISIBLE_DEVICES=1 \
#   bash scripts/hpc/run_support_aware_diagnostic.sh \
#       --config configs/experiment/support_aware_geometry3k_pilot.yaml \
#       --mode smoke
#
#   CUDA_VISIBLE_DEVICES=1 \
#   bash scripts/hpc/run_support_aware_diagnostic.sh \
#       --config configs/experiment/support_aware_geometry3k_pilot.yaml \
#       --mode full
# ===========================================================================

set -euo pipefail

# Resolve project root
DTOPD_ROOT="${DTOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${DTOPD_ROOT}"

# Default environment variables
DTOPD_MODEL_ROOT="${DTOPD_MODEL_ROOT:-${DTOPD_ROOT}/models}"
DTOPD_OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${DTOPD_ROOT}/fc-opd-storage/outputs}"

export DTOPD_ROOT DTOPD_MODEL_ROOT DTOPD_OUTPUT_ROOT

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CONFIG=""
MODE="smoke"
TEACHER_URL="http://127.0.0.1:18080"
STUDENT_MODEL=""
DATASET=""
OUTPUT_ROOT=""
SEED=""
DEVICE="cuda"
DTYPE="bfloat16"
PREFLIGHT_ONLY=false
RESUME=""
PROMPT_START=""
PROMPT_END=""
NUM_PROMPTS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"; shift 2 ;;
        --mode)
            MODE="$2"; shift 2 ;;
        --teacher-url)
            TEACHER_URL="$2"; shift 2 ;;
        --student-model)
            STUDENT_MODEL="$2"; shift 2 ;;
        --dataset)
            DATASET="$2"; shift 2 ;;
        --output-root)
            OUTPUT_ROOT="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --dtype)
            DTYPE="$2"; shift 2 ;;
        --resume)
            RESUME="$2"; shift 2 ;;
        --prompt-start)
            PROMPT_START="$2"; shift 2 ;;
        --prompt-end)
            PROMPT_END="$2"; shift 2 ;;
        --num-prompts)
            NUM_PROMPTS="$2"; shift 2 ;;
        --preflight-only)
            PREFLIGHT_ONLY=true; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
    esac
done

if [[ -z "${CONFIG}" ]]; then
    echo "ERROR: --config is required" >&2
    exit 1
fi

# Resolve config path
CONFIG_PATH="${DTOPD_ROOT}/${CONFIG}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    CONFIG_PATH="${CONFIG}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi
echo "Config: ${CONFIG_PATH}"

# ---------------------------------------------------------------------------
# Defaults from environment / config
# ---------------------------------------------------------------------------
STUDENT_MODEL="${STUDENT_MODEL:-${DTOPD_MODEL_ROOT}/Qwen3-VL-4B-Instruct}"
DATASET="${DATASET:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/geometry3k_full/train.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DTOPD_OUTPUT_ROOT}}"

echo "=== Support-Aware Diagnostic ==="
echo "Mode:          ${MODE}"
echo "Student:       ${STUDENT_MODEL}"
echo "Teacher URL:   ${TEACHER_URL}"
echo "Dataset:       ${DATASET}"
echo "Output root:   ${OUTPUT_ROOT}"
echo "Device:        ${DEVICE} (${DTYPE})"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

# GPU check
if command -v nvidia-smi &>/dev/null; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
        echo "Requested GPUs: ${GPU_COUNT}"
    else
        echo "WARNING: CUDA_VISIBLE_DEVICES is not set"
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
fi

# Student model check
if [[ ! -d "${STUDENT_MODEL}" ]]; then
    echo "FATAL: Student model not found: ${STUDENT_MODEL}" >&2
    exit 1
fi
echo "✓ Student model exists"

# Dataset check
if [[ ! -f "${DATASET}" ]]; then
    echo "FATAL: Dataset not found: ${DATASET}" >&2
    exit 1
fi
echo "✓ Dataset exists"

# Teacher health check
echo -n "Checking teacher at ${TEACHER_URL} ... "
if curl -s --max-time 5 "${TEACHER_URL}/health" | grep -q '"ok"'; then
    echo "OK"
else
    echo "UNREACHABLE"
    echo ""
    echo "FATAL: Teacher service is not running at ${TEACHER_URL}"
    echo "Start it in another terminal:"
    echo ""
    echo "  FC_OPD_TEACHER_MODEL=\"\${DTOPD_MODEL_ROOT}/Qwen3-VL-32B-Instruct\" \\"
    echo "  CUDA_VISIBLE_DEVICES=0 bash scripts/hpc/start_fc_teacher.sh"
    echo ""
    exit 1
fi

# Python environment
# Prefer $DTOPD_PYTHON, then any `python` that already has torch, then the
# vision-opd-cu128 env python on this HPC.  Avoids "No module named torch"
# when the calling shell's default python is not the research env.
resolve_python() {
    if [[ -n "${DTOPD_PYTHON:-}" && -x "${DTOPD_PYTHON}" ]]; then
        echo "${DTOPD_PYTHON}"
        return
    fi
    if command -v python >/dev/null 2>&1 && python -c 'import torch' >/dev/null 2>&1; then
        echo "python"
        return
    fi
    local envs_root
    envs_root="$(cd "$(dirname "${DTOPD_ROOT}")/envs" 2>/dev/null && pwd || true)"
    for candidate in "${envs_root}/vision-opd-cu128/bin/python" \
                     "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python"; do
        if [[ -x "${candidate}" ]]; then
            echo "${candidate}"
            return
        fi
    done
    echo ""
}

PYTHON_BIN="$(resolve_python)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "FATAL: no Python with torch found." >&2
    echo "       Activate the research env (conda activate vision-opd-cu128)" >&2
    echo "       or set DTOPD_PYTHON to an interpreter that has torch." >&2
    exit 1
fi

"${PYTHON_BIN}" -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || {
    echo "FATAL: Python/torch not available (PYTHON_BIN=${PYTHON_BIN})" >&2
    exit 1
}

if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
    echo ""
    echo "Pre-flight checks passed. Exiting (--preflight-only)."
    exit 0
fi

# ---------------------------------------------------------------------------
# Run diagnostic
# ---------------------------------------------------------------------------
echo ""
echo "=== Running diagnostic (mode=${MODE}) ==="

CMD=(
    "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.diagnostic
    --config "${CONFIG_PATH}"
    --mode "${MODE}"
    --student-model-path "${STUDENT_MODEL}"
    --teacher-url "${TEACHER_URL}"
    --dataset "${DATASET}"
    --output-root "${OUTPUT_ROOT}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
)

if [[ -n "${SEED}" ]]; then
    CMD+=(--seed "${SEED}")
fi

if [[ -n "${RESUME}" ]]; then
    CMD+=(--resume "${RESUME}")
fi

if [[ -n "${PROMPT_START}" ]]; then
    CMD+=(--prompt-start "${PROMPT_START}")
fi

if [[ -n "${PROMPT_END}" ]]; then
    CMD+=(--prompt-end "${PROMPT_END}")
fi

if [[ -n "${NUM_PROMPTS}" ]]; then
    CMD+=(--num-prompts "${NUM_PROMPTS}")
fi

echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}"
EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo ""
    echo "=== Diagnostic PASSED — all gates green ==="
else
    echo ""
    echo "=== Diagnostic FAILED (exit code ${EXIT_CODE}) — check summary.json ==="
fi

exit ${EXIT_CODE}

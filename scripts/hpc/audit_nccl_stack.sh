#!/usr/bin/env bash
# audit_nccl_stack.sh — NCCL/CUDA/torch/vllm environment audit
#
# Prints GPU driver info, CUDA toolkit version, PyTorch CUDA version,
# and all key package versions.  Checks for cu129/cu130 contamination.
# Optionally runs nccl-tests all_reduce_perf if available.
#
# Must be run on a GPU instance (nvidia-smi available).
#
# Usage:
#   bash scripts/hpc/audit_nccl_stack.sh

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/runs/env_audit/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_FILE="${OUTPUT_DIR}/audit_${TIMESTAMP}.log"

CONDA_BASE="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/miniconda3"
GKD_ENV="${CONDA_BASE}/envs/vaopd-gkd-cu128"
PYTHON="${GKD_ENV}/bin/python"

log() {
    echo "$@" | tee -a "${OUTPUT_FILE}"
}

fatal() {
    log "FATAL: $*"
    exit 1
}

log "════════════════════════════════════════════════════════"
log "  NCCL Stack Audit"
log "  Timestamp: ${TIMESTAMP}"
log "  Output:    ${OUTPUT_FILE}"
log "════════════════════════════════════════════════════════"
log ""

# ── 1. nvidia-smi ─────────────────────────────────────────────────────────
log "=== 1. nvidia-smi ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi 2>&1 | tee -a "${OUTPUT_FILE}"
else
    log "WARNING: nvidia-smi not found — are you on a GPU instance?"
fi
log ""

# ── 2. nvcc ───────────────────────────────────────────────────────────────
log "=== 2. nvcc --version ==="
if command -v nvcc &>/dev/null; then
    nvcc --version 2>&1 | tee -a "${OUTPUT_FILE}"
else
    log "(nvcc not found on PATH — may be absent if using conda cudatoolkit only)"
fi
log ""

# ── 3. Python / torch / vllm / ray versions ───────────────────────────────
log "=== 3. Python / torch / vllm / ray versions ==="
if [[ ! -x "${PYTHON}" ]]; then
    fatal "Python not found at ${PYTHON}. Run setup_vaopd_gkd_cu128.sh --execute first."
fi

"${PYTHON}" -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'torch: {torch.__version__}')
    print(f'torch.version.cuda: {torch.version.cuda}')
except ImportError:
    print('torch: NOT INSTALLED')

try:
    import vllm
    print(f'vllm: {vllm.__version__}')
except ImportError:
    print('vllm: NOT INSTALLED')

try:
    import ray
    print(f'ray: {ray.__version__}')
except ImportError:
    print('ray: NOT INSTALLED')

try:
    import flash_attn
    print(f'flash_attn: {flash_attn.__version__}')
except ImportError:
    print('flash_attn: NOT INSTALLED')

try:
    import transformer_engine
    print(f'transformer_engine: {transformer_engine.__version__}')
except ImportError:
    print('transformer_engine: NOT INSTALLED')

try:
    import megatron
    print(f'megatron: found ({megatron.__file__})')
except ImportError:
    print('megatron: NOT INSTALLED')
" 2>&1 | tee -a "${OUTPUT_FILE}"
log ""

# ── 4. pip show key packages ──────────────────────────────────────────────
log "=== 4. pip show (torch, vllm, ray, nvidia-nccl-cu12, nvidia-cudnn-cu12) ==="
for pkg in torch vllm ray nvidia-nccl-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12; do
    log "--- ${pkg} ---"
    "${PYTHON}" -m pip show "${pkg}" 2>&1 | tee -a "${OUTPUT_FILE}" || log "(not installed)"
done
log ""

# ── 5. pip freeze key packages ────────────────────────────────────────────
log "=== 5. pip freeze (key packages) ==="
"${PYTHON}" -m pip freeze 2>/dev/null | grep -E 'torch|vllm|ray|flash|nccl|nvidia-cudnn|transformer-engine|megatron' | tee -a "${OUTPUT_FILE}"
log ""

# ── 6. cu129 / cu130 scan ─────────────────────────────────────────────────
log "=== 6. cu129 / cu130 contamination scan ==="
CONTAMINATION=$("${PYTHON}" -m pip freeze 2>/dev/null | grep -E 'cu129|cu130' || true)
if [[ -n "${CONTAMINATION}" ]]; then
    log "CONTAMINATION FOUND:"
    log "${CONTAMINATION}"
    fatal "cu129/cu130 wheels detected!  These are incompatible with driver 570.x (CUDA 12.8 max)."
else
    log "OK — no cu129/cu130 wheels found."
fi
log ""

# ── 7. nccl-tests (optional) ──────────────────────────────────────────────
log "=== 7. nccl-tests all_reduce_perf ==="
NCCL_TESTS_DIR="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/nccl-tests"
if [[ -f "${NCCL_TESTS_DIR}/build/all_reduce_perf" ]]; then
    log "Running all_reduce_perf -b 8M -e 2G -f 2 -g 1 (single GPU health check)..."
    NCCL_DEBUG=INFO "${NCCL_TESTS_DIR}/build/all_reduce_perf" -b 8M -e 2G -f 2 -g 1 2>&1 | tee -a "${OUTPUT_FILE}"
    log ""
    log "To test multi-GPU (4 GPUs), run on GPU instance:"
    log "  NCCL_DEBUG=INFO ${NCCL_TESTS_DIR}/build/all_reduce_perf -b 8M -e 2G -f 2 -g 4"
else
    log "nccl-tests not found at ${NCCL_TESTS_DIR}/build/all_reduce_perf"
    log "To install:"
    log "  git clone https://github.com/NVIDIA/nccl-tests.git ${NCCL_TESTS_DIR}"
    log "  cd ${NCCL_TESTS_DIR} && make MPI=0 CUDA_HOME=/usr/local/cuda"
fi
log ""

# ── done ──────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════════════════"
log "  Audit complete."
log "  Full log: ${OUTPUT_FILE}"
log "════════════════════════════════════════════════════════"

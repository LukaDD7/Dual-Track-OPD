#!/usr/bin/env bash
# audit_nccl_stack.sh — NCCL/CUDA/torch/vllm environment audit
#
# Prints GPU driver info, CUDA toolkit version, PyTorch CUDA version,
# and all key package versions.  Checks for cu129/cu130 contamination.
# Optionally runs nccl-tests all_reduce_perf.
#
# Usage:
#   bash scripts/hpc/audit_nccl_stack.sh                          # full audit (needs GPU)
#   bash scripts/hpc/audit_nccl_stack.sh --cpu-ok                 # skip nvidia-smi requirement
#   bash scripts/hpc/audit_nccl_stack.sh --run-nccl-tests --gpus 4  # run multi-GPU nccl-tests
#   bash scripts/hpc/audit_nccl_stack.sh --run-vllm-smoke --vllm-smoke-model /path/to/model  # GPU vLLM runtime smoke
#   bash scripts/hpc/audit_nccl_stack.sh --env-path /path/to/env  # use specific env

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/runs/env_audit/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
AUDIT_LOG="${OUTPUT_DIR}/audit.log"

GKD_ENV_DEFAULT="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vaopd-gkd-cu128"

# ── parse args ─────────────────────────────────────────────────────────────
CPU_OK=false
RUN_NCCL_TESTS=false
RUN_VLLM_SMOKE=false
RUN_TE_SMOKE=false
NCCL_GPUS=4
VLLM_SMOKE_MODEL=""
ENV_PATH="${GKD_ENV_DEFAULT}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-ok)            CPU_OK=true; shift ;;
        --run-nccl-tests)    RUN_NCCL_TESTS=true; shift ;;
        --run-vllm-smoke)    RUN_VLLM_SMOKE=true; shift ;;
        --run-te-smoke)      RUN_TE_SMOKE=true; shift ;;
        --vllm-smoke-model)  VLLM_SMOKE_MODEL="${2:?--vllm-smoke-model needs a path}"; shift 2 ;;
        --gpus)              NCCL_GPUS="${2:?--gpus needs a value}"; shift 2 ;;
        --env-path)          ENV_PATH="${2:?--env-path needs a value}"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PYTHON="${ENV_PATH}/bin/python"

log() {
    echo "$@" | tee -a "${AUDIT_LOG}"
}

fatal() {
    log "FATAL: $*"
    exit 1
}

warn() {
    log "WARNING: $*"
}

log "════════════════════════════════════════════════════════"
log "  NCCL Stack Audit"
log "  Timestamp:  ${TIMESTAMP}"
log "  Output:     ${AUDIT_LOG}"
log "  CPU-OK:     ${CPU_OK}"
log "  NCCL-Tests: ${RUN_NCCL_TESTS}"
log "  NCCL_GPUS:  ${NCCL_GPUS}"
log "  Env path:   ${ENV_PATH}"
log "════════════════════════════════════════════════════════"
log ""

# ── 1. nvidia-smi ─────────────────────────────────────────────────────────
log "=== 1. nvidia-smi ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi 2>&1 | tee -a "${AUDIT_LOG}"
else
    if ${CPU_OK}; then
        warn "nvidia-smi not found — running on CPU node (--cpu-ok)"
    else
        fatal "nvidia-smi not found. Are you on a GPU instance? Use --cpu-ok to skip."
    fi
fi
log ""

# ── 2. nvcc ───────────────────────────────────────────────────────────────
log "=== 2. nvcc --version ==="
if command -v nvcc &>/dev/null; then
    nvcc --version 2>&1 | tee -a "${AUDIT_LOG}"
else
    log "(nvcc not found on PATH)"
fi
log ""

# ── 3. Python / torch / vllm / ray versions ───────────────────────────────
log "=== 3. Python / torch / vllm / ray versions ==="
if [[ ! -x "${PYTHON}" ]]; then
    fatal "Python not found at ${PYTHON}. Run setup_vaopd_gkd_cu128.sh --execute first, or pass --env-path."
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
" 2>&1 | tee -a "${AUDIT_LOG}"
log ""

# ── 4. pip show key packages ──────────────────────────────────────────────
log "=== 4. pip show (torch, vllm, ray, nvidia-nccl-cu12, nvidia-cudnn-cu12) ==="
for pkg in torch vllm ray nvidia-nccl-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12; do
    log "--- ${pkg} ---"
    "${PYTHON}" -m pip show "${pkg}" 2>&1 | tee -a "${AUDIT_LOG}" || log "(not installed)"
done
log ""

# ── 5. pip freeze key packages (no pipefail exit on empty grep) ───────────
log "=== 5. pip freeze (key packages) ==="
"${PYTHON}" -m pip freeze 2>/dev/null | grep -E 'torch|vllm|ray|flash|nccl|nvidia-cudnn|transformer-engine|megatron' || true
log ""

# ── 6. cu129 / cu130 scan ─────────────────────────────────────────────────
log "=== 6. cu129 / cu130 contamination scan ==="
CONTAMINATION=$("${PYTHON}" -m pip freeze 2>/dev/null | grep -E 'cu129|cu130' || true)
if [[ -n "${CONTAMINATION}" ]]; then
    log "CONTAMINATION FOUND:"
    log "${CONTAMINATION}"
    fatal "cu129/cu130 wheels detected! These are incompatible with driver 570.x (CUDA 12.8 max)."
else
    log "OK — no cu129/cu130 wheels found."
fi
log ""

# ── 7. nccl-tests ─────────────────────────────────────────────────────────
log "=== 7. nccl-tests ==="
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/nccl-tests}"
ALL_REDUCE_BIN="${NCCL_TESTS_DIR}/build/all_reduce_perf"

if ! ${RUN_NCCL_TESTS}; then
    log "Skipping nccl-tests (use --run-nccl-tests to enable)."
    if [[ -f "${ALL_REDUCE_BIN}" ]]; then
        log "all_reduce_perf found at ${ALL_REDUCE_BIN}"
        log "To run Gate 1: bash scripts/hpc/audit_nccl_stack.sh --run-nccl-tests --gpus 4"
    else
        log "nccl-tests not found at ${ALL_REDUCE_BIN}"
        log "To install: git clone https://github.com/NVIDIA/nccl-tests.git ${NCCL_TESTS_DIR}"
        log "             cd ${NCCL_TESTS_DIR} && make MPI=0 CUDA_HOME=/usr/local/cuda"
    fi
else
    if [[ ! -f "${ALL_REDUCE_BIN}" ]]; then
        fatal "all_reduce_perf not found at ${ALL_REDUCE_BIN}. Install nccl-tests first."
    fi

    if ! command -v nvidia-smi &>/dev/null; then
        fatal "Cannot run nccl-tests without nvidia-smi (GPU required)."
    fi

    log "Running all_reduce_perf -b 8M -e 2G -f 2 -g ${NCCL_GPUS} ..."
    log ""
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH
    export NCCL_CUMEM_ENABLE=0
    set +e
    "${ALL_REDUCE_BIN}" -b 8M -e 2G -f 2 -g "${NCCL_GPUS}" 2>&1 | tee -a "${AUDIT_LOG}"
    NCCL_EXIT=$?
    set -e
    log ""
    if [[ ${NCCL_EXIT} -eq 0 ]]; then
        log "nccl-tests PASSED (exit 0)"
    else
        fatal "nccl-tests FAILED (exit ${NCCL_EXIT})"
    fi
fi
log ""

# ── 8. vLLM GPU runtime smoke ─────────────────────────────────────────────
log "=== 8. vLLM GPU runtime smoke ==="
if ! ${RUN_VLLM_SMOKE}; then
    log "Skipping vLLM GPU smoke (use --run-vllm-smoke --vllm-smoke-model /path/to/model)."
    log "This smoke validates that PyPI vLLM works at runtime with torch cu128 on real GPU."
else
    if ! command -v nvidia-smi &>/dev/null; then
        fatal "Cannot run vLLM GPU smoke without nvidia-smi (GPU required)."
    fi
    if [[ -z "${VLLM_SMOKE_MODEL}" ]]; then
        fatal "--vllm-smoke-model is required (e.g. /path/to/Qwen3-0.6B)."
    fi
    if [[ ! -d "${VLLM_SMOKE_MODEL}" ]]; then
        fatal "Model not found at ${VLLM_SMOKE_MODEL}."
    fi

    log "Running vLLM GPU runtime smoke with model ${VLLM_SMOKE_MODEL}..."
    log ""
    set +e
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" - "${VLLM_SMOKE_MODEL}" <<'PY' 2>&1 | tee -a "${AUDIT_LOG}"
import sys, os
model_id = sys.argv[1]
print(f"Model: {model_id}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  gpu_count: {torch.cuda.device_count()}")

from vllm import LLM, SamplingParams
print("vLLM import OK, creating LLM instance...")

llm = LLM(
    model=model_id,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.20,
    max_model_len=512,
    enforce_eager=True,
)
print("LLM instance created, running inference...")

out = llm.generate(["Hello, world!"], SamplingParams(temperature=0, max_tokens=8))
print(f"Inference output: {out[0].outputs[0].text}")

del llm
torch.cuda.empty_cache()
print("vLLM GPU smoke PASS")
PY
    VLLM_EXIT=$?
    set -e
    log ""
    if [[ ${VLLM_EXIT} -eq 0 ]]; then
        log "vLLM GPU smoke PASSED"
    else
        log "vLLM GPU smoke FAILED (exit ${VLLM_EXIT})"
        log ""
        log "=== Collecting diagnostics ==="
        "${PYTHON}" -m vllm.collect_env 2>&1 | tee -a "${AUDIT_LOG}" || true
        log ""
        log "=== pip freeze ==="
        "${PYTHON}" -m pip freeze 2>/dev/null | grep -E 'torch|vllm|nvidia-cu|nvidia-nccl' | tee -a "${AUDIT_LOG}" || true
        log ""
        fatal "vLLM GPU runtime smoke failed. Do NOT switch vLLM version yet — collect full diagnostics first."
    fi
fi
log ""

# ── 9. TE GPU runtime smoke ──────────────────────────────────────────────
log "=== 9. TE GPU runtime smoke ==="
if ! ${RUN_TE_SMOKE}; then
    log "Skipping TE GPU smoke (use --run-te-smoke)."
    log "Validates TransformerEngine GPU ops (Linear forward+backward) on real GPU."
else
    if ! command -v nvidia-smi &>/dev/null; then
        fatal "Cannot run TE GPU smoke without nvidia-smi (GPU required)."
    fi
    log "Running TE GPU smoke on GPU 0..."
    set +e
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" - 2>&1 <<'PY' | tee -a "${AUDIT_LOG}"
import torch
import transformer_engine.pytorch as te

print(f"cuda available: {torch.cuda.is_available()}")
print(f"device: {torch.cuda.get_device_name(0)}")
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}")

layer = te.Linear(16, 16).cuda().to(dtype=torch.bfloat16)
x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
y = layer(x)
loss = y.float().sum()
loss.backward()

print(f"TE GPU smoke PASS  y.shape={y.shape}")
PY
    TE_EXIT=$?
    set -e
    log ""
    if [[ ${TE_EXIT} -eq 0 ]]; then
        log "TE GPU smoke PASSED"
    else
        log "TE GPU smoke FAILED (exit ${TE_EXIT})"
        log "Collecting diagnostics before considering source build..."
        "${PYTHON}" -m pip freeze 2>/dev/null | grep -E 'torch|vllm|transformer|cuda|nccl' | tee -a "${AUDIT_LOG}" || true
        "${PYTHON}" -m pip show transformer-engine transformer-engine-cu12 torch 2>&1 | tee -a "${AUDIT_LOG}" || true
        nvidia-smi 2>&1 | tee -a "${AUDIT_LOG}" || true
        log ""
        log "TE GPU smoke failed. Debug first, then consider source build."
        log "Source build toolchain: bash scripts/setup/setup_cuda128_toolchain.sh"
        log "Build activation:     source scripts/env/activate_cuda128_build.sh"
        fatal "TE GPU runtime smoke failed"
    fi
fi
log ""

# ── done ──────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════════════════"
log "  Audit complete."
log "  Full log: ${AUDIT_LOG}"
log "════════════════════════════════════════════════════════"

#!/usr/bin/env bash
# build_te_torch_offline.sh — Offline GPU build of transformer_engine_torch
#
# Compiles the PyTorch bindings for TransformerEngine v2.6.0.post1
# on a GPU node (H200, SM90) using the CUDA 12.8 NFS toolchain.
# Uses the wheelhouse prepared by setup_vaopd_gkd_cu128.sh.
#
# Prerequisites (run on CPU node first):
#   bash scripts/setup/setup_vaopd_gkd_cu128.sh --execute
#
# Usage:
#   bash scripts/hpc/build_te_torch_offline.sh
#   bash scripts/hpc/build_te_torch_offline.sh --gpu 0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── paths ─────────────────────────────────────────────────────────────────
ENV_PATH="${ENV_PATH:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vaopd-gkd-cu128}"
PYTHON="${ENV_PATH}/bin/python"
CUDA_TOOLCHAIN="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/cuda128-toolchain"
WHEELHOUSE="${REPO_ROOT}/wheelhouse/te260-cu128-torch280-py312"
TE_TORCH_VER="2.6.0.post1"
TE_BUILD_DIR="${REPO_ROOT}/external/te_torch_build"
GPU_ID=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_ID="${2:?--gpu needs a value}"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "══════════════════════════════════════════════════════════════"
echo "  Build transformer_engine_torch (offline GPU)"
echo "  GPU:       ${GPU_ID}"
echo "  Env:       ${ENV_PATH}"
echo "  Toolchain: ${CUDA_TOOLCHAIN}"
echo "  Wheelhouse: ${WHEELHOUSE}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── verify GPU ───────────────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "FATAL: nvidia-smi not found. This script must run on a GPU node."
    exit 1
fi
echo "GPU ${GPU_ID}: $(nvidia-smi -i "${GPU_ID}" --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# ── verify prerequisites ─────────────────────────────────────────────────
if [[ ! -f "${WHEELHOUSE}/transformer_engine_torch-${TE_TORCH_VER}.tar.gz" ]]; then
    echo "FATAL: TE torch source not found at ${WHEELHOUSE}"
    echo "Run setup_vaopd_gkd_cu128.sh --execute on CPU node first."
    exit 1
fi

if [[ ! -d "${CUDA_TOOLCHAIN}" ]]; then
    echo "FATAL: CUDA 12.8 toolchain not found at ${CUDA_TOOLCHAIN}"
    echo "Run: bash scripts/setup/setup_cuda128_toolchain.sh"
    exit 1
fi

# ── activate CUDA 12.8 build env ─────────────────────────────────────────
source "${REPO_ROOT}/scripts/env/activate_cuda128_build.sh"

# ── find CUDNN from pip-installed nvidia-cudnn-cu12 ─────────────────────
CUDNN_LIB=$("${PYTHON}" -c "
import nvidia.cudnn, pathlib, os
cudnn_dir = pathlib.Path(nvidia.cudnn.__file__).parent
lib_dir = cudnn_dir / 'lib'
print(lib_dir)
")
if [[ -d "${CUDNN_LIB}" ]]; then
    export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH}"
    echo "CUDNN lib: ${CUDNN_LIB}"
else
    echo "WARNING: CUDNN lib not found at ${CUDNN_LIB}"
fi

# ── H200 / SM90 ──────────────────────────────────────────────────────────
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export CUDAARCHS="${CUDAARCHS:-90}"
export MAX_JOBS="${MAX_JOBS:-1}"
export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export NVTE_FRAMEWORK="pytorch"

echo "Build env:"
echo "  CUDA_HOME=${CUDA_HOME}"
echo "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "  NVTE_FRAMEWORK=${NVTE_FRAMEWORK}"
echo "  MAX_JOBS=${MAX_JOBS}"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo ""

# ── build ────────────────────────────────────────────────────────────────
mkdir -p "${TE_BUILD_DIR}"

echo "=== Building transformer_engine_torch ${TE_TORCH_VER} ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" -m pip install \
    --no-build-isolation \
    --no-index \
    --find-links "${WHEELHOUSE}" \
    --target "${ENV_PATH}/lib/python3.12/site-packages" \
    "transformer-engine-torch==${TE_TORCH_VER}" \
    -v \
    2>&1 | tee "${TE_BUILD_DIR}/build.log"
BUILD_EXIT=$?

if [[ ${BUILD_EXIT} -ne 0 ]]; then
    echo ""
    echo "FATAL: TE torch build failed (exit ${BUILD_EXIT})"
    echo "Build log: ${TE_BUILD_DIR}/build.log"
    exit ${BUILD_EXIT}
fi

echo ""
echo "=== Verify TE torch import ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" - <<'PY'
import torch
import transformer_engine
import transformer_engine.pytorch as te

print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}")
print(f"transformer_engine: {getattr(transformer_engine, '__version__', 'unknown')}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"device: {torch.cuda.get_device_name(0)}")
print("TE torch build + import: PASS")
PY

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  TE torch build complete."
echo "  Run smoke: bash scripts/hpc/smoke_te_gpu.sh"
echo "══════════════════════════════════════════════════════════════"

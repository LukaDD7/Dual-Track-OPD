# activate_cuda128_build.sh — Environment for CUDA 12.8 source builds
#
# Source this script BEFORE building CUDA extensions from source.
# Do NOT source in normal training scripts — runtime relies on installed
# torch/vLLM/TE packages, not /usr/local/cuda.
#
# Usage:
#   source scripts/env/activate_cuda128_build.sh
#   pip install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.6

CUDA128_ROOT="${CUDA128_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/cuda128-toolchain}"

if [[ ! -d "${CUDA128_ROOT}" ]]; then
    echo "CUDA 12.8 toolchain not found at ${CUDA128_ROOT}"
    echo "Run: bash scripts/setup/setup_cuda128_toolchain.sh"
    return 1 2>/dev/null || exit 1
fi

if [[ ! -x "${CUDA128_ROOT}/bin/nvcc" ]]; then
    echo "nvcc not found at ${CUDA128_ROOT}/bin/nvcc"
    return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="${CUDA128_ROOT}"
export CUDA_PATH="${CUDA128_ROOT}"
export PATH="${CUDA128_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA128_ROOT}/lib64:${CUDA128_ROOT}/lib:${LD_LIBRARY_PATH:-}"

# H200 = SM90
export TORCH_CUDA_ARCH_LIST="9.0"
export FLASH_ATTN_CUDA_ARCHS="90"

# Limit parallel builds to avoid OOM during compilation
export MAX_JOBS="${MAX_JOBS:-1}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

echo "[cuda128-build] CUDA_HOME=${CUDA_HOME}"
echo "[cuda128-build] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "[cuda128-build] MAX_JOBS=${MAX_JOBS}"

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

# Conda-installed CUDA toolkit puts headers in targets/<arch>/include, not include/
_CUDA_TARGET_INCLUDE="${CUDA128_ROOT}/targets/x86_64-linux/include"
if [[ -f "${_CUDA_TARGET_INCLUDE}/cuda_runtime_api.h" ]]; then
    export CPLUS_INCLUDE_PATH="${_CUDA_TARGET_INCLUDE}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
    export C_INCLUDE_PATH="${_CUDA_TARGET_INCLUDE}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
    echo "[cuda128-build] Added conda CUDA include: ${_CUDA_TARGET_INCLUDE}"
fi
# Pip-installed nvidia-* packages carry their own include dirs (cusparse, cublas, etc.)
# The conda cuda-12.8.0 channel has no dev packages for these.
_GKD_ENV="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vaopd-gkd-cu128"
_NVIDIA_GLOB="${_GKD_ENV}/lib/python3.12/site-packages/nvidia/*/include"
for _incdir in ${_NVIDIA_GLOB}; do
    if [[ -d "${_incdir}" ]]; then
        export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH}:${_incdir}"
        export C_INCLUDE_PATH="${C_INCLUDE_PATH:-}:${_incdir}"
    fi
done
echo "[cuda128-build] Added pip nvidia includes"
# Fallback: system CUDA headers on GPU node
_SYS_CUDA_INCLUDE="/usr/local/cuda/include"
if [[ -f "${_SYS_CUDA_INCLUDE}/cuda_runtime_api.h" ]]; then
    export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH}:${_SYS_CUDA_INCLUDE}"
    export C_INCLUDE_PATH="${C_INCLUDE_PATH:-}:${_SYS_CUDA_INCLUDE}"
    echo "[cuda128-build] Added system CUDA include: ${_SYS_CUDA_INCLUDE}"
fi

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

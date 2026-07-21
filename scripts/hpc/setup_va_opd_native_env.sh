#!/usr/bin/env bash
# Build the shared CPU-prepared environment for the native verl VA-OPD path.
# Run on the CPU instance; consume the same prefix from the GPU instance.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HPC_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
ENV_PREFIX="${VA_OPD_ENV_PREFIX:-${HPC_ROOT}/fc-opd-storage/envs/va-opd-verl-e003-cu128}"
VERL_DIR="${VERL_VA_OPD_DIR:-${HPC_ROOT}/fc-opd-storage/backends/verl-va-opd-e0031631}"
CUDA_TOOLCHAIN="${VA_OPD_CUDA_TOOLCHAIN:-${HPC_ROOT}/fc-opd-storage/toolchains/cuda-12.8}"
CONSTRAINTS="${REPO_ROOT}/configs/environment/verl_va_opd_e003_cu128.constraints.txt"

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    CONDA_TOOL="${CONDA_EXE}"
elif command -v micromamba >/dev/null 2>&1; then
    CONDA_TOOL="$(command -v micromamba)"
elif command -v conda >/dev/null 2>&1; then
    CONDA_TOOL="$(command -v conda)"
else
    echo "FATAL: conda or micromamba is required on the CPU instance" >&2
    exit 1
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    "${CONDA_TOOL}" create -y -p "${ENV_PREFIX}" python=3.12 pip=25.1 setuptools wheel
fi

export VERL_VA_OPD_DIR="${VERL_DIR}"
bash "${REPO_ROOT}/scripts/setup/prepare_va_opd_native_verl.sh"

PYTHON="${ENV_PREFIX}/bin/python"
unset CUDA_HOME CUDA_PATH NVCC

"${PYTHON}" -m pip install --upgrade pip==25.1.1 setuptools wheel
"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" vllm==0.18.0
"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" -r "${VERL_DIR}/requirements.txt"
"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" \
    transformers==5.5.0 qwen-vl-utils==0.0.14 pillow flashinfer-python==0.6.6

if [[ "${VA_OPD_SKIP_FLASH_ATTN_BUILD:-0}" != "1" ]]; then
    if [[ ! -x "${CUDA_TOOLCHAIN}/bin/nvcc" ]]; then
        "${CONDA_TOOL}" create -y -p "${CUDA_TOOLCHAIN}" -c nvidia \
            cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8
    fi
    export CUDA_HOME="${CUDA_TOOLCHAIN}"
    export CUDA_PATH="${CUDA_TOOLCHAIN}"
    export PATH="${CUDA_TOOLCHAIN}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
    export MAX_JOBS="${MAX_JOBS:-16}"
    "${PYTHON}" -m pip install --no-build-isolation flash-attn==2.8.3
fi

"${PYTHON}" -m pip install --no-deps -e "${VERL_DIR}"
"${PYTHON}" -m pip install --no-deps -e "${REPO_ROOT}"

PYTHONPATH="${REPO_ROOT}/src:${VERL_DIR}" "${PYTHON}" - <<'PY'
import importlib.metadata as md
import torch
from dual_track_opd.va_opd.native_verl import prepare_native_verl_batch
from verl.trainer.distillation.losses import get_distillation_loss_settings

assert torch.__version__.startswith("2.10.0"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert get_distillation_loss_settings("va_opd_k1").use_estimator
for name in ("torch", "vllm", "transformers", "tensordict", "ray"):
    print(f"{name}=={md.version(name)}")
print("native VA-OPD environment: PASS")
PY

echo "Environment prefix: ${ENV_PREFIX}"
echo "Backend checkout:   ${VERL_DIR}"
echo "CUDA toolchain:      ${CUDA_TOOLCHAIN}"

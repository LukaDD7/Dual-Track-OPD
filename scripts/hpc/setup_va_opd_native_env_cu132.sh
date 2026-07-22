#!/usr/bin/env bash
# Build the shared environment for the native verl VA-OPD path — CUDA 13.2 (cu132).
# Run on a node with internet access (CPU instance); consume from GPU instance.
#
# Usage:
#   export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
#   MAX_JOBS=16 bash scripts/hpc/setup_va_opd_native_env_cu132.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HPC_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
ENV_PREFIX="${VA_OPD_ENV_PREFIX:-${HPC_ROOT}/fc-opd-storage/envs/va-opd-verl-e003-cu132}"
VERL_DIR="${VERL_VA_OPD_DIR:-${HPC_ROOT}/fc-opd-storage/backends/verl-va-opd-e0031631}"
CUDA_TOOLCHAIN="${VA_OPD_CUDA_TOOLCHAIN:-${HPC_ROOT}/fc-opd-storage/toolchains/cuda-13.2}"
CONSTRAINTS="${REPO_ROOT}/configs/environment/verl_va_opd_e003_cu132.constraints.txt"

PYTORCH_INDEX="https://download.pytorch.org/whl/cu132"

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    CONDA_TOOL="${CONDA_EXE}"
elif command -v micromamba >/dev/null 2>&1; then
    CONDA_TOOL="$(command -v micromamba)"
elif command -v conda >/dev/null 2>&1; then
    CONDA_TOOL="$(command -v conda)"
else
    echo "FATAL: conda or micromamba is required" >&2
    exit 1
fi

# ---- Python prefix ----
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    "${CONDA_TOOL}" create -y -p "${ENV_PREFIX}" python=3.12 pip=25.1 setuptools wheel
fi

# ---- verl backend (shared with cu128 setup — same commit, same patch) ----
export VERL_VA_OPD_DIR="${VERL_DIR}"
bash "${REPO_ROOT}/scripts/setup/prepare_va_opd_native_verl.sh"

PYTHON="${ENV_PREFIX}/bin/python"
unset CUDA_HOME CUDA_PATH NVCC

"${PYTHON}" -m pip install --upgrade pip==25.1.1 setuptools wheel

# Install PyTorch from cu132 index FIRST (before vllm, to satisfy its torch==2.11.0 pin)
"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" \
    --index-url "${PYTORCH_INDEX}" \
    torch==2.11.0 torchvision==0.26.0

# vllm — generic manylinux wheel, CUDA compat comes from torch
"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" --no-build-isolation vllm==0.25.1

# verl runtime dependencies
"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" -r "${VERL_DIR}/requirements.txt"

# Additional packages
"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" \
    transformers==5.13.1 \
    qwen-vl-utils==0.0.14 \
    pillow \
    flashinfer-python==0.6.13 \
    flashinfer-cubin==0.6.13

# ---- flash-attn from source (needs CUDA 13.2 toolchain) ----
if [[ "${VA_OPD_SKIP_FLASH_ATTN_BUILD:-0}" != "1" ]]; then
    if [[ ! -x "${CUDA_TOOLCHAIN}/bin/nvcc" ]]; then
        echo "Installing CUDA 13.2 conda toolchain to ${CUDA_TOOLCHAIN} ..."
        "${CONDA_TOOL}" create -y -p "${CUDA_TOOLCHAIN}" -c nvidia -c conda-forge \
            cuda-nvcc=13.2 \
            cuda-cudart-dev=13.2 \
            cuda-cccl=13.2 \
            libcublas-dev || {
            echo "WARNING: cuda-nvcc=13.2 not available via conda. Checking available versions..."
            "${CONDA_TOOL}" search cuda-nvcc -c nvidia --info 2>/dev/null | head -20 || true
            echo "Falling back: set VA_OPD_SKIP_FLASH_ATTN_BUILD=1 to skip flash-attn (may impact performance)"
            echo "Or install CUDA 13.2 toolkit manually to ${CUDA_TOOLCHAIN}"
            exit 1
        }
    fi
    export CUDA_HOME="${CUDA_TOOLCHAIN}"
    export CUDA_PATH="${CUDA_TOOLCHAIN}"
    export PATH="${CUDA_TOOLCHAIN}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
    export MAX_JOBS="${MAX_JOBS:-16}"
    # flash-attn 2.8.3 has cu132 source compatibility (Hopper SM90, not Blackwell)
    "${PYTHON}" -m pip install --no-build-isolation flash-attn==2.8.3
fi

# ---- Editable installs ----
"${PYTHON}" -m pip install --no-deps -e "${VERL_DIR}"
"${PYTHON}" -m pip install --no-deps -e "${REPO_ROOT}"

# ---- Smoke test (offline, no GPU needed) ----
PYTHONPATH="${REPO_ROOT}/src:${VERL_DIR}" "${PYTHON}" - <<'PY'
import importlib.metadata as md
import torch
from dual_track_opd.va_opd.native_verl import prepare_native_verl_batch
from verl.trainer.distillation.losses import get_distillation_loss_settings

assert torch.__version__.startswith("2.11.0"), f"torch version mismatch: {torch.__version__}"
cuda_ver = torch.version.cuda
print(f"torch=={torch.__version__}  cuda=={cuda_ver}  nccl=={torch.cuda.nccl.version()}")
assert get_distillation_loss_settings("va_opd_k1").use_estimator, "va_opd_k1 loss not registered"
for name in ("torch", "vllm", "transformers", "tensordict", "ray"):
    print(f"  {name}=={md.version(name)}")
print("native VA-OPD cu132 environment: PASS")
PY

echo ""
echo "===== cu132 environment ready ====="
echo "Environment prefix: ${ENV_PREFIX}"
echo "Backend checkout:   ${VERL_DIR}"
echo "CUDA toolchain:     ${CUDA_TOOLCHAIN}"
echo ""
echo "Next: run GPU verification commands from docs/va_opd_cu132_gpu_verify.md"

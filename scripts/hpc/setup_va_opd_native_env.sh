#!/usr/bin/env bash
# Build the shared CPU-prepared environment for the native verl VA-OPD path.
# Run on the CPU instance; consume the same prefix from the GPU instance.
#
# The supported user-space stack is CUDA 12.8.  It is valid on both the old
# R570 nodes and the newer R595 nodes whose nvidia-smi reports CUDA 13.2.
# Do not install the published vLLM wheel: vLLM 0.12.0 defaults can contain
# CUDA 12.9 binaries.
# This script compiles the exact vLLM tag against torch/cu128 and a separate
# conda CUDA toolkit, without consulting /usr/bin/nvcc.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HPC_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
ENV_PREFIX="${VA_OPD_ENV_PREFIX:-${HPC_ROOT}/envs/va-opd-native-e003-cu128-r595-v1}"
VERL_DIR="${VERL_VA_OPD_DIR:-${HPC_ROOT}/fc-opd-storage/backends/verl-va-opd-e0031631-clean}"
CUDA_TOOLCHAIN="${VA_OPD_CUDA_TOOLCHAIN:-${HPC_ROOT}/envs/cuda128-toolchain}"
VLLM_SOURCE="${VA_OPD_VLLM_SOURCE:-${HPC_ROOT}/fc-opd-storage/backends/vllm-va-opd-v0120}"
WHEELHOUSE="${VA_OPD_WHEELHOUSE:-${HPC_ROOT}/fc-opd-storage/wheelhouse/va-opd-cu128-r595-v1}"
CONSTRAINTS="${REPO_ROOT}/configs/environment/verl_va_opd_e003_cu128.constraints.txt"
VLLM_COMMIT="4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

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
unset CUDA_HOME CUDA_PATH NVCC CUDACXX CUDAHOSTCXX
unset CC CXX CPP CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LDFLAGS_LD
unset CMAKE_ARGS CMAKE_PREFIX_PATH TORCH_CUDA_ARCH_LIST

"${PYTHON}" -m pip install --upgrade pip==25.1.1 setuptools wheel

# Install torch from the explicit cu128 index before either native extension is
# compiled.  The constraints prevent a later dependency solve from replacing
# this family with a default CUDA 12.9/13 build.
"${PYTHON}" -m pip install --index-url "${PYTORCH_INDEX}" --constraint "${CONSTRAINTS}" \
    torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
"${PYTHON}" - <<'PY'
import torch

assert torch.__version__.startswith("2.9.0"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
print(f"torch build gate: {torch.__version__} CUDA {torch.version.cuda}")
PY

# A complete developer toolkit is intentional: vLLM and flash-attn both build
# on a CPU-only node, so headers and libraries must not be borrowed from /usr.
if [[ ! -x "${CUDA_TOOLCHAIN}/bin/nvcc" ]]; then
    "${CONDA_TOOL}" create -y -p "${CUDA_TOOLCHAIN}" -c nvidia -c conda-forge \
        cuda-toolkit=12.8 gcc_linux-64=12 gxx_linux-64=12
else
    "${CONDA_TOOL}" install -y -p "${CUDA_TOOLCHAIN}" -c nvidia -c conda-forge \
        cuda-toolkit=12.8 gcc_linux-64=12 gxx_linux-64=12
fi

NVCC_REAL="$(realpath "${CUDA_TOOLCHAIN}/bin/nvcc")"
case "${NVCC_REAL}" in
    /usr/*|/usr/local/*)
        echo "FATAL: managed CUDA prefix resolves to a system nvcc: ${NVCC_REAL}" >&2
        exit 1
        ;;
esac
"${CUDA_TOOLCHAIN}/bin/nvcc" --version | grep -q 'release 12\.8'

CC_BIN=""
CXX_BIN=""
for candidate in \
    "${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-cc" \
    "${CUDA_TOOLCHAIN}/bin/gcc"; do
    if [[ -x "${candidate}" ]]; then CC_BIN="${candidate}"; break; fi
done
for candidate in \
    "${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-c++" \
    "${CUDA_TOOLCHAIN}/bin/g++"; do
    if [[ -x "${candidate}" ]]; then CXX_BIN="${candidate}"; break; fi
done
if [[ -z "${CC_BIN}" || -z "${CXX_BIN}" ]]; then
    echo "FATAL: managed GCC/G++ are missing from ${CUDA_TOOLCHAIN}" >&2
    exit 1
fi

export CUDA_HOME="${CUDA_TOOLCHAIN}"
export CUDA_PATH="${CUDA_TOOLCHAIN}"
export CUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLCHAIN}"
export CUDACXX="${CUDA_TOOLCHAIN}/bin/nvcc"
export CUDAHOSTCXX="${CXX_BIN}"
export CC="${CC_BIN}"
export CXX="${CXX_BIN}"
export PATH="${ENV_PREFIX}/bin:${CUDA_TOOLCHAIN}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${CUDA_TOOLCHAIN}:${ENV_PREFIX}"
export TORCH_CUDA_ARCH_LIST="9.0"
export CMAKE_POLICY_DEFAULT_CMP0146="OLD"  # restore FindCUDA (removed in cmake ≥3.27 default)
export MAX_JOBS="${MAX_JOBS:-16}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export VLLM_TARGET_DEVICE="cuda"
export VLLM_VERSION_OVERRIDE="0.12.0"

if [[ ! -e "${VLLM_SOURCE}/.git" ]]; then
    mkdir -p "$(dirname "${VLLM_SOURCE}")"
    git clone https://github.com/vllm-project/vllm.git "${VLLM_SOURCE}"
fi
git -C "${VLLM_SOURCE}" fetch origin "${VLLM_COMMIT}" --depth 1
if [[ "$(git -C "${VLLM_SOURCE}" rev-parse HEAD)" != "${VLLM_COMMIT}" ]]; then
    if [[ -n "$(git -C "${VLLM_SOURCE}" status --porcelain --untracked-files=no)" ]]; then
        echo "FATAL: refusing to switch a modified vLLM source checkout: ${VLLM_SOURCE}" >&2
        git -C "${VLLM_SOURCE}" status --short >&2
        exit 1
    fi
    git -C "${VLLM_SOURCE}" switch --detach "${VLLM_COMMIT}"
fi
if [[ -n "$(git -C "${VLLM_SOURCE}" status --porcelain --untracked-files=no)" ]]; then
    echo "FATAL: vLLM source checkout must remain pristine: ${VLLM_SOURCE}" >&2
    git -C "${VLLM_SOURCE}" status --short >&2
    exit 1
fi

mkdir -p "${WHEELHOUSE}"
VLLM_WHEEL="$(find "${WHEELHOUSE}" -maxdepth 1 -type f -name 'vllm-0.12.0*.whl' -print -quit)"
if [[ -z "${VLLM_WHEEL}" ]]; then
    BUILD_PARENT="$(mktemp -d "${HPC_ROOT}/fc-opd-storage/vllm-build-XXXXXX")"
    BUILD_ROOT="${BUILD_PARENT}/vllm"
    trap 'rm -rf "${BUILD_PARENT}"' EXIT
    git clone --shared --no-checkout "${VLLM_SOURCE}" "${BUILD_ROOT}"
    git -C "${BUILD_ROOT}" checkout --detach "${VLLM_COMMIT}"
    (
        cd "${BUILD_ROOT}"
        "${PYTHON}" use_existing_torch.py
        # cmake >= 4.0 enables CMP0146=NEW which removes the FindCUDA module
        # required by torch's Caffe2 CMake config.  Downgrade if needed.
        "${PYTHON}" -m pip install 'cmake>=3.26.1,<4.0'
        "${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" -r requirements/build.txt
        # vLLM's setup.py only passes -DCMAKE_CUDA_COMPILER from CUDA_HOME but
        # does NOT pass -DCUDA_TOOLKIT_ROOT_DIR.  FindCUDA needs the toolkit root
        # to locate CUDA headers in the conda CUDA layout.
        sed -i '/cmake_args += \[f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}\/bin\/nvcc"\]/a\            cmake_args += [f"-DCUDA_TOOLKIT_ROOT_DIR={CUDA_HOME}"]' setup.py 2>/dev/null || true
        "${PYTHON}" -m pip wheel --no-build-isolation --no-deps --wheel-dir "${WHEELHOUSE}" .
    )
    VLLM_WHEEL="$(find "${WHEELHOUSE}" -maxdepth 1 -type f -name 'vllm-0.12.0*.whl' -print -quit)"
    if [[ -z "${VLLM_WHEEL}" ]]; then
        echo "FATAL: vLLM build completed without the expected 0.12.0 wheel" >&2
        exit 1
    fi
    rm -rf "${BUILD_PARENT}"
    trap - EXIT
fi

# Resolve the runtime as one transaction.  Supplying the locally built wheel
# makes pip validate vLLM's real Requires-Dist metadata without fetching the
# incompatible published binary.  This also repairs a partially populated v2
# prefix by downgrading/replacing packages to the constraints.
RUNTIME_REQUIREMENTS=(
    "${VLLM_WHEEL}"
    "transformers==4.57.3"
    "qwen-vl-utils==0.0.14"
    "pillow"
    "flashinfer-python==0.5.3"
)
"${PYTHON}" -m pip install --dry-run --constraint "${CONSTRAINTS}" \
    -r "${VERL_DIR}/requirements.txt" "${RUNTIME_REQUIREMENTS[@]}"
"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" \
    -r "${VERL_DIR}/requirements.txt" "${RUNTIME_REQUIREMENTS[@]}"

# Force the cu128 source-built vLLM wheel over any same-version wheel that may
# already be present in a reused prefix, then build flash-attn for H200 (SM90).
"${PYTHON}" -m pip install --force-reinstall --no-deps "${VLLM_WHEEL}"
"${PYTHON}" -m pip install --no-build-isolation --constraint "${CONSTRAINTS}" flash-attn==2.8.3

"${PYTHON}" -m pip install --no-deps -e "${VERL_DIR}"
"${PYTHON}" -m pip install --no-deps -e "${REPO_ROOT}"
"${PYTHON}" -m pip check

ENV_MANIFEST="${ENV_PREFIX}/share/dual-track-opd/va_opd_environment_manifest.json"
GENERIC_ENV_MANIFEST="${ENV_PREFIX}/share/dual-track-opd/environment_manifest.json"
mkdir -p "$(dirname "${ENV_MANIFEST}")"
VA_OPD_ENV_MANIFEST="${ENV_MANIFEST}" \
VA_OPD_GENERIC_ENV_MANIFEST="${GENERIC_ENV_MANIFEST}" \
VA_OPD_ENV_PREFIX="${ENV_PREFIX}" \
VA_OPD_REPO_ROOT="${REPO_ROOT}" \
VA_OPD_VLLM_WHEEL="${VLLM_WHEEL}" \
VA_OPD_VLLM_COMMIT="${VLLM_COMMIT}" \
VA_OPD_CUDA_TOOLCHAIN="${CUDA_TOOLCHAIN}" \
VA_OPD_CONSTRAINTS="${CONSTRAINTS}" \
VA_OPD_VERL_DIR="${VERL_DIR}" \
"${PYTHON}" - <<'PY'
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

wheel = Path(os.environ["VA_OPD_VLLM_WHEEL"]).resolve()
constraints = Path(os.environ["VA_OPD_CONSTRAINTS"]).resolve()
manifest = {
    "schema_version": 1,
    "environment_name": Path(os.environ["VA_OPD_ENV_PREFIX"]).name,
    "environment_prefix": str(Path(os.environ["VA_OPD_ENV_PREFIX"]).resolve()),
    "purpose": "native verl OPD and VA-OPD on H200",
    "status_at_build": "candidate",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "build_kind": "cpu-source-build-cu128-h200-sm90",
    "packages": {
        name: md.version(name)
        for name in ("torch", "torchvision", "torchaudio", "vllm", "transformers", "flashinfer-python", "flash-attn")
    },
    "torch_cuda": torch.version.cuda,
    "nccl": ".".join(str(part) for part in torch.cuda.nccl.version()),
    "torch_cuda_arch_list": "9.0",
    "target_gpu": "NVIDIA H200 (SM90)",
    "build_node_kind": "CPU with internet; shared prefix consumed on GPU node",
    "repo_commit": subprocess.check_output(
        ["git", "-C", os.environ["VA_OPD_REPO_ROOT"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "repo_dirty": bool(
        subprocess.check_output(
            ["git", "-C", os.environ["VA_OPD_REPO_ROOT"], "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
    ),
    "vllm_source_commit": os.environ["VA_OPD_VLLM_COMMIT"],
    "vllm_wheel": str(wheel),
    "vllm_wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "constraints_sha256": hashlib.sha256(constraints.read_bytes()).hexdigest(),
    "verl_backend_commit": subprocess.check_output(
        ["git", "-C", os.environ["VA_OPD_VERL_DIR"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "cuda_toolchain": str(Path(os.environ["VA_OPD_CUDA_TOOLCHAIN"]).resolve()),
    "nvcc_version": subprocess.check_output(
        [str(Path(os.environ["VA_OPD_CUDA_TOOLCHAIN"]) / "bin/nvcc"), "--version"], text=True
    ).strip(),
    "host_compiler": subprocess.check_output([os.environ["CC"], "--version"], text=True).splitlines()[0],
    "verification": {
        "pip_check": "pass",
        "gpu_kernel_smoke": "pending",
        "nccl_smoke": "pending",
        "training_smoke": "pending",
    },
}
payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
Path(os.environ["VA_OPD_ENV_MANIFEST"]).write_text(payload, encoding="utf-8")
Path(os.environ["VA_OPD_GENERIC_ENV_MANIFEST"]).write_text(payload, encoding="utf-8")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

PYTHONPATH="${REPO_ROOT}/src:${VERL_DIR}" "${PYTHON}" - <<'PY'
import importlib.metadata as md
import torch
from dual_track_opd.va_opd.native_verl import prepare_native_verl_batch
from verl.trainer.distillation.losses import get_distillation_loss_settings

assert torch.__version__.startswith("2.9.0"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert get_distillation_loss_settings("va_opd_k1").use_estimator
expected = {
    "torch": "2.9.0",
    "vllm": "0.12.0+cu128",
    "transformers": "4.57.3",
    "tensordict": "0.10.0",
    "ray": "2.53.0",
    "flash-attn": "2.8.3",
}
for name, wanted in expected.items():
    assert md.version(name) == wanted, (name, md.version(name), wanted)
    print(f"{name}=={md.version(name)}")
print("native VA-OPD environment: PASS")
PY

echo "Environment prefix: ${ENV_PREFIX}"
echo "Backend checkout:   ${VERL_DIR}"
echo "CUDA toolchain:      ${CUDA_TOOLCHAIN}"
echo "vLLM source:         ${VLLM_SOURCE} @ ${VLLM_COMMIT}"
echo "vLLM wheel:          ${VLLM_WHEEL}"
echo "Build manifest:      ${ENV_MANIFEST}"
echo "Environment registry: ${REPO_ROOT}/docs/environment_registry.md"

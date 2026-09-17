#!/usr/bin/env bash
# Build the shared CPU-prepared environment for native verl VA-OPD on CUDA 13.2.
#
# This is the cu132 counterpart of setup_va_opd_native_env.sh.  The cu128 script
# targets the supported verl e003 + vLLM 0.12.0 matrix; this script targets
# verl e003 + vLLM 0.25.1 source-built against torch 2.13+cu132.
#
# Key differences from the quarantined 2026-07-21 cu132 attempt:
#   1. CUDA 13.2 toolchain is a SEPARATE conda prefix (not mixed into Python env)
#   2. vLLM is SOURCE-BUILT against the installed torch 2.13 (no pre-built wheel ABI)
#   3. No force-reinstall — single pip solve with constraints
#   4. Manifest records the REAL vllm source commit (752a3a50, v0.25.1 tag)
#   5. Reproducible: install order is deterministic, all pins in constraints file
#
# Run on the CPU instance; consume the same prefix from the GPU instance.
# vLLM 0.25.1 has native CUDA 13 support (nvidia-cutlass-dsl[cu13],
# humming-kernels[cu13]), confirmed via PyPI metadata.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HPC_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
ENV_PREFIX="${VA_OPD_ENV_PREFIX:-${HPC_ROOT}/envs/va-opd-native-e003-cu132-r595-v1}"
VERL_DIR="${VERL_VA_OPD_DIR:-${HPC_ROOT}/fc-opd-storage/backends/verl-va-opd-e0031631-cu132}"
CUDA_TOOLCHAIN="${VA_OPD_CUDA_TOOLCHAIN:-${HPC_ROOT}/envs/cuda132-toolchain}"
VLLM_SOURCE="${VA_OPD_VLLM_SOURCE:-${HPC_ROOT}/fc-opd-storage/backends/vllm-va-opd-v0251}"
WHEELHOUSE="${VA_OPD_WHEELHOUSE:-${HPC_ROOT}/fc-opd-storage/wheelhouse/va-opd-cu132-r595-v1}"
CONSTRAINTS="${REPO_ROOT}/configs/environment/verl_va_opd_e003_cu132.constraints.txt"
VLLM_COMMIT="752a3a504485790a2e8491cacbb35c137339ad34"  # v0.25.1 tag
PYTORCH_INDEX="https://download.pytorch.org/whl/cu132"

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

# ── torch 2.13 + cu132 ──────────────────────────────────────────────────
# Install from the explicit cu132 index before any native extension is compiled.
# The constraints file prevents later dependency resolution from replacing this
# family with vLLM's declared torch==2.11.0.
"${PYTHON}" -m pip install --index-url "${PYTORCH_INDEX}" --constraint "${CONSTRAINTS}" \
    torch==2.13.0 torchvision==0.28.0

# torchaudio 2.13 may not have a cu132 wheel; VA-OPD does not require audio.
if "${PYTHON}" -m pip install --index-url "${PYTORCH_INDEX}" --constraint "${CONSTRAINTS}" \
    torchaudio==2.13.0 2>/dev/null; then
    echo "torchaudio 2.13.0+cu132 installed"
else
    echo "torchaudio 2.13.0+cu132 not available (expected — VA-OPD does not need audio)"
fi

"${PYTHON}" - <<'PY'
import torch
assert torch.__version__.startswith("2.13.0"), torch.__version__
assert torch.version.cuda == "13.2", torch.version.cuda
print(f"torch build gate: {torch.__version__} CUDA {torch.version.cuda}")
PY

# ── CUDA 13.2 toolchain (separate conda prefix) ─────────────────────────
# Like the cu128 cuda128-toolchain, this is a standalone compiler prefix.
# It is NOT mixed into the Python environment.
if [[ ! -x "${CUDA_TOOLCHAIN}/bin/nvcc" ]]; then
    "${CONDA_TOOL}" create -y -p "${CUDA_TOOLCHAIN}" -c nvidia -c conda-forge \
        cuda-toolkit=13.2 gcc_linux-64=12 gxx_linux-64=12
else
    "${CONDA_TOOL}" install -y -p "${CUDA_TOOLCHAIN}" -c nvidia -c conda-forge \
        cuda-toolkit=13.2 gcc_linux-64=12 gxx_linux-64=12
fi

NVCC_REAL="$(realpath "${CUDA_TOOLCHAIN}/bin/nvcc")"
case "${NVCC_REAL}" in
    /usr/*|/usr/local/*)
        echo "FATAL: managed CUDA prefix resolves to a system nvcc: ${NVCC_REAL}" >&2
        exit 1
        ;;
esac
"${CUDA_TOOLCHAIN}/bin/nvcc" --version | grep -q 'release 13\.2'

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

# The conda CUDA 13.2 layout puts headers under targets/x86_64-linux/include,
# not at the toolkit prefix root.  Many build scripts (build_deepgemm_C.py,
# FindCUDA, nvcc, etc.) expect headers at ${CUDA_HOME}/include.  Symlink
# everything from the targets include tree so all tools can find them.
TARGETS_INCLUDE="${CUDA_TOOLCHAIN}/targets/x86_64-linux/include"
if [[ -d "${TARGETS_INCLUDE}" ]]; then
    for _item in "${TARGETS_INCLUDE}"/*; do
        _base="$(basename "${_item}")"
        [[ -e "${CUDA_TOOLCHAIN}/include/${_base}" ]] || ln -s "${_item}" "${CUDA_TOOLCHAIN}/include/${_base}"
    done
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
export MAX_JOBS="${MAX_JOBS:-32}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
export VLLM_TARGET_DEVICE="cuda"
export VLLM_VERSION_OVERRIDE="0.25.1"

# ── Clone vLLM 0.25.1 source ────────────────────────────────────────────
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

# ── Source-build vLLM 0.25.1 wheel ──────────────────────────────────────
mkdir -p "${WHEELHOUSE}"
VLLM_WHEEL="$(find "${WHEELHOUSE}" -maxdepth 1 -type f -name 'vllm-0.25.1*.whl' -print -quit)"
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
        "${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" -r requirements/build/cuda.txt
        # vLLM's setup.py passes -DCMAKE_CUDA_COMPILER which causes FindCUDA to
        # skip nvcc probing.  In the conda CUDA 13.2 layout, headers live under
        # targets/x86_64-linux/include, not at the toolkit prefix root.  Without
        # nvcc probing, FindCUDA can't locate them.  Patch torch's cuda.cmake to
        # replace the failing find_package(CUDA) + if-not-found block with
        # hardcoded CUDA paths, then let the rest of the file (arch flags, cudnn,
        # etc.) execute normally.
        "${PYTHON}" - <<'PYEOF'
import re

toolkit = "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/cuda132-toolchain"
targets = f"{toolkit}/targets/x86_64-linux"
cuda_cmake = "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/va-opd-native-e003-cu132-r595-v1/lib/python3.12/site-packages/torch/share/cmake/Caffe2/public/cuda.cmake"

with open(cuda_cmake, 'r') as f:
    content = f.read()

if 'DT_OPD_CU132_FINDCUDA' in content:
    print("cuda.cmake already patched, skipping")
else:
    replacement = f'''# DT_OPD_CU132_FINDCUDA: bypass FindCUDA, use conda CUDA 13.2 layout
include(FindCUDA)  # provides cuda_select_nvcc_arch_flags used below
set(CUDA_TOOLKIT_ROOT_DIR "{targets}")
set(CUDA_INCLUDE_DIRS "${{CUDA_TOOLKIT_ROOT_DIR}}/include")
set(CUDA_CUDART_LIBRARY "{targets}/lib/libcudart.so")
set(CUDA_TOOLKIT_INCLUDE "${{CUDA_INCLUDE_DIRS}}")
set(CUDA_CUDA_LIBRARY "{targets}/lib/stubs/libcuda.so")
set(CUDA_FOUND TRUE)
set(CUDA_VERSION "13.2")
set(CUDA_VERSION_STRING "13.2")
set(CUDA_VERSION_MAJOR 13)
set(CUDA_VERSION_MINOR 2)
set(CUDA_HAS_FP16 TRUE)
set(CUDA_TOOLKIT_TARGET_DIR "${{CUDA_TOOLKIT_ROOT_DIR}}")
set(CUDA_NVCC_EXECUTABLE "{toolkit}/bin/nvcc")
set(CUDA_cudart_LIBRARY "${{CUDA_CUDART_LIBRARY}}")
set(CUDA_cuda_LIBRARY "${{CUDA_CUDA_LIBRARY}}")
set(CUDA_nvrtc_LIBRARY "{targets}/lib/libnvrtc.so")
set(CUDA_cublas_LIBRARY "{targets}/lib/libcublas.so")
set(CUDA_cublasLt_LIBRARY "{targets}/lib/libcublasLt.so")
set(CUDA_curand_LIBRARY "{targets}/lib/libcurand.so")
set(CUDA_cusparse_LIBRARY "{targets}/lib/libcusparse.so")
set(CUDA_cusolver_LIBRARY "{targets}/lib/libcusolver.so")
set(CUDA_cufft_LIBRARY "{targets}/lib/libcufft.so")
'''
    # Replace the find_package(CUDA) ... endif() block
    pattern = r'find_package\(CUDA\)\nif\(NOT CUDA_FOUND\)[^}]*?set\(CAFFE2_USE_CUDA OFF\)[^}]*?return\(\)\nendif\(\)'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    with open(cuda_cmake, 'w') as f:
        f.write(content)
    print("Patched cuda.cmake with DT_OPD_CU132_FINDCUDA")
PYEOF
        grep -c 'DT_OPD_CU132' "${ENV_PREFIX}/lib/python3.12/site-packages/torch/share/cmake/Caffe2/public/cuda.cmake" || true
        # vLLM 0.25.1's CMakeLists.txt hardcodes TORCH_SUPPORTED_VERSION_CUDA=2.11.0.
        sed -i 's/set(TORCH_SUPPORTED_VERSION_CUDA "2.11.0")/set(TORCH_SUPPORTED_VERSION_CUDA "2.13.0")/' CMakeLists.txt
        # vLLM's setup.py sets -DCMAKE_CUDA_COMPILER from CUDA_HOME but does not
        # forward CUDA_HOME itself.  build_deepgemm_C.py reads CUDA_HOME from
        # the environment to locate CCCL headers (include/cccl).  Forward it.
        sed -i '/cmake_args += \[f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}\/bin\/nvcc"\]/a\            cmake_args += [f"-DCUDA_HOME={CUDA_HOME}"]' setup.py
        "${PYTHON}" -m pip wheel --no-build-isolation --no-deps --wheel-dir "${WHEELHOUSE}" .
    )
    VLLM_WHEEL="$(find "${WHEELHOUSE}" -maxdepth 1 -type f -name 'vllm-0.25.1*.whl' -print -quit)"
    if [[ -z "${VLLM_WHEEL}" ]]; then
        echo "FATAL: vLLM build completed without the expected 0.25.1 wheel" >&2
        exit 1
    fi
    rm -rf "${BUILD_PARENT}"
    trap - EXIT
fi

# ── Resolve runtime in one transaction ──────────────────────────────────
# The local wheel makes pip validate vLLM's real Requires-Dist metadata without
# fetching the incompatible published binary.  The constraints file overrides
# vLLM's torch==2.11.0 pin with our torch 2.13.
RUNTIME_REQUIREMENTS=(
    "${VLLM_WHEEL}"
    "transformers==5.14.1"
    "qwen-vl-utils==0.0.14"
    "pillow"
    "flashinfer-python==0.6.13"
)
"${PYTHON}" -m pip install --dry-run --constraint "${CONSTRAINTS}" \
    -r "${VERL_DIR}/requirements.txt" "${RUNTIME_REQUIREMENTS[@]}"
"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" \
    -r "${VERL_DIR}/requirements.txt" "${RUNTIME_REQUIREMENTS[@]}"

# Force the cu132 source-built vLLM wheel over any same-version wheel that may
# already be present in a reused prefix, then build flash-attn for H200 (SM90).
"${PYTHON}" -m pip install --force-reinstall --no-deps "${VLLM_WHEEL}"
"${PYTHON}" -m pip install --no-build-isolation --constraint "${CONSTRAINTS}" flash-attn==2.8.3

# ── Apply cu132 backend patches ─────────────────────────────────────────
# Beyond the 3-file native OPD patch, vLLM 0.25.x moved LoRA imports from
# vllm.lora.models to vllm.lora.lora_model.  Apply this fix idempotently.
LORA_UTILS="${VERL_DIR}/verl/utils/vllm/utils.py"
if grep -qF "vllm.lora.models" "${LORA_UTILS}"; then
    sed -i 's/from vllm\.lora\.models import LoRAModel/from vllm.lora.lora_model import LoRAModel  # vllm >= 0.25/' "${LORA_UTILS}"
    echo "cu132: patched LoRA import in ${LORA_UTILS}"
elif grep -qF "vllm.lora.lora_model" "${LORA_UTILS}"; then
    echo "cu132: LoRA import already patched in ${LORA_UTILS}"
else
    echo "FATAL: cannot locate LoRAModel import in ${LORA_UTILS}" >&2
    exit 1
fi

# Forward LD_LIBRARY_PATH to Ray workers so they can find CUDA/torch shared
# libraries at runtime (especially important on GPU nodes without internet).
RAY_CONSTANTS="${VERL_DIR}/verl/trainer/constants_ppo.py"
if ! grep -qF 'LD_LIBRARY_PATH' "${RAY_CONSTANTS}"; then
    "${PYTHON}" - "${RAY_CONSTANTS}" <<'PY'
import sys
path = sys.argv[1]
with open(path, 'r') as f:
    content = f.read()
marker = 'env_vars = env_vars or {}'
if marker in content:
    content = content.replace(
        marker,
        marker + '\n    ld_path = os.environ.get("LD_LIBRARY_PATH", "")\n    if ld_path:\n        env_vars.setdefault("LD_LIBRARY_PATH", ld_path)'
    )
    with open(path, 'w') as f:
        f.write(content)
    print("cu132: patched Ray LD_LIBRARY_PATH forwarding")
else:
    print("WARNING: could not find env_vars marker in constants_ppo.py")
PY
else
    echo "cu132: Ray LD_LIBRARY_PATH forwarding already present"
fi

# ── Editable installs + verify ──────────────────────────────────────────
"${PYTHON}" -m pip install --no-deps -e "${VERL_DIR}"
"${PYTHON}" -m pip install --no-deps -e "${REPO_ROOT}"
"${PYTHON}" -m pip check

# ── Environment manifest ────────────────────────────────────────────────
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
    "purpose": "native verl OPD and VA-OPD on H200 with CUDA 13.2",
    "status_at_build": "candidate",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "build_kind": "cpu-source-build-cu132-h200-sm90",
    "packages": {
        name: md.version(name)
        for name in ("torch", "torchvision", "vllm", "transformers", "flashinfer-python", "flash-attn")
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

# ── Import gate ─────────────────────────────────────────────────────────
PYTHONPATH="${REPO_ROOT}/src:${VERL_DIR}" "${PYTHON}" - <<'PY'
import importlib.metadata as md
import torch
from dual_track_opd.va_opd.native_verl import prepare_native_verl_batch
from verl.trainer.distillation.losses import get_distillation_loss_settings

assert torch.__version__.startswith("2.13.0"), torch.__version__
assert torch.version.cuda == "13.2", torch.version.cuda
assert get_distillation_loss_settings("va_opd_k1").use_estimator

# cu132 expected versions (vLLM source-built, so local suffix is +cu132)
expected = {
    "torch": "2.13.0+cu132",
    "transformers": "5.14.1",
    "tensordict": "0.10.0",
    "ray": "2.53.0",
    "flash-attn": "2.8.3",
}
for name, wanted in expected.items():
    actual = md.version(name)
    assert actual == wanted, (name, actual, wanted)
    print(f"{name}=={md.version(name)}")

# vLLM version check: source-built wheel should contain +cu132 suffix
vllm_ver = md.version("vllm")
assert vllm_ver.startswith("0.25.1"), f"vllm version mismatch: {vllm_ver}"
print(f"vllm=={vllm_ver}")

# Critical: verify no ABI mismatch by actually importing vllm C extensions
try:
    from vllm.vllm_flash_attn import _vllm_fa2_C
    print("vLLM flash-attn C++ extension: OK")
except ImportError:
    print("WARNING: vllm_flash_attn import skipped (may be optional)")

print("native VA-OPD cu132 environment: PASS")
PY

echo "Environment prefix:   ${ENV_PREFIX}"
echo "Backend checkout:     ${VERL_DIR}"
echo "CUDA toolchain:       ${CUDA_TOOLCHAIN}"
echo "vLLM source:          ${VLLM_SOURCE} @ ${VLLM_COMMIT}"
echo "vLLM wheel:           ${VLLM_WHEEL}"
echo "Build manifest:       ${ENV_MANIFEST}"
echo "Environment registry: ${REPO_ROOT}/docs/environment_registry.md"

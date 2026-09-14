#!/usr/bin/env bash
# Resume: build vLLM + install runtime deps (PyTorch already installed)
set -euo pipefail

DTOPD_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
ENV_PREFIX="${DTOPD_ROOT}/envs/va-opd-native-e003-cu128-r595-v1"
CUDA_TOOLCHAIN="${DTOPD_ROOT}/envs/cuda128-toolchain"
VERL_DIR="${DTOPD_ROOT}/fc-opd-storage/backends/verl-va-opd-e0031631-clean"
VLLM_SOURCE="${DTOPD_ROOT}/fc-opd-storage/backends/vllm-va-opd-v0120"
WHEELHOUSE="${DTOPD_ROOT}/fc-opd-storage/wheelhouse/va-opd-cu128-r595-v1"
REPO_ROOT="${DTOPD_ROOT}/projects/Dual-Track-OPD"
CONSTRAINTS="${REPO_ROOT}/configs/environment/verl_va_opd_e003_cu128.constraints.txt"
PYTHON="${ENV_PREFIX}/bin/python"
VLLM_COMMIT="4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e"
VLLM_VERSION="0.12.0"

echo "=== Resume: Building vLLM ${VLLM_VERSION} ==="

# ── Compiler setup ──────────────────────────────────────────
CC_BIN=""
CXX_BIN=""
for candidate in "${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-gcc" "${CUDA_TOOLCHAIN}/bin/gcc" "${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-cc"; do
    if [[ -x "${candidate}" ]]; then CC_BIN="${candidate}"; break; fi
done
for candidate in "${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-g++" "${CUDA_TOOLCHAIN}/bin/g++" "${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-c++"; do
    if [[ -x "${candidate}" ]]; then CXX_BIN="${candidate}"; break; fi
done
echo "CC=${CC_BIN}"
echo "CXX=${CXX_BIN}"
[[ -z "${CC_BIN}" || -z "${CXX_BIN}" ]] && echo "FATAL: compiler not found" && exit 1

unset CUDA_HOME CUDA_PATH NVCC CUDACXX CUDAHOSTCXX
unset CC CXX CPP CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LDFLAGS_LD
unset CMAKE_ARGS CMAKE_PREFIX_PATH TORCH_CUDA_ARCH_LIST

export CUDA_HOME="${CUDA_TOOLCHAIN}"
export CUDA_PATH="${CUDA_TOOLCHAIN}"
export CUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLCHAIN}"
export CUDACXX="${CUDA_TOOLCHAIN}/bin/nvcc"
export CUDAHOSTCXX="${CXX_BIN}"
export CC="${CC_BIN}"
export CXX="${CXX_BIN}"
export PATH="${ENV_PREFIX}/bin:${CUDA_TOOLCHAIN}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib64:${CUDA_TOOLCHAIN}/lib64/stubs:${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export C_INCLUDE_PATH="${CUDA_TOOLCHAIN}/include:${CUDA_TOOLCHAIN}/targets/x86_64-linux/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_TOOLCHAIN}/include:${CUDA_TOOLCHAIN}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export CMAKE_PREFIX_PATH="${CUDA_TOOLCHAIN}:${ENV_PREFIX}"
export TORCH_CUDA_ARCH_LIST="9.0"
export CMAKE_POLICY_DEFAULT_CMP0146="OLD"
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export VLLM_TARGET_DEVICE="cuda"
export VLLM_VERSION_OVERRIDE="${VLLM_VERSION}"

# ── Build vLLM ──────────────────────────────────────────────
mkdir -p "${WHEELHOUSE}"
VLLM_WHEEL="$(find "${WHEELHOUSE}" -maxdepth 1 -type f -name "vllm-${VLLM_VERSION}*.whl" -print -quit 2>/dev/null || true)"

if [[ -n "${VLLM_WHEEL}" ]]; then
    echo "Cached vLLM wheel found: ${VLLM_WHEEL}"
else
    # Ensure source exists
    if [[ ! -e "${VLLM_SOURCE}/.git" ]]; then
        echo "Cloning vLLM..."
        mkdir -p "$(dirname "${VLLM_SOURCE}")"
        git clone https://github.com/vllm-project/vllm.git "${VLLM_SOURCE}" --depth 1
    fi
    git -C "${VLLM_SOURCE}" fetch origin "${VLLM_COMMIT}" --depth 1
    CURRENT="$(git -C "${VLLM_SOURCE}" rev-parse HEAD)"
    if [[ "${CURRENT}" != "${VLLM_COMMIT}" ]]; then
        git -C "${VLLM_SOURCE}" checkout -- . 2>/dev/null || true
        git -C "${VLLM_SOURCE}" clean -fd 2>/dev/null || true
        git -C "${VLLM_SOURCE}" switch --detach "${VLLM_COMMIT}"
    fi
    echo "vLLM source: $(git -C "${VLLM_SOURCE}" rev-parse --short HEAD)"

    BUILD_PARENT="$(mktemp -d "${DTOPD_ROOT}/fc-opd-storage/vllm-build-XXXXXX")"
    BUILD_ROOT="${BUILD_PARENT}/vllm"
    trap 'rm -rf "${BUILD_PARENT}"' EXIT
    git clone --shared --no-checkout "${VLLM_SOURCE}" "${BUILD_ROOT}"
    git -C "${BUILD_ROOT}" checkout --detach "${VLLM_COMMIT}"

    echo "=== Building vLLM (MAX_JOBS=${MAX_JOBS}) ==="
    cd "${BUILD_ROOT}"
    "${PYTHON}" use_existing_torch.py
    "${PYTHON}" -m pip install 'cmake>=3.26.1,<4.0' ninja 2>&1 | tail -2
    "${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" -r requirements/build.txt 2>&1 | tail -5

    # Inject CUDA_TOOLKIT_ROOT_DIR + VLLM_PYTHON_EXECUTABLE for conda CUDA layout
    sed -i '/cmake_args += \[f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}\/bin\/nvcc"\]/a\            cmake_args += [f"-DCUDA_TOOLKIT_ROOT_DIR={CUDA_HOME}"]' setup.py 2>/dev/null || true
    # vLLM cmake needs VLLM_PYTHON_EXECUTABLE to locate Python headers/libs
    export VLLM_PYTHON_EXECUTABLE="${PYTHON}"

    echo "Starting pip wheel (VLLM_PYTHON_EXECUTABLE=${VLLM_PYTHON_EXECUTABLE})..."
    "${PYTHON}" -m pip wheel --no-build-isolation --no-deps --wheel-dir "${WHEELHOUSE}" . 2>&1 | tail -20

    VLLM_WHEEL="$(find "${WHEELHOUSE}" -maxdepth 1 -type f -name "vllm-${VLLM_VERSION}*.whl" -print -quit)"
    if [[ -z "${VLLM_WHEEL}" ]]; then
        echo "FATAL: vLLM build failed" >&2
        exit 1
    fi
    rm -rf "${BUILD_PARENT}"
    trap - EXIT
    echo "vLLM wheel: ${VLLM_WHEEL}"
fi

# ── Install runtime dependencies ────────────────────────────
echo "=== Installing runtime dependencies ==="
RUNTIME_REQUIREMENTS=(
    "${VLLM_WHEEL}"
    "transformers==4.57.3"
    "qwen-vl-utils==0.0.14"
    "pillow"
    "flashinfer-python==0.5.3"
)

"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" \
    -r "${VERL_DIR}/requirements.txt" "${RUNTIME_REQUIREMENTS[@]}" 2>&1 | tail -10

"${PYTHON}" -m pip install --force-reinstall --no-deps "${VLLM_WHEEL}" 2>&1 | tail -3

echo "=== Building flash-attn ==="
"${PYTHON}" -m pip install --no-build-isolation --constraint "${CONSTRAINTS}" flash-attn==2.8.3 2>&1 | tail -10

echo "=== Installing verl (editable) ==="
"${PYTHON}" -m pip install --no-deps -e "${VERL_DIR}" 2>&1 | tail -3

echo "=== Installing dual-track-opd (editable) ==="
"${PYTHON}" -m pip install --no-deps -e "${REPO_ROOT}" 2>&1 | tail -3

# ── Verify ──────────────────────────────────────────────────
echo "=== Verification ==="
"${PYTHON}" -m pip check 2>&1 | tail -5

"${PYTHON}" - <<'PY'
import importlib.metadata as md
import torch

assert torch.__version__.startswith("2.9.0")
assert torch.version.cuda == "12.8"

expected = {"torch": "2.9.0", "vllm": "0.12.0+cu128", "transformers": "4.57.3", "tensordict": "0.10.0", "flash-attn": "2.8.3"}
for name, wanted in expected.items():
    got = md.version(name)
    assert got == wanted, f"{name}: want={wanted}, got={got}"
    print(f"  {name}=={got} ✓")

from verl.trainer.distillation.losses import DISTILLATION_LOSS_REGISTRY, get_distillation_loss_settings
assert "va_opd_k1" in DISTILLATION_LOSS_REGISTRY
assert "forward_kl_topk" in DISTILLATION_LOSS_REGISTRY
print("  OPD losses registered ✓")
print("=== ALL CHECKS PASSED ===")
PY

echo "=== Build complete at $(date) ==="

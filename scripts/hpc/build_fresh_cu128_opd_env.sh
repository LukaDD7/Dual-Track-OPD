#!/usr/bin/env bash
# Build a fresh cu128 environment with latest verl (e0031631) + OPD + VA-OPD patch.
# Designed to run on a CPU node with internet access.
# Run: nohup bash scripts/hpc/build_fresh_cu128_opd_env.sh > /tmp/build_cu128_opd.log 2>&1 &

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
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
VLLM_COMMIT="4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e"
VLLM_VERSION="0.12.0"

echo "========================================="
echo "Build started at $(date)"
echo "ENV_PREFIX:  ${ENV_PREFIX}"
echo "VERL_DIR:    ${VERL_DIR}"
echo "WHEELHOUSE:  ${WHEELHOUSE}"
echo "========================================="

# ── Check prerequisites ───────────────────────────────────────────
if [[ ! -x "${CUDA_TOOLCHAIN}/bin/nvcc" ]]; then
    echo "FATAL: CUDA toolchain not found: ${CUDA_TOOLCHAIN}" >&2
    exit 1
fi
NVCC_VER="$("${CUDA_TOOLCHAIN}/bin/nvcc" --version | grep release)"
echo "CUDA toolchain: ${NVCC_VER}"

if [[ ! -d "${VERL_DIR}/.git" ]]; then
    echo "FATAL: verl backend not found: ${VERL_DIR}" >&2
    exit 1
fi
echo "verl commit: $(git -C "${VERL_DIR}" rev-parse --short HEAD)"

# ── Create env if needed ──────────────────────────────────────────
if [[ ! -x "${PYTHON}" ]]; then
    echo "Creating conda env at ${ENV_PREFIX}..."
    conda create -y -p "${ENV_PREFIX}" python=3.12 pip=25.1 setuptools wheel
fi
"${PYTHON}" -m pip install --upgrade pip==25.1.1 setuptools wheel 2>&1 | tail -3

# ── Step 1: Install PyTorch ──────────────────────────────────────
echo ""
echo "=== Step 1: Installing PyTorch 2.9.0 from cu128 index ==="
echo "This downloads ~3GB — may take a while on slow network..."
"${PYTHON}" -m pip install \
    --index-url "${PYTORCH_INDEX}" \
    --constraint "${CONSTRAINTS}" \
    torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0

echo "Verifying torch..."
"${PYTHON}" - <<'PY'
import torch
assert torch.__version__.startswith("2.9.0"), f"torch version: {torch.__version__}"
assert torch.version.cuda == "12.8", f"torch cuda: {torch.version.cuda}"
print(f"torch {torch.__version__} CUDA {torch.version.cuda} — PASS")
PY

# ── Step 2: Set up build environment ──────────────────────────────
echo ""
echo "=== Step 2: Setting up build environment ==="

# Compiler setup (same as setup_va_opd_native_env.sh)
CC_BIN=""
CXX_BIN=""
for candidate in "${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-gcc" "${CUDA_TOOLCHAIN}/bin/gcc" "${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-cc"; do
    if [[ -x "${candidate}" ]]; then CC_BIN="${candidate}"; break; fi
done
for candidate in "${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-g++" "${CUDA_TOOLCHAIN}/bin/g++" "${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-c++"; do
    if [[ -x "${candidate}" ]]; then CXX_BIN="${candidate}"; break; fi
done

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
export MAX_JOBS="${MAX_JOBS:-16}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export VLLM_TARGET_DEVICE="cuda"
export VLLM_VERSION_OVERRIDE="${VLLM_VERSION}"

echo "CC=${CC}"
echo "CXX=${CXX}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# ── Step 3: Build vLLM from source ─────────────────────────────────
echo ""
echo "=== Step 3: vLLM ${VLLM_VERSION} source build ==="

mkdir -p "${WHEELHOUSE}"
VLLM_WHEEL="$(find "${WHEELHOUSE}" -maxdepth 1 -type f -name "vllm-${VLLM_VERSION}*.whl" -print -quit 2>/dev/null || true)"

if [[ -n "${VLLM_WHEEL}" ]]; then
    echo "Cached vLLM wheel found: ${VLLM_WHEEL}"
else
    echo "No cached wheel — building vLLM from source (this takes ~30-60 min on CPU)..."

    # Clone vLLM if needed
    if [[ ! -e "${VLLM_SOURCE}/.git" ]]; then
        echo "Cloning vLLM from GitHub..."
        mkdir -p "$(dirname "${VLLM_SOURCE}")"
        git clone https://github.com/vllm-project/vllm.git "${VLLM_SOURCE}" --depth 1 2>&1 | tail -3
    fi
    git -C "${VLLM_SOURCE}" fetch origin "${VLLM_COMMIT}" --depth 1 2>&1 | tail -3
    CURRENT_VLLM="$(git -C "${VLLM_SOURCE}" rev-parse HEAD 2>/dev/null || echo '')"
    if [[ "${CURRENT_VLLM}" != "${VLLM_COMMIT}" ]]; then
        if [[ -n "$(git -C "${VLLM_SOURCE}" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
            echo "WARNING: vLLM source is dirty, resetting..."
            git -C "${VLLM_SOURCE}" checkout -- . 2>/dev/null
            git -C "${VLLM_SOURCE}" clean -fd 2>/dev/null
        fi
        git -C "${VLLM_SOURCE}" switch --detach "${VLLM_COMMIT}"
    fi
    echo "vLLM source at: $(git -C "${VLLM_SOURCE}" rev-parse --short HEAD)"

    # Build in a temp clone
    BUILD_PARENT="$(mktemp -d "${DTOPD_ROOT}/fc-opd-storage/vllm-build-XXXXXX")"
    BUILD_ROOT="${BUILD_PARENT}/vllm"
    trap 'rm -rf "${BUILD_PARENT}"' EXIT
    git clone --shared --no-checkout "${VLLM_SOURCE}" "${BUILD_ROOT}" 2>&1 | tail -1
    git -C "${BUILD_ROOT}" checkout --detach "${VLLM_COMMIT}" 2>&1 | tail -1

    (
        cd "${BUILD_ROOT}"
        "${PYTHON}" use_existing_torch.py 2>&1 | tail -1
        "${PYTHON}" -m pip install 'cmake>=3.26.1,<4.0' 2>&1 | tail -1
        "${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" -r requirements/build.txt 2>&1 | tail -5
        # Inject CUDA_TOOLKIT_ROOT_DIR for conda CUDA layout
        sed -i '/cmake_args += \[f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}\/bin\/nvcc"\]/a\            cmake_args += [f"-DCUDA_TOOLKIT_ROOT_DIR={CUDA_HOME}"]' setup.py 2>/dev/null || true
        # vLLM cmake needs VLLM_PYTHON_EXECUTABLE to locate Python headers/libs
        export VLLM_PYTHON_EXECUTABLE="${PYTHON}"
        echo "Building vLLM wheel (MAX_JOBS=${MAX_JOBS})..."
        "${PYTHON}" -m pip wheel --no-build-isolation --no-deps --wheel-dir "${WHEELHOUSE}" . 2>&1 | tail -10
    )

    VLLM_WHEEL="$(find "${WHEELHOUSE}" -maxdepth 1 -type f -name "vllm-${VLLM_VERSION}*.whl" -print -quit)"
    if [[ -z "${VLLM_WHEEL}" ]]; then
        echo "FATAL: vLLM build failed — no wheel found" >&2
        exit 1
    fi
    rm -rf "${BUILD_PARENT}"
    trap - EXIT
    echo "vLLM wheel built: ${VLLM_WHEEL}"
fi

# ── Step 4: Install runtime dependencies ───────────────────────────
echo ""
echo "=== Step 4: Installing runtime dependencies ==="
echo "This downloads and installs verl, transformers, flash-attn, etc..."

RUNTIME_REQUIREMENTS=(
    "${VLLM_WHEEL}"
    "transformers==4.57.3"
    "qwen-vl-utils==0.0.14"
    "pillow"
    "flashinfer-python==0.5.3"
)

# Dry-run first
echo "Dry-run resolution..."
"${PYTHON}" -m pip install --dry-run --constraint "${CONSTRAINTS}" \
    -r "${VERL_DIR}/requirements.txt" "${RUNTIME_REQUIREMENTS[@]}" 2>&1 | tail -5

echo "Installing..."
"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" \
    -r "${VERL_DIR}/requirements.txt" "${RUNTIME_REQUIREMENTS[@]}" 2>&1 | tail -10

# Force the cu128 vLLM wheel
"${PYTHON}" -m pip install --force-reinstall --no-deps "${VLLM_WHEEL}" 2>&1 | tail -3

# Build flash-attn for H200 SM90
echo "Building flash-attn (SM90)..."
"${PYTHON}" -m pip install --no-build-isolation --constraint "${CONSTRAINTS}" flash-attn==2.8.3 2>&1 | tail -5

# Editable installs
echo "Installing verl (editable)..."
"${PYTHON}" -m pip install --no-deps -e "${VERL_DIR}" 2>&1 | tail -3

echo "Installing dual-track-opd (editable)..."
"${PYTHON}" -m pip install --no-deps -e "${REPO_ROOT}" 2>&1 | tail -3

# ── Step 5: Verify ─────────────────────────────────────────────────
echo ""
echo "=== Step 5: Verification ==="

echo "pip check..."
"${PYTHON}" -m pip check 2>&1 | tail -5

echo ""
echo "Import gate..."
"${PYTHON}" - <<'PY'
import importlib.metadata as md
import torch

# Basic checks
assert torch.__version__.startswith("2.9.0"), f"torch: {torch.__version__}"
assert torch.version.cuda == "12.8", f"CUDA: {torch.version.cuda}"

# Check key packages
expected = {
    "torch": "2.9.0",
    "vllm": "0.12.0+cu128",
    "transformers": "4.57.3",
    "tensordict": "0.10.0",
    "flash-attn": "2.8.3",
}
for name, wanted in expected.items():
    got = md.version(name)
    assert got == wanted, f"{name}: want={wanted}, got={got}"
    print(f"  {name}=={got} ✓")

# Verify OPD distillation API is available
from verl.trainer.distillation.losses import DISTILLATION_LOSS_REGISTRY, get_distillation_loss_settings
print(f"\nRegistered distillation losses: {list(DISTILLATION_LOSS_REGISTRY.keys())}")

# Verify VA-OPD custom losses are registered
assert "va_opd_k1" in DISTILLATION_LOSS_REGISTRY, "va_opd_k1 NOT registered!"
assert get_distillation_loss_settings("va_opd_k1").use_estimator
print("va_opd_k1 registered ✓")

# Verify forward_kl_topk (GKD-style)
assert "forward_kl_topk" in DISTILLATION_LOSS_REGISTRY, "forward_kl_topk NOT registered!"
print("forward_kl_topk registered ✓")

# Verify k1, k3 (PG-based reverse KL)
for mode in ["k1", "k3"]:
    assert mode in DISTILLATION_LOSS_REGISTRY, f"{mode} NOT registered!"
print("k1, k3 registered ✓")

print("\n=== ALL CHECKS PASSED ===")
PY

# ── Done ───────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "Build completed at $(date)"
echo "Environment: ${ENV_PREFIX}"
echo "Activate:    conda activate ${ENV_PREFIX}"
echo "========================================="

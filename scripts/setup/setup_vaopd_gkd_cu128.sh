#!/usr/bin/env bash
# setup_vaopd_gkd_cu128.sh — Isolated conda environment for GKD/Megatron VA-OPD
#
# Creates conda env "vaopd-gkd-cu128" with torch 2.8.0+cu128 and clones verl
# at the GKD recipe pin into external/verl_gkd/ (does NOT touch third_party/verl).
#
# Default: dry-run.  Pass --execute to actually install.
#
# Usage:
#   bash scripts/setup/setup_vaopd_gkd_cu128.sh                  # dry-run
#   bash scripts/setup/setup_vaopd_gkd_cu128.sh --execute        # real install
#   bash scripts/setup/setup_vaopd_gkd_cu128.sh --execute --force  # nuke external/verl_gkd first

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── env var overrides ─────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/miniconda3}"
CUDA_HOME="${CUDA_HOME:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/cuda128-toolchain}"
MODEL_ROOT="${MODEL_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models}"
CONDA_ENV_DIR="${CONDA_BASE}/envs"
ENV_NAME="vaopd-gkd-cu128"
ENV_PATH="${CONDA_ENV_DIR}/${ENV_NAME}"
CONDA="${CONDA_BASE}/bin/conda"

# ── pinned versions ─────────────────────────────────────────────────────────
PYTHON_VER="3.12"
TORCH_VER="2.8.0"
TORCHVISION_VER="0.23.0"
TORCHAUDIO_VER="2.8.0"
CUDA_INDEX="https://download.pytorch.org/whl/cu128"
VERL_REPO="https://github.com/verl-project/verl.git"
VERL_COMMIT="bcb638649a50e58494a8ddd92085ad1174f674b8"
RECIPE_COMMIT="ba246418f4de12b845a09bba975f1a5242adc898"
VLLM_VER="0.11.0"
FLASH_ATTN_VER="2.8.1"
FLASHINFER_VER="0.3.1"
TE_TAG="v2.6"
MEGATRON_TAG="core_v0.13.1"
CUDNN_VER="9.10.2.21"
VERL_GKD_DIR="${REPO_ROOT}/external/verl_gkd"

# ── platform detection (deferred until after env creation) ─────────────────
ARCH="$(uname -m)"

# ── parse args ─────────────────────────────────────────────────────────────
DRY_RUN=true
FORCE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --execute) DRY_RUN=false; shift ;;
        --force)   FORCE=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── helpers ─────────────────────────────────────────────────────────────────
_ver_preamble() {
    echo "=== Version preamble ==="
    if [[ -x "${ENV_PATH}/bin/python" ]]; then
        echo ""
        echo "--- torch version ---"
        "${ENV_PATH}/bin/python" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)" 2>&1 || echo "(torch not installed)"
        echo ""
        echo "--- pip freeze (key packages) ---"
        "${ENV_PATH}/bin/pip" freeze 2>/dev/null | grep -E 'torch|vllm|ray|flash|nccl|nvidia-cudnn|transformer-engine|megatron' || true
    else
        echo "(env does not exist yet — no packages to show)"
    fi
    echo "========================="
    echo ""
}

_cmd() {
    if ${DRY_RUN}; then
        echo "[DRY-RUN] $*"
    else
        echo "[EXEC] $*"
        eval "$@"
    fi
}

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

# ── main ───────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  setup_vaopd_gkd_cu128"
echo "  Mode:      $(${DRY_RUN} && echo 'DRY-RUN' || echo 'EXECUTE')"
echo "  Env name:  ${ENV_NAME}"
echo "  Env path:  ${ENV_PATH}"
echo "  Verl dir:  ${VERL_GKD_DIR}"
echo "  Python:    ${PYTHON_VER}"
echo "  Torch:     ${TORCH_VER}+cu128"
echo "  CUDA_HOME: ${CUDA_HOME}"
echo "  ARCH:      ${ARCH}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── pre-install preamble ───────────────────────────────────────────────────
_ver_preamble

# ── 1. conda env ───────────────────────────────────────────────────────────
if [[ -d "${ENV_PATH}" ]]; then
    echo "=== Conda env '${ENV_NAME}' already exists at ${ENV_PATH} ==="
else
    echo "=== Creating conda env '${ENV_NAME}' (Python ${PYTHON_VER}) ==="
    _cmd "${CONDA}" create -y -n "${ENV_NAME}" python="${PYTHON_VER}"
fi

# ── 2. PyTorch ─────────────────────────────────────────────────────────────
echo ""
echo "=== Installing PyTorch ${TORCH_VER}+cu128 ==="
_cmd "${ENV_PATH}/bin/pip" install --index-url "${CUDA_INDEX}" \
    "torch==${TORCH_VER}" \
    "torchvision==${TORCHVISION_VER}" \
    "torchaudio==${TORCHAUDIO_VER}"

# ── 3. CUDA libraries ──────────────────────────────────────────────────────
echo ""
echo "=== Installing nvidia-cudnn-cu12==${CUDNN_VER} ==="
_cmd "${ENV_PATH}/bin/pip" install "nvidia-cudnn-cu12==${CUDNN_VER}"

# ── 4. vLLM (explicit cu128 wheel from GitHub releases) ────────────────────
echo ""
echo "=== Installing vLLM ${VLLM_VER} (cu128 wheel) ==="
VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VER}/vllm-${VLLM_VER}+cu128-cp38-abi3-manylinux_2_35_${ARCH}.whl"

# Check wheel URL exists before attempting install
if ${DRY_RUN}; then
    _cmd "curl -I ${VLLM_WHEEL_URL}"
else
    echo "Checking vLLM wheel URL: ${VLLM_WHEEL_URL}"
    HTTP_STATUS=$(curl -sL -o /dev/null -w "%{http_code}" "${VLLM_WHEEL_URL}" 2>/dev/null || echo "000")
    if [[ "${HTTP_STATUS}" != "200" ]] && [[ "${HTTP_STATUS}" != "302" ]]; then
        fatal "vLLM wheel not found at ${VLLM_WHEEL_URL} (HTTP ${HTTP_STATUS}). No fallback to PyPI default wheel allowed."
    fi
    echo "  URL OK (HTTP ${HTTP_STATUS})"
    "${ENV_PATH}/bin/pip" install "${VLLM_WHEEL_URL}" --extra-index-url "${CUDA_INDEX}"
fi

# ── 5. flash-attn (pre-built wheel, no source compile) ─────────────────────
echo ""
echo "=== Installing flash-attn ${FLASH_ATTN_VER} (pre-built wheel) ==="
if [[ "${ARCH}" != "x86_64" ]]; then
    if ${DRY_RUN}; then
        echo "[DRY-RUN] would fatal: ARCH=${ARCH} not x86_64"
    else
        fatal "flash-attn pre-built wheel only available for x86_64, got ${ARCH}. Cannot auto-build from source."
    fi
fi
# Detect env Python short version (deferred — env must exist by now in --execute mode)
if [[ -x "${ENV_PATH}/bin/python" ]]; then
    PY_SHORT="cp$("${ENV_PATH}/bin/python" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null)"
else
    PY_SHORT="cp312"  # assume target (env not created yet in dry-run)
fi
if [[ "${PY_SHORT}" != "cp312" ]]; then
    if ${DRY_RUN}; then
        echo "[DRY-RUN] would fatal: env Python is ${PY_SHORT}, need cp312"
    else
        fatal "flash-attn pre-built wheel requires Python cp312, got ${PY_SHORT}. Cannot auto-build from source."
    fi
fi
FLASH_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VER}/flash_attn-${FLASH_ATTN_VER}+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
_cmd "${ENV_PATH}/bin/pip" install "${FLASH_WHEEL_URL}"

# ── 6. flashinfer ──────────────────────────────────────────────────────────
echo ""
echo "=== Installing flashinfer-python ${FLASHINFER_VER} ==="
_cmd "${ENV_PATH}/bin/pip" install "flashinfer-python==${FLASHINFER_VER}"

# ── 7. TransformerEngine (from git tag, requires CUDA_HOME) ────────────────
echo ""
echo "=== Installing TransformerEngine ${TE_TAG} from git ==="
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    fatal "nvcc not found at ${CUDA_HOME}/bin/nvcc. Set CUDA_HOME to the CUDA 12.8 toolchain on NFS (e.g. /inspire/hdd/.../lzy/envs/cuda128-toolchain)."
fi
echo "  Using CUDA_HOME=${CUDA_HOME} (nvcc: $("${CUDA_HOME}/bin/nvcc" --version 2>&1 | head -1))"
_cmd "CUDA_HOME=${CUDA_HOME} NVTE_FRAMEWORK=pytorch ${ENV_PATH}/bin/pip install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@${TE_TAG}"

# ── 8. Megatron-LM (from git tag, not pip package) ─────────────────────────
echo ""
echo "=== Installing Megatron-LM ${MEGATRON_TAG} from git ==="
_cmd "${ENV_PATH}/bin/pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@${MEGATRON_TAG}"

# ── 9. clone verl at GKD pin ───────────────────────────────────────────────
echo ""
echo "=== Setting up verl GKD checkout ==="
if [[ -d "${VERL_GKD_DIR}" ]]; then
    if ${FORCE}; then
        echo "Removing existing ${VERL_GKD_DIR} (--force)"
        _cmd rm -rf "${VERL_GKD_DIR}"
    else
        echo "=== ${VERL_GKD_DIR} already exists, skipping clone (use --force to replace) ==="
    fi
fi

if [[ ! -d "${VERL_GKD_DIR}" ]]; then
    _cmd mkdir -p "${VERL_GKD_DIR}"
    _cmd git clone "${VERL_REPO}" "${VERL_GKD_DIR}/verl"
    _cmd git -C "${VERL_GKD_DIR}/verl" checkout "${VERL_COMMIT}"

    # Verify verl checkout (only in --execute mode)
    if ${DRY_RUN}; then
        echo "[DRY-RUN] would verify verl checkout at ${VERL_COMMIT}"
    else
        _VERL_ACTUAL=$(git -C "${VERL_GKD_DIR}/verl" rev-parse HEAD)
        if [[ "${_VERL_ACTUAL}" != "${VERL_COMMIT}" ]]; then
            fatal "verl checkout mismatch: expected ${VERL_COMMIT}, got ${_VERL_ACTUAL}"
        fi
        echo "  verl HEAD: ${_VERL_ACTUAL} ✓"
    fi

    echo ""
    echo "=== Initializing recipe submodule ==="
    _cmd git -C "${VERL_GKD_DIR}/verl" submodule update --init --recursive recipe

    # Checkout recipe to pinned commit
    _cmd git -C "${VERL_GKD_DIR}/verl/recipe" fetch origin 2>/dev/null || true
    _cmd git -C "${VERL_GKD_DIR}/verl/recipe" checkout "${RECIPE_COMMIT}"

    # Verify recipe checkout — fatal on mismatch, no silent swallowing (only in --execute)
    if ${DRY_RUN}; then
        echo "[DRY-RUN] would verify recipe checkout at ${RECIPE_COMMIT}"
    else
        _RECIPE_ACTUAL=$(git -C "${VERL_GKD_DIR}/verl/recipe" rev-parse HEAD)
        if [[ "${_RECIPE_ACTUAL}" != "${RECIPE_COMMIT}" ]]; then
            fatal "recipe checkout mismatch: expected ${RECIPE_COMMIT}, got ${_RECIPE_ACTUAL}"
        fi
        echo "  recipe HEAD: ${_RECIPE_ACTUAL} ✓"
    fi
fi

# ── 10. install verl (editable, no-deps) ───────────────────────────────────
echo ""
echo "=== Installing verl (editable, --no-deps) ==="
_cmd "${ENV_PATH}/bin/pip install --no-deps -e ${VERL_GKD_DIR}/verl"

# ── 11. Ray ────────────────────────────────────────────────────────────────
echo ""
echo "=== Ensuring ray is installed ==="
_cmd "${ENV_PATH}/bin/pip install ray[default]" 2>/dev/null || \
    _cmd "${ENV_PATH}/bin/pip install ray"

# ── 12. post-install audit ─────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  POST-INSTALL AUDIT"
echo "══════════════════════════════════════════════════════════════"

_audit_failures=0

# a. torch.version.cuda must be 12.8
echo ""
echo "--- Audit: torch.version.cuda == 12.8 ---"
if ${DRY_RUN}; then
    echo "[DRY-RUN] would check torch.version.cuda"
else
    _CUDA_VER=$("${ENV_PATH}/bin/python" -c "import torch; print(torch.version.cuda)" 2>&1) || {
        echo "FAIL: could not import torch"
        _audit_failures=$((_audit_failures + 1))
    }
    if [[ "${_CUDA_VER}" == "12.8" ]]; then
        echo "OK: torch.version.cuda = ${_CUDA_VER}"
    else
        echo "FAIL: torch.version.cuda = ${_CUDA_VER}, expected 12.8"
        _audit_failures=$((_audit_failures + 1))
    fi
fi

# b. No cu129/cu130 in pip freeze
echo ""
echo "--- Audit: no cu129/cu130 in pip freeze ---"
if ${DRY_RUN}; then
    echo "[DRY-RUN] would scan for cu129/cu130"
else
    _CONTAM=$("${ENV_PATH}/bin/pip" freeze 2>/dev/null | grep -E 'cu129|cu130' || true)
    if [[ -n "${_CONTAM}" ]]; then
        echo "FAIL: cu129/cu130 contamination found:"
        echo "${_CONTAM}"
        _audit_failures=$((_audit_failures + 1))
    else
        echo "OK: no cu129/cu130 found"
    fi
fi

# c. import torch, vllm, ray
echo ""
echo "--- Audit: import torch, vllm, ray ---"
if ${DRY_RUN}; then
    echo "[DRY-RUN] would test imports"
else
    "${ENV_PATH}/bin/python" -c "
import torch; print(f'  torch={torch.__version__} ✓')
import vllm; print(f'  vllm={vllm.__version__} ✓')
import ray; print(f'  ray={ray.__version__} ✓')
" 2>&1 || {
        echo "FAIL: one or more imports failed (torch/vllm/ray)"
        _audit_failures=$((_audit_failures + 1))
    }
fi

# d. import megatron, transformer_engine
echo ""
echo "--- Audit: import megatron, transformer_engine ---"
if ${DRY_RUN}; then
    echo "[DRY-RUN] would test megatron/te imports"
else
    "${ENV_PATH}/bin/python" -c "
import megatron; print(f'  megatron ✓ ({megatron.__file__})')
import transformer_engine; print(f'  transformer_engine={transformer_engine.__version__} ✓')
" 2>&1 || {
        echo "FAIL: one or more imports failed (megatron/transformer_engine)"
        _audit_failures=$((_audit_failures + 1))
    }
fi

# ── post-install preamble ──────────────────────────────────────────────────
echo ""
_ver_preamble

echo ""
echo "══════════════════════════════════════════════════════════════"
if ${DRY_RUN}; then
    echo "  DRY-RUN complete.  Re-run with --execute to install."
else
    if (( _audit_failures > 0 )); then
        echo "  POST-INSTALL AUDIT: ${_audit_failures} FAILURE(S) — see above"
        echo "  Setup incomplete. Fix failures before using this env."
        exit 1
    fi
    echo "  Setup complete — all audits passed."
    echo "  Activate:  conda activate ${ENV_NAME}"
    echo "  Verl dir:  ${VERL_GKD_DIR}/verl"
fi
echo "══════════════════════════════════════════════════════════════"

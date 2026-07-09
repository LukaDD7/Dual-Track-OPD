#!/usr/bin/env bash
# setup_vaopd_gkd_cu128.sh — Isolated conda environment for GKD/Megatron VA-OPD
#
# Creates conda env "vaopd-gkd-cu128" with torch 2.8.0+cu128 and clones verl
# at the GKD recipe pin into external/verl_gkd/ (does NOT touch third_party/verl).
#
# Default: dry-run.  Pass --execute to actually install.
#
# Usage:
#   bash scripts/setup/setup_vaopd_gkd_cu128.sh            # dry-run
#   bash scripts/setup/setup_vaopd_gkd_cu128.sh --execute  # real install
#   bash scripts/setup/setup_vaopd_gkd_cu128.sh --execute --force  # nuke external/verl_gkd first

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/miniconda3"
CONDA_ENV_DIR="${CONDA_BASE}/envs"
ENV_NAME="vaopd-gkd-cu128"
ENV_PATH="${CONDA_ENV_DIR}/${ENV_NAME}"

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
TE_VER="2.6"
MEGATRON_VER="core_v0.13.1"
CUDNN_VER="9.10.2.21"
VERL_GKD_DIR="${REPO_ROOT}/external/verl_gkd"

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
        "${ENV_PATH}/bin/pip" freeze 2>/dev/null | grep -E 'torch|vllm|ray|flash|nccl|nvidia-cudnn|transformer-engine|megatron' || echo "(no matching packages)"
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

# ── main ───────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  setup_vaopd_gkd_cu128"
echo "  Mode:      $(${DRY_RUN} && echo 'DRY-RUN' || echo 'EXECUTE')"
echo "  Env name:  ${ENV_NAME}"
echo "  Env path:  ${ENV_PATH}"
echo "  Verl dir:  ${VERL_GKD_DIR}"
echo "  Python:    ${PYTHON_VER}"
echo "  Torch:     ${TORCH_VER}+cu128"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── pre-install preamble ───────────────────────────────────────────────────
_ver_preamble

# ── 1. conda env ───────────────────────────────────────────────────────────
if [[ -d "${ENV_PATH}" ]]; then
    echo "=== Conda env '${ENV_NAME}' already exists at ${ENV_PATH} ==="
else
    echo "=== Creating conda env '${ENV_NAME}' (Python ${PYTHON_VER}) ==="
    _cmd conda create -y -n "${ENV_NAME}" python="${PYTHON_VER}"
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

# ── 4. vLLM ────────────────────────────────────────────────────────────────
echo ""
echo "=== Installing vLLM ${VLLM_VER} ==="
_cmd "${ENV_PATH}/bin/pip" install "vllm==${VLLM_VER}"

# ── 5. flash-attn and flashinfer ───────────────────────────────────────────
echo ""
echo "=== Installing flash-attn ${FLASH_ATTN_VER} ==="
_cmd "${ENV_PATH}/bin/pip" install "flash-attn==${FLASH_ATTN_VER}"

echo ""
echo "=== Installing flashinfer-python ${FLASHINFER_VER} ==="
_cmd "${ENV_PATH}/bin/pip" install "flashinfer-python==${FLASHINFER_VER}"

# ── 6. TransformerEngine ───────────────────────────────────────────────────
echo ""
echo "=== Installing TransformerEngine ${TE_VER} ==="
_cmd "${ENV_PATH}/bin/pip" install "transformer-engine==${TE_VER}"

# ── 7. Megatron-LM ─────────────────────────────────────────────────────────
echo ""
echo "=== Installing Megatron-LM ${MEGATRON_VER} ==="
_cmd "${ENV_PATH}/bin/pip" install "megatron-lm==${MEGATRON_VER}"

# ── 8. clone verl at GKD pin ───────────────────────────────────────────────
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
    echo ""
    echo "=== Initializing recipe submodule ==="
    _cmd git -C "${VERL_GKD_DIR}/verl" submodule update --init --recursive recipe
    _cmd git -C "${VERL_GKD_DIR}/verl/recipe" checkout "${RECIPE_COMMIT}" 2>/dev/null || true
    _cmd git -C "${VERL_GKD_DIR}/verl" submodule update --init --recursive
fi

# ── 9. install verl (editable) ─────────────────────────────────────────────
echo ""
echo "=== Installing verl (editable) ==="
_cmd "${ENV_PATH}/bin/pip" install -e "${VERL_GKD_DIR}/verl"

# ── 10. Ray (if not pulled by verl) ────────────────────────────────────────
echo ""
echo "=== Ensuring ray is installed ==="
_cmd "${ENV_PATH}/bin/pip" install "ray[default]" 2>/dev/null || \
    _cmd "${ENV_PATH}/bin/pip" install "ray"

# ── 11. verify git state ───────────────────────────────────────────────────
echo ""
echo "=== Git state verification ==="
if [[ -d "${VERL_GKD_DIR}/verl/.git" ]]; then
    echo "verl HEAD:"
    _cmd git -C "${VERL_GKD_DIR}/verl" log --oneline -1
    echo "recipe HEAD:"
    _cmd git -C "${VERL_GKD_DIR}/verl/recipe" log --oneline -1 2>/dev/null || echo "(recipe not a git repo — may be submodule)"
else
    echo "WARNING: ${VERL_GKD_DIR}/verl is not a git repo"
fi

# ── post-install preamble ──────────────────────────────────────────────────
echo ""
_ver_preamble

echo ""
echo "══════════════════════════════════════════════════════════════"
if ${DRY_RUN}; then
    echo "  DRY-RUN complete.  Re-run with --execute to install."
else
    echo "  Setup complete."
    echo "  Activate:  conda activate ${ENV_NAME}"
    echo "  Verl dir:  ${VERL_GKD_DIR}/verl"
fi
echo "══════════════════════════════════════════════════════════════"

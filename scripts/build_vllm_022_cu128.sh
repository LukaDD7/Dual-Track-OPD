#!/usr/bin/env bash
#
# 在 qwen35 环境（torch 2.11+cu128）中用 cu128 工具链源码构建 vLLM 0.22.x
# 目的：qwen3.5/3.6 蒸馏唯一可行的 vLLM 路径（PyPI 0.23 wheel 是 CUDA 13 构建，
#       本节点 driver 570.124.06 最大只支持 CUDA 12.8，实测 CUDA error insufficient）
#
# 前置条件：网络可达内网 PyPI 镜像（至少一次，用于 setuptools-rust/setuptools-scm）
#   nexus.sii.shaipower.online
#
# 用法：
#   bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/build_vllm_022_cu128.sh
#
# 可选：
#   DEEPGEMM_SRC_DIR=/path/to/DeepGEMM   # 提供则构建 DeepGEMM；否则默认跳过（fp8 GEMM 非必需）
set -euo pipefail

DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
ENV_PREFIX="${DTOPD_ROOT}/envs/va-opd-qwen35-cu128"
CUDA_HOME="${DTOPD_ROOT}/envs/cuda128-toolchain"
REPO="${DTOPD_ROOT}/repos/vllm"
DEPS_CACHE="${DTOPD_ROOT}/fc-opd-storage/backends/vllm-va-opd-v0120/.deps"
PATCH="${DTOPD_ROOT}/projects/Dual-Track-OPD/patches/vllm-022-skip-deepgemm.patch"
WHEEL_OUT="${DTOPD_ROOT}/fc-opd-storage/wheelhouse/va-opd-qwen35-cu128-vllm022"
NEXUS="http://nexus.sii.shaipower.online/repository/pypi/simple"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${DTOPD_ROOT}/.cargo/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="${DTOPD_ROOT}/.cache/flashinfer"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# shellcheck disable=SC1091
source "${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

echo "== python=$(command -v python3) =="
echo "== nvcc=$(${CUDA_HOME}/bin/nvcc --version | tail -1) =="
echo "== cmake=$(${CUDA_HOME}/bin/cmake --version | head -1) =="
echo "== cargo=$(cargo --version 2>/dev/null | head -1) =="

# ---- 1. 构建期依赖（必须联网一次）----
if ! "${ENV_PREFIX}/bin/pip" show setuptools-rust >/dev/null 2>&1; then
  echo "[1] installing setuptools-rust/setuptools-scm from ${NEXUS}"
  "${ENV_PREFIX}/bin/pip" install "setuptools-rust>=1.9.0" "setuptools-scm>=8.0" \
    --index-url "${NEXUS}" --trusted-host nexus.sii.shaipower.online
fi
"${ENV_PREFIX}/bin/pip" show setuptools-rust setuptools-scm >/dev/null || {
  echo "FATAL: setuptools-rust/setuptools-scm 不可用（网络不通时无法继续）"
  echo "      请在 GPU 节点上执行："
  echo "        getent hosts nexus.sii.shaipower.online pypi.org github.com"
  echo "      或等 CPU 实例网络恢复后重跑本脚本。"
  exit 1
}

# ---- 2. 本地依赖源码（缓存复用，避免 FetchContent 拉网络）----
# QUTLASS_SRC_DIR 必须是 qutlass git 根（含 CMakeLists.txt），不是 python 包子目录
QUTLASS_SRC="${DEPS_CACHE}/qutlass-src"
TRITON_KERNELS_SRC="${DEPS_CACHE}/triton_kernels-src/python/triton_kernels/triton_kernels"
for p in "${QUTLASS_SRC}" "${TRITON_KERNELS_SRC}"; do
  [ -d "${p}" ] || { echo "FATAL: missing cached dep ${p}"; exit 1; }
done
export QUTLASS_SRC_DIR="${QUTLASS_SRC}"
export TRITON_KERNELS_SRC_DIR="${TRITON_KERNELS_SRC}"
echo "[2] QUTLASS_SRC_DIR=${QUTLASS_SRC_DIR}"
echo "    TRITON_KERNELS_SRC_DIR=${TRITON_KERNELS_SRC_DIR}  # 缓存为 v3.5.0，0.22 目标 v3.5.1"
echo "    注意：v0.22/0.23 需要 CUTLASS v4.4.2（缓存为 v4.2.1），若未设置 VLLM_CUTLASS_SRC_DIR"
echo "          构建期会从 GitHub FetchContent 拉取 v4.4.2，离线时无法继续。"

# ---- 3. DeepGEMM：默认跳过（bf16 蒸馏不需要 fp8 GEMM）；显式提供源码则构建 ----
if [ -n "${DEEPGEMM_SRC_DIR:-}" ]; then
  export DEEPGEMM_SRC_DIR
  echo "[3] DeepGEMM 将从 ${DEEPGEMM_SRC_DIR} 构建"
else
  export VLLM_SKIP_DEEPGEMM=1
  echo "[3] VLLM_SKIP_DEEPGEMM=1（跳过，fp8 GEMM 非蒸馏必需）"
fi

# ---- 4. 应用 skip-deepgemm patch（幂等）----
cd "${REPO}"
if ! rg -q 'VLLM_SKIP_DEEPGEMM' setup.py CMakeLists.txt; then
  git apply --check "${PATCH}" || { echo "FATAL: patch 无法应用: ${PATCH}"; exit 1; }
  git apply "${PATCH}"
  echo "[4] applied ${PATCH}"
else
  echo "[4] patch already applied"
fi

# ---- 5. 构建 wheel 并安装 ----
mkdir -p "${WHEEL_OUT}"
echo "[5] building vllm wheel (MAX_JOBS=${MAX_JOBS:-32}, 约 30-90 分钟)..."
MAX_JOBS="${MAX_JOBS:-32}" "${ENV_PREFIX}/bin/pip" wheel . \
  --no-deps --no-build-isolation -w "${WHEEL_OUT}"

echo "[6] installing wheel"
"${ENV_PREFIX}/bin/pip" install --no-deps --force-reinstall "${WHEEL_OUT}"/vllm-*.whl

# ---- 7. 验证：必须是 libcudart.so.12（cu128）----
SO=$("${ENV_PREFIX}/bin/python" - <<'EOF'
import glob, os
import vllm
root = os.path.dirname(vllm.__file__)
cands = glob.glob(os.path.join(root, "_C*.so")) + glob.glob(os.path.join(root, "_C*abi3*.so"))
print(sorted(cands)[0])
EOF
)
echo "== vllm version: $("${ENV_PREFIX}/bin/python" -c 'import vllm; print(vllm.__version__)') =="
echo "== NEEDED of ${SO}: =="
readelf -d "${SO}" | grep -E 'NEEDED' | grep -E 'cudart|torch' || true
echo "DONE. 期望看到 libcudart.so.12（而非 .13）。"

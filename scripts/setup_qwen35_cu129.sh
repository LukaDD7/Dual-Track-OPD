#!/usr/bin/env bash
#
# 将 qwen35 环境从「torch cu128 + vllm cu130(cu13) 混合」切换为
# 「torch 2.11.0+cu129 + vllm 0.23.0+cu129」的纯 CUDA 12.9 栈。
#
# 为什么 cu129 而不是 cu128 源码重建：
#   - 本节点 driver 570.124.06（nvidia-smi 显示最大 CUDA 12.8）。
#   - CUDA 官方 MVC（Minor Version Compatibility）表：
#       CUDA 12.x 应用最小驱动 >= 525（Linux），570.124 在范围内 → cu129 可跑
#       CUDA 13.x 应用最小驱动 >= 580           → cu130/cu132 不可跑（已实测报错）
#   - vllm 0.23.0 官方提供 cu129 变体 wheel（wheels.vllm.ai/0.23.0/cu129），
#     默认 PyPI wheel 是 cu130（要 libcudart.so.13），必须显式装 cu129 变体。
#   - torch 2.11 官方有 +cu129 构建；本脚本已把全部 wheel 预先下载到
#     ${DTOPD_ROOT}/fc-opd-storage/wheelhouse/va-opd-qwen35-cu129/，GPU 节点全程离线。
#
# 用法（GPU 节点，无需网络）：
#   bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/setup_qwen35_cu129.sh
# 之后：
#   bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/run_qwen35_gpu_smoke.sh
set -euo pipefail

DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
ENV_PREFIX="${DTOPD_ROOT}/envs/va-opd-qwen35-cu128"
PIP="${ENV_PREFIX}/bin/python -m pip"
WH="${DTOPD_ROOT}/fc-opd-storage/wheelhouse/va-opd-qwen35-cu129"
WH_NVIDIA="${WH}/nvidia-cu129"

source "${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

echo "== python=$(command -v python3) =="
echo "== wheelhouse=${WH} =="
[ -d "${WH_NVIDIA}" ] || { echo "FATAL: missing ${WH_NVIDIA}"; exit 1; }

echo "== 当前 vllm/torch =="
"${ENV_PREFIX}/bin/pip" list 2>/dev/null | rg -i '^(vllm|torch |torchvision|torchaudio)' || true

# ---- 1. 卸载 cu13 残留 + 旧 vllm cu130 wheel ----
echo "[1] 卸载 cu13 残留组件与旧 vllm"
"${ENV_PREFIX}/bin/pip" uninstall -y \
  vllm nvidia-cuda-runtime nvidia-cuda-crt nvidia-cuda-cccl nvidia-cuda-nvcc \
  nvidia-cuda-nvrtc nvidia-cuda-tileiras nvidia-nvjitlink nvidia-nvvm \
  nvidia-cutlass-dsl-libs-cu13 2>/dev/null || true

# ---- 2. torch 家族切到 +cu129（--no-deps：nvidia 库下一步单独装）----
echo "[2] torch/torchvision/torchaudio -> 2.11.0+cu129"
"${ENV_PREFIX}/bin/pip" install --no-index --no-deps \
  "${WH_NVIDIA}/torch-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
  "${WH_NVIDIA}/torchvision-0.26.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
  "${WH_NVIDIA}/torchaudio-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"

# ---- 3. nvidia 运行时库升级到 12.9（libcudart.so.12 -> 12.9.79）----
echo "[3] nvidia runtime libs -> 12.9"
"${ENV_PREFIX}/bin/pip" install --no-index --no-deps \
  "${WH_NVIDIA}"/nvidia_cublas_cu12-12.9*.whl \
  "${WH_NVIDIA}"/nvidia_cudnn_cu12-9.17*.whl \
  "${WH_NVIDIA}"/nvidia_cuda_runtime_cu12-12.9*.whl \
  "${WH_NVIDIA}"/nvidia_cuda_nvrtc_cu12-12.9*.whl \
  "${WH_NVIDIA}"/nvidia_cuda_cupti_cu12-12.9*.whl \
  "${WH_NVIDIA}"/nvidia_nvjitlink_cu12-12.9*.whl \
  "${WH_NVIDIA}"/nvidia_cufft_cu12-11.4*.whl \
  "${WH_NVIDIA}"/nvidia_curand_cu12-10.3*.whl \
  "${WH_NVIDIA}"/nvidia_cusolver_cu12-11.7*.whl \
  "${WH_NVIDIA}"/nvidia_cusparse_cu12-12.5*.whl \
  "${WH_NVIDIA}"/nvidia_cusparselt_cu12-0.7*.whl \
  "${WH_NVIDIA}"/nvidia_nccl_cu12-2.28*.whl \
  "${WH_NVIDIA}"/nvidia_nvshmem_cu12-3.4*.whl \
  "${WH_NVIDIA}"/nvidia_nvtx_cu12-12.9*.whl \
  "${WH_NVIDIA}"/nvidia_cufile_cu12-1.14*.whl \
  "${WH_NVIDIA}"/cuda_bindings-12.9*.whl \
  "${WH_NVIDIA}"/cuda_toolkit-12.9*.whl \
  "${WH_NVIDIA}"/cuda_pathfinder-1.2*.whl \
  "${WH}"/nvidia_cuda_cccl_cu12-12.9*.whl \
  "${WH}"/nvidia_cuda_nvcc_cu12-12.9*.whl

# ---- 4. vllm 0.23.0+cu129 wheel（显式文件，避免解析回 cu130）----
echo "[4] vllm -> 0.23.0+cu129"
"${ENV_PREFIX}/bin/pip" install --no-index --no-deps \
  "${WH}/vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"

# ---- 5. cu12 风味 kernel 依赖（cutlass-dsl 基座 / humming-kernels）----
echo "[5] cutlass-dsl / humming-kernels cu12 对齐"
"${ENV_PREFIX}/bin/pip" install --no-index --no-deps \
  "${WH}/nvidia_cutlass_dsl-4.5.2-py3-none-any.whl" \
  "${WH}/nvidia_cutlass_dsl_libs_base-4.5.2-cp312-cp312-manylinux_2_28_x86_64.whl" \
  "${WH}/humming_kernels-0.1.4-py3-none-any.whl"

# ---- 6. 验证：版本 + _C 链接的必须是 libcudart.so.12（12.9）----
echo "[6] 验证"
"${ENV_PREFIX}/bin/python" - <<'EOF'
import glob, os
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
import vllm
print("vllm:", vllm.__version__)
root = os.path.dirname(vllm.__file__)
cands = glob.glob(os.path.join(root, "_C*abi3*.so")) + glob.glob(os.path.join(root, "_C*.so"))
so = sorted(cands)[0]
print("engine ext:", os.path.basename(so))
EOF
SO=$("${ENV_PREFIX}/bin/python" - <<'EOF'
import glob, os, vllm
root = os.path.dirname(vllm.__file__)
cands = glob.glob(os.path.join(root, "_C*abi3*.so")) + glob.glob(os.path.join(root, "_C*.so"))
print(sorted(cands)[0])
EOF
)
echo "== NEEDED of ${SO}: =="
readelf -d "${SO}" | grep -E 'NEEDED' | grep -E 'cudart' || true
echo "== libcudart in env: =="
ls -l "${ENV_PREFIX}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/" | grep cudart || true
echo "DONE. 期望: NEEDED libcudart.so.12 且 site-packages 里 libcudart.so.12.9.x（而非 13.x）。"
echo "下一步: bash ${DTOPD_ROOT}/projects/Dual-Track-OPD/scripts/run_qwen35_gpu_smoke.sh"

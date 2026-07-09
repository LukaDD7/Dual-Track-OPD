#!/usr/bin/env bash
# setup_cuda128_toolchain.sh — Reusable CUDA 12.8 build toolchain (NFS)
#
# Creates a conda env with the CUDA 12.8 toolkit (nvcc, headers, libraries)
# for building CUDA extensions from source.  NOT needed for normal runtime.
#
# Usage:
#   bash scripts/setup/setup_cuda128_toolchain.sh
#
# To activate before a source build:
#   source scripts/env/activate_cuda128_build.sh

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/miniconda3}"
CONDA="${CONDA_BASE}/bin/conda"
CUDA_TOOLCHAIN="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/cuda128-toolchain"

echo "=== Setting up CUDA 12.8 build toolchain ==="
echo "  Target: ${CUDA_TOOLCHAIN}"
echo ""

if [[ -d "${CUDA_TOOLCHAIN}/bin/nvcc" ]]; then
    echo "Toolchain already exists at ${CUDA_TOOLCHAIN}"
    "${CUDA_TOOLCHAIN}/bin/nvcc" --version 2>&1 | head -1
    echo "Done."
    exit 0
fi

echo "Creating conda env with CUDA 12.8.0 toolkit..."
"${CONDA}" create -y -p "${CUDA_TOOLCHAIN}" -c nvidia/label/cuda-12.8.0 cuda

echo ""
echo "Verifying..."
"${CUDA_TOOLCHAIN}/bin/nvcc" --version 2>&1 | head -3
echo ""
echo "CUDA 12.8 toolchain installed at ${CUDA_TOOLCHAIN}"
echo "To use: source scripts/env/activate_cuda128_build.sh"

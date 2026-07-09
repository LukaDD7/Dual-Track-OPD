#!/usr/bin/env bash
# smoke_te_gpu.sh — GPU smoke test for TransformerEngine PyTorch bindings
#
# Validates that transformer_engine.pytorch is correctly built and can
# execute a forward+backward pass on GPU.
#
# Usage:
#   bash scripts/hpc/smoke_te_gpu.sh
#   bash scripts/hpc/smoke_te_gpu.sh --gpu 0

set -euo pipefail

ENV_PATH="${ENV_PATH:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vaopd-gkd-cu128}"
PYTHON="${ENV_PATH}/bin/python"
GPU_ID=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_ID="${2:?--gpu needs a value}"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== TE GPU Smoke (GPU ${GPU_ID}) ==="

if ! command -v nvidia-smi &>/dev/null; then
    echo "FATAL: nvidia-smi not found. This script must run on a GPU node."
    exit 1
fi

echo "GPU: $(nvidia-smi -i "${GPU_ID}" --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" - <<'PY'
import torch
import transformer_engine
import transformer_engine.pytorch as te

print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}")
print(f"transformer_engine: {getattr(transformer_engine, '__version__', 'unknown')}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"device: {torch.cuda.get_device_name(0)}")

layer = te.Linear(16, 16).cuda().to(dtype=torch.bfloat16)
x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
y = layer(x)
loss = y.float().sum()
loss.backward()

print(f"TE GPU smoke PASS  y.shape={y.shape}")
PY

echo ""
echo "TE GPU smoke complete."

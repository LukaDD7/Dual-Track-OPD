#!/usr/bin/env bash
# FC-OPD verl one-step smoke.
#
# Prerequisites (all on NFS, visible to GPU node):
#   1. Patches applied to third_party/verl (already done via NFS)
#   2. Teacher running on GPU 0: curl -s http://127.0.0.1:18080/health
#   3. Parquet at fc-opd-storage/outputs/fc_opd/verl_smoke/train.parquet
#   4. Student model at models/Qwen3-VL-4B-Instruct
#
# Usage (from repo root, on GPU node with 2 free GPUs):
#   CUDA_VISIBLE_DEVICES=1,2 bash scripts/hpc/run_verl_fc_opd_smoke.sh
#
# Post-run:  ray stop -f

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL_PATH="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-4B-Instruct"
PARQUET="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/verl_smoke/train.parquet"

echo "=== Prerequisites ==="
echo "Model: ${MODEL_PATH}"
echo "Parquet: ${PARQUET}"
echo "Teacher: $(curl -s http://127.0.0.1:18080/health || echo 'NOT REACHABLE')"
echo ""

# Verify patches (check for the injected code, not config values)
if ! grep -q "compute_verl_sparse_topk_kd" third_party/verl/verl/workers/actor/dp_actor.py; then
    echo "FATAL: actor patch not applied"
    exit 1
fi
if ! grep -q "fc_opd_hook_fqn" third_party/verl/verl/trainer/ppo/ray_trainer.py; then
    echo "FATAL: trainer patch not applied"
    exit 1
fi
echo "Patches: OK"

# Start Ray if not running
if ! ray status &>/dev/null 2>&1; then
    echo "Starting Ray..."
    ray start --head --num-gpus=2 --disable-usage-stats
fi
echo "Ray: OK"
echo ""

# vLLM v1 + Triton need a C compiler.  If CC is unset or points to a
# non-existent binary (e.g. a stale conda env path), fall back to gcc on PATH.
if [ -z "${CC:-}" ] || ! command -v "${CC}" >/dev/null 2>&1; then
    export CC=gcc
fi
echo "CC=${CC}"

# Run one PPO step with FC-OPD.
# Configuration is in configs/experiment/verl_fc_opd_smoke.yaml.
# CLI overrides are only for paths that differ between environments.
echo "=== Launching verl FC-OPD smoke ==="

CONFIG_DIR="${REPO_ROOT}/configs/experiment"
python -m verl.trainer.main_ppo \
    --config-path="${CONFIG_DIR}" \
    --config-name=verl_fc_opd_smoke \
    "data.train_files=${PARQUET}" \
    "data.val_files=${PARQUET}" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "$@"

echo ""
echo "=== Done ==="
echo "Look for 'actor/fc_opd_loss' in the output above."
echo "Cleanup: ray stop -f"

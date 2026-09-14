#!/usr/bin/env bash
# Run or resume one Phase-5B wrong-control generation shard on a single GPU.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 SHARD_INDEX NUM_SHARDS [RUN_PREFIX]" >&2
    exit 2
fi

SHARD_INDEX="$1"
NUM_SHARDS="$2"
RUN_PREFIX="${3:-${WRONG_CONTROL_PREFIX:-wrong_control_pool_20260815}}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"
CONFIG="${WRONG_CONTROL_CONFIG:-${PROJECT_ROOT}/configs/experiment/reachability_wrong_controls.yaml}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
RUN_DIR="${OUTPUT_ROOT}/support_aware_opd/${RUN_PREFIX}_s${SHARD_INDEX}"

[[ -f "${CONFIG}" ]] || { echo "FATAL: missing config ${CONFIG}" >&2; exit 1; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "FATAL: expose exactly one GPU" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo "FATAL: expected one visible GPU" >&2; exit 1; }

export DTOPD_OUTPUT_ROOT="${OUTPUT_ROOT}"
export DTOPD_MODEL_ROOT="${DTOPD_MODEL_ROOT:-${PROJECT_ROOT}/models}"

exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.reachability_wrong_controls run \
    --config "${CONFIG}" \
    --output-dir "${RUN_DIR}" \
    --shard-index "${SHARD_INDEX}" \
    --num-shards "${NUM_SHARDS}"

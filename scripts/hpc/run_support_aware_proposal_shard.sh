#!/usr/bin/env bash
# Run or resume one teacher-proposal feasibility shard on a two-GPU pair.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 SHARD_INDEX NUM_SHARDS [RUN_PREFIX]" >&2
    exit 2
fi

SHARD_INDEX="$1"
NUM_SHARDS="$2"
RUN_PREFIX="${3:-${PROPOSAL_RUN_PREFIX:-proposal_feasibility_20260805}}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"
CONFIG="${PROPOSAL_CONFIG:-${PROJECT_ROOT}/configs/experiment/support_aware_teacher_proposal_feasibility.yaml}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
RUN_DIR="${OUTPUT_ROOT}/support_aware_opd/${RUN_PREFIX}_s${SHARD_INDEX}"

[[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]] || { echo "FATAL: SHARD_INDEX must be an integer" >&2; exit 1; }
[[ "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: NUM_SHARDS must be positive" >&2; exit 1; }
(( SHARD_INDEX < NUM_SHARDS )) || { echo "FATAL: shard index is outside [0, NUM_SHARDS)" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "FATAL: missing config ${CONFIG}" >&2; exit 1; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
    echo "FATAL: expose exactly one teacher/student GPU pair with CUDA_VISIBLE_DEVICES" >&2
    exit 1
}
IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#VISIBLE_GPUS[@]}" -eq 2 ]] || {
    echo "FATAL: expected exactly two visible GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
}

export DTOPD_OUTPUT_ROOT="${OUTPUT_ROOT}"
export DTOPD_MODEL_ROOT="${DTOPD_MODEL_ROOT:-${PROJECT_ROOT}/models}"

exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.proposal_feasibility run \
    --config "${CONFIG}" \
    --output-dir "${RUN_DIR}" \
    --shard-index "${SHARD_INDEX}" \
    --num-shards "${NUM_SHARDS}"

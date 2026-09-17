#!/usr/bin/env bash
# Launch one frozen student-only prefix-intervention shard per GPU.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 GPU_CSV [RUN_PREFIX]" >&2
    echo "Example: $0 0,1,2,3,4,5,6,7 prefix_intervention_20260806" >&2
    exit 2
fi

GPU_CSV="$1"
RUN_PREFIX="${2:-${PREFIX_RUN_PREFIX:-prefix_intervention_20260806}}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
LOG_DIR="${PREFIX_LOG_DIR:-${OUTPUT_ROOT}/../logs/support_aware_prefix/${RUN_PREFIX}}"
IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
NUM_SHARDS="${#GPUS[@]}"
(( NUM_SHARDS > 0 )) || { echo "FATAL: no GPUs supplied" >&2; exit 1; }

mkdir -p "${LOG_DIR}"
PIDS=()
for (( shard_index=0; shard_index<NUM_SHARDS; shard_index++ )); do
    gpu="${GPUS[shard_index]}"
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "FATAL: invalid GPU ${gpu}" >&2; exit 1; }
    log_path="${LOG_DIR}/shard_${shard_index}.log"
    echo "Launching shard ${shard_index}/${NUM_SHARDS} on GPU ${gpu}; log=${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    DTOPD_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    PREFIX_RUN_PREFIX="${RUN_PREFIX}" \
    bash "${PROJECT_ROOT}/scripts/hpc/run_support_aware_prefix_intervention_shard.sh" \
        "${shard_index}" "${NUM_SHARDS}" "${RUN_PREFIX}" \
        >"${log_path}" 2>&1 &
    PIDS+=("$!")
done

status=0
for (( shard_index=0; shard_index<NUM_SHARDS; shard_index++ )); do
    if wait "${PIDS[shard_index]}"; then
        echo "Prefix shard ${shard_index} completed"
    else
        echo "Prefix shard ${shard_index} failed; inspect ${LOG_DIR}/shard_${shard_index}.log" >&2
        status=1
    fi
done
exit "${status}"

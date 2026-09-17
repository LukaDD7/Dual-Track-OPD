#!/usr/bin/env bash
# Launch all K=32 student shards in parallel against one synchronous teacher.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 COHORT_DIR STUDENT_GPUS_CSV [NUM_SHARDS]" >&2
    echo "Example: $0 /path/k32_cohort 1,2,3,4 4" >&2
    exit 2
fi

COHORT_DIR="$1"
STUDENT_GPUS_CSV="$2"
NUM_SHARDS="${3:-${K32_NUM_SHARDS:-4}}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
RUN_PREFIX="${K32_RUN_PREFIX:-diag_full_k32_20260804}"
LOG_DIR="${K32_LOG_DIR:-${OUTPUT_ROOT}/../logs/support_aware_k32/${RUN_PREFIX}}"

IFS=',' read -r -a STUDENT_GPUS <<< "${STUDENT_GPUS_CSV}"
[[ "${#STUDENT_GPUS[@]}" -eq "${NUM_SHARDS}" ]] || {
    echo "FATAL: received ${#STUDENT_GPUS[@]} GPU entries for ${NUM_SHARDS} shards" >&2
    exit 1
}

mkdir -p "${LOG_DIR}"
PIDS=()
for (( shard_index=0; shard_index<NUM_SHARDS; shard_index++ )); do
    gpu="${STUDENT_GPUS[shard_index]}"
    log_path="${LOG_DIR}/shard_${shard_index}.log"
    echo "Launching shard ${shard_index} on CUDA_VISIBLE_DEVICES=${gpu}; log=${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    DTOPD_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    K32_RUN_PREFIX="${RUN_PREFIX}" \
    bash "${PROJECT_ROOT}/scripts/hpc/run_support_aware_k32_shard.sh" \
        "${COHORT_DIR}" "${shard_index}" "${NUM_SHARDS}" \
        >"${log_path}" 2>&1 &
    PIDS+=("$!")
done

status=0
for (( shard_index=0; shard_index<NUM_SHARDS; shard_index++ )); do
    if wait "${PIDS[shard_index]}"; then
        echo "Shard ${shard_index} completed"
    else
        echo "Shard ${shard_index} failed; inspect ${LOG_DIR}/shard_${shard_index}.log" >&2
        status=1
    fi
done

if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
fi

echo "All shards completed. Merge with:"
echo "  DTOPD_OUTPUT_ROOT=${OUTPUT_ROOT} K32_RUN_PREFIX=${RUN_PREFIX} bash ${PROJECT_ROOT}/scripts/hpc/merge_support_aware_k32.sh ${COHORT_DIR} ${NUM_SHARDS}"

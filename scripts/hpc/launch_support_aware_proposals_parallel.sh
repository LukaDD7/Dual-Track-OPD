#!/usr/bin/env bash
# Launch proposal-feasibility shards, one 32B-teacher/4B-student GPU pair each.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 GPU_PAIRS_CSV [RUN_PREFIX]" >&2
    echo "Example: $0 0:1,2:3,4:5,6:7 proposal_feasibility_20260805" >&2
    exit 2
fi

GPU_PAIRS_CSV="$1"
RUN_PREFIX="${2:-${PROPOSAL_RUN_PREFIX:-proposal_feasibility_20260805}}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
LOG_DIR="${PROPOSAL_LOG_DIR:-${OUTPUT_ROOT}/../logs/support_aware_proposals/${RUN_PREFIX}}"
IFS=',' read -r -a GPU_PAIRS <<< "${GPU_PAIRS_CSV}"
NUM_SHARDS="${#GPU_PAIRS[@]}"
(( NUM_SHARDS > 0 )) || { echo "FATAL: no GPU pairs supplied" >&2; exit 1; }

mkdir -p "${LOG_DIR}"
PIDS=()
for (( shard_index=0; shard_index<NUM_SHARDS; shard_index++ )); do
    pair="${GPU_PAIRS[shard_index]}"
    [[ "${pair}" =~ ^[0-9]+:[0-9]+$ ]] || {
        echo "FATAL: invalid GPU pair ${pair}; expected TEACHER:STUDENT" >&2
        exit 1
    }
    teacher_gpu="${pair%%:*}"
    student_gpu="${pair##*:}"
    log_path="${LOG_DIR}/shard_${shard_index}.log"
    echo "Launching shard ${shard_index}: teacher=${teacher_gpu}, student=${student_gpu}; log=${log_path}"
    CUDA_VISIBLE_DEVICES="${teacher_gpu},${student_gpu}" \
    DTOPD_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    PROPOSAL_RUN_PREFIX="${RUN_PREFIX}" \
    bash "${PROJECT_ROOT}/scripts/hpc/run_support_aware_proposal_shard.sh" \
        "${shard_index}" "${NUM_SHARDS}" "${RUN_PREFIX}" \
        >"${log_path}" 2>&1 &
    PIDS+=("$!")
done

status=0
for (( shard_index=0; shard_index<NUM_SHARDS; shard_index++ )); do
    if wait "${PIDS[shard_index]}"; then
        echo "Proposal shard ${shard_index} completed"
    else
        echo "Proposal shard ${shard_index} failed; inspect ${LOG_DIR}/shard_${shard_index}.log" >&2
        status=1
    fi
done
exit "${status}"

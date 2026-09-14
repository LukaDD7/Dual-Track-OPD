#!/usr/bin/env bash
# Phase-B2: rescore the symmetric Top-100 cache across parallel GPU pairs.
# Usage: bash rescore_symmetric_8gpu.sh STUDENT:TEACHER,STUDENT:TEACHER,...
# Example (8 GPUs): bash rescore_symmetric_8gpu.sh 0:1,2:3,4:5,6:7
# The pair order is STUDENT:TEACHER because the proxy config maps cuda:0
# (student) and cuda:1 (teacher).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 STUDENT:TEACHER,STUDENT:TEACHER,..." >&2
    echo "Example: $0 0:1,2:3,4:5,6:7" >&2
    exit 2
fi

GPU_PAIRS_CSV="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
OUT="${OUTPUT_ROOT}/support_aware_opd/reachability_proxy_20260815"
LOG_DIR="${RESCORE_LOG_DIR:-${OUTPUT_ROOT}/../logs}"

export DTOPD_OUTPUT_ROOT="${OUTPUT_ROOT}"
export DTOPD_MODEL_ROOT="${DTOPD_MODEL_ROOT:-${PROJECT_ROOT}/models}"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPU_PAIRS <<< "${GPU_PAIRS_CSV}"
NUM_SHARDS="${#GPU_PAIRS[@]}"
(( NUM_SHARDS > 0 )) || { echo "FATAL: no GPU pairs supplied" >&2; exit 1; }
for pair in "${GPU_PAIRS[@]}"; do
    [[ "${pair}" =~ ^[0-9]+:[0-9]+$ ]] || {
        echo "FATAL: invalid pair ${pair}; expected STUDENT:TEACHER" >&2
        exit 1
    }
done

UIDS="$("${PYTHON_BIN}" -c "
import json, os
path = os.path.join(os.environ['DTOPD_OUTPUT_ROOT'],
                    'support_aware_opd/proposal_feasibility_20260815_merged/retained_proposals.jsonl')
print(' '.join(sorted({json.loads(line)['sample_uid'] for line in open(path)})))
")"
echo "Rescoring $(wc -w <<< "${UIDS}") traces across ${NUM_SHARDS} shards"

PIDS=()
for (( shard_index=0; shard_index<NUM_SHARDS; shard_index++ )); do
    pair="${GPU_PAIRS[shard_index]}"
    student_gpu="${pair%%:*}"
    teacher_gpu="${pair##*:}"
    log_path="${LOG_DIR}/rescore_symmetric_s${shard_index}.log"
    echo "  shard ${shard_index}: student=${student_gpu}, teacher=${teacher_gpu}; log=${log_path}"
    CUDA_VISIBLE_DEVICES="${student_gpu},${teacher_gpu}" \
    "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.reachability_proxy score \
        --config "${PROJECT_ROOT}/configs/diagnostics/reachability_proxy_expansion.yaml" \
        --prompt-uids ${UIDS} \
        --shard-index "${shard_index}" \
        --num-shards "${NUM_SHARDS}" \
        >"${log_path}" 2>&1 &
    PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
    echo "FATAL: one or more shards failed; inspect ${LOG_DIR}/rescore_symmetric_s*.log" >&2
    exit 1
fi
echo "ALL SHARDS DONE"

TOKEN_FILES=()
for (( shard_index=0; shard_index<NUM_SHARDS; shard_index++ )); do
    TOKEN_FILES+=("${OUT}/proxy_token_rows.s${shard_index}.jsonl")
done
"${PYTHON_BIN}" -m dual_track_opd.support_aware.reachability_proxy merge-tokens \
    --token-files "${TOKEN_FILES[@]}" \
    --output-dir "${OUT}"

echo "Rescoring the 12 Experiment-B traces (legacy cache)"
CUDA_VISIBLE_DEVICES="${GPU_PAIRS[0]%%:*},${GPU_PAIRS[0]##*:}" \
"${PYTHON_BIN}" -u -m dual_track_opd.support_aware.reachability_proxy score \
    --config "${PROJECT_ROOT}/configs/diagnostics/reachability_proxy.yaml"

echo "DONE. Symmetric v2 cache is ready; salvage v2 can now run on any node."

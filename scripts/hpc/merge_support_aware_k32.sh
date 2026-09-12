#!/usr/bin/env bash
# Strictly merge and validate all K=32 shards after they finish.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 COHORT_DIR [NUM_SHARDS]" >&2
    exit 2
fi

K32_COHORT_DIR="$(cd "$1" && pwd)"
NUM_SHARDS="${2:-${K32_NUM_SHARDS:-4}}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
RUN_PREFIX="${K32_RUN_PREFIX:-diag_full_k32_20260804}"
MANIFEST="${K32_COHORT_DIR}/cohort_manifest.json"

[[ -f "${MANIFEST}" ]] || { echo "FATAL: missing ${MANIFEST}" >&2; exit 1; }
TOTAL_PROMPTS=$("${PYTHON_BIN}" -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["num_prompts"]))' \
    "${MANIFEST}")
MANIFEST_SHARDS=$("${PYTHON_BIN}" -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["num_shards"]))' \
    "${MANIFEST}")
[[ "${NUM_SHARDS}" -eq "${MANIFEST_SHARDS}" ]] || {
    echo "FATAL: NUM_SHARDS=${NUM_SHARDS} differs from cohort manifest ${MANIFEST_SHARDS}" >&2
    exit 1
}

SHARD_DIRS=()
for (( shard_index=0; shard_index<NUM_SHARDS; shard_index++ )); do
    prompt_start=$(( TOTAL_PROMPTS * shard_index / NUM_SHARDS ))
    prompt_end=$(( TOTAL_PROMPTS * (shard_index + 1) / NUM_SHARDS ))
    shard_dir="${OUTPUT_ROOT}/support_aware_opd/${RUN_PREFIX}_s${shard_index}_${prompt_start}_${prompt_end}"
    [[ -f "${shard_dir}/summary.json" ]] || {
        echo "FATAL: shard is incomplete: ${shard_dir}" >&2
        exit 1
    }
    SHARD_DIRS+=("${shard_dir}")
done

MERGED_DIR="${OUTPUT_ROOT}/support_aware_opd/${RUN_PREFIX}_merged"
if [[ ! -d "${MERGED_DIR}" ]]; then
    "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.diagnostic \
        --merge-shards "${SHARD_DIRS[@]}" \
        --merge-output "${MERGED_DIR}" \
        --merge-mode full
fi

cp "${K32_COHORT_DIR}/cohort_manifest.json" "${MERGED_DIR}/source_cohort_manifest.json"
cp "${K32_COHORT_DIR}/cohort_prompts.csv" "${MERGED_DIR}/source_cohort_prompts.csv"

exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.k32_cohort validate \
    --run-dir "${MERGED_DIR}" \
    --cohort-dir "${K32_COHORT_DIR}" \
    --expected-k 32

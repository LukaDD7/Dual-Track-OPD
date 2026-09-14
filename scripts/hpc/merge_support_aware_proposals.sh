#!/usr/bin/env bash
# Strictly merge completed teacher-proposal feasibility shards.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 NUM_SHARDS [RUN_PREFIX] [MERGED_NAME]" >&2
    exit 2
fi

NUM_SHARDS="$1"
RUN_PREFIX="${2:-${PROPOSAL_RUN_PREFIX:-proposal_feasibility_20260805}}"
MERGED_NAME="${3:-${RUN_PREFIX}_merged}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
SHARD_DIRS=()
for (( shard_index=0; shard_index<NUM_SHARDS; shard_index++ )); do
    shard_dir="${OUTPUT_ROOT}/support_aware_opd/${RUN_PREFIX}_s${shard_index}"
    [[ -f "${shard_dir}/run_manifest.json" ]] || {
        echo "FATAL: missing shard manifest ${shard_dir}" >&2
        exit 1
    }
    SHARD_DIRS+=("${shard_dir}")
done

exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.proposal_feasibility merge \
    --shard-dirs "${SHARD_DIRS[@]}" \
    --output-dir "${OUTPUT_ROOT}/support_aware_opd/${MERGED_NAME}"

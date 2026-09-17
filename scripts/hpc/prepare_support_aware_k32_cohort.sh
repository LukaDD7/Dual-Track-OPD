#!/usr/bin/env bash
# Build the immutable 64-prompt K=32 confirmation cohort on a CPU/network node.

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 SCREENING_RUN SOURCE_DATASET OUTPUT_DIR" >&2
    exit 2
fi

SCREENING_RUN="$1"
SOURCE_DATASET="$2"
OUTPUT_DIR="$3"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -c 'import pandas, pyarrow' >/dev/null 2>&1 || {
    echo "FATAL: ${PYTHON_BIN} must provide pandas and pyarrow" >&2
    exit 1
}

exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.k32_cohort build \
    --diagnostic-run "${SCREENING_RUN}" \
    --source-dataset "${SOURCE_DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --per-stratum "${K32_PER_STRATUM:-16}" \
    --length-cutoff "${K32_LENGTH_CUTOFF:-2048}" \
    --seed "${K32_SELECTION_SEED:-20260804}" \
    --num-shards "${K32_NUM_SHARDS:-4}"

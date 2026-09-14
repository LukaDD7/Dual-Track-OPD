#!/usr/bin/env bash
# CPU-only robust teacher-gap analysis on immutable K=32 exact-token arrays.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
K32_RUN="${K32_RUN_DIR:-${OUTPUT_ROOT}/support_aware_opd/diag_full_k32_20260804_merged}"
ANALYSIS_DIR="${GAP_ANALYSIS_DIR:-${OUTPUT_ROOT}/support_aware_opd/gap_robustness_20260806}"

exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.gap_robustness \
    --run-dir "${K32_RUN}" \
    --output-dir "${ANALYSIS_DIR}" \
    --prefix-horizons 128 256 512 \
    --bootstrap-seed 42 \
    --bootstrap-resamples 10000

#!/usr/bin/env bash
# CPU-only offline TOPD-style OT probe on immutable cached trajectories.
# Optional SMOKE=1 reduces to 2 prompts / K=16 windows / stride 8 and
# validates the summary before the full run.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
MODEL_ROOT="${DTOPD_MODEL_ROOT:-${PROJECT_ROOT}/../models}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/experiment/support_aware_offline_topd_probe.yaml}"
OUTPUT_DIR="${OUTPUT_ROOT}/support_aware_opd/offline_topd_probe_20260806"

if [[ "${SMOKE:-0}" == "1" ]]; then
  exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.offline_topd_probe \
    run \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}_smoke" \
    --max-prompts 2 \
    --window-size 16 \
    --stride 8 \
    --bootstrap-resamples 200
fi

exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.offline_topd_probe \
  run \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}"

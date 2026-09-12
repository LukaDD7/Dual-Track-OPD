#!/usr/bin/env bash
# Smoke-test Vision-OPD-6K adapter behavior without teacher service or training.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"
DATASET="${DATASET:-${PROJECT_ROOT}/third_party/Vision-OPD/data/train.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${DTOPD_OUTPUT_ROOT:-${REPO_ROOT}/artifacts}/fc_opd/vision_opd_adapter_smoke}"

args=(
  "${REPO_ROOT}/scripts/hpc/run_fc_opd_vision_opd_adapter_smoke.py"
  --project-root "${PROJECT_ROOT}"
  --dataset-type "${DATASET_TYPE:-vision_opd_parquet}"
  --source-dataset "${SOURCE_DATASET:-vision-opd-6k}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -f "${DATASET}" ]]; then
  args+=(--dataset "${DATASET}")
fi
if [[ "${MATERIALIZE_DEGRADED_IMAGES:-1}" != "1" ]]; then
  args+=(--no-materialize-degraded-images)
fi

python "${args[@]}"

#!/usr/bin/env bash
# Audit Vision-OPD images for red-box localization contamination.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

: "${DATASET:?set DATASET to Vision-OPD train.parquet}"
: "${DTOPD_OUTPUT_ROOT:?set DTOPD_OUTPUT_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-${DTOPD_OUTPUT_ROOT}/fc_opd/red_box_audit/vision_opd6k}"

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi

python "${REPO_ROOT}/scripts/hpc/audit_vision_opd_red_box_contamination.py" \
  --dataset "${DATASET}" \
  --dataset-type "${DATASET_TYPE:-vision_opd_parquet}" \
  --source-dataset "${SOURCE_DATASET:-vision-opd-6k}" \
  --red-threshold "${RED_THRESHOLD:-0.002}" \
  --output-jsonl "${OUTPUT_DIR}/red_box_audit.jsonl" \
  --summary-json "${OUTPUT_DIR}/red_box_audit_summary.json" \
  --contact-sheet "${OUTPUT_DIR}/red_box_contact_sheet.jpg" \
  --examples-dir "${OUTPUT_DIR}/examples" \
  --num-examples "${NUM_EXAMPLES:-32}" \
  "${EXTRA_ARGS[@]}"

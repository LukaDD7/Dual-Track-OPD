#!/usr/bin/env bash
# Run a small FC-OPD condition-signal audit before selecting training data.
#
# Required:
#   DATASET=/path/to/train.jsonl or .json or .parquet
#   SOURCE_DATASET=vision_opd_6k | geometry3k | virl39k | ...
#
# Optional:
#   DATASET_TYPE=auto
#   LIMIT=128
#   TEACHER_URL=http://127.0.0.1:18080
#   TOKENIZER="hf:${DTOPD_MODEL_ROOT}/Qwen3-VL-4B-Instruct"
#   OUTPUT_DIR="${DTOPD_OUTPUT_ROOT}/fc_opd/dataset_signal_audit/${SOURCE_DATASET}"
#   DEGRADED_DIR=/path/to/blurred/images
#   TASK_EVIDENCE_MODE=none
#   DRY_RUN=1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${DATASET:?set DATASET to a candidate JSON/JSONL/Parquet file}"
: "${SOURCE_DATASET:?set SOURCE_DATASET to the dataset name}"

DATASET_TYPE="${DATASET_TYPE:-auto}"
LIMIT="${LIMIT:-128}"
TEACHER_URL="${TEACHER_URL:-http://127.0.0.1:18080}"
TOKENIZER="${TOKENIZER:-hf:\$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${DTOPD_OUTPUT_ROOT:-${REPO_ROOT}/artifacts}/fc_opd/dataset_signal_audit/${SOURCE_DATASET}}"
CONDITIONS="${CONDITIONS:-full,blur,free,task}"
BLUR_SIGMA="${BLUR_SIGMA:-2.0}"
TASK_EVIDENCE_MODE="${TASK_EVIDENCE_MODE:-none}"

args=(
  "${REPO_ROOT}/scripts/hpc/run_fc_opd_dataset_signal_audit.py"
  --dataset "${DATASET}"
  --dataset-type "${DATASET_TYPE}"
  --source-dataset "${SOURCE_DATASET}"
  --limit "${LIMIT}"
  --teacher-url "${TEACHER_URL}"
  --tokenizer "${TOKENIZER}"
  --conditions "${CONDITIONS}"
  --blur-sigma "${BLUR_SIGMA}"
  --task-evidence-mode "${TASK_EVIDENCE_MODE}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${DEGRADED_DIR:-}" ]]; then
  args+=(--degraded-dir "${DEGRADED_DIR}")
fi
if [[ "${MATERIALIZE_DEGRADED_IMAGES:-0}" == "1" ]]; then
  args+=(--materialize-degraded-images)
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  args+=(--skip-existing)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

python "${args[@]}"

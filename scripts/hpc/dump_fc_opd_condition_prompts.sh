#!/usr/bin/env bash
# Dump exact FC-OPD teacher prompts for a few candidate dataset records.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"
SOURCE_DATASET="${SOURCE_DATASET:-vision-opd-6k}"
DATASET="${DATASET:-${PROJECT_ROOT}/third_party/Vision-OPD/data/train.parquet}"
DATASET_TYPE="${DATASET_TYPE:-auto}"
LIMIT="${LIMIT:-3}"
CONDITIONS="${CONDITIONS:-full,blur,free,task}"
BLUR_SIGMA="${BLUR_SIGMA:-2.0}"
TASK_EVIDENCE_MODE="${TASK_EVIDENCE_MODE:-none}"
OUTPUT="${OUTPUT:-${DTOPD_OUTPUT_ROOT:-${REPO_ROOT}/artifacts}/fc_opd/condition_prompt_audit/${SOURCE_DATASET}_condition_prompts.md}"

args=(
  "${REPO_ROOT}/scripts/hpc/dump_fc_opd_condition_prompts.py"
  --dataset "${DATASET}"
  --dataset-type "${DATASET_TYPE}"
  --source-dataset "${SOURCE_DATASET}"
  --limit "${LIMIT}"
  --conditions "${CONDITIONS}"
  --blur-sigma "${BLUR_SIGMA}"
  --task-evidence-mode "${TASK_EVIDENCE_MODE}"
  --output "${OUTPUT}"
  --include-images-as-paths
)

if [[ -n "${DEGRADED_DIR:-}" ]]; then
  args+=(--degraded-dir "${DEGRADED_DIR}")
fi
if [[ "${MATERIALIZE_DEGRADED_IMAGES:-0}" == "1" ]]; then
  args+=(--materialize-degraded-images)
fi

python "${args[@]}"

#!/usr/bin/env bash
# Generate student rollouts and teacher-force score them under FC-OPD conditions.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${DATASET:?set DATASET to Vision-OPD train.parquet}"
: "${DTOPD_OUTPUT_ROOT:?set DTOPD_OUTPUT_ROOT}"

python "${REPO_ROOT}/scripts/hpc/run_fc_opd_student_rollout_signal_audit.py" \
  --dataset "${DATASET}" \
  --dataset-type "${DATASET_TYPE:-vision_opd_parquet}" \
  --source-dataset "${SOURCE_DATASET:-vision-opd-6k}" \
  --limit "${LIMIT:-4}" \
  --conditions "${CONDITIONS:-full,blur}" \
  --teacher-url "${TEACHER_URL:-http://127.0.0.1:18080}" \
  --student-model-path "${STUDENT_MODEL_PATH:-hf:\$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct}" \
  --rollouts-per-prompt "${ROLLOUTS_PER_PROMPT:-2}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.9}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --seed "${SEED:-42}" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-bfloat16}" \
  --rollout-response-format "${ROLLOUT_RESPONSE_FORMAT:-fc_opd_structured}" \
  --min-response-tokens-for-warning "${MIN_RESPONSE_TOKENS_FOR_WARNING:-16}" \
  --blur-sigma "${BLUR_SIGMA:-2.0}" \
  --materialize-degraded-images \
  --output-dir "${OUTPUT_DIR:-${DTOPD_OUTPUT_ROOT}/fc_opd/student_rollout_signal_audit}"

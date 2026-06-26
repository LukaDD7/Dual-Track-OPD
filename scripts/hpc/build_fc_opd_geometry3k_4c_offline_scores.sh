#!/usr/bin/env bash
# Build Geometry3K clean-data 4C trainable offline scores.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

: "${DATASET:?set DATASET to Geometry3K data file}"
: "${EVIDENCE_CACHE:?set EVIDENCE_CACHE to evidence_cache.jsonl}"
: "${DTOPD_OUTPUT_ROOT:?set DTOPD_OUTPUT_ROOT}"

LIMIT="${LIMIT:-8}"
ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-${DTOPD_OUTPUT_ROOT}/fc_opd/offline_scores/geometry3k_4c_limit${LIMIT}_k${ROLLOUTS_PER_PROMPT}}"

EXTRA_ARGS=()
if [[ -n "${END_INDEX:-}" ]]; then EXTRA_ARGS+=(--end-index "${END_INDEX}"); fi
if [[ -n "${DEGRADED_DIR:-}" ]]; then EXTRA_ARGS+=(--degraded-dir "${DEGRADED_DIR}"); fi
if [[ -n "${RESUME:-}" ]]; then EXTRA_ARGS+=(--resume); fi
if [[ -n "${SKIP_EXISTING:-}" ]]; then EXTRA_ARGS+=(--skip-existing); fi

python "${REPO_ROOT}/scripts/hpc/build_fc_opd_geometry3k_4c_offline_scores.py" \
  --dataset "${DATASET}" \
  --evidence-cache "${EVIDENCE_CACHE}" \
  --dataset-type geometry3k \
  --source-dataset "${SOURCE_DATASET:-geometry3k}" \
  --student-model-path "${STUDENT_MODEL_PATH:-hf:\$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct}" \
  --teacher-url "${TEACHER_URL:-http://127.0.0.1:18080}" \
  --limit "${LIMIT}" \
  --start-index "${START_INDEX:-0}" \
  --rollouts-per-prompt "${ROLLOUTS_PER_PROMPT}" \
  --rollout-response-format "${ROLLOUT_RESPONSE_FORMAT:-fc_opd_structured}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.9}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --seed "${SEED:-42}" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-bfloat16}" \
  --degraded-mode "${DEGRADED_MODE:-lowres_10pct_nearest}" \
  --output-jsonl "${OUTPUT_DIR}/geometry3k_4c_offline_scores.jsonl" \
  --summary-json "${OUTPUT_DIR}/geometry3k_4c_offline_scores_summary.json" \
  "${EXTRA_ARGS[@]}"

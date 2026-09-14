#!/usr/bin/env bash
# Build free/task evidence cache for clean-data 4C FC-OPD.
# Keep pipefail enabled when piping this command to tee, so Python load/generation
# failures are not hidden by tee's exit code.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

: "${DATASET:?set DATASET to Geometry3K/ViRL39K/generic VQA file}"
if [[ -z "${DRY_RUN_INSPECT:-}" ]]; then
  : "${DTOPD_OUTPUT_ROOT:?set DTOPD_OUTPUT_ROOT}"
fi

SOURCE_DATASET="${SOURCE_DATASET:-geometry3k}"
DATASET_TYPE="${DATASET_TYPE:-geometry3k}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-/tmp/dtopd_dry_run}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/fc_opd/evidence_cache/${SOURCE_DATASET}_4c_limit${LIMIT:-8}}"

EXTRA_ARGS=()
if [[ -n "${GENERATOR_MODEL_PATH:-}" ]]; then
  EXTRA_ARGS+=(--generator-model-path "${GENERATOR_MODEL_PATH}")
fi
if [[ -n "${GENERATOR_URL:-}" ]]; then
  EXTRA_ARGS+=(--generator-url "${GENERATOR_URL}")
fi
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${END_INDEX:-}" ]]; then
  EXTRA_ARGS+=(--end-index "${END_INDEX}")
fi
if [[ -n "${RESUME:-}" ]]; then
  EXTRA_ARGS+=(--resume)
fi
if [[ -n "${SKIP_EXISTING:-}" ]]; then
  EXTRA_ARGS+=(--skip-existing)
fi
if [[ -n "${DRY_RUN_INSPECT:-}" ]]; then
  EXTRA_ARGS+=(--dry-run-inspect)
fi

python "${REPO_ROOT}/scripts/hpc/build_fc_opd_4c_evidence_cache.py" \
  --dataset "${DATASET}" \
  --dataset-type "${DATASET_TYPE}" \
  --source-dataset "${SOURCE_DATASET}" \
  --start-index "${START_INDEX:-0}" \
  --seed "${SEED:-42}" \
  --temperature "${TEMPERATURE:-0.2}" \
  --top-p "${TOP_P:-0.9}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --output-jsonl "${OUTPUT_DIR}/evidence_cache.jsonl" \
  --summary-json "${OUTPUT_DIR}/evidence_cache_summary.json" \
  "${EXTRA_ARGS[@]}"

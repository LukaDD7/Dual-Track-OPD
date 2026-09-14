#!/usr/bin/env bash
# Build trainable Vision-OPD-6K FC-OPD 2C offline scores and optionally run the
# real-student optimizer-step smoke. No verl integration is used.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

: "${DATASET:?set DATASET to Vision-OPD train.parquet}"
: "${DTOPD_OUTPUT_ROOT:?set DTOPD_OUTPUT_ROOT}"

MODE="${1:-build}"
LIMIT="${LIMIT:-4}"
ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-2}"

case "${MODE}" in
  limit4_k2)
    LIMIT=4
    ROLLOUTS_PER_PROMPT=2
    ;;
  limit16_k4)
    LIMIT=16
    ROLLOUTS_PER_PROMPT=4
    ;;
  build|validate|real-min-train)
    ;;
  *)
    echo "usage: $0 [build|limit4_k2|limit16_k4|validate|real-min-train]" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${DTOPD_OUTPUT_ROOT}/fc_opd/offline_scores/vision_opd6k_2c_structured_limit${LIMIT}_k${ROLLOUTS_PER_PROMPT}}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/vision_opd6k_2c_offline_scores.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_DIR}/vision_opd6k_2c_offline_scores_summary.json}"
MODEL_PATH="${STUDENT_MODEL_PATH:-hf:\$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct}"

if [[ "${MODE}" == "validate" ]]; then
  python "${REPO_ROOT}/scripts/hpc/build_fc_opd_vision_opd_2c_offline_scores.py" \
    --dataset "${DATASET}" \
    --output-jsonl "${OUTPUT_JSONL}" \
    --summary-json "${SUMMARY_JSON}" \
    --validate-only
  exit $?
fi

if [[ "${MODE}" == "real-min-train" ]]; then
  python "${REPO_ROOT}/scripts/hpc/run_fc_opd_real_student_min_train_smoke.py" \
    --scores "${OUTPUT_JSONL}" \
    --model-path "${FC_OPD_STUDENT_MODEL:-${DTOPD_MODEL_ROOT}/Qwen3-VL-4B-Instruct}" \
    --condition-set 2c \
    --limit "${FC_OPD_REAL_STUDENT_LIMIT:-2}" \
    --steps "${FC_OPD_REAL_STUDENT_STEPS:-1}" \
    --lr "${FC_OPD_REAL_STUDENT_LR:-1e-4}" \
    --device "${FC_OPD_STUDENT_DEVICE:-cuda}" \
    --dtype "${FC_OPD_STUDENT_DTYPE:-bfloat16}" \
    --freeze-all-but-lm-head
  exit $?
fi

EXTRA_ARGS=()
if [[ -n "${END_INDEX:-}" ]]; then
  EXTRA_ARGS+=(--end-index "${END_INDEX}")
fi
if [[ -n "${MATERIALIZE_DEGRADED_IMAGES:-}" ]]; then
  EXTRA_ARGS+=(--materialize-degraded-images)
fi
if [[ -n "${DEGRADED_DIR:-}" ]]; then
  EXTRA_ARGS+=(--degraded-dir "${DEGRADED_DIR}")
fi
if [[ -n "${SKIP_EXISTING:-}" ]]; then
  EXTRA_ARGS+=(--skip-existing)
fi
if [[ -n "${RESUME:-}" ]]; then
  EXTRA_ARGS+=(--resume)
fi
if [[ -n "${SHARD_ID:-}" ]]; then
  EXTRA_ARGS+=(--shard-id "${SHARD_ID}")
fi
if [[ -n "${NUM_SHARDS:-}" ]]; then
  EXTRA_ARGS+=(--num-shards "${NUM_SHARDS}")
fi
if [[ -n "${ALLOW_RED_BOX_CONTAMINATED_IMAGES:-}" ]]; then
  EXTRA_ARGS+=(--allow-red-box-contaminated-images)
fi

python "${REPO_ROOT}/scripts/hpc/build_fc_opd_vision_opd_2c_offline_scores.py" \
  --dataset "${DATASET}" \
  --dataset-type "${DATASET_TYPE:-vision_opd_parquet}" \
  --source-dataset "${SOURCE_DATASET:-vision-opd-6k}" \
  --student-model-path "${MODEL_PATH}" \
  --teacher-url "${TEACHER_URL:-http://127.0.0.1:18080}" \
  --conditions "${CONDITIONS:-full,blur}" \
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
  --blur-sigma "${BLUR_SIGMA:-2.0}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  "${EXTRA_ARGS[@]}"

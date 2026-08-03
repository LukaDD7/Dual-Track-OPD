#!/usr/bin/env bash
# Run or safely resume one exact-token K=32 shard.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 COHORT_DIR SHARD_INDEX [NUM_SHARDS]" >&2
    exit 2
fi

K32_COHORT_DIR="$(cd "$1" && pwd)"
SHARD_INDEX="$2"
NUM_SHARDS="${3:-${K32_NUM_SHARDS:-4}}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"
CONFIG="${K32_CONFIG:-${PROJECT_ROOT}/configs/experiment/support_aware_geometry3k_k32.yaml}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
RUN_PREFIX="${K32_RUN_PREFIX:-diag_full_k32_20260804}"
TEACHER_URL="${K32_TEACHER_URL:-http://127.0.0.1:18080}"
STUDENT_MODEL="${K32_STUDENT_MODEL:-${DTOPD_MODEL_ROOT:-${PROJECT_ROOT}/models}/Qwen3-VL-4B-Instruct}"
MANIFEST="${K32_COHORT_DIR}/cohort_manifest.json"
DATASET="${K32_COHORT_DIR}/cohort.parquet"

for path in "${MANIFEST}" "${DATASET}" "${CONFIG}"; do
    [[ -f "${path}" ]] || { echo "FATAL: missing ${path}" >&2; exit 1; }
done
[[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]] || { echo "FATAL: SHARD_INDEX must be an integer" >&2; exit 1; }
[[ "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: NUM_SHARDS must be positive" >&2; exit 1; }
(( SHARD_INDEX < NUM_SHARDS )) || { echo "FATAL: shard index is outside [0, NUM_SHARDS)" >&2; exit 1; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
    echo "FATAL: set CUDA_VISIBLE_DEVICES explicitly for this student shard" >&2
    exit 1
}

TOTAL_PROMPTS=$("${PYTHON_BIN}" -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["num_prompts"]))' \
    "${MANIFEST}")
MANIFEST_SHARDS=$("${PYTHON_BIN}" -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["num_shards"]))' \
    "${MANIFEST}")
[[ "${NUM_SHARDS}" -eq "${MANIFEST_SHARDS}" ]] || {
    echo "FATAL: NUM_SHARDS=${NUM_SHARDS} differs from cohort manifest ${MANIFEST_SHARDS}" >&2
    exit 1
}

PROMPT_START=$(( TOTAL_PROMPTS * SHARD_INDEX / NUM_SHARDS ))
PROMPT_END=$(( TOTAL_PROMPTS * (SHARD_INDEX + 1) / NUM_SHARDS ))
RUN_ID="${RUN_PREFIX}_s${SHARD_INDEX}_${PROMPT_START}_${PROMPT_END}"
RUN_DIR="${OUTPUT_ROOT}/support_aware_opd/${RUN_ID}"

export K32_COHORT_DIR DTOPD_OUTPUT_ROOT="${OUTPUT_ROOT}"
export DTOPD_MODEL_ROOT="${DTOPD_MODEL_ROOT:-$(dirname "${STUDENT_MODEL}")}"

RESUME_ARGS=()
if [[ -f "${RUN_DIR}/run_manifest.json" && -f "${RUN_DIR}/summary.json" ]]; then
    EXIT_STATUS=$("${PYTHON_BIN}" -c \
        'import json,sys; print(json.load(open(sys.argv[1])).get("exit_status", ""))' \
        "${RUN_DIR}/run_manifest.json")
    if [[ "${EXIT_STATUS}" == "PASS" || "${EXIT_STATUS}" == "GATE_FAIL" ]]; then
        echo "Shard already complete: ${RUN_DIR} (${EXIT_STATUS})"
        exit 0
    fi
fi
if [[ -f "${RUN_DIR}/prompt_support_summary.jsonl" ]]; then
    RESUME_ARGS=(--resume "${RUN_ID}")
elif [[ -d "${RUN_DIR}" ]]; then
    echo "FATAL: existing run dir has no resumable prompt summary: ${RUN_DIR}" >&2
    exit 1
else
    RESUME_ARGS=(--run-id "${RUN_ID}")
fi

echo "K=32 shard ${SHARD_INDEX}/${NUM_SHARDS}: prompts [${PROMPT_START}, ${PROMPT_END})"
echo "Run directory: ${RUN_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

exec bash "${PROJECT_ROOT}/scripts/hpc/run_support_aware_diagnostic.sh" \
    --config "${CONFIG}" \
    --mode full \
    --teacher-url "${TEACHER_URL}" \
    --student-model "${STUDENT_MODEL}" \
    --dataset "${DATASET}" \
    --output-root "${OUTPUT_ROOT}" \
    --num-prompts "${TOTAL_PROMPTS}" \
    --prompt-start "${PROMPT_START}" \
    --prompt-end "${PROMPT_END}" \
    --device cuda \
    --dtype bfloat16 \
    --exit-zero-on-complete \
    "${RESUME_ARGS[@]}"

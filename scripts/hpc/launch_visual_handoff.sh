#!/usr/bin/env bash
# Visual Handoff Diagnostic (frozen plan Question A):
#   full/degraded counterfactual scoring (teacher+student, fixed trace) ->
#   reasoning-block DeltaV curve -> BIC change point -> continuation
#   validation at h_V +/- 1 block.  Sharded across STUDENT:TEACHER GPU pairs,
#   then merged into the Visual Handoff Report.  No training, no RL.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 STUDENT:TEACHER,STUDENT:TEACHER,..." >&2
    echo "Example: $0 0:1,2:3,4:5,6:7" >&2
    exit 2
fi

GPU_PAIRS_CSV="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
OUT="${OUTPUT_ROOT}/support_aware_opd/${VISUAL_HANDOFF_PREFIX:-visual_handoff_js_v3_20260817}"
COHORT_DIR="${VISUAL_HANDOFF_COHORT_DIR:-${OUTPUT_ROOT}/support_aware_opd/k32_cohort_20260804}"
COHORT_PARQUET="${VISUAL_HANDOFF_COHORT_PARQUET:-${OUTPUT_ROOT}/fc_opd/geometry3k_full/train.parquet}"
LOG_DIR="${VISUAL_HANDOFF_LOG_DIR:-${OUTPUT_ROOT}/../logs/visual_handoff}"
CONTINUATION_K="${VISUAL_HANDOFF_K:-4}"
MAX_TOKENS="${VISUAL_HANDOFF_MAX_TOKENS:-256}"
MAX_PROMPTS="${VISUAL_HANDOFF_MAX_PROMPTS:-}"
STUDENT_MODEL="${DTOPD_MODEL_ROOT}/Qwen3-VL-4B-Instruct"
TEACHER_MODEL="${DTOPD_MODEL_ROOT}/Qwen3-VL-32B-Instruct"

PROPOSAL_DIRS=(
    "${OUTPUT_ROOT}/support_aware_opd/proposal_feasibility_20260805_merged"
    "${OUTPUT_ROOT}/support_aware_opd/proposal_feasibility_20260815_merged"
    "${OUTPUT_ROOT}/support_aware_opd/proposal_feasibility_20260816_merged"
)
RESCUE_DIRS=(
    "${OUTPUT_ROOT}/support_aware_opd/prefix_intervention_20260806_merged"
    "${OUTPUT_ROOT}/support_aware_opd/prefix_intervention_20260815_merged"
    "${OUTPUT_ROOT}/support_aware_opd/prefix_intervention_20260815b_merged"
    "${OUTPUT_ROOT}/support_aware_opd/prefix_intervention_20260816_merged"
)

mkdir -p "${OUT}" "${LOG_DIR}"
IFS=',' read -r -a GPU_PAIRS <<< "${GPU_PAIRS_CSV}"
NUM_SHARDS="${#GPU_PAIRS[@]}"
(( NUM_SHARDS > 0 )) || { echo "FATAL: no GPU pairs" >&2; exit 1; }

echo "== Visual Handoff: ${NUM_SHARDS} shards, output ${OUT} =="
PIDS=()
for (( i=0; i<NUM_SHARDS; i++ )); do
    pair="${GPU_PAIRS[i]}"
    [[ "${pair}" =~ ^[0-9]+:[0-9]+$ ]] || { echo "FATAL: invalid pair ${pair}" >&2; exit 1; }
    student_gpu="${pair%%:*}"
    teacher_gpu="${pair##*:}"
    log="${LOG_DIR}/visual_handoff_s${i}.log"
    max_prompt_args=()
    if [[ -n "${MAX_PROMPTS}" ]]; then max_prompt_args=(--max-prompts "${MAX_PROMPTS}"); fi
    CUDA_VISIBLE_DEVICES="${student_gpu},${teacher_gpu}" \
    "${PYTHON_BIN}" -m dual_track_opd.support_aware.visual_handoff run \
        --student-model "${STUDENT_MODEL}" \
        --teacher-model "${TEACHER_MODEL}" \
        --student-device cuda:0 \
        --teacher-device cuda:1 \
        --cohort-dir "${COHORT_DIR}" \
        --cohort-parquet-path "${COHORT_PARQUET}" \
        --proposal-dirs "${PROPOSAL_DIRS[@]}" \
        --rescue-dirs "${RESCUE_DIRS[@]}" \
        --output-dir "${OUT}" \
        --continuation-k "${CONTINUATION_K}" \
        --max-continuation-tokens "${MAX_TOKENS}" \
        --shard-index "${i}" \
        --num-shards "${NUM_SHARDS}" \
        "${max_prompt_args[@]}" \
        >"${log}" 2>&1 &
    PIDS+=("$!")
    echo "  shard ${i}: pair=${pair}; log=${log}"
done
for pid in "${PIDS[@]}"; do wait "${pid}"; done

echo "== Merge Visual Handoff Report =="
"${PYTHON_BIN}" -m dual_track_opd.support_aware.visual_handoff merge \
    --shard-dir "${OUT}" \
    --output-dir "${OUT}"

echo "DONE. See ${OUT}/visual_handoff_report.json"

#!/usr/bin/env bash
# Real student optimizer-step smoke for the offline FC-OPD loss.
#
# Loads a real Qwen3-VL / Qwen3.5-VL student, runs a teacher-forced multimodal
# forward on offline-score records, computes the FC-OPD loss, backpropagates, and
# runs a few Adam steps, verifying that a real trainable parameter changes.
# Defaults to a single record, 3 steps, bfloat16 on CUDA, and
# --freeze-all-but-lm-head. No teacher service required; third_party/verl
# untouched.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCORES="${1:-${DTOPD_OUTPUT_ROOT}/fc_opd/offline_scores/vstar16_real_teacher/vstar_offline_scores.jsonl}"
MODEL_PATH="${FC_OPD_STUDENT_MODEL:-${DTOPD_MODEL_ROOT}/Qwen3-VL-4B-Instruct}"
LIMIT="${FC_OPD_REAL_STUDENT_LIMIT:-1}"
STEPS="${FC_OPD_REAL_STUDENT_STEPS:-3}"
LR="${FC_OPD_REAL_STUDENT_LR:-1e-4}"
DEVICE="${FC_OPD_STUDENT_DEVICE:-cuda}"
DTYPE="${FC_OPD_STUDENT_DTYPE:-bfloat16}"

EXTRA_ARGS=()
if [[ -n "${FC_OPD_MAX_PROMPT_LENGTH:-}" ]]; then
  EXTRA_ARGS+=(--max-prompt-length "${FC_OPD_MAX_PROMPT_LENGTH}")
fi
if [[ -n "${FC_OPD_MAX_RESPONSE_TOKENS:-}" ]]; then
  EXTRA_ARGS+=(--max-response-tokens "${FC_OPD_MAX_RESPONSE_TOKENS}")
fi
if [[ "${FC_OPD_FREEZE_ALL_BUT_LM_HEAD:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-freeze-all-but-lm-head)
fi

python "${REPO_ROOT}/scripts/hpc/run_fc_opd_real_student_min_train_smoke.py" \
  --scores "${SCORES}" \
  --model-path "${MODEL_PATH}" \
  --limit "${LIMIT}" \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  "${EXTRA_ARGS[@]}"

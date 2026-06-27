#!/usr/bin/env bash
# Real-student optimizer-step smoke for Geometry3K clean-data 4C offline scores.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

: "${SCORES:?set SCORES to Geometry3K 4C offline-score JSONL}"

python "${REPO_ROOT}/scripts/hpc/run_fc_opd_real_student_min_train_smoke.py" \
  --scores "${SCORES}" \
  --model-path "${FC_OPD_STUDENT_MODEL:-${DTOPD_MODEL_ROOT}/Qwen3-VL-4B-Instruct}" \
  --condition-set "${FC_OPD_CONDITION_SET:-4c-clean}" \
  --routing-mode "${FC_OPD_ROUTING_MODE:-uniform_all_conditions}" \
  --limit "${FC_OPD_REAL_STUDENT_LIMIT:-2}" \
  --steps "${FC_OPD_REAL_STUDENT_STEPS:-1}" \
  --lr "${FC_OPD_REAL_STUDENT_LR:-1e-4}" \
  --device "${FC_OPD_STUDENT_DEVICE:-cuda}" \
  --dtype "${FC_OPD_STUDENT_DTYPE:-bfloat16}" \
  --freeze-all-but-lm-head

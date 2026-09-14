#!/usr/bin/env bash
# Minimal Adam optimizer-update smoke for the offline FC-OPD loss.
#
# Attaches synthetic student logits (requires_grad) to a recorded offline-score
# dataset and runs Adam for a few steps against the recorded teacher scores.
# Reports per-step loss / grad_norm / logits_delta_norm / consumed_conditions and
# verifies finite loss and gradients, that every optimizer.step() moves the
# logits, and that all four conditions are consumed. No GPU, model, or teacher
# service required, and third_party/verl is untouched.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCORES="${1:-${DTOPD_OUTPUT_ROOT}/fc_opd/offline_scores/vstar16_real_teacher/vstar_offline_scores.jsonl}"
STEPS="${FC_OPD_MIN_TRAIN_STEPS:-10}"

python "${REPO_ROOT}/scripts/hpc/run_fc_opd_min_train_smoke.py" \
  --scores "${SCORES}" \
  --steps "${STEPS}"

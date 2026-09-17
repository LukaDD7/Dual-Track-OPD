#!/usr/bin/env bash
# Offline FC-OPD loss/backward smoke over a recorded offline-score dataset.
#
# Reads recorded teacher scores, rebuilds tensors and chunk masks, attaches
# synthetic student logits (requires_grad), and runs the real router -> loss ->
# backward path. Verifies finite loss, present/finite gradients, aligned chunk
# masks, and that all four conditions are consumed. No GPU, model, or teacher
# service required, and third_party/verl is untouched.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCORES="${1:-${DTOPD_OUTPUT_ROOT}/fc_opd/offline_scores/vstar16_real_teacher/vstar_offline_scores.jsonl}"

python "${REPO_ROOT}/scripts/hpc/run_fc_opd_offline_loss_smoke.py" \
  --scores "${SCORES}"

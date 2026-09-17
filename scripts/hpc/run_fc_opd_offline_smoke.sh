#!/usr/bin/env bash
# Self-contained offline FC-OPD scoring smoke run.
#
# Builds offline scores for the first 16 synthetic VStar samples against an
# in-process synthetic teacher (top-k 32), writes a JSONL dataset, and verifies
# that every sample has all four conditions with [T, 32] tensors. Requires no
# GPU, no model weights, and no external teacher service.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${DTOPD_OFFLINE_SMOKE_DIR:-${TMPDIR:-/tmp}/fc_opd_offline_smoke}"

python "${REPO_ROOT}/scripts/hpc/build_fc_opd_offline_scores.py" \
  --self-contained-smoke \
  --limit 16 \
  --source-dataset vstar \
  --mode protocol_smoke \
  --tokenizer byte \
  --smoke-top-k 32 \
  --output-dir "${OUTPUT_DIR}" \
  --output-name vstar_offline_scores_smoke

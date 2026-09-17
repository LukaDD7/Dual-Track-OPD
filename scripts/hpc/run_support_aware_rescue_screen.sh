#!/usr/bin/env bash
# Adaptive rescue screen on fresh Geometry3K prompts (frozen policy).
# Uses GPUs 0-1 by default (teacher cuda:0, student cuda:1); shard across
# GPU pairs with --shard-index/--num-shards if a full cohort is screened.
# SMOKE=1 reduces to 1 prompt / 1 proposal / K=2 / max 256.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs}"
MODEL_ROOT="${DTOPD_MODEL_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/experiment/support_aware_rescue_screen.yaml}"
COHORT="${COHORT:-${OUTPUT_ROOT}/fc_opd/geometry3k_gkd/rescue_screen_cohort_20260806.parquet}"
OUTPUT_DIR="${OUTPUT_ROOT}/support_aware_opd/rescue_screen_20260806"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.rescue_screen \
    run --config "${CONFIG}" \
    --cohort-path "${COHORT}" \
    --output-dir "${OUTPUT_DIR}_smoke" \
    --max-prompts 1 \
    --proposals-per-prompt 1 \
    --stage1-k 2 \
    --stage2-k 2 \
    --wrong-source-k 4 \
    --max-continuation-tokens 256 \
    --teacher-device cuda:0 \
    --student-device cuda:1
fi

exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.rescue_screen \
  run --config "${CONFIG}" \
  --cohort-path "${COHORT}" \
  --output-dir "${OUTPUT_DIR}" \
  --teacher-device cuda:0 \
  --student-device cuda:1 \
  --shard-index "${SHARD_INDEX}" \
  --num-shards "${NUM_SHARDS}"

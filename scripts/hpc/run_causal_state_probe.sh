#!/usr/bin/env bash
# Frozen-policy causal state localization on one student GPU + one teacher GPU.
#
# Examples:
#   MODE=preflight bash scripts/hpc/run_causal_state_probe.sh
#   CUDA_VISIBLE_DEVICES=3,4 SMOKE=1 bash scripts/hpc/run_causal_state_probe.sh
#   CUDA_VISIBLE_DEVICES=3,4 bash scripts/hpc/run_causal_state_probe.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER_ROOT="${DTOPD_CLUSTER_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
PYTHON_BIN="${DTOPD_PYTHON:-${CLUSTER_ROOT}/envs/va-opd-native-e003-cu128-r595-v1/bin/python}"
MODEL_ROOT="${DTOPD_MODEL_ROOT:-${CLUSTER_ROOT}/models}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${CLUSTER_ROOT}/fc-opd-storage/outputs}"
CONFIG="${CAUSAL_PROBE_CONFIG:-${PROJECT_ROOT}/configs/diagnostics/causal_state_probe.yaml}"
OUTPUT_DIR="${CAUSAL_PROBE_OUTPUT_DIR:-${OUTPUT_ROOT}/support_aware_opd/causal_state_probe_20260808}"
MODE="${MODE:-run}"

[[ -x "${PYTHON_BIN}" ]] || { echo "FATAL: Python environment missing: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "FATAL: config missing: ${CONFIG}" >&2; exit 1; }

export DTOPD_MODEL_ROOT="${MODEL_ROOT}"
export DTOPD_OUTPUT_ROOT="${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

COMMON=(
  --config "${CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --student-device cuda:0
  --teacher-device cuda:1
  --shard-index "${SHARD_INDEX:-0}"
  --num-shards "${NUM_SHARDS:-1}"
)

if [[ "${MODE}" == "preflight" ]]; then
  exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.causal_state_probe \
    preflight "${COMMON[@]}" --load-tokenizers
fi

[[ "${MODE}" == "run" ]] || { echo "FATAL: MODE must be preflight or run" >&2; exit 1; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "FATAL: expose one student and one teacher GPU" >&2; exit 1; }
GPU_COUNT="$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | wc -l | tr -d ' ')"
[[ "${GPU_COUNT}" == "2" ]] || { echo "FATAL: expected exactly two visible GPUs, got ${CUDA_VISIBLE_DEVICES}" >&2; exit 1; }

if [[ "${SMOKE:-0}" == "1" ]]; then
  exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.causal_state_probe \
    run "${COMMON[@]}" \
    --output-dir "${OUTPUT_DIR}_smoke" \
    --max-prompts 1 \
    --continuation-k 2 \
    --relay-k 2 \
    --transport-k 2 \
    --answer-k 2 \
    --relay-lengths 32 \
    --max-continuation-tokens 256 \
    --max-answer-tokens 32
fi

exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.causal_state_probe \
  run "${COMMON[@]}"

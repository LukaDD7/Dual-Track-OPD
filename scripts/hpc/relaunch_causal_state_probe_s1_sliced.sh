#!/usr/bin/env bash
# Relaunch ONLY the unfinished s1 shard of the 20260808 causal-state probe,
# splitting its remaining work units across all 4 GPU pairs (8 GPUs, 4 slices).
#
# Motivation: s0/s2/s3 are complete; s1 has N remaining units.  Launching 4
# concurrent runners against the same s1 output dir with different
# --work-slice-index values splits the pending work ~evenly, so the last shard
# finishes in ~1/4 of the wall time and the GPU instance is never idle.
#
# Safety: every result is committed atomically to
#   .../causal_state_probe_20260808_s1/trajectory_results/
# and each runner skips work ids already present, so re-running this script
# after a recycle only picks up the still-missing units.  The final slice
# (index 3) writes summary.json + finalizes run_manifest.json; the other
# slices return after contributing their results.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER_ROOT="${DTOPD_CLUSTER_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
PYTHON_BIN="${DTOPD_PYTHON:-${CLUSTER_ROOT}/envs/va-opd-qwen35-cu128/bin/python}"
OUT="${DTOPD_OUTPUT_ROOT:-${CLUSTER_ROOT}/fc-opd-storage/outputs}/support_aware_opd"
OUTPUT_DIR="${CAUSAL_PROBE_OUTPUT_DIR:-${OUT}/causal_state_probe_20260808_s1}"
CONFIG="${CAUSAL_PROBE_CONFIG:-${PROJECT_ROOT}/configs/diagnostics/causal_state_probe.yaml}"
PAIRS=("0,1" "2,3" "4,5" "6,7")
LOG_DIR="${PROJECT_ROOT}/artifacts/fc_opd"

[[ -x "${PYTHON_BIN}" ]] || { echo "FATAL: Python environment missing: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "FATAL: config missing: ${CONFIG}" >&2; exit 1; }
[[ -f "${OUTPUT_DIR}/run_manifest.json" ]] || { echo "FATAL: missing s1 manifest at ${OUTPUT_DIR}" >&2; exit 1; }

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

export DTOPD_MODEL_ROOT="${CLUSTER_ROOT}/models"
export DTOPD_OUTPUT_ROOT="${CLUSTER_ROOT}/fc-opd-storage/outputs"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PIDS=()
for i in 0 1 2 3; do
  nohup env \
    CUDA_VISIBLE_DEVICES="${PAIRS[$i]}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.causal_state_probe \
    run \
      --config "${CONFIG}" \
      --output-dir "${OUTPUT_DIR}" \
      --student-device cuda:0 \
      --teacher-device cuda:1 \
      --shard-index 1 \
      --num-shards 4 \
      --work-slice-index "${i}" \
      --work-slice-total 4 \
    > "${LOG_DIR}/nohup_causal_state_probe_s1_slice${i}_20260812.log" 2>&1 &
  PIDS+=("$!")
  echo "slice ${i}/4 on GPUs ${PAIRS[$i]}: pid $!"
done

echo "launched 4 slices; logs: ${LOG_DIR}/nohup_causal_state_probe_s1_slice*_20260812.log"
echo "pids: ${PIDS[*]}"

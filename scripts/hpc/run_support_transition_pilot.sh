#!/usr/bin/env bash
# STP-OPD mechanics pilot launcher — orchestration only (handoff §4.3/§6).
#
# P0 gate: `DRY_RUN=1` resolves and prints the four arm commands without GPU
# or model access.  Real launch requires the training integration module and
# user approval; this script intentionally does not start training by itself.
#
# Usage:
#   DRY_RUN=1 bash scripts/hpc/run_support_transition_pilot.sh
#   bash scripts/hpc/run_support_transition_pilot.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER_ROOT="${DTOPD_CLUSTER_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
PYTHON_BIN="${DTOPD_PYTHON:-${CLUSTER_ROOT}/envs/va-opd-native-e003-cu128-r595-v1/bin/python}"
CONFIG="${STP_PILOT_CONFIG:-${PROJECT_ROOT}/configs/experiment/support_transition_prefix_opd_pilot.yaml}"
ARMS="${STP_PILOT_ARMS:-A0 A1 A2 A3}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "${CONFIG}" ]] || { echo "FATAL: config missing: ${CONFIG}" >&2; exit 1; }

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "=== STP-OPD pilot dry-run (no GPU/model access) ==="
  for arm in ${ARMS}; do
    echo "arm ${arm}:"
    echo "  ${PYTHON_BIN} -u -m dual_track_opd.support_aware.support_transition_train"
    echo "    --config ${CONFIG} --arm ${arm} --steps 60 --dry-run"
  done
  echo "=== dry-run complete: four arms resolved; nothing launched ==="
  exit 0
fi

echo "FATAL: real launch not implemented — training integration module and"
echo "       user approval are required before starting any arm (handoff §6)." >&2
exit 1

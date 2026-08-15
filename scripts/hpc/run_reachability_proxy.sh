#!/usr/bin/env bash
# Frozen-policy reachability-proxy study: preflight -> score -> analyze.
# Inference/export only; no training is launched by this script.

set -euo pipefail

PHASE="${1:-all}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"
CONFIG="${REACHABILITY_CONFIG:-${PROJECT_ROOT}/configs/diagnostics/reachability_proxy.yaml}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"

export DTOPD_OUTPUT_ROOT="${OUTPUT_ROOT}"
export DTOPD_MODEL_ROOT="${DTOPD_MODEL_ROOT:-${PROJECT_ROOT}/models}"

[[ -f "${CONFIG}" ]] || { echo "FATAL: missing config ${CONFIG}" >&2; exit 1; }

case "${PHASE}" in
    preflight)
        exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.reachability_proxy preflight --config "${CONFIG}"
        ;;
    score)
        [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "FATAL: expose student+teacher GPUs" >&2; exit 1; }
        exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.reachability_proxy score --config "${CONFIG}"
        ;;
    analyze)
        exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.reachability_proxy analyze --config "${CONFIG}"
        ;;
    all)
        "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.reachability_proxy preflight --config "${CONFIG}"
        [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "FATAL: expose student+teacher GPUs" >&2; exit 1; }
        "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.reachability_proxy score --config "${CONFIG}"
        exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.reachability_proxy analyze --config "${CONFIG}"
        ;;
    *)
        echo "Usage: $0 [preflight|score|analyze|all]" >&2
        exit 2
        ;;
esac

#!/usr/bin/env bash
# Post-hoc frontier analysis for an existing support-aware diagnostic run.
# This script is CPU-only and never loads student or teacher checkpoints.

set -euo pipefail

DTOPD_ROOT="${DTOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${DTOPD_ROOT}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 RUN_DIR [--after-run RUN_DIR] [frontier-analysis options...]" >&2
    exit 2
fi

resolve_python() {
    if [[ -n "${DTOPD_PYTHON:-}" && -x "${DTOPD_PYTHON}" ]]; then
        echo "${DTOPD_PYTHON}"
        return
    fi
    if command -v python >/dev/null 2>&1; then
        echo "python"
        return
    fi
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
        return
    fi
    echo ""
}

PYTHON_BIN="$(resolve_python)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "FATAL: no Python interpreter found; set DTOPD_PYTHON." >&2
    exit 1
fi

echo "Python: ${PYTHON_BIN}"
echo "Command: ${PYTHON_BIN} -m dual_track_opd.support_aware.frontier_analysis $*"
exec "${PYTHON_BIN}" -m dual_track_opd.support_aware.frontier_analysis "$@"

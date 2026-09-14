#!/usr/bin/env bash
# Pre-score every retained Phase-5 teacher trace on an idle GPU instance.
# Writes into the final expansion output dir; the main orchestrator skips
# prompts that are already completely scored (idempotent, lock-protected).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"
CONFIG="${REACHABILITY_PRESCORE_CONFIG:-${PROJECT_ROOT}/configs/diagnostics/reachability_proxy_expansion.yaml}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
PROPOSAL_MERGED="${OUTPUT_ROOT}/support_aware_opd/proposal_feasibility_20260815_merged"

[[ -f "${CONFIG}" ]] || { echo "FATAL: missing config ${CONFIG}" >&2; exit 1; }
[[ -f "${PROPOSAL_MERGED}/retained_proposals.jsonl" ]] || {
    echo "FATAL: missing merged proposals ${PROPOSAL_MERGED}" >&2
    exit 1
}
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "FATAL: set CUDA_VISIBLE_DEVICES (student,teacher)" >&2; exit 1; }

export DTOPD_OUTPUT_ROOT="${OUTPUT_ROOT}"
export DTOPD_MODEL_ROOT="${DTOPD_MODEL_ROOT:-${PROJECT_ROOT}/models}"

UIDS="$("${PYTHON_BIN}" -c "
import json, os
path = os.path.join(os.environ['DTOPD_OUTPUT_ROOT'],
                    'support_aware_opd/proposal_feasibility_20260815_merged/retained_proposals.jsonl')
uids = sorted({json.loads(line)['sample_uid'] for line in open(path)})
print(' '.join(uids))
")"

echo "Prescoring $(wc -w <<< "${UIDS}") prompts into ${OUTPUT_ROOT}/support_aware_opd/reachability_proxy_20260815"
exec "${PYTHON_BIN}" -u -m dual_track_opd.support_aware.reachability_proxy score \
    --config "${CONFIG}" \
    --prompt-uids ${UIDS}

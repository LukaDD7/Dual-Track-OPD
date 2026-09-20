#!/usr/bin/env bash
set -euo pipefail

# Run one isolated VA-OPD performance arm. These are execution-performance
# experiments only: they do not change VA loss, grouping, degradation, scoring,
# model identity, dataset, batch size, K, or learning rate.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_LAUNCHER="${REPO_ROOT}/scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k.sh"
CONFIG_REFERENCE="${REPO_ROOT}/configs/experiment/qwen3vl_32b_8b_virl39k_va_opd.yaml"

ARM="A"
STEPS=4
RUN_ID=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k_perf_arm.sh \
    --arm A|B|C|D|E|F --steps N --run-id ID [base launcher options]

Arms (K=8, batch=16, 32B->8B, ViRL39K):
  A: token_limit=10240, gradient_checkpointing=true  (current baseline)
  B: token_limit=16384, gradient_checkpointing=true
  C: token_limit=20480, gradient_checkpointing=true
  D: token_limit=16384, gradient_checkpointing=false
  E: token_limit=20480, gradient_checkpointing=false
  F: token_limit=32768, gradient_checkpointing=false (advanced OOM screen)

Extra base-launcher options are intentionally restricted to --resume and
--preflight-only. Batch, K, dataset, models, and full/epoch semantics are fixed.
Run only after the active full run has produced a healthy step-25 checkpoint.
Use separate run IDs; do not run two arms concurrently on one host.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm) ARM="${2:?missing arm}"; shift 2 ;;
        --steps) STEPS="${2:?missing step count}"; shift 2 ;;
        --run-id) RUN_ID="${2:?missing run id}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) break ;;
    esac
done

case " ${*} " in
    *" --batch-size "*|*" --full "*|*" --data-only "*|*" --steps "*|*" --run-id "*)
        echo "FATAL: performance arms fix batch, step count, and run lineage" >&2
        exit 2
        ;;
esac

case "${ARM}" in
    A) TOKEN_LIMIT=10240; GRADIENT_CHECKPOINTING=true ;;
    B) TOKEN_LIMIT=16384; GRADIENT_CHECKPOINTING=true ;;
    C) TOKEN_LIMIT=20480; GRADIENT_CHECKPOINTING=true ;;
    D) TOKEN_LIMIT=16384; GRADIENT_CHECKPOINTING=false ;;
    E) TOKEN_LIMIT=20480; GRADIENT_CHECKPOINTING=false ;;
    F) TOKEN_LIMIT=32768; GRADIENT_CHECKPOINTING=false ;;
    *) echo "FATAL: unknown performance arm: ${ARM}" >&2; usage >&2; exit 2 ;;
esac

if ! [[ "${STEPS}" =~ ^[0-9]+$ ]] || (( STEPS < 1 )); then
    echo "FATAL: --steps must be a positive integer" >&2
    exit 2
fi
if [[ -z "${RUN_ID}" ]]; then
    echo "FATAL: --run-id is required for performance lineage" >&2
    exit 2
fi

export VA_OPD_ROLLOUT_N=8
export VA_OPD_CONFIG_REFERENCE="${CONFIG_REFERENCE}"
export VA_OPD_RUN_ID="${RUN_ID}"
export VA_OPD_PPO_MAX_TOKEN_LEN_PER_GPU="${TOKEN_LIMIT}"
export VA_OPD_GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING}"
export VA_OPD_MAX_ACTOR_CKPT_TO_KEEP="${VA_OPD_MAX_ACTOR_CKPT_TO_KEEP:-2}"

echo "VA-OPD performance arm ${ARM}:"
echo "  token_limit: ${TOKEN_LIMIT}"
echo "  gradient_checkpointing: ${GRADIENT_CHECKPOINTING}"
echo "  K: 8"
echo "  run ID: ${RUN_ID}"

exec bash "${BASE_LAUNCHER}" \
    --steps "${STEPS}" \
    --batch-size 16 \
    --run-id "${RUN_ID}" \
    "$@"

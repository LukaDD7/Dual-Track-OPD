#!/usr/bin/env bash
set -euo pipefail

# Paper-contract VA-OPD on project-scale models: 32B teacher -> 8B student,
# ViRL39K, K=4. This entrypoint deliberately wraps the existing launcher and
# fixes the K=4 lineage/config so it cannot silently become the K=8 scaling run.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_LAUNCHER="${REPO_ROOT}/scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k.sh"
CONFIG_REFERENCE="${REPO_ROOT}/configs/experiment/qwen3vl_32b_8b_virl39k_va_opd_paper_k4.yaml"
RUN_ID="${VA_OPD_RUN_ID:-qwen3vl_32b_teacher_8b_student_virl39k_va_opd_paper_k4_v1}"

if [[ ! -f "${CONFIG_REFERENCE}" ]]; then
    echo "FATAL: missing K=4 config: ${CONFIG_REFERENCE}" >&2
    exit 2
fi

export VA_OPD_ROLLOUT_N=4
export VA_OPD_CONFIG_REFERENCE="${CONFIG_REFERENCE}"
export VA_OPD_RUN_ID="${RUN_ID}"
export VA_OPD_MAX_ACTOR_CKPT_TO_KEEP="${VA_OPD_MAX_ACTOR_CKPT_TO_KEEP:-3}"

echo "VA-OPD paper-K4 entrypoint:"
echo "  student: Qwen3-VL-8B-Instruct"
echo "  teacher: Qwen3-VL-32B-Instruct"
echo "  dataset: ViRL39K"
echo "  K:      4"
echo "  config: ${CONFIG_REFERENCE}"
echo "  run ID: ${RUN_ID}"

exec bash "${BASE_LAUNCHER}" "$@"

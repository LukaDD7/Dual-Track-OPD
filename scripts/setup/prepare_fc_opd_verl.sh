#!/usr/bin/env bash
# Apply the minimal project-owned online-distillation overlays to verl v0.7.1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERL_DIR="${VERL_DIR:-${REPO_ROOT}/third_party/verl}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -e "${VERL_DIR}/.git" ]]; then
    echo "FATAL: verl submodule is not initialized: ${VERL_DIR}" >&2
    echo "Run: git submodule update --init third_party/verl" >&2
    exit 1
fi

apply_patch_if_needed() {
    local patch_path="$1"
    local marker_file="$2"
    local marker="$3"
    if grep -qF "${marker}" "${marker_file}" 2>/dev/null; then
        echo "[OK] $(basename "${patch_path}") already applied"
        return
    fi
    if git -C "${VERL_DIR}" apply --recount --check "${patch_path}"; then
        git -C "${VERL_DIR}" apply --recount "${patch_path}"
        echo "[OK] applied $(basename "${patch_path}")"
        return
    fi
    echo "FATAL: cannot apply $(basename "${patch_path}") cleanly to ${VERL_DIR}" >&2
    echo "Preserving the existing verl worktree for inspection." >&2
    git -C "${VERL_DIR}" status --short >&2 || true
    exit 1
}

apply_patch_if_needed \
    "${REPO_ROOT}/patches/verl/fc_opd_ray_trainer_post_rollout_hook.patch" \
    "${VERL_DIR}/verl/trainer/ppo/ray_trainer.py" \
    "fc_opd_hook_fqn"
apply_patch_if_needed \
    "${REPO_ROOT}/patches/verl/fc_opd_fsdp_actor_aux_kd.patch" \
    "${VERL_DIR}/verl/workers/actor/dp_actor.py" \
    "compute_verl_fc_opd_actor_loss"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/hpc/patch_verl_pure_distill_modes.py" \
    "${VERL_DIR}/verl/workers/actor/dp_actor.py"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/hpc/patch_verl_fc_opd_global_norm.py" \
    "${VERL_DIR}/verl/workers/actor/dp_actor.py"

grep -q '"gkd", "gkd_forward"' "${VERL_DIR}/verl/workers/actor/dp_actor.py"
grep -q '"va_opd_jsd"' "${VERL_DIR}/verl/workers/actor/dp_actor.py"
grep -q 'fc_opd_global_normalizer' "${VERL_DIR}/verl/workers/actor/dp_actor.py"

echo "FC/VA-OPD verl backend: PASS"
echo "  upstream: $(git -C "${VERL_DIR}" rev-parse HEAD)"
echo "  path:     ${VERL_DIR}"

#!/usr/bin/env bash
# Check out and patch the exact native-OPD verl backend used by VA-OPD.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPECTED_COMMIT="e003163181731412595257a72ec173071efb125f"
VERL_DIR="${VERL_VA_OPD_DIR:-${REPO_ROOT}/external/verl-va-opd-e0031631}"
PATCH_PATH="${REPO_ROOT}/patches/verl/va_opd_native_e0031631.patch"

if [[ ! -e "${VERL_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${VERL_DIR}")"
    git clone --filter=blob:none https://github.com/verl-project/verl.git "${VERL_DIR}"
fi

git -C "${VERL_DIR}" fetch origin "${EXPECTED_COMMIT}" --depth 1
CURRENT_COMMIT="$(git -C "${VERL_DIR}" rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
    if [[ -n "$(git -C "${VERL_DIR}" status --porcelain --untracked-files=no)" ]]; then
        echo "FATAL: refusing to switch a modified backend: ${VERL_DIR}" >&2
        git -C "${VERL_DIR}" status --short >&2
        exit 1
    fi
    git -C "${VERL_DIR}" switch --detach "${EXPECTED_COMMIT}"
fi

MARKER="register_native_verl_loss"
MARKER_FILE="${VERL_DIR}/verl/trainer/distillation/losses.py"
if ! grep -qF "${MARKER}" "${MARKER_FILE}"; then
    git -C "${VERL_DIR}" apply --check "${PATCH_PATH}"
    git -C "${VERL_DIR}" apply "${PATCH_PATH}"
fi

# A marker alone is insufficient: reject stale or hand-edited variants of the
# three-file overlay before the environment is built around them.
git -C "${VERL_DIR}" apply --reverse --check "${PATCH_PATH}"

grep -qF "teacher_degraded_logprobs" "${VERL_DIR}/verl/experimental/agent_loop/agent_loop.py"
grep -qF "prepare_native_verl_batch" "${VERL_DIR}/verl/trainer/ppo/ray_trainer.py"
grep -qF "register_native_verl_loss" "${MARKER_FILE}"

echo "Native VA-OPD verl backend: PASS"
echo "  commit: ${EXPECTED_COMMIT}"
echo "  path:   ${VERL_DIR}"
echo "  patch:  ${PATCH_PATH}"

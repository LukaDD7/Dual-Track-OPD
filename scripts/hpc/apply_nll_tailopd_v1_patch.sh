#!/usr/bin/env bash
# Apply the default-off NLL-TailOPD v1 integration to the pinned external verl backend.
#
# The backend is intentionally dirty in this research workspace. This script is
# idempotent: it detects an already-applied patch by reverse-checking it and
# never resets, cleans, or discards unrelated backend changes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_ROOT="${NLL_TAILOPD_VERL_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/backends/verl-qwen35-v090-cu132}"
PATCH_FILE="${REPO_ROOT}/patches/verl/nll_tailopd_v1.patch"
PYTHON_BIN="${NLL_TAILOPD_PYTHON:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/va-opd-qwen35-v090-cu132-r595-v1/bin/python}"

if [[ ! -d "${BACKEND_ROOT}/.git" ]]; then
    echo "ERROR: backend root is not a Git worktree: ${BACKEND_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "ERROR: patch not found: ${PATCH_FILE}" >&2
    exit 1
fi

cd "${BACKEND_ROOT}"

if git apply --check --reverse "${PATCH_FILE}"; then
    echo "NLL-TailOPD v1 patch is already applied."
else
    if ! git apply --check "${PATCH_FILE}"; then
        echo "ERROR: NLL-TailOPD patch does not apply cleanly; inspect backend changes manually." >&2
        exit 1
    fi
    git apply "${PATCH_FILE}"
    echo "Applied NLL-TailOPD v1 patch."
fi

"${PYTHON_BIN}" -m py_compile \
    verl/trainer/config/algorithm.py \
    verl/trainer/distillation/losses.py \
    verl/trainer/ppo/ray_trainer.py
bash -n examples/on_policy_distillation_trainer/run_qwen3_5_4b_fsdp.sh

echo "Backend commit: $(git rev-parse HEAD)"
echo "Backend dirty status:"
git status --short
echo "NLL-TailOPD v1 patch check: PASS"

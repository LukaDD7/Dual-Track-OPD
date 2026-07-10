#!/usr/bin/env bash
# Prepare the verl revision that contains the GKD recipe and its matching
# in-process vLLM rollout.  This reuses the existing clone and never modifies
# or deletes the server's patched bcb638 checkout.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_VERL="${SOURCE_VERL:-${REPO_ROOT}/external/verl_gkd/verl}"
TARGET_VERL="${TARGET_VERL:-${REPO_ROOT}/external/verl_gkd_compatible/verl}"
GKD_COMPAT_COMMIT="d8e97e1724e348658c670b9160f1393d4fb20678"

if [[ ! -d "${SOURCE_VERL}/.git" && ! -f "${SOURCE_VERL}/.git" ]]; then
    echo "FATAL: source verl checkout not found: ${SOURCE_VERL}" >&2
    exit 1
fi

if ! git -C "${SOURCE_VERL}" cat-file -e "${GKD_COMPAT_COMMIT}^{commit}" 2>/dev/null; then
    echo "FATAL: ${GKD_COMPAT_COMMIT} is absent from the source clone." >&2
    echo "On the CPU node, run: git -C ${SOURCE_VERL} fetch origin ${GKD_COMPAT_COMMIT}" >&2
    exit 1
fi

if [[ -e "${TARGET_VERL}" ]]; then
    actual="$(git -C "${TARGET_VERL}" rev-parse HEAD 2>/dev/null || true)"
    if [[ "${actual}" != "${GKD_COMPAT_COMMIT}" ]]; then
        echo "FATAL: existing target is at ${actual:-unknown}, expected ${GKD_COMPAT_COMMIT}" >&2
        echo "Refusing to overwrite: ${TARGET_VERL}" >&2
        exit 1
    fi
    echo "Compatible GKD checkout already exists: ${TARGET_VERL}"
else
    mkdir -p "$(dirname "${TARGET_VERL}")"
    git -C "${SOURCE_VERL}" worktree add --detach "${TARGET_VERL}" "${GKD_COMPAT_COMMIT}"
fi

test -f "${TARGET_VERL}/recipe/gkd/main_gkd.py"
test -f "${TARGET_VERL}/verl/workers/rollout/vllm_rollout/vllm_rollout.py"
grep -q 'class vLLMAsyncRollout' \
    "${TARGET_VERL}/verl/workers/rollout/vllm_rollout/vllm_rollout.py"
grep -q 'def generate_sequences' \
    "${TARGET_VERL}/verl/workers/rollout/vllm_rollout/vllm_rollout.py"

echo "GKD compatible checkout: PASS"
echo "  commit: ${GKD_COMPAT_COMMIT}"
echo "  path:   ${TARGET_VERL}"
echo "  The smoke launcher will select this checkout automatically."

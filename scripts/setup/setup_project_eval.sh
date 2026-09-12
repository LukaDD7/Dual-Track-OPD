#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-${REPO_ROOT}/third_party_runtime/lmms-eval}"
LMMS_EVAL_COMMIT="88b23e2bfa16a1edbc16e9e238ed82130b3a4f56"
LMMS_EVAL_REPO="https://github.com/EvolvingLMMs-Lab/lmms-eval.git"
INSTALL_EXTRAS="${LMMS_EVAL_EXTRAS:-all}"

if [[ ! -d "${LMMS_EVAL_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${LMMS_EVAL_ROOT}")"
  git clone "${LMMS_EVAL_REPO}" "${LMMS_EVAL_ROOT}"
fi

git -C "${LMMS_EVAL_ROOT}" fetch --tags origin
git -C "${LMMS_EVAL_ROOT}" checkout --detach "${LMMS_EVAL_COMMIT}"

python -m pip install -e "${LMMS_EVAL_ROOT}[${INSTALL_EXTRAS}]"
python -m pip install -e "${REPO_ROOT}"

ACTUAL_COMMIT="$(git -C "${LMMS_EVAL_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${LMMS_EVAL_COMMIT}" ]]; then
  echo "ERROR: lmms-eval commit mismatch: ${ACTUAL_COMMIT}" >&2
  exit 1
fi

python -m lmms_eval --tasks list >/dev/null
echo "[setup-project-eval] lmms-eval=${ACTUAL_COMMIT}"
echo "[setup-project-eval] environment ready"

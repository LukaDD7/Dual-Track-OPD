#!/usr/bin/env bash
# Canonical VA-OPD launcher.
#
# This script is intentionally a thin compatibility wrapper.  The maintained
# VA-only configuration lives in run_verl_fc_opd_overnight.sh:
#   - conditions=[full,degraded]
#   - no StudentScorer
#   - rollout weights sum to 1 per prompt sibling group
#   - loss_mode=va_opd with pure distillation actor bypass

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${REPO_ROOT}/scripts/hpc/run_verl_fc_opd_overnight.sh" "$@"

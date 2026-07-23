#!/usr/bin/env bash
# Historical entrypoint for the unsupported native VA-OPD cu132 experiment.
#
# The 2026-07-21 environment combined a vLLM 0.25.1 wheel built for torch
# 2.11 with torch 2.13+cu132, while verl e0031631 supports vLLM only through
# 0.12.0.  It failed with a libtorch ABI undefined symbol.  Reinstalling torch,
# changing LD_LIBRARY_PATH, or selecting another attention backend cannot make
# that binary matrix reproducible.

set -euo pipefail

cat >&2 <<'EOF'
FATAL: the native VA-OPD cu132 environment is quarantined and must not be rebuilt.

Use the supported cu128 user-space stack on the R595/CUDA-13.2-capable H200
node.  A newer driver can run the pinned CUDA 12.8 application stack.

Canonical setup:
  export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
  export VA_OPD_ENV_PREFIX="$DTOPD_ROOT/envs/va-opd-native-e003-cu128-r595-v1"
  export VERL_VA_OPD_DIR="$DTOPD_ROOT/fc-opd-storage/backends/verl-va-opd-e0031631-clean"
  export VA_OPD_CUDA_TOOLCHAIN="$DTOPD_ROOT/envs/cuda128-toolchain"
  bash scripts/hpc/setup_va_opd_native_env.sh

Read docs/environment_registry.md and docs/va_opd_cu132_status_20260722.md.
EOF

exit 2

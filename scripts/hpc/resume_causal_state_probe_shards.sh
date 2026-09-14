#!/usr/bin/env bash
# Resume the 4-shard causal-state probe (20260808) after an instance recycle.
#
# Safe to re-run: the runner reuses the existing per-shard run_manifest.json and
# skips any trajectory JSON already present in trajectory_results/.  The three
# env vars SHARD_INDEX / NUM_SHARDS / CAUSAL_PROBE_OUTPUT_DIR must stay
# identical across relaunches (the manifest check enforces this).
#
# GPU layout assumption: 8 GPUs as pairs 0,1 / 2,3 / 4,5 / 6,7, student on the
# first device of each pair, teacher on the second (as the current run does).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${DTOPD_OUTPUT_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs}/support_aware_opd"
PAIRS=("0,1" "2,3" "4,5" "6,7")

cd "${PROJECT_ROOT}"

for s in 0 1 2 3; do
  nohup env \
    CUDA_VISIBLE_DEVICES="${PAIRS[$s]}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SHARD_INDEX="${s}" \
    NUM_SHARDS=4 \
    CAUSAL_PROBE_OUTPUT_DIR="${OUT}/causal_state_probe_20260808_s${s}" \
    DTOPD_PYTHON=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/va-opd-qwen35-cu128/bin/python \
    bash scripts/hpc/run_causal_state_probe.sh \
    > "artifacts/fc_opd/nohup_causal_state_probe_s${s}_20260808.log" 2>&1 &
  echo "shard ${s} pid $!"
done

echo "all 4 shards launched (resume mode). logs: artifacts/fc_opd/nohup_causal_state_probe_s*_20260808.log"

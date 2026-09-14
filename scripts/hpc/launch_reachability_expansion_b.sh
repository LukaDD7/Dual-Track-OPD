#!/usr/bin/env bash
# Phase-5B on the stable GPU instance: student wrong controls -> adaptive
# rescue for the 19 expansion prompts -> merge -> combined proxy analysis.
# No training is launched.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 GPU0,GPU1,...   e.g. 0,1,2,3" >&2
    exit 2
fi

GPU_CSV="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
WRONG_PREFIX="${WRONG_CONTROL_PREFIX:-wrong_control_pool_20260815}"
INT_PREFIX="${EXPANSION_INTERVENTION_PREFIX:-prefix_intervention_20260815b}"
LOG_DIR="${EXPANSION_LOG_DIR:-${OUTPUT_ROOT}/../logs/reachability_expansion_b}"
WRONG_POOL="${OUTPUT_ROOT}/support_aware_opd/${WRONG_PREFIX}"
INT_MERGED="${OUTPUT_ROOT}/support_aware_opd/${INT_PREFIX}_merged"

export DTOPD_ALLOW_DIRTY_MERGE=1
export DTOPD_OUTPUT_ROOT="${OUTPUT_ROOT}"
export DTOPD_MODEL_ROOT="${DTOPD_MODEL_ROOT:-${PROJECT_ROOT}/models}"
export EXPANSION_MANIFEST="${WRONG_POOL}/intervention_subset_manifest.jsonl"
export EXPANSION_INTERVENTION_CONFIG="${EXPANSION_INTERVENTION_CONFIG:-${PROJECT_ROOT}/configs/experiment/reachability_expansion_intervention_b.yaml}"
export EXPANSION_INTERVENTION_PREFIX="${INT_PREFIX}"

IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
NUM_SHARDS="${#GPUS[@]}"
(( NUM_SHARDS > 0 )) || { echo "FATAL: no GPUs" >&2; exit 1; }
mkdir -p "${LOG_DIR}"

echo "== Phase B1: student wrong controls (${NUM_SHARDS} shards) =="
PIDS=()
for (( i=0; i<NUM_SHARDS; i++ )); do
    CUDA_VISIBLE_DEVICES="${GPUS[i]}" \
    WRONG_CONTROL_PREFIX="${WRONG_PREFIX}" \
    bash "${PROJECT_ROOT}/scripts/hpc/run_wrong_controls_shard.sh" \
        "${i}" "${NUM_SHARDS}" "${WRONG_PREFIX}" >"${LOG_DIR}/wrong_s${i}.log" 2>&1 &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "${pid}"; done

echo "== Phase B2: merge wrong-control pool =="
SHARD_DIRS=()
for (( i=0; i<NUM_SHARDS; i++ )); do
    SHARD_DIRS+=("${OUTPUT_ROOT}/support_aware_opd/${WRONG_PREFIX}_s${i}")
done
[[ -d "${WRONG_POOL}" ]] && rm -rf "${WRONG_POOL}"
"${PYTHON_BIN}" -m dual_track_opd.support_aware.reachability_wrong_controls merge \
    --shard-dirs "${SHARD_DIRS[@]}" \
    --output-dir "${WRONG_POOL}"

echo "== Phase B3: adaptive rescue (${NUM_SHARDS} shards, one GPU each) =="
PIDS=()
for (( i=0; i<NUM_SHARDS; i++ )); do
    shard_dir="${OUTPUT_ROOT}/support_aware_opd/${INT_PREFIX}_s${i}"
    [[ -f "${shard_dir}/run_manifest.json" ]] && rm -f "${shard_dir}/run_manifest.json"
    [[ -f "${shard_dir}/summary.json" ]] && rm -f "${shard_dir}/summary.json"
    CUDA_VISIBLE_DEVICES="${GPUS[i]}" \
    EXPANSION_INTERVENTION_PREFIX="${INT_PREFIX}" \
    bash "${PROJECT_ROOT}/scripts/hpc/run_reachability_expansion_intervention_shard.sh" \
        "${i}" "${NUM_SHARDS}" "${INT_PREFIX}" >"${LOG_DIR}/intervention_s${i}.log" 2>&1 &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "${pid}"; done

echo "== Phase B4: merge adaptive rescue =="
SHARD_DIRS=()
for (( i=0; i<NUM_SHARDS; i++ )); do
    SHARD_DIRS+=("${OUTPUT_ROOT}/support_aware_opd/${INT_PREFIX}_s${i}")
done
[[ -d "${INT_MERGED}" ]] && rm -rf "${INT_MERGED}"
"${PYTHON_BIN}" -m dual_track_opd.support_aware.prefix_intervention merge \
    --shard-dirs "${SHARD_DIRS[@]}" \
    --output-dir "${INT_MERGED}"

echo "== Phase B5: combined proxy analysis (12 + 33 + 19 prompts) =="
OUT="${OUTPUT_ROOT}/support_aware_opd"
"${PYTHON_BIN}" -m dual_track_opd.support_aware.reachability_proxy analyze-combined \
    --token-dirs "${OUT}/reachability_proxy_20260814" "${OUT}/reachability_proxy_20260815" \
    --rescue-dirs "${OUT}/prefix_intervention_20260806_merged" "${OUT}/prefix_intervention_20260815_merged" "${INT_MERGED}" \
    --proposal-dirs "${OUT}/proposal_feasibility_20260805_merged" "${OUT}/proposal_feasibility_20260815_merged" \
    --output-dir "${OUT}/reachability_proxy_combined_final"

echo "DONE. See ${OUT}/reachability_proxy_combined_final/proxy_analysis.json"

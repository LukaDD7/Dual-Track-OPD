#!/usr/bin/env bash
# Phase-5 end-to-end on GPU instances (brief 2026-08-14, section 8):
#   freeze manifest -> teacher proposals (GPU pairs, shards) -> merge ->
#   adaptive K=4->K=8 rescue (single GPUs, shards) -> merge ->
#   frozen proxy scoring + analysis.
# No training is launched.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 TEACHER:STUDENT,TEACHER:STUDENT,..." >&2
    echo "Example: $0 0:1,2:3,4:5,6:7" >&2
    exit 2
fi

GPU_PAIRS_CSV="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DTOPD_PYTHON:-python}"
OUTPUT_ROOT="${DTOPD_OUTPUT_ROOT:-${PROJECT_ROOT}/fc-opd-storage/outputs}"
REACH_OUT="${OUTPUT_ROOT}/support_aware_opd/reachability_proxy_20260814"
POOL256="${OUTPUT_ROOT}/support_aware_opd/diag_full_20260802_256_merged.rescored-exact-v1"
RESCUE_12="${OUTPUT_ROOT}/support_aware_opd/prefix_intervention_20260806_merged"
MANIFEST="${EXPANSION_MANIFEST:-${REACH_OUT}/expansion_manifest_20260815.jsonl}"
PROPOSAL_PREFIX="${EXPANSION_PROPOSAL_PREFIX:-proposal_feasibility_20260815}"
INTERVENTION_PREFIX="${EXPANSION_INTERVENTION_PREFIX:-prefix_intervention_20260815}"
LOG_DIR="${EXPANSION_LOG_DIR:-${OUTPUT_ROOT}/../logs/reachability_expansion}"

# Pre-existing untracked artifacts make the shared worktree dirty; git commit
# identity is still enforced by the shard manifests.
export DTOPD_ALLOW_DIRTY_MERGE=1
export DTOPD_OUTPUT_ROOT="${OUTPUT_ROOT}"
export DTOPD_MODEL_ROOT="${DTOPD_MODEL_ROOT:-${PROJECT_ROOT}/models}"
export EXPANSION_MANIFEST="${MANIFEST}"

IFS=',' read -r -a GPU_PAIRS <<< "${GPU_PAIRS_CSV}"
NUM_PROP_SHARDS="${#GPU_PAIRS[@]}"
(( NUM_PROP_SHARDS > 0 )) || { echo "FATAL: no GPU pairs" >&2; exit 1; }
GPU_CSV=""
for pair in "${GPU_PAIRS[@]}"; do
    [[ "${pair}" =~ ^[0-9]+:[0-9]+$ ]] || { echo "FATAL: invalid pair ${pair}" >&2; exit 1; }
    if [[ -n "${GPU_CSV}" ]]; then GPU_CSV="${GPU_CSV},"; fi
    GPU_CSV="${GPU_CSV}${pair%%:*},${pair##*:}"
done
IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
NUM_INT_SHARDS="${#GPUS[@]}"

echo "== Phase 0: freeze expansion manifest =="
mkdir -p "${REACH_OUT}" "${LOG_DIR}"
"${PYTHON_BIN}" -m dual_track_opd.support_aware.reachability_expansion freeze \
    --pool256-dir "${POOL256}" \
    --rescue-dir "${RESCUE_12}" \
    --output-dir "${REACH_OUT}"
"${PYTHON_BIN}" -m dual_track_opd.support_aware.reachability_expansion verify \
    --manifest "${MANIFEST}" --pool256-dir "${POOL256}"

echo "== Phase 1: teacher proposals (${NUM_PROP_SHARDS} shards) =="
PIDS=()
for (( i=0; i<NUM_PROP_SHARDS; i++ )); do
    pair="${GPU_PAIRS[i]}"
    log="${LOG_DIR}/proposals_s${i}.log"
    CUDA_VISIBLE_DEVICES="${pair%%:*},${pair##*:}" \
    EXPANSION_PROPOSAL_PREFIX="${PROPOSAL_PREFIX}" \
    bash "${PROJECT_ROOT}/scripts/hpc/run_reachability_expansion_proposal_shard.sh" \
        "${i}" "${NUM_PROP_SHARDS}" "${PROPOSAL_PREFIX}" >"${log}" 2>&1 &
    PIDS+=("$!")
    echo "  shard ${i}: teacher=${pair%%:*}, student=${pair##*:}; log=${log}"
done
for pid in "${PIDS[@]}"; do wait "${pid}"; done

echo "== Phase 2: merge proposals =="
SHARD_DIRS=()
for (( i=0; i<NUM_PROP_SHARDS; i++ )); do
    SHARD_DIRS+=("${OUTPUT_ROOT}/support_aware_opd/${PROPOSAL_PREFIX}_s${i}")
done
# Regenerable merge artifacts; remove before re-merge on resume.
PROPOSAL_MERGED="${OUTPUT_ROOT}/support_aware_opd/${PROPOSAL_PREFIX}_merged"
[[ -d "${PROPOSAL_MERGED}" ]] && rm -rf "${PROPOSAL_MERGED}"
"${PYTHON_BIN}" -m dual_track_opd.support_aware.proposal_feasibility merge \
    --shard-dirs "${SHARD_DIRS[@]}" \
    --output-dir "${PROPOSAL_MERGED}"

echo "== Phase 3: adaptive rescue (${NUM_INT_SHARDS} shards, one GPU each) =="
PIDS=()
for (( i=0; i<NUM_INT_SHARDS; i++ )); do
    shard_dir="${OUTPUT_ROOT}/support_aware_opd/${INTERVENTION_PREFIX}_s${i}"
    # Drop stale resume metadata so shards adopt the current merged proposal
    # hash; prompt_results are preserved for resumption.
    [[ -f "${shard_dir}/run_manifest.json" ]] && rm -f "${shard_dir}/run_manifest.json"
    [[ -f "${shard_dir}/summary.json" ]] && rm -f "${shard_dir}/summary.json"
    log="${LOG_DIR}/intervention_s${i}.log"
    CUDA_VISIBLE_DEVICES="${GPUS[i]}" \
    EXPANSION_INTERVENTION_PREFIX="${INTERVENTION_PREFIX}" \
    bash "${PROJECT_ROOT}/scripts/hpc/run_reachability_expansion_intervention_shard.sh" \
        "${i}" "${NUM_INT_SHARDS}" "${INTERVENTION_PREFIX}" >"${log}" 2>&1 &
    PIDS+=("$!")
    echo "  shard ${i}: gpu=${GPUS[i]}; log=${log}"
done
for pid in "${PIDS[@]}"; do wait "${pid}"; done

echo "== Phase 4: merge intervention =="
SHARD_DIRS=()
for (( i=0; i<NUM_INT_SHARDS; i++ )); do
    SHARD_DIRS+=("${OUTPUT_ROOT}/support_aware_opd/${INTERVENTION_PREFIX}_s${i}")
done
INTERVENTION_MERGED="${OUTPUT_ROOT}/support_aware_opd/${INTERVENTION_PREFIX}_merged"
[[ -d "${INTERVENTION_MERGED}" ]] && rm -rf "${INTERVENTION_MERGED}"
"${PYTHON_BIN}" -m dual_track_opd.support_aware.prefix_intervention merge \
    --shard-dirs "${SHARD_DIRS[@]}" \
    --output-dir "${INTERVENTION_MERGED}"

echo "== Phase 5: frozen proxy scoring + analysis =="
FIRST_PAIR="${GPU_PAIRS[0]}"
CUDA_VISIBLE_DEVICES="${FIRST_PAIR%%:*},${FIRST_PAIR##*:}" \
"${PYTHON_BIN}" -m dual_track_opd.support_aware.reachability_proxy score \
    --config "${PROJECT_ROOT}/configs/diagnostics/reachability_proxy_expansion.yaml"
CUDA_VISIBLE_DEVICES="${FIRST_PAIR%%:*},${FIRST_PAIR##*:}" \
"${PYTHON_BIN}" -m dual_track_opd.support_aware.reachability_proxy analyze \
    --config "${PROJECT_ROOT}/configs/diagnostics/reachability_proxy_expansion.yaml"

echo "DONE. See ${OUTPUT_ROOT}/support_aware_opd/reachability_proxy_20260815/"

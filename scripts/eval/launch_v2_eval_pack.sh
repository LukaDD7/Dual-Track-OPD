#!/usr/bin/env bash
# Launch the four-task v2 diagnostic for base, TailSFT, and PTD-PO.
#
# Usage:
#   bash scripts/eval/launch_v2_eval_pack.sh [gpu] [arms]
#
# Default:
#   GPU 0, arms=base,tailsft,ptdpo
#
# Each arm runs sequentially on one model server. The v2 protocol has no judge
# tasks, so only one GPU is needed. Output is written under:
#   eval_runs/vision_opd_project_v2

set -euo pipefail

DTOPD_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
REPO_ROOT="${DTOPD_ROOT}/projects/Dual-Track-OPD"
GPU_ID="${1:-0}"
ARMS="${2:-base,tailsft,ptdpo}"
PORT="${V2_PACK_PORT:-8100}"

PTDPO_HF="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/hf/qwen3vl_ptdpo_r4_step390"

IFS=',' read -r -a ARM_LIST <<< "${ARMS}"

for arm in "${ARM_LIST[@]}"; do
  case "${arm}" in
    base)
      ckpt="${DTOPD_ROOT}/models/Qwen3-VL-8B-Instruct"
      run_name="v2_base_qwen3vl8b"
      served_name="Qwen3-VL-8B-Base"
      ;;
    tailsft)
      ckpt="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_tailsft_mmf122k_1ep/global_step_1774/huggingface"
      run_name="v2_tailsft_mmf122k_1ep"
      served_name="Qwen3-VL-8B-TailSFT"
      ;;
    ptdpo)
      ckpt="${PTDPO_HF}"
      run_name="v2_ptdpo_r4_step390"
      served_name="Qwen3-VL-8B-PTDPO-R4"
      ;;
    *)
      echo "FATAL: unknown arm ${arm}" >&2
      exit 1
      ;;
  esac

  echo "== launching v2 arm ${arm} on GPU ${GPU_ID} =="
  EVAL_CKPT="${ckpt}" \
  EVAL_RUN_NAME="${run_name}" \
  EVAL_SERVED_MODEL="${served_name}" \
  EVAL_GPU="${GPU_ID}" \
  EVAL_PORT="${PORT}" \
    bash "${REPO_ROOT}/scripts/eval/run_target_benchmarks_v2.sh"

  PORT="$((PORT + 1))"
done

echo "== v2 eval pack complete =="

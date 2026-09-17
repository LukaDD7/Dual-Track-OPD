#!/usr/bin/env bash
# Track B fixed offline eval: ckpt 0/60/120 x 3 seeds on the 200-prompt val
# set, N=8 per prompt with a fixed seed grid, paired bootstrap summary.
#
# Steps: 1) merge FSDP checkpoints to HF (CPU, one-off); 2) vLLM offline eval
# per condition (GPU); 3) prompt-paired summary with bootstrap CIs.
#
# Env overrides:
#   EVAL_ROOT      output root (default fc-opd-storage/outputs/qwen35_fixed_eval_20260806)
#   SEEDS          space list (default "r1 seed2 seed3")
#   STEPS          space list (default "60 120")
#   N_SAMPLES      samples per prompt (default 8)
#   EVAL_GPUS      cuda device for eval (default 0)
#   SKIP_MERGE=1   reuse existing merged dirs
#   SKIP_EVAL=1    reuse existing eval dumps
set -euo pipefail

DTOPD_ROOT=${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}
PROJECT_ROOT="${PROJECT_ROOT:-${DTOPD_ROOT}/projects/Dual-Track-OPD}"
PYTHON="${PYTHON:-${DTOPD_ROOT}/envs/va-opd-qwen35-cu128/bin/python}"
EVAL_ROOT="${EVAL_ROOT:-${DTOPD_ROOT}/fc-opd-storage/outputs/qwen35_fixed_eval_20260806}"
MERGE_DIR="${EVAL_ROOT}/merged"
EVAL_DIR="${EVAL_ROOT}/eval_dumps"
SUMMARY_DIR="${EVAL_ROOT}/summary"
VAL_PARQUET="${VAL_PARQUET:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/geometry3k_gkd/val_text_only.parquet}"
BASE_MODEL="${BASE_MODEL:-${DTOPD_ROOT}/models/Qwen3.5-4B}"
CKPT_ROOT="${CKPT_ROOT:-${DTOPD_ROOT}/repos/verl-cu130-vllm/examples/on_policy_distillation_trainer/checkpoints/verl_distill_qwen35}"
SEEDS=${SEEDS:-"r1 seed2 seed3"}
STEPS=${STEPS:-"60 120"}
N_SAMPLES=${N_SAMPLES:-8}
EVAL_GPUS=${EVAL_GPUS:-0}

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
# vLLM 0.23 + FlashInfer JIT requires the same CUDA toolchain env as the
# qwen35 training scripts (run_qwen35_formal.sh); standalone eval must
# replicate it or FlashInfer cannot find nvcc.
CUDA_HOME="${CUDA_HOME:-${DTOPD_ROOT}/envs/cuda128-toolchain}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${DTOPD_ROOT}/.cache/flashinfer}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$MERGE_DIR" "$EVAL_DIR" "$SUMMARY_DIR"

echo "== eval root: ${EVAL_ROOT} =="
echo "== val: ${VAL_PARQUET} =="
echo "== base: ${BASE_MODEL} =="
git rev-parse HEAD
git status --porcelain --untracked-files=no || true

for seed in ${SEEDS}; do
  case "${seed}" in r1|seed2|seed3) ;; *) echo "FATAL: unknown seed tag ${seed}"; exit 1 ;; esac
  for step in ${STEPS}; do
    case "${step}" in 30|60|90|120) ;; *) echo "FATAL: unknown step ${step}"; exit 1 ;; esac
    tag="${seed}_step${step}"
    ckpt_dir="${CKPT_ROOT}/qwen3_6_27b_to_qwen3_5_4b_k1_tasktrue_fcop_pvboxed_only_nonthinking_sampled_r4096_n4_overnight120_${seed}/global_step_${step}"
    actor_dir="${ckpt_dir}/actor"
    [ -d "${actor_dir}" ] || { echo "FATAL: missing checkpoint ${actor_dir}"; exit 1; }

    if [ ! -d "${MERGE_DIR}/${tag}" ] || [ "${SKIP_MERGE:-0}" != "1" ]; then
      echo "== merge ${tag} =="
      "$PYTHON" -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${actor_dir}" \
        --target_dir "${MERGE_DIR}/${tag}" \
        --trust-remote-code \
        --use_cpu_initialization
    else
      echo "== skip merge ${tag} (exists) =="
    fi

    if [ ! -f "${EVAL_DIR}/${tag}.jsonl" ] || [ "${SKIP_EVAL:-0}" != "1" ]; then
      echo "== eval ${tag} on cuda:${EVAL_GPUS} =="
      CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
      "$PYTHON" -u scripts/qwen35_fixed_offline_eval.py \
        --model-path "${MERGE_DIR}/${tag}" \
        --val-parquet "${VAL_PARQUET}" \
        --output-jsonl "${EVAL_DIR}/${tag}.jsonl" \
        --condition "${tag}" \
        --n "${N_SAMPLES}" \
        --device cuda:0 \
        --max-response-length 4096 \
        --max-model-len 8192
    else
      echo "== skip eval ${tag} (exists) =="
    fi
  done
done

if [ ! -f "${EVAL_DIR}/base.jsonl" ] || [ "${SKIP_EVAL:-0}" != "1" ]; then
  echo "== eval base on cuda:${EVAL_GPUS} =="
  CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
  "$PYTHON" -u scripts/qwen35_fixed_offline_eval.py \
    --model-path "${BASE_MODEL}" \
    --val-parquet "${VAL_PARQUET}" \
    --output-jsonl "${EVAL_DIR}/base.jsonl" \
    --condition "base" \
    --n "${N_SAMPLES}" \
    --device cuda:0 \
    --max-response-length 4096 \
    --max-model-len 8192
else
  echo "== skip eval base (exists) =="
fi

conditions="base"
for seed in ${SEEDS}; do
  for step in ${STEPS}; do
    conditions="${conditions},seed${seed}_step${step}"
  done
done

echo "== summary =="
"$PYTHON" -u scripts/qwen35_fixed_offline_eval_summary.py \
  --eval-dir "${EVAL_DIR}" \
  --conditions "${conditions}" \
  --output-json "${SUMMARY_DIR}/summary.json"

echo "== DONE =="
echo "summary: ${SUMMARY_DIR}/summary.json"

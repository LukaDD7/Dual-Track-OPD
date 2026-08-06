#!/usr/bin/env bash
# Unattended multi-seed Track-A sequence: N consecutive 120-step runs with the
# frozen D3/n4 config, each with fresh metadata/val/rollout/checkpoint dirs and
# a per-run group-level clip summary.  The instrumentation canary was already
# validated by the overnight sequence, so no canary is repeated here.
#
# Env overrides:
#   SEEDS="2 3"        seeds to run (default: 2 3); e.g. SEEDS=2 for one seed
#   TOTAL_STEPS=120    steps per run (default 120)
#   DRY_RUN=1          compose all seeds without running (skips clean gate)
#
# Launch (Track A uses GPUs 0-3; keepalive/other work must use GPUs 4+):
#   nohup bash scripts/hpc/run_qwen35_v1_n4_seed_sequence.sh \
#     > artifacts/fc_opd/nohup_qwen35_n4_seed_sequence_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/../.."

DTOPD_ROOT=${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}
DTOPD_PYTHON=${DTOPD_PYTHON:-${DTOPD_ROOT}/envs/va-opd-qwen35-cu128/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-${DTOPD_ROOT}/fc-opd-storage/logs}
SEEDS=${SEEDS:-"2 3"}
TOTAL_STEPS=${TOTAL_STEPS:-120}

if [ "${DRY_RUN:-0}" != "1" ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "FATAL: project has tracked changes; use a clean checkout for the seed sequence."
  git status --short --untracked-files=no
  exit 1
fi

run_seed() {
  local seed=$1
  local tag="k1_boxedonly_nonthinking_sampled_r4096_n4_overnight120_seed${seed}"
  local experiment="qwen3_6_27b_to_qwen3_5_4b_k1_tasktrue_fcop_pvboxed_only_nonthinking_sampled_r4096_n4_overnight120_seed${seed}"
  local meta="${OUTPUT_ROOT}/qwen35_runs/${tag}"
  local val="${OUTPUT_ROOT}/val_dump_${tag}"
  local rollout="${meta}/train_rollouts"
  local group="${meta}/group_clip_summary.json"

  echo "== seed ${seed}: ${TOTAL_STEPS}-step run =="
  FORMAL_GPUS=0,1,2,3 \
  NGPUS_PER_NODE=3 \
  TRAIN_BATCH_SIZE=6 \
  PPO_MINI_BATCH_SIZE=6 \
  USE_FCOP_DATASET=1 \
  PROMPT_VERSION=boxed_only \
  USE_TASK_REWARDS=True \
  ROLLOUT_N=4 \
  ROLLOUT_NUM_WORKERS=8 \
  TRAINER_USE_V1=True \
  RESUME_MODE=disable \
  VAL_BEFORE_TRAIN=True \
  TOTAL_TRAINING_STEPS="${TOTAL_STEPS}" \
  MAX_RESPONSE_LENGTH=4096 \
  PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
  SAVE_FREQ=30 \
  TEST_FREQ=20 \
  EXPERIMENT_NAME="${experiment}" \
  VALIDATION_DATA_DIR="${val}" \
  TRAIN_ROLLOUT_DATA_DIR="${rollout}" \
  RUN_METADATA_DIR="${meta}" \
  bash scripts/run_qwen35_formal.sh \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    data.seed="${seed}" \
    actor_rollout_ref.actor.data_loader_seed="${seed}"

  if [ "${DRY_RUN:-0}" != "1" ]; then
    "${DTOPD_PYTHON}" scripts/qwen35_group_clip_analysis.py \
      --rollout-dir "${rollout}" \
      --max-response-length 4096 \
      --output "${group}"
    echo "== seed ${seed} group metrics: ${group} =="
  fi
}

for seed in ${SEEDS}; do
  case "${seed}" in
    ''|*[!0-9]*) echo "FATAL: SEEDS must be a whitespace list of integers (got: '${seed}')"; exit 1 ;;
  esac
done

for seed in ${SEEDS}; do
  run_seed "${seed}"
done

echo "== seed sequence completed =="
echo "runs: ${OUTPUT_ROOT}/qwen35_runs/k1_boxedonly_nonthinking_sampled_r4096_n4_overnight120_seed{${SEEDS// /,}}/"

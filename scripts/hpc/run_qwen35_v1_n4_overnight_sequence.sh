#!/usr/bin/env bash
# One unattended Track-A sequence for a roughly six-hour remaining instance:
# 5-step instrumentation canary -> structural group-metric gate -> fresh
# 120-step run -> final group analysis.  Launch this script itself with nohup.
set -euo pipefail

cd "$(dirname "$0")/../.."

DTOPD_ROOT=${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}
DTOPD_PYTHON=${DTOPD_PYTHON:-${DTOPD_ROOT}/envs/va-opd-qwen35-cu128/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-${DTOPD_ROOT}/fc-opd-storage/logs}

if [ "${DRY_RUN:-0}" != "1" ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "FATAL: project has tracked changes; use a clean checkout for the overnight research run."
  git status --short --untracked-files=no
  exit 1
fi

run_phase() {
  local experiment_name=$1
  local total_steps=$2
  local test_freq=$3
  local save_freq=$4
  local validation_dir=$5
  local rollout_dir=$6
  local metadata_dir=$7

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
  TOTAL_TRAINING_STEPS="${total_steps}" \
  MAX_RESPONSE_LENGTH=4096 \
  PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
  SAVE_FREQ="${save_freq}" \
  TEST_FREQ="${test_freq}" \
  EXPERIMENT_NAME="${experiment_name}" \
  VALIDATION_DATA_DIR="${validation_dir}" \
  TRAIN_ROLLOUT_DATA_DIR="${rollout_dir}" \
  RUN_METADATA_DIR="${metadata_dir}" \
  bash scripts/run_qwen35_formal.sh \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
}

CANARY_TAG=k1_boxedonly_nonthinking_sampled_r4096_n4_group_canary_r1
CANARY_META=${OUTPUT_ROOT}/qwen35_runs/${CANARY_TAG}
CANARY_VAL=${OUTPUT_ROOT}/val_dump_${CANARY_TAG}
CANARY_ROLLOUT=${CANARY_META}/train_rollouts
CANARY_GROUP=${CANARY_META}/group_clip_summary.json

echo "== overnight phase 1/3: 5-step group instrumentation canary =="
run_phase \
  qwen3_6_27b_to_qwen3_5_4b_k1_tasktrue_fcop_pvboxed_only_nonthinking_sampled_r4096_n4_group_canary_r1 \
  5 5 -1 "${CANARY_VAL}" "${CANARY_ROLLOUT}" "${CANARY_META}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  run_phase \
    qwen3_6_27b_to_qwen3_5_4b_k1_tasktrue_fcop_pvboxed_only_nonthinking_sampled_r4096_n4_overnight120_r1 \
    120 20 30 \
    "${OUTPUT_ROOT}/val_dump_k1_boxedonly_nonthinking_sampled_r4096_n4_overnight120_r1" \
    "${OUTPUT_ROOT}/qwen35_runs/k1_boxedonly_nonthinking_sampled_r4096_n4_overnight120_r1/train_rollouts" \
    "${OUTPUT_ROOT}/qwen35_runs/k1_boxedonly_nonthinking_sampled_r4096_n4_overnight120_r1"
  echo "DRY_RUN composed both canary and long-run contracts successfully; stopping before artifact analysis."
  exit 0
fi

"${DTOPD_PYTHON}" scripts/qwen35_group_clip_analysis.py \
  --rollout-dir "${CANARY_ROLLOUT}" \
  --max-response-length 4096 \
  --output "${CANARY_GROUP}"

"${DTOPD_PYTHON}" - "${CANARY_GROUP}" <<'PY'
import json
import sys

path = sys.argv[1]
summary = json.load(open(path, encoding="utf-8"))
steps = summary.get("steps", {})
if len(steps) != 5:
    raise SystemExit(f"FATAL: expected 5 canary rollout dumps, found {len(steps)}")
for step, metrics in steps.items():
    group = metrics.get("group", {})
    if metrics.get("n") != 24:
        raise SystemExit(f"FATAL: step {step} expected 24 sequences, got {metrics.get('n')}")
    if group.get("n_groups") != 6 or group.get("rollouts_per_group") != 4.0:
        raise SystemExit(f"FATAL: step {step} invalid n=4 grouping: {group}")
print("Canary group contract: PASS (5 steps, 6 groups x 4 rollouts)")
PY

LONG_TAG=k1_boxedonly_nonthinking_sampled_r4096_n4_overnight120_r1
LONG_META=${OUTPUT_ROOT}/qwen35_runs/${LONG_TAG}
LONG_VAL=${OUTPUT_ROOT}/val_dump_${LONG_TAG}
LONG_ROLLOUT=${LONG_META}/train_rollouts
LONG_GROUP=${LONG_META}/group_clip_summary.json

echo "== overnight phase 2/3: fresh 120-step run =="
run_phase \
  qwen3_6_27b_to_qwen3_5_4b_k1_tasktrue_fcop_pvboxed_only_nonthinking_sampled_r4096_n4_overnight120_r1 \
  120 20 30 "${LONG_VAL}" "${LONG_ROLLOUT}" "${LONG_META}"

echo "== overnight phase 3/3: final group analysis =="
"${DTOPD_PYTHON}" scripts/qwen35_group_clip_analysis.py \
  --rollout-dir "${LONG_ROLLOUT}" \
  --max-response-length 4096 \
  --output "${LONG_GROUP}"

echo "== overnight sequence completed naturally =="
echo "manifest: ${LONG_META}/run_manifest.json"
echo "group metrics: ${LONG_GROUP}"

#!/usr/bin/env bash
# First n=4 trainability smoke after D3.  This freezes the D3 sampler and
# non-thinking contract explicitly so training and validation cannot drift.
set -euo pipefail

cd "$(dirname "$0")/../.."

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
TOTAL_TRAINING_STEPS=20 \
MAX_RESPONSE_LENGTH=4096 \
PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
SAVE_FREQ=-1 \
TEST_FREQ=5 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_tasktrue_fcop_pvboxed_only_nonthinking_sampled_r4096_n4_v1_smoke \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_boxedonly_nonthinking_sampled_r4096_n4_smoke \
RUN_METADATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/qwen35_runs/k1_boxedonly_nonthinking_sampled_r4096_n4_smoke \
nohup bash scripts/run_qwen35_formal.sh \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  > "artifacts/fc_opd/nohup_v1_n4_nonthinking_sampled_r4096_smoke_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "n=4 smoke launched pid=$!; 6 prompts x 4 rollouts, non-thinking sampled cap=4096"

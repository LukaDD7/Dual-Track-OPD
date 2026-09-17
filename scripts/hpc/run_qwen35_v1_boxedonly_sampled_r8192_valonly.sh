#!/usr/bin/env bash
# Step D1-L: observe the natural completion-length and answer-quality
# distribution under the same non-greedy sampler as D1, changing only the
# response cap from 2048 to 8192.  Validation only; this does not approve an
# 8192-token training contract.
set -euo pipefail

cd "$(dirname "$0")/../.."

FORMAL_GPUS=0,1,2,3 \
NGPUS_PER_NODE=3 \
TRAIN_BATCH_SIZE=24 \
PPO_MINI_BATCH_SIZE=24 \
USE_FCOP_DATASET=1 \
PROMPT_VERSION=boxed_only \
USE_TASK_REWARDS=False \
ROLLOUT_N=1 \
TRAINER_USE_V1=True \
RESUME_MODE=disable \
VAL_BEFORE_TRAIN=True \
TOTAL_TRAINING_STEPS=1 \
MAX_RESPONSE_LENGTH=8192 \
PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
SAVE_FREQ=-1 \
TEST_FREQ=5 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_taskfalse_fcop_pvboxed_only_sampled_r8192_n1_v1_valonly \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_boxedonly_sampled_r8192_d1l \
RUN_METADATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/qwen35_runs/k1_boxedonly_sampled_r8192_d1l \
nohup bash scripts/run_qwen35_formal.sh \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  trainer.val_only=True \
  > "artifacts/fc_opd/nohup_v1_boxedonly_sampled_r8192_d1l_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "Step D1-L launched pid=$!; sampled validation cap=8192, thinking unchanged"

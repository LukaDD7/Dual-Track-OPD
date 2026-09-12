#!/usr/bin/env bash
# Step D2: same sampled validation as D1, with the single additional change
# enable_thinking=False.  The boxed_only prompt still requests brief visible
# reasoning, so this is not an answer-only contract.
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
MAX_RESPONSE_LENGTH=2048 \
PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
SAVE_FREQ=-1 \
TEST_FREQ=5 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_taskfalse_fcop_pvboxed_only_nonthinking_sampled_n1_v1_valonly \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_boxedonly_nonthinking_sampled_d2 \
RUN_METADATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/qwen35_runs/k1_boxedonly_nonthinking_sampled_d2 \
nohup bash scripts/run_qwen35_formal.sh \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  trainer.val_only=True \
  > "artifacts/fc_opd/nohup_v1_boxedonly_nonthinking_sampled_d2_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "Step D2 launched pid=$!; sampled validation adds enable_thinking=False"

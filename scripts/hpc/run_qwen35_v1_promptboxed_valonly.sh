#!/usr/bin/env bash
# Step C launcher: 2048-token val-only truncation ablation with the versioned
# "concise closing" prompt (PROMPT_VERSION=boxed_only).
# Purpose: after Step B showed clip rate stays at 96% at 4096, test whether a
# separately versioned closing instruction (final line = \boxed{<answer>} only)
# stops the model writing reasoning up to the token budget. One variable changed
# vs experiment #2 step 0: the prompt text. Length stays 2048; scorer untouched.
# See docs/qwen35_prompt_boxedonly_ablation_next_run.md.
# Run: bash scripts/hpc/run_qwen35_v1_promptboxed_valonly.sh
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
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_taskfalse_fcop_pvboxed_only_n1_v1_valonly \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_promptfix_boxedonly_r3 \
RUN_METADATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/qwen35_runs/k1_promptfix_boxedonly_r3 \
nohup bash scripts/run_qwen35_formal.sh trainer.val_only=True > "artifacts/fc_opd/nohup_v1_pvboxed_only_valonly_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "Step C launched pid=$!; tail -f artifacts/fc_opd/nohup_v1_pvboxed_only_valonly_*.log"

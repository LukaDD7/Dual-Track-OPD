#!/usr/bin/env bash
# Step B launcher: 4096-token validation-only truncation ablation (1 val step).
# See docs/qwen35_v1_truncation_next_steps_for_claude.md section 4.
# Run: bash scripts/hpc/run_qwen35_v1_stepb.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

FORMAL_GPUS=0,1,2,3 \
NGPUS_PER_NODE=3 \
TRAIN_BATCH_SIZE=24 \
PPO_MINI_BATCH_SIZE=24 \
USE_FCOP_DATASET=1 \
USE_TASK_REWARDS=False \
ROLLOUT_N=1 \
TRAINER_USE_V1=True \
RESUME_MODE=disable \
VAL_BEFORE_TRAIN=True \
TOTAL_TRAINING_STEPS=1 \
MAX_RESPONSE_LENGTH=4096 \
PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
SAVE_FREQ=-1 \
TEST_FREQ=5 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_taskfalse_fcop_n1_v1_r4096_valonly \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_promptfix_r4096_valonly \
RUN_METADATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/qwen35_runs/k1_promptfix_r4096_valonly \
nohup bash scripts/run_qwen35_formal.sh trainer.val_only=True > "artifacts/fc_opd/nohup_v1_r4096_valonly_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "Step B launched pid=$!; tail -f artifacts/fc_opd/nohup_v1_r4096_valonly_*.log"

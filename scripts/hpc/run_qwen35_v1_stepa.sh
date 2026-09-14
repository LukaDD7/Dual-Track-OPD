#!/usr/bin/env bash
# Step A launcher: v1 task-reward integration smoke (20 steps).
# See docs/qwen35_v1_truncation_next_steps_for_claude.md section 3.
# Run: bash scripts/hpc/run_qwen35_v1_stepa.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

FORMAL_GPUS=0,1,2,3 \
NGPUS_PER_NODE=3 \
TRAIN_BATCH_SIZE=24 \
PPO_MINI_BATCH_SIZE=24 \
USE_FCOP_DATASET=1 \
USE_TASK_REWARDS=True \
ROLLOUT_N=1 \
TRAINER_USE_V1=True \
RESUME_MODE=disable \
VAL_BEFORE_TRAIN=True \
TOTAL_TRAINING_STEPS=20 \
SAVE_FREQ=-1 \
TEST_FREQ=5 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_tasktrue_fcop_n1_v1_reward_smoke \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_reward_v1_smoke \
RUN_METADATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/qwen35_runs/k1_reward_v1_smoke_r2 \
nohup bash scripts/run_qwen35_formal.sh > "artifacts/fc_opd/nohup_v1_reward_smoke_r2_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "Step A launched pid=$!; tail -f artifacts/fc_opd/nohup_v1_reward_smoke_r2_*.log"

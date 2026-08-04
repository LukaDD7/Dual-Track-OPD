#!/usr/bin/env bash
# Qwen3.5 Step D: answer-only closing-behavior gate at 2048 tokens.
#
# Validation only.  This launcher does not approve answer_only as a training
# prompt.  Compare clip/boxed/accuracy with Step C before any optimizer run.
# See docs/qwen35_stepc_codex_decisions_and_next_run.md.
set -euo pipefail

cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
mkdir -p artifacts/fc_opd

FORMAL_GPUS=0,1,2,3 \
NGPUS_PER_NODE=3 \
TRAIN_BATCH_SIZE=24 \
PPO_MINI_BATCH_SIZE=24 \
USE_FCOP_DATASET=1 \
PROMPT_VERSION=answer_only \
USE_TASK_REWARDS=False \
ROLLOUT_N=1 \
TRAINER_USE_V1=True \
RESUME_MODE=disable \
VAL_BEFORE_TRAIN=False \
TOTAL_TRAINING_STEPS=1 \
SAVE_FREQ=-1 \
TEST_FREQ=-1 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_taskfalse_fcop_pvanswer_only_n1_v1_valonly \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_promptfix_answeronly_r1 \
RUN_METADATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/qwen35_runs/k1_promptfix_answeronly_r1 \
nohup bash scripts/run_qwen35_formal.sh trainer.val_only=True > "artifacts/fc_opd/nohup_v1_pvanswer_only_valonly_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

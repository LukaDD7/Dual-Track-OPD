#!/usr/bin/env bash
#
# SFT-then-RL 管线 | RL 段（今晚窗口 / 常规全量）
#   数据: geometry3k 全量 train 2101 / 官方 test 601（verl GRPO V0 + vLLM rollout）
#   模型: SFT_RL_MODEL 必须指向 SFT ckpt 的 huggingface 目录
#         （如 <ckpt>/global_step_*/huggingface，需含 model.safetensors）
#   奖励: scripts/sft_rl/geo3k_reward.py（mathruler + LaTeX 归一化，已验证）
#
# 用法（GPU 节点）:
#   SFT_RL_MODEL=<sft-ckpt>/global_step_*/huggingface \
#     bash scripts/sft_rl/run_grpo_overnight.sh
# 可选 env:
#   SFT_RL_GPUS="0,1,2,3"  TRAIN_BATCH_SIZE=16  PPO_MINI_BATCH_SIZE=8
#   ROLLOUT_N=4  MAX_RESPONSE_LENGTH=1024  TOTAL_EPOCHS=1
#   SFT_RL_SMOKE=1  改用 64/32 行 smoke 集（默认 0=全量）
#   SFT_RL_TRAIN/SFT_RL_VAL  覆盖数据文件（如 MMF RL: mmfinereason_rl.parquet）
#   SFT_RL_REWARD=file://.../mmf_reward.py  覆盖规则奖励
#   SFT_RL_MAX_SAMPLES=N   训练集采样上限（GRPO 大池首跑建议 2000）
set -euo pipefail

DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
ENV_PREFIX="${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1"
CUDA_HOME="${DTOPD_ROOT}/envs/cuda132-toolchain"
VERL_DIR="${DTOPD_ROOT}/fc-opd-storage/backends/verl-qwen35-v090-cu132"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${ENV_PREFIX}/lib/python3.12/site-packages/torch/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="${DTOPD_ROOT}/.cache/flashinfer"
export VERL_DIAG_DIR="${DTOPD_ROOT}/fc-opd-storage/logs"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export RAY_LOG_TO_STDERR=1 RAY_DEDUP_LOGS=0
export VLLM_LOGGING_STREAM=ext://sys.stderr
ulimit -c 0

source "${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
ray stop --force 2>/dev/null || true
sleep 2

SFT_RL_GPUS="${SFT_RL_GPUS:-0,1,2,3}"
SFT_RL_MODEL="${SFT_RL_MODEL:-}"
export CUDA_VISIBLE_DEVICES="${SFT_RL_GPUS}"
N_GPUS="$(echo "${SFT_RL_GPUS}" | tr ',' '\n' | wc -l)"

SMOKE_TRAIN="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/geo3k_grpo_smoke64.parquet"
SMOKE_VAL="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/geo3k_grpo_val_smoke32.parquet"
FULL_TRAIN="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/geometry3k_full/train.parquet"
FULL_VAL="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/geo3k_grpo_val_test.parquet"
if [[ "${SFT_RL_SMOKE:-0}" == "1" ]]; then
    TRAIN_FILE="${SMOKE_TRAIN}"; VAL_FILE="${SMOKE_VAL}"
else
    TRAIN_FILE="${FULL_TRAIN}"; VAL_FILE="${FULL_VAL}"
fi
TRAIN_FILE="${SFT_RL_TRAIN:-${TRAIN_FILE}}"
VAL_FILE="${SFT_RL_VAL:-${VAL_FILE}}"

[ -n "${SFT_RL_MODEL}" ] || { echo "FATAL: SFT_RL_MODEL required (point to SFT ckpt huggingface dir)"; exit 1; }
for p in "${SFT_RL_MODEL}/config.json" "${SFT_RL_MODEL}/model.safetensors" "${TRAIN_FILE}" "${VAL_FILE}"; do
  [ -f "${p}" ] || { echo "FATAL: missing ${p}"; exit 1; }
done

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-4}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="null"
PROJECT_NAME="verl_sftrl"
EXPERIMENT_NAME="qwen3vl_grpo_geo3k_overnight_$(date +%Y%m%d_%H%M)"
LOG_FILE="${DTOPD_ROOT}/fc-opd-storage/logs/${EXPERIMENT_NAME}.log"
CKPT_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/${EXPERIMENT_NAME}"
REWARD_PATH="${SFT_RL_REWARD:-file://${DTOPD_ROOT}/projects/Dual-Track-OPD/scripts/sft_rl/geo3k_reward.py}"
MAX_SAMPLES_ARG=""
if [[ -n "${SFT_RL_MAX_SAMPLES:-}" ]]; then
    MAX_SAMPLES_ARG="data.train_max_samples=${SFT_RL_MAX_SAMPLES}"
fi

echo "== GRPO overnight: model=${SFT_RL_MODEL} GPUs=${CUDA_VISIBLE_DEVICES} (n=${N_GPUS}) =="
echo "== batch=${TRAIN_BATCH_SIZE} mini=${PPO_MINI_BATCH_SIZE} rollout_n=${ROLLOUT_N} epochs=${TOTAL_EPOCHS} =="
echo "== train=${TRAIN_FILE} =="
echo "== val=${VAL_FILE} =="

cd "${VERL_DIR}/examples/grpo_trainer"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  ${MAX_SAMPLES_ARG:+${MAX_SAMPLES_ARG}} \
  data.image_key=images \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=1024 \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${SFT_RL_MODEL}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq=25 \
  trainer.test_freq=10 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.use_v1=False \
  reward.custom_reward_function.path="${REWARD_PATH}" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  2>&1 | tee "${LOG_FILE}"

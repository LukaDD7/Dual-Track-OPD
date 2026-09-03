#!/usr/bin/env bash
#
# SFT-then-RL 管线验证 | Stage 2: GRPO (verl 0.9.0 V0) on geometry3k
#   环境: va-opd-qwen35-v090-cu132-r595-v1 (torch 2.13+cu132 / vllm 0.27.1 / verl 0.9.0)
#   数据: fc-opd-storage/outputs/fc_opd/geometry3k_full/train.parquet (2101 rows)
#   用途: GPU 节点上验证 Qwen3-VL 的 GRPO vLLM rollout + FSDP2 训练链路
#
# 用法（GPU 节点）:
#   bash scripts/sft_rl/run_grpo_qwen3vl_geo3k_smoke.sh
# 可选:
#   SFT_RL_GPUS="0,1,2,3"   参与 GPU（默认 0,1,2,3，可只给 2 卡）
#   SFT_RL_MODEL=...        默认 Qwen3-VL-4B-Instruct（本地路径）
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
SFT_RL_MODEL="${SFT_RL_MODEL:-${DTOPD_ROOT}/models/Qwen3-VL-4B-Instruct}"
export CUDA_VISIBLE_DEVICES="${SFT_RL_GPUS}"
N_GPUS="$(echo "${SFT_RL_GPUS}" | tr ',' '\n' | wc -l)"

SMOKE_TRAIN="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/geo3k_grpo_smoke64.parquet"
SMOKE_VAL="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/geo3k_grpo_val_smoke32.parquet"
FULL_TRAIN="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/geometry3k_full/train.parquet"
FULL_VAL="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/geo3k_grpo_val_test.parquet"
TRAIN_FILE="${SFT_RL_SMOKE:-1}"
if [[ "${SFT_RL_SMOKE:-1}" == "1" && -f "${SMOKE_TRAIN}" ]]; then
    TRAIN_FILE="${SMOKE_TRAIN}"
    VAL_FILE="${SMOKE_VAL}"
else
    TRAIN_FILE="${FULL_TRAIN}"
    VAL_FILE="${FULL_VAL}"
fi
for p in "${SFT_RL_MODEL}/config.json" "${TRAIN_FILE}" "${VAL_FILE}"; do
  [ -f "${p}" ] || { echo "FATAL: missing ${p}"; exit 1; }
done

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-2}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
if [[ "${SFT_RL_SMOKE:-1}" == "1" && -f "${SMOKE_TRAIN}" ]]; then
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-8}"
else
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
fi
PROJECT_NAME="verl_sftrl"
EXPERIMENT_NAME="qwen3vl_grpo_geo3k_smoke_$(date +%Y%m%d_%H%M)"
LOG_FILE="${DTOPD_ROOT}/fc-opd-storage/logs/${EXPERIMENT_NAME}.log"
CKPT_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/${EXPERIMENT_NAME}"
REWARD_PATH="file://${DTOPD_ROOT}/projects/Dual-Track-OPD/scripts/sft_rl/geo3k_reward.py"

echo "== GRPO smoke: model=${SFT_RL_MODEL} GPUs=${CUDA_VISIBLE_DEVICES} (n=${N_GPUS}) =="
echo "== batch=${TRAIN_BATCH_SIZE} mini=${PPO_MINI_BATCH_SIZE} rollout_n=${ROLLOUT_N} =="
echo "== train=${TRAIN_FILE} =="
echo "== val=${VAL_FILE} =="
echo "== total_training_steps=${TOTAL_TRAINING_STEPS} =="

cd "${VERL_DIR}/examples/grpo_trainer"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
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
  trainer.save_freq=100 \
  trainer.test_freq=5 \
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

#!/usr/bin/env bash
#
# SFT-then-RL 管线验证 | Stage 1: SFT (verl sft_trainer) on geometry3k smoke
#   环境: va-opd-qwen35-v090-cu132-r595-v1 (torch 2.13+cu132 / verl 0.9.0)
#   数据: fc-opd-storage/outputs/fc_opd/sft_rl/geo3k_sft_smoke84.parquet
#         （由 scripts/sft_rl/prep_geometry3k_sft_smoke.py 生成，messages+images 列）
#   用途: GPU 节点上验证 Qwen3-VL SFT (FSDP2 + VLM processor) 链路
#
# 用法（GPU 节点）:
#   bash scripts/sft_rl/run_sft_qwen3vl_geo3k_smoke.sh
# 可选:
#   SFT_RL_GPUS="0,1,2,3"   参与 GPU（默认 0,1,2,3）
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

DATA_FILE="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/geo3k_sft_smoke84.parquet"
for p in "${SFT_RL_MODEL}/config.json" "${DATA_FILE}"; do
  [ -f "${p}" ] || { echo "FATAL: missing ${p}"; exit 1; }
done

EXPERIMENT_NAME="qwen3vl_sft_geo3k_smoke_$(date +%Y%m%d_%H%M)"
LOG_FILE="${DTOPD_ROOT}/fc-opd-storage/logs/${EXPERIMENT_NAME}.log"
CKPT_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/${EXPERIMENT_NAME}"
mkdir -p "${CKPT_DIR}"

echo "== SFT smoke: model=${SFT_RL_MODEL} GPUs=${CUDA_VISIBLE_DEVICES} (n=${N_GPUS}) =="

cd "${VERL_DIR}/examples/sft/vlm"
torchrun --standalone --nnodes=1 --nproc-per-node="${N_GPUS}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${DATA_FILE}" \
  data.val_files="${DATA_FILE}" \
  data.train_batch_size=8 \
  data.max_length=2048 \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.use_dynamic_bsz=True \
  data.max_token_len_per_gpu=65536 \
  model.path="${SFT_RL_MODEL}" \
  model.use_remove_padding=True \
  engine=fsdp \
  optim=fsdp \
  optim.lr=2e-5 \
  optim.lr_warmup_steps_ratio=0.01 \
  optim.weight_decay=0.1 \
  optim.betas="[0.9,0.95]" \
  optim.clip_grad=1.0 \
  optim.min_lr_ratio=0.1 \
  optim.warmup_style=cosine \
  engine.ulysses_sequence_parallel_size=1 \
  engine.strategy=fsdp2 \
  engine.fsdp_size=-1 \
  trainer.test_freq=5 \
  trainer.save_freq=100 \
  trainer.logger='["console"]' \
  trainer.project_name=verl_sftrl \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=1 \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.resume_mode=disable \
  trainer.max_ckpt_to_keep=2 \
  checkpoint.save_contents="[model,hf_model]" \
  2>&1 | tee "${LOG_FILE}"

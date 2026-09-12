#!/usr/bin/env bash
#
# 从 verl GRPO FSDP checkpoint 导出 HF 权重（评测 / 给 RL-only 对齐用）
#   verl GRPO 每 25 步存的 ckpt 是 FSDP 分片（model_world_size_*_rank_*.pt），
#   其中的 huggingface/ 只有 config/tokenizer，没有权重；必须用 model_merger
#   合并导出成标准 HF 目录（model.safetensors）。
#
# 用法（GPU 节点，1 张卡即可）:
#   bash scripts/sft_rl/export_hf_from_grpo_ckpt.sh
# 可选 env:
#   SFT_RL_CKPT=.../global_step_184   默认 1511 实验的 global_step_184
#   SFT_RL_GPUS=0                     默认 GPU 0（避开 4-7 的常驻进程）
#   SFT_RL_OUT=...                    默认 fc-opd-storage/outputs/fc_opd/sft_rl/hf/...
set -euo pipefail

DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
ENV_PREFIX="${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1"
CUDA_HOME="${DTOPD_ROOT}/envs/cuda132-toolchain"
VERL_DIR="${DTOPD_ROOT}/fc-opd-storage/backends/verl-qwen35-v090-cu132"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${ENV_PREFIX}/lib/python3.12/site-packages/torch/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_NVLS_ENABLE=0

source "${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

SFT_RL_GPUS="${SFT_RL_GPUS:-0}"
export CUDA_VISIBLE_DEVICES="${SFT_RL_GPUS}"
SFT_RL_CKPT="${SFT_RL_CKPT:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_grpo_mmf_overnight_20260820_1511/global_step_184}"
SFT_RL_OUT="${SFT_RL_OUT:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/hf/qwen3vl_grpo_mmf_184}"

[ -d "${SFT_RL_CKPT}/actor" ] || { echo "FATAL: missing ${SFT_RL_CKPT}/actor"; exit 1; }
[ -f "${SFT_RL_CKPT}/actor/huggingface/config.json" ] || { echo "FATAL: missing config in actor/huggingface"; exit 1; }
mkdir -p "$(dirname "${SFT_RL_OUT}")"

echo "== Export HF: ckpt=${SFT_RL_CKPT} GPU=${CUDA_VISIBLE_DEVICES} =="
echo "== out=${SFT_RL_OUT} =="

cd "${VERL_DIR}"
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${SFT_RL_CKPT}/actor" \
  --target_dir "${SFT_RL_OUT}" \
  --trust-remote-code

[ -f "${SFT_RL_OUT}/model.safetensors" ] || { echo "FATAL: export missing model.safetensors"; exit 1; }
echo "== OK: ${SFT_RL_OUT}/model.safetensors =="

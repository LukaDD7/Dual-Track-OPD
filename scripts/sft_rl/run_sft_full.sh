#!/usr/bin/env bash
#
# SFT-then-RL 管线 | 全量池 SFT（比 warmup 更强的 SFT，供直接 SFT-then-RL 重训）
#   数据: sft_full/（prep_sft_full.py 产物）：
#         MMFineReason 全量 + the_cauldron 16 子集全量
#         （教师 CoT<=8192、整序列估长<=12288 过滤，绝不静默截断）
#   模型: Qwen3-VL-8B-Instruct
#   与 warmup 的关键差异:
#     - 2 epoch（全量池 token 口径约 1.5~2 轮）
#     - 完整 cosine 衰减（lr_scheduler_type=cosine, warmup 5%, min_lr_ratio 0.1）
#     - lr 2e-5, wd 0.01, betas [0.9,0.999], seed 1234
#     - max_length=12288（长 CoT 不截断）、train_batch_size=512（token 预算主导）
#     - checkpoint 每 400 步保存、最多留 2 份（省磁盘）
#     - resume_mode=auto（实例回收后重挂可续训）
#
# 用法（GPU 节点）:
#   bash scripts/sft_rl/run_sft_full.sh
# 可选 env:
#   SFT_RL_DATA=...     训练 parquet 目录（默认 .../sft_rl/sft_full/train）
#   SFT_RL_VAL=...      验证 parquet（默认 .../sft_rl/sft_full/val）
#   SFT_RL_MODEL=...    基座模型
#   SFT_RL_GPUS="0,1,2,3,4,5,6,7"
#   SFT_RL_EPOCHS=2     total_epochs
#   SFT_RL_BATCH=512    train_batch_size（软上限，token 预算主导）
#   SFT_RL_MAX_TOKEN=98304  动态 bsz 每 GPU token 预算（8B 若 OOM 降到 65536）
#   SFT_RL_MAX_LEN=12288   整序列上限（须与 prep 的 --max-seq-tokens 一致）
#   SFT_RL_SAVE_FREQ=400   checkpoint 频率（低频率省磁盘；可再调高）
#   SFT_RL_NAME=...     固定实验名（续训时与原实验一致）
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
if [[ "${SFT_RL_SKIP_RAY_STOP:-0}" != "1" ]]; then
    ray stop --force 2>/dev/null || true
    sleep 2
fi

SFT_RL_GPUS="${SFT_RL_GPUS:-0,1,2,3,4,5,6,7}"
SFT_RL_MODEL="${SFT_RL_MODEL:-${DTOPD_ROOT}/models/Qwen3-VL-8B-Instruct}"
SFT_RL_DATA="${SFT_RL_DATA:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/sft_full/train}"
SFT_RL_VAL="${SFT_RL_VAL:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/sft_full/val}"
export CUDA_VISIBLE_DEVICES="${SFT_RL_GPUS}"
N_GPUS="$(echo "${SFT_RL_GPUS}" | tr ',' '\n' | wc -l)"
SHARDS=()
VAL_SHARDS=()

if [[ -d "${SFT_RL_DATA}" ]]; then
    mapfile -t SHARDS < <(ls "${SFT_RL_DATA}"/sft_full_train__part_*.parquet 2>/dev/null)
    if (( ${#SHARDS[@]} == 0 )); then
        echo "FATAL: no sft_full_train__part_*.parquet under ${SFT_RL_DATA}"; exit 1
    fi
    SFT_RL_DATA_ARG="data.train_files=[$(IFS=,; echo "${SHARDS[*]}")]"
    mapfile -t VAL_SHARDS < <(ls "${SFT_RL_VAL}"/sft_full_val__part_*.parquet 2>/dev/null)
    if (( ${#VAL_SHARDS[@]} == 0 )); then
        echo "FATAL: no sft_full_val__part_*.parquet under ${SFT_RL_VAL}"; exit 1
    fi
    SFT_RL_VAL_ARG="data.val_files=[$(IFS=,; echo "${VAL_SHARDS[*]}")]"
else
    SFT_RL_DATA_ARG="data.train_files=${SFT_RL_DATA}"
    SFT_RL_VAL_ARG="data.val_files=${SFT_RL_VAL}"
fi

for p in "${SFT_RL_MODEL}/config.json"; do
  [ -f "${p}" ] || { echo "FATAL: missing ${p}"; exit 1; }
done

SFT_RL_EPOCHS="${SFT_RL_EPOCHS:-2}"
SFT_RL_SAVE_FREQ="${SFT_RL_SAVE_FREQ:-400}"
SFT_RL_MAX_LEN="${SFT_RL_MAX_LEN:-12288}"
SFT_RL_MAX_TOKEN="${SFT_RL_MAX_TOKEN:-98304}"
SFT_RL_BATCH="${SFT_RL_BATCH:-512}"
EXPERIMENT_NAME="${SFT_RL_NAME:-qwen3vl_sft_full_$(date +%Y%m%d_%H%M)}"
LOG_FILE="${DTOPD_ROOT}/fc-opd-storage/logs/${EXPERIMENT_NAME}.log"
CKPT_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/${EXPERIMENT_NAME}"
mkdir -p "${CKPT_DIR}"

echo "== SFT full: model=${SFT_RL_MODEL} GPUs=${CUDA_VISIBLE_DEVICES} (n=${N_GPUS}) =="
echo "== train=${SFT_RL_DATA} ($((${#SHARDS[@]})) shards) =="
echo "== val=${SFT_RL_VAL} ($((${#VAL_SHARDS[@]})) shards) epochs=${SFT_RL_EPOCHS} =="
echo "== save_freq=${SFT_RL_SAVE_FREQ} max_len=${SFT_RL_MAX_LEN} token_budget=${SFT_RL_MAX_TOKEN} batch=${SFT_RL_BATCH} =="
echo "== ckpt=${CKPT_DIR} log=${LOG_FILE} =="

cd "${VERL_DIR}/examples/sft/vlm"
torchrun --standalone --nnodes=1 --nproc-per-node="${N_GPUS}" \
  -m verl.trainer.sft_trainer \
  "${SFT_RL_DATA_ARG}" \
  "${SFT_RL_VAL_ARG}" \
  data.train_batch_size="${SFT_RL_BATCH}" \
  data.max_length="${SFT_RL_MAX_LEN}" \
  data.pad_mode=no_padding \
  data.truncation=right \
  data.use_dynamic_bsz=True \
  data.max_token_len_per_gpu="${SFT_RL_MAX_TOKEN}" \
  model.path="${SFT_RL_MODEL}" \
  model.use_remove_padding=True \
  engine=fsdp \
  optim=fsdp \
  optim.lr="${SFT_RL_LR:-2e-5}" \
  optim.lr_scheduler_type=cosine \
  optim.lr_warmup_steps_ratio="${SFT_RL_WARMUP:-0.05}" \
  optim.min_lr_ratio=0.1 \
  optim.weight_decay="${SFT_RL_WD:-0.01}" \
  optim.betas="[0.9,0.999]" \
  optim.clip_grad=1.0 \
  optim.warmup_style=cosine \
  engine.ulysses_sequence_parallel_size=1 \
  engine.strategy=fsdp2 \
  engine.fsdp_size=-1 \
  trainer.seed="${SFT_RL_SEED:-1234}" \
  trainer.test_freq=100 \
  trainer.save_freq="${SFT_RL_SAVE_FREQ}" \
  trainer.logger='["console"]' \
  trainer.project_name=verl_sftrl \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs="${SFT_RL_EPOCHS}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.resume_mode=auto \
  trainer.max_ckpt_to_keep=2 \
  checkpoint.save_contents="[model,hf_model]" \
  2>&1 | tee "${LOG_FILE}"

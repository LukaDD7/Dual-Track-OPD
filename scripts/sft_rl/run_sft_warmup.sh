#!/usr/bin/env bash
#
# SFT-then-RL 管线 | SFT warmup（今晚 7 小时窗口用）
#   数据: sft_warmup_train/val.parquet（prep_sft_warmup.py 生成：
#         MMFineReason 长 CoT 分层 100K + the_cauldron 16 子集每子集 4K，
#         长 CoT 约 61% 行 / ~85% token，避免短 GT 淹没长 CoT）
#   模型: Qwen3-VL-8B-Instruct（SFT_RL_MODEL 可覆盖）
#   SFT 超参对齐 arXiv:2604.23747 §A：lr 5e-5、cosine、10% warmup、min_ratio 0.1、
#         wd 0.01、betas [0.9,0.999]、3 epoch（全量 487K 口径；本 warmup 因
#         数据仅 ~164K 且 token 预算主导，默认 epoch 也为 3 对齐见 --epochs 讨论）
#   特点: max_length=12288（长 CoT 不截尾，与 prep 的 --max-seq-tokens 一致）、
#         truncation=right、save_contents=[model,hf_model]、resume_mode=auto
#
# 用法（GPU 节点）:
#   bash scripts/sft_rl/run_sft_warmup.sh
# 可选 env:
#   SFT_RL_DATA=...     训练 parquet（默认 warmup 产物）
#   SFT_RL_VAL=...      验证 parquet
#   SFT_RL_MODEL=...    基座模型
#   SFT_RL_GPUS="4,5,6,7"   (0-3 留给 PTD-PO track)
#   SFT_RL_EPOCHS=3     total_epochs（论文 3）
#   SFT_RL_BATCH=64     train_batch_size（论文 64；本文档动批量下是软上界）
#   SFT_RL_LR=5e-5      峰值学习率（论文 5e-5）
#   SFT_RL_WARMUP=0.10  lr warmup 比例（论文 10%）
#   SFT_RL_MAX_TOKEN=98304  动态 bsz 每 GPU token 预算（8B 若 OOM 降到 65536）
#   SFT_RL_MAX_LEN=12288   整序列上限（须与 prep 的 --max-seq-tokens 一致）
#   SFT_RL_SAVE_FREQ=...   checkpoint 频率
#   SFT_RL_NAME=...        固定实验名（续训时与原实验一致）
#   SFT_RL_SKIP_RAY_STOP=1 跳过 ray stop（与正在运行的 GRPO 并发时必设，
#                          否则会杀掉 GRPO 的 Ray；SFT 本身不用 Ray）
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
# dataloader 的 "Kwargs passed to processor.__call__" 警告曾以 ~40-250 条/step
# 灌爆 stdout 管道（tee 一停读，rank0 就阻塞在 pipe_write，全训练冻结 8h+，
# 20260826 run 两次事故）。压到 error 级别，管道流量降 99%。
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
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

SFT_RL_GPUS="${SFT_RL_GPUS:-4,5,6,7}"
SFT_RL_MODEL="${SFT_RL_MODEL:-${DTOPD_ROOT}/models/Qwen3-VL-8B-Instruct}"
SFT_RL_DATA="${SFT_RL_DATA:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/warmup_t2}"
SFT_RL_VAL="${SFT_RL_VAL:-}"
export CUDA_VISIBLE_DEVICES="${SFT_RL_GPUS}"
N_GPUS="$(echo "${SFT_RL_GPUS}" | tr ',' '\n' | wc -l)"
SHARDS=()

# 数据默认是分片目录：verl SFT 用 pandas 读 parquet，大文件会触发 pyarrow
# 嵌套 chunk 限制，所以以 hydra list 形式传多个小分片（<1500 行/文件）。
if [[ -d "${SFT_RL_DATA}" ]]; then
    mapfile -t SHARDS < <(ls "${SFT_RL_DATA}"/sft_warmup_train__part_*.parquet 2>/dev/null)
    if (( ${#SHARDS[@]} == 0 )); then
        echo "FATAL: no sft_warmup_train__part_*.parquet under ${SFT_RL_DATA}"; exit 1
    fi
    SFT_RL_DATA_ARG="data.train_files=[$(IFS=,; echo "${SHARDS[*]}")]"
    if [[ -z "${SFT_RL_VAL}" ]]; then
        mapfile -t VAL_SHARDS < <(ls "${SFT_RL_DATA}"/sft_warmup_val__part_*.parquet 2>/dev/null)
        SFT_RL_VAL="${VAL_SHARDS[0]:-}"
    fi
else
    SFT_RL_DATA_ARG="data.train_files=${SFT_RL_DATA}"
fi

for p in "${SFT_RL_MODEL}/config.json" "${SFT_RL_VAL}"; do
  [ -f "${p}" ] || { echo "FATAL: missing ${p}"; exit 1; }
done

SFT_RL_EPOCHS="${SFT_RL_EPOCHS:-3}"
SFT_RL_SAVE_FREQ="${SFT_RL_SAVE_FREQ:-200}"
SFT_RL_MAX_LEN="${SFT_RL_MAX_LEN:-12288}"
SFT_RL_MAX_TOKEN="${SFT_RL_MAX_TOKEN:-98304}"
SFT_RL_BATCH="${SFT_RL_BATCH:-64}"
SFT_RL_LR="${SFT_RL_LR:-5e-5}"
SFT_RL_WARMUP="${SFT_RL_WARMUP:-0.10}"
SFT_RL_WD="${SFT_RL_WD:-0.01}"
SFT_RL_SEED="${SFT_RL_SEED:-1234}"
EXPERIMENT_NAME="${SFT_RL_NAME:-qwen3vl_sft_warmup_$(date +%Y%m%d_%H%M)}"
LOG_FILE="${DTOPD_ROOT}/fc-opd-storage/logs/${EXPERIMENT_NAME}.log"
CKPT_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/${EXPERIMENT_NAME}"
mkdir -p "${CKPT_DIR}"

echo "== SFT warmup: model=${SFT_RL_MODEL} GPUs=${CUDA_VISIBLE_DEVICES} (n=${N_GPUS}) =="
echo "== train=${SFT_RL_DATA} (${#SHARDS[@]} shards) =="
echo "== val=${SFT_RL_VAL} epochs=${SFT_RL_EPOCHS} save_freq=${SFT_RL_SAVE_FREQ} max_len=${SFT_RL_MAX_LEN} token_budget=${SFT_RL_MAX_TOKEN} batch=${SFT_RL_BATCH} lr=${SFT_RL_LR} warmup=${SFT_RL_WARMUP} wd=${SFT_RL_WD} =="
echo "== ckpt=${CKPT_DIR} log=${LOG_FILE} =="

cd "${VERL_DIR}/examples/sft/vlm"
torchrun --standalone --nnodes=1 --nproc-per-node="${N_GPUS}" \
  -m verl.trainer.sft_trainer \
  "${SFT_RL_DATA_ARG}" \
  data.val_files="${SFT_RL_VAL}" \
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
  optim.lr="${SFT_RL_LR}" \
  optim.lr_scheduler_type=cosine \
  optim.lr_warmup_steps_ratio="${SFT_RL_WARMUP}" \
  optim.weight_decay="${SFT_RL_WD}" \
  optim.betas="[0.9,0.999]" \
  optim.clip_grad=1.0 \
  optim.min_lr_ratio=0.1 \
  optim.warmup_style=cosine \
  engine.ulysses_sequence_parallel_size=1 \
  engine.strategy=fsdp2 \
  engine.fsdp_size=-1 \
  trainer.seed="${SFT_RL_SEED}" \
  trainer.test_freq=100 \
  trainer.save_freq="${SFT_RL_SAVE_FREQ}" \
  trainer.logger='["console"]' \
  trainer.project_name=verl_sftrl \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs="${SFT_RL_EPOCHS}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.resume_mode=auto \
  trainer.max_ckpt_to_keep=3 \
  checkpoint.save_contents="[model,hf_model]" \
  2>&1 | tee "${LOG_FILE}"

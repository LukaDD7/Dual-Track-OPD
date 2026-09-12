#!/usr/bin/env bash
#
# TailSFT 复现（arXiv:2608.25756）：MMF-only 单源 SFT 的在线过滤训练臂。
#   与 mmf122k_1ep（qwen3vl_sft_mmf122k_1ep/global_step_1774，B 段 avg 0.3714）
#   唯一差异 = TailSFT 在线序列级过滤（γt 由 ramp 0→0.5），其余超参完全一致。
#   ⚠ 本脚本不覆盖/不动任何标准 SFT 产物：实验名、ckpt 目录、数据目录均为
#     新的 *_tailsft 命名；verl 后端 sft_loss 原样保留（tailsft 仅在
#     data.tailsft.enabled=True 时启用，默认 False 走原路径）。
#
# 数据: $DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft_tailsft/
#       （由 scripts/sft_rl/annotate_tailsft_init_ce.py 生成：train 各 shard
#         追加 init_ce 列 = base 模型 ℓ0；val 原样拷贝，val 路径自动退回
#         纯 CE——所以 val/loss 曲线与标准 SFT 口径可比）
#
# 用法（GPU 节点，先跑完 annotate 脚本）:
#   bash scripts/sft_rl/run_sft_tailsft_mmf.sh
# 可选 env:
#   SFT_RL_TAILSFT_F=0.5          目标过滤比例 γ
#   SFT_RL_TAILSFT_SCHEDULE=ramp  static | ramp
#   SFT_RL_TAILSFT_RAMP=800       ramp 达到满 γ 的步数（≈总步数一半）
#   其余 SFT_RL_* 与 run_sft_warmup.sh 完全同义（GPUS/EPOCHS/LR/...），
#   默认值与本臂设计固定：EPOCHS=1（对齐 mmf122k_1ep 的 1-epoch 口径）。
set -euo pipefail

DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
ENV_PREFIX="${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1"
CUDA_HOME="${DTOPD_ROOT}/envs/cuda132-toolchain"
VERL_DIR="${DTOPD_ROOT}/fc-opd-storage/backends/verl-qwen35-v090-cu132"
REPO_ROOT="${DTOPD_ROOT}/projects/Dual-Track-OPD"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${ENV_PREFIX}/lib/python3.12/site-packages/torch/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="${DTOPD_ROOT}/.cache/flashinfer"
export VERL_DIAG_DIR="${DTOPD_ROOT}/fc-opd-storage/logs"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
# OOM 防护（2026-09-06, step-589 崩溃复盘）: 8B fp32 主权重 + 动态 bsz 峰值
#   ~111G allocated / ~137G reserved, logprobs_from_logits 的温度除法要再
#   alloc ~25G 时撞碎片墙。expandable_segments 消除碎片化保留段, 让 cuda
#   allocator 能复用 reserved-but-unallocated 的 ~35G; verl 官方 fully_async
#   脚本同款设置。与基线 70G 峰值相比 tailsft 高出的 ~40G 来自 torch.compile
#   编译产物 + 温度归一 eager 路径, 属该后端已知行为, 不影响数值。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RAY_LOG_TO_STDERR=1 RAY_DEDUP_LOGS=0
export VLLM_LOGGING_STREAM=ext://sys.stderr
ulimit -c 0

source "${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
if [[ "${SFT_RL_SKIP_RAY_STOP:-0}" != "1" ]]; then
    ray stop --force 2>/dev/null || true
    sleep 2
fi

SFT_RL_GPUS="${SFT_RL_GPUS:-0,1,2,3}"
SFT_RL_MODEL="${SFT_RL_MODEL:-${DTOPD_ROOT}/models/Qwen3-VL-8B-Instruct}"
SFT_RL_DATA="${SFT_RL_DATA:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft_tailsft}"
SFT_RL_VAL="${SFT_RL_VAL:-}"
export SFT_RL_DATA  # the init_ce guard heredoc reads os.environ["SFT_RL_DATA"]
export CUDA_VISIBLE_DEVICES="${SFT_RL_GPUS}"
N_GPUS="$(echo "${SFT_RL_GPUS}" | tr ',' '\n' | wc -l)"
SHARDS=()

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

# init_ce 列必须存在（防误用未标注的普通 pool）
if [[ -d "${SFT_RL_DATA}" ]] && ! "${ENV_PREFIX}/bin/python" - <<'PYEOF'
import sys, pyarrow.parquet as pq, os
data = os.environ["SFT_RL_DATA"]
shard = sorted(f for f in os.listdir(data) if f.startswith("sft_warmup_train__part_"))[0]
schema = pq.read_schema(os.path.join(data, shard))
sys.exit(0 if "init_ce" in schema.names else 1)
PYEOF
then
    echo "FATAL: ${SFT_RL_DATA} train shards lack init_ce column."
    echo "       Run scripts/sft_rl/annotate_tailsft_init_ce.py first."
    exit 1
fi

for p in "${SFT_RL_MODEL}/config.json" "${SFT_RL_VAL}"; do
  [ -f "${p}" ] || { echo "FATAL: missing ${p}"; exit 1; }
done

# TailSFT 固定口径：1 epoch 对齐 mmf122k_1ep；其余超参默认与 run_sft_warmup.sh 相同
SFT_RL_EPOCHS="${SFT_RL_EPOCHS:-1}"
SFT_RL_SAVE_FREQ="${SFT_RL_SAVE_FREQ:-200}"
SFT_RL_MAX_LEN="${SFT_RL_MAX_LEN:-12288}"
SFT_RL_MAX_TOKEN="${SFT_RL_MAX_TOKEN:-98304}"
SFT_RL_BATCH="${SFT_RL_BATCH:-64}"
SFT_RL_LR="${SFT_RL_LR:-5e-5}"
SFT_RL_WARMUP="${SFT_RL_WARMUP:-0.10}"
SFT_RL_WD="${SFT_RL_WD:-0.01}"
SFT_RL_SEED="${SFT_RL_SEED:-1234}"
EXPERIMENT_NAME="${SFT_RL_NAME:-qwen3vl_sft_tailsft_mmf122k_1ep}"
TAILSFT_F="${SFT_RL_TAILSFT_F:-0.5}"
TAILSFT_SCHEDULE="${SFT_RL_TAILSFT_SCHEDULE:-ramp}"
TAILSFT_RAMP="${SFT_RL_TAILSFT_RAMP:-800}"
LOG_FILE="${DTOPD_ROOT}/fc-opd-storage/logs/${EXPERIMENT_NAME}.log"
CKPT_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/${EXPERIMENT_NAME}"
mkdir -p "${CKPT_DIR}"

echo "== TailSFT SFT: model=${SFT_RL_MODEL} GPUs=${CUDA_VISIBLE_DEVICES} (n=${N_GPUS}) =="
echo "== train=${SFT_RL_DATA} (${#SHARDS[@]} shards) val=${SFT_RL_VAL} =="
echo "== tailsft: f=${TAILSFT_F} schedule=${TAILSFT_SCHEDULE} ramp=${TAILSFT_RAMP} =="
echo "== epochs=${SFT_RL_EPOCHS} save_freq=${SFT_RL_SAVE_FREQ} max_len=${SFT_RL_MAX_LEN} token_budget=${SFT_RL_MAX_TOKEN} batch=${SFT_RL_BATCH} lr=${SFT_RL_LR} warmup=${SFT_RL_WARMUP} wd=${SFT_RL_WD} =="
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
  data.custom_cls.path="${REPO_ROOT}/scripts/sft_rl/tailsft_dataset.py" \
  data.custom_cls.name=TailsFTDataset \
  data.tailsft.enabled=True \
  data.tailsft.filter_fraction="${TAILSFT_F}" \
  data.tailsft.schedule="${TAILSFT_SCHEDULE}" \
  data.tailsft.ramp_steps="${TAILSFT_RAMP}" \
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

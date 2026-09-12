#!/usr/bin/env bash
#
# qwen3.5/3.6 家族 cu132 环境 GPU smoke（自包含）
#   student = teacher = Qwen3.5-4B（本地路径）
#   纯 RKL：k1 + use_policy_gradient=True（use_task_rewards=False 由脚本固定）
#   栈：torch 2.13.0+cu132 / vLLM 0.27.1 (cu130 wheel) / flashinfer 0.6.16.post3 /
#       flash-attn 2.8.3 (SM90 源码编译) / verl v0.9.0
#   默认 6 卡（GPU2-7，GPU0-1 被 vaopd-gkd-qwen35 占用）：5 actor + 1 teacher
#   数据：geometry3k text-only smoke 子集（train 84 / val 21）→ 1 epoch = 4 步
#   batch 必须能被 trainer GPU 数整除（默认 n_trainer * 4）
#
# 用法（GPU 节点，无需任何前置环境变量）：
#   bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/run_qwen35_cu132_gpu_smoke.sh
#
# 可选：
#   SMOKE_GPUS=...     默认 2,3,4,5,6,7（8 卡自蒸馏请用 0,1,2,3,4,5,6,7）
#   PREFLIGHT_EAGER=0  开启 CUDA graphs 的 preflight（验证 vLLM #50445 warmup）
#   ATTENTION_IMPL=sdpa 显式覆盖注意力实现（默认 verl 的 flash_attention_2）
#   USE_V1=1           强制 verl V1 trainer（默认 0 = V0，见下方注释）
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 本实例实测 NVLink SHARP (NVLS) multicast 绑定失败（CUDA error 401，疑似
# Fabric Manager/NVSwitch 配置）；NCCL 2.29.7 默认开 NVLS，显式关闭回退到
# NVLink P2P（仍为 18-link 全互联，性能影响小）。如需开启：NCCL_NVLS_ENABLE=1。
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
# 日志可见性：ray/worker 与 vLLM 输出统一到 stderr，随训练 tee 落盘，便于排障
export RAY_LOG_TO_STDERR=1
export RAY_DEDUP_LOGS=0
export VLLM_LOGGING_STREAM=ext://sys.stderr
ulimit -c 0

# shellcheck disable=SC1091
source "${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

# 干净启动：清理可能残留的 ray cluster（上次中断可能留下半死 raylet）
ray stop --force 2>/dev/null || true
sleep 2

SMOKE_GPUS="${SMOKE_GPUS:-2,3,4,5,6,7}"
# verl 0.9.0 的 V1 trainer 无条件强制 transfer_queue（SimpleStorage 8 个存储单元），
# 在本共享 GPU 节点上实测 actor 创建卡死 ~27min 后 worker 死亡（14:43 日志），
# CPU 节点同版本可正常创建 → 平台/负载相关。smoke 先走 V0（与已验证的 qwen35
# cu129 栈一致，不碰 transfer_queue），跑通栈后再单独攻 V1。
USE_V1="${USE_V1:-0}"
if [[ "${USE_V1}" == "1" ]]; then
    USE_V1_FLAG="True"
else
    USE_V1_FLAG="False"
fi
export CUDA_VISIBLE_DEVICES="${SMOKE_GPUS}"
N_GPUS="$(echo "${SMOKE_GPUS}" | tr ',' '\n' | wc -l)"
TEACHER_WORLD_SIZE=1
NGPUS_PER_NODE=$((N_GPUS - TEACHER_WORLD_SIZE))
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((NGPUS_PER_NODE * 4))}"
ROLLOUT_NUM_WORKERS="${ROLLOUT_NUM_WORKERS:-${NGPUS_PER_NODE}}"

STUDENT_MODEL="${DTOPD_ROOT}/models/Qwen3.5-4B"
TEACHER_MODEL="${DTOPD_ROOT}/models/Qwen3.5-4B"
TRAIN_FILE="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/geometry3k_gkd/train_text_only_smoke84.parquet"
VAL_FILE="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/geometry3k_gkd/val_text_only_smoke21.parquet"

for p in \
  "${STUDENT_MODEL}/config.json" \
  "${TEACHER_MODEL}/config.json" \
  "${TRAIN_FILE}" \
  "${VAL_FILE}"; do
  [ -f "${p}" ] || { echo "FATAL: missing ${p}"; exit 1; }
done

echo "== stack: torch 2.13.0+cu132 / vllm 0.27.1 / verl v0.9.0 =="
echo "== student=${STUDENT_MODEL} =="
echo "== teacher=${TEACHER_MODEL} =="
echo "== GPUs=${CUDA_VISIBLE_DEVICES} (n=${N_GPUS}, trainer=${NGPUS_PER_NODE}, teacher=${TEACHER_WORLD_SIZE}) =="
echo "== batch=${TRAIN_BATCH_SIZE} rollout_workers=${ROLLOUT_NUM_WORKERS} =="
echo "== python=$(command -v python3) =="

# ---- GPU 预检：torch CUDA + vllm qwen3_5 引擎（vLLM spawn 需要真实文件作为 __main__）----
echo "== [preflight] torch.cuda + vllm qwen3_5 engine =="
python3 "${DTOPD_ROOT}/projects/Dual-Track-OPD/scripts/qwen35_vllm_preflight.py"

cd "${VERL_DIR}/examples/on_policy_distillation_trainer"

ATTN_ARGS=()
if [[ -n "${ATTENTION_IMPL:-}" ]]; then
    ATTN_ARGS+=("+actor_rollout_ref.model.override_config.attn_implementation=${ATTENTION_IMPL}")
fi

STUDENT_MODEL="${STUDENT_MODEL}" \
TEACHER_MODEL="${TEACHER_MODEL}" \
NNODES=1 \
NGPUS_PER_NODE="${NGPUS_PER_NODE}" \
TEACHER_WORLD_SIZE="${TEACHER_WORLD_SIZE}" \
TEACHER_TP=1 \
TEACHER_EP=1 \
ROLLOUT_TP=1 \
ROLLOUT_NUM_WORKERS="${ROLLOUT_NUM_WORKERS}" \
TRAIN_FILE="${TRAIN_FILE}" \
VAL_FILE="${VAL_FILE}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
PPO_MINI_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
MAX_RESPONSE_LENGTH=1024 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=10 \
TEST_FREQ=5 \
ACTOR_LR=1e-6 \
DISTILLATION_LOSS_MODE=k1 \
USE_POLICY_GRADIENT=True \
USE_TASK_REWARDS=False \
DISTILLATION_TOPK=64 \
PROJECT_NAME=verl_distill_qwen35 \
EXPERIMENT_NAME=qwen3_5_4b_self_gpu_smoke_cu132 \
bash run_qwen3_5_4b_fsdp.sh "${ATTN_ARGS[@]}" trainer.use_v1=${USE_V1_FLAG}

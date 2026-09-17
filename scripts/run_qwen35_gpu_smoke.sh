#!/usr/bin/env bash
#
# qwen3.5/3.6 家族 cu128 环境 GPU smoke（自包含）
#   student = teacher = Qwen3.5-4B（本地路径）
#   纯 RKL：k1 + use_policy_gradient=True（use_task_rewards=False 由脚本固定）
#   资源：8 卡（trainer 7 卡 + teacher 1 卡，verl-cu130 的 teacher_pool 独立计数）
#   数据：geometry3k text-only smoke 子集（train 84 / val 21）→ 1 epoch = 4 步
#   batch=21 必须为 7 的倍数（minimal_bsz = n_gpus = 7）
#   attn=sdpa：qwen35 环境暂无 flash-attn，verl 默认 flash_attention_2 会直接 ImportError
#   vllm：cu129 wheel（wheels.vllm.ai/0.23.0/cu129）或 cu128 源码构建均可；
#   若环境仍残留 cu13 wheel（不推荐），下方会按需把 site-packages/nvidia/cu13/lib 加进
#   LD_LIBRARY_PATH。本节点 nvidia-smi 显示最大 CUDA 12.8：cu13 runtime 不可用（需 R580+），
#   cu129 经 MVC（12.x 最小驱动 >=525）可尝试，实际能否初始化由下方 GPU 预检实测。
#
# 用法（GPU 节点，无需任何前置环境变量）：
#   bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/run_qwen35_gpu_smoke.sh
#
# 可选：SMOKE_GPUS=... bash ...（默认全部 8 卡 0,1,2,3,4,5,6,7）
set -euo pipefail

DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
ENV_PREFIX="${DTOPD_ROOT}/envs/va-opd-qwen35-cu128"
CUDA_HOME="${DTOPD_ROOT}/envs/cuda128-toolchain"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
# 仅当环境中残留 cu13 组件时才需要补这些路径（cu12/cu129 wheel 由 pip/torch loader 自处理）
NV_LIB="${ENV_PREFIX}/lib/python3.12/site-packages/nvidia"
if [ -d "${NV_LIB}/cu13/lib" ]; then
  export LD_LIBRARY_PATH="${NV_LIB}/cu13/lib:${NV_LIB}/cuda_runtime/lib:${NV_LIB}/cublas/lib:${NV_LIB}/cusparse/lib:${NV_LIB}/cusolver/lib:${NV_LIB}/nvrtc/lib:${NV_LIB}/nvjitlink/lib:${NV_LIB}/cudnn/lib:${LD_LIBRARY_PATH:-}"
fi
export FLASHINFER_WORKSPACE_BASE="${DTOPD_ROOT}/.cache/flashinfer"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${SMOKE_GPUS:-0,1,2,3,4,5,6,7}"
ulimit -c 0

# shellcheck disable=SC1091
source "${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

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

echo "== student=${STUDENT_MODEL} =="
echo "== teacher=${TEACHER_MODEL} =="
echo "== train=${TRAIN_FILE} =="
echo "== val=${VAL_FILE} =="
echo "== GPUs=${CUDA_VISIBLE_DEVICES} =="
echo "== python=$(command -v python3) =="

# ---- GPU 预检：torch CUDA + vllm 0.23 qwen3.5 引擎能否在本驱动上拉起 ----
# 注意：必须用真实 .py 文件执行，vLLM spawn 子进程无法从 stdin 重新加载 __main__
echo "== [preflight] torch.cuda + vllm qwen3_5 engine on $(echo ${CUDA_VISIBLE_DEVICES} | cut -d, -f1) =="
python3 "${DTOPD_ROOT}/projects/Dual-Track-OPD/scripts/qwen35_vllm_preflight.py"

cd "${DTOPD_ROOT}/repos/verl-cu130-vllm/examples/on_policy_distillation_trainer"

STUDENT_MODEL="${STUDENT_MODEL}" \
TEACHER_MODEL="${TEACHER_MODEL}" \
NNODES=1 \
NGPUS_PER_NODE=7 \
TEACHER_WORLD_SIZE=1 \
TEACHER_TP=1 \
TEACHER_EP=1 \
ROLLOUT_TP=1 \
ROLLOUT_NUM_WORKERS=7 \
TRAIN_FILE="${TRAIN_FILE}" \
VAL_FILE="${VAL_FILE}" \
TRAIN_BATCH_SIZE=21 \
PPO_MINI_BATCH_SIZE=21 \
MAX_RESPONSE_LENGTH=1024 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=10 \
TEST_FREQ=5 \
ACTOR_LR=1e-6 \
DISTILLATION_LOSS_MODE=k1 \
USE_POLICY_GRADIENT=True \
DISTILLATION_TOPK=64 \
PROJECT_NAME=verl_distill_qwen35 \
EXPERIMENT_NAME=qwen3_5_4b_self_gpu_smoke \
bash run_qwen3_5_4b_fsdp.sh +actor_rollout_ref.model.override_config.attn_implementation=sdpa

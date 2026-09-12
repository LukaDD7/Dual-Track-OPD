#!/usr/bin/env bash
#
# SFT-then-RL 管线 | RL 段（MMFineReason，规则奖励；与 SFT-only / RL-only 三臂对齐）
#   数据: scripts/sft_rl/sample_mmf_rl.py 产出的 mmf_rl_20k 分片
#         （MMF answer 非空 + GT 形态 letter/yesno/numeric/LaTeX；不再剔除 warmup
#          重叠——RL 可在 SFT warmup 的同域任务上继续提升能力，见报告 Q3）
#   RL 数据规模定为 20K train / 500 val（原 3K 多样性不足、巩固不充分；池 95,128
#         行，20K 分层抽样覆盖全部 22 个 source，多样性与原池一致）
#   模型: SFT_RL_MODEL 必须指向 SFT warmup ckpt 的 huggingface 目录
#         （如 <ckpt>/global_step_1086/huggingface，需含 model.safetensors）
#   奖励: scripts/sft_rl/mmf_reward.py（MCQ/是或否/数值/LaTeX+mathruler，10/10 单测通过）
#   RL 超参对齐 arXiv:2604.23747 §4.3 短程方案（SFT warmup 好则 50 步足够）：
#         lr 5e-6 恒定、epsilon_low 0.2 / epsilon_high 0.28 不对称 clip、
#         token 级 loss（loss_agg_mode=token-mean）、去掉 KL 惩罚、
#         去掉 length / std 归一化、max_response_length=12288（压 clip_ratio）
#
# 用法（GPU 节点）:
#   SFT_RL_MODEL=<sft-ckpt>/global_step_*/huggingface \
#     bash scripts/sft_rl/run_grpo_mmf.sh
# 可选 env:
#   SFT_RL_GPUS="4,5,6,7"  默认 4 卡 (0-3 留给 PTD-PO track)
#   TRAIN_BATCH_SIZE=16  PPO_MINI_BATCH_SIZE=8  ROLLOUT_N=8
#     （PPO_MINI_BATCH_SIZE 是 prompt 口径：verl 内部 ×ROLLOUT_N 得序列数，
#       8×8=64 样本 = 论文 RL hyperparameters 的 mini-batch 64；128 样本总批 = 2 个 mini-batch）
#   MAX_RESPONSE_LENGTH=12288  MAX_PROMPT_LENGTH=2048
#   SFT_RL_RL_LR=5e-6          恒定学习率（论文 §4.3 短程方案 5e-6）
#   SFT_RL_TOTAL_STEPS=50      总步数（默认短程 50；设为 0/空则回退 total_epochs）
#   SFT_RL_CLIP_LOW=0.2  SFT_RL_CLIP_HIGH=0.28   epsilon_low/high（不对称 clip）
#   SFT_RL_ACTOR_TOKEN_BUDGET=24576
#   SFT_RL_FUSED_KERNELS=0   (1 可开回 fused PPO 反向；warmup ckpt 的 lm_head 是
#                             F32，verl fused bwd 会 dtype 崩，默认关闭)
#   SFT_RL_NAME=...          固定实验名（续训必设：与原实验名一致，
#                             resume_mode=auto 从 latest_checkpointed_iteration 续）
#   SFT_RL_TRAIN/SFT_RL_VAL  覆盖数据文件（支持 glob 分片）
#   SFT_RL_REWARD=file://.../xxx.py  覆盖规则奖励
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
# 同 run_sft_warmup.sh: 压掉 processor 警告刷屏，防 stdout 管道堵塞冻结训练
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export RAY_LOG_TO_STDERR=1 RAY_DEDUP_LOGS=0
export VLLM_LOGGING_STREAM=ext://sys.stderr
ulimit -c 0

source "${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
ray stop --force 2>/dev/null || true
sleep 2

SFT_RL_GPUS="${SFT_RL_GPUS:-4,5,6,7}"
SFT_RL_MODEL="${SFT_RL_MODEL:-}"
export CUDA_VISIBLE_DEVICES="${SFT_RL_GPUS}"
N_GPUS="$(echo "${SFT_RL_GPUS}" | tr ',' '\n' | wc -l)"

DEFAULT_TRAIN="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_20k/mmf_rl_train__part_*.parquet"
DEFAULT_VAL="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_20k/mmf_rl_val__part_0000.parquet"
TRAIN_FILE="${SFT_RL_TRAIN:-${DEFAULT_TRAIN}}"
VAL_FILE="${SFT_RL_VAL:-${DEFAULT_VAL}}"

[ -n "${SFT_RL_MODEL}" ] || { echo "FATAL: SFT_RL_MODEL required (point to SFT ckpt huggingface dir)"; exit 1; }
[ -f "${SFT_RL_MODEL}/config.json" ] || { echo "FATAL: missing ${SFT_RL_MODEL}/config.json"; exit 1; }
# 权重校验: 单文件 model.safetensors 或分片 model-*.safetensors（如师弟的 617 模型是
# 5 分片 + index.json）都接受。verl/transformers 按 index.json 自动加载分片。
[ -f "${SFT_RL_MODEL}/model.safetensors" ] || compgen -G "${SFT_RL_MODEL}/model-*.safetensors" >/dev/null \
  || { echo "FATAL: no model weights in ${SFT_RL_MODEL}"; exit 1; }
ls ${TRAIN_FILE} >/dev/null 2>&1 || { echo "FATAL: no train shards matching ${TRAIN_FILE}"; exit 1; }
[ -f "${VAL_FILE}" ] || { echo "FATAL: missing val ${VAL_FILE}"; exit 1; }

# verl 数据文件传参：分片 glob -> hydra list（同 run_sft_warmup.sh 的做法）
if [[ "${TRAIN_FILE}" == *'*'* ]]; then
    mapfile -t TRAIN_PARTS < <(ls ${TRAIN_FILE})
    TRAIN_ARG="data.train_files=[$(IFS=,; echo "${TRAIN_PARTS[*]}")]"
else
    TRAIN_ARG="data.train_files=${TRAIN_FILE}"
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12288}"
ACTOR_TOKEN_BUDGET="${SFT_RL_ACTOR_TOKEN_BUDGET:-24576}"
if [[ "${SFT_RL_FUSED_KERNELS:-0}" == "1" ]]; then FUSED_KERNELS="True"; else FUSED_KERNELS="False"; fi
RL_LR="${SFT_RL_RL_LR:-5e-6}"
CLIP_LOW="${SFT_RL_CLIP_LOW:-0.2}"
CLIP_HIGH="${SFT_RL_CLIP_HIGH:-0.28}"
SFT_RL_TOTAL_STEPS="${SFT_RL_TOTAL_STEPS:-50}"
if [[ "${SFT_RL_TOTAL_STEPS}" != "0" ]]; then
    TOTAL_TRAINING_STEPS="${SFT_RL_TOTAL_STEPS}"
else
    TOTAL_TRAINING_STEPS="null"
fi
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
PROJECT_NAME="verl_sftrl"
EXPERIMENT_NAME="${SFT_RL_NAME:-qwen3vl_grpo_mmf_overnight_$(date +%Y%m%d_%H%M)}"
LOG_FILE="${DTOPD_ROOT}/fc-opd-storage/logs/${EXPERIMENT_NAME}.log"
CKPT_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/${EXPERIMENT_NAME}"
REWARD_PATH="${SFT_RL_REWARD:-file://${DTOPD_ROOT}/projects/Dual-Track-OPD/scripts/sft_rl/mmf_reward.py}"
MAX_SAMPLES_ARG=""
if [[ -n "${SFT_RL_MAX_SAMPLES:-}" ]]; then
    MAX_SAMPLES_ARG="data.train_max_samples=${SFT_RL_MAX_SAMPLES}"
fi

echo "== GRPO MMF overnight: model=${SFT_RL_MODEL} GPUs=${CUDA_VISIBLE_DEVICES} (n=${N_GPUS}) =="
echo "== batch=${TRAIN_BATCH_SIZE} mini=${PPO_MINI_BATCH_SIZE} rollout_n=${ROLLOUT_N} epochs=${TOTAL_EPOCHS} steps=${TOTAL_TRAINING_STEPS} =="
echo "== prompt_max=${MAX_PROMPT_LENGTH} response_max=${MAX_RESPONSE_LENGTH} actor_token_budget=${ACTOR_TOKEN_BUDGET} =="
echo "== rl_lr=${RL_LR} clip=[${CLIP_LOW},${CLIP_HIGH}] loss_agg=token-mean std_norm=False kl=False =="
echo "== train=${TRAIN_FILE} =="
echo "== val=${VAL_FILE} =="
echo "== reward=${REWARD_PATH} =="
echo "== fused_kernels=${FUSED_KERNELS} (warmup ckpt lm_head=F32, 必须 False) =="

cd "${VERL_DIR}/examples/grpo_trainer"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=False \
  ${TRAIN_ARG} \
  data.val_files="${VAL_FILE}" \
  ${MAX_SAMPLES_ARG:+${MAX_SAMPLES_ARG}} \
  data.image_key=images \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${SFT_RL_MODEL}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr="${RL_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_TOKEN_BUDGET}" \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_HIGH}" \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ACTOR_TOKEN_BUDGET}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${ACTOR_TOKEN_BUDGET}" \
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
  trainer.resume_mode=auto \
  reward.custom_reward_function.path="${REWARD_PATH}" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.use_fused_kernels="${FUSED_KERNELS}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.actor.checkpoint.save_contents="[model,optimizer,extra,hf_model]" \
  2>&1 | tee "${LOG_FILE}"

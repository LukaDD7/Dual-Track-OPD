#!/usr/bin/env bash
#
# SFT-then-RL | RL 段（MMF TailSFT 臂, 论文对齐 GSPO）
#   对齐 MMFineReason 论文（arXiv:2601.21821 Table 9 + §B.1）的 RL 配置:
#     算法 GSPO[61] / AdamW lr 1e-6 恒定 / wd 0.1 / 300 步 / warmup 10 /
#     batch 256（序列口径: 16 prompt × G16 rollouts）/ prompt 8192 /
#     output 16384 / rollout T=1.0 / G=16 / εlow 3e-4 εhigh 4e-4。
#   论文 ε 即 GSPO 的 clip 窗口（clip_ratio_low/high），不是独立 KL 项
#     （官方 verl examples/gspo_trainer 同款: use_kl_loss=False +
#       clip_ratio_c=10.0 + loss_agg_mode=seq-mean-token-mean）。
#
# 与 run_grpo_mmf.sh（grpo50 臂, 对齐 arXiv:2604.23747）的关键差异:
#     - policy_loss.loss_mode=gspo（序列级重要性比率, 官方实现原样启用）
#     - clip 3e-4/4e-4（原 0.2/0.28）+ clip_ratio_c=10.0 双 clip 保护
#     - loss_agg_mode=seq-mean-token-mean（原 token-mean）
#     - lr 1e-6（原 5e-6）/ wd 0.1（原默认 0.01）/ warmup 10 步（原 0）
#     - G=16（原 8）/ 300 步（原 50）/ prompt 8192 response 16384（原 2048/12288）
#     - norm_adv_by_std_in_grpo 回默认 True（官方 GSPO 示例口径; 原脚本按
#       2604.23747 关闭, 本臂跟随 GSPO 官方实现）
#   本脚本不动 run_grpo_mmf.sh, 产物全部走新命名 qwen3vl_gspo_mmf_tailsft_*。
#
# 已知偏差（论文口径 vs 本实验, 记录进 manuscript）:
#     - RL 数据: mmf_rl_20k（20K 分层抽样, 无 pass-rate 列）vs 论文 40K
#       难度过滤（Qwen3-VL-4B-Thinking ×4 全错才保留）—— pass-rate 不可复现
#     - batch 256 按序列口径落地（16 prompt × 16）; 论文 4 卡 H200 vs 论文
#       未披露的更大规模集群, 步时长按 4 卡折算
#     - SFT 起点: tailsft 1ep/122K（试水）vs 论文 3ep/1.8M MFR-8B-SFT
#
# 用法（GPU 节点, 与 MMF 评测并行时各占不同卡）:
#   SFT_RL_MODEL=<tailsft-ckpt>/global_step_1774/huggingface \
#     bash scripts/sft_rl/run_gspo_mmf_tailsft.sh
# 可选 env:
#   SFT_RL_GPUS="0,1,2,3"      默认 4 卡（新结点 0-3 留给本任务, 4-6 给评测）
#   TRAIN_BATCH_SIZE=16  PPO_MINI_BATCH_SIZE=8  ROLLOUT_N=16
#     （TRAIN/PPO_MINI 是 prompt 口径; verl 内部 ×ROLLOUT_N 得序列数,
#       16×16=256 序列/步 = 论文 batch 256）
#   MAX_PROMPT_LENGTH=8192  MAX_RESPONSE_LENGTH=16384
#   SFT_RL_RL_LR=1e-6   SFT_RL_WD=0.1   SFT_RL_WARMUP_STEPS=10
#   SFT_RL_TOTAL_STEPS=300
#   SFT_RL_CLIP_LOW=3e-4  SFT_RL_CLIP_HIGH=4e-4  SFT_RL_CLIP_C=10.0
#   SFT_RL_ACTOR_TOKEN_BUDGET=24576
#   SFT_RL_SAVE_FREQ=75  SFT_RL_TEST_FREQ=50  SFT_RL_MAX_CKPT_KEEP=2
#     （ckpt 含 optimizer ≈50G/个, 磁盘紧张, 默认只留最近 2 个）
#   SFT_RL_ACTOR_OFFLOAD=1 / SFT_RL_OPTIMIZER_OFFLOAD=1  OOM 时再开
#   SFT_RL_FUSED_KERNELS=0  （tailsft ckpt lm_head F32, 必须 0, 同 grpo 脚本）
#   SFT_RL_SKIP_RAY_STOP=1   同结点有其他 Ray 任务时保留集群
#   SFT_RL_NAME=...  固定实验名（续训必设: 与原实验名一致, resume_mode=auto）
#   SFT_RL_TRAIN/SFT_RL_VAL/SFT_RL_REWARD  同 run_grpo_mmf.sh
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
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export RAY_LOG_TO_STDERR=1 RAY_DEDUP_LOGS=0
export VLLM_LOGGING_STREAM=ext://sys.stderr
ulimit -c 0

source "${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
# 评测（judge vllm serve + VLMEvalKit run.py）不用 Ray, 常态 ray stop 不伤它们;
# SKIP=1 仅供同结点确有其他 Ray 任务时使用。
if [[ "${SFT_RL_SKIP_RAY_STOP:-0}" != "1" ]]; then
    ray stop --force 2>/dev/null || true
    sleep 2
fi

SFT_RL_GPUS="${SFT_RL_GPUS:-0,1,2,3}"
SFT_RL_MODEL="${SFT_RL_MODEL:-}"
export CUDA_VISIBLE_DEVICES="${SFT_RL_GPUS}"
N_GPUS="$(echo "${SFT_RL_GPUS}" | tr ',' '\n' | wc -l)"

DEFAULT_TRAIN="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_20k/mmf_rl_train__part_*.parquet"
DEFAULT_VAL="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_20k/mmf_rl_val__part_0000.parquet"
TRAIN_FILE="${SFT_RL_TRAIN:-${DEFAULT_TRAIN}}"
VAL_FILE="${SFT_RL_VAL:-${DEFAULT_VAL}}"

[ -n "${SFT_RL_MODEL}" ] || { echo "FATAL: SFT_RL_MODEL required (point to tailsft ckpt huggingface dir)"; exit 1; }
[ -f "${SFT_RL_MODEL}/config.json" ] || { echo "FATAL: missing ${SFT_RL_MODEL}/config.json"; exit 1; }
[ -f "${SFT_RL_MODEL}/model.safetensors" ] || compgen -G "${SFT_RL_MODEL}/model-*.safetensors" >/dev/null \
  || { echo "FATAL: no model weights in ${SFT_RL_MODEL}"; exit 1; }
ls ${TRAIN_FILE} >/dev/null 2>&1 || { echo "FATAL: no train shards matching ${TRAIN_FILE}"; exit 1; }
[ -f "${VAL_FILE}" ] || { echo "FATAL: missing val ${VAL_FILE}"; exit 1; }

if [[ "${TRAIN_FILE}" == *'*'* ]]; then
    mapfile -t TRAIN_PARTS < <(ls ${TRAIN_FILE})
    TRAIN_ARG="data.train_files=[$(IFS=,; echo "${TRAIN_PARTS[*]}")]"
else
    TRAIN_ARG="data.train_files=${TRAIN_FILE}"
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
ACTOR_TOKEN_BUDGET="${SFT_RL_ACTOR_TOKEN_BUDGET:-24576}"
if [[ "${SFT_RL_FUSED_KERNELS:-0}" == "1" ]]; then FUSED_KERNELS="True"; else FUSED_KERNELS="False"; fi
RL_LR="${SFT_RL_RL_LR:-1e-6}"
RL_WD="${SFT_RL_WD:-0.1}"
WARMUP_STEPS="${SFT_RL_WARMUP_STEPS:-10}"
CLIP_LOW="${SFT_RL_CLIP_LOW:-3e-4}"
CLIP_HIGH="${SFT_RL_CLIP_HIGH:-4e-4}"
CLIP_C="${SFT_RL_CLIP_C:-10.0}"
SFT_RL_TOTAL_STEPS="${SFT_RL_TOTAL_STEPS:-300}"
if [[ "${SFT_RL_TOTAL_STEPS}" != "0" ]]; then
    TOTAL_TRAINING_STEPS="${SFT_RL_TOTAL_STEPS}"
else
    TOTAL_TRAINING_STEPS="null"
fi
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SFT_RL_SAVE_FREQ:-75}"
TEST_FREQ="${SFT_RL_TEST_FREQ:-50}"
MAX_CKPT_KEEP="${SFT_RL_MAX_CKPT_KEEP:-2}"
if [[ "${SFT_RL_ACTOR_OFFLOAD:-0}" == "1" ]]; then ACTOR_OFFLOAD="True"; else ACTOR_OFFLOAD="False"; fi
if [[ "${SFT_RL_OPTIMIZER_OFFLOAD:-0}" == "1" ]]; then OPTIMIZER_OFFLOAD="True"; else OPTIMIZER_OFFLOAD="False"; fi
PROJECT_NAME="verl_sftrl"
EXPERIMENT_NAME="${SFT_RL_NAME:-qwen3vl_gspo_mmf_tailsft_lr1e6_steps300}"
LOG_FILE="${DTOPD_ROOT}/fc-opd-storage/logs/${EXPERIMENT_NAME}.log"
CKPT_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/${EXPERIMENT_NAME}"
REWARD_PATH="${SFT_RL_REWARD:-file://${DTOPD_ROOT}/projects/Dual-Track-OPD/scripts/sft_rl/mmf_reward.py}"
MAX_SAMPLES_ARG=""
if [[ -n "${SFT_RL_MAX_SAMPLES:-}" ]]; then
    MAX_SAMPLES_ARG="data.train_max_samples=${SFT_RL_MAX_SAMPLES}"
fi

echo "== GSPO MMF tailsft (paper-aligned): model=${SFT_RL_MODEL} GPUs=${CUDA_VISIBLE_DEVICES} (n=${N_GPUS}) =="
echo "== batch=${TRAIN_BATCH_SIZE}x${ROLLOUT_N}=$(( TRAIN_BATCH_SIZE * ROLLOUT_N ))seqs mini=${PPO_MINI_BATCH_SIZE} steps=${TOTAL_TRAINING_STEPS} warmup=${WARMUP_STEPS} =="
echo "== prompt_max=${MAX_PROMPT_LENGTH} response_max=${MAX_RESPONSE_LENGTH} actor_token_budget=${ACTOR_TOKEN_BUDGET} =="
echo "== gspo: lr=${RL_LR} wd=${RL_WD} sched=constant clip=[${CLIP_LOW},${CLIP_HIGH}] c=${CLIP_C} kl=False temp=1.0(default) =="
echo "== train=${TRAIN_FILE} =="
echo "== val=${VAL_FILE} =="
echo "== reward=${REWARD_PATH} =="
echo "== fused_kernels=${FUSED_KERNELS} (lm_head=F32, 必须 False) save_freq=${SAVE_FREQ} test_freq=${TEST_FREQ} keep=${MAX_CKPT_KEEP} =="
echo "== ckpt=${CKPT_DIR} log=${LOG_FILE} =="

cd "${VERL_DIR}/examples/grpo_trainer"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
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
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_HIGH}" \
  actor_rollout_ref.actor.clip_ratio_c="${CLIP_C}" \
  actor_rollout_ref.actor.optim.lr="${RL_LR}" \
  actor_rollout_ref.actor.optim.weight_decay="${RL_WD}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${WARMUP_STEPS}" \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_TOKEN_BUDGET}" \
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
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.max_actor_ckpt_to_keep="${MAX_CKPT_KEEP}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.use_v1=False \
  trainer.resume_mode=auto \
  reward.custom_reward_function.path="${REWARD_PATH}" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.use_fused_kernels="${FUSED_KERNELS}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.actor.checkpoint.save_contents="[model,optimizer,extra,hf_model]" \
  2>&1 | tee "${LOG_FILE}"

#!/usr/bin/env bash
#
# SFT-then-RL + PTD-PO (arXiv:2606.07000) — GRPO with privileged tutoring distillation.
#   数据: scripts/sft_rl/build_mmf_hints.py 产出的 hint 训练集
#         （默认 fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_3k_hint/，val 用原 mmf_rl_3k val）
#   模型: SFT_RL_MODEL 必须指向 SFT warmup ckpt 的 huggingface 目录（同 run_grpo_mmf.sh）
#   奖励: mmf_reward.py（不变）
#
# PTD 机制: 失败轨迹上, 冻结 ref 模型以 hint 增强上下文 (prompt_with_hint) 生成教师
# top-K 分布, 学生用 Top-K JSD (K=100, tail 补偿) 对齐; 策略仍在 question-only 下 GRPO。
# 实现复用 verl 内置 top-K 蒸馏管线: algorithm.ptd.* + loss_mode=jsd_topk。
#
# 用法（GPU 节点, 先跑 hint 生成）:
#   SFT_RL_MODEL=<sft-ckpt>/global_step_*/huggingface \
#     bash scripts/sft_rl/run_grpo_mmf_ptd.sh
# 可选 env:
#   PTD_COEF=5e-2 (8B 论文值; 4B 用 5e-1)   PTD_TOP_K=100
#   PTD_THRESHOLD=1.0   PTD_KL_DIRECTION=jsd_kl   PTD_ALL_TRAJECTORIES=0
#   MAX_RESPONSE_LENGTH=12288 (默认; 与 run_grpo_mmf.sh 对齐。8192 时
#     clip_ratio=0.42, 被截断的失败轨迹会把 PTD 的对齐目标变成"不完整前缀",
#     对 PTD 的伤害比纯 GRPO 更大)
#   其余同 run_grpo_mmf.sh (SFT_RL_GPUS/TRAIN_BATCH_SIZE/ROLLOUT_N/MAX_RESPONSE_LENGTH/...)
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

SFT_RL_GPUS="${SFT_RL_GPUS:-0,1,2,3,4,5,6,7}"
SFT_RL_MODEL="${SFT_RL_MODEL:-}"
export CUDA_VISIBLE_DEVICES="${SFT_RL_GPUS}"
N_GPUS="$(echo "${SFT_RL_GPUS}" | tr ',' '\n' | wc -l)"

DEFAULT_TRAIN="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/hint_pools/mmf_hint/mmf_hint__part_*.parquet"
DEFAULT_VAL="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_3k/mmf_rl_val__part_0000.parquet"
TRAIN_FILE="${SFT_RL_TRAIN:-${DEFAULT_TRAIN}}"
VAL_FILE="${SFT_RL_VAL:-${DEFAULT_VAL}}"

[ -n "${SFT_RL_MODEL}" ] || { echo "FATAL: SFT_RL_MODEL required (point to SFT ckpt huggingface dir)"; exit 1; }
[ -f "${SFT_RL_MODEL}/config.json" ] || { echo "FATAL: missing ${SFT_RL_MODEL}/config.json"; exit 1; }
[ -f "${SFT_RL_MODEL}/model.safetensors" ] || { echo "FATAL: missing ${SFT_RL_MODEL}/model.safetensors"; exit 1; }
ls ${TRAIN_FILE} >/dev/null 2>&1 || { echo "FATAL: no hint shards matching ${TRAIN_FILE} (run build_mmf_hints.py first)"; exit 1; }
[ -f "${VAL_FILE}" ] || { echo "FATAL: missing val ${VAL_FILE}"; exit 1; }

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
# PTD requires the eager logits path (fused kernels do not emit the top-K aux outputs)
SFT_RL_FUSED_KERNELS="${SFT_RL_FUSED_KERNELS:-0}"
if [[ "${SFT_RL_FUSED_KERNELS}" == "1" ]]; then FUSED_KERNELS="True"; else FUSED_KERNELS="False"; fi
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-25}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-10}"
PROJECT_NAME="verl_sftrl"
EXPERIMENT_NAME="${SFT_RL_NAME:-qwen3vl_grpo_mmf_ptd_$(date +%Y%m%d_%H%M)}"
LOG_FILE="${DTOPD_ROOT}/fc-opd-storage/logs/${EXPERIMENT_NAME}.log"
CKPT_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/${EXPERIMENT_NAME}"
REWARD_PATH="${SFT_RL_REWARD:-file://${DTOPD_ROOT}/projects/Dual-Track-OPD/scripts/sft_rl/mmf_reward.py}"
MAX_SAMPLES_ARG=""
if [[ -n "${SFT_RL_MAX_SAMPLES:-}" ]]; then
    MAX_SAMPLES_ARG="data.train_max_samples=${SFT_RL_MAX_SAMPLES}"
fi

PTD_ENABLE="${PTD_ENABLE:-true}"
PTD_COEF="${PTD_COEF:-5e-2}"
PTD_TOP_K="${PTD_TOP_K:-100}"
PTD_THRESHOLD="${PTD_THRESHOLD:-1.0}"
PTD_KL_DIRECTION="${PTD_KL_DIRECTION:-jsd_kl}"
PTD_ALL_TRAJECTORIES="${PTD_ALL_TRAJECTORIES:-false}"
PTD_USE_REF_TEACHER="${PTD_USE_REF_TEACHER:-true}"

echo "== GRPO+PTD MMF: model=${SFT_RL_MODEL} GPUs=${CUDA_VISIBLE_DEVICES} (n=${N_GPUS}) =="
echo "== batch=${TRAIN_BATCH_SIZE} mini=${PPO_MINI_BATCH_SIZE} rollout_n=${ROLLOUT_N} epochs=${TOTAL_EPOCHS} =="
echo "== ptd: enable=${PTD_ENABLE} coef=${PTD_COEF} top_k=${PTD_TOP_K} threshold=${PTD_THRESHOLD} kl=${PTD_KL_DIRECTION} =="
echo "== train=${TRAIN_FILE} =="
echo "== val=${VAL_FILE} =="
echo "== fused_kernels=${FUSED_KERNELS} (PTD 必须 False) =="

cd "${VERL_DIR}/examples/grpo_trainer"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.ptd.enable="${PTD_ENABLE}" \
  algorithm.ptd.coef="${PTD_COEF}" \
  algorithm.ptd.top_k="${PTD_TOP_K}" \
  algorithm.ptd.threshold="${PTD_THRESHOLD}" \
  algorithm.ptd.kl_direction="${PTD_KL_DIRECTION}" \
  algorithm.ptd.all_trajectories="${PTD_ALL_TRAJECTORIES}" \
  algorithm.ptd.use_ref_teacher="${PTD_USE_REF_TEACHER}" \
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
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_TOKEN_BUDGET}" \
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
  trainer.save_freq="${TRAINER_SAVE_FREQ}" \
  trainer.test_freq="${TRAINER_TEST_FREQ}" \
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
  2>&1 | tee "${LOG_FILE}"

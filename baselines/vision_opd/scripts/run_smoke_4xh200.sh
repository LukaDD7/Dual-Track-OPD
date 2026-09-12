#!/usr/bin/env bash
set -euo pipefail

# 4xH200 smoke test for Vision-OPD.
# Run from anywhere after activating vision-opd-cu128.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
UPSTREAM="${REPO_ROOT}/third_party/Vision-OPD"

cd "${UPSTREAM}"

export PYTHONPATH="${UPSTREAM}:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
unset VLLM_ATTENTION_BACKEND

# Required paths. Override from shell if needed.
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3.5-4B}"
TRAIN_FILE="${TRAIN_FILE:-${UPSTREAM}/data/train.parquet}"

# Smoke output.
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/baselines/vision_opd/logs/smoke_4xh200}"
mkdir -p "${OUT_DIR}/ckpt" "${OUT_DIR}/rollouts"

echo "[smoke] repo root: ${REPO_ROOT}"
echo "[smoke] upstream: ${UPSTREAM}"
echo "[smoke] model: ${MODEL_PATH}"
echo "[smoke] train file: ${TRAIN_FILE}"
echo "[smoke] out dir: ${OUT_DIR}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[smoke] ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[smoke] ERROR: TRAIN_FILE does not exist: ${TRAIN_FILE}" >&2
  exit 1
fi

python3 - <<'PY'
import torch
print("[smoke] torch:", torch.__version__, "cuda:", torch.version.cuda)
print("[smoke] cuda available:", torch.cuda.is_available())
print("[smoke] cuda device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"[smoke] gpu {i}: {torch.cuda.get_device_name(i)}")
PY

python3 -m verl.trainer.main_ppo --config-name vopd \
  data.train_files="[\"${TRAIN_FILE}\"]" \
  data.val_files="[]" \
  data.train_batch_size=4 \
  data.max_prompt_length=8192 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.shuffle=False \
  data.trust_remote_code=True \
  data.return_multi_modal_inputs=True \
  data.image_key=images \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_model_len=8320 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8320 \
  actor_rollout_ref.rollout.response_length=128 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8320 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.clip_ratio_high=0.3 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
  actor_rollout_ref.actor.calculate_entropy=False \
  actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
  actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
  actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
  actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
  actor_rollout_ref.actor.self_distillation.teacher_model_source=legacy \
  actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
  actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05 \
  actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images \
  actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
  actor_rollout_ref.actor.self_distillation.alpha=0.5 \
  actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  reward_model.enable=False \
  reward_model.use_reward_loop=False \
  critic.model.path="${MODEL_PATH}" \
  custom_reward_function.path=null \
  trainer.project_name=Vision-OPD-Smoke \
  trainer.group_name=smoke_4xh200 \
  trainer.experiment_name=smoke_4xh200 \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=False \
  trainer.default_local_dir="${OUT_DIR}/ckpt" \
  trainer.rollout_data_dir="${OUT_DIR}/rollouts"
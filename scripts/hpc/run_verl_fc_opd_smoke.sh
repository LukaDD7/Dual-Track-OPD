#!/usr/bin/env bash
# FC-OPD verl one-step smoke.
#
# Prerequisites (all on NFS, visible to GPU node):
#   1. Patches applied to third_party/verl (already done via NFS)
#   2. Teacher running on GPU 0: curl -s http://127.0.0.1:18080/health
#   3. Parquet at fc-opd-storage/outputs/fc_opd/verl_smoke/train.parquet
#   4. Student model at models/Qwen3-VL-4B-Instruct
#
# Usage (from repo root, on GPU node with 2 free GPUs):
#   CUDA_VISIBLE_DEVICES=1,2 bash scripts/hpc/run_verl_fc_opd_smoke.sh
#
# Post-run:  ray stop -f

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL_PATH="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-4B-Instruct"
PARQUET="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/verl_smoke/train.parquet"

echo "=== Prerequisites ==="
echo "Model: ${MODEL_PATH}"
echo "Parquet: ${PARQUET}"
echo "Teacher: $(curl -s http://127.0.0.1:18080/health || echo 'NOT REACHABLE')"
echo ""

# Verify patches (check for the injected code, not config values)
if ! grep -q "compute_verl_sparse_topk_kd" third_party/verl/verl/workers/actor/dp_actor.py; then
    echo "FATAL: actor patch not applied"
    exit 1
fi
if ! grep -q "fc_opd_hook_fqn" third_party/verl/verl/trainer/ppo/ray_trainer.py; then
    echo "FATAL: trainer patch not applied"
    exit 1
fi
echo "Patches: OK"

# Start Ray if not running
if ! ray status &>/dev/null 2>&1; then
    echo "Starting Ray..."
    ray start --head --num-gpus=2 --disable-usage-stats
fi
echo "Ray: OK"
echo ""

# vLLM v1 + Triton need a C compiler.  If CC is unset or points to a
# non-existent binary (e.g. a stale conda env path), fall back to gcc on PATH.
if [ -z "${CC:-}" ] || ! command -v "${CC}" >/dev/null 2>&1; then
    export CC=gcc
fi
echo "CC=${CC}"

# Run one PPO step with FC-OPD.
# We use verl's standard ppo_trainer config and override everything via CLI.
# This smoke is GPU-first: it must instantiate the real student forced scorer.
# CPU fallback is intentionally not used because it cannot validate Qwen vocab
# alignment, real student logits, GPU memory pressure, or the FC-OPD training path.
echo "=== Launching verl FC-OPD smoke ==="

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    "+algorithm.fc_opd.post_rollout_hook=dual_track_opd.fc_opd.verl_post_rollout_hook.fc_opd_post_rollout_hook" \
    "+algorithm.fc_opd.student_scorer_fqn=dual_track_opd.fc_opd.student_scorer.StudentScorer" \
    "+algorithm.fc_opd.student_scorer_kwargs.model_path=${MODEL_PATH}" \
    "+algorithm.fc_opd.student_scorer_kwargs.device=cuda" \
    "+algorithm.fc_opd.student_scorer_kwargs.dtype=bfloat16" \
    "+algorithm.fc_opd.student_scorer_kwargs.top_k=32" \
    "+algorithm.fc_opd.teacher_url=http://127.0.0.1:18080" \
    "+algorithm.fc_opd.conditions=[full,degraded,free,task_visible,task_infer,task_solve]" \
    "+algorithm.fc_opd.loss_coef=0.1" \
    "+algorithm.fc_opd.renormalize_topk=true" \
    "+algorithm.fc_opd.include_tail=true" \
    "data.train_files=${PARQUET}" \
    "data.val_files=${PARQUET}" \
    data.train_batch_size=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=false \
    data.truncation=error \
    data.image_key=images \
    "reward.custom_reward_function.path=file://${REPO_ROOT}/src/dual_track_opd/fc_opd/smoke_reward.py" \
    "reward.custom_reward_function.name=compute_score" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    '+actor_rollout_ref.model.override_config.attn_implementation=sdpa' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    trainer.critic_warmup=0 \
    'trainer.logger=["console"]' \
    trainer.project_name=verl_fc_opd_smoke \
    trainer.experiment_name=fc_opd_one_step \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=false \
    "$@"

echo ""
echo "=== Done ==="
echo "Look for 'actor/fc_opd_loss' in the output above."
echo "Cleanup: ray stop -f"

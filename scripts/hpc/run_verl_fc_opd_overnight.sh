#!/usr/bin/env bash
# FC-OPD overnight multi-step training (reverse KL, 200 steps).
#
# Uses the same smoke infra but runs many steps to validate stability:
#   - Reverse KL (mode-seeking) as default
#   - Checkpoint every 50 steps
#   - Log to file for post-hoc analysis
#
# Usage:
#   bash scripts/hpc/run_verl_fc_opd_overnight.sh [--gpus N] [--steps S]
#
# Default: 4 GPUs, 200 steps (~2 hours for 2-sample batch)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

GPU_COUNT=${GPU_COUNT:-4}
NUM_STEPS=${NUM_STEPS:-200}
TOP_K=32
TEACHER_PORT=18080

# ── parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)  GPU_COUNT="$2"; shift 2 ;;
        --steps) NUM_STEPS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MODEL_PATH="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-4B-Instruct"
TEACHER_MODEL="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct"
PARQUET="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/verl_smoke/train.parquet"
CONDA_ENV="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/envs/fc-opd-verl071-cu128"
REPO_ROOT_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="fc_opd_overnight_$(date +%Y%m%d_%H%M%S)"
TEACHER_LOG="${REPO_ROOT_ABS}/artifacts/fc_opd/teacher_${RUN_ID}.log"
TRAIN_LOG="${REPO_ROOT_ABS}/artifacts/fc_opd/train_${RUN_ID}.log"
VERL_CONFIG_DIR="${REPO_ROOT_ABS}/third_party/verl/verl/trainer/config"
REWARD_FN="file://${REPO_ROOT_ABS}/src/dual_track_opd/fc_opd/smoke_reward.py"
mkdir -p "$(dirname "${TEACHER_LOG}")"

# ── GPU math ────────────────────────────────────────────────────────────────
VERL_GPUS=$(( GPU_COUNT - 1 ))
TRAIN_GPUS=$(( VERL_GPUS - 1 ))
TEACHER_GPU=0
VERL_GPU_LIST=$(seq -s, 1 $(( GPU_COUNT - 1 )))
SAVE_FREQ=25
CHECKPOINT_DIR="${REPO_ROOT_ABS}/checkpoints/verl_fc_opd_overnight/${RUN_ID}"

echo "══════════════════════════════════════════════════════════════"
echo "  FC-OPD Overnight Training"
echo "  Run ID:    ${RUN_ID}"
echo "  Steps:     ${NUM_STEPS}"
echo "  Save freq: ${SAVE_FREQ}"
echo "  Loss mode: reverse (mode-seeking KL)"
echo "  GPU:       teacher=0, train=1..$((TRAIN_GPUS)), scorer=$((GPU_COUNT-1))"
echo "  Train log: ${TRAIN_LOG}"
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── verify ──────────────────────────────────────────────────────────────────
if ! grep -q "compute_verl_sparse_topk_kd" third_party/verl/verl/workers/actor/dp_actor.py; then
    echo "FATAL: actor patch not applied"; exit 1
fi
if [ -z "${CC:-}" ] || ! command -v "${CC}" >/dev/null 2>&1; then
    export CC=/usr/bin/gcc
fi
echo "[OK] CC=${CC}"

# ── cleanup ─────────────────────────────────────────────────────────────────
echo "=== Cleanup ==="
ray stop -f 2>/dev/null || true
EXISTING_TEACHER=$(lsof -ti:${TEACHER_PORT} 2>/dev/null || true)
if [ -n "${EXISTING_TEACHER}" ]; then
    kill -9 ${EXISTING_TEACHER} 2>/dev/null || true; sleep 2
fi
rm -rf /dev/shm/*vllm* /dev/shm/*psm_* 2>/dev/null || true
sleep 2

# ── 1) Teacher ──────────────────────────────────────────────────────────────
echo "=== Teacher (GPU ${TEACHER_GPU}) ==="
CUDA_VISIBLE_DEVICES=${TEACHER_GPU} \
    ${CONDA_ENV}/bin/python -m dual_track_opd.fc_opd.teacher_service \
    --backend transformers --model "${TEACHER_MODEL}" \
    --port "${TEACHER_PORT}" --top-k "${TOP_K}" --dtype bfloat16 --device "cuda:0" \
    > "${TEACHER_LOG}" 2>&1 &
TEACHER_PID=$!
echo -n "  Waiting ."
for i in $(seq 1 180); do
    if curl -s "http://127.0.0.1:${TEACHER_PORT}/health" >/dev/null 2>&1; then echo " OK"; break; fi
    if ! kill -0 ${TEACHER_PID} 2>/dev/null; then echo " DIED"; tail -20 "${TEACHER_LOG}"; exit 1; fi
    echo -n "."; sleep 1
done

# ── 2) Ray ──────────────────────────────────────────────────────────────────
echo "=== Ray (${VERL_GPUS} GPUs) ==="
CUDA_VISIBLE_DEVICES=${VERL_GPU_LIST} ray start --head --num-gpus=${VERL_GPUS} --disable-usage-stats
sleep 3

# ── 3) PPO ──────────────────────────────────────────────────────────────────
echo "=== Training (${NUM_STEPS} steps, reverse KL) ==="
set +e
${CONDA_ENV}/bin/python -m verl.trainer.main_ppo \
    --config-path="${VERL_CONFIG_DIR}" \
    --config-name=ppo_trainer \
    "data.train_files=${PARQUET}" \
    "data.val_files=${PARQUET}" \
    "data.train_batch_size=${TRAIN_GPUS}" \
    "data.max_prompt_length=1024" \
    "data.max_response_length=512" \
    "data.filter_overlong_prompts=false" \
    "data.truncation=error" \
    "data.image_key=images" \
    "data.custom_cls.path=file://${REPO_ROOT_ABS}/src/dual_track_opd/fc_opd/verl_dataset.py" \
    "data.custom_cls.name=FCOPDDataset" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.model.use_remove_padding=false" \
    "actor_rollout_ref.model.use_fused_kernels=false" \
    "actor_rollout_ref.model.enable_gradient_checkpointing=true" \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    "actor_rollout_ref.actor.optim.lr=1e-6" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_GPUS}" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.actor.use_kl_loss=false" \
    "actor_rollout_ref.actor.fsdp_config.param_offload=false" \
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false" \
    "actor_rollout_ref.rollout.name=vllm" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.3" \
    "actor_rollout_ref.rollout.max_model_len=2048" \
    "actor_rollout_ref.rollout.n=1" \
    "actor_rollout_ref.rollout.free_cache_engine=true" \
    "actor_rollout_ref.rollout.enforce_eager=true" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2" \
    "actor_rollout_ref.rollout.agent.num_workers=2" \
    "actor_rollout_ref.ref.fsdp_config.param_offload=true" \
    "reward_model.enable=false" \
    "reward_model.num_workers=null" \
    "reward_model.reward_manager=null" \
    "reward_model.reward_loop_source=null" \
    "reward_model.reward_loop_module_path=null" \
    "reward_model.reward_loop_class_name=null" \
    "reward_model.model.path=null" \
    "custom_reward_function.path=${REWARD_FN}" \
    "custom_reward_function.name=compute_score" \
    "algorithm.adv_estimator=grpo" \
    "algorithm.use_kl_in_reward=false" \
    "+algorithm.fc_opd.post_rollout_hook=dual_track_opd.fc_opd.verl_post_rollout_hook.fc_opd_post_rollout_hook" \
    "+algorithm.fc_opd.student_scorer_fqn=dual_track_opd.fc_opd.student_scorer.StudentScorer" \
    "+algorithm.fc_opd.student_scorer_kwargs.model_path=${MODEL_PATH}" \
    "+algorithm.fc_opd.student_scorer_kwargs.device=cuda" \
    "+algorithm.fc_opd.student_scorer_kwargs.dtype=bfloat16" \
    "+algorithm.fc_opd.student_scorer_kwargs.top_k=${TOP_K}" \
    "+algorithm.fc_opd.teacher_url=http://127.0.0.1:${TEACHER_PORT}" \
    "+algorithm.fc_opd.conditions=[full,degraded,free,task_visible,task_infer,task_solve]" \
    "+algorithm.fc_opd.loss_coef=0.1" \
    "+algorithm.fc_opd.loss_mode=reverse" \
    "+algorithm.fc_opd.renormalize_topk=true" \
    "+algorithm.fc_opd.include_tail=true" \
    "trainer.total_epochs=${NUM_STEPS}" \
    "trainer.n_gpus_per_node=${TRAIN_GPUS}" \
    "trainer.nnodes=1" \
    "trainer.critic_warmup=0" \
    "trainer.logger=['console']" \
    "trainer.project_name=fc_opd_overnight" \
    "trainer.experiment_name=${RUN_ID}" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.test_freq=-1" \
    "trainer.default_local_dir=${CHECKPOINT_DIR}" \
    "trainer.val_before_train=false" \
    2>&1 | tee "${TRAIN_LOG}"
VERL_EXIT=$?

# ── cleanup ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Cleanup ==="
ray stop -f 2>/dev/null || true
kill ${TEACHER_PID} 2>/dev/null || true
sleep 2

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Run:    ${RUN_ID}"
echo "  Steps:  ${NUM_STEPS}"
echo "  Log:    ${TRAIN_LOG}"
echo "  CKPT:   ${CHECKPOINT_DIR}"
echo "  Exit:   ${VERL_EXIT}"
echo "══════════════════════════════════════════════════════════════"
exit ${VERL_EXIT}

#!/usr/bin/env bash
# VA-OPD: Visual-Advantage On-Policy Distillation (arXiv 2605.21924 §3.2-3.3)
# ========================================================================
# Grouped reverse KL distillation, weighted by rollout-level visual advantage.
# Auxiliary loss added to GRPO (NOT advantage modulation).
#
#   VA_t = max(log P_T(y_t|full) - log P_T(y_t|degraded), 0)        (§3.1)
#   w^(k) = K * softmax(z_score(mean_t(VA_t)) / τ)                  (§3.2)
#   L_group = λ·mean(KL_rev, HighVA) + (1-λ)·mean(KL_rev, LowVA)   (§3.3)
#   L_total = L_GRPO + α * Σ_k w^(k) * L_group^(k)                  (auxiliary)
#
# Teacher VA uses EXACT full-vocab gather (no top-K tail approximation).
# GPU layout:  0=Teacher(32B), 1..N-2=verl+vLLM, N-1=StudentScorer(4B)
# Usage:  bash scripts/hpc/run_verl_vgpo.sh [--steps S] [--gpus N] [--bg]

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

GPU_COUNT=4; NUM_STEPS=200; TOP_K=32
TEACHER_PORT=18080; SCORER_PORT=18081
RUN_BG=false; PARQUET_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)  GPU_COUNT="${2:?}"; shift 2 ;;
        --steps) NUM_STEPS="${2:?}"; shift 2 ;;
        --data)  PARQUET_OVERRIDE="${2:?}"; shift 2 ;;
        --bg)    RUN_BG=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if ${RUN_BG}; then
    RELAUNCH_ARGS=()
    for a in "$@"; do [[ "$a" != "--bg" ]] && RELAUNCH_ARGS+=("$a"); done
    [[ -n "${PARQUET_OVERRIDE}" ]] && RELAUNCH_ARGS+=(--data "${PARQUET_OVERRIDE}")
    NOHUP_LOG="${REPO_ROOT}/artifacts/fc_opd/nohup_vgpo_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "${NOHUP_LOG}")"
    echo "Launching VGPO. Monitor: tail -f ${NOHUP_LOG}"
    nohup bash "$0" --gpus "${GPU_COUNT}" --steps "${NUM_STEPS}" "${RELAUNCH_ARGS[@]}" > "${NOHUP_LOG}" 2>&1 &
    disown; echo "BG PID: $!"; exit 0
fi

MODEL_PATH="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-4B-Instruct"
TEACHER_MODEL="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct"
PARQUET="${PARQUET_OVERRIDE:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/geometry3k_vgg/train.parquet}"
CONDA_ENV="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/envs/fc-opd-verl071-cu128"
REPO_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="va_opd_$(date +%Y%m%d_%H%M%S)"
TEACHER_LOG="${REPO_ABS}/artifacts/fc_opd/teacher_${RUN_ID}.log"
SCORER_LOG="${REPO_ABS}/artifacts/fc_opd/student_scorer_${RUN_ID}.log"
TRAIN_LOG="${REPO_ABS}/artifacts/fc_opd/train_${RUN_ID}.log"
VERL_CFG="${REPO_ABS}/third_party/verl/verl/trainer/config"
REWARD_FN="file://${REPO_ABS}/src/dual_track_opd/fc_opd/smoke_reward.py"
mkdir -p "$(dirname "${TEACHER_LOG}")"

TEACHER_GPU=0; SCORER_GPU=$(( GPU_COUNT - 1 ))
TRAIN_GPUS=$(( GPU_COUNT - 2 ))
VERL_GPU_LIST=$(seq -s, 1 $(( GPU_COUNT - 2 )))
TRAIN_BATCH_SIZE=2; ROLLOUT_N=4
CHECKPOINT_DIR="${REPO_ABS}/checkpoints/verl_vgpo/${RUN_ID}"

echo "══════════════════════════════════════════════════════════════"
echo "  VGPO — ACL 2026 (arXiv 2604.09349)"
echo "  Run: ${RUN_ID}  Steps: ${NUM_STEPS}"
echo "  Loss: standard GRPO PPO + VGPO advantage modulation"
echo "  GPU: teacher=0 train=${VERL_GPU_LIST} scorer=${SCORER_GPU}"
echo "  Data: ${PARQUET}"
echo "══════════════════════════════════════════════════════════════"

# verify
grep -q "vgpo" "${REPO_ABS}/third_party/verl/verl/workers/actor/dp_actor.py" || { echo "FATAL: VGPO patch missing"; exit 1; }
[ -z "${CC:-}" ] && export CC=/usr/bin/gcc

# cleanup
ray stop -f 2>/dev/null || true
kill $(lsof -ti:${TEACHER_PORT}) 2>/dev/null || true; sleep 2
kill $(lsof -ti:${SCORER_PORT}) 2>/dev/null || true; sleep 2
rm -rf /dev/shm/*vllm* /dev/shm/*psm_* 2>/dev/null || true; sleep 2

# Teacher
CUDA_VISIBLE_DEVICES=${TEACHER_GPU} ${CONDA_ENV}/bin/python \
    -m dual_track_opd.fc_opd.teacher_service \
    --backend transformers --model "${TEACHER_MODEL}" \
    --port "${TEACHER_PORT}" --top-k "${TOP_K}" --dtype bfloat16 --device "cuda:0" \
    > "${TEACHER_LOG}" 2>&1 &
TEACHER_PID=$!
for i in $(seq 1 300); do
    curl -s "http://127.0.0.1:${TEACHER_PORT}/health" >/dev/null 2>&1 && break
    kill -0 ${TEACHER_PID} 2>/dev/null || { echo "TEACHER DIED"; tail -20 "${TEACHER_LOG}"; exit 1; }
    sleep 1
done

# StudentScorer
CUDA_VISIBLE_DEVICES=${SCORER_GPU} ${CONDA_ENV}/bin/python \
    -m dual_track_opd.fc_opd.student_scorer_service \
    --model "${MODEL_PATH}" --port "${SCORER_PORT}" --top-k "${TOP_K}" --dtype bfloat16 --device "cuda:0" \
    > "${SCORER_LOG}" 2>&1 &
SCORER_PID=$!
for i in $(seq 1 300); do
    curl -s "http://127.0.0.1:${SCORER_PORT}/health" >/dev/null 2>&1 && break
    kill -0 ${SCORER_PID} 2>/dev/null || { echo "SCORER DIED"; tail -20 "${SCORER_LOG}"; exit 1; }
    sleep 1
done

# Ray + Training
CUDA_VISIBLE_DEVICES=${VERL_GPU_LIST} ray start --head --num-gpus=${TRAIN_GPUS} --disable-usage-stats
sleep 3

set +e
CUDA_VISIBLE_DEVICES=${VERL_GPU_LIST} ${CONDA_ENV}/bin/python -m verl.trainer.main_ppo \
    --config-path="${VERL_CFG}" --config-name=ppo_trainer \
    "data.train_files=${PARQUET}" "data.val_files=${PARQUET}" \
    "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "data.max_prompt_length=2048" "data.max_response_length=768" \
    "data.filter_overlong_prompts=false" "data.truncation=error" \
    "data.image_key=images" "data.dataloader_num_workers=8" \
    "data.custom_cls.path=file://${REPO_ABS}/src/dual_track_opd/fc_opd/verl_dataset.py" \
    "data.custom_cls.name=FCOPDDataset" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.model.use_remove_padding=false" \
    "actor_rollout_ref.model.use_fused_kernels=false" \
    "actor_rollout_ref.model.enable_gradient_checkpointing=true" \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    "actor_rollout_ref.actor.optim.lr=2e-6" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.actor.use_dynamic_bsz=true" \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384" \
    "actor_rollout_ref.actor.use_kl_loss=false" \
    "actor_rollout_ref.actor.fsdp_config.param_offload=true" \
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true" \
    "actor_rollout_ref.rollout.name=vllm" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    "actor_rollout_ref.rollout.max_model_len=4096" \
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8" \
    "actor_rollout_ref.rollout.agent.num_workers=8" \
    "actor_rollout_ref.ref.fsdp_config.param_offload=true" \
    "reward_model.enable=false" "reward_model.num_workers=null" \
    "reward_model.reward_manager=null" "reward_model.reward_loop_source=null" \
    "reward_model.reward_loop_module_path=null" "reward_model.reward_loop_class_name=null" \
    "reward_model.model.path=null" \
    "custom_reward_function.path=${REWARD_FN}" \
    "custom_reward_function.name=compute_score" \
    "algorithm.adv_estimator=grpo" "algorithm.use_kl_in_reward=false" \
    "+algorithm.fc_opd.post_rollout_hook=dual_track_opd.fc_opd.verl_post_rollout_hook.fc_opd_post_rollout_hook" \
    "+algorithm.fc_opd.student_scorer_fqn=dual_track_opd.fc_opd.student_scorer_client.StudentScorerClient" \
    "+algorithm.fc_opd.student_scorer_kwargs.base_url=http://127.0.0.1:${SCORER_PORT}" \
    "+algorithm.fc_opd.student_scorer_kwargs.timeout_seconds=300" \
    "+algorithm.fc_opd.compute_hook_loss=false" \
    "+algorithm.fc_opd.teacher_url=http://127.0.0.1:${TEACHER_PORT}" \
    "+algorithm.fc_opd.conditions=[full,degraded]" \
    "+algorithm.fc_opd.loss_mode=va_opd" \
    "+algorithm.fc_opd.renormalize_topk=true" \
    "+algorithm.fc_opd.include_tail=true" \
    "trainer.total_training_steps=${NUM_STEPS}" \
    "trainer.n_gpus_per_node=${TRAIN_GPUS}" "trainer.nnodes=1" \
    "trainer.critic_warmup=0" "trainer.logger=['console']" \
    "trainer.project_name=va_opd" "trainer.experiment_name=${RUN_ID}" \
    "trainer.save_freq=50" "trainer.test_freq=-1" \
    "trainer.default_local_dir=${CHECKPOINT_DIR}" "trainer.val_before_train=false" \
    2>&1 | tee "${TRAIN_LOG}"
VERL_EXIT=$?

ray stop -f 2>/dev/null || true
kill ${TEACHER_PID} 2>/dev/null || true; kill ${SCORER_PID} 2>/dev/null || true

echo "══════════════════════════════════════════════════════════════"
echo "  Run: ${RUN_ID}  Exit: ${VERL_EXIT}"
echo "  Log: ${TRAIN_LOG}  CKPT: ${CHECKPOINT_DIR}"
echo "══════════════════════════════════════════════════════════════"
exit ${VERL_EXIT}

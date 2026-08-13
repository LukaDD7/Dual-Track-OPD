#!/usr/bin/env bash
# STP-OPD 4-step real-image canary (single arm; handoff Phase 5).
#
# Prereqs (built earlier):
#   - stp_prefixes.json (verified answer-free teacher prefixes, 4 rescue-positive)
#   - canary training parquet (scripts/hpc/build_stp_canary_parquet.py)
#   - verl 0.7.1 with fc_opd x2 + stp_opd_combined + 0003 patches applied
#
# Usage:
#   bash scripts/hpc/run_stp_opd_canary.sh --arm A3 --steps 4 --gpus 2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

ARM="A3"
NUM_STEPS=4
GPU_COUNT=2
TEACHER_PORT=18081
ROLLOUT_N=4
LR=1e-6
GPU_MEM_UTIL=0.7

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm)       ARM="$2"; shift 2 ;;
        --steps)     NUM_STEPS="$2"; shift 2 ;;
        --gpus)      GPU_COUNT="$2"; shift 2 ;;
        --dry-run)   echo "DRY-RUN: arm=${ARM} steps=${NUM_STEPS} gpus=${GPU_COUNT}"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

CLUSTER_ROOT="/inspire/hdd/global_user/mengweicheng-240108120092/lzy"
CONDA_ENV="${CLUSTER_ROOT}/fc-opd-storage/envs/fc-opd-verl071-cu128"
MODEL_PATH="${CLUSTER_ROOT}/models/Qwen3-VL-4B-Instruct"
TEACHER_MODEL="${CLUSTER_ROOT}/models/Qwen3-VL-32B-Instruct"
OUT_ROOT="${CLUSTER_ROOT}/fc-opd-storage/outputs/support_aware_opd/support_transition_canary"
PARQUET="${OUT_ROOT}/train.parquet"
PREFIX_MANIFEST="${CLUSTER_ROOT}/fc-opd-storage/outputs/support_aware_opd/stp_prefixes_20260813/stp_prefixes.json"
REWARD_FN="file://${REPO_ROOT}/src/dual_track_opd/fc_opd/smoke_reward.py"
TEACHER_LOG="${REPO_ROOT}/artifacts/fc_opd/teacher_stp_$(date +%Y%m%d_%H%M%S).log"
VERL_CONFIG_DIR="${REPO_ROOT}/third_party/verl/verl/trainer/config"

[[ -f "${PARQUET}" ]] || { echo "FATAL: canary parquet missing (run build_stp_canary_parquet.py): ${PARQUET}" >&2; exit 1; }
[[ -f "${PREFIX_MANIFEST}" ]] || { echo "FATAL: prefix manifest missing: ${PREFIX_MANIFEST}" >&2; exit 1; }
[[ $(( GPU_COUNT - 1 )) -ge 1 ]] || { echo "ERROR: need at least 2 GPUs (1 teacher + >=1 train)"; exit 1; }

VERL_GPUS=$(( GPU_COUNT - 1 ))
TRAIN_BATCH_SIZE=${VERL_GPUS}
PPO_MINI_BATCH_SIZE=$(( TRAIN_BATCH_SIZE * ROLLOUT_N ))
VERL_GPU_LIST=$(seq -s, 1 $(( GPU_COUNT - 1 )))

echo "=== STP-OPD canary (arm ${ARM}, ${NUM_STEPS} steps, ${GPU_COUNT} GPUs) ==="
echo "  Teacher: ${TEACHER_MODEL} (GPU 0)"
echo "  Student: ${MODEL_PATH} (GPUs ${VERL_GPU_LIST})"

mkdir -p "${OUT_ROOT}"

# ── 1) teacher ──────────────────────────────────────────────────────────────
ray stop -f >/dev/null 2>&1 || true
echo "=== Starting teacher (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ${CONDA_ENV}/bin/python -m dual_track_opd.fc_opd.teacher_service \
    --backend transformers --model "${TEACHER_MODEL}" \
    --port "${TEACHER_PORT}" --dtype bfloat16 --device cuda:0 \
    >"${TEACHER_LOG}" 2>&1 &
TEACHER_PID=$!
for _ in $(seq 1 300); do
    curl -fsS "http://127.0.0.1:${TEACHER_PORT}/health" >/dev/null 2>&1 && break
    sleep 1
done
curl -fsS "http://127.0.0.1:${TEACHER_PORT}/health" >/dev/null 2>&1 || {
    echo "FATAL: teacher not healthy within 300s"; tail -50 "${TEACHER_LOG}"; exit 1; }
echo "  Teacher PID ${TEACHER_PID} healthy"

# ── 2) Ray ─────────────────────────────────────────────────────────────────
echo "=== Starting Ray (GPUs ${VERL_GPU_LIST}) ==="
CUDA_VISIBLE_DEVICES=${VERL_GPU_LIST} ray start --head --num-gpus=${VERL_GPUS} --disable-usage-stats
sleep 3

# ── 3) verl PPO ─────────────────────────────────────────────────────────────
echo "=== Running verl PPO (arm ${ARM}, ${NUM_STEPS} steps) ==="
set +e
${CONDA_ENV}/bin/python -m verl.trainer.main_ppo \
    --config-path="${VERL_CONFIG_DIR}" \
    --config-name=ppo_trainer \
    "data.train_files=${PARQUET}" \
    "data.val_files=${PARQUET}" \
    "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "data.max_prompt_length=1024" \
    "data.max_response_length=512" \
    "data.filter_overlong_prompts=false" \
    "data.truncation=error" \
    "data.image_key=images" \
    "data.dataloader_num_workers=8" \
    "data.custom_cls.path=file://${REPO_ROOT}/src/dual_track_opd/support_aware/verl_stp_dataset.py" \
    "data.custom_cls.name=STPTransitionDataset" \
    "data.prefix_manifest=${PREFIX_MANIFEST}" \
    "data.scaffold_all=true" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.model.use_remove_padding=false" \
    "actor_rollout_ref.model.use_fused_kernels=false" \
    "actor_rollout_ref.model.enable_gradient_checkpointing=true" \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    "actor_rollout_ref.actor.optim.lr=${LR}" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.actor.use_dynamic_bsz=true" \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384" \
    "actor_rollout_ref.actor.use_kl_loss=false" \
    "actor_rollout_ref.actor.fsdp_config.param_offload=true" \
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true" \
    "actor_rollout_ref.rollout.name=vllm" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}" \
    "actor_rollout_ref.rollout.max_model_len=2048" \
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8" \
    "actor_rollout_ref.rollout.agent.num_workers=8" \
    "actor_rollout_ref.ref.fsdp_config.param_offload=true" \
    "reward_model.enable=false" \
    "reward_model.reward_manager=null" \
    "reward_model.reward_loop_source=null" \
    "reward_model.model.path=null" \
    "custom_reward_function.path=${REWARD_FN}" \
    "custom_reward_function.name=compute_score" \
    "algorithm.adv_estimator=grpo" \
    "algorithm.use_kl_in_reward=false" \
    "+algorithm.stp_opd.post_rollout_hook=dual_track_opd.support_aware.verl_stp_opd_integration.stp_opd_post_rollout_hook" \
    "+algorithm.stp_opd.arm=${ARM}" \
    "+algorithm.stp_opd.loss_coef=1.0" \
    "+algorithm.stp_opd.lambda_prefix=1.0" \
    "+algorithm.stp_opd.lambda_distill=1.0" \
    "+algorithm.stp_opd.lambda_task=1.0" \
    "+algorithm.stp_opd.teacher_url=http://127.0.0.1:${TEACHER_PORT}" \
    "trainer.total_epochs=1" \
    "trainer.n_gpus_per_node=${VERL_GPUS}" \
    "trainer.nnodes=1" \
    "trainer.critic_warmup=0" \
    "trainer.logger=['console']" \
    "trainer.project_name=stp_opd_canary" \
    "trainer.experiment_name=arm_${ARM}_${NUM_STEPS}steps" \
    "trainer.save_freq=-1" \
    "trainer.test_freq=-1" \
    "trainer.val_before_train=false" \
    "trainer.max_training_steps=${NUM_STEPS}" \
    "trainer.record_per_steps=1" \
    "trainer.rollout_per_steps=1" \
    "trainer.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "trainer.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
VERL_EXIT=$?
set -e

ray stop -f >/dev/null 2>&1 || true
kill "${TEACHER_PID}" 2>/dev/null || true
echo "=== verl exit code: ${VERL_EXIT} ==="
echo "  Teacher log: ${TEACHER_LOG}"
exit ${VERL_EXIT}

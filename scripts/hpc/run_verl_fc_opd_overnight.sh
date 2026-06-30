#!/usr/bin/env bash
# FC-OPD overnight multi-step training (reverse KL, up to 200 steps).
#
# Scaled-up config referencing Vision-OPD (VA-OPD) paper settings:
#   - Rollout n=8 (VA-OPD: 8, OPD-SFT: 16).  Was 1.
#   - LR 2e-6 (VA-OPD: 2e-6 for 4B).
#   - GPU memory 0.5 for vLLM (VA-OPD: 0.7; we're conservative with FSDP).
#   - Reverse KL (mode-seeking) as default.
#   - Checkpoint every 25 steps.
#   - Log to file + console.
#
# Usage:
#   bash scripts/hpc/run_verl_fc_opd_overnight.sh [--gpus N] [--steps S] [--background] [--data /path/to/train.parquet]
#
#   --gpus N        GPUs to use (default: 4).  GPU 0=teacher, 1..N-2=verl, N-1=scorer.
#   --steps S       PPO steps (default: 200).
#   --background    Detach from terminal via nohup — safe to close code-server.
#   --data PATH     Override parquet path (default: verl_smoke/train.parquet).
#
# Background mode:
#   When --background is passed, the script re-launches itself under nohup and
#   exits immediately.  The training runs in the background and survives
#   terminal / code-server disconnects.  Check progress with:
#     tail -f artifacts/fc_opd/train_fc_opd_overnight_*.log

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

GPU_COUNT=4
NUM_STEPS=660  # 5 epochs × (2101 prompts / 16 batch), align VA-OPD
TOP_K=32
TEACHER_PORT=18080
RUN_BACKGROUND=false
PARQUET_OVERRIDE=""

# ── parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)       GPU_COUNT="$2"; shift 2 ;;
        --steps)      NUM_STEPS="$2"; shift 2 ;;
        --data)       PARQUET_OVERRIDE="$2"; shift 2 ;;
        --background) RUN_BACKGROUND=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── background re-launch ────────────────────────────────────────────────────
if ${RUN_BACKGROUND}; then
    # Re-invoke this script without --background, under nohup.
    # Build args without --background
    RELAUNCH_ARGS=()
    for arg in "$@"; do
        [[ "$arg" != "--background" ]] || continue
        RELAUNCH_ARGS+=("$arg")
    done
    if [[ -n "${PARQUET_OVERRIDE}" ]]; then
        RELAUNCH_ARGS+=(--data "${PARQUET_OVERRIDE}")
    fi
    NOHUP_LOG="${REPO_ROOT}/artifacts/fc_opd/nohup_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "${NOHUP_LOG}")"
    echo "Launching background training (PID will be printed, then exits)."
    echo "Monitor:  tail -f ${NOHUP_LOG}"
    nohup bash "$0" --gpus "${GPU_COUNT}" --steps "${NUM_STEPS}" "${RELAUNCH_ARGS[@]}" \
        > "${NOHUP_LOG}" 2>&1 &
    disown
    echo "Background PID: $!"
    exit 0
fi

MODEL_PATH="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-4B-Instruct"
TEACHER_MODEL="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct"
PARQUET="${PARQUET_OVERRIDE:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/geometry3k_full/train.parquet}"
CONDA_ENV="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/envs/fc-opd-verl071-cu128"
REPO_ROOT_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="fc_opd_overnight_$(date +%Y%m%d_%H%M%S)"
TEACHER_LOG="${REPO_ROOT_ABS}/artifacts/fc_opd/teacher_${RUN_ID}.log"
TRAIN_LOG="${REPO_ROOT_ABS}/artifacts/fc_opd/train_${RUN_ID}.log"
VERL_CONFIG_DIR="${REPO_ROOT_ABS}/third_party/verl/verl/trainer/config"
REWARD_FN="file://${REPO_ROOT_ABS}/src/dual_track_opd/fc_opd/smoke_reward.py"
mkdir -p "$(dirname "${TEACHER_LOG}")"

# ── GPU math ────────────────────────────────────────────────────────────────
# GPU 0: Teacher (32B, 66 GB)
# GPU 1..N-2: verl PPO training (WorkerDict + vLLM)
# GPU N-1: StudentScorer Ray actor (4B, ~8 GB)
if (( GPU_COUNT < 4 )); then
    echo "ERROR: need at least 4 GPUs (1=teacher, 2=verl train, 1=StudentScorer)"
    exit 1
fi
VERL_GPUS=$(( GPU_COUNT - 1 ))
TRAIN_GPUS=$(( VERL_GPUS - 1 ))
TEACHER_GPU=0
VERL_GPU_LIST=$(seq -s, 1 $(( GPU_COUNT - 1 )))
SAVE_FREQ=25

# Aligned with VA-OPD: batch_size=16, rollout_n=4, 5 epochs.
# Geometry3K: 2101 prompts / 16 batch ≈ 132 steps/epoch × 5 = 660 steps.
TRAIN_BATCH_SIZE=16
ROLLOUT_N=4
PPO_MINI_BATCH_SIZE=${TRAIN_BATCH_SIZE}   # verl requires mini <= train_batch_size (both in prompts)
MICRO_BATCH_PER_GPU=1

CHECKPOINT_DIR="${REPO_ROOT_ABS}/checkpoints/verl_fc_opd_overnight/${RUN_ID}"

echo "══════════════════════════════════════════════════════════════"
echo "  FC-OPD Training — VA-OPD aligned"
echo "  Run ID:       ${RUN_ID}"
echo "  Steps:        ${NUM_STEPS} (5 epochs × 2101/16 batch)"
echo "  Save freq:    ${SAVE_FREQ}"
echo "  Loss mode:    reverse (mode-seeking KL)"
echo "  Top-K:        ${TOP_K}"
echo "  Rollout n:    ${ROLLOUT_N} (VA-OPD: 4)"
echo "  Train batch:  ${TRAIN_BATCH_SIZE} (VA-OPD: 16)"
echo "  LR:           2e-6 (VA-OPD: 2e-6)"
echo "  GPU layout:   teacher=0, train=1..$((TRAIN_GPUS)), scorer=$((GPU_COUNT-1))"
echo "  Data:         ${PARQUET} ($(python3 -c \"import pandas as pd; print(len(pd.read_parquet('${PARQUET}')))\" 2>/dev/null || echo '?') rows)"
echo "  Train log:    ${TRAIN_LOG}"
echo "  Checkpoint:   ${CHECKPOINT_DIR}"
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
for i in $(seq 1 300); do
    if curl -s "http://127.0.0.1:${TEACHER_PORT}/health" >/dev/null 2>&1; then echo " OK"; break; fi
    if ! kill -0 ${TEACHER_PID} 2>/dev/null; then echo " DIED"; tail -20 "${TEACHER_LOG}"; exit 1; fi
    echo -n "."; sleep 1
done

# ── 2) Ray ──────────────────────────────────────────────────────────────────
echo "=== Ray (${VERL_GPUS} GPUs) ==="
CUDA_VISIBLE_DEVICES=${VERL_GPU_LIST} ray start --head --num-gpus=${VERL_GPUS} --disable-usage-stats
sleep 3

# ── 3) PPO ──────────────────────────────────────────────────────────────────
echo "=== Training (${NUM_STEPS} steps, reverse KL, rollout n=${ROLLOUT_N}) ==="
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
    "data.custom_cls.path=file://${REPO_ROOT_ABS}/src/dual_track_opd/fc_opd/verl_dataset.py" \
    "data.custom_cls.name=FCOPDDataset" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.model.use_remove_padding=false" \
    "actor_rollout_ref.model.use_fused_kernels=false" \
    "actor_rollout_ref.model.enable_gradient_checkpointing=true" \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    "actor_rollout_ref.actor.optim.lr=2e-6" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU}" \
    "actor_rollout_ref.actor.use_dynamic_bsz=true" \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384" \
    "actor_rollout_ref.actor.use_kl_loss=false" \
    "actor_rollout_ref.actor.fsdp_config.param_offload=true" \
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true" \
    "actor_rollout_ref.rollout.name=vllm" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    "actor_rollout_ref.rollout.max_model_len=2048" \
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8" \
    "actor_rollout_ref.rollout.agent.num_workers=8" \
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
    "+algorithm.fc_opd.loss_coef=0.01" \
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

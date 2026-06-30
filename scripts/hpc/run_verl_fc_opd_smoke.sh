#!/usr/bin/env bash
# FC-OPD verl smoke — teacher + PPO, parameterized GPU count.
#
# Usage:
#   bash scripts/hpc/run_verl_fc_opd_smoke.sh [--gpus N] [--top-k K] [--steps S] [--loss-mode forward|reverse] [--background] [--dry-run]
#
#   --gpus N       GPUs to use (default: 8). GPU 0 → teacher; GPU 1..N-1 → verl.
#   --top-k K      Teacher/student top-K (default: 32).
#   --steps S      PPO steps to run (default: 1).
#   --loss-mode    forward (default) or reverse KL.
#   --background   Detach via nohup — safe to close terminal/code-server.
#   --dry-run      Print the command without running.
#
#   Current defaults align with Vision-OPD (VA-OPD) paper: rollout n=8, LR=2e-6.
#
# Examples:
#   bash scripts/hpc/run_verl_fc_opd_smoke.sh --gpus 4 --steps 2
#   bash scripts/hpc/run_verl_fc_opd_smoke.sh --gpus 4 --steps 200 --background
#   bash scripts/hpc/run_verl_fc_opd_smoke.sh --gpus 8 --top-k 100 --loss-mode reverse

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ── defaults ────────────────────────────────────────────────────────────────
GPU_COUNT=8
TOP_K=32
NUM_STEPS=1
LOSS_MODE="reverse"
DRY_RUN=false
RUN_BACKGROUND=false
KEEP_TEACHER=false
TEACHER_PORT=18080
ROLLOUT_N=8
PPO_MINI_BATCH_SIZE=2  # must be <= train_batch_size; matches TRAIN_GPUS for smoke
LR=2e-6
GPU_MEM_UTIL=0.7

# ── paths (NFS, visible to all nodes) ───────────────────────────────────────
MODEL_PATH="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-4B-Instruct"
TEACHER_MODEL="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct"
PARQUET="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/verl_smoke/train.parquet"
CONDA_ENV="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/envs/fc-opd-verl071-cu128"
REPO_ROOT_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEACHER_LOG="${REPO_ROOT_ABS}/artifacts/fc_opd/teacher_$(date +%Y%m%d_%H%M%S).log"
VERL_CONFIG_DIR="${REPO_ROOT_ABS}/third_party/verl/verl/trainer/config"
REWARD_FN="file://${REPO_ROOT_ABS}/src/dual_track_opd/fc_opd/smoke_reward.py"
mkdir -p "$(dirname "${TEACHER_LOG}")"

# ── parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)       GPU_COUNT="$2"; shift 2 ;;
        --top-k)      TOP_K="$2";      shift 2 ;;
        --steps)      NUM_STEPS="$2";  shift 2 ;;
        --loss-mode)  LOSS_MODE="$2";  shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        --background) RUN_BACKGROUND=true; shift ;;
        --keep-teacher) KEEP_TEACHER=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── background re-launch ────────────────────────────────────────────────────
if ${RUN_BACKGROUND}; then
    RELAUNCH_ARGS=()
    for arg in "$@"; do
        [[ "$arg" != "--background" ]] || continue
        RELAUNCH_ARGS+=("$arg")
    done
    NOHUP_LOG="${REPO_ROOT_ABS}/artifacts/fc_opd/nohup_smoke_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "${NOHUP_LOG}")"
    echo "Launching background smoke test.  Monitor:  tail -f ${NOHUP_LOG}"
    nohup bash "$0" "${RELAUNCH_ARGS[@]}" > "${NOHUP_LOG}" 2>&1 &
    disown
    echo "Background PID: $!"
    exit 0
fi

if (( GPU_COUNT < 4 )); then
    echo "ERROR: need at least 4 GPUs (1=teacher, 2=verl train, 1=StudentScorer)"
    exit 1
fi

VERL_GPUS=$(( GPU_COUNT - 1 ))
# Reserve 1 GPU from the verl pool for the StudentScorer Ray actor.
# Verl PPO training uses (VERL_GPUS - 1) GPUs; the last GPU hosts StudentScorer.
TRAIN_GPUS=$(( VERL_GPUS - 1 ))
TEACHER_GPU=0
VERL_GPU_LIST=$(seq -s, 1 $(( GPU_COUNT - 1 )))

echo "══════════════════════════════════════════════════════════════"
echo "  FC-OPD Smoke — $(date)"
echo "══════════════════════════════════════════════════════════════"
echo "  GPU count:      ${GPU_COUNT} (teacher=0, train=1..$((TRAIN_GPUS)), scorer=$((GPU_COUNT-1)))"
echo "  Top-K:          ${TOP_K}"
echo "  Steps:          ${NUM_STEPS}"
echo "  Loss mode:      ${LOSS_MODE}"
echo "  Rollout n:      ${ROLLOUT_N}"
echo "  LR:             ${LR}"
echo "  GPU mem util:   ${GPU_MEM_UTIL}"
echo "  Model:          ${MODEL_PATH}"
echo "  Parquet:        ${PARQUET}"
echo "  Teacher log:    ${TEACHER_LOG}"
echo ""

# ── verify prerequisites ────────────────────────────────────────────────────
if ! grep -q "compute_verl_sparse_topk_kd" third_party/verl/verl/workers/actor/dp_actor.py; then
    echo "FATAL: actor patch not applied to dp_actor.py"
    exit 1
fi
if ! grep -q "fc_opd_hook_fqn" third_party/verl/verl/trainer/ppo/ray_trainer.py; then
    echo "FATAL: trainer patch not applied to ray_trainer.py"
    exit 1
fi
echo "[OK] verl patches verified"

# ── CC check ────────────────────────────────────────────────────────────────
if [ -z "${CC:-}" ] || ! command -v "${CC}" >/dev/null 2>&1; then
    export CC=gcc
fi
echo "[OK] CC=${CC} ($(command -v "${CC}"))"

# ── stop any previous instances ─────────────────────────────────────────────
echo ""
echo "=== Stopping previous instances ==="
ray stop -f 2>/dev/null || true
# If --keep-teacher, leave the teacher running for iterative debugging.
if ${KEEP_TEACHER}; then
    echo "[keep-teacher] Skipping teacher restart — reusing port ${TEACHER_PORT}"
else
    EXISTING_TEACHER=$(lsof -ti:${TEACHER_PORT} 2>/dev/null || true)
    if [ -n "${EXISTING_TEACHER}" ]; then
        echo "Killing existing teacher on port ${TEACHER_PORT} (PID ${EXISTING_TEACHER})"
        kill -9 ${EXISTING_TEACHER} 2>/dev/null || true
        sleep 2
    fi
fi
# Clean stale vLLM shared memory
rm -rf /dev/shm/*vllm* /dev/shm/*psm_* 2>/dev/null || true
sleep 2

# ── dry-run: print and exit ─────────────────────────────────────────────────
if ${DRY_RUN}; then
    echo ""
    echo "=== DRY RUN — would execute: ==="
    echo "1) Teacher on GPU ${TEACHER_GPU}, port ${TEACHER_PORT}"
    echo "2) Ray on GPUs ${VERL_GPU_LIST} (${VERL_GPUS} GPUs)"
    echo "3) verl PPO with --config-path=${VERL_CONFIG_DIR} --config-name=ppo_trainer"
    echo "   + FC-OPD overrides: top_k=${TOP_K}, loss_mode=${LOSS_MODE}, coef=0.1"
    echo "   + Conditions: full,degraded,free,task_visible,task_infer,task_solve"
    exit 0
fi

# ── 1) Start teacher ────────────────────────────────────────────────────────
TEACHER_PID=""
if curl -s "http://127.0.0.1:${TEACHER_PORT}/health" >/dev/null 2>&1; then
    echo ""
    echo "=== Teacher already running on port ${TEACHER_PORT} — reusing ==="
else
    echo ""
    echo "=== Starting teacher (GPU ${TEACHER_GPU}) ==="
    CUDA_VISIBLE_DEVICES=${TEACHER_GPU} \
        ${CONDA_ENV}/bin/python -m dual_track_opd.fc_opd.teacher_service \
        --backend transformers \
        --model "${TEACHER_MODEL}" \
        --port "${TEACHER_PORT}" \
        --top-k "${TOP_K}" \
        --dtype bfloat16 \
        --device "cuda:0" \
        > "${TEACHER_LOG}" 2>&1 &

    TEACHER_PID=$!
    echo "  Teacher PID: ${TEACHER_PID}"

    # Wait for teacher health check (up to 300s for 32B model load: shards 15s + GPU 60s + processor init)
    echo -n "  Waiting for teacher ."
    HEALTHY=false
    for i in $(seq 1 300); do
        if curl -s "http://127.0.0.1:${TEACHER_PORT}/health" >/dev/null 2>&1; then
            HEALTHY=true
            echo " OK ($(curl -s http://127.0.0.1:${TEACHER_PORT}/health))"
            break
        fi
        if ! kill -0 ${TEACHER_PID} 2>/dev/null; then
            echo ""
            echo "FATAL: teacher process died. Last 20 lines of ${TEACHER_LOG}:"
            tail -20 "${TEACHER_LOG}"
            exit 1
        fi
        echo -n "."
        sleep 1
    done
    if ! ${HEALTHY}; then
        echo ""
        echo "FATAL: teacher did not become healthy within 300s"
        tail -50 "${TEACHER_LOG}"
        exit 1
    fi
fi

# ── 2) Start Ray ────────────────────────────────────────────────────────────
echo ""
echo "=== Starting Ray (GPUs ${VERL_GPU_LIST}, ${VERL_GPUS} GPUs) ==="
CUDA_VISIBLE_DEVICES=${VERL_GPU_LIST} ray start --head --num-gpus=${VERL_GPUS} --disable-usage-stats
echo "[OK] Ray started"
sleep 3

# ── 3) Run verl PPO ─────────────────────────────────────────────────────────
echo ""
echo "=== Running verl PPO (${NUM_STEPS} step(s)) ==="

# Use verl's own ppo_trainer config as base; inject FC-OPD via CLI overrides.
# ``+algorithm.fc_opd.*`` uses ``+`` because these keys are NEW (not in verl's
# structured AlgoConfig).  Other keys use ``++`` to force-set struct fields.
set +e  # capture exit code for cleanup
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
    "data.dataloader_num_workers=8" \
    "data.custom_cls.path=file://${REPO_ROOT_ABS}/src/dual_track_opd/fc_opd/verl_dataset.py" \
    "data.custom_cls.name=FCOPDDataset" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.model.use_remove_padding=true" \
    "actor_rollout_ref.model.use_fused_kernels=true" \
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
    "+algorithm.fc_opd.loss_mode=${LOSS_MODE}" \
    "+algorithm.fc_opd.renormalize_topk=true" \
    "+algorithm.fc_opd.include_tail=true" \
    "trainer.total_epochs=${NUM_STEPS}" \
    "trainer.n_gpus_per_node=${TRAIN_GPUS}" \
    "trainer.nnodes=1" \
    "trainer.critic_warmup=0" \
    "trainer.logger=['console']" \
    "trainer.project_name=verl_fc_opd_smoke" \
    "trainer.experiment_name=fc_opd_one_step" \
    "trainer.save_freq=-1" \
    "trainer.test_freq=-1" \
    "trainer.val_before_train=false"
VERL_EXIT=$?

# ── cleanup ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Cleanup ==="
ray stop -f 2>/dev/null || true
if ${KEEP_TEACHER}; then
    echo "[keep-teacher] Teacher left running on port ${TEACHER_PORT}"
else
    kill ${TEACHER_PID} 2>/dev/null || true
fi
sleep 2

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Teacher log: ${TEACHER_LOG}"
echo "  verl exit code: ${VERL_EXIT}"
echo "══════════════════════════════════════════════════════════════"

exit ${VERL_EXIT}

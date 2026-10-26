#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG="${SCOPED_CONFIG:-$ROOT/configs/external/scoped_expert_reproduction.yaml}"
REPO_ROOT="${SCOPED_REPO_ROOT:-$ROOT/fc-opd-storage/external/scoped/repo}"
STORAGE_ROOT="${SCOPED_STORAGE_ROOT:-$ROOT/fc-opd-storage/external/scoped/storage}"
GPU_A="${SCOPED_GPU_GROUP_A:-0,1,2,3}"
GPU_B="${SCOPED_GPU_GROUP_B:-4,5,6,7}"
BASE_MODEL="${BASE_MODEL:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct}"
STUDENT_MODEL="${STUDENT_MODEL:-$BASE_MODEL}"
AUTO_DOWNLOAD_DATA="${AUTO_DOWNLOAD_DATA:-1}"
SAVE_MODEL_ONLY="${SCOPED_SAVE_MODEL_ONLY:-true}"
SAVE_LIMIT="${SCOPED_SAVE_LIMIT:-1}"
PYTHON_INCLUDE_DIR="${SCOPED_PYTHON_INCLUDE_DIR:-$STORAGE_ROOT/runtime/python3.12-include}"
RUNTIME_PYTHONPATH="$ROOT/src:$ROOT/src/dual_track_opd/scoped_runtime"
PAIR=a
STAGE=rl
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: bash scripts/hpc/run_scoped_expert_pair.sh [options]

Stages:
  rl       Run one two-expert parallel pair (default)
  export   Export all four completed RL checkpoints as teachers
  skill    Run the upstream four-capability skill extraction stage

Options:
  --pair a|b   Pair A = capability2 + capability3; Pair B = capability1 + capability4
  --dry-run    Print the exact upstream commands without launching them
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pair) PAIR="${2:?missing pair}"; shift 2 ;;
        --stage) STAGE="${2:?missing stage}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$PAIR" in
    a) CAP_A=capability2; CAP_B=capability3 ;;
    b) CAP_A=capability1; CAP_B=capability4 ;;
    *) echo "Pair must be a or b" >&2; exit 2 ;;
esac

case "$STAGE" in rl|export|skill) ;; *) echo "Stage must be rl, export, or skill" >&2; exit 2 ;; esac

[[ -d "$REPO_ROOT/.git" ]] || { echo "Missing SCOPED checkout: $REPO_ROOT" >&2; exit 1; }
[[ "$GPU_A" != "$GPU_B" ]] || { echo "GPU groups must differ" >&2; exit 1; }

RL_PYTHON="${RL_PYTHON:-$REPO_ROOT/.venv-rl/bin/python}"
SCOPED_PYTHON="${SCOPED_PYTHON:-$REPO_ROOT/.venv-scoped/bin/python}"
SCOPED_CUDA12_RUNTIME="$REPO_ROOT/.venv-scoped/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
if [[ "$DRY_RUN" != 1 ]]; then
    if [[ "$STAGE" == rl && ! -f "$PYTHON_INCLUDE_DIR/Python.h" ]]; then
        echo "Missing Python 3.12 headers: $PYTHON_INCLUDE_DIR" >&2
        exit 1
    fi
    if [[ "$STAGE" == rl ]]; then
        [[ -x "$RL_PYTHON" ]] || { echo "Missing executable: $RL_PYTHON" >&2; exit 1; }
    else
        [[ -x "$SCOPED_PYTHON" ]] || { echo "Missing executable: $SCOPED_PYTHON" >&2; exit 1; }
    fi
fi

DATA_DIR="$STORAGE_ROOT/data/hf_dataset"
OUTPUT_DIR="$STORAGE_ROOT/outputs"
LOG_DIR="$STORAGE_ROOT/logs"
RECORD_DIR="$STORAGE_ROOT/records"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR" "$RECORD_DIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RAY_HEAD_PORT="${SCOPED_RAY_PORT:-6379}"
RAY_HEAD_TMP="/tmp/scoped-ray-$STAMP-head"
RAY_BIN="$REPO_ROOT/.venv-rl/bin/ray"
CLEAN_RAY="${SCOPED_CLEAN_RAY:-1}"

cleanup_ray() {
    if [[ "$STAGE" == rl && "$CLEAN_RAY" == 1 ]]; then
        "$RAY_BIN" stop --force >/dev/null 2>&1 || true
    fi
}
trap cleanup_ray EXIT

record_contract() {
    local stage=$1
    python "$ROOT/scripts/hpc/scoped_reproduction.py" \
        --config "$CONFIG" \
        --repo-root "$REPO_ROOT" \
        --storage-root "$STORAGE_ROOT" \
        --stage "$stage" \
        --gpu-group-a "$GPU_A" \
        --gpu-group-b "$GPU_B" \
        --command "run_scoped_expert_pair.sh --stage $stage --pair $PAIR" \
        --output "$RECORD_DIR/${stage}_${STAMP}.json"
}

run_expert() {
    local capability=$1 gpu_group=$2 ray_tmp=$3 log=$4
    local command=(
        env
        CUDA_VISIBLE_DEVICES="$gpu_group"
        RAY_TMPDIR="$ray_tmp"
        RAY_ADDRESS="127.0.0.1:$RAY_HEAD_PORT"
        SCOPED_STORAGE="$STORAGE_ROOT"
        DATA_DIR="$DATA_DIR"
        OUTPUT_DIR="$OUTPUT_DIR"
        BASE_MODEL="$BASE_MODEL"
        MODEL_PATH="$BASE_MODEL"
        STUDENT_MODEL="$STUDENT_MODEL"
        AUTO_DOWNLOAD_DATA="$AUTO_DOWNLOAD_DATA"
        HF_HUB_OFFLINE=1
        TRANSFORMERS_OFFLINE=1
        CPATH="$PYTHON_INCLUDE_DIR${CPATH:+:$CPATH}"
        PYTHONPATH="$RUNTIME_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
        bash "$REPO_ROOT/scripts/lib/train_capability.sh" "$capability" \
            "trainer.save_model_only=$SAVE_MODEL_ONLY" \
            "trainer.save_limit=$SAVE_LIMIT"
    )
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '%q ' "${command[@]}" >&2
        printf '\n' >&2
    else
        "${command[@]}" >"$log" 2>&1
    fi
}

case "$STAGE" in
    rl)
        if [[ "$DRY_RUN" != 1 ]]; then
            if [[ "$CLEAN_RAY" == 1 ]]; then
                "$RAY_BIN" stop --force >/dev/null 2>&1 || true
            fi
            mkdir -p "$RAY_HEAD_TMP"
            "$RAY_BIN" start --head \
                --port "$RAY_HEAD_PORT" \
                --temp-dir "$RAY_HEAD_TMP" \
                --disable-usage-stats \
                --include-dashboard=false \
                >"$LOG_DIR/ray_head_$STAMP.log" 2>&1
            record_contract "rl_pair_${PAIR}_planned"
            "$RL_PYTHON" "$REPO_ROOT/scripts/lib/prepare_data.py" \
                --capability all --output "$DATA_DIR" --offline
            record_contract "rl_pair_${PAIR}_data_ready"
        fi

        if [[ "$DRY_RUN" == 1 ]]; then
            run_expert "$CAP_A" "$GPU_A" "$RAY_HEAD_TMP" "$LOG_DIR/rl_${CAP_A}_${STAMP}.log"
            run_expert "$CAP_B" "$GPU_B" "$RAY_HEAD_TMP" "$LOG_DIR/rl_${CAP_B}_${STAMP}.log"
            exit 0
        fi
        run_expert "$CAP_A" "$GPU_A" "$RAY_HEAD_TMP" "$LOG_DIR/rl_${CAP_A}_${STAMP}.log" &
        PID_A=$!
        run_expert "$CAP_B" "$GPU_B" "$RAY_HEAD_TMP" "$LOG_DIR/rl_${CAP_B}_${STAMP}.log" &
        PID_B=$!
        echo "Launched $CAP_A on GPUs $GPU_A (pid $PID_A)" >&2
        echo "Launched $CAP_B on GPUs $GPU_B (pid $PID_B)" >&2
        STATUS_A=0
        STATUS_B=0
        wait "$PID_A" || STATUS_A=$?
        wait "$PID_B" || STATUS_B=$?
        if [[ "$STATUS_A" != 0 || "$STATUS_B" != 0 ]]; then
            echo "RL pair failed: $CAP_A=$STATUS_A, $CAP_B=$STATUS_B" >&2
            exit 1
        fi
        ;;
    export)
        if [[ "$DRY_RUN" == 1 ]]; then
            printf '%q ' env LD_LIBRARY_PATH="$SCOPED_CUDA12_RUNTIME${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
                SCOPED_STORAGE="$STORAGE_ROOT" OUTPUT_DIR="$OUTPUT_DIR" \
                "$SCOPED_PYTHON" "$REPO_ROOT/scripts/lib/export_teachers.py" \
                --checkpoint-root "$OUTPUT_DIR/rl" --env-file "$OUTPUT_DIR/teachers.env"
            printf '\n'
            exit 0
        fi
        record_contract "export_teachers"
        env LD_LIBRARY_PATH="$SCOPED_CUDA12_RUNTIME${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            SCOPED_STORAGE="$STORAGE_ROOT" OUTPUT_DIR="$OUTPUT_DIR" \
            "$SCOPED_PYTHON" "$REPO_ROOT/scripts/lib/export_teachers.py" \
            --checkpoint-root "$OUTPUT_DIR/rl" --env-file "$OUTPUT_DIR/teachers.env"
        ;;
    skill)
        if [[ "$DRY_RUN" == 1 ]]; then
            printf '%q ' env LD_LIBRARY_PATH="$SCOPED_CUDA12_RUNTIME${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
                CUDA_VISIBLE_DEVICES="$GPU_A" RAY_TMPDIR="/tmp/scoped-skill-$STAMP" \
                SCOPED_STORAGE="$STORAGE_ROOT" DATA_DIR="$DATA_DIR" OUTPUT_DIR="$OUTPUT_DIR" \
                AUTO_DOWNLOAD_DATA=0 bash "$REPO_ROOT/scripts/train.sh" skill all
            printf '\n'
            exit 0
        fi
        record_contract "skill_extraction"
        env LD_LIBRARY_PATH="$SCOPED_CUDA12_RUNTIME${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            CUDA_VISIBLE_DEVICES="$GPU_A" RAY_TMPDIR="/tmp/scoped-skill-$STAMP" \
            SCOPED_STORAGE="$STORAGE_ROOT" DATA_DIR="$DATA_DIR" OUTPUT_DIR="$OUTPUT_DIR" \
            AUTO_DOWNLOAD_DATA=0 bash "$REPO_ROOT/scripts/train.sh" skill all
        ;;
esac

echo "SCOPED stage complete: $STAGE (pair=$PAIR)" >&2

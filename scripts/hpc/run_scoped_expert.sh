#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG="${SCOPED_CONFIG:-$ROOT/configs/external/scoped_expert_reproduction.yaml}"
REPO_ROOT="${SCOPED_REPO_ROOT:-$ROOT/fc-opd-storage/external/scoped/repo}"
STORAGE_ROOT="${SCOPED_STORAGE_ROOT:-$ROOT/fc-opd-storage/external/scoped/storage}"
GPU_GROUP="${SCOPED_GPU_GROUP:-0,1,2,3}"
BASE_MODEL="${BASE_MODEL:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct}"
AUTO_DOWNLOAD_DATA="${AUTO_DOWNLOAD_DATA:-1}"
SAVE_MODEL_ONLY="${SCOPED_SAVE_MODEL_ONLY:-true}"
SAVE_LIMIT="${SCOPED_SAVE_LIMIT:-1}"
CAPABILITY=""
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: bash scripts/hpc/run_scoped_expert.sh --capability capabilityN [options]

Run one SCOPED RL expert in a 4-GPU allocation.  Use this when experts run on
separate nodes; use run_scoped_expert_pair.sh when two experts share one node.

Options:
  --capability capability1|capability2|capability3|capability4
  --gpu-group  Comma-separated CUDA_VISIBLE_DEVICES value
  --dry-run    Print the upstream command without launching it
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --capability) CAPABILITY="${2:?missing capability}"; shift 2 ;;
        --gpu-group) GPU_GROUP="${2:?missing GPU group}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$CAPABILITY" in
    capability1|capability2|capability3|capability4) ;;
    *) echo "Capability must be capability1..capability4" >&2; exit 2 ;;
esac

[[ -d "$REPO_ROOT/.git" ]] || { echo "Missing SCOPED checkout: $REPO_ROOT" >&2; exit 1; }
RL_PYTHON="${RL_PYTHON:-$REPO_ROOT/.venv-rl/bin/python}"
if [[ "$DRY_RUN" != 1 ]]; then
    [[ -x "$RL_PYTHON" ]] || { echo "Missing executable: $RL_PYTHON" >&2; exit 1; }
fi

DATA_DIR="$STORAGE_ROOT/data/hf_dataset"
OUTPUT_DIR="$STORAGE_ROOT/outputs"
LOG_DIR="$STORAGE_ROOT/logs"
RECORD_DIR="$STORAGE_ROOT/records"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR" "$RECORD_DIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RAY_TMPDIR="/tmp/scoped-ray-${CAPABILITY}-${STAMP}"
LOG="$LOG_DIR/rl_${CAPABILITY}_${STAMP}.log"

if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' env CUDA_VISIBLE_DEVICES="$GPU_GROUP" RAY_TMPDIR="$RAY_TMPDIR" \
        SCOPED_STORAGE="$STORAGE_ROOT" DATA_DIR="$DATA_DIR" OUTPUT_DIR="$OUTPUT_DIR" \
        BASE_MODEL="$BASE_MODEL" AUTO_DOWNLOAD_DATA="$AUTO_DOWNLOAD_DATA" \
        bash "$REPO_ROOT/scripts/train.sh" "$CAPABILITY" \
            "trainer.save_model_only=$SAVE_MODEL_ONLY" \
            "trainer.save_limit=$SAVE_LIMIT" >&2
    printf '\n' >&2
    exit 0
fi

python "$ROOT/scripts/hpc/scoped_reproduction.py" \
    --config "$CONFIG" \
    --repo-root "$REPO_ROOT" \
    --storage-root "$STORAGE_ROOT" \
    --stage "rl_${CAPABILITY}_planned" \
    --gpu-group-a "$GPU_GROUP" \
    --gpu-group-b "" \
    --command "run_scoped_expert.sh --capability $CAPABILITY" \
    --output "$RECORD_DIR/rl_${CAPABILITY}_${STAMP}_planned.json"

"$RL_PYTHON" "$REPO_ROOT/scripts/lib/prepare_data.py" \
    --capability all --output "$DATA_DIR" --offline

python "$ROOT/scripts/hpc/scoped_reproduction.py" \
    --config "$CONFIG" \
    --repo-root "$REPO_ROOT" \
    --storage-root "$STORAGE_ROOT" \
    --stage "rl_${CAPABILITY}_data_ready" \
    --gpu-group-a "$GPU_GROUP" \
    --gpu-group-b "" \
    --command "run_scoped_expert.sh --capability $CAPABILITY" \
    --output "$RECORD_DIR/rl_${CAPABILITY}_${STAMP}_data_ready.json"

echo "Launching $CAPABILITY on GPUs $GPU_GROUP; log: $LOG" >&2
env CUDA_VISIBLE_DEVICES="$GPU_GROUP" RAY_TMPDIR="$RAY_TMPDIR" \
    SCOPED_STORAGE="$STORAGE_ROOT" DATA_DIR="$DATA_DIR" OUTPUT_DIR="$OUTPUT_DIR" \
    BASE_MODEL="$BASE_MODEL" AUTO_DOWNLOAD_DATA="$AUTO_DOWNLOAD_DATA" \
    bash "$REPO_ROOT/scripts/train.sh" "$CAPABILITY" \
        "trainer.save_model_only=$SAVE_MODEL_ONLY" \
        "trainer.save_limit=$SAVE_LIMIT" >"$LOG" 2>&1

echo "SCOPED RL expert complete: $CAPABILITY" >&2

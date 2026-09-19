#!/usr/bin/env bash
set -euo pipefail

# Project-domain VA-OPD: Qwen3-VL-32B-Instruct teacher ->
# Qwen3-VL-8B-Instruct student, ViRL39K, rollout K=8.
# Default is a bounded 500-step pilot. Use --full for epoch-driven training.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HPC_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
PYTHON="${VA_OPD_ENV_PREFIX:-${HPC_ROOT}/envs/va-opd-native-e003-cu128-r595-v1}/bin/python"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

STUDENT_MODEL="${VA_OPD_STUDENT_MODEL:-${HPC_ROOT}/models/Qwen3-VL-8B-Instruct}"
TEACHER_MODEL="${VA_OPD_TEACHER_MODEL:-${HPC_ROOT}/models/Qwen3-VL-32B-Instruct}"
CONFIG_REFERENCE="${VA_OPD_CONFIG_REFERENCE:-${REPO_ROOT}/configs/experiment/qwen3vl_32b_8b_virl39k_va_opd.yaml}"

SOURCE_DATA="${VIRL39K_SOURCE:-${HPC_ROOT}/dataset/ViRL39K/39Krelease.parquet}"
IMAGE_ROOT="${VIRL39K_IMAGE_ROOT:-${HPC_ROOT}/dataset/ViRL39K}"
DATA_ROOT="${VIRL39K_VA_OPD_DATA_ROOT:-${HPC_ROOT}/fc-opd-storage/outputs/fc_opd/virl39k_va_opd}"
TRAIN_DATA="${DATA_ROOT}/train.parquet"
VAL_DATA="${DATA_ROOT}/val_monitor.parquet"
ASSET_DIR="${DATA_ROOT}/assets"
MANIFEST="${DATA_ROOT}/train.parquet.manifest.json"

VISIBLE_GPUS="${VA_OPD_VISIBLE_GPUS:-0,1,2,3,4,5,6,7}"
ACTOR_GPUS="${VA_OPD_ACTOR_GPUS:-4}"
TEACHER_GPUS="${VA_OPD_TEACHER_GPUS:-4}"
TEACHER_TP="${VA_OPD_TEACHER_TP:-2}"
ROLLOUT_N="${VA_OPD_ROLLOUT_N:-8}"
BATCH_SIZE="${VA_OPD_BATCH_SIZE:-16}"
SAVE_FREQ="${VA_OPD_SAVE_FREQ:-25}"
TEST_FREQ="${VA_OPD_TEST_FREQ:-25}"
RUN_ID="${VA_OPD_RUN_ID:-qwen3vl_32b_teacher_8b_student_virl39k_va_opd_pilot500_v1}"
STEPS="${VA_OPD_STEPS:-500}"

DATA_ONLY=false
PREFLIGHT_ONLY=false
FULL_RUN=false
RESUME_ARG=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k.sh [options]

Options:
  --data-only       Only build/validate the ViRL39K VA-OPD parquet assets
  --preflight-only  Build data if missing, then stop after preflight
  --full            Ignore --steps and run epoch-driven full training (5 epochs)
  --steps N         Bounded optimizer steps (default: 500)
  --batch-size N    Prompt batch size (default: 16)
  --run-id ID       Fixed lineage/run ID; reuse it with --resume
  --resume          Enable trainer.resume_mode=auto

Environment:
  VA_OPD_VISIBLE_GPUS default 0,1,2,3,4,5,6,7
  VA_OPD_ACTOR_GPUS   default 4
  VA_OPD_TEACHER_GPUS default 4 (two TP2 teacher replicas)
  VA_OPD_ROLLOUT_N    default 8
  VA_OPD_STEPS        default 500
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-only) DATA_ONLY=true; shift ;;
    --preflight-only) PREFLIGHT_ONLY=true; shift ;;
    --full) FULL_RUN=true; STEPS=0; shift ;;
    --steps) STEPS="${2:?missing step count}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?missing batch size}"; shift 2 ;;
    --run-id) RUN_ID="${2:?missing run id}"; shift 2 ;;
    --resume) RESUME_ARG=(--resume); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in "${STUDENT_MODEL}" "${TEACHER_MODEL}" "${CONFIG_REFERENCE}" "${SOURCE_DATA}" "${IMAGE_ROOT}"; do
  [[ -e "${required}" ]] || {
    echo "FATAL: required path does not exist: ${required}" >&2
    exit 2
  }
done

if [[ ! -x "${PYTHON}" ]]; then
  echo "FATAL: VA-OPD Python is not executable: ${PYTHON}" >&2
  exit 2
fi

student_name="$(basename "${STUDENT_MODEL}")"
teacher_name="$(basename "${TEACHER_MODEL}")"
[[ "${student_name}" == "Qwen3-VL-8B-Instruct" ]] || {
  echo "FATAL: expected Qwen3-VL-8B-Instruct student, got ${student_name}" >&2
  exit 2
}
[[ "${teacher_name}" == "Qwen3-VL-32B-Instruct" ]] || {
  echo "FATAL: expected Qwen3-VL-32B-Instruct teacher, got ${teacher_name}" >&2
  exit 2
}

"${PYTHON}" - "${STUDENT_MODEL}" "${TEACHER_MODEL}" <<'PY'
import json
import sys
from pathlib import Path

student = Path(sys.argv[1])
teacher = Path(sys.argv[2])
for label, path in (("student", student), ("teacher", teacher)):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise SystemExit(f"FATAL: {label} model_type is not qwen3_vl: {path}")
if (student / "tokenizer.json").read_bytes() != (teacher / "tokenizer.json").read_bytes():
    raise SystemExit("FATAL: 8B student and 32B teacher tokenizer.json differ")
PY

if [[ ! -f "${TRAIN_DATA}" || ! -f "${VAL_DATA}" || ! -f "${MANIFEST}" ]]; then
  echo "Preparing ViRL39K VA-OPD data: ${DATA_ROOT}" >&2
  CONVERT_ARGS=()
  if [[ -e "${TRAIN_DATA}" || -e "${VAL_DATA}" || -e "${MANIFEST}" ]]; then
    CONVERT_ARGS=(--overwrite)
  fi
  "${PYTHON}" "${REPO_ROOT}/scripts/gen_virl39k_va_opd_parquet.py" \
    --source "${SOURCE_DATA}" \
    --image-root "${IMAGE_ROOT}" \
    --output "${TRAIN_DATA}" \
    --val-output "${VAL_DATA}" \
    --asset-dir "${ASSET_DIR}" \
    --manifest "${MANIFEST}" \
    --val-rows 500 \
    "${CONVERT_ARGS[@]}"
fi

if ${DATA_ONLY}; then
  mkdir -p "${DATA_ROOT}/preflight"
  "${PYTHON}" "${REPO_ROOT}/scripts/hpc/preflight_va_opd_native.py" \
    --repo-root "${REPO_ROOT}" \
    --backend-dir "${HPC_ROOT}/fc-opd-storage/backends/verl-va-opd-e0031631-clean" \
    --env-prefix "$(dirname "${PYTHON}")" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --student-model "${STUDENT_MODEL}" \
    --teacher-model "${TEACHER_MODEL}" \
    --objective va_opd \
    --visible-gpus "${VISIBLE_GPUS}" \
    --actor-gpus "${ACTOR_GPUS}" \
    --teacher-gpus "${TEACHER_GPUS}" \
    --teacher-tp "${TEACHER_TP}" \
    --prompt-batch-size "${BATCH_SIZE}" \
    --rollout-n "${ROLLOUT_N}" \
    --max-prompt-length 6144 \
    --max-response-length 2048 \
    --config-reference "${CONFIG_REFERENCE}" \
    --run-dir "${DATA_ROOT}/preflight" \
    --audit-all-images
  echo "ViRL39K VA-OPD data preparation: PASS"
  exit 0
fi

STEP_ARGS=(--steps "${STEPS}")
if ${FULL_RUN}; then
  STEP_ARGS=(--steps 0)
fi

PREFLIGHT_ARGS=()
if ${PREFLIGHT_ONLY}; then
  PREFLIGHT_ARGS=(--preflight-only)
fi

export VA_OPD_MAX_ACTOR_CKPT_TO_KEEP="${VA_OPD_MAX_ACTOR_CKPT_TO_KEEP:-null}"

exec bash "${REPO_ROOT}/scripts/hpc/run_va_opd_native.sh" \
  --objective va_opd \
  --profile paper \
  --visible-gpus "${VISIBLE_GPUS}" \
  --actor-gpus "${ACTOR_GPUS}" \
  --teacher-gpus "${TEACHER_GPUS}" \
  --teacher-tp "${TEACHER_TP}" \
  --rollout-n "${ROLLOUT_N}" \
  --batch-size "${BATCH_SIZE}" \
  --student-model "${STUDENT_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --config-reference "${CONFIG_REFERENCE}" \
  --train-data "${TRAIN_DATA}" \
  --val-data "${VAL_DATA}" \
  --run-id "${RUN_ID}" \
  --save-freq "${SAVE_FREQ}" \
  --test-freq "${TEST_FREQ}" \
  "${STEP_ARGS[@]}" \
  "${RESUME_ARG[@]}" \
  "${PREFLIGHT_ARGS[@]}"

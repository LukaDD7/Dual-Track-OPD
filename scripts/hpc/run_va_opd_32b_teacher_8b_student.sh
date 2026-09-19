#!/usr/bin/env bash
set -euo pipefail

# Explicit, fail-fast launcher for the project-scale VA-OPD experiment:
# Qwen3-VL-32B-Instruct teacher -> Qwen3-VL-8B-Instruct student.
#
# This wrapper exists because the generic paper profile defaults to the
# paper's 8B teacher -> 2B student pair.  It refuses to start unless the
# resolved model identity is exactly the 32B/8B pair, preventing accidental
# mislabelled runs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HPC_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"

STUDENT_MODEL="${VA_OPD_STUDENT_MODEL:-${HPC_ROOT}/models/Qwen3-VL-8B-Instruct}"
TEACHER_MODEL="${VA_OPD_TEACHER_MODEL:-${HPC_ROOT}/models/Qwen3-VL-32B-Instruct}"
CONFIG_REFERENCE="${VA_OPD_CONFIG_REFERENCE:-${REPO_ROOT}/configs/experiment/qwen3vl_32b_8b_geometry3k_va_opd_paper.yaml}"

VISIBLE_GPUS="${VA_OPD_VISIBLE_GPUS:-0,1,2,3,4,5,6,7}"
ACTOR_GPUS="${VA_OPD_ACTOR_GPUS:-4}"
TEACHER_GPUS="${VA_OPD_TEACHER_GPUS:-4}"
TEACHER_TP="${VA_OPD_TEACHER_TP:-2}"
RUN_ID="${VA_OPD_RUN_ID:-qwen3vl_32b_teacher_8b_student_va_opd_full5e_v1}"
SAVE_FREQ="${VA_OPD_SAVE_FREQ:-50}"
TEST_FREQ="${VA_OPD_TEST_FREQ:-25}"
RESUME_ARG=()
if [[ "${VA_OPD_RESUME:-0}" == "1" ]]; then
  RESUME_ARG=(--resume)
fi

usage() {
  cat <<'EOF'
Usage:
  bash scripts/hpc/run_va_opd_32b_teacher_8b_student.sh

Environment overrides:
  VA_OPD_VISIBLE_GPUS   default 0,1,2,3,4,5,6,7
  VA_OPD_ACTOR_GPUS     default 4
  VA_OPD_TEACHER_GPUS   default 4 (two TP2 teacher replicas)
  VA_OPD_TEACHER_TP     default 2
  VA_OPD_RUN_ID         default qwen3vl_32b_teacher_8b_student_va_opd_full5e_v1
  VA_OPD_SAVE_FREQ      default 50
  VA_OPD_TEST_FREQ      default 25
  VA_OPD_RESUME=1       enable resume

The model paths and config are intentionally not exposed as loose CLI flags
here. Use the environment overrides only when the paths are deliberate.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in "${STUDENT_MODEL}" "${TEACHER_MODEL}" "${CONFIG_REFERENCE}"; do
  [[ -e "${required}" ]] || {
    echo "FATAL: required path does not exist: ${required}" >&2
    exit 2
  }
done

student_name="$(basename "${STUDENT_MODEL}")"
teacher_name="$(basename "${TEACHER_MODEL}")"
[[ "${student_name}" == "Qwen3-VL-8B-Instruct" ]] || {
  echo "FATAL: expected 8B student, got ${student_name}" >&2
  exit 2
}
[[ "${teacher_name}" == "Qwen3-VL-32B-Instruct" ]] || {
  echo "FATAL: expected 32B teacher, got ${teacher_name}" >&2
  exit 2
}

python3 - <<'PY'
import json
from pathlib import Path

student = Path("/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct")
teacher = Path("/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct")

for label, path in (("student", student), ("teacher", teacher)):
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen3_vl":
        raise SystemExit(f"FATAL: {label} model_type is not qwen3_vl: {path}")

# Shared token-ID semantics are mandatory for exact response-ID alignment.
student_tok = student / "tokenizer.json"
teacher_tok = teacher / "tokenizer.json"
if student_tok.read_bytes() != teacher_tok.read_bytes():
    raise SystemExit("FATAL: 8B student and 32B teacher tokenizer.json differ")
PY

export VA_OPD_STUDENT_MODEL="${STUDENT_MODEL}"
export VA_OPD_TEACHER_MODEL="${TEACHER_MODEL}"
export VA_OPD_MAX_ACTOR_CKPT_TO_KEEP="${VA_OPD_MAX_ACTOR_CKPT_TO_KEEP:-null}"

exec bash "${REPO_ROOT}/scripts/hpc/run_va_opd_native.sh" \
  --objective va_opd \
  --profile paper \
  --visible-gpus "${VISIBLE_GPUS}" \
  --actor-gpus "${ACTOR_GPUS}" \
  --teacher-gpus "${TEACHER_GPUS}" \
  --teacher-tp "${TEACHER_TP}" \
  --student-model "${STUDENT_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --config-reference "${CONFIG_REFERENCE}" \
  --run-id "${RUN_ID}" \
  --save-freq "${SAVE_FREQ}" \
  --test-freq "${TEST_FREQ}" \
  "${RESUME_ARG[@]}"

#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${FC_OPD_TEACHER_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
HOST="${FC_OPD_TEACHER_HOST:-127.0.0.1}"
PORT="${FC_OPD_TEACHER_PORT:-18080}"
TOP_K="${FC_OPD_TEACHER_TOP_K:-32}"
DTYPE="${FC_OPD_TEACHER_DTYPE:-bfloat16}"
DEVICE="${FC_OPD_TEACHER_DEVICE:-cuda}"
REVISION="${FC_OPD_TEACHER_REVISION:-main}"

python -m dual_track_opd.fc_opd.teacher_service \
  --backend transformers \
  --model "$MODEL_ID" \
  --revision "$REVISION" \
  --host "$HOST" \
  --port "$PORT" \
  --top-k "$TOP_K" \
  --dtype "$DTYPE" \
  --device "$DEVICE"

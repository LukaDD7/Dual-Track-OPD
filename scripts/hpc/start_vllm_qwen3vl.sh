#!/usr/bin/env bash
set -euo pipefail
: "${DTOPD_MODEL_ROOT:?Set DTOPD_MODEL_ROOT}"
vllm serve "${DTOPD_MODEL_ROOT}/Qwen3-VL-8B-Instruct"


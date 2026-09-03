#!/usr/bin/env bash
#
# vLLM serve the privileged-hint teacher for PTD-PO (offline hint generation).
# Default: models/qwen3.6-35B-A3B (complete local VLM, qwen3_5_moe arch,
# newer-generation distribution than the Qwen3-VL student) on the cu132 env
# (vLLM 0.27.1 natively supports Qwen3_5MoeForConditionalGeneration).
# NOTE: the old Qwen3-VL-235B-A22B-Instruct-FP8 was 17/24 shards incomplete and
# silently produced degenerate outputs; it was deleted on 2026-08-25.
#
# 用法 (GPU 节点):
#   bash scripts/sft_rl/serve_hint_gen.sh                # 只起 server (默认后台 + health check)
#   HINT_BUILD=1 bash scripts/sft_rl/serve_hint_gen.sh   # 起 server 后自动跑 hint 生成
#   HINT_SERVE_ONLY=1 bash scripts/sft_rl/serve_hint_gen.sh  # 起 server 后立即返回, server 常驻
# 可选 env:
#   HINT_ENV / HINT_MODEL / HINT_GPUS / HINT_PORT / HINT_TP / HINT_MEM / HINT_MAX_LEN
#   HINT_INPUT_DIR / HINT_OUT_DIR / HINT_LIMIT
# 输出: hint 生成结果写到 $HINT_OUT_DIR (默认 fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_3k_hint/)
set -euo pipefail

DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
REPO_ROOT="${DTOPD_ROOT}/projects/Dual-Track-OPD"
HINT_ENV="${HINT_ENV:-${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1}"
VLLM_BIN="${HINT_ENV}/bin/vllm"
LOG_DIR="${DTOPD_ROOT}/fc-opd-storage/logs"
mkdir -p "${LOG_DIR}"

CUDA_HOME="${DTOPD_ROOT}/envs/cuda132-toolchain"
export CUDA_HOME
# HINT_ENV/bin must precede PATH: flashinfer JIT spawns `ninja` and `nvcc` in
# child processes, which inherit this PATH (fixes 2026-08-25 startup failure).
export PATH="${HINT_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${HINT_ENV}/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${HINT_HF_CACHE:-${DTOPD_ROOT}/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

HINT_MODEL="${HINT_MODEL:-${DTOPD_ROOT}/models/qwen3.6-35B-A3B}"
HINT_GPUS="${HINT_GPUS:-0}"
HINT_PORT="${HINT_PORT:-8010}"
HINT_TP="${HINT_TP:-1}"
HINT_MEM="${HINT_MEM:-0.9}"
HINT_MAX_LEN="${HINT_MAX_LEN:-32768}"
SERVED_NAME="${HINT_SERVED_NAME:-Qwen3.6-35B-A3B}"

HINT_INPUT_DIR="${HINT_INPUT_DIR:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_3k}"
HINT_OUT_DIR="${HINT_OUT_DIR:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_3k_hint}"

[ -d "${HINT_MODEL}" ] || { echo "FATAL: model dir missing: ${HINT_MODEL}"; exit 1; }
[ -x "${VLLM_BIN}" ] || { echo "FATAL: vllm missing in ${HINT_ENV}"; exit 1; }
[ -d "${HINT_INPUT_DIR}" ] || { echo "FATAL: input dir missing: ${HINT_INPUT_DIR}"; exit 1; }
if [[ ! -f "${HINT_MODEL}/tokenizer_config.json" || ! -f "${HINT_MODEL}/vocab.json" ]]; then
  echo "FATAL: ${HINT_MODEL} is missing tokenizer files (tokenizer_config.json/vocab.json/tokenizer.json)."
  echo "       Repair (Qwen3-VL 同族 tokenizer, ~11MB):"
  echo "       cp ${DTOPD_ROOT}/models/Qwen3-VL-32B-Instruct/{vocab.json,tokenizer.json,tokenizer_config.json} ${HINT_MODEL}/"
  exit 1
fi
if [[ ! -f "${HINT_MODEL}/preprocessor_config.json" ]]; then
  echo "FATAL: ${HINT_MODEL} is missing preprocessor_config.json (image processor)."
  echo "       Repair: cp ${DTOPD_ROOT}/models/Qwen3-VL-32B-Instruct/preprocessor_config.json ${HINT_MODEL}/"
  exit 1
fi

HINT_LOG="${LOG_DIR}/hint_gen_server_${HINT_PORT}.log"
cleanup() {
  [[ -n "${HINT_PID:-}" ]] && kill "${HINT_PID}" 2>/dev/null || true
}
if [[ "${HINT_SERVE_ONLY:-0}" != "1" ]]; then
  trap cleanup EXIT
fi

echo "== hint-gen server: model=${HINT_MODEL} gpus=${HINT_GPUS} tp=${HINT_TP} port=${HINT_PORT} =="
echo "== input:  ${HINT_INPUT_DIR}"
echo "== output: ${HINT_OUT_DIR} (hint 结果写到这; HINT_BUILD=1 才自动跑, 否则单独跑 build_mmf_hints.py)"

nohup env CUDA_VISIBLE_DEVICES="${HINT_GPUS}" \
  "${VLLM_BIN}" serve "${HINT_MODEL}" \
  --host 127.0.0.1 --port "${HINT_PORT}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size "${HINT_TP}" \
  --gpu-memory-utilization "${HINT_MEM}" \
  --max-model-len "${HINT_MAX_LEN}" \
  --trust-remote-code > "${HINT_LOG}" 2>&1 &
HINT_PID=$!

echo "[hint-gen] waiting for server health (pid=${HINT_PID}, log=${HINT_LOG})..."
for i in $(seq 1 300); do
  if curl -s --max-time 5 "http://127.0.0.1:${HINT_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[hint-gen] server ready: http://127.0.0.1:${HINT_PORT}/v1"
    break
  fi
  if ! kill -0 "${HINT_PID}" 2>/dev/null; then
    echo "FATAL: server died; tail ${HINT_LOG}"; tail -30 "${HINT_LOG}"; exit 1
  fi
  sleep 2
done

if [[ "${HINT_BUILD:-0}" == "1" ]]; then
  "${HINT_ENV}/bin/python" "${REPO_ROOT}/scripts/sft_rl/build_mmf_hints.py" \
    --input-dir "${HINT_INPUT_DIR}" \
    --out-dir "${HINT_OUT_DIR}" \
    --api-base "http://127.0.0.1:${HINT_PORT}/v1" \
    --model "${SERVED_NAME}" \
    ${HINT_LIMIT:+--limit "${HINT_LIMIT}"} \
    2>&1 | tee "${LOG_DIR}/hint_gen_build_$(date +%Y%m%d_%H%M).log"
fi

if [[ "${HINT_SERVE_ONLY:-0}" == "1" ]]; then
  echo "[hint-gen] server detached: pid=${HINT_PID}, log=${HINT_LOG}"
  echo "[hint-gen] stop with: kill ${HINT_PID}"
  exit 0
fi

wait "${HINT_PID}"

#!/usr/bin/env bash
#
# 一键后台跑 hint 生成 (build_mmf_hints.py), 日志落 fc-opd-storage/logs/,
# 进度实时写 <out>/hint_gen_progress.json, 便于共享 NFS 监控。
#
# 用法 (GPU 节点, 8010 server 已在跑):
#   bash scripts/sft_rl/run_hint_gen.sh                  # 全量 3000
#   bash scripts/sft_rl/run_hint_gen.sh --limit 200      # 冒烟 200
# 可选 env: HINT_INPUT_DIR / HINT_OUT_DIR / HINT_PORT / HINT_WORKERS / HINT_MODEL
set -euo pipefail

DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
REPO_ROOT="${DTOPD_ROOT}/projects/Dual-Track-OPD"
PY="${HINT_PY:-${DTOPD_ROOT}/envs/vision-opd-cu128/bin/python}"
INPUT_DIR="${HINT_INPUT_DIR:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_3k}"
OUT_DIR="${HINT_OUT_DIR:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_3k_hint}"
PORT="${HINT_PORT:-8010}"
MODEL="${HINT_MODEL:-Qwen3-VL-235B-Instruct}"
WORKERS="${HINT_WORKERS:-8}"
API="http://127.0.0.1:${PORT}/v1"
LOG="${DTOPD_ROOT}/fc-opd-storage/logs/hint_gen_build_full.log"

[ -x "${PY}" ] || { echo "FATAL: python missing: ${PY}"; exit 1; }
[ -d "${INPUT_DIR}" ] || { echo "FATAL: input dir missing: ${INPUT_DIR}"; exit 1; }

if ! curl -s --max-time 5 "${API}/models" >/dev/null 2>&1; then
  echo "FATAL: hint-gen server not reachable on port ${PORT}."
  echo "       Start it first:  HINT_SERVE_ONLY=1 bash ${REPO_ROOT}/scripts/sft_rl/serve_hint_gen.sh"
  exit 1
fi

nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${PY}" "${REPO_ROOT}/scripts/sft_rl/build_mmf_hints.py" \
  --input-dir "${INPUT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --api-base "${API}" \
  --model "${MODEL}" \
  --n-workers "${WORKERS}" \
  --shard-prefix "${HINT_SHARD_PREFIX:-mmf_rl_train_hint}" \
  "$@" \
  > "${LOG}" 2>&1 &

echo "PID: $!"
echo "log:      ${LOG}"
echo "progress: ${OUT_DIR}/hint_gen_progress.json"
echo "output:   ${OUT_DIR}/${HINT_SHARD_PREFIX:-mmf_rl_train_hint}__part_*.parquet"

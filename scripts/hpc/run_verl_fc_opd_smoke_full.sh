#!/usr/bin/env bash
# Full FC-OPD smoke: restart teacher, run smoke, save ALL logs to NFS.
#
# Usage (on GPU node):
#   bash scripts/hpc/run_verl_fc_opd_smoke_full.sh 2>&1 | tee /tmp/smoke_full.log

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/artifacts/fc_opd/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
TEACHER_LOG="${LOG_DIR}/teacher_${TS}.log"
SMOKE_LOG="${LOG_DIR}/smoke_${TS}.log"

echo "=== FC-OPD Full Smoke ==="
echo "Logs: teacher=${TEACHER_LOG}  smoke=${SMOKE_LOG}"

# 1. Kill old teacher and Ray
pkill -9 -f teacher_service 2>/dev/null || true
ray stop -f 2>/dev/null || true
rm -rf /tmp/ray/*
sleep 2

# 2. Start teacher on GPU 0, log to NFS
echo "[$(date)] Starting teacher..." | tee -a "${TEACHER_LOG}"
CUDA_VISIBLE_DEVICES=0 nohup python -m dual_track_opd.fc_opd.teacher_service \
    --backend transformers \
    --model /inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct \
    --port 18080 --top-k 32 --dtype bfloat16 --device cuda:0 \
    >> "${TEACHER_LOG}" 2>&1 &
TEACHER_PID=$!
echo "Teacher PID: ${TEACHER_PID}"

# 3. Wait for teacher to be ready
echo "Waiting for teacher (up to 60s)..." | tee -a "${SMOKE_LOG}"
for i in $(seq 1 12); do
    sleep 5
    if curl -s http://127.0.0.1:18080/health 2>/dev/null | grep -q '"status":"ok"'; then
        echo "Teacher ready after $((i*5))s" | tee -a "${SMOKE_LOG}"
        break
    fi
    echo "  still waiting... ($((i*5))s)" | tee -a "${SMOKE_LOG}"
done

# 4. Run smoke, log to NFS
echo "[$(date)] Running smoke..." | tee -a "${SMOKE_LOG}"
set +e
CUDA_VISIBLE_DEVICES=1,2 bash scripts/hpc/run_verl_fc_opd_smoke.sh >> "${SMOKE_LOG}" 2>&1
RC=$?
set -e
echo "[$(date)] Smoke exit code: ${RC}" | tee -a "${SMOKE_LOG}"

# 5. Show key lines from both logs
echo ""
echo "=== Teacher log (last 10 lines) ==="
tail -10 "${TEACHER_LOG}"
echo ""
echo "=== Smoke: errors and key metrics ==="
grep -E "Error|Traceback|hook_loss|fc_opd_loss|Done|SUCCESS|FAIL" "${SMOKE_LOG}" || echo "(no matches)"
echo ""
echo "Full logs at: ${TEACHER_LOG}  ${SMOKE_LOG}"

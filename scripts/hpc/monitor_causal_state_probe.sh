#!/usr/bin/env bash
# Poll the four 20260808 causal-probe shards on shared storage.
# Prints a progress table; exits 0 when all 101 units are done.
#
# Usage: bash scripts/hpc/monitor_causal_state_probe.sh [interval_seconds]
set -euo pipefail

OUT="${DTOPD_OUTPUT_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs}/support_aware_opd"
INTERVAL="${1:-600}"
SILENCE_LIMIT="${SILENCE_LIMIT:-21600}"   # 6h with no new result => warn stall

python3 - "${OUT}" "${INTERVAL}" "${SILENCE_LIMIT}" <<'PY'
import json
import sys
import time
from pathlib import Path

out_root = Path(sys.argv[1])
interval = int(sys.argv[2])
silence_limit = int(sys.argv[3])

shards = [f"causal_state_probe_20260808_s{s}" for s in range(4)]

def shard_state(shard):
    root = out_root / shard
    manifest = json.loads((root / "run_manifest.json").read_text())
    expected = set(manifest["provenance"]["expected_work_ids"])
    done = set()
    newest = 0.0
    for path in (root / "trajectory_results").glob("*.json"):
        done.add(json.loads(path.read_text())["trajectory_id"])
        newest = max(newest, path.stat().st_mtime)
    missing = expected - done
    return shard, len(expected), len(done), len(missing), newest

while True:
    now = time.time()
    print(f"\n=== causal-state probe status {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(now))} ===")
    rows = [shard_state(s) for s in shards]
    done_total = sum(r[2] for r in rows)
    missing_total = sum(r[3] for r in rows)
    for shard, n_expected, n_done, n_missing, newest in rows:
        age_h = (now - newest) / 3600 if newest else float("inf")
        flag = ""
        if n_missing and age_h > silence_limit / 3600:
            flag = "  <-- STALL? no result for %.1fh" % age_h
        elif not n_missing:
            flag = "  DONE"
        newest_s = time.strftime("%H:%M", time.gmtime(newest)) if newest else "never"
        print(f"  {shard}: {n_done}/{n_expected} done, {n_missing} left, newest {newest_s} UTC ({age_h:.1f}h ago){flag}")
    print(f"  TOTAL: {done_total}/101 done, {missing_total} left")
    if missing_total == 0:
        print("=== ALL SHARDS COMPLETE ===")
        sys.exit(0)
    time.sleep(interval)
PY

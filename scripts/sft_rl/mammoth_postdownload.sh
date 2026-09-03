#!/usr/bin/env bash
# MAmmoTH-VL multi_image_data post-download driver (detached, survives shell exit).
#
# Waits for the in-flight `snapshot_download` (8 workers) to finish all 20
# multi_image_data shards, verifies each shard's byte size against the
# ModelScope-listed size, extracts the tar.gz trees into `extracted/`, runs the
# SFT converter, and writes a completion marker + stats summary.
#
# Usage (already running as a detached background job):
#   setsid nohup bash scripts/sft_rl/mammoth_postdownload.sh > /tmp/mammoth_postdl.log 2>&1 &

set -u

DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
PY="${DTOPD_ROOT}/envs/va-opd-qwen35-cu128/bin/python"
DS_DIR="${DTOPD_ROOT}/dataset/MAmmoTH-VL-Instruct-12M"
MI_DIR="${DS_DIR}/multi_image_data"
IMG_ROOT="${MI_DIR}/extracted"
OUT_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/mammothvl_ov2m"
DONE_MARK="${DS_DIR}/multi_image_CONVERTED.ok"
REPO=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD

# Expected byte sizes from ModelScope get_dataset_files (AI-ModelScope/MAmmoTH-VL-Instruct-12M).
cat > /tmp/mammoth_expected_sizes.txt <<'EOF'
shard_1.tar.gz 8919149843
shard_2.tar.gz 9879750915
shard_3.tar.gz 31115340978
shard_4.tar.gz 31322943254
shard_5.tar.gz 15921373509
shard_6.tar.gz 170680093
shard_7.tar.gz 227810915
shard_8.tar.gz 200495527
shard_9.tar.gz 1132982368
shard_10.tar.gz 5716696358
shard_11.tar.gz 9788083160
shard_12.tar.gz 12656004383
shard_13.tar.gz 1860887960
shard_14.tar.gz 847835660
shard_15.tar.gz 3697442016
shard_16.tar.gz 11900317960
shard_17.tar.gz 9416776568
shard_18.tar.gz 12368546830
shard_19.tar.gz 17333379121
shard_20.tar.gz 17456817341
EOF

log() { echo "[$(date '+%F %T')] $*"; }

# ---- 1. wait for download completion ----
log "waiting for all 20 shards to finish downloading (no .incomplete left) ..."
START=$(date +%s)
while true; do
    n_inc=$(ls "${MI_DIR}"/*.incomplete 2>/dev/null | wc -l)
    if [ "$n_inc" -eq 0 ]; then
        break
    fi
    now=$(date +%s); elapsed=$(( (now - START) / 60 ))
    log "still downloading: ${n_inc} incomplete (elapsed ${elapsed}m; disk free $(df -h "${MI_DIR}" | awk 'NR==2{print $4}'))"
    sleep 600
done
log "all .incomplete cleared; verifying sizes."

# ---- 2. verify each shard size matches expected ----
BAD=0
while read -r name want; do
    f="${MI_DIR}/${name}"
    if [ ! -f "$f" ]; then
        log "MISSING: $name"; BAD=1; continue
    fi
    got=$(stat -c %s "$f")
    if [ "$got" -ne "$want" ]; then
        log "SIZE-MISMATCH: $name got=$got want=$want (diff=$((want-got)))"; BAD=1
    else
        log "OK: $name ($got bytes)"
    fi
done < /tmp/mammoth_expected_sizes.txt

if [ "$BAD" -ne 0 ]; then
    log "ABORT: shard size verification failed; leaving tar.gz in place for re-download."
    exit 1
fi
log "all 20 shards verified (size match)."

# ---- 3. extract all shards into extracted/ (flat merge on M4-Instruct-Data/) ----
mkdir -p "${IMG_ROOT}"
for name in shard_{1..20}.tar.gz; do
    f="${MI_DIR}/${name}"
    log "extracting $name ..."
    tar -xzf "$f" -C "${IMG_ROOT}"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        log "ABORT: tar extraction failed on $name (rc=$rc)."
        exit 1
    fi
done
log "extraction done. image tree:"
du -sh "${IMG_ROOT}" 2>/dev/null

# Release the compressed shards once the tree is on disk — the extracted tree
# is byte-identical content and re-downloadable if ever needed; keeps ~200 GB
# of headroom for embedding image bytes into parquet (bytes are re-embedded,
# so steady-state disk ≈ tar(202G, deleted) + tree(207G) + parquet(207G)).
log "deleting completed tar.gz shards to free disk (extracted tree is the durable asset) ..."
for name in shard_{1..20}.tar.gz; do
    rm -f "${MI_DIR}/${name}"
done
log "tar.gz shards removed. disk now: $(df -h "${MI_DIR}" | awk 'NR==2{print "free "$4" ("$5" used)"}')"

# ---- 4. run the SFT converter ----
log "running convert_mammothvl.py ..."
cd "${REPO}"
"${PY}" scripts/sft_rl/convert_mammothvl.py \
    --input-json "${DS_DIR}/mammoth_ov_2M.json" \
    --image-root "${IMG_ROOT}" \
    --out-dir "${OUT_DIR}" \
    > /tmp/mammoth_convert.log 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
    log "ABORT: converter failed (rc=$rc); see /tmp/mammoth_convert.log"
    exit 1
fi
log "conversion done:"
tail -5 /tmp/mammoth_convert.log

# ---- 5. completion marker ----
touch "${DONE_MARK}"
log "DONE. marker=${DONE_MARK}  out=${OUT_DIR}"
log "disk after: $(df -h "${MI_DIR}" | awk 'NR==2{print "free "$4" ("$5" used)"}')"
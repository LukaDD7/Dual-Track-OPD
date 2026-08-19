#!/usr/bin/env python3
"""Parallel Range downloader for MMFineReason-SFT-123K HF parquet shards.

Why this exists: the xet CDN used by huggingface_hub >= 1.26 is unreachable
from this cluster (endless ``s3::get_range`` retries), and ``hf_transfer`` is
deprecated/no-op in current huggingface_hub.  Direct parallel Range requests to
the HF resolve URL are the measured-fastest path here (~2.5-3.7 MB/s aggregate,
vs ~0.65 MB/s single-stream).

Usage:
    python scripts/hpc/download_mmfinereason_parquet.py \
        --dest $DTOPD_ROOT/dataset/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking/data

Run it under ``setsid`` so it survives the launching exec session:
    setsid python scripts/hpc/download_mmfinereason_parquet.py > log 2>&1 &
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import time

import requests

DEFAULT_BASE = (
    "https://huggingface.co/datasets/"
    "OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking/resolve/main/data"
)
DEFAULT_DEST = (
    "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/"
    "MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking/data"
)
HEADERS = {"User-Agent": "parallel-range-downloader/1.0"}


def get_size(url: str) -> int:
    r = requests.head(url, headers=HEADERS, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int, idx: int, retries: int) -> tuple[int, bytes]:
    headers = dict(HEADERS)
    headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=180)
            if r.status_code in (200, 206) and len(r.content) == end - start + 1:
                return idx, r.content
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"range {start}-{end} failed after {retries} attempts")


def download_file(url: str, path: str, workers: int, retries: int) -> None:
    size = get_size(url)
    name = os.path.basename(path)
    if os.path.exists(path) and os.path.getsize(path) == size:
        print(f"skip {name} (already complete)", flush=True)
        return
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.truncate(size)
    n = workers
    step = (size + n - 1) // n
    ranges = [(i * step, min((i + 1) * step - 1, size - 1)) for i in range(n)]
    done = [0] * n
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futs = {
            ex.submit(fetch_range, url, s, e, i, retries): i
            for i, (s, e) in enumerate(ranges)
        }
        for fut in cf.as_completed(futs):
            i, data = fut.result()
            with open(tmp, "r+b") as f:
                f.seek(ranges[i][0])
                f.write(data)
            done[i] = len(data)
            got = sum(done)
            pct = 100.0 * got / size
            rate = got / max(time.time() - t0, 1e-6) / 1e6
            print(
                f"{name}: {got/1e6:.1f}/{size/1e6:.1f} MB "
                f"({pct:.1f}%) {rate:.2f} MB/s",
                flush=True,
            )
    os.rename(tmp, path)
    print(f"DONE {name} ({size/1e6:.1f} MB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE, help="HF resolve URL prefix")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="target data/ directory")
    parser.add_argument("--shards", type=int, default=18)
    parser.add_argument("--workers", type=int, default=12, help="range workers per file")
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    for i in range(args.shards):
        name = f"train-{i:05d}-of-{args.shards:05d}.parquet"
        url = f"{args.base}/{name}?download=true"
        download_file(url, os.path.join(args.dest, name), args.workers, args.retries)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

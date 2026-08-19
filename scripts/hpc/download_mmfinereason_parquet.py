#!/usr/bin/env python3
"""Parallel Range downloader for MMFineReason-SFT-123K HF parquet shards.

Why this exists: the xet CDN used by huggingface_hub >= 1.26 is unreachable
from this cluster (endless ``s3::get_range`` retries), and ``hf_transfer`` is
deprecated/no-op in current huggingface_hub.  Direct parallel Range requests to
the HF resolve URL are the measured-fastest path here (~2.4-3.7 MB/s aggregate
vs ~0.65 MB/s single-stream).

Design notes (v2):
- Downloads several shards concurrently so a single hung range cannot block
  the whole dataset.
- Resumes partially written shards at range granularity (a range whose final
  byte is non-zero is treated as complete).
- Per-range timeout is bounded; failed ranges retry with short backoff.

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
HEADERS = {"User-Agent": "parallel-range-downloader/2.0"}


def get_size(url: str) -> int:
    r = requests.head(url, headers=HEADERS, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def _range_complete(path: str, start: int, end: int) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(end)
            return f.read(1) != b"\x00"
    except OSError:
        return False


def fetch_range(
    url: str,
    start: int,
    end: int,
    idx: int,
    timeout: int,
    retries: int,
) -> tuple[int, bytes]:
    headers = dict(HEADERS)
    headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code in (200, 206) and len(r.content) == end - start + 1:
                return idx, r.content
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"range {start}-{end} failed after {retries} attempts")


def download_file(
    url: str,
    path: str,
    ranges_per_file: int,
    timeout: int,
    retries: int,
) -> None:
    size = get_size(url)
    name = os.path.basename(path)
    if os.path.exists(path) and os.path.getsize(path) == size:
        print(f"skip {name} (complete)", flush=True)
        return
    tmp = path + ".part"
    if not (os.path.exists(tmp) and os.path.getsize(tmp) == size):
        with open(tmp, "wb") as f:
            f.truncate(size)
    n = ranges_per_file
    step = (size + n - 1) // n
    ranges = [(i * step, min((i + 1) * step - 1, size - 1)) for i in range(n)]
    todo = [r for r in ranges if not _range_complete(tmp, *r)]
    if len(todo) < len(ranges):
        print(
            f"{name}: resume {len(todo)}/{len(ranges)} ranges "
            f"({100.0 * (size - sum(e - s + 1 for s, e in todo)) / size:.0f}% already on disk)",
            flush=True,
        )
    remaining = sum(e - s + 1 for s, e in todo)
    got = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futs = {
            ex.submit(fetch_range, url, s, e, i, timeout, retries): i
            for i, (s, e) in enumerate(todo)
        }
        for fut in cf.as_completed(futs):
            i, data = fut.result()
            with open(tmp, "r+b") as f:
                f.seek(todo[i][0])
                f.write(data)
            got += len(data)
            pct = 100.0 * (size - remaining + got) / size
            rate = (size - remaining + got) / max(time.time() - t0, 1e-6) / 1e6
            print(f"{name}: {pct:.1f}% ({rate:.2f} MB/s)", flush=True)
    os.rename(tmp, path)
    print(f"DONE {name} ({size/1e6:.1f} MB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE, help="HF resolve URL prefix")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="target data/ directory")
    parser.add_argument("--shards", type=int, default=18)
    parser.add_argument("--files-parallel", type=int, default=6)
    parser.add_argument("--ranges-per-file", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    tasks = []
    for i in range(args.shards):
        name = f"train-{i:05d}-of-{args.shards:05d}.parquet"
        url = f"{args.base}/{name}?download=true"
        path = os.path.join(args.dest, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"skip {name} (candidate complete)", flush=True)
            continue
        tasks.append((url, path))
    if not tasks:
        print("ALL_DONE", flush=True)
        return
    with cf.ThreadPoolExecutor(max_workers=args.files_parallel) as ex:
        futs = [
            ex.submit(
                download_file,
                url,
                path,
                args.ranges_per_file,
                args.timeout,
                args.retries,
            )
            for url, path in tasks
        ]
        for fut in cf.as_completed(futs):
            fut.result()  # propagate failures
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

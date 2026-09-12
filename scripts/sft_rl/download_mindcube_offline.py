#!/usr/bin/env python3
"""Download only the MindCube parquet shards needed by mindcube_full."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import urllib.request


REPO = "oscarqjh/MindCube_lmmseval"
REVISION = "7dd2725d9bd4149f2aad00a9843f72a3824da003"


def url_for(index: int) -> str:
    name = f"combined-{index:05d}-of-00096.parquet"
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/full/{name}"


def target_for(root: Path, index: int) -> Path:
    return root / "full" / f"combined-{index:05d}-of-00096.parquet"


def content_length(index: int) -> int | None:
    request = urllib.request.Request(url_for(index), method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.headers.get("Content-Length")
    return int(value) if value is not None else None


def download_one(root: Path, index: int) -> tuple[int, str]:
    target = target_for(root, index)
    expected = content_length(index)
    if target.exists() and (expected is None or target.stat().st_size == expected):
        return index, "exists"

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url_for(index))
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    if expected is not None and partial.stat().st_size != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"size mismatch for {target.name}: expected {expected}, got {partial.stat().st_size}"
        )
    partial.replace(target)
    return index, "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=Path(
            "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/MindCube_lmmseval"
        ),
        type=Path,
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, args.output_root, index) for index in range(96)]
        for future in concurrent.futures.as_completed(futures):
            index, status = future.result()
            completed += 1
            print(f"[{completed:03d}/096] {status} combined-{index:05d}", flush=True)


if __name__ == "__main__":
    main()

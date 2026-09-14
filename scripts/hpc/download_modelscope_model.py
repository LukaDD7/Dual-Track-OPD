#!/usr/bin/env python3
"""Download a ModelScope model with resumable, verified file transfers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Namespace/repository, e.g. Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--api-base", default="https://modelscope.cn")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--required-free-gb",
        type=float,
        default=5.0,
        help="Minimum free space required after the planned download, excluding existing partial files",
    )
    parser.add_argument("--curl", default="curl")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file_list(args: argparse.Namespace) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"Revision": args.revision, "Root": ""})
    url = f"{args.api_base.rstrip('/')}/api/v1/models/{args.model}/repo/files?{query}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("Code") != 200:
        raise RuntimeError(f"ModelScope API failed for {args.model}: {payload}")
    files = payload.get("Data", {}).get("Files", [])
    blobs = [item for item in files if item.get("Type") == "blob"]
    if not blobs:
        raise RuntimeError(f"ModelScope returned no blob files for {args.model}")
    return blobs


def download_file(args: argparse.Namespace, item: dict[str, Any]) -> None:
    path = args.output / item["Path"]
    expected_size = int(item["Size"])
    expected_hash = str(item["Sha256"]).lower()
    if path.is_file() and path.stat().st_size == expected_size and sha256_file(path) == expected_hash:
        print(f"OK      {path} ({expected_size} bytes)", flush=True)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() and partial.stat().st_size > expected_size:
        partial.unlink()
    url = f"{args.api_base.rstrip('/')}/models/{args.model}/resolve/{args.revision}/{urllib.parse.quote(item['Path'])}"
    command = [
        args.curl,
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    subprocess.run(command, check=True)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for {path}: expected {expected_size}, got {partial.stat().st_size}")
    actual_hash = sha256_file(partial)
    if actual_hash != expected_hash:
        partial.unlink()
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected_hash}, got {actual_hash}")
    partial.replace(path)
    print(f"DOWNLOADED {path} ({expected_size} bytes)", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output = args.output.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else args.output / "modelscope_download_manifest.json"
    )
    files = fetch_file_list(args)
    planned_bytes = sum(int(item["Size"]) for item in files)
    free_bytes = shutil.disk_usage(args.output.parent).free
    required_bytes = planned_bytes + int(args.required_free_gb * (1024**3))
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"insufficient disk space: need {required_bytes / 1024**3:.2f} GiB including headroom, "
            f"have {free_bytes / 1024**3:.2f} GiB"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    for item in files:
        download_file(args, item)

    manifest = {
        "schema_version": 1,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "modelscope",
        "model": args.model,
        "revision": args.revision,
        "output": str(args.output),
        "total_bytes": planned_bytes,
        "file_count": len(files),
        "files": [
            {
                "path": item["Path"],
                "size": int(item["Size"]),
                "sha256": item["Sha256"],
                "revision": item.get("Revision"),
            }
            for item in files
        ],
        "verification": "all_file_sizes_and_sha256_hashes_matched",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MODELSCOPE DOWNLOAD PASSED: {args.output}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise

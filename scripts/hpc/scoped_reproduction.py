#!/usr/bin/env python3
"""Validate and record the pinned SCOPED expert-reproduction contract.

The upstream SCOPED repository and its datasets are external artifacts.  This
module does not vendor or execute them; it checks the pins that the HPC launch
scripts rely on and writes a small provenance manifest outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/external/scoped_expert_reproduction.yaml"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def inspect_repo(repo: Path, expected_commit: str) -> dict[str, Any]:
    if not (repo / ".git").exists():
        raise ValueError(f"Not a Git checkout: {repo}")
    commit = git_output(repo, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError(
            f"SCOPED checkout is {commit}, expected {expected_commit}"
        )
    status = git_output(repo, "status", "--short")
    if status:
        preview = "\n".join(status.splitlines()[:20])
        raise ValueError(f"SCOPED checkout is dirty:\n{preview}")
    return {
        "commit": commit,
        "dirty": False,
        "url": git_output(repo, "remote", "get-url", "origin"),
    }


def inspect_dataset(
    storage_root: Path,
    expected_revision: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    data_root = storage_root / "data" / "hf_dataset"
    manifest = data_root / "manifest.json"
    result: dict[str, Any] = {
        "present": manifest.is_file(),
        "root": str(data_root),
    }
    if not manifest.is_file():
        return result
    digest = file_sha256(manifest)
    if digest != expected_manifest_sha256:
        raise ValueError(
            f"SCOPED dataset manifest hash mismatch: {digest} != "
            f"{expected_manifest_sha256}"
        )
    payload = json.loads(manifest.read_text())
    if not payload.get("complete") or payload.get("limited_export"):
        raise ValueError("SCOPED dataset manifest is not a complete export")
    counts = payload.get("source_counts", {})
    result.update(
        {
            "manifest_sha256": digest,
            "revision": expected_revision,
            "source_counts": counts,
            "verified": True,
        }
    )
    return result


def validate_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())["scoped_reproduction"]
    caps = config["capabilities"]
    expected = {
        "capability1": (9499, 501, 5, 128, 4096),
        "capability2": (9500, 500, 3, 128, 4096),
        "capability3": (9500, 500, 3, 128, 8192),
        "capability4": (9500, 500, 3, 64, 4096),
    }
    if len(caps) != 4:
        raise ValueError("SCOPED contract must contain exactly four experts")
    for cap in caps:
        name = cap["name"]
        got = (
            cap["train_rows"],
            cap["val_rows"],
            cap["total_epochs"],
            cap["rollout_batch_size"],
            cap["max_response_length"],
        )
        if got != expected[name]:
            raise ValueError(f"Unexpected SCOPED contract for {name}: {got}")
        if cap["rollout_n"] != 8:
            raise ValueError(f"{name} must use rollout.n=8")
    pairs = config["execution_plan"]["parallel_pairs"]
    if set(pairs["a"]) | set(pairs["b"]) != set(expected):
        raise ValueError("Parallel pairs must cover all four capabilities once")
    if set(pairs["a"]) & set(pairs["b"]):
        raise ValueError("Parallel pairs must not overlap")
    return config


def build_record(
    config_path: Path,
    repo_root: Path,
    storage_root: Path,
    stage: str,
    gpu_group_a: str,
    gpu_group_b: str,
    command: str,
) -> dict[str, Any]:
    config = validate_config(config_path)
    upstream = inspect_repo(repo_root, config["upstream"]["commit"])
    dataset = inspect_dataset(
        storage_root,
        config["dataset"]["revision"],
        config["dataset"]["manifest_sha256"],
    )
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["name"],
        "stage": stage,
        "upstream": upstream,
        "dataset": dataset,
        "base_model": os.environ.get("BASE_MODEL", config["defaults"]["base_model"]),
        "student_model": os.environ.get("STUDENT_MODEL", config["defaults"]["base_model"]),
        "gpu_group_a": gpu_group_a,
        "gpu_group_b": gpu_group_b,
        "command": command,
        "capabilities": config["capabilities"],
        "execution_plan": config["execution_plan"],
    }


def write_record(record: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--gpu-group-a", default="0,1,2,3")
    parser.add_argument("--gpu-group-b", default="4,5,6,7")
    parser.add_argument("--command", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = build_record(
        args.config,
        args.repo_root,
        args.storage_root,
        args.stage,
        args.gpu_group_a,
        args.gpu_group_b,
        args.command,
    )
    write_record(record, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

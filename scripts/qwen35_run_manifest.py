#!/usr/bin/env python3
"""Create and finalize provenance for qwen3.5/3.6 distillation runs.

This helper intentionally uses only the Python standard library so it can run
inside the pinned server environment without importing the research package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGES = (
    "torch",
    "transformers",
    "vllm",
    "ray",
    "tensordict",
    "flash-attn",
    "flashinfer-python",
    "verl",
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run(command: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"unavailable: {exc}"
    output = (result.stdout + result.stderr).strip()
    return result.returncode, output


def _git(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        code, output = _run(["git", "-C", str(repo), *args])
        return output if code == 0 else "unavailable"

    tracked_status = git("status", "--porcelain", "--untracked-files=no")
    untracked = git("ls-files", "--others", "--exclude-standard")
    diff_code, diff = _run(["git", "-C", str(repo), "diff", "--binary", "HEAD"])
    return {
        "path": str(repo.resolve()),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tracked_status": tracked_status.splitlines() if tracked_status not in ("", "unavailable") else [],
        "tracked_dirty": tracked_status not in ("", "unavailable"),
        "untracked_count": 0 if untracked in ("", "unavailable") else len(untracked.splitlines()),
        "diff_sha256": _sha256_bytes(diff.encode()) if diff_code == 0 and diff else None,
    }


def _file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(resolved),
    }


def _model(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    config = resolved / "config.json"
    result: dict[str, Any] = {"path": str(resolved)}
    if config.is_file():
        result["config"] = _file(config)
        try:
            payload = json.loads(config.read_text(encoding="utf-8"))
            result["model_type"] = payload.get("model_type")
            result["architectures"] = payload.get("architectures")
        except (OSError, json.JSONDecodeError):
            result["config_parse_error"] = True
    return result


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _environment_manifest(env_prefix: Path) -> dict[str, Any] | None:
    candidates = (
        env_prefix / "share/dual-track-opd/environment_manifest.json",
        env_prefix / "share/dual-track-opd/va_opd_environment_manifest.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return _file(candidate)
    return None


def _hardware() -> dict[str, Any]:
    code, output = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_returncode": code,
        "gpus": output.splitlines() if code == 0 else [],
        "nvidia_smi_error": None if code == 0 else output,
    }


def _parse_config(values: list[str]) -> dict[str, str]:
    config: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--config must use KEY=VALUE, got {value!r}")
        key, item = value.split("=", 1)
        if not key:
            raise ValueError("--config key must be non-empty")
        config[key] = item
    return config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _start(argv: list[str]) -> int:
    if "--" in argv:
        split = argv.index("--")
        option_argv, backend_argv = argv[:split], argv[split + 1 :]
    else:
        option_argv, backend_argv = argv, []

    parser = argparse.ArgumentParser(prog="qwen35_run_manifest.py start")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--env-prefix", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--config", action="append", default=[])
    args = parser.parse_args(option_argv)

    metadata_dir = args.metadata_dir.resolve()
    metadata_dir.mkdir(parents=True, exist_ok=True)
    wrapper_config = _parse_config(args.config)
    launcher_argv = [str(args.launcher.resolve()), *json.loads(wrapper_config.pop("launcher_args_json", "[]"))]
    exact_backend_command = ["bash", "run_qwen3_5_4b_fsdp.sh", *backend_argv]

    launch_config = {
        "wrapper": wrapper_config,
        "launcher_argv": launcher_argv,
        "backend_argv": exact_backend_command,
        "backend_command_shell": shlex.join(exact_backend_command),
    }
    launch_config_path = metadata_dir / "resolved_launch_config.json"
    _write_json(launch_config_path, launch_config)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "started",
        "started_at": _now(),
        "finished_at": None,
        "returncode": None,
        "host": platform.node(),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "repo": _git(args.repo_root.resolve()),
        "backend": _git(args.backend_root.resolve()),
        "environment": {
            "prefix": str(args.env_prefix.resolve()),
            "manifest": _environment_manifest(args.env_prefix.resolve()),
            "versions": _versions(),
        },
        "hardware": _hardware(),
        "datasets": {
            "train": _file(args.train_file),
            "validation": _file(args.val_file),
        },
        "models": {
            "student": _model(args.student_model),
            "teacher": _model(args.teacher_model),
        },
        "tokenizer_alignment": _file(metadata_dir / "tokenizer_alignment.json")
        if (metadata_dir / "tokenizer_alignment.json").is_file()
        else None,
        "launch_config": {
            **_file(launch_config_path),
            "hydra_composed_config": None,
            "hydra_overrides": None,
            "train_log": str((metadata_dir / "train.log").resolve()),
        },
    }
    manifest_path = metadata_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Run manifest started: {manifest_path}")
    return 0


def _finish(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qwen35_run_manifest.py finish")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--returncode", type=int, required=True)
    args = parser.parse_args(argv)

    metadata_dir = args.metadata_dir.resolve()
    manifest_path = metadata_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hydra_root = metadata_dir / "hydra" / ".hydra"
    config_path = hydra_root / "config.yaml"
    overrides_path = hydra_root / "overrides.yaml"
    train_log = metadata_dir / "train.log"
    manifest["status"] = "completed" if args.returncode == 0 else "failed"
    manifest["finished_at"] = _now()
    manifest["returncode"] = args.returncode
    manifest["launch_config"]["hydra_composed_config"] = _file(config_path) if config_path.is_file() else None
    manifest["launch_config"]["hydra_overrides"] = _file(overrides_path) if overrides_path.is_file() else None
    manifest["launch_config"]["train_log"] = _file(train_log) if train_log.is_file() else None
    _write_json(manifest_path, manifest)
    print(f"Run manifest finalized: {manifest_path} ({manifest['status']})")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"start", "finish"}:
        print("usage: qwen35_run_manifest.py {start,finish} ...", file=sys.stderr)
        return 2
    command = sys.argv[1]
    return _start(sys.argv[2:]) if command == "start" else _finish(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())

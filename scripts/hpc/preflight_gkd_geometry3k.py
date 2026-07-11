#!/usr/bin/env python3
"""Validate and record a Geometry3K Qwen3-VL online-distillation run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {}
    return {}


def _model_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", ""))
    architectures = [str(item) for item in config.get("architectures", [])]
    identity = " ".join([model_type, *architectures]).lower()
    if "qwen3_vl" not in identity and "qwen3vl" not in identity:
        raise ValueError(f"expected a Qwen3-VL checkpoint at {path}, got {model_type!r}/{architectures!r}")
    return config


def _vocab_size(config: dict[str, Any]) -> int:
    value = config.get("vocab_size")
    if value is None and isinstance(config.get("text_config"), dict):
        value = config["text_config"].get("vocab_size")
    if not isinstance(value, int) or value < 2:
        raise ValueError("Qwen3-VL config is missing a valid text vocabulary size")
    return value


def _dataset_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset parquet not found: {path}")
    frame = pd.read_parquet(path)
    required = {
        "data_source",
        "prompt",
        "images",
        "reward_model",
        "extra_info",
        "question",
        "condition_inputs",
        "answer",
        "sample_uid",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows")
    sample = frame.iloc[0].to_dict()
    condition_inputs = _mapping(sample["condition_inputs"])
    for condition in ("full_image", "degraded_image"):
        image = _mapping(condition_inputs.get(condition))
        image_path = Path(str(image.get("path", "")))
        if not image_path.is_file():
            raise FileNotFoundError(f"{condition} asset missing for first row: {image_path}")
    stat = path.stat()
    fingerprint_payload = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": len(frame),
        "columns": sorted(frame.columns.tolist()),
        "first_uid": str(frame.iloc[0]["sample_uid"]),
        "last_uid": str(frame.iloc[-1]["sample_uid"]),
    }
    encoded = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    fingerprint_payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return fingerprint_payload


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--objective", choices=("gkd", "va_opd", "va_opd_jsd"), required=True)
    parser.add_argument("--teacher-gpu", type=int, required=True)
    parser.add_argument("--train-gpus", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--env-path", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument("--max-response-length", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--ignore-eos", choices=("true", "false"), required=True)
    parser.add_argument("--save-freq", type=int, required=True)
    parser.add_argument("--teacher-port", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config-reference", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    backend = args.backend_dir.resolve()
    train_gpus = [int(value) for value in args.train_gpus.split(",") if value.strip()]
    if not train_gpus or len(set(train_gpus)) != len(train_gpus):
        raise ValueError("--train-gpus must contain unique GPU indices")
    if args.teacher_gpu in train_gpus:
        raise ValueError("teacher GPU must be disjoint from training GPUs")
    if args.steps < 1 or args.train_batch_size < 1 or args.rollout_n < 1 or args.top_k < 1:
        raise ValueError("steps, train batch size, rollout n, and top-k must be positive")
    if args.max_prompt_length < 1 or args.max_response_length < 1 or args.save_freq < 1:
        raise ValueError("sequence lengths and save frequency must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0 or args.learning_rate <= 0:
        raise ValueError("GPU memory utilization must be in (0,1) and learning rate must be positive")
    if args.temperature <= 0 or not 0.0 <= args.top_p <= 1.0:
        raise ValueError("temperature must be positive and top_p must be in [0,1]")
    if args.train_batch_size % len(train_gpus):
        raise ValueError("train batch size must be divisible by the training GPU count")

    actor_file = backend / "verl/workers/actor/dp_actor.py"
    trainer_file = backend / "verl/trainer/ppo/ray_trainer.py"
    actor_text = actor_file.read_text(encoding="utf-8")
    trainer_text = trainer_file.read_text(encoding="utf-8")
    for marker, text in (
        ("compute_verl_fc_opd_actor_loss", actor_text),
        ('"gkd", "gkd_forward"', actor_text),
        ("fc_opd_hook_fqn", trainer_text),
    ):
        if marker not in text:
            raise RuntimeError(f"verl overlay marker missing: {marker}")

    student_config = _model_config(args.student_model.resolve())
    teacher_config = _model_config(args.teacher_model.resolve())
    student_vocab_size = _vocab_size(student_config)
    teacher_vocab_size = _vocab_size(teacher_config)
    if student_vocab_size != teacher_vocab_size:
        raise ValueError("student and teacher vocab_size differ; exact-token GKD is unsafe")
    if args.top_k >= student_vocab_size:
        raise ValueError("teacher top-k must be smaller than the shared vocabulary")

    train_summary = _dataset_summary(args.train_data.resolve())
    val_summary = _dataset_summary(args.val_data.resolve())
    prepared_manifest_path = args.train_data.resolve().with_suffix(".manifest.json")
    prepared_manifest = None
    if prepared_manifest_path.is_file():
        prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
    config_reference = args.config_reference.resolve()
    config_bytes = config_reference.read_bytes()
    manifest = {
        "objective": args.objective,
        "repo_commit": _git(repo, "rev-parse", "HEAD"),
        "repo_dirty": bool(dirty),
        "repo_tracked_changes": dirty.splitlines(),
        "backend_commit": _git(backend, "rev-parse", "HEAD"),
        "backend_dirty": bool(_git(backend, "status", "--porcelain", "--untracked-files=no")),
        "config_reference": str(config_reference),
        "config_reference_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "student_model": str(args.student_model.resolve()),
        "teacher_model": str(args.teacher_model.resolve()),
        "model_type": student_config.get("model_type"),
        "vocab_size": student_vocab_size,
        "train_dataset": train_summary,
        "val_dataset": val_summary,
        "prepared_dataset_manifest": prepared_manifest,
        "teacher_gpu": args.teacher_gpu,
        "train_gpus": train_gpus,
        "steps": args.steps,
        "train_batch_size": args.train_batch_size,
        "rollout_n": args.rollout_n,
        "top_k": args.top_k,
        "env_path": str(args.env_path.resolve()),
        "conditions": ["full"] if args.objective == "gkd" else ["full", "degraded"],
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "ignore_eos": args.ignore_eos == "true",
        "save_freq": args.save_freq,
        "teacher_port": args.teacher_port,
        "run_dir": str(args.run_dir.resolve()),
        "train_log": str((args.run_dir / "train.log").resolve()),
        "teacher_log": str((args.run_dir / "teacher.log").resolve()),
        "validation_output_dir": str((args.run_dir / "validation").resolve()),
        "versions": {
            package: _version(package)
            for package in ("torch", "transformers", "vllm", "ray", "tensordict", "flashinfer-python")
        },
    }
    manifest["resolved_config_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = args.run_dir / "run_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"GKD Geometry3K preflight: PASS ({output})")


if __name__ == "__main__":
    main()

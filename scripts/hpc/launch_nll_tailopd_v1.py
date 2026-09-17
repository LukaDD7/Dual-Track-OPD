#!/usr/bin/env python3
"""Config-driven launcher and provenance recorder for NLL-TailOPD v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def git_state(root: Path) -> dict[str, Any]:
    status = run_git(root, "status", "--porcelain").splitlines()
    return {
        "commit": run_git(root, "rev-parse", "HEAD"),
        "dirty": bool(status),
        "tracked_changes": sum(not line.startswith("??") for line in status),
        "untracked_entries": sum(line.startswith("??") for line in status),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a YAML mapping")
    return config


def patch_is_applied(backend_root: Path, patch_path: Path) -> bool:
    result = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(patch_path)],
        cwd=backend_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def validate_inputs(config: dict[str, Any], run_name: str) -> None:
    backend = Path(config["backend"]["root"])
    launcher = backend / config["backend"]["launcher"]
    patch = REPO_ROOT / config["backend"]["patch"]
    python_bin = Path(config["environment"]["python"])
    paths = config["paths"]

    for label, path in (
        ("backend", backend),
        ("launcher", launcher),
        ("patch", patch),
        ("python", python_bin),
        ("student model", Path(paths["student_model"])),
        ("teacher model", Path(paths["teacher_model"])),
        ("train file", Path(paths["train_file"])),
        ("val file", Path(paths["val_file"])),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if run_name not in config["runs"]:
        raise ValueError(f"unknown run: {run_name}; expected one of {sorted(config['runs'])}")
    if int(config["training"]["rollout_n"]) < 2:
        raise ValueError("NLL-TailOPD requires rollout_n >= 2 for within-prompt normalization")
    if config["training"]["loss_agg_mode"] != "seq-mean-token-mean":
        raise ValueError("TailOPD v1 requires loss_agg_mode=seq-mean-token-mean")
    hardware = config.get("hardware", {})
    actor_gpus = int(hardware.get("ngpus_per_node", 0))
    teacher_gpus = int(hardware.get("teacher_world_size", 0))
    visible_gpu_count = len(str(hardware.get("visible_gpus", "")).split(","))
    if actor_gpus + teacher_gpus != 4:
        raise ValueError(
            f"TailOPD v1 smoke expects 4 total GPUs; got actor={actor_gpus}, teacher={teacher_gpus}"
        )
    if visible_gpu_count != 4:
        raise ValueError(f"TailOPD v1 smoke expects 4 visible GPUs, got {visible_gpu_count}")


def build_environment(config: dict[str, Any], run_name: str, stage: str) -> dict[str, str]:
    training = config["training"]
    stage_config = config["stages"][stage]
    train_batch_size = stage_config.get("train_batch_size", training["train_batch_size"])
    ppo_mini_batch_size = stage_config.get("ppo_mini_batch_size", train_batch_size)
    paths = config["paths"]
    tail_enabled = bool(config["runs"][run_name]["tail_opd_enabled"])
    tail = config["tail_opd"]
    hardware = config["hardware"]
    experiment_name = os.environ.get("NLL_TAILOPD_EXPERIMENT_NAME")
    if experiment_name is None:
        experiment_name = f"{run_name}_{stage}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    env = os.environ.copy()
    python_bin = Path(config["environment"]["python"])
    python_env_bin = python_bin.parent
    python_env_root = python_env_bin.parent
    env.update(
        {
            "STUDENT_MODEL": str(paths["student_model"]),
            "TEACHER_MODEL": str(paths["teacher_model"]),
            "TRAIN_FILE": str(paths["train_file"]),
            "VAL_FILE": str(paths["val_file"]),
            "ROLLOUT_N": str(training["rollout_n"]),
            "TRAIN_BATCH_SIZE": str(train_batch_size),
            "PPO_MINI_BATCH_SIZE": str(ppo_mini_batch_size),
            "NGPUS_PER_NODE": str(hardware["ngpus_per_node"]),
            "TEACHER_WORLD_SIZE": str(hardware["teacher_world_size"]),
            "TEACHER_TP": str(hardware["teacher_tp"]),
            "TEACHER_EP": str(hardware["teacher_ep"]),
            "ROLLOUT_TP": str(hardware["rollout_tp"]),
            "ROLLOUT_NUM_WORKERS": str(hardware["rollout_num_workers"]),
            "CUDA_VISIBLE_DEVICES": str(hardware["visible_gpus"]),
            "TOTAL_EPOCHS": str(stage_config["total_epochs"]),
            "TEST_FREQ": str(stage_config["eval_frequency"]),
            "SAVE_FREQ": str(stage_config["checkpoint_frequency"]),
            "ACTOR_LOSS_AGG_MODE": str(training["loss_agg_mode"]),
            "DISTILLATION_LOSS_MODE": str(training["distillation_loss_mode"]),
            "USE_POLICY_GRADIENT": str(bool(training["use_policy_gradient"])).lower(),
            "USE_TASK_REWARDS": str(bool(training["use_task_rewards"])).lower(),
            "TAIL_OPD_ENABLED": str(tail_enabled).lower(),
            "TAIL_OPD_TEMPERATURE": str(tail["temperature"]),
            "TAIL_OPD_EPS": str(tail["eps"]),
            "PROJECT_NAME": "nll_tailopd_v1",
            "EXPERIMENT_NAME": experiment_name,
            "PYTHON_BIN": str(python_bin),
            "PYTHONUNBUFFERED": "1",
        }
    )
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    # The external verl launcher invokes plain ``python3``. Force it to resolve
    # to the pinned training environment even when this launcher itself runs
    # from a lightweight environment that provides PyYAML/pytest.
    env["PATH"] = f"{python_env_bin}{os.pathsep}{env.get('PATH', '')}"
    env["CONDA_PREFIX"] = str(python_env_root)
    env["CONDA_DEFAULT_ENV"] = python_env_root.name
    return env


def build_extra_args(config: dict[str, Any], stage: str) -> list[str]:
    stage_config = config["stages"][stage]
    if "total_steps" in stage_config:
        return [f"trainer.total_training_steps={int(stage_config['total_steps'])}"]
    return []


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/experiment/nll_tailopd_geometry3k_v1.yaml")
    parser.add_argument("--run", choices=("tail_opd",), default="tail_opd")
    parser.add_argument("--stage", choices=("smoke", "stability", "formal"), default="smoke")
    parser.add_argument("--apply-patch", action="store_true", help="apply the pinned backend patch if needed")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the command without launching")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_inputs(config, args.run)

    backend_root = Path(config["backend"]["root"])
    patch_path = REPO_ROOT / config["backend"]["patch"]
    if not patch_is_applied(backend_root, patch_path):
        if not args.apply_patch:
            raise RuntimeError("backend patch is not applied; rerun with --apply-patch")
        apply_script = REPO_ROOT / "scripts/hpc/apply_nll_tailopd_v1_patch.sh"
        subprocess.check_call(["bash", str(apply_script)], cwd=REPO_ROOT)
        if not patch_is_applied(backend_root, patch_path):
            raise RuntimeError("patch application did not verify")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(config["paths"]["output_root"])
    run_dir = output_root / f"{timestamp}_{args.run}_{args.stage}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "train.log"
    manifest_path = run_dir / "manifest.json"

    environment = build_environment(config, args.run, args.stage)
    extra_args = build_extra_args(config, args.stage)
    launcher = str(backend_root / config["backend"]["launcher"])
    command = ["bash", launcher, *extra_args]

    dataset_hashes = {
        label: sha256_file(Path(config["paths"][f"{label}_file"]))
        for label in ("train", "val")
    }
    manifest = {
        "schema_version": 1,
        "experiment": config["experiment"]["name"],
        "run": args.run,
        "stage": args.stage,
        "created_utc": timestamp,
        "repo": git_state(REPO_ROOT),
        "backend": git_state(backend_root),
        "backend_patch": {
            "path": str(patch_path),
            "sha256": sha256_file(patch_path),
            "applied": True,
        },
        "config_path": str(args.config),
        "resolved_config": config,
        "dataset_manifest_hashes": dataset_hashes,
        "model_checkpoint": str(config["paths"]["student_model"]),
        "teacher_model": str(config["paths"]["teacher_model"]),
        "command": command,
        "environment_overrides": {
            key: value for key, value in environment.items() if key in {
                "STUDENT_MODEL", "TEACHER_MODEL", "TRAIN_FILE", "VAL_FILE",
                "ROLLOUT_N", "TRAIN_BATCH_SIZE", "TOTAL_EPOCHS", "TEST_FREQ",
                "SAVE_FREQ", "PPO_MINI_BATCH_SIZE", "ACTOR_LOSS_AGG_MODE", "DISTILLATION_LOSS_MODE",
                "USE_POLICY_GRADIENT", "USE_TASK_REWARDS", "TAIL_OPD_ENABLED",
                "TAIL_OPD_TEMPERATURE", "TAIL_OPD_EPS", "PROJECT_NAME",
                "EXPERIMENT_NAME", "PYTHON_BIN", "NGPUS_PER_NODE",
                "TEACHER_WORLD_SIZE", "TEACHER_TP", "TEACHER_EP",
                "ROLLOUT_TP", "ROLLOUT_NUM_WORKERS", "CUDA_VISIBLE_DEVICES",
            }
        },
        "raw_output_path": str(run_dir),
        "train_log": str(log_path),
        "status": "prepared",
    }
    write_json(manifest_path, manifest)

    print(f"Run directory: {run_dir}")
    print("Command:", " ".join(command))
    if args.dry_run:
        manifest["status"] = "dry_run"
        write_json(manifest_path, manifest)
        return 0

    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=backend_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()

    manifest["status"] = "completed" if return_code == 0 else "failed"
    manifest["return_code"] = return_code
    write_json(manifest_path, manifest)
    print(f"Return code: {return_code}")
    print(f"Log: {log_path}")
    print(f"Manifest: {manifest_path}")
    return return_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

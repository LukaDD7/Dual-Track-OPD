#!/usr/bin/env python3
"""Fail-fast validation and manifest creation for native verl VA-OPD."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

EXPECTED_BACKEND_COMMIT = "e003163181731412595257a72ec173071efb125f"

# ── cu128 (verl-supported vLLM 0.12.0) ──────────────────────────────────
EXPECTED_VLLM_SOURCE_COMMIT_CU128 = "4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e"
EXPECTED_BACKEND_CHANGES_CU128 = {
    "verl/experimental/agent_loop/agent_loop.py",
    "verl/trainer/distillation/losses.py",
    "verl/trainer/ppo/ray_trainer.py",
}
EXPECTED_RUNTIME_CU128 = ("2.9.0", "12.8", "0.12.0+cu128", "4.57.3")
EXPECTED_BUILD_KIND_CU128 = "cpu-source-build-cu128-h200-sm90"
EXPECTED_VERSIONS_CU128 = {
    "vllm": "0.12.0+cu128",
    "transformers": "4.57.3",
    "tensordict": "0.10.0",
    "flash-attn": "2.8.3",
    "flashinfer-python": "0.5.3",
}

# ── cu132 (source-built vLLM 0.25.1) ────────────────────────────────────
EXPECTED_VLLM_SOURCE_COMMIT_CU132 = "752a3a504485790a2e8491cacbb35c137339ad34"
EXPECTED_BACKEND_CHANGES_CU132 = {
    "verl/experimental/agent_loop/agent_loop.py",
    "verl/trainer/distillation/losses.py",
    "verl/trainer/ppo/ray_trainer.py",
    "verl/utils/vllm/utils.py",           # LoRA import: vllm.lora.lora_model (vLLM >= 0.25)
    "verl/trainer/constants_ppo.py",      # Ray runtime_env LD_LIBRARY_PATH forwarding
}
EXPECTED_RUNTIME_CU132 = ("2.13.0", "13.2", "0.25.1+cu132", "5.14.1")
EXPECTED_BUILD_KIND_CU132 = "cpu-source-build-cu132-h200-sm90"
EXPECTED_VERSIONS_CU132 = {
    "vllm": "0.25.1+cu132",
    "transformers": "5.14.1",
    "tensordict": "0.10.0",
    "flash-attn": "2.8.3",
    "flashinfer-python": "0.6.13",
}

# Patch SHA-256 are only validated for cu128 (exact known-good overlay).
EXPECTED_PATCHED_FILE_SHA256 = {
    "verl/experimental/agent_loop/agent_loop.py": "97be16d52f92ed6dee42d1cbdbe7200b8105e842f86a229f7acc9ace766602b8",
    "verl/trainer/distillation/losses.py": "41f8959296620b0e08bed59719a405e7d7b835653ff86a17340ed495bdb5197b",
    "verl/trainer/ppo/ray_trainer.py": "23bacdc7b537cc73c3677dbc478984c9c9b6c6ea31f1514c31cd6ca2006cd543",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).rstrip("\n")


def mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "tolist"):
        return mapping(value.tolist())
    if isinstance(value, (list, tuple)):
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {}
    return {}


def model_summary(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    identity = " ".join(
        [str(config.get("model_type", "")), *[str(item) for item in config.get("architectures", [])]]
    ).lower()
    if "qwen3_vl" not in identity and "qwen3vl" not in identity:
        raise ValueError(f"expected Qwen3-VL checkpoint at {path}, got {identity!r}")
    vocab_size = config.get("vocab_size")
    if vocab_size is None:
        vocab_size = mapping(config.get("text_config")).get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size < 2:
        raise ValueError(f"invalid vocabulary size in {config_path}")
    tokenizer_files = []
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        candidate = path / name
        if candidate.is_file():
            tokenizer_files.append({"name": name, "sha256": sha256_file(candidate)})
    if not tokenizer_files:
        raise FileNotFoundError(f"no tokenizer metadata found under {path}")
    return {
        "path": str(path.resolve()),
        "model_type": config.get("model_type"),
        "vocab_size": vocab_size,
        "config_sha256": sha256_file(config_path),
        "tokenizer_files": tokenizer_files,
    }


def tokenizer_identity(summary: dict[str, Any]) -> dict[str, str]:
    """Return the files that define token-ID semantics, excluding cosmetic config drift."""

    hashes = {item["name"]: item["sha256"] for item in summary["tokenizer_files"]}
    if "tokenizer.json" in hashes:
        return {"tokenizer.json": hashes["tokenizer.json"]}
    fallback = {name: hashes[name] for name in ("vocab.json", "merges.txt") if name in hashes}
    if not fallback:
        raise ValueError(f"cannot establish tokenizer identity for {summary['path']}")
    return fallback


def dataset_summary(path: Path, *, audit_all_images: bool) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset parquet is missing: {path}")
    frame = pd.read_parquet(path)
    required = {"prompt", "images", "extra_info", "condition_inputs", "sample_uid"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing VA-OPD columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} has no rows")

    rows = range(len(frame)) if audit_all_images else range(min(8, len(frame)))
    degraded_modes: set[str] = set()
    for row_index in rows:
        row = frame.iloc[row_index]
        condition_inputs = mapping(row["condition_inputs"])
        full = mapping(condition_inputs.get("full_image"))
        degraded = mapping(condition_inputs.get("degraded_image"))
        full_path = Path(str(full.get("path", ""))).expanduser()
        degraded_path = Path(str(degraded.get("path", ""))).expanduser()
        if not full_path.is_file() or not degraded_path.is_file():
            raise FileNotFoundError(
                f"row {row_index} image pair is missing: full={full_path}, degraded={degraded_path}"
            )
        with Image.open(full_path) as full_image, Image.open(degraded_path) as degraded_image:
            if full_image.size != degraded_image.size:
                raise ValueError(
                    f"row {row_index} image dimensions differ: full={full_image.size}, degraded={degraded_image.size}"
                )
        transform = mapping(degraded.get("transform"))
        source_transform = mapping(transform.get("source_transform"))
        mode = str(transform.get("degraded_mode") or source_transform.get("degraded_mode") or "")
        if mode:
            degraded_modes.add(mode)
    if degraded_modes and degraded_modes != {"lowres_10pct_nearest"}:
        raise ValueError(f"unexpected degradation modes in {path}: {sorted(degraded_modes)}")

    return {
        "path": str(path.resolve()),
        "rows": len(frame),
        "columns": sorted(frame.columns.tolist()),
        "sha256": sha256_file(path),
        "image_rows_audited": len(list(rows)),
        "degraded_modes": sorted(degraded_modes),
        "first_sample_uid": str(frame.iloc[0]["sample_uid"]),
        "last_sample_uid": str(frame.iloc[-1]["sample_uid"]),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path, required=True)
    parser.add_argument("--env-prefix", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--objective", choices=("opd", "va_opd"), required=True)
    parser.add_argument("--visible-gpus", required=True)
    parser.add_argument("--actor-gpus", type=int, required=True)
    parser.add_argument("--teacher-gpus", type=int, required=True)
    parser.add_argument("--teacher-tp", type=int, required=True)
    parser.add_argument("--prompt-batch-size", type=int, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument("--max-response-length", type=int, required=True)
    parser.add_argument("--config-reference", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-all-images", action="store_true")
    parser.add_argument("--allow-system-nvcc", action="store_true")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    backend = args.backend_dir.resolve()
    env_prefix = args.env_prefix.resolve()
    visible_gpus = [int(value) for value in args.visible_gpus.split(",") if value.strip()]
    if len(visible_gpus) != len(set(visible_gpus)):
        raise ValueError("visible GPU indices must be unique")
    required_gpus = args.actor_gpus + args.teacher_gpus
    if len(visible_gpus) != required_gpus:
        raise ValueError(
            f"visible GPU count must equal actor+teacher pools ({required_gpus}), got {len(visible_gpus)}"
        )
    if args.teacher_gpus % args.teacher_tp:
        raise ValueError("teacher GPU pool must be divisible by teacher tensor parallel size")
    if args.rollout_n != 4:
        raise ValueError("paper-faithful OPD/VA-OPD comparison requires K=4")
    if args.prompt_batch_size % args.actor_gpus:
        raise ValueError("prompt batch size must be divisible by actor GPU count")
    if args.max_prompt_length < 1 or args.max_response_length < 2:
        raise ValueError("sequence lengths are invalid")

    if not str(Path(sys.executable).resolve()).startswith(str(env_prefix) + "/"):
        raise RuntimeError(f"preflight must run inside {env_prefix}; got {sys.executable}")
    backend_commit = git(backend, "rev-parse", "HEAD")
    if backend_commit != EXPECTED_BACKEND_COMMIT:
        raise RuntimeError(f"backend commit drift: expected {EXPECTED_BACKEND_COMMIT}, got {backend_commit}")
    changed = {
        line[3:]
        for line in git(backend, "status", "--porcelain", "--untracked-files=no").splitlines()
        if len(line) > 3
    }
    for relative, marker in (
        ("verl/experimental/agent_loop/agent_loop.py", "teacher_degraded_logprobs"),
        ("verl/trainer/distillation/losses.py", "register_native_verl_loss"),
        ("verl/trainer/ppo/ray_trainer.py", "prepare_native_verl_batch"),
    ):
        if marker not in (backend / relative).read_text(encoding="utf-8"):
            raise RuntimeError(f"native VA-OPD patch marker {marker!r} is missing from {relative}")
    for relative, expected_sha256 in EXPECTED_PATCHED_FILE_SHA256.items():
        actual_sha256 = sha256_file(backend / relative)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"patched backend file drift: {relative} expected {expected_sha256}, got {actual_sha256}"
            )

    import torch

    # ── Read manifest first to determine build kind ─────────────────────
    environment_manifest_path = env_prefix / "share/dual-track-opd/va_opd_environment_manifest.json"
    if not environment_manifest_path.is_file():
        raise FileNotFoundError(f"environment build manifest is missing: {environment_manifest_path}")
    environment_manifest = json.loads(environment_manifest_path.read_text(encoding="utf-8"))
    build_kind = environment_manifest.get("build_kind")

    if build_kind == EXPECTED_BUILD_KIND_CU128:
        expected_vllm_source_commit = EXPECTED_VLLM_SOURCE_COMMIT_CU128
        expected_backend_changes = EXPECTED_BACKEND_CHANGES_CU128
        expected_runtime = EXPECTED_RUNTIME_CU128
        expected_versions = dict(EXPECTED_VERSIONS_CU128)
        expected_torch_cuda = "12.8"
        check_backend_sha = True
    elif build_kind == EXPECTED_BUILD_KIND_CU132:
        expected_vllm_source_commit = EXPECTED_VLLM_SOURCE_COMMIT_CU132
        expected_backend_changes = EXPECTED_BACKEND_CHANGES_CU132
        expected_runtime = EXPECTED_RUNTIME_CU132
        expected_versions = dict(EXPECTED_VERSIONS_CU132)
        expected_torch_cuda = "13.2"
        check_backend_sha = False  # cu132 patches differ per-build; validate markers instead
    else:
        raise RuntimeError(
            f"unrecognized environment build kind: {build_kind!r}. "
            f"Expected {EXPECTED_BUILD_KIND_CU128!r} or {EXPECTED_BUILD_KIND_CU132!r}."
        )

    # ── Validate backend ─────────────────────────────────────────────────
    if changed != expected_backend_changes:
        raise RuntimeError(
            f"backend patch scope drift for {build_kind}: "
            f"expected {sorted(expected_backend_changes)}, got {sorted(changed)}"
        )
    for relative, marker in (
        ("verl/experimental/agent_loop/agent_loop.py", "teacher_degraded_logprobs"),
        ("verl/trainer/distillation/losses.py", "register_native_verl_loss"),
        ("verl/trainer/ppo/ray_trainer.py", "prepare_native_verl_batch"),
    ):
        if marker not in (backend / relative).read_text(encoding="utf-8"):
            raise RuntimeError(f"native VA-OPD patch marker {marker!r} is missing from {relative}")

    if check_backend_sha:
        for relative, expected_sha256 in EXPECTED_PATCHED_FILE_SHA256.items():
            actual_sha256 = sha256_file(backend / relative)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"patched backend file drift: {relative} expected {expected_sha256}, got {actual_sha256}"
                )
    else:
        # cu132: verify LoRA import patch is applied (vllm >= 0.25 requirement)
        lora_utils = backend / "verl/utils/vllm/utils.py"
        if "vllm.lora.lora_model" not in lora_utils.read_text(encoding="utf-8"):
            raise RuntimeError("cu132 backend is missing the LoRA import patch (vllm.lora.lora_model)")
        ray_constants = backend / "verl/trainer/constants_ppo.py"
        if "LD_LIBRARY_PATH" not in ray_constants.read_text(encoding="utf-8"):
            raise RuntimeError("cu132 backend is missing the Ray LD_LIBRARY_PATH forwarding patch")

    # ── Validate runtime versions ────────────────────────────────────────
    actual_runtime = (
        torch.__version__.split("+")[0],
        torch.version.cuda,
        package_version("vllm").split("+")[0],
        package_version("transformers").split("+")[0],
    )
    expected_runtime_normalized = tuple(value.split("+")[0] for value in expected_runtime)
    if actual_runtime != expected_runtime_normalized:
        raise RuntimeError(
            f"unsupported VA-OPD runtime for {build_kind}: "
            f"torch={actual_runtime[0]}, CUDA={actual_runtime[1]}, vllm={actual_runtime[2]}, "
            f"transformers={actual_runtime[3]}; expected {expected_runtime}."
        )
    for package, expected in expected_versions.items():
        actual = package_version(package).split("+")[0]
        expected_normalized = expected.split("+")[0]
        if actual != expected_normalized:
            raise RuntimeError(f"{package} version drift: expected {expected}, got {actual}")

    nvcc = shutil.which("nvcc")
    if nvcc and Path(nvcc).resolve().as_posix().startswith("/usr/") and not args.allow_system_nvcc:
        raise RuntimeError(f"system nvcc is forbidden for this pipeline: {Path(nvcc).resolve()}")

    # ── Validate environment manifest provenance ─────────────────────────
    if environment_manifest.get("vllm_source_commit") != expected_vllm_source_commit:
        raise RuntimeError(
            f"vLLM source provenance drift for {build_kind}: expected "
            f"{expected_vllm_source_commit}, got {environment_manifest.get('vllm_source_commit')}"
        )
    if environment_manifest.get("torch_cuda") != expected_torch_cuda:
        raise RuntimeError(
            f"environment manifest CUDA mismatch for {build_kind}: "
            f"expected {expected_torch_cuda}, got {environment_manifest.get('torch_cuda')!r}"
        )
    if environment_manifest.get("torch_cuda_arch_list") != "9.0":
        raise RuntimeError(
            f"environment manifest was not built for H200 SM90: {environment_manifest.get('torch_cuda_arch_list')!r}"
        )
    if environment_manifest.get("verl_backend_commit") != EXPECTED_BACKEND_COMMIT:
        raise RuntimeError(
            "environment was built against the wrong verl backend: "
            f"{environment_manifest.get('verl_backend_commit')!r}"
        )
    if build_kind == EXPECTED_BUILD_KIND_CU128:
        constraints_path = repo / "configs/environment/verl_va_opd_e003_cu128.constraints.txt"
        manifest_torch_version = "2.9.0"
    else:
        constraints_path = repo / "configs/environment/verl_va_opd_e003_cu132.constraints.txt"
        manifest_torch_version = "2.13.0"
    if environment_manifest.get("constraints_sha256") != sha256_file(constraints_path):
        raise RuntimeError("environment constraints hash differs from the checked-out project")
    manifest_packages = mapping(environment_manifest.get("packages"))
    for package, expected in {"torch": manifest_torch_version, **expected_versions}.items():
        if manifest_packages.get(package) is None:
            continue
        actual_manifest = str(manifest_packages.get(package) or "").split("+")[0]
        expected_manifest = expected.split("+")[0]
        if actual_manifest != expected_manifest:
            raise RuntimeError(
                f"environment manifest {package} drift: expected {expected}, got {manifest_packages.get(package)}"
            )
    vllm_wheel = Path(str(environment_manifest.get("vllm_wheel", ""))).expanduser()
    if not vllm_wheel.is_file():
        raise FileNotFoundError(f"source-built vLLM wheel is missing: {vllm_wheel}")
    actual_wheel_sha256 = sha256_file(vllm_wheel)
    if actual_wheel_sha256 != environment_manifest.get("vllm_wheel_sha256"):
        raise RuntimeError(
            "source-built vLLM wheel hash drift: expected "
            f"{environment_manifest.get('vllm_wheel_sha256')}, got {actual_wheel_sha256}"
        )

    student = model_summary(args.student_model.resolve())
    teacher = model_summary(args.teacher_model.resolve())
    if student["vocab_size"] != teacher["vocab_size"]:
        raise ValueError("student and teacher vocabularies differ; sampled-token reverse KL is invalid")
    student_tokenizer = tokenizer_identity(student)
    teacher_tokenizer = tokenizer_identity(teacher)
    if student_tokenizer != teacher_tokenizer:
        raise ValueError(
            "student and teacher tokenizer-ID mappings differ; exact sampled-token reverse KL is invalid"
        )
    train = dataset_summary(args.train_data.resolve(), audit_all_images=args.audit_all_images)
    val = dataset_summary(args.val_data.resolve(), audit_all_images=args.audit_all_images)
    config_reference = args.config_reference.resolve()
    patch_path = repo / "patches/verl/va_opd_native_e0031631.patch"

    repo_dirty = git(repo, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all")
    manifest = {
        "schema_version": 1,
        "objective": args.objective,
        "repo_commit": git(repo, "rev-parse", "HEAD"),
        "repo_dirty": bool(repo_dirty),
        "repo_tracked_changes": repo_dirty.splitlines(),
        "backend_commit": backend_commit,
        "backend_changed_files": sorted(changed),
        "backend_patch": str(patch_path.resolve()),
        "backend_patch_sha256": sha256_file(patch_path),
        "config_reference": str(config_reference),
        "config_reference_sha256": sha256_file(config_reference),
        "environment_prefix": str(env_prefix),
        "environment_build_manifest": str(environment_manifest_path.resolve()),
        "environment_build_manifest_sha256": sha256_file(environment_manifest_path),
        "vllm_source_commit": environment_manifest["vllm_source_commit"],
        "python": sys.version,
        "packages": {
            name: package_version(name)
            for name in ("torch", "vllm", "transformers", "tensordict", "ray", "numpy", "pyarrow")
        },
        "torch_cuda": torch.version.cuda,
        "nvcc": str(Path(nvcc).resolve()) if nvcc else None,
        "student": student,
        "teacher": teacher,
        "shared_tokenizer_identity": student_tokenizer,
        "train_dataset": train,
        "val_dataset": val,
        "visible_gpus": visible_gpus,
        "actor_gpus": args.actor_gpus,
        "teacher_gpus": args.teacher_gpus,
        "teacher_tensor_parallel_size": args.teacher_tp,
        "prompt_batch_size": args.prompt_batch_size,
        "rollouts_per_prompt": args.rollout_n,
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
        "run_dir": str(args.run_dir.resolve()),
        "raw_outputs_outside_git": str((args.run_dir / "rollouts").resolve()),
        "summary_metrics": str((args.run_dir / "result.json").resolve()),
    }
    manifest["resolved_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = args.run_dir / "run_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Native VA-OPD preflight: PASS ({output})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify repository-local parts of a backend contract without touching VERL."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = REPO_ROOT / "configs/backend/verl_qwen35_v090_cu132.yaml"


class ContractError(ValueError):
    """Raised when a backend contract is incomplete or inconsistent."""


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"{label} does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a YAML mapping: {path}")
    return value


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ContractError(f"contract field {key!r} must be a mapping")
    return value


def _nonempty_string(parent: dict[str, Any], key: str, prefix: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"contract field {prefix}.{key} must be a non-empty string")
    return value


def _repo_path(repo_root: Path, value: str, label: str) -> Path:
    if Path(value).is_absolute():
        raise ContractError(f"{label} must be repository-relative, got {value}")
    root = repo_root.resolve()
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ContractError(f"{label} escapes repository root: {value}") from error
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_contract(contract_path: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return locally verified contract facts or raise ``ContractError``."""
    contract = _load_mapping(contract_path, "backend contract")
    if contract.get("schema_version") != 1:
        raise ContractError("contract schema_version must equal 1")
    name = _nonempty_string(contract, "name", "contract")
    status = _nonempty_string(contract, "verification_status", "contract")
    if status != "TO_BE_CONFIRMED_ON_SERVER":
        raise ContractError("verification_status must remain TO_BE_CONFIRMED_ON_SERVER before audit")

    upstream = _mapping(contract, "upstream")
    _nonempty_string(upstream, "repo", "upstream")
    base_commit = _nonempty_string(upstream, "nominal_base_commit", "upstream")
    if len(base_commit) != 40 or any(character not in "0123456789abcdef" for character in base_commit):
        raise ContractError("upstream.nominal_base_commit must be a 40-character lowercase Git SHA")

    runtime = _mapping(contract, "runtime")
    python_env = _nonempty_string(runtime, "python_env", "runtime")
    backend_path = _nonempty_string(runtime, "external_backend_path", "runtime")

    experiment_section = _mapping(contract, "experiment")
    experiment_value = _nonempty_string(experiment_section, "config", "experiment")
    experiment_path = _repo_path(repo_root, experiment_value, "experiment.config")
    experiment = _load_mapping(experiment_path, "experiment config")

    patch_stack = _mapping(contract, "patch_stack")
    if patch_stack.get("status") != "TO_BE_CONFIRMED_ON_SERVER":
        raise ContractError("patch_stack.status must be TO_BE_CONFIRMED_ON_SERVER before audit")
    if patch_stack.get("ordered_patches") is not None:
        raise ContractError("patch_stack.ordered_patches must remain null until the server audit")
    patch_value = _nonempty_string(patch_stack, "tail_opd_patch", "patch_stack")
    patch_path = _repo_path(repo_root, patch_value, "patch_stack.tail_opd_patch")
    if not patch_path.is_file():
        raise ContractError(f"TailOPD patch does not exist: {patch_path}")

    invariants = contract.get("required_invariants")
    smoke_checks = contract.get("required_smoke_checks")
    if not isinstance(invariants, list) or not invariants:
        raise ContractError("required_invariants must be a non-empty list")
    if not isinstance(smoke_checks, list) or not smoke_checks:
        raise ContractError("required_smoke_checks must be a non-empty list")

    experiment_backend = _mapping(experiment, "backend")
    experiment_contract = _nonempty_string(experiment_backend, "contract", "backend")
    expected_contract = contract_path.resolve().relative_to(repo_root.resolve()).as_posix()
    if experiment_contract != expected_contract:
        raise ContractError(
            f"experiment backend.contract={experiment_contract!r} does not reference {expected_contract!r}"
        )
    if _nonempty_string(experiment_backend, "patch", "backend") != patch_value:
        raise ContractError("experiment backend.patch does not match contract TailOPD patch")
    if _nonempty_string(experiment_backend, "root", "backend") != backend_path:
        raise ContractError("experiment backend.root does not match contract external backend path")

    environment = _mapping(experiment, "environment")
    expected_python = f"{python_env.rstrip('/')}/bin/python"
    if _nonempty_string(environment, "python", "environment") != expected_python:
        raise ContractError("experiment environment.python does not match contract python_env")

    training = _mapping(experiment, "training")
    rollout_n = training.get("rollout_n")
    if isinstance(rollout_n, bool) or not isinstance(rollout_n, int) or rollout_n <= 0:
        raise ContractError("experiment training.rollout_n must be a positive integer")

    return {
        "name": name,
        "nominal_base_commit": base_commit,
        "patch_path": patch_value,
        "patch_sha256": _sha256(patch_path),
        "experiment_path": experiment_value,
        "rollout_n": rollout_n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", nargs="?", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()

    try:
        result = verify_contract(args.contract)
    except ContractError as error:
        print(f"FAIL: {error}")
        return 1

    print(f"PASS: contract schema and local references ({result['name']})")
    print(f"PASS: TailOPD patch exists: {result['patch_path']}")
    print(f"PASS: TailOPD patch sha256: {result['patch_sha256']}")
    print(
        "PASS: experiment config references the contract, patch, backend path, "
        f"runtime environment, and rollout_n={result['rollout_n']}"
    )
    print(
        "SKIP: requires active backend worktree: current HEAD/status, diff from nominal base, "
        "ordered patch stack, unexplained diffs, backend integration test, and GPU smoke"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

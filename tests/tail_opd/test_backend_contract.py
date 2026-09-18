from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/verify_backend_contract.py"
SPEC = importlib.util.spec_from_file_location("verify_backend_contract", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tailopd_backend_contract_local_references_are_consistent():
    contract_path = REPO_ROOT / "configs/backend/verl_qwen35_v090_cu132.yaml"
    result = MODULE.verify_contract(contract_path, repo_root=REPO_ROOT)

    assert result["nominal_base_commit"].startswith("483b8a00")
    assert result["patch_path"] == "patches/verl/nll_tailopd_v1.patch"
    assert len(result["patch_sha256"]) == 64
    assert result["rollout_n"] == 4
    assert result["verification_status"] == "TO_BE_CONFIRMED_ON_SERVER"
    assert result["verified_patch_count"] == 0


def test_backend_contract_rejects_missing_patch(tmp_path):
    contract_path = REPO_ROOT / "configs/backend/verl_qwen35_v090_cu132.yaml"
    contract_text = contract_path.read_text(encoding="utf-8").replace(
        "patches/verl/nll_tailopd_v1.patch",
        "patches/verl/does_not_exist.patch",
    )
    broken_contract = tmp_path / "broken.yaml"
    broken_contract.write_text(contract_text, encoding="utf-8")

    with pytest.raises(MODULE.ContractError, match="TailOPD patch does not exist"):
        MODULE.verify_contract(broken_contract, repo_root=REPO_ROOT)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_verified_contract_fixture(root: Path, *, include_evidence: bool = True) -> Path:
    patch_path = root / "patches/verl/tail.patch"
    audit_path = root / "docs/backend_audit.md"
    experiment_path = root / "configs/experiment/tail.yaml"
    contract_path = root / "configs/backend/contract.yaml"
    for path in (patch_path, audit_path, experiment_path, contract_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text("verified patch\n", encoding="utf-8")
    audit_path.write_text("verified reconstruction evidence\n", encoding="utf-8")

    experiment = {
        "backend": {
            "contract": "configs/backend/contract.yaml",
            "root": "/server/backend",
            "patch": "patches/verl/tail.patch",
        },
        "environment": {"python": "/server/env/bin/python"},
        "training": {"rollout_n": 4},
    }
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")

    tree_sha = "2" * 40
    patch_stack = {
        "status": "VERIFIED_ON_SERVER",
        "ordered_patches": [
            {"path": "patches/verl/tail.patch", "sha256": _sha256(patch_path)}
        ],
        "reconstruction_evidence": None,
        "tail_opd_patch": "patches/verl/tail.patch",
    }
    if include_evidence:
        patch_stack["reconstruction_evidence"] = {
            "audit_report": "docs/backend_audit.md",
            "audit_report_sha256": _sha256(audit_path),
            "active_backend_head": "1" * 40,
            "active_backend_tree": tree_sha,
            "reconstructed_backend_tree": tree_sha,
            "clean_reconstruction_diff": True,
        }

    contract = {
        "schema_version": 1,
        "name": "verified-test-backend",
        "verification_status": "VERIFIED_ON_SERVER",
        "upstream": {
            "repo": "https://github.com/verl-project/verl.git",
            "nominal_base_commit": "0" * 40,
        },
        "runtime": {
            "python_env": "/server/env",
            "external_backend_path": "/server/backend",
        },
        "experiment": {"config": "configs/experiment/tail.yaml"},
        "patch_stack": patch_stack,
        "required_invariants": [{"id": "TAIL-01"}],
        "required_smoke_checks": [{"id": "SMOKE-01"}],
    }
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    return contract_path


def test_verified_contract_requires_and_validates_reconstruction_evidence(tmp_path):
    contract_path = _write_verified_contract_fixture(tmp_path)

    result = MODULE.verify_contract(contract_path, repo_root=tmp_path)

    assert result["verification_status"] == "VERIFIED_ON_SERVER"
    assert result["verified_patch_count"] == 1
    assert result["audit_report"] == "docs/backend_audit.md"


def test_verified_contract_rejects_missing_reconstruction_evidence(tmp_path):
    contract_path = _write_verified_contract_fixture(tmp_path, include_evidence=False)

    with pytest.raises(MODULE.ContractError, match="requires patch_stack.reconstruction_evidence"):
        MODULE.verify_contract(contract_path, repo_root=tmp_path)

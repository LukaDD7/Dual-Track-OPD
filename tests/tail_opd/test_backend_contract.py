from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


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

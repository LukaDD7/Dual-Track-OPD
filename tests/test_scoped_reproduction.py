from pathlib import Path

import importlib.util
import json

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/external/scoped_expert_reproduction.yaml"
SETUP = ROOT / "scripts/hpc/setup_scoped_reproduction.sh"
PAIR_RUNNER = ROOT / "scripts/hpc/run_scoped_expert_pair.sh"
EXPERT_RUNNER = ROOT / "scripts/hpc/run_scoped_expert.sh"
MODULE_PATH = ROOT / "scripts/hpc/scoped_reproduction.py"


def load_module():
    spec = importlib.util.spec_from_file_location("scoped_reproduction", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contract_pins_upstream_and_four_expert_shapes():
    module = load_module()
    config = module.validate_config(CONFIG)

    assert config["upstream"]["commit"] == "74e112e29e2be0657d1d521c1ecc3be53df99e44"
    assert config["dataset"]["revision"] == "912185f76295f847c09bc44f379cf85513472ebb"
    assert config["dataset"]["manifest_sha256"] == (
        "bc3340721edddbd869259cb1db50ca54446f8379deb4078067eb83896e47ca96"
    )
    assert config["defaults"]["base_model"] == (
        "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/"
        "Qwen3-VL-8B-Instruct"
    )
    assert config["execution_plan"]["moped_stage"].startswith("deferred")


def test_dataset_manifest_verification_rejects_wrong_hash(tmp_path):
    module = load_module()
    manifest = tmp_path / "data" / "hf_dataset" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"complete": True, "limited_export": False}))

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        module.inspect_dataset(
            tmp_path,
            expected_revision="912185f",
            expected_manifest_sha256="0" * 64,
        )


def test_repo_dirty_check_ignores_only_python_runtime_cache():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'line.startswith("?? ") and "__pycache__/" in line' in source
    assert 'git_output(repo, "status", "--short", "--untracked-files=all")' in source


def test_pair_runner_uses_two_isolated_ray_directories_and_all_four_caps():
    source = PAIR_RUNNER.read_text(encoding="utf-8")

    assert "CAP_A=capability2; CAP_B=capability3" in source
    assert "CAP_A=capability1; CAP_B=capability4" in source
    assert "/tmp/scoped-ray-$STAMP-a" in source
    assert "/tmp/scoped-ray-$STAMP-b" in source
    assert "SCOPED_STORAGE=" in source
    assert "AUTO_DOWNLOAD_DATA=" in source
    assert '--capability all --output "$DATA_DIR" --offline' in source
    assert "bash \"$REPO_ROOT/scripts/train.sh\"" in source
    assert "bash scripts/train.sh opd" not in source


def test_setup_pins_commit_and_refuses_dirty_checkout_without_reset():
    source = SETUP.read_text(encoding="utf-8")

    assert 'git -C "$REPO_ROOT" checkout "$COMMIT"' in source
    assert "git reset --hard" not in source
    assert "git clean" not in source
    assert 'git -C "$REPO_ROOT" status --short' in source
    assert "requirements-rl.txt" in source
    assert "requirements.txt" in source


def test_single_expert_runner_supports_separate_four_gpu_allocations():
    source = EXPERT_RUNNER.read_text(encoding="utf-8")

    assert "capability1|capability2|capability3|capability4" in source
    assert "SCOPED_GPU_GROUP" in source
    assert "RAY_TMPDIR=\"/tmp/scoped-ray-${CAPABILITY}-${STAMP}\"" in source
    assert '--capability all --output "$DATA_DIR" --offline' in source
    assert "bash \"$REPO_ROOT/scripts/train.sh\" \"$CAPABILITY\"" in source
    assert "scripts/train.sh skill" not in source


def test_record_writer_is_atomic(tmp_path):
    module = load_module()
    output = tmp_path / "records" / "setup.json"
    record = {
        "schema_version": 1,
        "experiment": "scoped-four-expert-reproduction",
        "stage": "setup",
    }
    module.write_record(record, output)

    assert json.loads(output.read_text()) == record

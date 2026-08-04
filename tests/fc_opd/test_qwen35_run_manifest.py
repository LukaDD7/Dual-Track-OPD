from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "qwen35_run_manifest.py"
TOKENIZER_ALIGNMENT_SCRIPT = REPO_ROOT / "scripts" / "qwen35_tokenizer_alignment.py"
WRAPPER = REPO_ROOT / "scripts" / "run_qwen35_formal.sh"


def _git_repo(path: Path, filename: str) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / filename).write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", filename], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "initial"], check=True)


def _model(path: Path, architecture: str) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl", "architectures": [architecture]}) + "\n",
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}}, "added_tokens": []}) + "\n",
        encoding="utf-8",
    )


def test_manifest_records_full_hashes_and_finalizes_hydra_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    backend = tmp_path / "backend"
    _git_repo(repo, "README.md")
    _git_repo(backend, "backend.py")
    (backend / "backend.py").write_text("dirty\n", encoding="utf-8")

    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    train.write_bytes(b"train-data")
    val.write_bytes(b"val-data")
    student = tmp_path / "student"
    teacher = tmp_path / "teacher"
    _model(student, "StudentModel")
    _model(teacher, "TeacherModel")
    env_prefix = tmp_path / "env"
    env_manifest = env_prefix / "share/dual-track-opd/environment_manifest.json"
    env_manifest.parent.mkdir(parents=True)
    env_manifest.write_text('{"environment": "test"}\n', encoding="utf-8")
    metadata = tmp_path / "metadata"

    start = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "start",
            "--metadata-dir",
            str(metadata),
            "--repo-root",
            str(repo),
            "--backend-root",
            str(backend),
            "--env-prefix",
            str(env_prefix),
            "--train-file",
            str(train),
            "--val-file",
            str(val),
            "--student-model",
            str(student),
            "--teacher-model",
            str(teacher),
            "--launcher",
            str(REPO_ROOT / "scripts/run_qwen35_formal.sh"),
            "--config",
            'launcher_args_json=["trainer.val_only=True"]',
            "--config",
            "trainer_use_v1=True",
            "--",
            "trainer.use_v1=True",
            "trainer.val_only=True",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    )
    assert start.returncode == 0, start.stdout + start.stderr

    manifest_path = metadata / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "started"
    assert manifest["datasets"]["train"]["sha256"] == hashlib.sha256(b"train-data").hexdigest()
    assert len(manifest["datasets"]["validation"]["sha256"]) == 64
    assert manifest["backend"]["tracked_dirty"] is True
    assert len(manifest["backend"]["diff_sha256"]) == 64
    assert manifest["hardware"]["cuda_visible_devices"] == "0,1,2,3"
    assert manifest["environment"]["manifest"]["sha256"] == hashlib.sha256(
        env_manifest.read_bytes()
    ).hexdigest()

    hydra = metadata / "hydra/.hydra"
    hydra.mkdir(parents=True)
    (hydra / "config.yaml").write_text("trainer:\n  use_v1: true\n", encoding="utf-8")
    (hydra / "overrides.yaml").write_text("- trainer.use_v1=True\n", encoding="utf-8")
    (metadata / "train.log").write_text("resolved config and training output\n", encoding="utf-8")

    finish = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "finish",
            "--metadata-dir",
            str(metadata),
            "--returncode",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert finish.returncode == 0, finish.stdout + finish.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["returncode"] == 0
    assert len(manifest["launch_config"]["hydra_composed_config"]["sha256"]) == 64
    assert len(manifest["launch_config"]["hydra_overrides"]["sha256"]) == 64
    assert len(manifest["launch_config"]["train_log"]["sha256"]) == 64
    launch = json.loads((metadata / "resolved_launch_config.json").read_text(encoding="utf-8"))
    assert launch["wrapper"]["trainer_use_v1"] == "True"
    assert launch["launcher_argv"][-1] == "trainer.val_only=True"
    assert launch["backend_argv"][-1] == "trainer.val_only=True"


def test_wrapper_real_mode_writes_and_finalizes_manifest_with_stubs(tmp_path: Path) -> None:
    backend = tmp_path / "backend"
    _git_repo(backend, "README.md")
    backend_run = backend / "examples/on_policy_distillation_trainer"
    backend_run.mkdir(parents=True)
    backend_script = backend_run / "run_qwen3_5_4b_fsdp.sh"
    backend_script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
hydra_dir=
trainer_v1=
for arg in "$@"; do
  case "$arg" in
    hydra.run.dir=*) hydra_dir=${arg#*=} ;;
    trainer.use_v1=*) trainer_v1=${arg#*=} ;;
  esac
done
[ "$trainer_v1" = "True" ]
mkdir -p "$hydra_dir/.hydra"
printf 'trainer:\n  use_v1: true\n' > "$hydra_dir/.hydra/config.yaml"
printf '%s\n' '- trainer.use_v1=True' > "$hydra_dir/.hydra/overrides.yaml"
echo 'TaskRunnerV1 stub completed'
""",
        encoding="utf-8",
    )

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(SCRIPT, scripts / SCRIPT.name)
    shutil.copy2(TOKENIZER_ALIGNMENT_SCRIPT, scripts / TOKENIZER_ALIGNMENT_SCRIPT.name)
    (scripts / "qwen35_vllm_preflight.py").write_text("print('preflight stub: PASS')\n", encoding="utf-8")

    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text("conda() { return 0; }\n", encoding="utf-8")
    cuda_home = tmp_path / "cuda"
    cuda_bin = cuda_home / "bin"
    cuda_bin.mkdir(parents=True)
    nvidia_smi = cuda_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/usr/bin/env bash\necho '0, NVIDIA H200, GPU-test, 999.0, 143771 MiB'\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)

    env_prefix = tmp_path / "env"
    env_manifest = env_prefix / "share/dual-track-opd/environment_manifest.json"
    env_manifest.parent.mkdir(parents=True)
    env_manifest.write_text('{"environment": "stub"}\n', encoding="utf-8")
    student = tmp_path / "student"
    teacher = tmp_path / "teacher"
    _model(student, "StudentModel")
    _model(teacher, "TeacherModel")
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    train.write_bytes(b"train")
    val.write_bytes(b"val")
    outputs = tmp_path / "outputs"
    validation = outputs / "validation"
    metadata = outputs / "metadata"

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "DTOPD_ROOT": str(tmp_path),
            "PROJECT_ROOT": str(REPO_ROOT),
            "BACKEND_ROOT": str(backend),
            "BACKEND_RUN_DIR": str(backend_run),
            "ENV_PREFIX": str(env_prefix),
            "CUDA_HOME": str(cuda_home),
            "CONDA_SH": str(conda_sh),
            "SCRIPTS": str(scripts),
            "STUDENT_MODEL": str(student),
            "TEACHER_MODEL": str(teacher),
            "TRAIN_FILE": str(train),
            "VAL_FILE": str(val),
            "FORMAL_GPUS": "0,1,2,3",
            "NGPUS_PER_NODE": "3",
            "TRAIN_BATCH_SIZE": "24",
            "PPO_MINI_BATCH_SIZE": "24",
            "ROLLOUT_NUM_WORKERS": "8",
            "USE_FCOP_DATASET": "1",
            "USE_TASK_REWARDS": "True",
            "TRAINER_USE_V1": "True",
            "TOTAL_TRAINING_STEPS": "1",
            "SAVE_FREQ": "-1",
            "EXPERIMENT_NAME": "stub_v1_run",
            "PROJECT_NAME": "stub_project",
            "VALIDATION_DATA_DIR": str(validation),
            "RUN_METADATA_DIR": str(metadata),
            "CKPT_ROOT": str(outputs / "checkpoints"),
        },
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((metadata / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["returncode"] == 0
    assert manifest["launch_config"]["hydra_composed_config"] is not None
    assert manifest["launch_config"]["hydra_overrides"] is not None
    assert manifest["launch_config"]["train_log"] is not None
    assert manifest["tokenizer_alignment"] is not None
    tokenizer_alignment = json.loads((metadata / "tokenizer_alignment.json").read_text(encoding="utf-8"))
    assert tokenizer_alignment["valid"] is True
    assert "TaskRunnerV1 stub completed" in (metadata / "train.log").read_text(encoding="utf-8")
    launch = json.loads((metadata / "resolved_launch_config.json").read_text(encoding="utf-8"))
    assert "trainer.use_v1=True" in launch["backend_argv"]

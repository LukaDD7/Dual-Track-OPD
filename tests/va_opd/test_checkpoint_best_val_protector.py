import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "hpc" / "protect_va_opd_best_checkpoint.py"


def _write_checkpoint(root: Path, step: int) -> Path:
    path = root / f"global_step_{step}" / "actor"
    path.mkdir(parents=True)
    weight = path / "model.safetensors"
    weight.write_bytes(f"weights-{step}".encode())
    return path


def test_best_checkpoint_protector_dry_run_and_apply(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    _write_checkpoint(root, 1)
    _write_checkpoint(root, 2)
    log = tmp_path / "train.log"
    log.write_text(
        "\x1b[36m(TaskRunnerV1) step:1 - val-core/ViRL39K/reward/mean@1:np.float64(0.55)\x1b[0m\n"
        "(TaskRunnerV1) step:2 - val-core/ViRL39K/reward/mean@1:np.float64(0.62)\n",
        encoding="utf-8",
    )

    command = [sys.executable, str(SCRIPT), "--checkpoint-root", str(root), "--train-log", str(log)]
    dry = subprocess.run(command, check=True, text=True, capture_output=True)
    assert "step 2" in dry.stdout
    assert not (root / "best_val").exists()

    subprocess.run(command + ["--apply"], check=True, text=True, capture_output=True)
    protected = root / "best_val" / "global_step_2" / "actor" / "model.safetensors"
    assert protected.read_bytes() == b"weights-2"
    source = root / "global_step_2" / "actor" / "model.safetensors"
    assert protected.stat().st_nlink == source.stat().st_nlink == 2
    marker = json.loads((root / "best_val.json").read_text(encoding="utf-8"))
    assert marker["step"] == 2
    assert marker["validation_score"] == 0.62


def test_best_checkpoint_protector_replaces_previous_best(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    _write_checkpoint(root, 1)
    _write_checkpoint(root, 2)
    _write_checkpoint(root, 3)
    log = tmp_path / "train.log"
    log.write_text(
        "step:1 - val-core/ViRL39K/reward/mean@1:np.float64(0.55)\n"
        "step:2 - val-core/ViRL39K/reward/mean@1:np.float64(0.62)\n"
        "step:3 - val-core/ViRL39K/reward/mean@1:np.float64(0.67)\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(SCRIPT),
        "--checkpoint-root",
        str(root),
        "--train-log",
        str(log),
        "--apply",
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)
    assert (root / "best_val" / "global_step_3").is_dir()
    assert not (root / "best_val" / "global_step_2").exists()


def test_best_checkpoint_protector_prefers_latest_on_tie(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    _write_checkpoint(root, 25)
    _write_checkpoint(root, 350)
    log = tmp_path / "train.log"
    log.write_text(
        "step:25 - val-core/ViRL39K/reward/mean@1:np.float64(0.624)\n"
        "step:350 - val-core/ViRL39K/reward/mean@1:np.float64(0.624)\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(SCRIPT),
        "--checkpoint-root",
        str(root),
        "--train-log",
        str(log),
        "--apply",
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)

    marker = json.loads((root / "best_val.json").read_text(encoding="utf-8"))
    assert marker["step"] == 350
    assert marker["validation_score"] == 0.624
    assert (root / "best_val" / "global_step_350" / "actor" / "model.safetensors").is_file()
    assert not (root / "best_val" / "global_step_25").exists()


def test_best_checkpoint_protector_preserves_history_across_resume(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    _write_checkpoint(root, 25)
    _write_checkpoint(root, 100)
    logs = tmp_path / "logs"
    logs.mkdir()
    launcher_log = logs / "launcher.log"
    launcher_log.write_text(
        "step:25 - val-core/ViRL39K/reward/mean@1:np.float64(0.624)\n"
        "step:100 - val-core/ViRL39K/reward/mean@1:np.float64(0.598)\n",
        encoding="utf-8",
    )
    train_log = logs / "train.log"
    train_log.write_text(
        "step:100 - val-core/ViRL39K/reward/mean@1:np.float64(0.576)\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(SCRIPT),
        "--checkpoint-root",
        str(root),
        "--train-log",
        str(train_log),
        "--apply",
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)

    marker = json.loads((root / "best_val.json").read_text(encoding="utf-8"))
    assert marker["step"] == 25
    assert marker["validation_score"] == 0.624
    history = json.loads((root / "best_val_history.json").read_text(encoding="utf-8"))
    # A repeated validation at the same step uses the newest train.log score.
    assert history["scores"] == {"25": 0.624, "100": 0.576}


def test_best_checkpoint_protector_updates_after_source_pruned(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    _write_checkpoint(root, 25)
    _write_checkpoint(root, 100)
    logs = tmp_path / "logs"
    logs.mkdir()
    launcher_log = logs / "launcher.log"
    launcher_log.write_text(
        "step:25 - val-core/ViRL39K/reward/mean@1:np.float64(0.624)\n"
        "step:100 - val-core/ViRL39K/reward/mean@1:np.float64(0.598)\n",
        encoding="utf-8",
    )
    train_log = logs / "train.log"
    train_log.write_text(
        "step:100 - val-core/ViRL39K/reward/mean@1:np.float64(0.598)\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--checkpoint-root",
        str(root),
        "--train-log",
        str(train_log),
        "--apply",
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)

    # Simulate trainer retention: the original step-25 source is gone, while
    # its protected hard-link tree remains. A later higher score must replace it.
    shutil.rmtree(root / "global_step_25")
    _write_checkpoint(root, 125)
    train_log.write_text(
        "step:125 - val-core/ViRL39K/reward/mean@1:np.float64(0.630)\n",
        encoding="utf-8",
    )
    subprocess.run(command, check=True, text=True, capture_output=True)

    marker = json.loads((root / "best_val.json").read_text(encoding="utf-8"))
    assert marker["step"] == 125
    assert marker["validation_score"] == 0.630
    assert (root / "best_val" / "global_step_125" / "actor" / "model.safetensors").is_file()
    assert not (root / "best_val" / "global_step_25").exists()

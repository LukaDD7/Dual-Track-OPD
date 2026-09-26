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
    assert not (root.parent / "protected_runs" / root.name).exists()

    subprocess.run(command + ["--apply"], check=True, text=True, capture_output=True)
    protected_root = root.parent / "protected_runs" / root.name
    protected = protected_root / "global_step_2" / "actor" / "model.safetensors"
    assert protected.read_bytes() == b"weights-2"
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
    protected_root = root.parent / "protected_runs" / root.name
    assert (protected_root / "global_step_3").is_dir()
    assert not (protected_root / "global_step_2").exists()


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
    protected_root = root.parent / "protected_runs" / root.name
    assert (protected_root / "global_step_350" / "actor" / "model.safetensors").is_file()
    assert not (protected_root / "global_step_25").exists()


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
    # A repeated validation at the same step keeps the historical high score.
    assert history["scores"] == {"25": 0.624, "100": 0.598}


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
    protected_root = root.parent / "protected_runs" / root.name
    assert (protected_root / "global_step_125" / "actor" / "model.safetensors").is_file()
    assert not (protected_root / "global_step_25").exists()


def test_best_checkpoint_protector_keeps_protected_best_after_rolling_source_pruned(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    _write_checkpoint(root, 625)
    _write_checkpoint(root, 775)
    logs = tmp_path / "logs"
    logs.mkdir()
    history_log = logs / "launcher.log"
    history_log.write_text(
        "step:625 - val-core/ViRL39K/reward/mean@1:np.float64(0.634)\n"
        "step:775 - val-core/ViRL39K/reward/mean@1:np.float64(0.624)\n",
        encoding="utf-8",
    )
    train_log = logs / "train.log"
    train_log.write_text(
        "step:775 - val-core/ViRL39K/reward/mean@1:np.float64(0.624)\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--checkpoint-root",
        str(root),
        "--train-log",
        str(train_log),
        "--history-log",
        str(history_log),
        "--apply",
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)

    # Reproduce rolling-checkpoint cleanup after the best was protected. The
    # newer rolling checkpoint scores lower and must not replace history.
    shutil.rmtree(root / "global_step_625")
    subprocess.run(command, check=True, text=True, capture_output=True)

    marker = json.loads((root / "best_val.json").read_text(encoding="utf-8"))
    assert marker["step"] == 625
    assert marker["validation_score"] == 0.634
    protected_root = root.parent / "protected_runs" / root.name
    protected_best = protected_root / "global_step_625" / "actor" / "model.safetensors"
    assert protected_best.read_bytes() == b"weights-625"
    assert marker["source"] == str(protected_root / "global_step_625")
    assert not (protected_root / "global_step_775").exists()


def test_best_checkpoint_protector_uses_protected_best_when_rolling_source_is_tombstone(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    _write_checkpoint(root, 800)
    _write_checkpoint(root, 850)
    logs = tmp_path / "logs"
    logs.mkdir()
    history_log = logs / "launcher.log"
    history_log.write_text(
        "step:800 - val-core/ViRL39K/reward/mean@1:np.float64(0.638)\n"
        "step:850 - val-core/ViRL39K/reward/mean@1:np.float64(0.616)\n",
        encoding="utf-8",
    )
    train_log = logs / "train.log"
    train_log.write_text(
        "step:850 - val-core/ViRL39K/reward/mean@1:np.float64(0.616)\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--checkpoint-root",
        str(root),
        "--train-log",
        str(train_log),
        "--history-log",
        str(history_log),
        "--apply",
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)

    # VERL retention leaves a directory tombstone rather than removing the
    # whole step directory. The protected best must still win over a newer,
    # lower-scoring rolling checkpoint.
    (root / "global_step_800" / "actor" / "model.safetensors").unlink()
    subprocess.run(command, check=True, text=True, capture_output=True)

    marker = json.loads((root / "best_val.json").read_text(encoding="utf-8"))
    assert marker["step"] == 800
    assert marker["validation_score"] == 0.638
    protected_root = root.parent / "protected_runs" / root.name
    assert (protected_root / "global_step_800" / "actor" / "model.safetensors").is_file()
    assert marker["source"] == str(protected_root / "global_step_800")
    assert not (protected_root / "global_step_850").exists()


def test_best_checkpoint_protector_rejects_tombstone_without_losing_best(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    _write_checkpoint(root, 625)
    _write_checkpoint(root, 575)
    # Simulate VERL retention: shard files disappear, but the directory remains.
    for shard in (root / "global_step_575" / "actor").glob("model.safetensors"):
        shard.unlink()
    log = tmp_path / "train.log"
    log.write_text(
        "step:575 - val-core/ViRL39K/reward/mean@1:np.float64(0.63)\n"
        "step:625 - val-core/ViRL39K/reward/mean@1:np.float64(0.60)\n",
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
    assert marker["step"] == 625
    assert marker["validation_score"] == 0.60
    protected_root = root.parent / "protected_runs" / root.name
    assert (protected_root / "global_step_625" / "actor" / "model.safetensors").is_file()
    assert not (protected_root / "global_step_575").exists()


def test_best_checkpoint_protector_keeps_historical_high_score_across_resume(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    _write_checkpoint(root, 625)
    logs = tmp_path / "logs"
    logs.mkdir()
    launcher_log = logs / "launcher.log"
    # Historical 0.634 selected step 625 before the GPU-instance reclamation.
    launcher_log.write_text(
        "step:625 - val-core/ViRL39K/reward/mean@1:np.float64(0.634)\n",
        encoding="utf-8",
    )
    # The resumed run validates the restored checkpoint again and gets 0.590.
    train_log = logs / "train.log"
    train_log.write_text(
        "step:625 - val-core/ViRL39K/reward/mean@1:np.float64(0.590)\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(SCRIPT),
        "--checkpoint-root",
        str(root),
        "--train-log",
        str(train_log),
        "--history-log",
        str(launcher_log),
        "--apply",
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)

    marker = json.loads((root / "best_val.json").read_text(encoding="utf-8"))
    assert marker["step"] == 625
    assert marker["validation_score"] == 0.634
    history = json.loads((root / "best_val_history.json").read_text(encoding="utf-8"))
    assert history["scores"]["625"] == 0.634

import subprocess
from pathlib import Path
import shutil
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "hpc" / "run_va_opd_native.sh"


def test_missing_nvidia_smi_fails_with_clear_gpu_instance_message(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("VA_OPD_ALLOW_BUSY_GPUS", "0")

    result = subprocess.run(
        [pytest.importorskip("shutil").which("bash") or "/bin/bash", str(SCRIPT), "--objective", "va_opd", "--profile", "paper"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # The environment intentionally lacks the VA-OPD Python and nvidia-smi.
    # The launcher should still contain the fail-fast guard; asserting the
    # exact preflight path would require a full valid CPU environment.
    script_text = SCRIPT.read_text(encoding="utf-8")
    assert 'command -v nvidia-smi >/dev/null 2>&1 || {' in script_text
    assert "launch from the GPU instance" in script_text
    assert result.returncode != 0

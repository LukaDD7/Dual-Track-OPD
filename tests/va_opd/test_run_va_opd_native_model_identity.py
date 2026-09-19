import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "hpc" / "run_va_opd_native.sh"
LAUNCHER = REPO_ROOT / "scripts" / "hpc" / "run_va_opd_32b_teacher_8b_student.sh"
CONFIG = REPO_ROOT / "configs" / "experiment" / "qwen3vl_32b_8b_geometry3k_va_opd_paper.yaml"


def _resolved_models(monkeypatch, cli_models, env_models=None):
    monkeypatch.setenv("VA_OPD_ENV_PREFIX", "/tmp/nonexistent-va-opd-env")
    for key, value in (env_models or {}).items():
        monkeypatch.setenv(key, value)

    command = [
        "bash",
        str(SCRIPT),
        "--objective",
        "va_opd",
        "--profile",
        "paper",
        "--preflight-only",
    ]
    for value in cli_models:
        command.extend(["--student-model", value[0], "--teacher-model", value[1]])

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result


def test_cli_models_survive_paper_profile(monkeypatch, tmp_path):
    student = tmp_path / "Qwen3-VL-8B-Instruct"
    teacher = tmp_path / "Qwen3-VL-32B-Instruct"
    student.mkdir()
    teacher.mkdir()

    result = _resolved_models(
        monkeypatch,
        [(str(student), str(teacher))],
    )

    # The environment is intentionally invalid so the script exits before
    # preflight. It must not exit from the model-path validation path.
    assert result.returncode != 0
    assert "native VA-OPD environment is incomplete" in (result.stdout + result.stderr)

    # Reparse the exact final command as a subprocess-free check.
    script_text = SCRIPT.read_text(encoding="utf-8")
    assert "MODEL_EXPLICIT=true" in script_text
    assert "if ! ${MODEL_EXPLICIT}; then" in script_text


def test_cli_data_paths_survive_paper_profile():
    script_text = SCRIPT.read_text(encoding="utf-8")
    assert "TRAIN_DATA_EXPLICIT=true" in script_text
    assert "VAL_DATA_EXPLICIT=true" in script_text
    assert "if ! ${TRAIN_DATA_EXPLICIT}; then" in script_text
    assert "if ! ${VAL_DATA_EXPLICIT}; then" in script_text


def test_paper_profile_defaults_to_8b_teacher_2b_student():
    script_text = SCRIPT.read_text(encoding="utf-8")
    assert "Qwen3-VL-2B-Instruct" in script_text
    assert "Qwen3-VL-8B-Instruct" in script_text


def test_32b_8b_launcher_defaults_to_rollout_8():
    script_text = LAUNCHER.read_text(encoding="utf-8")
    assert 'ROLLOUT_N="${VA_OPD_ROLLOUT_N:-8}"' in script_text
    assert '--rollout-n "${ROLLOUT_N}"' in script_text


def test_32b_8b_config_uses_rollout_8():
    config_text = CONFIG.read_text(encoding="utf-8")
    assert "rollouts_per_prompt: 8" in config_text

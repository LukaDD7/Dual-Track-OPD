"""GPU-free checks for ``scripts/run_qwen35_formal.sh`` argument composition.

The wrapper supports ``DRY_RUN=1`` (compose + print effective settings, exit
before any GPU/preflight step) and ``CKPT_ROOT``/``ALLOW_EXISTING_RUN_DIR``
overrides so the resume guard can be exercised without touching the real
checkpoint area.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_qwen35_formal.sh"

EXP2_ENV = {
    "FORMAL_GPUS": "0,1,2,3",
    "NGPUS_PER_NODE": "3",
    "TRAIN_BATCH_SIZE": "24",
    "PPO_MINI_BATCH_SIZE": "24",
    "USE_FCOP_DATASET": "1",
    "USE_TASK_REWARDS": "False",
    "ROLLOUT_N": "1",
    "RESUME_MODE": "disable",
    "VAL_BEFORE_TRAIN": "True",
    "TOTAL_TRAINING_STEPS": "20",
    "SAVE_FREQ": "-1",
    "TEST_FREQ": "5",
    "EXPERIMENT_NAME": "qwen3_6_27b_to_qwen3_5_4b_k1_taskfalse_fcop_n1_promptfix_smoke",
    "VALIDATION_DATA_DIR": "/tmp/qwen35_test_val_dump",
    "DRY_RUN": "1",
}


def run_script(env: dict[str, str], *extra_args: str) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    full_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *extra_args],
        capture_output=True,
        text=True,
        env=full_env,
        timeout=300,
    )


def test_script_passes_bash_syntax_check() -> None:
    res = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_default_experiment_name_has_task_dataset_rollout_semantics() -> None:
    res = run_script(
        {
            "DRY_RUN": "1",
            "PROJECT_NAME": "__unit_test__",
            "ALLOW_EXISTING_RUN_DIR": "1",
        }
    )
    assert res.returncode == 0, res.stdout + res.stderr
    # Defaults are raw prompt, task rewards off, rollout n=1, and the target v1 trainer.
    assert "qwen3_6_27b_to_qwen3_5_4b_k1_taskfalse_raw_n1_v1" in res.stdout


def test_exp2_composed_command_contains_required_overrides(tmp_path: Path) -> None:
    env = dict(EXP2_ENV)
    env["VALIDATION_DATA_DIR"] = str(tmp_path / "val_dump")
    env["PROJECT_NAME"] = "__unit_test__"
    res = run_script(env)
    assert res.returncode == 0, res.stdout + res.stderr
    out = res.stdout
    for expected in (
        "data.custom_cls.path=pkg://dual_track_opd.fc_opd.verl_dataset",
        "data.custom_cls.name=FCOPDDataset",
        "trainer.resume_mode=disable",
        "trainer.val_before_train=True",
        "trainer.use_v1=True",
        "trainer.total_training_steps=20",
        "actor_rollout_ref.rollout.n=1",
        "+data.prompt_version=v1",
        "hydra.run.dir=",
        f"trainer.validation_data_dir={env['VALIDATION_DATA_DIR']}",
        "qwen3_6_27b_to_qwen3_5_4b_k1_taskfalse_fcop_n1_promptfix_smoke",
    ):
        assert expected in out, expected
    assert "Using dataset class" not in out  # 由 verl 训练时打印，DRY_RUN 不应伪造


def test_prompt_version_boxed_only_wired_into_args_and_default_name(tmp_path: Path) -> None:
    env = dict(EXP2_ENV)
    env.pop("EXPERIMENT_NAME", None)
    env["PROMPT_VERSION"] = "boxed_only"
    env["VALIDATION_DATA_DIR"] = str(tmp_path / "val_dump")
    env["PROJECT_NAME"] = "__unit_test__"
    res = run_script(env)
    assert res.returncode == 0, res.stdout + res.stderr
    out = res.stdout
    assert "+data.prompt_version=boxed_only" in out
    assert "_fcop_pvboxed_only_n1_v1" in out  # 默认实验名带独立版本 tag
    assert "== prompt_version=boxed_only ==" in out


def test_wrapper_rejects_unknown_prompt_version(tmp_path: Path) -> None:
    env = dict(EXP2_ENV)
    env["PROMPT_VERSION"] = "nope"
    env["VALIDATION_DATA_DIR"] = str(tmp_path / "val_dump")
    res = run_script(env)
    assert res.returncode != 0
    assert "PROMPT_VERSION 必须为 v1 或 boxed_only" in res.stdout + res.stderr


def test_wrapper_rejects_cli_override_of_prompt_version(tmp_path: Path) -> None:
    env = dict(EXP2_ENV)
    env["VALIDATION_DATA_DIR"] = str(tmp_path / "val_dump")
    res = run_script(env, "+data.prompt_version=boxed_only")
    assert res.returncode != 0
    assert "由 wrapper 管理" in res.stdout + res.stderr


def test_wrapper_forwards_cli_overrides_after_generated_args() -> None:
    res = run_script(
        {
            "DRY_RUN": "1",
            "PROJECT_NAME": "__unit_test__",
            "ALLOW_EXISTING_RUN_DIR": "1",
        },
        "trainer.test_freq=3",
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert res.stdout.rstrip().endswith("trainer.test_freq=3")


def test_wrapper_rejects_cli_override_that_would_desync_run_identity() -> None:
    res = run_script(
        {
            "DRY_RUN": "1",
            "PROJECT_NAME": "__unit_test__",
        },
        "trainer.use_v1=False",
    )
    assert res.returncode != 0
    assert "由 wrapper 管理" in res.stdout + res.stderr


def test_v0_override_is_explicit_in_default_name_and_hydra_args() -> None:
    res = run_script(
        {
            "DRY_RUN": "1",
            "TRAINER_USE_V1": "False",
            "PROJECT_NAME": "__unit_test__",
            "ALLOW_EXISTING_RUN_DIR": "1",
        }
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "qwen3_6_27b_to_qwen3_5_4b_k1_taskfalse_raw_n1_v0" in res.stdout
    assert "trainer.use_v1=False" in res.stdout


def test_resume_disable_guard_blocks_nonempty_checkpoint_dir(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpts" / "proj" / "exp"
    ckpt.mkdir(parents=True)
    (ckpt / "global_step_1").touch()
    res = run_script(
        {
            "DRY_RUN": "1",
            "RESUME_MODE": "disable",
            "PROJECT_NAME": "proj",
            "EXPERIMENT_NAME": "exp",
            "CKPT_ROOT": str(tmp_path / "ckpts"),
            "ALLOW_EXISTING_RUN_DIR": "0",
        }
    )
    assert res.returncode != 0
    assert "FATAL" in res.stdout + res.stderr


def test_resume_disable_guard_allows_explicit_override(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpts" / "proj" / "exp"
    ckpt.mkdir(parents=True)
    (ckpt / "global_step_1").touch()
    res = run_script(
        {
            "DRY_RUN": "1",
            "RESUME_MODE": "disable",
            "PROJECT_NAME": "proj",
            "EXPERIMENT_NAME": "exp",
            "CKPT_ROOT": str(tmp_path / "ckpts"),
            "ALLOW_EXISTING_RUN_DIR": "1",
        }
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_resume_disable_guard_blocks_nonempty_validation_dir(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "0.jsonl").write_text("{}\n", encoding="utf-8")
    res = run_script(
        {
            "DRY_RUN": "1",
            "RESUME_MODE": "disable",
            "VALIDATION_DATA_DIR": str(validation),
            "RUN_METADATA_DIR": str(tmp_path / "metadata"),
            "PROJECT_NAME": "__unit_test__",
            "ALLOW_EXISTING_OUTPUT_DIR": "0",
        }
    )
    assert res.returncode != 0
    assert "validation 目录已有内容" in res.stdout + res.stderr


def test_resume_disable_guard_blocks_nonempty_metadata_dir(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    res = run_script(
        {
            "DRY_RUN": "1",
            "RESUME_MODE": "disable",
            "RUN_METADATA_DIR": str(metadata),
            "PROJECT_NAME": "__unit_test__",
            "ALLOW_EXISTING_OUTPUT_DIR": "0",
        }
    )
    assert res.returncode != 0
    assert "metadata 目录已有内容" in res.stdout + res.stderr


def test_numeric_validation_rejects_mangled_env() -> None:
    # 复现手滑把多个 env 挤成一坨（TRAIN_BATCH_SIZE=24PPO_MINI_BATCH_SIZE=24）
    res = run_script(
        {
            "DRY_RUN": "1",
            "TRAIN_BATCH_SIZE": "24PPO_MINI_BATCH_SIZE=24",
            "PROJECT_NAME": "__unit_test__",
            "ALLOW_EXISTING_RUN_DIR": "1",
        }
    )
    assert res.returncode != 0
    assert "FATAL: TRAIN_BATCH_SIZE" in res.stdout + res.stderr


def test_boolean_validation_rejects_ambiguous_trainer_value() -> None:
    res = run_script(
        {
            "DRY_RUN": "1",
            "TRAINER_USE_V1": "maybe",
            "PROJECT_NAME": "__unit_test__",
        }
    )
    assert res.returncode != 0
    assert "FATAL: TRAINER_USE_V1" in res.stdout + res.stderr

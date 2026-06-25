from pathlib import Path

import pytest

from dual_track_opd.fc_opd.conditions import Condition, build_condition_inputs


def _record(full: str, degraded: str) -> dict:
    return {
        "condition_inputs": {
            "full_image": {"path": full},
            "degraded_image": {
                "path": degraded,
                "transform": {"type": "gaussian_blur", "sigma": 2.0},
            },
            "free_caption": "A chart with two bars.",
            "task_evidence": "The blue bar is taller than the red bar.",
        }
    }


def test_build_condition_inputs_resolves_paths_and_conditions(tmp_path: Path):
    full = tmp_path / "full.png"
    degraded = tmp_path / "blur.png"
    full.write_bytes(b"full")
    degraded.write_bytes(b"blur")

    inputs = build_condition_inputs(
        _record("full.png", "blur.png"),
        {"data_root": tmp_path, "require_paths": True},
    )

    assert inputs.full_image.path == str(full)
    assert inputs.degraded_image.transform == {"type": "gaussian_blur", "sigma": 2.0}
    assert inputs.available_conditions() == (
        Condition.FULL,
        Condition.BLUR,
        Condition.FREE,
        Condition.TASK,
    )


def test_verified_facts_require_source():
    record = _record("full.png", "blur.png")
    record["condition_inputs"]["verified_facts"] = "There are two bars."
    with pytest.raises(ValueError, match="provided together"):
        build_condition_inputs(record)


def test_only_gaussian_blur_is_accepted():
    record = _record("full.png", "blur.png")
    record["condition_inputs"]["degraded_image"]["transform"]["type"] = "crop"
    with pytest.raises(ValueError, match="gaussian_blur"):
        build_condition_inputs(record)

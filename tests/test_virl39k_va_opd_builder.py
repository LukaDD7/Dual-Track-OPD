from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


pytest.importorskip("pyarrow")


def _load_builder():
    path = Path("scripts/gen_virl39k_va_opd_parquet.py")
    spec = importlib.util.spec_from_file_location("gen_virl39k_va_opd_parquet", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _image(path: Path, color: str) -> Path:
    Image.new("RGB", (31, 27), color).save(path)
    return path


def _source(tmp_path: Path) -> Path:
    image_one = _image(tmp_path / "one.png", "red")
    image_two = _image(tmp_path / "two.png", "blue")
    source = tmp_path / "source.parquet"
    rows = [
        {
            "question": "<image><image>\nWhich images are shown?\n(A) red and blue",
            "answer": "\\boxed{A}",
            "qid": "multi-1",
            "source": "MMK12",
            "image": [str(image_one), str(image_two)],
        },
        {
            "question": "<image>\nWhat number is shown?\n(A) 1",
            "answer": "\\boxed{3}",
            "qid": "number-1",
            "source": "Processed",
            "image": [str(image_one)],
        },
        {
            "question": "<image>\nUnsupported answer",
            "answer": "\\boxed{unknown}",
            "qid": "other-1",
            "source": "Processed",
            "image": [str(image_one)],
        },
    ]
    pd.DataFrame(rows).to_parquet(source, index=False)
    return source


def test_builder_preserves_multi_images_and_overlap_monitor(tmp_path):
    module = _load_builder()
    output = tmp_path / "prepared" / "train.parquet"
    val_output = tmp_path / "prepared" / "val.parquet"
    manifest = tmp_path / "prepared" / "train.parquet.manifest.json"
    args = module._parse_args(
        [
            "--source",
            str(_source(tmp_path)),
            "--image-root",
            str(tmp_path),
            "--output",
            str(output),
            "--val-output",
            str(val_output),
            "--asset-dir",
            str(tmp_path / "assets"),
            "--manifest",
            str(manifest),
            "--val-rows",
            "1",
        ]
    )

    result = module.build_virl39k_va_opd_parquet(args)

    train = pd.read_parquet(output)
    val = pd.read_parquet(val_output)
    assert len(train) == 2
    assert len(val) == 1
    assert set(train.sample_uid).issuperset(set(val.sample_uid))
    assert result["holdout_val"] is False
    assert result["degradation"] == "lowres_10pct_nearest"

    multi = train.iloc[0]
    assert multi["prompt"][0]["content"].count("<image>") == 2
    assert len(multi["images"]) == 2
    full_paths = multi["condition_inputs"]["full_images"]
    degraded_paths = multi["condition_inputs"]["degraded_images"]
    assert len(full_paths) == len(degraded_paths) == 2
    for full_entry, degraded_entry in zip(full_paths, degraded_paths):
        full_path = Path(full_entry["path"])
        degraded_path = Path(degraded_entry["path"])
        assert full_path.is_file() and degraded_path.is_file()
        with Image.open(full_path) as full, Image.open(degraded_path) as degraded:
            assert full.size == degraded.size
        transform = degraded_entry["transform"]
        assert transform["source_transform"]["scale"] == 0.1
        assert transform["source_transform"]["downsample"] == "bilinear"
        assert transform["source_transform"]["upsample"] == "nearest"


def test_builder_can_make_disjoint_holdout_monitor(tmp_path):
    module = _load_builder()
    output = tmp_path / "prepared" / "train.parquet"
    val_output = tmp_path / "prepared" / "val.parquet"
    args = module._parse_args(
        [
            "--source",
            str(_source(tmp_path)),
            "--image-root",
            str(tmp_path),
            "--output",
            str(output),
            "--val-output",
            str(val_output),
            "--asset-dir",
            str(tmp_path / "assets"),
            "--val-rows",
            "1",
            "--holdout-val",
        ]
    )

    module.build_virl39k_va_opd_parquet(args)

    train = pd.read_parquet(output)
    val = pd.read_parquet(val_output)
    assert len(train) == 1
    assert set(train.sample_uid).isdisjoint(set(val.sample_uid))


def test_virl39k_32b_8b_launcher_defaults_are_strict():
    launcher = Path("scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k.sh").read_text()
    config = Path("configs/experiment/qwen3vl_32b_8b_virl39k_va_opd.yaml").read_text()
    assert 'ROLLOUT_N="${VA_OPD_ROLLOUT_N:-8}"' in launcher
    assert 'STEPS="${VA_OPD_STEPS:-500}"' in launcher
    assert "virl39k_va_opd" in launcher
    assert 'TRAIN_DATA="${DATA_ROOT}/train.parquet"' in launcher
    assert "rollouts_per_prompt: 8" in config
    assert "expected_train_rows: 38348" in config

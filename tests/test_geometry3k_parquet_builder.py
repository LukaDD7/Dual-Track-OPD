from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pandas as pd
import pytest


pytest.importorskip("pyarrow")
PIL = pytest.importorskip("PIL.Image")


def _load_builder():
    path = Path("scripts/gen_geometry3k_parquet.py")
    spec = importlib.util.spec_from_file_location("gen_geometry3k_parquet", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _png_bytes() -> bytes:
    image = PIL.new("RGB", (32, 32), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_builder_creates_disjoint_train_val_assets_and_manifest(tmp_path):
    module = _load_builder()
    source = tmp_path / "source.parquet"
    rows = [
        {
            "images": [{"bytes": _png_bytes()}],
            "problem": f"Find angle {index}.\nA. 30\nB. 60",
            "answer": "B",
        }
        for index in range(3)
    ]
    pd.DataFrame(rows).to_parquet(source, index=False)
    train = tmp_path / "prepared" / "train.parquet"
    val = tmp_path / "prepared" / "val.parquet"
    manifest = tmp_path / "prepared" / "manifest.json"
    args = module._parse_args(
        [
            "--source",
            str(source),
            "--output",
            str(train),
            "--val-output",
            str(val),
            "--val-size",
            "1",
            "--holdout-val",
            "--manifest",
            str(manifest),
        ]
    )

    result = module.build_geometry3k_parquet(args)

    train_df = pd.read_parquet(train)
    val_df = pd.read_parquet(val)
    assert len(train_df) == 2
    assert len(val_df) == 1
    assert set(train_df.sample_uid).isdisjoint(set(val_df.sample_uid))
    assert result["schema_version"] == "geometry3k-qwen3vl-gkd-v1"
    assert len(result["manifest_sha256"]) == 64
    assert manifest.is_file()
    first = train_df.iloc[0]
    assert first["prompt"][0]["content"].startswith("<image>")
    assert Path(first["condition_inputs"]["full_image"]["path"]).is_file()
    assert Path(first["condition_inputs"]["degraded_image"]["path"]).is_file()


def test_builder_preserves_geometry3k_zero_based_choice_indices():
    module = _load_builder()
    choices = ["A. 30", "B. 60", "C. 90"]
    assert module._answer_label("0", choices) == "A"
    assert module._answer_label("1", choices) == "B"
    assert module._answer_label("2", choices) == "C"

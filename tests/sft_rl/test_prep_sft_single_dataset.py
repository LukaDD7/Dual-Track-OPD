"""Unit tests for the per-dataset one-epoch SFT pool preparation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sft_rl"))

import io

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from prep_sft_single_dataset import SCHEMA, load_cauldron, load_mmf, main  # noqa: E402

from prep_sft_warmup import DEFAULT_CAULDRON_SUBSETS, subset_of_part  # noqa: E402


def _png_bytes(w: int = 32, h: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


PNG = _png_bytes()


# --- fixtures ----------------------------------------------------------------


def _row(idx: int, source: str, cot: str = "short answer") -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"<image>\nquestion {idx}"},
            {"role": "assistant", "content": cot},
        ],
        "images": [{"bytes": PNG, "path": None}],
        "source": source,
        "image_hash": f"hash{idx}",
        # extra column the loader must drop (MMF carries pass_rate)
        "pass_rate": 0.5,
    }


def _write_parts(tmp_path: Path, prefix: str, rows: list[dict]) -> None:
    schema = pa.schema([
        pa.field("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        pa.field("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
        pa.field("source", pa.string()),
        pa.field("image_hash", pa.string()),
        pa.field("pass_rate", pa.float64()),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), str(tmp_path / prefix))


class _StubTok:
    """encode() length = len(text) so tests control length without a model."""

    def encode(self, text, add_special_tokens: bool = False) -> list[int]:
        return [0] * len(text)


def _patch_tokenizer(monkeypatch):
    import prep_sft_single_dataset as mod

    class _StubAuto:
        @staticmethod
        def from_pretrained(*a, **k):
            return _StubTok()

    monkeypatch.setattr("transformers.AutoTokenizer", _StubAuto)


# --- loaders -----------------------------------------------------------------


def test_load_mmf_reads_all_parts_and_keeps_extra_columns(tmp_path):
    _write_parts(tmp_path, "mmfinereason_sft__part_train-00000-of-00002.parquet", [_row(0, "mmf-a"), _row(1, "mmf-a")])
    _write_parts(tmp_path, "mmfinereason_sft__part_train-00001-of-00002.parquet", [_row(2, "mmf-b")])
    rows = load_mmf(str(tmp_path))
    assert len(rows) == 3
    assert {r["source"] for r in rows} == {"mmf-a", "mmf-b"}


def test_load_mmf_missing_dir_exits(tmp_path):
    try:
        load_mmf(str(tmp_path / "nonexistent"))
    except SystemExit:
        return
    raise AssertionError("expected SystemExit on missing MMF dir")


def test_load_cauldron_filters_to_default16(tmp_path):
    # one target subset, two non-target subsets -> only target rows survive
    for sub in ("tallyqa", "clevr", "datikz"):
        rows = [_row(0, sub), _row(1, sub)]
        _write_parts(
            tmp_path,
            f"cauldron_sft__part_{sub}_train-00000-of-00001-abc{len(sub)}.parquet",
            rows,
        )
    # partial shard naming (cauldron_sft_partial__part_*) must NOT be picked up
    _write_parts(tmp_path, "cauldron_sft_partial__part_finqa_x.parquet", [_row(9, "finqa")])
    rows = load_cauldron(str(tmp_path), DEFAULT_CAULDRON_SUBSETS)
    assert len(rows) == 2
    assert {r["source"] for r in rows} == {"tallyqa"}


def test_subset_of_part_parses():
    p = Path("cauldron_sft__part_tallyqa_train-00000-of-00001-abc.parquet")
    assert subset_of_part(p) == "tallyqa"


# --- end-to-end main() on synthetic data --------------------------------------


def _run_main(tmp_path, dataset: str, argv: list[str], capsys):
    sys.argv = ["prep_sft_single_dataset.py", "--dataset", dataset,
                "--mmf-dir", str(tmp_path / "mmf"),
                "--cauldron-dir", str(tmp_path / "cauldron"),
                "--out-dir", str(tmp_path / "out"),
                "--model", "stub"] + argv
    main()
    return capsys.readouterr().out


def test_main_mmf_writes_warmup_prefix_shards(tmp_path, monkeypatch, capsys):
    _patch_tokenizer(monkeypatch)
    mmf = tmp_path / "mmf"
    mmf.mkdir()
    rows = [_row(i, "mmf-a", cot="x" * 5) for i in range(30)]
    _write_parts(mmf, "mmfinereason_sft__part_train-00000-of-00001.parquet", rows)
    out = _run_main(tmp_path, "mmf", ["--shard-rows", "10", "--val-frac", "0.1"], capsys)
    assert "raw rows: 30" in out

    out_dir = tmp_path / "out"
    train_shards = sorted(out_dir.glob("sft_warmup_train__part_*.parquet"))
    val_shards = sorted(out_dir.glob("sft_warmup_val__part_*.parquet"))
    assert len(train_shards) == 3  # 27 train rows / 10 per shard
    assert len(val_shards) == 1    # 3 val rows

    t = pq.read_table(train_shards[0])
    assert t.schema.names == ["messages", "images", "source", "image_hash"]
    assert t["images"][0].as_py()[0]["bytes"] == PNG


def test_main_mmf_default_val_frac_is_2pct(tmp_path, monkeypatch, capsys):
    _patch_tokenizer(monkeypatch)
    mmf = tmp_path / "mmf"
    mmf.mkdir()
    _write_parts(mmf, "mmfinereason_sft__part_train-00000-of-00001.parquet", [_row(i, "s") for i in range(100)])
    out = _run_main(tmp_path, "mmf", [], capsys)
    assert "val_frac=0.02" in out


def test_main_cauldron_default_val_frac_and_subsets(tmp_path, monkeypatch, capsys):
    _patch_tokenizer(monkeypatch)
    cau = tmp_path / "cauldron"
    cau.mkdir()
    rows = [_row(i, "iconqa", cot="y" * 3) for i in range(50)]
    _write_parts(cau, "cauldron_sft__part_iconqa_train-00000-of-00001-abc.parquet", rows)
    out = _run_main(tmp_path, "cauldron", ["--shard-rows", "25"], capsys)
    assert "val_frac=0.008" in out
    assert "raw rows: 50" in out
    out_dir = tmp_path / "out"
    assert len(list(out_dir.glob("sft_warmup_train__part_*.parquet"))) == 2  # 50*0.992=49 rows


def test_main_deterministic_split_same_seed(tmp_path, monkeypatch, capsys):
    _patch_tokenizer(monkeypatch)
    mmf = tmp_path / "mmf"
    mmf.mkdir()
    rows = [_row(i, "s", cot=f"z{i}") for i in range(40)]
    _write_parts(mmf, "mmfinereason_sft__part_train-00000-of-00001.parquet", rows)

    _run_main(tmp_path, "mmf", ["--val-frac", "0.1", "--shard-rows", "100"], capsys)
    first = (tmp_path / "out" / "sft_warmup_train__part_0000.parquet")
    a = pq.read_table(first).to_pylist()
    _run_main(tmp_path, "mmf", ["--val-frac", "0.1", "--shard-rows", "100"], capsys)
    b = pq.read_table(first).to_pylist()
    assert a == b  # same seed -> identical shards


def test_main_limit_rows_smoke_cap(tmp_path, monkeypatch, capsys):
    _patch_tokenizer(monkeypatch)
    mmf = tmp_path / "mmf"
    mmf.mkdir()
    _write_parts(mmf, "mmfinereason_sft__part_train-00000-of-00001.parquet", [_row(i, "s") for i in range(100)])
    out = _run_main(tmp_path, "mmf", ["--limit-rows", "10", "--val-frac", "0.1"], capsys)
    assert "limit-rows cap applied: 10 raw rows" in out


def test_main_drops_overlength_rows(tmp_path, monkeypatch, capsys):
    _patch_tokenizer(monkeypatch)
    mmf = tmp_path / "mmf"
    mmf.mkdir()
    rows = [_row(0, "s", cot="a" * 10), _row(1, "s", cot="a" * 20000)]  # second row overlength
    _write_parts(mmf, "mmfinereason_sft__part_train-00000-of-00001.parquet", rows)
    # 32x32 PNG -> 64 img tokens + 224 overhead: short row ~308, long row ~20298
    out = _run_main(tmp_path, "mmf", ["--max-seq-len", "500", "--val-frac", "0.5"], capsys)
    assert "kept 1" in out

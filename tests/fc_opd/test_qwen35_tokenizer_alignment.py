from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "qwen35_tokenizer_alignment.py"
SPEC = importlib.util.spec_from_file_location("qwen35_tokenizer_alignment", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_tokenizer(path: Path, vocab: dict[str, int], *, metadata: str = "") -> None:
    path.mkdir()
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "model": {"type": "BPE", "vocab": vocab},
                "added_tokens": [{"id": 3, "content": "<eos>"}],
            }
        ),
        encoding="utf-8",
    )


def test_alignment_uses_token_id_semantics_not_raw_file_hash(tmp_path: Path) -> None:
    student = tmp_path / "student"
    teacher = tmp_path / "teacher"
    _write_tokenizer(student, {"a": 0, "b": 1}, metadata="student")
    _write_tokenizer(teacher, {"b": 1, "a": 0}, metadata="teacher")

    result = MODULE.compare_tokenizers(student, teacher)

    assert result["valid"] is True
    assert result["student"]["tokenizer_json_sha256"] != result["teacher"]["tokenizer_json_sha256"]
    assert result["student"]["id_to_token_sha256"] == result["teacher"]["id_to_token_sha256"]


def test_alignment_rejects_same_tokens_at_different_ids(tmp_path: Path) -> None:
    student = tmp_path / "student"
    teacher = tmp_path / "teacher"
    _write_tokenizer(student, {"a": 0, "b": 1})
    _write_tokenizer(teacher, {"a": 1, "b": 0})

    result = MODULE.compare_tokenizers(student, teacher)

    assert result["valid"] is False
    assert result["mismatch_count"] == 2
    assert result["first_mismatched_ids"] == [0, 1]

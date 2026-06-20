import csv
import json
from pathlib import Path

from dual_track_opd.eval.score_raw_responses import score_raw_responses


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_score_raw_responses_synthetic(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    scores = tmp_path / "scores.csv"
    audit = tmp_path / "audit.csv"

    _write_jsonl(
        raw_dir / "mcq.jsonl",
        [
            {"id": "ok", "prediction": "Answer: C", "answer": "C", "finish_reason": "stop"},
            {"id": "bad", "prediction": "Answer: A", "answer": "B", "finish_reason": "stop"},
            {"id": "unparsed", "prediction": "A cat appears.", "answer": "A", "finish_reason": "stop"},
            {"id": "length", "prediction": "Answer: A", "answer": "A", "finish_reason": "length"},
        ],
    )
    _write_jsonl(
        raw_dir / "short.jsonl",
        [
            {"sample_id": "s1", "response": "The red apple.", "ground_truth": "red apple"},
            {"sample_id": "s2", "response": "blue apple", "ground_truth": ["red apple"]},
        ],
    )
    _write_jsonl(
        raw_dir / "judge.jsonl",
        [{"id": "j1", "prediction": "A detailed free-form answer.", "answer": "rubric"}],
    )
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "MCQ",
                        "file": "mcq.jsonl",
                        "scoring_type": "mcq",
                        "notes": "test",
                    }
                ),
                json.dumps(
                    {
                        "dataset": "Short",
                        "file": "short.jsonl",
                        "scoring_type": "normalized_exact",
                        "notes": "test",
                    }
                ),
                json.dumps(
                    {
                        "dataset": "Judge",
                        "file": "judge.jsonl",
                        "scoring_type": "needs_judge",
                        "notes": "test",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    score_rows, audit_rows = score_raw_responses(raw_dir, manifest, scores, audit)

    by_dataset = {row["dataset"]: row for row in score_rows}
    assert by_dataset["MCQ"]["n"] == 4
    assert by_dataset["MCQ"]["scored_n"] == 2
    assert by_dataset["MCQ"]["correct"] == 1
    assert by_dataset["MCQ"]["length_rows"] == 1
    assert by_dataset["MCQ"]["unparsed_rows"] == 1
    assert by_dataset["Short"]["accuracy"] == "0.500000"
    assert by_dataset["Judge"]["needs_judge"] == "true"
    assert by_dataset["Judge"]["scored_n"] == 0

    audit_reasons = {row["audit_reason"] for row in audit_rows}
    assert {"incorrect", "unparsed", "length", "needs_judge"}.issubset(audit_reasons)

    with scores.open(newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    assert len(persisted) == 3

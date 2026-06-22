import json
from pathlib import Path

from dual_track_opd.eval.sample_raw_examples import sample_raw_examples


def test_sample_raw_examples_extracts_fields(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "toy.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "a",
                        "question": "What color is the object?",
                        "options": ["A. red", "B. blue"],
                        "prediction": "A. red",
                        "explanation": "It appears red.",
                        "answer": "A",
                        "finish_reason": "stop",
                    }
                ),
                json.dumps({"id": "b", "question": "Second?", "prediction": "B", "answer": "B"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "Toy",
                "file": "toy.jsonl",
                "scoring_type": "mcq",
                "notes": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = sample_raw_examples(raw_dir, manifest, samples_per_dataset=1)
    assert rows[0]["dataset"] == "Toy"
    assert rows[0]["question"] == "What color is the object?"
    assert rows[0]["prediction"] == "A. red"
    assert rows[0]["reasoning"] == "It appears red."


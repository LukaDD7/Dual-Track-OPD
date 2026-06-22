import json
from pathlib import Path

from dual_track_opd.eval.sample_raw_examples import collect_image_refs, sample_raw_examples


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
                        "metadata": {
                            "assets": [{"path": "/tmp/example.png"}],
                            "messages": [
                                {
                                    "content": [
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": "https://example.com/nested.jpg"
                                            },
                                        }
                                    ]
                                }
                            ],
                        },
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
    assert rows[0]["image"] == "/tmp/example.png"
    assert rows[0]["image_paths"] == ["/tmp/example.png", "https://example.com/nested.jpg"]
    assert "metadata" in rows[0]["original_metadata"]
    assert rows[0]["raw_response_file"].endswith("toy.jsonl")
    assert rows[0]["raw_response_row_index"] == 1


def test_collect_image_refs_from_nested_messages():
    refs = collect_image_refs(
        {
            "messages": [
                {
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "input_image",
                            "image_url": {"url": "https://host/image.webp"},
                        },
                    ]
                }
            ],
            "original_metadata": {"local_path": "/dataset/imgs/a.png"},
        }
    )
    assert refs == ["https://host/image.webp", "/dataset/imgs/a.png"]

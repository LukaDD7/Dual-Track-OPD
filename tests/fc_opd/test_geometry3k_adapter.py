import json

from dual_track_opd.fc_opd.geometry3k_adapter import load_geometry3k_records


def test_geometry3k_adapter_normalizes_question_choices_and_image(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"placeholder")
    dataset = tmp_path / "geometry.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "g1",
                    "diagram_path": "diagram.png",
                    "problem": "Find angle ABC.",
                    "choices": ["30", "45", "60"],
                    "answer": "B",
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = load_geometry3k_records(dataset)

    assert rows[0]["sample_uid"] == "geometry3k:g1"
    assert rows[0]["image_path"] == str(image.resolve(strict=False))
    assert rows[0]["choices"] == ["A. 30", "B. 45", "C. 60"]
    assert "A. 30" in rows[0]["question"]
    assert rows[0]["answer_metadata"] == "B"
    assert rows[0]["metadata"]["gold_answer_metadata_only"] is True
    assert rows[0]["bbox_image_paths"] == []

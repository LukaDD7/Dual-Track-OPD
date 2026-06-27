import json

from dual_track_opd.fc_opd.geometry3k_adapter import inspect_geometry3k_dataset, load_geometry3k_records


def _write_official_geometry3k_sample(tmp_path, *, split="train", sample_id="0"):
    root = tmp_path / "unzipped"
    sample_dir = root / split / split / sample_id
    sample_dir.mkdir(parents=True)
    image = sample_dir / "diagram.png"
    image.write_bytes(b"placeholder")
    (sample_dir / "data.json").write_text(
        json.dumps(
            {
                "id": sample_id,
                "problem_text": "Problem text fallback should not win.",
                "compact_text": "Use the diagram to find angle ABC.",
                "choices": ["30", "45", "60"],
                "answer": "42-gold",
                "detailed_solution": "metadata only",
                "problem_type_graph": "angle",
                "problem_type_goal": "calculation",
                "source": "unit-test",
                "data_type": split,
            }
        ),
        encoding="utf-8",
    )
    (sample_dir / "logic_form.json").write_text(
        json.dumps(
            {
                "text_logic_form": ["Find(Angle(ABC))"],
                "dissolved_text_logic_form": ["Find"],
                "diagram_logic_form": ["Line(A,B)", "Line(B,C)"],
                "line_instances": [["A", "B"], ["B", "C"]],
                "point_positions": {"A": [0, 0], "B": [1, 0], "C": [1, 1]},
                "circle_instances": [],
            }
        ),
        encoding="utf-8",
    )
    return root, sample_dir


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


def test_geometry3k_adapter_loads_official_directory_root(tmp_path):
    root, sample_dir = _write_official_geometry3k_sample(tmp_path)

    rows = load_geometry3k_records(root)

    row = rows[0]
    assert row["sample_uid"] == "geometry3k:train:0"
    assert row["split"] == "train"
    assert row["question"] == "Use the diagram to find angle ABC."
    assert row["choices"] == ["A. 30", "B. 45", "C. 60"]
    assert row["answer_metadata"] == "42-gold"
    assert row["outcome_metadata"]["answer"] == "42-gold"
    assert row["outcome_metadata"]["correctness_used_for_prompt"] is False
    assert row["image_path"] == str((sample_dir / "diagram.png").resolve(strict=False))
    assert row["image_exists"] is True
    assert row["data_json_path"] == str(sample_dir / "data.json")
    assert row["logic_form_json_path"] == str(sample_dir / "logic_form.json")
    assert row["problem_type_graph"] == "angle"
    assert row["problem_type_goal"] == "calculation"
    assert row["text_logic_form"] == ["Find(Angle(ABC))"]
    assert row["diagram_logic_form"] == ["Line(A,B)", "Line(B,C)"]
    assert row["line_instances"] == [["A", "B"], ["B", "C"]]
    assert row["point_positions"] == {"A": [0, 0], "B": [1, 0], "C": [1, 1]}
    assert row["circle_instances"] == []


def test_geometry3k_adapter_loads_official_split_subdir(tmp_path):
    root, _sample_dir = _write_official_geometry3k_sample(tmp_path)

    rows = load_geometry3k_records(root / "train")

    assert len(rows) == 1
    assert rows[0]["sample_uid"] == "geometry3k:train:0"
    assert rows[0]["split"] == "train"


def test_geometry3k_adapter_inspection_reports_counts_and_examples(tmp_path):
    root, _sample_dir = _write_official_geometry3k_sample(tmp_path)

    summary = inspect_geometry3k_dataset(root)

    assert summary["num_records"] == 1
    assert summary["split_counts"] == {"train": 1}
    assert summary["missing_image_count"] == 0
    assert summary["sample_examples"][0]["sample_uid"] == "geometry3k:train:0"

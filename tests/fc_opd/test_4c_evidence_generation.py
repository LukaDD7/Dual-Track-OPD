import json

from dual_track_opd.fc_opd.evidence_generation import (
    EvidenceGenerationConfig,
    TemplateEvidenceGenerator,
    run_evidence_generation,
    validate_evidence_row,
)


def _write_official_geometry3k_sample(tmp_path):
    root = tmp_path / "unzipped"
    sample_dir = root / "train" / "train" / "0"
    sample_dir.mkdir(parents=True)
    (sample_dir / "diagram.png").write_bytes(b"placeholder")
    (sample_dir / "data.json").write_text(
        json.dumps(
            {
                "id": "0",
                "problem_text": "Problem text fallback should not win.",
                "compact_text": "Use the diagram to find angle ABC.",
                "choices": ["30", "45", "60"],
                "answer": "42-gold",
                "problem_type_graph": "angle",
                "problem_type_goal": "calculation",
                "data_type": "train",
            }
        ),
        encoding="utf-8",
    )
    (sample_dir / "logic_form.json").write_text(
        json.dumps(
            {
                "text_logic_form": ["Find(Angle(ABC))"],
                "diagram_logic_form": ["Line(A,B)", "Line(B,C)"],
                "line_instances": [["A", "B"], ["B", "C"]],
                "point_positions": {"A": [0, 0], "B": [1, 0], "C": [1, 1]},
                "circle_instances": [],
            }
        ),
        encoding="utf-8",
    )
    return root, sample_dir


class SpyEvidenceGenerator:
    model_id = "spy"

    def __init__(self):
        self.free_prompt = None
        self.task_prompt = None
        self.task_question = None
        self.task_choices = None

    def generate_free_caption(self, *, image_path, prompt, seed):
        del image_path, seed
        self.free_prompt = prompt
        return "Visible diagram evidence only."

    def generate_task_evidence(self, *, image_path, question, choices, prompt, seed):
        del image_path, seed
        self.task_prompt = prompt
        self.task_question = question
        self.task_choices = list(choices)
        return "Visible angle labels and line relations only."


def test_evidence_generation_writes_auditable_rows_without_gold_use(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"placeholder")
    dataset = tmp_path / "geometry.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "g1",
                    "diagram_path": str(image),
                    "question": "What is angle ABC?",
                    "choices": ["30", "45"],
                    "answer": "B",
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evidence.jsonl"
    summary = tmp_path / "summary.json"

    result = run_evidence_generation(
        EvidenceGenerationConfig(
            dataset=dataset,
            dataset_type="geometry3k",
            source_dataset="geometry3k",
            output_jsonl=output,
            summary_json=summary,
            limit=1,
            include_prompts_in_output=True,
        ),
        generator=TemplateEvidenceGenerator(),
    )

    row = result.rows[0]
    assert row["no_gold_field_used"] is True
    assert row["answer_letter_forbidden"] is True
    assert row["final_answer_forbidden"] is True
    assert row["free_caption_prompt_hash"]
    assert row["task_evidence_prompt_hash"]
    assert validate_evidence_row(row) == []
    assert result.summary["validation_error_count"] == 0
    assert output.is_file() and summary.is_file()


def test_prompt_debug_fields_are_hidden_by_default(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"placeholder")
    dataset = tmp_path / "geometry.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "g1",
                    "diagram_path": str(image),
                    "question": "What is angle ABC?",
                    "choices": ["30", "45"],
                    "answer": "B",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = run_evidence_generation(
        EvidenceGenerationConfig(
            dataset=dataset,
            dataset_type="geometry3k",
            output_jsonl=tmp_path / "evidence.jsonl",
            summary_json=tmp_path / "summary.json",
            limit=1,
            condition_set="6c-solve",
        ),
        generator=TemplateEvidenceGenerator(),
    )

    row = result.rows[0]
    assert "free_caption_prompt" not in row
    assert "task_visible_prompt" not in row
    assert row["prompt_hashes"]["task_solve_prompt"]
    assert row["conditions"] == ["full", "degraded", "free", "task_visible", "task_infer", "task_solve"]
    assert row["condition_evidence"]["free"]
    assert row["condition_evidence"]["task_visible"]
    assert row["condition_evidence"]["task_infer"]
    assert row["condition_evidence"]["task_solve"]


def test_geometry3k_official_directory_evidence_generation_does_not_prompt_with_answer(tmp_path):
    root, _sample_dir = _write_official_geometry3k_sample(tmp_path)
    output = tmp_path / "evidence.jsonl"
    summary = tmp_path / "summary.json"
    generator = SpyEvidenceGenerator()

    result = run_evidence_generation(
        EvidenceGenerationConfig(
            dataset=root,
            dataset_type="geometry3k",
            source_dataset="geometry3k",
            output_jsonl=output,
            summary_json=summary,
            limit=1,
            include_prompts_in_output=True,
        ),
        generator=generator,
    )

    row = result.rows[0]
    assert row["sample_uid"] == "geometry3k:train:0"
    assert row["no_gold_field_used"] is True
    assert "42-gold" not in row["question"]
    assert "42-gold" not in row["free_caption_prompt"]
    assert "42-gold" not in row["task_evidence_prompt"]
    assert "42-gold" not in (generator.free_prompt or "")
    assert "42-gold" not in (generator.task_prompt or "")
    assert "42-gold" not in (generator.task_question or "")
    assert "42-gold" not in "\n".join(generator.task_choices or [])
    assert validate_evidence_row(row) == []


def test_evidence_generation_flags_degraded_source_image_by_default(tmp_path):
    image = tmp_path / "diagram.lowres_10pct_nearest.png"
    image.write_bytes(b"placeholder")
    dataset = tmp_path / "geometry.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "g1",
                    "diagram_path": str(image),
                    "question": "What is angle ABC?",
                    "choices": ["30", "45"],
                    "answer": "B",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = run_evidence_generation(
        EvidenceGenerationConfig(
            dataset=dataset,
            dataset_type="geometry3k",
            output_jsonl=tmp_path / "evidence.jsonl",
            summary_json=tmp_path / "summary.json",
            limit=1,
        ),
        generator=TemplateEvidenceGenerator(),
    )

    assert result.summary["degraded_source_image_count"] == 1
    assert "degraded_source_image_path" in result.rows[0]["errors"]
    assert "degraded_source_image_path" in validate_evidence_row(result.rows[0])
    assert result.summary["validation_error_count"] > 0


def test_evidence_generation_can_explicitly_allow_degraded_source_image(tmp_path):
    image = tmp_path / "diagram.lowres_10pct_nearest.png"
    image.write_bytes(b"placeholder")
    dataset = tmp_path / "geometry.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "g1",
                    "diagram_path": str(image),
                    "question": "What is angle ABC?",
                    "choices": ["30", "45"],
                    "answer": "B",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = run_evidence_generation(
        EvidenceGenerationConfig(
            dataset=dataset,
            dataset_type="geometry3k",
            output_jsonl=tmp_path / "evidence_allowed.jsonl",
            summary_json=tmp_path / "summary_allowed.json",
            limit=1,
            allow_degraded_source_images=True,
        ),
        generator=TemplateEvidenceGenerator(),
    )

    assert result.summary["degraded_source_image_count"] == 1
    assert validate_evidence_row(result.rows[0]) == []

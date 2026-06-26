import json

from dual_track_opd.fc_opd.evidence_generation import (
    EvidenceGenerationConfig,
    TemplateEvidenceGenerator,
    run_evidence_generation,
    validate_evidence_row,
)


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

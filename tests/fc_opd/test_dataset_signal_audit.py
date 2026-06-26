import json
from contextlib import ExitStack

import pytest

from dual_track_opd.fc_opd.dataset_signal_audit import (
    DatasetAuditConfig,
    detect_task_evidence_leakage,
    hash_text,
    hash_token_ids,
    load_candidate_records,
    run_dataset_signal_audit,
)
from dual_track_opd.fc_opd.offline_scoring import ByteTokenizer
from dual_track_opd.fc_opd.teacher_client import TeacherClient
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint
from dual_track_opd.fc_opd.teacher_scorer import SyntheticTeacherScorer
from dual_track_opd.fc_opd.teacher_service import running_teacher_server


def _write_dataset(path, image_path, *, task_evidence="Look at the colored marker."):
    records = [
        {
            "question_id": "sample-0001",
            "images": [str(image_path)],
            "query": "What color is the marker?",
            "answer": "red",
            "free_caption": "A marker is visible.",
            "task_evidence": task_evidence,
        }
    ]
    path.write_text(json.dumps(records), encoding="utf-8")
    return records


def test_dry_run_writes_outputs_and_flags_task_evidence_leakage(tmp_path):
    dataset = tmp_path / "candidate.json"
    missing_image = tmp_path / "missing.jpg"
    _write_dataset(dataset, missing_image, task_evidence="The correct answer is red.")

    result = run_dataset_signal_audit(
        DatasetAuditConfig(
            dataset=dataset,
            dataset_type="vision_opd_json",
            source_dataset="vision_opd_6k",
            output_dir=tmp_path / "audit",
            dry_run=True,
            task_evidence_mode="question_conditioned_caption",
        ),
        tokenizer=ByteTokenizer(),
    )

    assert result.jsonl_path is not None
    assert result.summary_json_path is not None
    assert result.summary_tsv_path is not None
    assert result.summary_md_path is not None
    assert result.summary["num_samples_requested"] == 1
    assert result.summary["num_samples_scored"] == 0
    assert result.summary["image_missing_rate"] == 1.0
    assert result.samples[0]["leakage_warnings"] == ["task_evidence_contains_answer"]
    assert result.samples[0]["teacher_model_id"] == "dry_run"

    parsed = json.loads(result.jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert parsed["sample_uid"] == "vision_opd_6k:sample-0001"


def test_dry_run_default_task_evidence_does_not_leak_answer(tmp_path):
    dataset = tmp_path / "candidate.json"
    missing_image = tmp_path / "missing.jpg"
    _write_dataset(dataset, missing_image, task_evidence="The correct answer is red.")

    result = run_dataset_signal_audit(
        DatasetAuditConfig(
            dataset=dataset,
            dataset_type="vision_opd_json",
            source_dataset="vision_opd_6k",
            output_dir=tmp_path / "audit",
            dry_run=True,
        ),
        tokenizer=ByteTokenizer(),
    )

    assert result.samples[0]["leakage_warnings"] == []
    assert result.samples[0]["metadata"]["task_evidence_mode"] == "none"
    assert result.samples[0]["response_source"] == "fixed_audit_response"
    assert "red" not in result.samples[0]["response_text"].lower()
    assert result.samples[0]["response_text_hash"] == hash_text(result.samples[0]["response_text"])
    assert result.samples[0]["response_token_hash"] == hash_token_ids(
        result.samples[0]["response_token_ids"]
    )


def test_dry_run_materializes_degraded_images_and_counts_them_present(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    dataset = tmp_path / "candidate.json"
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), color=(0, 0, 255)).save(image)
    _write_dataset(dataset, image)

    result = run_dataset_signal_audit(
        DatasetAuditConfig(
            dataset=dataset,
            dataset_type="vision_opd_json",
            source_dataset="vision_opd_6k",
            output_dir=tmp_path / "audit",
            dry_run=True,
            materialize_degraded_images=True,
        ),
        tokenizer=ByteTokenizer(),
    )

    degraded_path = result.samples[0]["degraded_image_path"]
    assert degraded_path.endswith("image.gaussian_blur_s2.png")
    assert result.samples[0]["degraded_image_exists"] is True
    assert result.summary["degraded_image_missing_rate"] == 0.0


def test_synthetic_teacher_audit_computes_signal_summary_and_cosines(tmp_path):
    dataset = tmp_path / "candidate.json"
    image = tmp_path / "image.jpg"
    image.write_bytes(b"placeholder")
    _write_dataset(dataset, image)

    tokenizer = ByteTokenizer()
    with ExitStack() as stack:
        scorer = SyntheticTeacherScorer(
            vocab_size=320,
            top_k=16,
            tokenizer_hash=tokenizer_fingerprint(tokenizer),
        )
        server = stack.enter_context(running_teacher_server(scorer))
        host, port = server.server_address
        client = TeacherClient(
            f"http://{host}:{port}",
            expected_tokenizer_hash=tokenizer_fingerprint(tokenizer),
        )
        result = run_dataset_signal_audit(
            DatasetAuditConfig(
                dataset=dataset,
                dataset_type="vision_opd_json",
                source_dataset="geometry3k",
                output_dir=tmp_path / "audit",
                dry_run=False,
            ),
            tokenizer=tokenizer,
            teacher_client=client,
        )

    sample = result.samples[0]
    assert all(sample["condition_score_available"].values())
    assert "visual_detail" in sample["condition_signals"]
    assert "task_extraction" in sample["condition_signals"]
    assert sample["condition_signal_summary"]["visual_detail"]["mean"] is not None
    assert "cos_g_full_blur" in sample["gradient_cosines"]
    assert result.summary["num_samples_scored"] == 1
    assert result.summary["mean_entropy"]["full"] is not None
    assert result.summary["mean_full_vs_blur_divergence"] is not None
    assert result.summary["gradient_cosines"]["cos_g_full_blur"] is not None
    assert result.summary["gradient_cosine_diagnostic"]["label"] == "condition redundancy diagnostic"
    assert result.summary["gradient_cosine_diagnostic"]["is_ideal_alignment"] is False


def test_identical_fixed_audit_responses_trigger_provenance_warning(tmp_path):
    dataset = tmp_path / "candidate.json"
    records = [
        {
            "question_id": f"sample-{index}",
            "images": [str(tmp_path / f"image-{index}.jpg")],
            "query": f"Question {index}?",
            "answer": "red",
        }
        for index in range(2)
    ]
    dataset.write_text(json.dumps(records), encoding="utf-8")

    result = run_dataset_signal_audit(
        DatasetAuditConfig(
            dataset=dataset,
            dataset_type="vision_opd_json",
            source_dataset="vision_opd_6k",
            output_dir=tmp_path / "audit",
            dry_run=True,
        ),
        tokenizer=ByteTokenizer(),
    )

    assert result.summary["response_source_counts"] == {"fixed_audit_response": 2}
    assert result.summary["unique_response_text_hash_count"] == 1
    assert result.summary["all_responses_identical"] is True
    assert "all_responses_identical" in result.summary["response_provenance_warning"]


def test_dataset_target_source_is_labeled_not_student_rollout(tmp_path):
    dataset = tmp_path / "candidate.json"
    image = tmp_path / "image.jpg"
    image.write_bytes(b"placeholder")
    _write_dataset(dataset, image)

    result = run_dataset_signal_audit(
        DatasetAuditConfig(
            dataset=dataset,
            dataset_type="vision_opd_json",
            source_dataset="vision_opd_6k",
            output_dir=tmp_path / "audit",
            dry_run=True,
            response_source="dataset_target",
        ),
        tokenizer=ByteTokenizer(),
    )

    assert result.samples[0]["response_source"] == "dataset_target"
    assert result.samples[0]["response_text"] == "red"
    assert "student" not in result.samples[0]["response_source"]
    assert result.samples[0]["response_provenance_note"] == "diagnostic target scoring, not student rollout"


def test_nested_answer_metadata_is_preserved_but_not_used_by_fixed_response(tmp_path):
    dataset = tmp_path / "candidate.json"
    image = tmp_path / "image.jpg"
    image.write_bytes(b"placeholder")
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "sample-0001",
                    "image_path": str(image),
                    "prompt": {"role": "user", "content": "<image>\nWhat color?"},
                    "reward_model": {"ground_truth": "B"},
                    "extra_info": {"answer": "blue", "question": "What color?"},
                }
            ]
        ),
        encoding="utf-8",
    )

    result = run_dataset_signal_audit(
        DatasetAuditConfig(
            dataset=dataset,
            dataset_type="vision_opd_json",
            source_dataset="vision_opd_6k",
            output_dir=tmp_path / "audit",
            dry_run=True,
        ),
        tokenizer=ByteTokenizer(),
    )

    sample = result.samples[0]
    assert sample["answer_available"] is True
    assert sample["answer_source"] == "reward_model.ground_truth"
    assert sample["reward_model_ground_truth"] == "B"
    assert sample["extra_info_answer"] == "blue"
    assert "B" not in sample["response_text"]


def test_load_candidate_records_accepts_jsonl_auto(tmp_path):
    dataset = tmp_path / "candidate.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "x",
                "image": "image.jpg",
                "question": "What is shown?",
                "answer": "a marker",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_candidate_records(dataset, "auto")
    assert records[0]["question"] == "What is shown?"


def test_detect_task_evidence_leakage_normalizes_case_and_punctuation():
    warnings = detect_task_evidence_leakage(
        answer="Red!",
        task_evidence="The answer is red.",
    )
    assert warnings == ["task_evidence_contains_answer"]

import json
from contextlib import ExitStack

from dual_track_opd.fc_opd.dataset_signal_audit import (
    DatasetAuditConfig,
    detect_task_evidence_leakage,
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

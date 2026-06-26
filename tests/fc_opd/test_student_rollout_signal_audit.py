import json
from contextlib import ExitStack

from dual_track_opd.fc_opd.offline_scoring import ByteTokenizer
from dual_track_opd.fc_opd.student_rollout_signal_audit import (
    FixedFakeRolloutGenerator,
    StudentRolloutAuditConfig,
    run_student_rollout_signal_audit,
)
from dual_track_opd.fc_opd.teacher_client import TeacherClient
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint
from dual_track_opd.fc_opd.teacher_scorer import SyntheticTeacherScorer
from dual_track_opd.fc_opd.teacher_service import running_teacher_server


def test_fake_student_rollout_produces_prompt_times_k_rows(tmp_path):
    dataset = tmp_path / "candidate.json"
    image = tmp_path / "image.jpg"
    image.write_bytes(b"placeholder")
    records = [
        {
            "id": f"sample-{index}",
            "image_path": str(image),
            "bbox_images": [str(tmp_path / "crop.jpg")],
            "question": f"What is shown in sample {index}?",
            "answer": "A",
        }
        for index in range(2)
    ]
    dataset.write_text(json.dumps(records), encoding="utf-8")

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
        result = run_student_rollout_signal_audit(
            StudentRolloutAuditConfig(
                dataset=dataset,
                dataset_type="vision_opd_json",
                source_dataset="vision-opd-6k",
                limit=2,
                rollouts_per_prompt=3,
                output_dir=tmp_path / "audit",
                student_model_path="fake/student",
            ),
            rollout_generator=FixedFakeRolloutGenerator(tokenizer),
            teacher_client=client,
        )

    assert result.expected_rows == 6
    assert result.actual_rows == 6
    assert result.summary["expected_rows"] == 6
    assert result.summary["actual_rows"] == 6
    assert {row["response_source"] for row in result.samples} == {"student_rollout"}
    assert {row["rollout_id"] for row in result.samples} == {0, 1, 2}
    assert all(row["bbox_metadata_unused_by_default"] is True for row in result.samples)
    assert result.summary["response_source_counts"] == {"student_rollout": 6}
    assert result.jsonl_path is not None and result.jsonl_path.is_file()

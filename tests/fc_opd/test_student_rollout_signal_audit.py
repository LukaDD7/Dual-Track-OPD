import json
from contextlib import ExitStack

from dual_track_opd.fc_opd.offline_scoring import ByteTokenizer
from dual_track_opd.fc_opd.student_rollout_signal_audit import (
    FixedFakeRolloutGenerator,
    StudentRolloutAuditConfig,
    build_rollout_prompt,
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
    assert all(
        row["crop_bbox_policy"] == "metadata_only_default_no_crop_condition"
        for row in result.samples
    )
    assert result.summary["response_source_counts"] == {"student_rollout": 6}
    assert result.summary["unique_response_per_prompt_mean"] == 3
    assert result.summary["duplicate_rollout_rate"] == 0
    assert all("cos_g_full_blur" in row["gradient_cosines"] for row in result.samples)
    assert result.jsonl_path is not None and result.jsonl_path.is_file()


def test_student_rollout_rows_include_image_exists_and_degraded_exists(tmp_path):
    import pytest

    pytest.importorskip("PIL")
    from PIL import Image

    dataset = tmp_path / "candidate.json"
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(image)
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "sample-0",
                    "image_path": str(image),
                    "question": "What color?",
                }
            ]
        ),
        encoding="utf-8",
    )

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
                limit=1,
                rollouts_per_prompt=1,
                materialize_degraded_images=True,
            ),
            rollout_generator=FixedFakeRolloutGenerator(tokenizer),
            teacher_client=client,
        )

    row = result.samples[0]
    assert row["image_exists"] is True
    assert row["degraded_image_exists"] is True
    assert result.summary["image_missing_rate"] == 0.0
    assert result.summary["degraded_image_missing_rate"] == 0.0


def test_rollout_prompt_structured_contains_xml_and_no_gold_answer():
    prompt = build_rollout_prompt(
        "What color?\nA. red\nB. blue\nAnswer with the option's letter.",
        response_format="fc_opd_structured",
    )
    assert "<visual_evidence>" in prompt.text
    assert "<reasoning>" in prompt.text
    assert "<answer>" in prompt.text
    assert "gold answer" in prompt.text
    assert "Correct answer:" not in prompt.text


def test_answer_only_mode_is_labeled_mechanical_and_short(tmp_path):
    dataset = tmp_path / "candidate.json"
    image = tmp_path / "image.jpg"
    image.write_bytes(b"placeholder")
    dataset.write_text(
        json.dumps([{"id": "sample-0", "image_path": str(image), "question": "What color?"}]),
        encoding="utf-8",
    )
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
                rollouts_per_prompt=2,
                rollout_response_format="answer_only",
                min_response_tokens_for_warning=16,
            ),
            rollout_generator=FixedFakeRolloutGenerator(tokenizer),
            teacher_client=client,
        )

    assert all(row["rollout_response_format"] == "answer_only" for row in result.samples)
    assert "answer_only_is_mechanical_smoke_only" in result.summary["rollout_diversity_warnings"]
    assert result.summary["short_response_rate"] == 1.0
    assert result.samples[0]["metadata"]["rollout_response_format_note"].startswith("mechanical")


class DuplicateFakeRolloutGenerator(FixedFakeRolloutGenerator):
    def generate(self, *, question, image_path, prompt_text, seed):
        del question, image_path, prompt_text, seed
        return "A. duplicate"


def test_duplicate_rollout_diagnostics_detect_identical_prompt_rollouts(tmp_path):
    dataset = tmp_path / "candidate.json"
    image = tmp_path / "image.jpg"
    image.write_bytes(b"placeholder")
    dataset.write_text(
        json.dumps([{"id": "sample-0", "image_path": str(image), "question": "What color?"}]),
        encoding="utf-8",
    )
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
                rollouts_per_prompt=3,
                rollout_response_format="answer_only",
            ),
            rollout_generator=DuplicateFakeRolloutGenerator(tokenizer),
            teacher_client=client,
        )

    assert [row["generation_seed"] for row in result.samples] == [42, 43, 44]
    assert result.summary["duplicate_rollout_rate"] == 2 / 3
    assert result.summary["all_rollouts_identical_per_prompt_count"] == 1
    assert "duplicate_rollout_rate_high" in result.summary["rollout_diversity_warnings"]

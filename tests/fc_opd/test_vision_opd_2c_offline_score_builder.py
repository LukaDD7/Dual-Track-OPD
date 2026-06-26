import json
from contextlib import ExitStack

from dual_track_opd.fc_opd.offline_loss import offline_record_to_tensors
from dual_track_opd.fc_opd.offline_scoring import ByteTokenizer
from dual_track_opd.fc_opd.student_rollout_signal_audit import FixedFakeRolloutGenerator
from dual_track_opd.fc_opd.teacher_client import TeacherClient
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint
from dual_track_opd.fc_opd.teacher_scorer import SyntheticTeacherScorer
from dual_track_opd.fc_opd.teacher_service import running_teacher_server
from dual_track_opd.fc_opd.vision_opd_2c_offline_builder import (
    CROP_BBOX_POLICY,
    VisionOPD2COfflineScoreConfig,
    run_vision_opd_2c_offline_score_builder,
    validate_vision_opd_2c_offline_scores,
)


def _teacher_client(stack, tokenizer):
    scorer = SyntheticTeacherScorer(
        vocab_size=320,
        top_k=16,
        tokenizer_hash=tokenizer_fingerprint(tokenizer),
    )
    server = stack.enter_context(running_teacher_server(scorer))
    host, port = server.server_address
    return TeacherClient(
        f"http://{host}:{port}",
        expected_tokenizer_hash=tokenizer_fingerprint(tokenizer),
    )


def test_vision_opd_2c_builder_writes_trainable_rows(tmp_path):
    dataset = tmp_path / "train.json"
    image = tmp_path / "image.jpg"
    crop = tmp_path / "crop.jpg"
    image.write_bytes(b"placeholder")
    crop.write_bytes(b"placeholder")
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": f"sample-{index}",
                    "image_path": str(image),
                    "bbox_images": [str(crop)],
                    "question": f"What is shown in sample {index}?",
                    "answer": "A",
                }
                for index in range(2)
            ]
        ),
        encoding="utf-8",
    )

    tokenizer = ByteTokenizer()
    output_jsonl = tmp_path / "scores.jsonl"
    summary_json = tmp_path / "summary.json"
    with ExitStack() as stack:
        client = _teacher_client(stack, tokenizer)
        result = run_vision_opd_2c_offline_score_builder(
            VisionOPD2COfflineScoreConfig(
                dataset=dataset,
                dataset_type="vision_opd_json",
                output_jsonl=output_jsonl,
                summary_json=summary_json,
                limit=2,
                rollouts_per_prompt=2,
                student_model_path="fake/student",
                allow_red_box_contaminated_images=True,
            ),
            rollout_generator=FixedFakeRolloutGenerator(tokenizer),
            teacher_client=client,
        )

    assert result.summary["expected_rows"] == 4
    assert result.summary["actual_rows"] == 4
    assert output_jsonl.is_file()
    assert summary_json.is_file()

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert {row["response_source"] for row in rows} == {"student_rollout"}
    assert {tuple(row["conditions"]) for row in rows} == {("full", "blur")}
    assert all(row["crop_bbox_policy"] == CROP_BBOX_POLICY for row in rows)
    assert all(row["bbox_metadata_unused_by_default"] is True for row in rows)
    assert all(row["red_box_contaminated"] is True for row in rows)
    assert all(row["localization_cued_ablation"] is True for row in rows)
    assert all(row["not_main_experiment"] is True for row in rows)
    assert result.summary["red_box_contaminated"] is True
    assert result.summary["localization_cued_ablation"] is True
    assert result.summary["not_main_experiment"] is True
    assert all(row["condition_inputs"]["full_image"]["path"] == str(image) for row in rows)
    assert all(row["condition_inputs"]["full_image"]["path"] != str(crop) for row in rows)

    for row in rows:
        tensors = offline_record_to_tensors(row)
        assert set(condition.value for condition in tensors.teacher_scores) == {"full", "blur"}
        assert tensors.seq_len == len(row["response_token_ids"])
        for block in row["condition_scores"].values():
            assert len(block["token_ids"]) == len(row["response_token_ids"])
            assert all(len(token_row) == 16 for token_row in block["token_ids"])
            assert len(block["log_probs"]) == len(row["response_token_ids"])

    validation = validate_vision_opd_2c_offline_scores(output_jsonl)
    assert validation["valid"], validation["errors"]


def test_builder_resume_skips_existing_rollout_uids(tmp_path):
    dataset = tmp_path / "train.json"
    image = tmp_path / "image.jpg"
    image.write_bytes(b"placeholder")
    dataset.write_text(
        json.dumps([{"id": "sample-0", "image_path": str(image), "question": "What color?"}]),
        encoding="utf-8",
    )
    tokenizer = ByteTokenizer()
    output_jsonl = tmp_path / "scores.jsonl"
    summary_json = tmp_path / "summary.json"
    config = VisionOPD2COfflineScoreConfig(
        dataset=dataset,
        dataset_type="vision_opd_json",
        output_jsonl=output_jsonl,
        summary_json=summary_json,
        limit=1,
        rollouts_per_prompt=2,
        student_model_path="fake/student",
        resume=True,
        allow_red_box_contaminated_images=True,
    )

    with ExitStack() as stack:
        client = _teacher_client(stack, tokenizer)
        run_vision_opd_2c_offline_score_builder(
            config,
            rollout_generator=FixedFakeRolloutGenerator(tokenizer),
            teacher_client=client,
        )
        run_vision_opd_2c_offline_score_builder(
            config,
            rollout_generator=FixedFakeRolloutGenerator(tokenizer),
            teacher_client=client,
        )

    assert len(output_jsonl.read_text(encoding="utf-8").splitlines()) == 2


def test_vision_opd_builder_hard_stops_without_contaminated_override(tmp_path):
    dataset = tmp_path / "train.json"
    dataset.write_text("[]", encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="red-box contamination"):
        VisionOPD2COfflineScoreConfig(
            dataset=dataset,
            dataset_type="vision_opd_parquet",
            output_jsonl=tmp_path / "scores.jsonl",
            summary_json=tmp_path / "summary.json",
            limit=4,
        )

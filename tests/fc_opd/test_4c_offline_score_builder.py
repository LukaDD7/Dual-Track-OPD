import json
import math
from contextlib import ExitStack
from types import SimpleNamespace

from PIL import Image

from dual_track_opd.fc_opd.evidence_generation import (
    EvidenceGenerationConfig,
    TemplateEvidenceGenerator,
    run_evidence_generation,
)
from dual_track_opd.fc_opd.four_condition_offline_builder import (
    FourConditionOfflineBuilderConfig,
    materialize_degraded_image,
    summarize_rows,
    validate_four_condition_rows,
    run_four_condition_offline_builder,
)
from dual_track_opd.fc_opd.geometry3k_adapter import load_geometry3k_records
from dual_track_opd.fc_opd.offline_loss import offline_record_to_tensors
from dual_track_opd.fc_opd.offline_scoring import ByteTokenizer
from dual_track_opd.fc_opd.student_rollout_signal_audit import FixedFakeRolloutGenerator
from dual_track_opd.fc_opd.teacher_client import TeacherClient
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint
from dual_track_opd.fc_opd.teacher_scorer import SyntheticTeacherScorer
from dual_track_opd.fc_opd.teacher_service import running_teacher_server
import pytest


def _dataset(tmp_path):
    image = tmp_path / "diagram.png"
    Image.new("RGB", (20, 20), "white").save(image)
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
    return dataset


def _official_geometry_sample(tmp_path):
    root = tmp_path / "unzipped"
    sample_dir = root / "train" / "train" / "0"
    sample_dir.mkdir(parents=True)
    Image.new("RGB", (20, 20), "white").save(sample_dir / "img_diagram.png")
    (sample_dir / "data.json").write_text(
        json.dumps(
            {
                "id": "0",
                "compact_text": "Use the diagram.",
                "choices": ["30", "45"],
                "answer": "B",
                "data_type": "train",
            }
        ),
        encoding="utf-8",
    )
    (sample_dir / "logic_form.json").write_text("{}", encoding="utf-8")
    return root, sample_dir


class FailingRolloutGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, *, question, image_path, prompt_text, seed):
        del question, image_path, prompt_text, seed
        raise RuntimeError("mock rollout failed")


class EmptyRolloutGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, *, question, image_path, prompt_text, seed):
        del question, image_path, prompt_text, seed
        return ""


def _write_evidence_cache(tmp_path, dataset, *, condition_set="4c-clean"):
    evidence_jsonl = tmp_path / "evidence.jsonl"
    run_evidence_generation(
        EvidenceGenerationConfig(
            dataset=dataset,
            dataset_type="geometry3k",
            output_jsonl=evidence_jsonl,
            summary_json=tmp_path / "evidence_summary.json",
            limit=1,
            condition_set=condition_set,
        ),
        generator=TemplateEvidenceGenerator(),
    )
    return evidence_jsonl


def _fake_teacher_client():
    return SimpleNamespace(metadata=SimpleNamespace(model_id="fake-teacher", top_k=8))


def test_default_degradation_cache_does_not_pollute_geometry3k_sample_dir(tmp_path, monkeypatch):
    root, sample_dir = _official_geometry_sample(tmp_path)
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("DTOPD_OUTPUT_ROOT", str(output_root))
    before = load_geometry3k_records(root)[0]["image_path"]

    config = FourConditionOfflineBuilderConfig(
        dataset=root,
        evidence_cache=tmp_path / "evidence.jsonl",
        output_jsonl=tmp_path / "scores.jsonl",
        summary_json=tmp_path / "summary.json",
    )
    first_degraded = materialize_degraded_image(before, config)
    second_degraded = materialize_degraded_image(before, config)
    after = load_geometry3k_records(root)[0]["image_path"]

    assert before == after == str((sample_dir / "img_diagram.png").resolve(strict=False))
    assert first_degraded == second_degraded
    assert str(first_degraded).startswith(str(output_root / "fc_opd" / "degraded_images"))
    assert not (sample_dir / "img_diagram.lowres_10pct_nearest.png").exists()


def test_rollout_exception_is_counted_and_summarized(tmp_path):
    dataset = _dataset(tmp_path)
    evidence_jsonl = _write_evidence_cache(tmp_path, dataset)
    summary_json = tmp_path / "summary.json"

    result = run_four_condition_offline_builder(
        FourConditionOfflineBuilderConfig(
            dataset=dataset,
            evidence_cache=evidence_jsonl,
            output_jsonl=tmp_path / "scores.jsonl",
            summary_json=summary_json,
            limit=1,
                rollouts_per_prompt=1,
                allow_empty_output=True,
                max_rollout_attempts=1,
            ),
        rollout_generator=FailingRolloutGenerator(ByteTokenizer()),
        teacher_client=_fake_teacher_client(),
    )

    assert result.rows == []
    assert result.summary["selected_records"] == 1
    assert result.summary["evidence_rows_loaded"] == 1
    assert result.summary["evidence_rows_matched"] == 1
    assert result.summary["rollout_attempt_count"] == 1
    assert result.summary["rollout_exception_count"] == 1
    assert result.summary["row_write_count"] == 0
    assert result.summary["skipped_examples_top20"][0]["stage"] == "student_rollout"
    assert "RuntimeError: mock rollout failed" in result.summary["skipped_examples_top20"][0]["reason"]
    assert summary_json.is_file()


def test_all_rows_skipped_raises_runtime_error_after_writing_summary(tmp_path):
    dataset = _dataset(tmp_path)
    evidence_jsonl = _write_evidence_cache(tmp_path, dataset)
    summary_json = tmp_path / "summary_empty_error.json"

    with pytest.raises(RuntimeError, match="wrote zero rows"):
        run_four_condition_offline_builder(
            FourConditionOfflineBuilderConfig(
                dataset=dataset,
                evidence_cache=evidence_jsonl,
                output_jsonl=tmp_path / "scores_empty_error.jsonl",
                summary_json=summary_json,
                limit=1,
                rollouts_per_prompt=1,
                max_rollout_attempts=1,
            ),
            rollout_generator=EmptyRolloutGenerator(ByteTokenizer()),
            teacher_client=_fake_teacher_client(),
        )

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["validation_errors_top10"] == ["no rows produced"]
    assert summary["rollout_empty_count"] == 1
    assert summary["skipped_examples_top20"][0]["reason"] == "attempt=1: empty rollout text"


def test_allow_empty_output_permits_empty_summary(tmp_path):
    dataset = _dataset(tmp_path)
    evidence_jsonl = _write_evidence_cache(tmp_path, dataset)

    result = run_four_condition_offline_builder(
        FourConditionOfflineBuilderConfig(
            dataset=dataset,
            evidence_cache=evidence_jsonl,
            output_jsonl=tmp_path / "scores_empty_allowed.jsonl",
            summary_json=tmp_path / "summary_empty_allowed.json",
            limit=1,
                rollouts_per_prompt=1,
                allow_empty_output=True,
                max_rollout_attempts=1,
            ),
        rollout_generator=EmptyRolloutGenerator(ByteTokenizer()),
        teacher_client=_fake_teacher_client(),
    )

    assert result.rows == []
    assert result.summary["rollout_empty_count"] == 1
    assert result.summary["row_write_count"] == 0
    assert result.summary["validation_valid"] is False
    assert result.summary["skipped_examples_top20"][0] == {
        "sample_uid": "geometry3k:g1:rollout-0",
        "stage": "student_rollout",
        "reason": "attempt=1: empty rollout text",
    }


def test_rollout_exhausted_writes_malformed_fallback_row(tmp_path):
    dataset = _dataset(tmp_path)
    evidence_jsonl = _write_evidence_cache(tmp_path, dataset)
    tokenizer = ByteTokenizer()
    with ExitStack() as stack:
        scorer = SyntheticTeacherScorer(
            vocab_size=320,
            top_k=8,
            tokenizer_hash=tokenizer_fingerprint(tokenizer),
        )
        server = stack.enter_context(running_teacher_server(scorer))
        host, port = server.server_address
        client = TeacherClient(
            f"http://{host}:{port}",
            expected_tokenizer_hash=tokenizer_fingerprint(tokenizer),
        )
        result = run_four_condition_offline_builder(
            FourConditionOfflineBuilderConfig(
                dataset=dataset,
                evidence_cache=evidence_jsonl,
                output_jsonl=tmp_path / "scores_fallback.jsonl",
                summary_json=tmp_path / "summary_fallback.json",
                limit=1,
                rollouts_per_prompt=1,
                max_rollout_attempts=1,
                verifier_gate="geometry3k_verifier",
            ),
            rollout_generator=EmptyRolloutGenerator(tokenizer),
            teacher_client=client,
        )

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row["response_source"] == "fallback_malformed_rollout"
    assert row["rollout_fallback_malformed"] is True
    assert row["verifier"]["malformed"] is True
    assert result.summary["rollout_exhausted_count"] == 1
    assert result.summary["row_write_count"] == 1


def test_4c_builder_writes_trainable_full_degraded_free_task_rows(tmp_path):
    dataset = _dataset(tmp_path)
    evidence_jsonl = tmp_path / "evidence.jsonl"
    run_evidence_generation(
        EvidenceGenerationConfig(
            dataset=dataset,
            dataset_type="geometry3k",
            output_jsonl=evidence_jsonl,
            summary_json=tmp_path / "evidence_summary.json",
            limit=1,
        ),
        generator=TemplateEvidenceGenerator(),
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
        output = tmp_path / "scores.jsonl"
        result = run_four_condition_offline_builder(
            FourConditionOfflineBuilderConfig(
                dataset=dataset,
                evidence_cache=evidence_jsonl,
                output_jsonl=output,
                summary_json=tmp_path / "summary.json",
                limit=1,
                rollouts_per_prompt=2,
                student_model_path="fake/student",
                degraded_dir=str(tmp_path / "degraded"),
            ),
            rollout_generator=FixedFakeRolloutGenerator(tokenizer),
            teacher_client=client,
        )

    assert result.summary["conditions"] == ["full", "degraded", "free", "task"]
    assert result.summary["degraded_mode"] == "lowres_10pct_nearest"
    assert len(result.rows) == 2
    for row in result.rows:
        assert row["conditions"] == ["full", "degraded", "free", "task"]
        assert set(row["condition_scores"]) == {"full", "degraded", "free", "task"}
        assert row["outcome_metadata"]["correctness_used_for_prompt"] is False
        assert row["outcome_metadata"]["correctness_used_for_evidence_generation"] is False
        tensors = offline_record_to_tensors(row)
        assert tensors.seq_len == len(row["response_token_ids"])
        for block in row["condition_scores"].values():
            assert len(block["token_ids"]) == len(row["response_token_ids"])
            assert all(len(token_row) == 16 for token_row in block["token_ids"])
            assert block["top_k"] == 16

    validation = validate_four_condition_rows(output)
    assert validation["valid"], validation["errors"]


def test_6c_builder_writes_expanded_condition_schema(tmp_path):
    dataset = _dataset(tmp_path)
    evidence_jsonl = tmp_path / "evidence_6c.jsonl"
    run_evidence_generation(
        EvidenceGenerationConfig(
            dataset=dataset,
            dataset_type="geometry3k",
            output_jsonl=evidence_jsonl,
            summary_json=tmp_path / "evidence_6c_summary.json",
            limit=1,
            condition_set="6c-solve",
        ),
        generator=TemplateEvidenceGenerator(),
    )

    tokenizer = ByteTokenizer()
    with ExitStack() as stack:
        scorer = SyntheticTeacherScorer(
            vocab_size=320,
            top_k=8,
            tokenizer_hash=tokenizer_fingerprint(tokenizer),
        )
        server = stack.enter_context(running_teacher_server(scorer))
        host, port = server.server_address
        client = TeacherClient(
            f"http://{host}:{port}",
            expected_tokenizer_hash=tokenizer_fingerprint(tokenizer),
        )
        output = tmp_path / "scores_6c.jsonl"
        result = run_four_condition_offline_builder(
            FourConditionOfflineBuilderConfig(
                dataset=dataset,
                evidence_cache=evidence_jsonl,
                output_jsonl=output,
                summary_json=tmp_path / "summary_6c.json",
                limit=1,
                rollouts_per_prompt=1,
                student_model_path="fake/student",
                degraded_dir=str(tmp_path / "degraded_6c"),
                condition_set="6c-solve",
                rollout_response_format="fc_opd_structured_v2",
                enable_student_condition_scoring=True,
                student_deficit_gate=True,
                verifier_gate="geometry3k_verifier",
                routing_mode="student_deficit_chunk_gated",
                grouped_loss_schema="capability_chunk_v1",
            ),
            rollout_generator=FixedFakeRolloutGenerator(tokenizer),
            teacher_client=client,
        )

    expected = ["full", "degraded", "free", "task_visible", "task_infer", "task_solve"]
    assert result.summary["condition_set_name"] == "6c-solve"
    assert result.summary["conditions"] == expected
    assert result.summary["chunk_parse_success_rate"] == 1.0
    assert result.summary["gate_ready_fields_available"] is True
    assert math.isfinite(result.summary["delta_means"]["visual_detail_delta"])
    assert result.summary["visual_detail_delta_count"] == 1
    assert result.summary["visual_detail_delta_invalid_count"] == 0
    assert result.summary["validation_valid"] is True
    assert result.summary["validation_error_count"] == 0
    assert result.summary["student_condition_score_success_rate"]["task_solve"] == 1.0
    assert result.summary["verifier_outcome_counts"]["wrong_but_format_valid"] == 1
    assert result.summary["wrong_valid_rollout_opd_weight_sum"] > result.summary["correct_rollout_opd_weight_sum"]
    assert result.summary["grouped_loss_ready"] is True
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row["conditions"] == expected
    assert set(row["condition_scores"]) == set(expected)
    assert set(row["student_condition_scores"]) == set(expected)
    assert row["verifier"]["format_valid"] is True
    assert row["verifier_learning_value_gate"]["outcome_class"] == "wrong_but_format_valid"
    assert 0.0 < row["verifier_learning_value_gate"]["chunk_gates"]["diagram_inference"] < 1.0
    assert row["verifier_learning_value_gate"]["chunk_gates"]["visible_evidence"] > row["verifier_learning_value_gate"]["chunk_gates"]["diagram_inference"]
    assert row["grouped_loss_plan"]["solve"] == ["solving"]
    assert row["routing_mode"] == "student_deficit_chunk_gated"
    assert row["chunk_spans"]["token_counts"]["diagram_inference"] > 0
    assert set(row["capability_scores"]) == {
        "visual_detail",
        "evidence_selection",
        "visual_text_inference",
        "solving",
    }

    validation = validate_four_condition_rows(output, condition_set="6c-solve")
    assert validation["valid"], validation["errors"]


def test_summary_reports_chunk_and_delta_validation_errors(tmp_path):
    config = FourConditionOfflineBuilderConfig(
        dataset=tmp_path / "dataset.json",
        evidence_cache=tmp_path / "evidence.jsonl",
        output_jsonl=tmp_path / "scores.jsonl",
        summary_json=tmp_path / "summary.json",
        condition_set="6c-solve",
        rollout_response_format="fc_opd_structured_v2",
        max_new_tokens=384,
    )
    row = {
        "sample_uid": "row-1",
        "prompt_sample_uid": "prompt-1",
        "conditions": ["full", "degraded", "free", "task_visible", "task_infer", "task_solve"],
        "response_token_count": 384,
        "response_text_hash": "abc",
        "response_source": "student_rollout",
        "tokenizer_hash": "tok",
        "chunk_spans": {
            "format_valid": False,
            "errors": ["reasoning:close_tag_count=0", "answer:open_tag_count=0"],
            "token_counts": {"visible_evidence": 4, "diagram_inference": 3, "reasoning": 377, "answer": 0},
        },
        "condition_signal_summary": {
            "visual_detail_delta": {"mean": float("nan"), "p50": None, "p90": None},
            "task_selection_delta": {"mean": 0.1, "p50": 0.1, "p90": 0.1},
            "diagram_infer_delta": {"mean": 0.2, "p50": 0.2, "p90": 0.2},
            "solve_delta": {"mean": 0.3, "p50": 0.3, "p90": 0.3},
        },
    }

    summary = summarize_rows(
        [row],
        config=config,
        teacher_client=SimpleNamespace(metadata=SimpleNamespace(model_id="teacher")),
    )

    assert summary["validation_valid"] is False
    assert summary["validation_error_count"] > 0
    assert "row-1" in summary["rows_with_chunk_parse_failure"]
    assert "row-1" in summary["rows_with_invalid_delta"]
    assert summary["visual_detail_delta_count"] == 0
    assert summary["visual_detail_delta_invalid_count"] == 1
    assert summary["delta_means"]["visual_detail_delta"] is None
    assert not any(
        isinstance(value, float) and math.isnan(value)
        for value in summary["delta_means"].values()
    )
    assert "reasoning:close_tag_count=0" in summary["validation_errors_top10"][0]
    assert summary["max_new_tokens_recommendation"]

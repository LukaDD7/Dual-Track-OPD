import json
from contextlib import ExitStack

from PIL import Image

from dual_track_opd.fc_opd.evidence_generation import (
    EvidenceGenerationConfig,
    TemplateEvidenceGenerator,
    run_evidence_generation,
)
from dual_track_opd.fc_opd.four_condition_offline_builder import (
    FourConditionOfflineBuilderConfig,
    validate_four_condition_rows,
    run_four_condition_offline_builder,
)
from dual_track_opd.fc_opd.offline_loss import offline_record_to_tensors
from dual_track_opd.fc_opd.offline_scoring import ByteTokenizer
from dual_track_opd.fc_opd.student_rollout_signal_audit import FixedFakeRolloutGenerator
from dual_track_opd.fc_opd.teacher_client import TeacherClient
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint
from dual_track_opd.fc_opd.teacher_scorer import SyntheticTeacherScorer
from dual_track_opd.fc_opd.teacher_service import running_teacher_server


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

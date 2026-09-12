import json

import pytest

from dual_track_opd.fc_opd.conditions import Condition
from dual_track_opd.fc_opd.offline_scoring import (
    ByteTokenizer,
    OfflineScoringConfig,
    build_protocol_smoke_response,
    derive_degraded_path,
    iter_offline_scores,
    load_student_responses,
    load_vision_opd_records,
    make_smoke_dataset,
    write_offline_scores,
)
from dual_track_opd.fc_opd.teacher_client import TeacherClient
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint
from dual_track_opd.fc_opd.teacher_scorer import SyntheticTeacherScorer
from dual_track_opd.fc_opd.teacher_service import running_teacher_server

TOP_K = 32
CONDITIONS = (Condition.FULL, Condition.BLUR, Condition.FREE, Condition.TASK)


def _tokenizer_and_client(stack):
    tokenizer = ByteTokenizer()
    scorer = SyntheticTeacherScorer(
        vocab_size=320,
        top_k=TOP_K,
        tokenizer_hash=tokenizer_fingerprint(tokenizer),
    )
    server = stack.enter_context(running_teacher_server(scorer))
    host, port = server.server_address
    client = TeacherClient(
        f"http://{host}:{port}",
        expected_tokenizer_hash=tokenizer_fingerprint(tokenizer),
    )
    return tokenizer, client


def test_byte_tokenizer_round_trips_and_fingerprint_is_stable():
    tokenizer = ByteTokenizer()
    text = "<answer>1</answer>"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer_fingerprint(tokenizer) == tokenizer_fingerprint(ByteTokenizer())


def test_derive_degraded_path_records_sigma():
    path = derive_degraded_path("/data/images/0001.jpg", 2.0, None)
    assert path.endswith("0001.gaussian_blur_s2.jpg")
    scoped = derive_degraded_path("/data/images/0001.jpg", 1.5, "/blur")
    assert scoped == "/blur/0001.gaussian_blur_s1_5.jpg"


def test_protocol_smoke_dataset_yields_four_conditions_with_topk_shapes():
    from contextlib import ExitStack

    with ExitStack() as stack:
        tokenizer, client = _tokenizer_and_client(stack)
        records = make_smoke_dataset(16, dataset_name="vstar")
        config = OfflineScoringConfig(source_dataset="vstar", conditions=CONDITIONS)
        scored = list(
            iter_offline_scores(
                records,
                config=config,
                tokenizer=tokenizer,
                teacher_client=client,
                mode="protocol_smoke",
            )
        )

    assert len(scored) == 16
    for record in scored:
        payload = record.payload
        scores = payload["condition_scores"]
        assert set(scores) == {c.value for c in CONDITIONS}
        seq_len = len(payload["response_token_ids"])
        for condition in CONDITIONS:
            block = scores[condition.value]
            assert len(block["token_ids"]) == seq_len
            assert all(len(row) == TOP_K for row in block["token_ids"])
            assert len(block["log_probs"]) == seq_len
            assert all(len(row) == TOP_K for row in block["log_probs"])
            assert len(block["tail_log_prob"]) == seq_len
            assert len(block["entropy"]) == seq_len
        signals = payload["condition_signals"]
        assert "visual_detail" in signals and len(signals["visual_detail"]) == seq_len
        assert "task_extraction" in signals and len(signals["task_extraction"]) == seq_len
        assert payload["protocol_version"]
        assert payload["tokenizer_hash"] == tokenizer_fingerprint(tokenizer)
        assert payload["metadata"]["top_k"] == TOP_K


def test_protocol_smoke_response_parses_into_valid_chunks():
    from contextlib import ExitStack

    with ExitStack() as stack:
        tokenizer, client = _tokenizer_and_client(stack)
        config = OfflineScoringConfig(source_dataset="vstar", conditions=CONDITIONS)
        scored = list(
            iter_offline_scores(
                make_smoke_dataset(1),
                config=config,
                tokenizer=tokenizer,
                teacher_client=client,
                mode="protocol_smoke",
            )
        )

    chunk_spans = scored[0].payload["chunk_spans"]
    assert chunk_spans["format_valid"] is True
    assert chunk_spans["errors"] == []
    assert chunk_spans["answer"], "answer chunk should map to at least one token span"


def test_student_mode_uses_provided_token_ids(tmp_path):
    from contextlib import ExitStack

    tokenizer = ByteTokenizer()
    text = build_protocol_smoke_response({"answer": "left"})
    student_path = tmp_path / "student.jsonl"
    student_path.write_text(
        json.dumps(
            {
                "question_id": "vstar-0000",
                "response_text": text,
                "response_token_ids": tokenizer.encode(text),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with ExitStack() as stack:
        tokenizer, client = _tokenizer_and_client(stack)
        config = OfflineScoringConfig(source_dataset="vstar", conditions=CONDITIONS)
        scored = list(
            iter_offline_scores(
                make_smoke_dataset(1),
                config=config,
                tokenizer=tokenizer,
                teacher_client=client,
                mode="student",
                student_responses=load_student_responses(student_path),
            )
        )

    assert scored[0].payload["response_token_ids"] == tokenizer.encode(text)
    assert scored[0].payload["metadata"]["response_mode"] == "student"


def test_student_mode_missing_response_raises():
    from contextlib import ExitStack

    with ExitStack() as stack:
        tokenizer, client = _tokenizer_and_client(stack)
        config = OfflineScoringConfig(source_dataset="vstar", conditions=CONDITIONS)
        with pytest.raises(KeyError):
            list(
                iter_offline_scores(
                    make_smoke_dataset(1),
                    config=config,
                    tokenizer=tokenizer,
                    teacher_client=client,
                    mode="student",
                    student_responses={"other-id": {"response_text": "x"}},
                )
            )


def test_write_offline_scores_round_trips_jsonl(tmp_path):
    from contextlib import ExitStack

    with ExitStack() as stack:
        tokenizer, client = _tokenizer_and_client(stack)
        config = OfflineScoringConfig(source_dataset="vstar", conditions=CONDITIONS)
        scored = list(
            iter_offline_scores(
                make_smoke_dataset(4),
                config=config,
                tokenizer=tokenizer,
                teacher_client=client,
                mode="protocol_smoke",
            )
        )

    result = write_offline_scores(scored, output_dir=tmp_path, filename="scores")
    assert result.jsonl_path is not None
    lines = result.jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    parsed = [json.loads(line) for line in lines]
    assert {record["sample_uid"] for record in parsed} == {
        f"vstar:vstar-{index:04d}" for index in range(4)
    }


def test_load_vision_opd_records_accepts_json_array(tmp_path):
    dataset = tmp_path / "ds.json"
    dataset.write_text(json.dumps(make_smoke_dataset(3)), encoding="utf-8")
    records = load_vision_opd_records(dataset)
    assert len(records) == 3
    assert records[0]["query"]

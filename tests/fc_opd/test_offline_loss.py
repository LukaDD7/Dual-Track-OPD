from contextlib import ExitStack

import pytest
import torch

from dual_track_opd.fc_opd.conditions import Condition
from dual_track_opd.fc_opd.offline_loss import (
    FOUR_CONDITIONS,
    load_offline_score_records,
    offline_record_to_tensors,
    run_offline_loss_backward,
    run_offline_loss_smoke,
)
from dual_track_opd.fc_opd.offline_scoring import (
    ByteTokenizer,
    OfflineScoringConfig,
    iter_offline_scores,
    make_smoke_dataset,
    write_offline_scores,
)
from dual_track_opd.fc_opd.teacher_client import TeacherClient
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint
from dual_track_opd.fc_opd.teacher_scorer import SyntheticTeacherScorer
from dual_track_opd.fc_opd.teacher_service import running_teacher_server

TOP_K = 32
CONDITIONS = FOUR_CONDITIONS


def _build_offline_payloads(num_samples: int = 4) -> list[dict]:
    """Produce real offline-score payloads via the synthetic teacher pipeline."""

    tokenizer = ByteTokenizer()
    scorer = SyntheticTeacherScorer(
        vocab_size=320, top_k=TOP_K, tokenizer_hash=tokenizer_fingerprint(tokenizer)
    )
    with ExitStack() as stack:
        server = stack.enter_context(running_teacher_server(scorer))
        host, port = server.server_address
        client = TeacherClient(
            f"http://{host}:{port}", expected_tokenizer_hash=tokenizer_fingerprint(tokenizer)
        )
        config = OfflineScoringConfig(source_dataset="vstar", conditions=CONDITIONS)
        scored = list(
            iter_offline_scores(
                make_smoke_dataset(num_samples),
                config=config,
                tokenizer=tokenizer,
                teacher_client=client,
                mode="protocol_smoke",
            )
        )
    return [record.payload for record in scored]


@pytest.fixture(scope="module")
def offline_payloads() -> list[dict]:
    return _build_offline_payloads(4)


def test_offline_record_to_tensors_aligns_with_response_length(offline_payloads):
    payload = offline_payloads[0]
    tensors = offline_record_to_tensors(payload)
    assert tensors.seq_len == tensors.num_response_tokens
    assert set(tensors.teacher_scores) == set(CONDITIONS)
    for mask in tensors.chunk_masks.values():
        assert mask.shape == (1, tensors.seq_len)
    for scores in tensors.teacher_scores.values():
        assert scores.token_ids.shape == (1, tensors.seq_len, TOP_K)
        assert scores.tail_log_prob.shape == (1, tensors.seq_len)
        assert scores.entropy.shape == (1, tensors.seq_len)


def test_run_offline_loss_backward_produces_finite_loss_and_gradients(offline_payloads):
    result = run_offline_loss_backward(offline_payloads[0])
    assert result.loss_is_finite
    assert result.grad_exists
    assert result.grad_is_finite
    assert result.grad_norm > 0.0
    assert result.seq_len == result.num_response_tokens
    # The protocol-smoke response is well-formed, so all four conditions are used:
    # task/free/full from the chunks and blur from the tag/whitespace fallback.
    assert result.consumed_conditions == set(CONDITIONS)


def test_offline_loss_smoke_report_passes_over_all_records(offline_payloads):
    report = run_offline_loss_smoke(offline_payloads)
    assert report.num_records == len(offline_payloads)
    assert report.all_loss_finite
    assert report.all_grads_present
    assert report.all_grads_finite
    assert report.all_masks_aligned
    assert report.four_conditions_consumed
    assert report.passed


def test_gradients_flow_to_student_logits_only(offline_payloads):
    payload = offline_payloads[0]
    tensors = offline_record_to_tensors(payload)
    for scores in tensors.teacher_scores.values():
        assert scores.log_probs.grad is None
    result = run_offline_loss_backward(payload)
    assert result.grad_exists
    # Teacher tensors are rebuilt per call and never require grad.
    for scores in tensors.teacher_scores.values():
        assert scores.log_probs.requires_grad is False


def test_smoke_reads_back_written_jsonl(tmp_path, offline_payloads):
    from dual_track_opd.fc_opd.offline_loss import load_offline_score_records as _reader
    from dual_track_opd.fc_opd.offline_scoring import OfflineScoreRecord

    records = [OfflineScoreRecord(payload=payload) for payload in offline_payloads]
    result = write_offline_scores(records, output_dir=tmp_path, filename="scores")
    loaded = _reader(result.jsonl_path)
    assert len(loaded) == len(offline_payloads)
    report = run_offline_loss_smoke(loaded)
    assert report.passed


def test_student_vocab_floor_is_respected(offline_payloads):
    payload = offline_payloads[0]
    tensors = offline_record_to_tensors(payload)
    with pytest.raises(ValueError, match="smaller than the largest teacher token id"):
        run_offline_loss_backward(payload, student_vocab_size=tensors.vocab_floor - 1)


def test_chunk_span_out_of_bounds_is_rejected():
    payload = {
        "sample_uid": "vstar:bad",
        "response_token_ids": [1, 2, 3],
        "condition_scores": {
            "full": {
                "token_ids": [[0], [1], [2]],
                "log_probs": [[0.0], [0.0], [0.0]],
                "tail_log_prob": None,
                "entropy": None,
            }
        },
        "chunk_spans": {
            "visual_evidence": [[0, 9]],
            "reasoning": [],
            "answer": [],
            "format_valid": True,
        },
    }
    with pytest.raises(ValueError, match="out of bounds"):
        offline_record_to_tensors(payload)


def test_load_offline_score_records_round_trips(tmp_path):
    from dual_track_opd.fc_opd.offline_scoring import OfflineScoreRecord

    payloads = _build_offline_payloads(2)
    records = [OfflineScoreRecord(payload=payload) for payload in payloads]
    result = write_offline_scores(records, output_dir=tmp_path, filename="scores")
    loaded = load_offline_score_records(result.jsonl_path)
    assert [r["sample_uid"] for r in loaded] == [p["sample_uid"] for p in payloads]

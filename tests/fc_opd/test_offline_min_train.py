from contextlib import ExitStack

import pytest

from dual_track_opd.fc_opd.conditions import Condition
from dual_track_opd.fc_opd.offline_loss import (
    FOUR_CONDITIONS,
    run_offline_min_train_smoke,
)
from dual_track_opd.fc_opd.offline_scoring import (
    ByteTokenizer,
    OfflineScoringConfig,
    iter_offline_scores,
    make_smoke_dataset,
)
from dual_track_opd.fc_opd.teacher_client import TeacherClient
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint
from dual_track_opd.fc_opd.teacher_scorer import SyntheticTeacherScorer
from dual_track_opd.fc_opd.teacher_service import running_teacher_server

TOP_K = 32


def _build_offline_payloads(num_samples: int = 4) -> list[dict]:
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
        config = OfflineScoringConfig(source_dataset="vstar", conditions=FOUR_CONDITIONS)
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


def test_min_train_smoke_passes(offline_payloads):
    report = run_offline_min_train_smoke(offline_payloads, num_steps=10, learning_rate=0.1)
    assert report.num_records == len(offline_payloads)
    assert report.num_steps == 10
    assert len(report.steps) == 10
    assert report.passed


def test_every_step_is_finite_and_updates_logits(offline_payloads):
    report = run_offline_min_train_smoke(offline_payloads, num_steps=5, learning_rate=0.1)
    for step in report.steps:
        assert step.loss_is_finite
        assert step.grad_is_finite
        assert step.grad_norm > 0.0
        assert step.logits_delta_norm > 0.0
        assert step.consumed_conditions == set(FOUR_CONDITIONS)


def test_loss_decreases_over_training(offline_payloads):
    report = run_offline_min_train_smoke(offline_payloads, num_steps=10, learning_rate=0.1)
    assert report.loss_decreased
    assert report.steps[-1].loss < report.steps[0].loss


def test_all_four_conditions_consumed(offline_payloads):
    report = run_offline_min_train_smoke(offline_payloads, num_steps=3, learning_rate=0.1)
    assert report.four_conditions_consumed
    assert report.consumed_conditions == set(FOUR_CONDITIONS)


def test_num_steps_must_be_positive(offline_payloads):
    with pytest.raises(ValueError, match="num_steps must be positive"):
        run_offline_min_train_smoke(offline_payloads, num_steps=0)


def test_student_vocab_floor_is_respected(offline_payloads):
    from dual_track_opd.fc_opd.offline_loss import offline_record_to_tensors

    floor = max(offline_record_to_tensors(p).vocab_floor for p in offline_payloads)
    with pytest.raises(ValueError, match="smaller than the largest teacher token id"):
        run_offline_min_train_smoke(offline_payloads, num_steps=2, student_vocab_size=floor - 1)


def test_step_count_matches_request(offline_payloads):
    report = run_offline_min_train_smoke(offline_payloads, num_steps=7, learning_rate=0.05)
    assert [step.step for step in report.steps] == list(range(7))


def test_default_is_ten_steps(offline_payloads):
    report = run_offline_min_train_smoke(offline_payloads)
    assert report.num_steps == 10
    assert len(report.steps) == 10
    assert report.passed


def test_empty_records_raises():
    with pytest.raises(ValueError, match="no offline-score records"):
        run_offline_min_train_smoke([], num_steps=3)

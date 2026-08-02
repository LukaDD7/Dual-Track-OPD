"""Prompt-level statistics and exact-token protocol gate tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from dual_track_opd.fc_opd.dataset_signal_audit import hash_token_ids
from dual_track_opd.support_aware.diagnostic import (
    _check_gates,
    _compute_prompt_signal_metrics,
    _random_first_correct_mrr,
    assert_exact_score_alignment,
    build_exact_scored_record,
    generation_record_from_token_ids,
)
from dual_track_opd.support_aware.scorer import StudentScorer, TeacherScorer


def _rollout(uid: str, gap: float, correct: bool) -> dict:
    return {
        "sample_uid": uid,
        "is_greedy": False,
        "teacher_gap": gap,
        "correct": correct,
    }


def test_prompt_macro_auc_and_bootstrap_are_reproducible():
    rows = [
        _rollout("p1", 3.0, True),
        _rollout("p1", 2.0, True),
        _rollout("p1", 1.0, False),
        _rollout("p1", 0.0, False),  # AUC p1 = 1.0
        _rollout("p2", 2.0, True),
        _rollout("p2", 2.0, False),
        _rollout("p2", 3.0, False),  # AUC p2 = (tie + loss) / 2 = .25
    ]
    first = _compute_prompt_signal_metrics(rows, seed=17, resamples=500)
    second = _compute_prompt_signal_metrics(rows, seed=17, resamples=500)

    assert first == second
    assert first["eligible_prompt_count"] == 2
    assert first["within_prompt_auc"]["mean"] == pytest.approx(0.625)
    # Prompt-specific random Rank@1 uses c_i/K, not 1/K.
    assert first["random_rank1"] == pytest.approx((0.5 + 1 / 3) / 2)


def test_tied_top_score_receives_fractional_rank1_credit():
    metrics = _compute_prompt_signal_metrics(
        [_rollout("p", 2.0, True), _rollout("p", 2.0, False)],
        seed=42,
        resamples=10,
    )
    assert metrics["rank1"]["mean"] == pytest.approx(0.5)
    assert metrics["rank1_lift"]["mean"] == pytest.approx(0.0)


def test_exact_random_mrr_baseline():
    assert _random_first_correct_mrr(4, 1) == pytest.approx(
        (1.0 + 0.5 + 1 / 3 + 0.25) / 4
    )
    assert _random_first_correct_mrr(4, 4) == pytest.approx(1.0)


def _gate_summary(auc_low: float = 0.51) -> dict:
    return {
        "tokenizer_match": True,
        "missing_image_count": 0,
        "non_finite_score_count": 0,
        "malformed_response_rate": 0.0,
        "overall_duplicate_rollout_rate": 0.0,
        "exact_token_alignment_rate": 1.0,
        "prompt_token_hash_rate": 1.0,
        "shard_completeness_rate": 1.0,
        "truncation_rate": 0.0,
        "correct_tail_count": 20,
        "correct_tail_rank_metrics": {
            "within_prompt_auc": {"mean": 0.7, "ci95_low": auc_low},
            "rank1_lift": {"mean": 0.1, "ci95_low": 0.01},
        },
    }


def test_signal_gates_use_confidence_interval_lower_bounds():
    passing = {name: passed for name, passed, _ in _check_gates(_gate_summary(), "full")}
    failing = {
        name: passed for name, passed, _ in _check_gates(_gate_summary(0.49), "full")
    }
    assert passing["teacher_gap_within_prompt_auc"] is True
    assert passing["correct_tail_rank1_above_random"] is True
    assert failing["teacher_gap_within_prompt_auc"] is False


class _Tokenizer:
    eos_token_id = 9

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        del clean_up_tokenization_spaces
        visible = [token_id for token_id in ids if not (skip_special_tokens and token_id == 9)]
        return " ".join(str(token_id) for token_id in visible)


def test_generation_record_keeps_eos_but_excludes_it_from_content_score():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=9))
    generation = generation_record_from_token_ids(
        model=model,
        tokenizer=_Tokenizer(),
        response_token_ids=(4, 5, 9),
        prompt_token_ids=(1, 2),
        max_new_tokens=8,
    )
    response_hash = hash_token_ids((4, 5, 9))
    teacher = TeacherScorer.ScoreResult(
        request_id="r",
        sampled_token_log_probs=(-1.0, -2.0, -8.0),
        mean_logp=-11 / 3,
        scored_token_ids=(4, 5, 9),
        scored_token_hash=response_hash,
        response_mask=(True, True, True),
    )
    student = StudentScorer.ScoreResult(
        sampled_token_log_probs=(-2.0, -3.0, -9.0),
        mean_logp=-14 / 3,
        token_count=3,
        scored_token_ids=(4, 5, 9),
        scored_token_hash=response_hash,
        response_mask=(True, True, True),
    )
    record = build_exact_scored_record(
        base_record={"gold_answer": "5"},
        generation=generation,
        verdict={"correct": True, "answer_extracted": "5", "gold_answer": "5"},
        teacher_result=teacher,
        student_result=student,
    )

    assert generation.response_text_display == "4 5"
    assert generation.response_token_ids_raw == (4, 5, 9)
    assert generation.content_mask == (True, True, False)
    assert generation.finish_reason == "stop"
    assert record["teacher_mean_logp"] == pytest.approx(-1.5)
    assert record["teacher_mean_logp_all"] == pytest.approx(-11 / 3)
    assert record["teacher_terminal_logp"] == pytest.approx(-8.0)
    assert record["teacher_gap"] == pytest.approx(1.0)

    with pytest.raises(RuntimeError, match="student scored token IDs"):
        assert_exact_score_alignment(
            expected_token_ids=(4, 5, 9),
            teacher_result=teacher,
            student_result=replace(student, scored_token_ids=(4, 6, 9)),
        )

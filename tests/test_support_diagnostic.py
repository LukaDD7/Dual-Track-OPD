"""Tests for support_aware.verifier — Geometry3K answer extraction and verification."""

from __future__ import annotations

import pytest

from dual_track_opd.support_aware.verifier import (
    extract_answer,
    normalize_gold_answer,
    verify_answer,
)


# -- extract_answer ------------------------------------------------------------

@pytest.mark.parametrize(
    "response, expected",
    [
        # Simple numeric
        ("The angle is 42 degrees. Answer: 42", "42"),
        ("Answer: 138", "138"),
        ("x = 27. Answer: 27", "27"),
        # Decimal
        ("Area = 374.1 square feet. Answer: 374.1", "374.1"),
        # Negative
        ("The result is -5. Answer: -5", "-5"),
        # Answer marker variations
        ("Some reasoning\nanswer: 120", "120"),
        ("**Answer:** 90", "90"),
        ("Answer: 3.14", "3.14"),
        # LaTeX expressions
        ("Answer: 2 \\SQRT { 221 }", "2sqrt(221)"),
        ("Answer: 12 \\PI", "12pi"),
        ("Answer: \\frac{1}{2}", "(1)/(2)"),
    ],
)
def test_extract_answer(response, expected):
    result = extract_answer(response)
    assert result == expected, f"Expected {expected!r}, got {result!r}"


def test_extract_answer_empty():
    assert extract_answer("") is None
    assert extract_answer("No numbers here, just text.") is None


def test_extract_answer_overlength_digits_no_crash():
    """Over-length digit runs (float→inf) must not crash _normalize_number."""
    huge = "1" + "0" * 400
    assert extract_answer("Answer: " + huge) == huge


def test_extract_answer_malformed():
    # Unmarked reasoning numbers are not reliable final answers.
    result = extract_answer("The answer might be around 42 or 43")
    assert result is None


@pytest.mark.parametrize(
    "response",
    [
        "The value of x is 42",
        "Therefore x = 27 and y = 35",
        "A long truncated trace with intermediate 42",
    ],
)
def test_extract_answer_rejects_unmarked_reasoning(response):
    assert extract_answer(response) is None


# -- normalize_gold_answer -----------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("3", "3"),
        ("120", "120"),
        ("2 \\SQRT { 221 }", "2sqrt(221)"),
        ("12 \\PI", "12pi"),
        (3, "3"),
        (3.0, "3"),
        (3.14159, "3.1416"),
        ({"answer": "42"}, "42"),
        ({"label": "90"}, "90"),
        ({"gold": "27"}, "27"),
        ("374.1", "374.1"),
        ("374.10", "374.1"),
    ],
)
def test_normalize_gold_answer(raw, expected):
    result = normalize_gold_answer(raw)
    assert result == expected, f"Expected {expected!r}, got {result!r}"


# -- verify_answer -------------------------------------------------------------

def test_verify_correct_numeric():
    result = verify_answer("x = 42. Answer: 42", "42")
    assert result["correct"] is True
    assert result["format_valid"] is True
    assert result["malformed"] is False


def test_verify_correct_decimal():
    result = verify_answer("Area = 374.1. Answer: 374.1", "374.1")
    assert result["correct"] is True


def test_verify_wrong():
    result = verify_answer("Answer: 42", "27")
    assert result["correct"] is False


def test_verify_malformed():
    result = verify_answer("No answer here", "42")
    assert result["correct"] is None
    assert result["malformed"] is True


def test_verify_correct_latex():
    result = verify_answer("Answer: 2 \\SQRT { 221 }", "2 \\SQRT { 221 }")
    assert result["correct"] is True


def test_verify_numeric_tolerance():
    # 3.1416 should match 3.14159
    result = verify_answer("Answer: 3.1416", "3.14159")
    assert result["correct"] is True


def test_verify_dict_gold():
    result = verify_answer("Answer: 90", {"answer": "90"})
    assert result["correct"] is True


# -- support_state classification ----------------------------------------------

from dual_track_opd.support_aware.support_state import (
    SupportState,
    classify_support_state,
)


def test_support_state_exposed_greedy():
    assert classify_support_state(greedy_correct=True, correct_count=0, K=8) == SupportState.EXPOSED


def test_support_state_exposed_majority():
    assert classify_support_state(greedy_correct=False, correct_count=4, K=8) == SupportState.EXPOSED


def test_support_state_correct_tail():
    assert classify_support_state(greedy_correct=False, correct_count=1, K=8) == SupportState.CORRECT_TAIL
    assert classify_support_state(greedy_correct=False, correct_count=3, K=8) == SupportState.CORRECT_TAIL


def test_support_state_no_correct():
    assert classify_support_state(greedy_correct=False, correct_count=0, K=8) == SupportState.NO_CORRECT_OBSERVED


def test_support_state_other_none_greedy():
    assert classify_support_state(greedy_correct=None, correct_count=1, K=8) == SupportState.OTHER


def test_support_state_boundary_exactly_half():
    # correct_count/K == 0.5 → exposed (>= 0.5)
    assert classify_support_state(greedy_correct=False, correct_count=4, K=8) == SupportState.EXPOSED


def test_support_state_boundary_k2():
    # K=2: correct_count=1 is half → exposed; correct_count=0 is no_correct
    assert classify_support_state(greedy_correct=False, correct_count=1, K=2) == SupportState.EXPOSED
    assert classify_support_state(greedy_correct=False, correct_count=0, K=2) == SupportState.NO_CORRECT_OBSERVED
    # correct_tail needs 0 < c < K/2, so c=0 doesn't qualify and c=1 is not < K/2
    assert classify_support_state(greedy_correct=False, correct_count=0, K=4) == SupportState.NO_CORRECT_OBSERVED
    assert classify_support_state(greedy_correct=False, correct_count=1, K=4) == SupportState.CORRECT_TAIL

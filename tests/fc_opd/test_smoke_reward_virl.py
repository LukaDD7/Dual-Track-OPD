from __future__ import annotations

import pytest

from dual_track_opd.fc_opd.answer_extraction import extract_final_answer_candidate
from dual_track_opd.fc_opd import smoke_reward
from dual_track_opd.fc_opd.smoke_reward import _is_single_math_expression, compute_score


@pytest.mark.parametrize(
    ("gold", "response"),
    [
        ("Yes", "\\boxed{Yes}"),
        ("No", "The answer is No."),
        ("4:05", "\\boxed{4:05}"),
        ("1:1", "\\boxed{1:1}"),
        ("4 - 3 = 1", "\\boxed{4 - 3 = 1}"),
        ("\\frac{10}{3}", "\\boxed{\\frac{10}{3}}"),
        ("\\dfrac{25\\pi }{8}", "\\boxed{\\dfrac{25\\pi }{8}}"),
        ("30 minutes", "\\boxed{30 minutes}"),
        ("\\{2,3,5\\}", "\\boxed{\\{2,3,5\\}}"),
        ("\\frac{{{2}^{2021}}}{{{3}^{2019}}}", "\\boxed{\\frac{{{2}^{2021}}}{{{3}^{2019}}}}"),
        ("E", "\\boxed{E}"),
    ],
)
def test_virl_gold_answer_in_boxed_is_correct(gold, response):
    result = compute_score(
        "ViRL39K",
        response,
        ground_truth=gold,
        extra_info={"answer": gold, "gt_type": "has_number"},
    )

    assert result == {"score": 1.0}


@pytest.mark.parametrize(
    ("gold", "response"),
    [
        ("Yes", "\\boxed{No}"),
        ("4:05", "\\boxed{4:06}"),
        ("1:1", "\\boxed{2:1}"),
        ("\\frac{10}{3}", "\\boxed{\\frac{11}{3}}"),
        ("30 minutes", "\\boxed{31 minutes}"),
    ],
)
def test_virl_gold_answer_mismatch_is_zero(monkeypatch, gold, response):
    # Unit tests must not silently depend on whether the local test environment
    # contains mathruler.  The pinned environment integration is covered by
    # preflight and by the actual training environment.
    monkeypatch.setattr(smoke_reward, "_math_equal", lambda extracted, gold: False)
    result = compute_score(
        "ViRL39K",
        response,
        ground_truth=gold,
        extra_info={"answer": gold, "gt_type": "has_number"},
    )

    assert result == {"score": 0.0}


@pytest.mark.parametrize(
    ("gold", "response", "choices"),
    [
        ("A", "\\boxed{A. 2}", ["A. 2", "B. 5", "C. 4", "D. 3"]),
        ("B", "\\boxed{\\text{(B) } 30^\\circ}", ["(A) 15°", "(B) 30°", "(C) 45°", "(D) 60°"]),
        ("C", "\\boxed{C. \\ 112^\\circ}", ["A. 40°", "B. 80°", "C. 112°", "D. 150°"]),
        ("E", "\\boxed{E. 30}", ["A. 10", "B. 20", "C. 30", "D. 40", "E. 50"]),
    ],
)
def test_mcq_letter_with_value_matches_gold(gold, response, choices):
    result = compute_score(
        "ViRL39K",
        response,
        ground_truth=gold,
        extra_info={"answer": gold, "choices": choices},
    )

    assert result == {"score": 1.0}


def test_mcq_letter_with_value_rejects_wrong_letter():
    result = compute_score(
        "ViRL39K",
        "\\boxed{B. 5}",
        ground_truth="A",
        extra_info={"answer": "A", "choices": ["A. 2", "B. 5", "C. 4", "D. 3"]},
    )

    assert result == {"score": 0.0}


def test_extract_final_answer_preserves_nested_latex():
    text = "Reasoning. \\boxed{\\frac{10}{3}}"

    assert extract_final_answer_candidate(text) == "\\frac{10}{3}"


def test_extract_final_answer_preserves_escaped_set_braces():
    text = "Reasoning. \\boxed{\\{2,3,5\\}}"

    assert extract_final_answer_candidate(text) == "\\{2,3,5\\}"


def test_math_grader_is_limited_to_single_math_expressions():
    allowed = ("\\frac{10}{3}", "\\dfrac{25\\pi}{8}", "72^\\circ", "24\\pi")
    disallowed = (
        "0/5 or 0",
        "3x = 17",
        "-3<x<-1",
        "30 minutes",
        "\\{2,3,5\\}",
        "(4,0)",
        "4:05",
        "5.0, but perhaps 6",
    )

    assert all(_is_single_math_expression(value) for value in allowed)
    assert not any(_is_single_math_expression(value) for value in disallowed)

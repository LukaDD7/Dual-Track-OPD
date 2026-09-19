from __future__ import annotations

import pytest

from dual_track_opd.fc_opd.answer_extraction import extract_final_answer_candidate
from dual_track_opd.fc_opd.smoke_reward import compute_score


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
def test_virl_gold_answer_mismatch_is_zero(gold, response):
    result = compute_score(
        "ViRL39K",
        response,
        ground_truth=gold,
        extra_info={"answer": gold, "gt_type": "has_number"},
    )

    assert result == {"score": 0.0}


def test_extract_final_answer_preserves_nested_latex():
    text = "Reasoning. \\boxed{\\frac{10}{3}}"

    assert extract_final_answer_candidate(text) == "\\frac{10}{3}"


def test_extract_final_answer_preserves_escaped_set_braces():
    text = "Reasoning. \\boxed{\\{2,3,5\\}}"

    assert extract_final_answer_candidate(text) == "\\{2,3,5\\}"

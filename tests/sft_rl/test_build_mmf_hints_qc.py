"""Unit tests for the PTD-PO hint QC + prompt construction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sft_rl"))

from build_mmf_hints import (  # noqa: E402
    build_prompt_with_hint,
    qc_hint,
)


def test_mcq_letter_leak_rejected():
    ok, hard, _ = qc_hint(
        "Focus on the left triangle; compare the two shaded regions.",
        "(B)",
    )
    assert ok and not hard
    ok, hard, _ = qc_hint("Compare option B's region with the others.", "(B)")
    assert not ok and "answer_leak_letter" in hard


def test_yesno_word_leak_rejected():
    ok, hard, _ = qc_hint("Check whether the two lines are parallel.", "yes")
    assert ok and not hard
    ok, hard, _ = qc_hint("The answer is simply yes.", "yes")
    assert not ok and "answer_leak_word" in hard


def test_numeric_leak_rejected():
    ok, hard, _ = qc_hint(
        "Estimate the total area from the shaded rectangle.",
        "65.12",
    )
    assert ok and not hard
    ok, hard, _ = qc_hint("The width is 65.12 units.", "65.12")
    assert not ok and "answer_leak_numeric" in hard


def test_expression_substr_leak_rejected():
    ok, hard, _ = qc_hint("Relate the areas of the two squares.", r"\frac{1}{4}")
    assert ok and not hard
    ok, hard, _ = qc_hint("The ratio is 1/4.", r"\frac{1}{4}")
    assert not ok and "answer_leak_substr" in hard


def test_multipart_numeric_component_leak_rejected():
    """Regression: a hint computing only one part ('= 50 days') of the
    multi-part answer '50, 57.6, 292' evaded the full-string containment check."""
    q = "<image>Use the bar chart to find the sampled days and percentages."
    ok, hard, _ = qc_hint(
        "The pie chart's good category is 64%, so verify the total: "
        "32 days = 64% -> total = 50 days.",
        "50, 57.6, 292",
        question=q,
    )
    assert not ok and "answer_leak_component_numeric" in hard


def test_multipart_component_word_leak_rejected():
    ok, hard, _ = qc_hint(
        "Focus on the trend of the leftmost bar.", "rectangle, cylinder"
    )
    assert ok and not hard
    ok, hard, _ = qc_hint(
        "The shape to focus on is the cylinder on the right.", "rectangle, cylinder"
    )
    assert not ok and "answer_leak_component_word" in hard


def test_other_answer_embedded_number_leak_rejected():
    q = "<image>What is the perimeter of the square tablecloth with side 20 dm?"
    ok, hard, _ = qc_hint(
        "Multiply the labeled side length by 4 to obtain the total lace length.",
        "80 decimeters",
        question=q,
    )
    assert ok and not hard
    ok, hard, _ = qc_hint(
        "Multiply 20 by 4, which gives 80 for the lace.",
        "80 decimeters",
        question=q,
    )
    assert not ok and "answer_leak_component_numeric" in hard


def test_assertion_phrase_rejected_as_cot_degenerate():
    """Regression: 'the correct response is null' asserts a specific response
    (and was also factually wrong), but the old regex did not match it."""
    ok, hard, _ = qc_hint(
        "Trace the push sequence and note the final position; the correct "
        "response is null.", "4"
    )
    assert not ok and "cot_degenerate" in hard


def test_degenerate_repeated_output_rejected():
    """Regression: a broken FP8-MoE serving returned 768x'!' for every sample
    and passed all leak checks. Low unique-character output is now hard-rejected."""
    ok, hard, _ = qc_hint("!" * 768, "9")
    assert not ok and "degenerate_output" in hard
    ok, hard, _ = qc_hint("Look at the top-left region of the chart.", "9")
    assert ok and not hard


def test_cot_degenerate_rejected():
    ok, hard, _ = qc_hint(
        "Step 1: ... Step 2: ... therefore the answer is obtained by the full "
        "derivation shown above in this long chain of thought.",
        "7",
    )
    assert not ok and "cot_degenerate" in hard


def test_too_long_rejected():
    ok, hard, _ = qc_hint("x" * 3000, "7", max_chars=2048)
    assert not ok and "too_long" in hard


def test_decisive_overlap_soft():
    ok, hard, soft = qc_hint(
        "Use the 42-degree angle and the 87-degree supplement and the 55 base.",
        "13",
        original_answer="sum 42+87+55=184, subtract from 197 to get 13",
        max_decisive_overlap=3,
    )
    assert ok and not hard
    assert any(s.startswith("decisive_overlap") for s in soft)


def test_no_spatial_soft():
    ok, _, soft = qc_hint("Check the parity and factor the expression.", "9")
    assert ok
    assert "no_spatial" in soft


def test_prompt_with_hint_keeps_image_placeholders():
    q = "<image>What is the total?"
    hint = "Look at the top-left region."
    msgs = build_prompt_with_hint(q, hint)
    assert len(msgs) == 1
    assert "<image>" in msgs[0]["content"]
    assert hint in msgs[0]["content"]
    assert build_prompt_with_hint(q, "   ") == []

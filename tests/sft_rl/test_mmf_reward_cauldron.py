"""Unit tests for the Cauldron-aware mmf_reward.py fix.

Regression coverage for the two gates that made Cauldron GT rows unmatchable:
  1. trailing-period GTs ("Yes.", "No.", "4.") previously fell through the
     yesno / letter fullmatch gates;
  2. raven letter GTs run A..H, previously only A-E was matchable.

Also pins the MMF-side behavior that must not change (A-E letters, clean
yes/no, numeric tolerance) — verified against all 95,128 MMF hint-pool rows
(only 2 routing flips, both repaired numeric "400." rows).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sft_rl"))

from mmf_reward import compute_score  # noqa: E402


def _score(gt: str, sol: str) -> float:
    return float(compute_score("mmfinereason", sol, gt)["score"])


# --- Cauldron: yesno GTs with trailing periods (vsr / raven / vqav2) --------


def test_yesno_trailing_period_matches():
    assert _score("Yes.", "The answer is yes.") == 1.0
    assert _score("No.", "No, the shapes are not the same size.") == 1.0
    assert _score("True.", "Yes, that is correct.") == 1.0
    assert _score("False.", "No, it is not.") == 1.0


def test_yesno_trailing_period_wrong_direction():
    assert _score("Yes.", "No, they do not match.") == 0.0
    assert _score("No.", "Yes, they match exactly.") == 0.0


# --- Cauldron: letter GTs beyond E (raven F..H) ----------------------------


def test_raven_letters_f_to_h():
    assert _score("F", "\\boxed{F}") == 1.0
    assert _score("G", "The answer is G.") == 1.0
    assert _score("H", "The correct figure is (H).") == 1.0
    assert _score("H", "The answer is F.") == 0.0


def test_letter_gt_with_trailing_period():
    assert _score("B.", "Based on the diagram, the answer is B.") == 1.0
    assert _score("(F).", "Final answer: (F)") == 1.0


# --- Cauldron: numeric GTs with trailing periods (tallyqa style) -----------


def test_numeric_trailing_period():
    assert _score("4.", "There are 4 triangles.") == 1.0
    assert _score("1948.", "Final answer: 1948") == 1.0
    assert _score("4.", "There are 5 triangles.") == 0.0


# --- MMF regression: existing clean forms must be untouched ----------------


def test_mmf_letter_a_e():
    assert _score("D", "Answer: D") == 1.0
    assert _score("D", "I think it's B") == 0.0
    assert _score("C", "\\boxed{C}") == 1.0
    assert _score("A", "(A) is correct") == 1.0


def test_mmf_yesno_clean():
    assert _score("Yes", "No, it is not") == 0.0
    assert _score("yes", "Yes, that's right") == 1.0


def test_mmf_numeric_tolerance():
    assert _score("65.12", "The answer is 65.12") == 1.0
    assert _score("65.12", "The answer is 65.13") == 0.0
    assert _score("8", "Final answer: 8") == 1.0


def test_return_contract():
    out = compute_score("mmfinereason", "Answer: D", "D")
    assert set(out.keys()) == {"score", "extracted"}
    out = compute_score("vsr", "The answer is yes.", "Yes.")
    assert out["extracted"] == "yes"

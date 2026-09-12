#!/usr/bin/env python3
"""Rule-based reward for MMFineReason RL (verl 0.9.0 custom_reward_function).

Extends scripts/sft_rl/geo3k_reward.py with the answer formats seen in
MMFineReason-123K (verified 2026-08-19):
  - single MCQ letter (A-E), with or without parentheses/period
  - yes/no, true/false
  - pure numeric (with tolerance)
  - LaTeX / symbolic expressions -> geo3k mathruler path

Signature required by verl: compute_score(data_source, solution_str,
ground_truth, extra_info=None, **kwargs) -> dict with a "score" key.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

# verl loads the reward via file:// spec_from_file_location without adding the
# script dir to sys.path, so make the sibling geo3k_reward import explicit.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from geo3k_reward import _extract_answer, _normalize_latex, _numeric_equal


def _match_mcq(solution_str: str, gt_letter: str) -> bool:
    sol = solution_str.upper()
    patterns = (
        r"\\BOXED\{([A-H])\}",
        r"<ANSWER>\s*([A-H])",
        r"ANSWER\s*[:：]?\s*\(?([A-H])\)?",
        r"FINAL\s+ANSWER\s*[:：]?\s*\(?([A-H])\)?",
        r"\(([A-H])\)",
    )
    for pat in patterns:
        m = re.search(pat, sol)
        if m and m.group(1) in "ABCDEFGH":
            return m.group(1) == gt_letter
    letters = re.findall(r"\b([A-H])\b", sol)
    return bool(letters) and letters[-1] == gt_letter


def _match_yesno(solution_str: str, gt: str) -> bool:
    sol = solution_str.strip().lower()
    yes_words = ("yes", "true")
    no_words = ("no", "false")
    gt_is_yes = gt in ("yes", "true")
    # Direct short answer first.
    tokens = [t for t in re.split(r"[^a-z]+", sol) if t]
    for tok in reversed(tokens):
        if tok in yes_words:
            return gt_is_yes
        if tok in no_words:
            return not gt_is_yes
    return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float | str]:
    del data_source, extra_info, kwargs
    gt = str(ground_truth).strip() if ground_truth is not None else ""
    # Cauldron GTs often carry a trailing period ("Yes.", "4.", "B."). Strip it
    # once so the gate below sees the clean form; MMF GTs are already clean
    # (verified 95,128 rows: only 2 routing flips, both repaired numeric rows).
    gt = re.sub(r"\.+$", "", gt).strip()
    if not gt:
        return {"score": 0.0, "extracted": ""}

    # MCQ letter: "D", "D.", "(D)", "Answer: C"  (A-H: raven GTs run A..H)
    m = re.fullmatch(r"\(?([A-Ha-h])\)?\.?", gt)
    if m:
        ok = _match_mcq(solution_str, m.group(1).upper())
        return {"score": 1.0 if ok else 0.0, "extracted": m.group(1).upper()}

    # yes/no or true/false
    yn = re.fullmatch(r"(yes|no|true|false)", gt.lower())
    if yn:
        ok = _match_yesno(solution_str, yn.group(1))
        return {"score": 1.0 if ok else 0.0, "extracted": yn.group(1)}

    # Numeric GT: exact with tolerance.
    if _numeric_equal(solution_str, gt):
        return {"score": 1.0, "extracted": gt}

    # Symbolic / LaTeX: normalize and ask mathruler.
    from mathruler.grader import grade_answer

    ans = _extract_answer(solution_str)
    if ans is None:
        return {"score": 0.0, "extracted": ""}
    try:
        ok = grade_answer(_normalize_latex(ans), _normalize_latex(gt))
    except Exception:
        ok = False
    return {"score": 1.0 if ok else 0.0, "extracted": ans}


if __name__ == "__main__":
    cases = [
        ("D", "Answer: D", 1.0),
            ("D", "I think it's B", 0.0),
            ("C", "\\boxed{C}", 1.0),
            ("A", "(A) is correct", 1.0),
            ("65.12", "The answer is 65.12", 1.0),
            ("65.12", "The answer is 65.13", 0.0),
            ("Yes", "No, it is not", 0.0),
            ("yes", "Yes, that's right", 1.0),
            ("B", "Based on the diagram, the answer is B.", 1.0),
            ("8", "Final answer: 8", 1.0),
            ("Yes.", "The answer is yes.", 1.0),
            ("No.", "No, the shapes are not the same size.", 1.0),
            ("No.", "Yes, they match exactly.", 0.0),
            ("F", "\\boxed{F}", 1.0),
            ("H", "The correct figure is (H).", 1.0),
            ("4.", "There are 4 triangles.", 1.0),
        ]
    n_fail = 0
    for gt, sol, expect in cases:
        got = compute_score("mmfinereason", sol, gt)["score"]
        flag = "OK " if got == expect else "FAIL"
        if got != expect:
            n_fail += 1
        print(f"{flag} gt={gt!r} sol={sol!r} -> {got} (expect {expect})")
    raise SystemExit(f"{len(cases) - n_fail}/{len(cases)} passed" if n_fail == 0 else f"FAILED {n_fail}")

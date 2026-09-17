#!/usr/bin/env python3
"""Rule-based geometry3k reward for verl 0.9.0 (custom_reward_function).

Signature required by verl: compute_score(data_source, solution_str,
ground_truth, extra_info=None, **kwargs) -> dict with a "score" key.

Ground truth in this repo's parquet is LaTeX-ish ("2 \\SQRT { 221 }",
"12 \\PI", "\\FRAC { 26 } { 3 }") which mathruler cannot parse directly, so we
normalize it (lowercase commands, strip spaces inside braces) before grading.
The solution answer is extracted from \\boxed{...} / <answer>...</answer> /
final-answer lines, with a numeric fallback.
"""

from __future__ import annotations

import re
from typing import Any

from mathruler.grader import grade_answer


def _normalize_latex(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\([A-Za-z]+)", lambda m: "\\" + m.group(1).lower(), s)

    def strip_braces(m: re.Match) -> str:
        return "{" + m.group(1).replace(" ", "") + "}"

    return re.sub(r"\{([^{}]*)\}", strip_braces, s)


def _extract_balanced(text: str, open_seq: str, close_seq: str) -> str | None:
    """Return the text inside the LAST balanced open_seq...close_seq."""
    start = text.rfind(open_seq)
    if start < 0:
        return None
    depth = 0
    for i in range(start + len(open_seq), len(text)):
        ch = text[i]
        if ch == "}":
            depth -= 1
            if depth < 0:
                return text[start + len(open_seq) : i].strip()
        elif ch == "{":
            depth += 1
    return text[start + len(open_seq) :].strip()


def _extract_answer(solution_str: str) -> str | None:
    # 1) \boxed{...} (balanced, last occurrence wins)
    boxed = _extract_balanced(solution_str, "\\boxed{", "}")
    if boxed:
        return boxed
    # 2) <answer>...</answer>
    m = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 3) "Final answer:" / "answer is" line
    for pat in (r"[Ff]inal answer[:：]\s*(.+)", r"[Aa]nswer is[:：]?\s*(.+)"):
        m = re.search(pat, solution_str)
        if m:
            return m.group(1).strip().rstrip(".")
    # 4) last pure number in the response
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", solution_str)
    return nums[-1] if nums else None


def _numeric_equal(pred: str, gt: str, tol: float = 1e-6) -> bool:
    try:
        a = float(pred.replace(",", "").replace("%", ""))
        b = float(gt.replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return False
    return abs(a - b) < tol


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float | str]:
    del data_source, extra_info, kwargs
    gt = str(ground_truth).strip() if ground_truth is not None else ""
    if not gt:
        return {"score": 0.0, "extracted": ""}
    ans = _extract_answer(solution_str)
    if ans is None:
        return {"score": 0.0, "extracted": ""}
    try:
        correct = grade_answer(ans, _normalize_latex(gt))
    except Exception:
        correct = False
    if not correct:
        correct = _numeric_equal(ans, gt)
    return {"score": 1.0 if correct else 0.0, "extracted": ans[:80]}

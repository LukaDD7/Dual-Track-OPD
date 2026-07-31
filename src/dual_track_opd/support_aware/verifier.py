"""Geometry3K answer extraction and verification.

Handles both numeric answers (1873/2101) and LaTeX expression answers
(~169/2101 with \\SQRT, \\PI, \\frac, etc.).
"""

from __future__ import annotations

import math
import re
from typing import Any

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Match "Answer: <stuff>" or "answer: <stuff>" or "**Answer:** <stuff>"
_ANSWER_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?\s*Answer\s*:?\s*(?:\*\*)?\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Numeric: integer, decimal, negative, optional units
_NUMERIC_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")

# LaTeX expressions that appear in Geometry3K gold answers
_LATEX_NORMALIZE = [
    (re.compile(r"\\SQRT\s*\{\s*(.+?)\s*\}"), r"sqrt(\1)"),
    (re.compile(r"\\sqrt\s*\{\s*(.+?)\s*\}"), r"sqrt(\1)"),
    (re.compile(r"\\PI"), "pi"),
    (re.compile(r"\\pi"), "pi"),
    (re.compile(r"\\frac\s*\{\s*(.+?)\s*\}\s*\{\s*(.+?)\s*\}"), r"(\1)/(\2)"),
    (re.compile(r"\\degree"), "deg"),
    (re.compile(r"°"), "deg"),
]

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_answer(response_text: str) -> str | None:
    """Extract the final answer from a student/teacher response.

    Strategy (in order):
    1. Look for "Answer:" marker and extract text after it.
    2. Fall back to the last non-empty line.
    3. From that text, extract a number or LaTeX expression.
    """
    if not response_text or not response_text.strip():
        return None

    # Try explicit answer marker first
    candidate_text = response_text
    marker_matches = list(_ANSWER_MARKER_RE.finditer(response_text))
    if marker_matches:
        candidate_text = marker_matches[-1].group(1)

    # Try to extract from the candidate text
    extracted = _extract_from_text(candidate_text)
    if extracted is not None:
        return extracted

    # Fall back: scan the whole response
    return _extract_from_text(response_text)


def _extract_from_text(text: str) -> str | None:
    """Extract a number or LaTeX expression from text.

    LaTeX expressions are tried first (they are more specific).  Plain numeric
    extraction is the fallback.
    """
    # Try LaTeX expressions first — they are more specific and would otherwise
    # be truncated by numeric extraction (e.g. "2 \\SQRT{221}" → "2").
    latex = _extract_latex_expression(text)
    if latex is not None:
        return _normalize_latex(latex)

    # Fall back to plain numeric
    numbers = _NUMERIC_RE.findall(text)
    if numbers:
        return _normalize_number(numbers[-1])

    return None


def _extract_latex_expression(text: str) -> str | None:
    """Extract a LaTeX math expression like '2 \\SQRT { 221 }'."""
    # Find patterns like \SQRT{...}, \frac{...}{...}, etc.
    patterns = [
        r"(\d+\.?\d*\s*\\SQRT\s*\{[^}]+\})",
        r"(\d+\.?\d*\s*\\sqrt\s*\{[^}]+\})",
        r"(\d+\.?\d*\s*\\PI)",
        r"(\d+\.?\d*\s*\\pi)",
        r"(\\SQRT\s*\{[^}]+\})",
        r"(\\frac\s*\{[^}]+\}\s*\{[^}]+\})",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            return matches[-1]
    return None


def _normalize_number(num_str: str) -> str:
    """Normalize a numeric string — round and strip trailing zeros."""
    try:
        value = float(num_str)
        if value == int(value):
            return str(int(value))
        # Round to 4 decimal places, then strip trailing zeros
        formatted = f"{value:.4f}"
        formatted = formatted.rstrip("0").rstrip(".") if "." in formatted else formatted
        return formatted
    except ValueError:
        return num_str.strip()


def _normalize_latex(latex_str: str) -> str:
    """Normalize LaTeX expression for comparison."""
    result = latex_str.strip()
    for pattern, replacement in _LATEX_NORMALIZE:
        result = pattern.sub(replacement, result)
    # Remove whitespace
    result = re.sub(r"\s+", "", result)
    return result


# ---------------------------------------------------------------------------
# Gold answer normalization
# ---------------------------------------------------------------------------


def normalize_gold_answer(answer_metadata: Any) -> str | None:
    """Normalize a gold answer from Geometry3K metadata.

    Handles:
    - Plain numbers / strings: "3", "120", "27"
    - LaTeX strings: "2 \\\\SQRT { 221 }", "12 \\\\PI"
    - Dicts with answer/label/gold/target/value keys
    """
    if answer_metadata is None:
        return None

    # Unwrap dict
    if isinstance(answer_metadata, dict):
        for key in ("answer", "label", "gold", "target", "value"):
            if key in answer_metadata:
                return normalize_gold_answer(answer_metadata[key])
        return None

    text = str(answer_metadata).strip()

    # Try numeric
    num_match = _NUMERIC_RE.match(text)
    if num_match and num_match.group() == text:
        return _normalize_number(text)

    # Try LaTeX
    latex = _extract_latex_expression(text)
    if latex is not None:
        return _normalize_latex(latex)

    # Fallback: return normalized text
    return _normalize_number(text)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_answer(
    response_text: str,
    gold_answer: Any,
) -> dict[str, Any]:
    """Verify whether a response answer matches the gold answer.

    Returns a dict with:
    - answer_extracted: the extracted answer string or None
    - gold_answer: normalized gold answer string
    - correct: True/False/None (None = unparseable)
    - format_valid: bool
    - malformed: bool
    """
    extracted = extract_answer(response_text)
    gold = normalize_gold_answer(gold_answer)

    if extracted is None or gold is None:
        return {
            "answer_extracted": extracted,
            "gold_answer": gold,
            "correct": None,
            "format_valid": False,
            "malformed": True,
        }

    correct = _answers_match(extracted, gold)
    return {
        "answer_extracted": extracted,
        "gold_answer": gold,
        "correct": correct,
        "format_valid": True,
        "malformed": False,
    }


def _answers_match(extracted: str, gold: str) -> bool:
    """Check if extracted answer matches gold answer.

    Comparison strategies:
    1. Exact string match (after normalization)
    2. Numeric match with tolerance
    3. LaTeX match (after normalization)
    """
    if extracted == gold:
        return True

    # Try numeric comparison
    try:
        extracted_val = float(extracted)
        gold_val = float(gold)
        return math.isclose(extracted_val, gold_val, rel_tol=1e-3, abs_tol=1e-3)
    except (ValueError, OverflowError):
        pass

    # Try LaTeX normalization
    extracted_norm = _normalize_latex(extracted)
    gold_norm = _normalize_latex(gold)
    if extracted_norm == gold_norm:
        return True

    return False

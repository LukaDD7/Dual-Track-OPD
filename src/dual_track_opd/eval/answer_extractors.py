"""Conservative answer extraction helpers for deterministic baseline scoring."""

from __future__ import annotations

import re
import string
from decimal import Decimal, InvalidOperation


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def extract_choice(text: object, choices: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> str | None:
    """Extract an explicit multiple-choice option.

    The extractor is intentionally conservative. It accepts common explicit
    formats such as ``Answer: C``, ``(C)``, ``C.``, and ``The answer is C``.
    It does not scan arbitrary long text for any standalone option letter.
    """

    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return None

    allowed = "".join(re.escape(c) for c in choices.upper())
    matches: list[str] = []

    patterns = [
        rf"(?i)\b(?:final\s+answer|answer|correct\s+answer)\s*(?:is|:|-)?\s*(?:option\s*)?\(?([{allowed}])\)?(?=$|[\s\.\),:;])",
        rf"(?i)\bthe\s+answer\s+is\s*(?:option\s*)?\(?([{allowed}])\)?(?=$|[\s\.\),:;])",
        rf"(?i)^\s*(?:option\s*)?\(?([{allowed}])\)?\s*[\.\):]\s+",
        rf"(?i)^\s*(?:option\s*)?\(?([{allowed}])\)?\s*[\.\):]\s*$",
        rf"(?i)^\s*(?:option\s*)?\(?([{allowed}])\)?\s*$",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, value):
            matches.append(match.group(1).upper())

    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return None


def normalize_text_answer(text: object) -> str:
    """Lowercase, remove punctuation/articles, and compress whitespace."""

    if text is None:
        return ""
    value = str(text).lower()
    value = value.translate(str.maketrans({char: " " for char in string.punctuation}))
    value = _ARTICLES_RE.sub(" ", value)
    return _SPACE_RE.sub(" ", value).strip()


def normalize_numeric_answer(text: object) -> str | None:
    """Return a canonical numeric string for simple numeric answers.

    This first-pass parser handles one explicit integer/decimal value and is
    deliberately unwilling to infer answers from multiple numbers or equations.
    """

    if text is None:
        return None
    value = str(text).strip().replace(",", "")
    if not value:
        return None

    answer_match = re.search(
        r"(?i)\b(?:final\s+answer|answer)\s*(?:is|:|-)?\s*([-+]?\d+(?:\.\d+)?)\b",
        value,
    )
    if answer_match:
        candidates = [answer_match.group(1)]
    else:
        candidates = re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])", value)

    unique = sorted(set(candidates))
    if len(unique) != 1:
        return None

    try:
        number = Decimal(unique[0])
    except InvalidOperation:
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized == "-0" else normalized

"""Conservative final-answer extraction shared by Geometry3K scorers."""

from __future__ import annotations

import re

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)
XML_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is\s*|[:=]\s*)+([^\n\r]+)",
    re.IGNORECASE,
)


def extract_final_answer_candidate(response_text: str) -> str | None:
    """Return an explicitly marked answer, never an early reasoning number.

    ``\\boxed{}`` remains preferred.  XML and natural-language final-answer
    markers are accepted as a compatibility fallback for native Instruct
    generations.  Unmarked long reasoning is deliberately rejected because
    taking its first or last number silently rewards intermediate calculations.
    """
    text = str(response_text or "").strip()
    if not text:
        return None
    for pattern in (BOXED_RE, XML_ANSWER_RE, FINAL_ANSWER_RE):
        matches = list(pattern.finditer(text))
        if matches:
            candidate = matches[-1].group(1).strip().strip("*_`$ ")
            return candidate or None
    # Preserve support for callers that pass an answer rather than a full
    # generation, while refusing ambiguous prose containing many values.
    if re.fullmatch(
        r"(?:[A-Da-d]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*[a-zA-Z°%²^]+)?)",
        text,
    ):
        return text.strip("*_`$ ") or None
    return None

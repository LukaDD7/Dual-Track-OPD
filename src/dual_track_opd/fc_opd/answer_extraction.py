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
    boxed = _extract_last_boxed(text)
    if boxed is not None:
        return boxed.strip().strip("*_`$ ") or None
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


def _extract_last_boxed(text: str) -> str | None:
    """Extract the last balanced ``\\boxed{...}`` including nested braces.

    The simple regex cannot represent LaTeX such as ``\\frac{10}{3}`` or sets
    with escaped braces.  Scan the brace span instead, treating braces escaped
    with a backslash as literal content rather than nesting delimiters.
    """

    matches = list(re.finditer(r"\\boxed\{", text, flags=re.IGNORECASE))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    index = start
    while index < len(text):
        char = text[index]
        escaped = index > 0 and text[index - 1] == "\\"
        if char == "{" and not escaped:
            depth += 1
        elif char == "}":
            trailing_outer_close = depth == 1 and not text[index + 1 :].strip()
            if escaped and not trailing_outer_close:
                # An escaped brace is literal unless it is the final character
                # of a malformed answer whose last source backslash precedes
                # the generated boxed closing delimiter.
                index += 1
                continue
            depth -= 1
            if depth == 0:
                return text[start:index]
        index += 1
    return None

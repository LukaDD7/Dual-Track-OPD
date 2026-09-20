"""FC-OPD native validation reward function.

Geometry3K MCQ remains the primary contract.  ViRL39K validation additionally
uses yes/no and answer strings containing times, ratios, expressions, LaTeX,
sets, coordinates, and units.  Explicit final answers are preserved as complete
strings so checkpoint selection does not systematically undercount those rows.
"""

from __future__ import annotations

import re
from math import isclose
from typing import Any, Sequence

from dual_track_opd.fc_opd.answer_extraction import extract_final_answer_candidate

LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
YESNO_RE = re.compile(r"(yes|no|true|false)", re.IGNORECASE)
MCQ_PREFIX_RE = re.compile(r"^\(?\s*([A-E])\s*[).:\]]\s*(.+)$", re.IGNORECASE)
MATH_GRADER_ALLOWED_RE = re.compile(
    r"^(?:\\(?:frac|dfrac|tfrac|sqrt|pi|circ|left|right)|[-+0-9^_{}()\s*/])+$"
)
NUMERIC_WITH_OPTIONAL_UNIT_RE = re.compile(
    r"^([-+]?(?:\d+(?:\.\d*)?|\.\d+))(?:\s*[a-zA-Z°%²^]+)?$"
)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Score a Geometry3K or ViRL39K-style response.

    Returns ``{"score": 1.0}`` for correct, ``{"score": 0.0}`` for wrong,
    and ``{"score": 0.0}`` for malformed responses.
    """
    del data_source, kwargs
    extra = extra_info or {}
    raw_choices = extra.get("choices")
    if raw_choices is None:
        choices = ()
    elif hasattr(raw_choices, "tolist"):
        choices = tuple(str(value) for value in raw_choices.tolist())
    elif isinstance(raw_choices, (list, tuple)):
        choices = tuple(raw_choices)
    else:
        choices = (str(raw_choices),)
    answer = extra.get("answer") or extra.get("answer_metadata") or ground_truth

    extracted = _extract_answer(solution_str)
    gold = _normalize_gold(answer, choices)
    if extracted is None or gold is None:
        return {"score": 0.0}
    return {"score": 1.0 if _answers_match(extracted, gold, choices) else 0.0}


def _extract_answer(response_text: str) -> str | None:
    candidate = extract_final_answer_candidate(response_text)
    if not candidate:
        return None
    yesno = YESNO_RE.fullmatch(_strip_answer_punctuation(candidate))
    if yesno:
        return "yes" if yesno.group(1).lower() in {"yes", "true"} else "no"
    letter = LETTER_RE.search(candidate)
    if letter and LETTER_RE.fullmatch(candidate.strip("() .")):
        return letter.group(1).upper()
    return candidate


def _normalize_gold(answer: Any, choices: Sequence[str]) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, dict):
        for key in ("answer", "label", "gold", "target", "value"):
            if key in answer:
                return _normalize_gold(answer[key], choices)
        return None
    text = str(answer).strip()
    if not text:
        return None
    if "\\boxed{" in text:
        unboxed = extract_final_answer_candidate(text)
        if unboxed is not None:
            text = unboxed
    yesno = YESNO_RE.fullmatch(_strip_answer_punctuation(text))
    if yesno:
        return "yes" if yesno.group(1).lower() in {"yes", "true"} else "no"
    letter = LETTER_RE.fullmatch(_strip_answer_punctuation(text))
    if letter:
        return letter.group(1).upper()
    if len(text) == 1 and text.isdigit():
        index = int(text)
        if 0 <= index < len(choices):
            return chr(ord("A") + index)
        if 1 <= index <= len(choices):
            return chr(ord("A") + index - 1)
    return text


def _answers_match(extracted: str, gold: str, choices: Sequence[str]) -> bool:
    # A gold MCQ letter is authoritative even when the model repeats the
    # option value after the label, e.g. ``A. 2`` or ``\text{(B) } 30``.
    gold_letter = _single_mcq_letter(gold)
    if gold_letter is not None:
        extracted_letter = _single_mcq_letter(extracted)
        if extracted_letter is not None:
            return extracted_letter == gold_letter

    extracted_norm = _canonical_answer(extracted)
    gold_norm = _canonical_answer(gold)
    if extracted_norm == gold_norm:
        return True
    if _numeric_equal(extracted_norm, gold_norm):
        return True
    if _math_equal(extracted, gold):
        return True
    extracted_letter = _single_mcq_letter(extracted)
    if extracted_letter and gold_letter:
        return extracted_letter == gold_letter
    if extracted_letter and 0 <= ord(extracted_letter) - ord("A") < len(choices):
        return _normalize_choice_value(choices[ord(extracted_letter) - ord("A")]) == gold_norm
    if gold_letter and 0 <= ord(gold_letter) - ord("A") < len(choices):
        return _normalize_choice_value(choices[ord(gold_letter) - ord("A")]) == extracted_norm
    return False


def _normalize_choice_value(text: str) -> str:
    stripped = str(text).strip()
    prefix = re.match(r"^\s*([A-E])[\s.)\);:-]+(.+)$", stripped, flags=re.IGNORECASE)
    if prefix:
        stripped = prefix.group(2).strip()
    number = NUMBER_RE.fullmatch(stripped)
    if number:
        return _normalize_number_text(stripped)
    return re.sub(r"\s+", "", stripped).lower()


def _canonical_answer(text: str) -> str:
    """Normalize formatting without discarding non-MCQ answer structure."""

    value = _strip_answer_punctuation(text)
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("\\left", "").replace("\\right", "")
    value = re.sub(r"\^\{?circ\}?", "°", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", "", value)
    return value.lower()


def _strip_answer_punctuation(text: str) -> str:
    value = str(text).strip().strip("*_`$ ")
    value = re.sub(r"[.。;；,，]+$", "", value).strip()
    return value


def _numeric_equal(extracted: str, gold: str) -> bool:
    extracted_match = NUMERIC_WITH_OPTIONAL_UNIT_RE.fullmatch(extracted)
    gold_match = NUMERIC_WITH_OPTIONAL_UNIT_RE.fullmatch(gold)
    if not extracted_match or not gold_match:
        return False
    try:
        extracted_value = float(extracted_match.group(1))
        gold_value = float(gold_match.group(1))
        return isclose(extracted_value, gold_value, rel_tol=1e-10, abs_tol=1e-12)
    except ValueError:
        return False


def _math_equal(extracted: str, gold: str) -> bool:
    """Use the pinned math grader only for single LaTeX math expressions.

    The grader is deliberately conservative.  It can be overly permissive for
    compound answers, equations, ranges, sets, coordinates, and text with
    units; those forms must rely on exact structural normalization instead.
    """

    if not (_is_single_math_expression(extracted) and _is_single_math_expression(gold)):
        return False

    try:
        from mathruler.grader import grade_answer
    except Exception as error:
        raise RuntimeError(
            "mathruler is required for math-expression validation but is unavailable"
        ) from error
    try:
        return bool(grade_answer(_normalize_latex(extracted), _normalize_latex(gold)))
    except Exception:
        return False


def _single_mcq_letter(text: str) -> str | None:
    value = _strip_answer_punctuation(_unwrap_latex_text(text))
    if not value:
        return None
    direct = re.fullmatch(r"\(?([A-E])\)?", value, flags=re.IGNORECASE)
    if direct:
        return direct.group(1).upper()
    prefix = MCQ_PREFIX_RE.fullmatch(value)
    if prefix:
        return prefix.group(1).upper()
    return None


def _unwrap_latex_text(text: str) -> str:
    """Unwrap simple ``\\text{...}`` and ``\\mathrm{...}`` wrappers."""

    value = str(text)
    for _ in range(3):
        replaced = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
        if replaced == value:
            break
        value = replaced
    return value


def _is_single_math_expression(text: str) -> bool:
    """Return True only for a conservative single-expression grader input."""

    value = _unwrap_latex_text(str(text)).strip().strip("$")
    if not value:
        return False
    if any(token in value.lower() for token in ("or ", " and ", "but")):
        return False
    return MATH_GRADER_ALLOWED_RE.fullmatch(value) is not None


def _normalize_latex(text: str) -> str:
    value = str(text).strip().strip("*_`$ ")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return re.sub(r"\s+", "", value)


def _normalize_number_text(text: str) -> str:
    value = float(text)
    return str(int(value)) if value.is_integer() else f"{value:.12g}"

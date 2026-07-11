"""Geometry3K reward function for FC-OPD training.

Uses the same extraction logic as the offline verifier so rewards are
consistent with the verifier_learning_value_gate.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from dual_track_opd.fc_opd.answer_extraction import extract_final_answer_candidate

LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Score a Geometry3K-style multiple-choice response.

    Returns ``{"score": 1.0}`` for correct, ``{"score": 0.0}`` for wrong,
    and ``{"score": 0.0}`` for malformed responses.
    """
    del data_source, kwargs
    extra = extra_info or {}
    choices = tuple(extra.get("choices") or ())
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
    letter = LETTER_RE.search(candidate)
    if letter:
        return letter.group(1).upper()
    number = NUMBER_RE.search(candidate)
    if number:
        return _normalize_number_text(number.group(0))
    return None


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
    letter = LETTER_RE.fullmatch(text)
    if letter:
        return letter.group(1).upper()
    if len(text) == 1 and text.isdigit():
        index = int(text)
        if 0 <= index < len(choices):
            return chr(ord("A") + index)
        if 1 <= index <= len(choices):
            return chr(ord("A") + index - 1)
    return _normalize_choice_value(text)


def _answers_match(extracted: str, gold: str, choices: Sequence[str]) -> bool:
    extracted_norm = _normalize_choice_value(extracted)
    gold_norm = _normalize_choice_value(gold)
    if extracted_norm == gold_norm:
        return True
    extracted_letter = extracted.upper() if extracted.upper() in {"A", "B", "C", "D"} else None
    gold_letter = gold.upper() if gold.upper() in {"A", "B", "C", "D"} else None
    if extracted_letter and gold_letter:
        return extracted_letter == gold_letter
    if extracted_letter and 0 <= ord(extracted_letter) - ord("A") < len(choices):
        return _normalize_choice_value(choices[ord(extracted_letter) - ord("A")]) == gold_norm
    if gold_letter and 0 <= ord(gold_letter) - ord("A") < len(choices):
        return _normalize_choice_value(choices[ord(gold_letter) - ord("A")]) == extracted_norm
    return False


def _normalize_choice_value(text: str) -> str:
    stripped = str(text).strip()
    prefix = re.match(r"^\s*([A-D])[\s.)\);:-]+(.+)$", stripped, flags=re.IGNORECASE)
    if prefix:
        stripped = prefix.group(2).strip()
    number = NUMBER_RE.fullmatch(stripped)
    if number:
        return _normalize_number_text(stripped)
    return re.sub(r"\s+", "", stripped).lower()


def _normalize_number_text(text: str) -> str:
    value = float(text)
    return str(int(value)) if value.is_integer() else f"{value:.12g}"

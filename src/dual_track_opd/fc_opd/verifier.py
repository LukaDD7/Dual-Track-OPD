"""Post-hoc Geometry3K verifier used only for offline outcome gates."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def verify_geometry3k_response(
    *,
    question: str,
    choices: Sequence[str],
    response_text: str,
    answer_metadata: Any,
) -> dict[str, Any]:
    del question
    extracted = extract_answer(response_text)
    gold = normalize_gold_answer(answer_metadata, choices)
    if extracted is None:
        return {
            "answer_extracted": None,
            "gold_answer": gold,
            "correct": None,
            "format_valid": False,
            "malformed": True,
            "reward": 0.0,
            "verifier_source": "geometry3k_rule",
        }

    correct = answers_match(extracted, gold, choices)
    return {
        "answer_extracted": extracted,
        "gold_answer": gold,
        "correct": bool(correct) if gold is not None else None,
        "format_valid": True,
        "malformed": False,
        "reward": 1.0 if correct else 0.25,
        "verifier_source": "geometry3k_rule",
    }


def extract_answer(response_text: str) -> str | None:
    match = ANSWER_RE.search(response_text or "")
    candidate = match.group(1).strip() if match else (response_text or "").strip()
    if not candidate:
        return None
    letter = LETTER_RE.search(candidate)
    if letter:
        return letter.group(1).upper()
    number = NUMBER_RE.search(candidate)
    if number:
        return _normalize_number_text(number.group(0))
    return None


def normalize_gold_answer(answer_metadata: Any, choices: Sequence[str]) -> str | None:
    if answer_metadata is None:
        return None
    if isinstance(answer_metadata, Mapping):
        for key in ("answer", "label", "gold", "target", "value"):
            if key in answer_metadata:
                normalized = normalize_gold_answer(answer_metadata[key], choices)
                if normalized is not None:
                    return normalized
        return None
    text = str(answer_metadata).strip()
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


def answers_match(extracted: str, gold: str | None, choices: Sequence[str]) -> bool:
    if gold is None:
        return False
    extracted_norm = _normalize_choice_value(extracted)
    gold_norm = _normalize_choice_value(gold)
    if extracted_norm == gold_norm:
        return True
    extracted_letter = extracted.upper() if extracted.upper() in {"A", "B", "C", "D"} else None
    gold_letter = gold.upper() if gold.upper() in {"A", "B", "C", "D"} else None
    if extracted_letter and gold_letter:
        return extracted_letter == gold_letter
    if extracted_letter:
        idx = ord(extracted_letter) - ord("A")
        return 0 <= idx < len(choices) and _normalize_choice_value(choices[idx]) == gold_norm
    if gold_letter:
        idx = ord(gold_letter) - ord("A")
        return 0 <= idx < len(choices) and _normalize_choice_value(choices[idx]) == extracted_norm
    return False


def _normalize_choice_value(text: str) -> str:
    stripped = str(text).strip()
    prefix = re.match(r"^\s*([A-D])[\s\.\):;-]+(.+)$", stripped, flags=re.IGNORECASE)
    if prefix:
        stripped = prefix.group(2).strip()
    number = NUMBER_RE.fullmatch(stripped)
    if number:
        return _normalize_number_text(stripped)
    return re.sub(r"\s+", "", stripped).lower()


def _normalize_number_text(text: str) -> str:
    value = float(text)
    if value.is_integer():
        return str(int(value))
    return f"{value:.12g}"

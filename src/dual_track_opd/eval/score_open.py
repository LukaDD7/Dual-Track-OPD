"""Open-ended scoring placeholders."""

from __future__ import annotations


def normalize_answer(text: str) -> str:
    """Normalize answer text for lightweight comparisons."""

    return " ".join(text.strip().lower().split())


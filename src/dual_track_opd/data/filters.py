"""Dataset filtering helpers."""

from __future__ import annotations


def keep_all(example: object) -> bool:
    """Default filter that keeps every example."""

    return True


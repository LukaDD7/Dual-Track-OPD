"""Data collator placeholders."""

from __future__ import annotations


def identity_collate(batch: list[object]) -> list[object]:
    """Return a batch unchanged for smoke tests."""

    return batch


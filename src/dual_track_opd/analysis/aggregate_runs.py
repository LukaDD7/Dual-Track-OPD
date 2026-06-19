"""Experiment aggregation placeholders."""

from __future__ import annotations


def aggregate_rows(rows: list[dict[str, object]]) -> dict[str, int]:
    """Return a minimal aggregate summary."""

    return {"runs": len(rows)}


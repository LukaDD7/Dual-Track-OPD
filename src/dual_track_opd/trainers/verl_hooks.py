"""`verl` hook placeholders."""

from __future__ import annotations


def describe_hooks() -> list[str]:
    """List planned hook points for the backend."""

    return ["loss", "rollout", "metrics"]


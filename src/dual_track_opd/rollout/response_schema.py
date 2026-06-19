"""Common rollout response schema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutResponse:
    text: str
    finish_reason: str = "unknown"
    error: str | None = None


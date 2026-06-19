"""Minimal VLM dataset container."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VLMExample:
    dataset: str
    question: str
    image_path: str | None = None
    answer: str | None = None


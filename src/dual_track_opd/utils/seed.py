"""Reproducibility seed helpers."""

from __future__ import annotations

import random


def seed_python(seed: int) -> None:
    """Seed Python's built-in random module."""

    random.seed(seed)


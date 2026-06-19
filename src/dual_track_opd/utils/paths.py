"""Path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def env_path(name: str, default: str | None = None) -> Path:
    """Return a path from an environment variable or default."""

    value = os.environ.get(name, default)
    if value is None:
        raise KeyError(f"missing environment variable: {name}")
    return Path(value).expanduser()


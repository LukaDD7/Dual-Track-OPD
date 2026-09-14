"""ViRL39K adapter placeholder.

ViRL39K is the second clean-data target after Geometry3K. The schema should be
inspected before implementing a normalizer, so this module intentionally raises
with a clear message for now.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_virl39k_records(path: str | Path, *, source_dataset: str = "virl39k") -> list[dict[str, Any]]:
    del path, source_dataset
    raise NotImplementedError("ViRL39K adapter requires schema inspection before implementation")

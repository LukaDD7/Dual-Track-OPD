#!/usr/bin/env python
"""Build clean-data 4C free/task evidence cache."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dual_track_opd.fc_opd.evidence_generation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

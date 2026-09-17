#!/usr/bin/env python
"""Geometry3K 4C signal-audit smoke.

This entry currently reuses the clean-data 4C offline builder path because the
trainable JSONL contains the raw teacher top-k scores plus signal summaries
needed for audit. Use small limits only.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dual_track_opd.fc_opd.four_condition_offline_builder import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

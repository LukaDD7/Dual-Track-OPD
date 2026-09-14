#!/usr/bin/env python
"""Build trainable Vision-OPD-6K FC-OPD 2C offline scores."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dual_track_opd.fc_opd.vision_opd_2c_offline_builder import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

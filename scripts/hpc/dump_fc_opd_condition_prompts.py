#!/usr/bin/env python
"""HPC entrypoint for dumping exact FC-OPD condition prompts."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dual_track_opd.fc_opd.condition_prompt_dump import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Run the FC-OPD dataset signal audit.

This is a thin HPC-friendly entrypoint around
``dual_track_opd.fc_opd.dataset_signal_audit``. It performs no verl integration
and starts no training.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dual_track_opd.fc_opd.dataset_signal_audit import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

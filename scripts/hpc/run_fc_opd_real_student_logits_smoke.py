#!/usr/bin/env python
"""Real student-logits adapter smoke for the offline FC-OPD loss.

Loads a real Qwen3-VL / Qwen3.5-VL student model, runs a teacher-forced
multimodal forward on records from an offline-score JSONL, extracts logits at the
exact response-token positions, feeds them into the existing FC-OPD loss/router
path, and verifies that backward reaches real model parameters.

This is the bridge from synthetic student logits to real student logits. It is
not verl integration: no teacher service is needed and third_party/verl is never
imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dual_track_opd.fc_opd.real_student_smoke import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

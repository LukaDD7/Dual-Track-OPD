#!/usr/bin/env python
"""Real student optimizer-step smoke for the offline FC-OPD loss.

Loads a real Qwen3-VL / Qwen3.5-VL student, runs a teacher-forced multimodal
forward on offline-score records, computes the FC-OPD loss, backpropagates, and
runs a few Adam steps. Verifies that a real trainable parameter actually changes
(``param_delta_norm > 0``), gradients are finite and non-zero, the loss is
finite, all four conditions are consumed, and reports the lm_head/embed tied
status.

This is the bridge to a real training step; it is not verl integration. No
teacher service is needed and ``third_party/verl`` is never imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dual_track_opd.fc_opd.real_student_smoke import min_train_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(min_train_main())

#!/usr/bin/env python
"""Offline FC-OPD loss/backward smoke over a recorded offline-score dataset.

Reads recorded teacher scores, rebuilds tensors and chunk masks, attaches
synthetic student logits (``requires_grad=True``), and runs the real
``route_condition_weights`` -> ``compute_fc_opd_loss`` -> ``backward`` path.
Verifies a finite loss, present and finite student-logit gradients, chunk masks
aligned to the response token length, and that all four conditions are consumed.

No GPU, student model, or teacher service is needed, and ``third_party/verl`` is
never imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dual_track_opd.fc_opd.offline_loss import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

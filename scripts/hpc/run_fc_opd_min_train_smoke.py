#!/usr/bin/env python
"""Minimal Adam optimizer-update smoke for the offline FC-OPD loss.

Reads a recorded offline-score dataset, attaches a synthetic ``student_logits``
parameter (``requires_grad=True``) per record, and runs Adam for a few steps
against the recorded teacher scores. Per step it reports loss, gradient norm,
the norm of the logit update, and the consumed conditions, then verifies that
the loss and gradients stay finite, that every ``optimizer.step()`` changes the
logits, and that all four conditions are consumed.

No student model or teacher service is loaded, and ``third_party/verl`` is never
imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dual_track_opd.fc_opd.offline_loss import min_train_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(min_train_main())

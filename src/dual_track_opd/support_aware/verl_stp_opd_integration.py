"""verl 0.7.1 integration entry for the STP-OPD pilot (thin, opt-in).

verl 0.7.1 exposes two official extension points that cover STP-OPD without a
source patch (see patches/verl/README.md for the rationale):

1. ``verl.trainer.ppo.core_algos.register_policy_loss`` — register the
   ``stp_opd`` policy loss and select it via
   ``actor_rollout_ref.actor.policy_loss``.
2. the post-rollout hook (``fc_opd_post_rollout_hook`` pattern) — attach
   teacher-scored tensors and masks to the rollout batch so the loss fn can
   read them from the batch's custom fields.

This module imports verl lazily; importing it without a verl install is a
no-op.  Until the end-to-end wiring is validated in the pinned HPC verl
environment, the registered loss raises (fails loudly) instead of silently
training the wrong objective (handoff §6 P0).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable


def _register_if_available() -> bool:
    try:
        core_algos = importlib.import_module("verl.trainer.ppo.core_algos")
    except ImportError:
        return False

    @core_algos.register_policy_loss("stp_opd")  # type: ignore[attr-defined]
    def stp_opd_policy_loss(
        old_log_prob: Any,
        log_prob: Any,
        advantages: Any,
        response_mask: Any,
        loss_agg_mode: str,
        config: Any,
        rollout_log_probs: Any = None,
    ) -> tuple[Any, dict[str, Any]]:
        """STP-OPD policy loss entry.

        Production wiring reads ``teacher_logits``/``sampled_ids``/
        ``prefix_mask``/``suffix_mask`` from the rollout batch's custom fields
        (attached by the post-rollout hook) and delegates to
        ``support_transition_train.train_step_loss``.  That wiring is not
        validated yet, so calling this raises rather than training a wrong
        objective silently.
        """

        del old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config, rollout_log_probs
        raise NotImplementedError(
            "stp_opd policy loss is registered but not wired: attach "
            "teacher_logits/prefix_mask/suffix_mask/sampled_ids via the "
            "post-rollout hook and delegate to "
            "dual_track_opd.support_aware.support_transition_train.train_step_loss "
            "(handoff §6 P0 validation required)"
        )

    return True


_REGISTERED = _register_if_available()


def stp_opd_loss_registered() -> bool:
    """True when verl was importable and the stp_opd loss was registered."""

    return _REGISTERED

"""Support-state classification for frozen-policy diagnostics.

Classifies each prompt based on the observed distribution of correct vs
incorrect student rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SupportState(str, Enum):
    EXPOSED = "exposed"
    CORRECT_TAIL = "correct_tail"
    NO_CORRECT_OBSERVED = "no_correct_observed"
    OTHER = "other"


@dataclass(frozen=True)
class SupportStateResult:
    sample_uid: str
    K: int
    correct_count: int
    greedy_correct: bool | None
    support_state: SupportState
    unique_response_count: int
    duplicate_rollout_rate: float
    note: str = ""


def classify_support_state(
    *,
    greedy_correct: bool | None,
    correct_count: int,
    K: int,
) -> SupportState:
    """Classify a prompt's observed support state.

    Args:
        greedy_correct: Whether the greedy (T=0) response was correct.  ``None``
            when greedy generation was skipped or the answer was unparseable.
        correct_count: Number of correct stochastic rollouts (0 ≤ c ≤ K).
        K: Total number of stochastic rollouts.

    Returns:
        The assigned support state.

    Classification rules (runbook §4.5):

    * **exposed**: greedy is correct OR at least half of stochastic rollouts
      are correct.  The correct mode is already well-represented.
    * **correct_tail**: greedy is NOT correct AND 0 < correct_count < K/2.
      Correct answers exist but are not the dominant mode — this is the
      regime where support-gated OPD may help.
    * **no_correct_observed**: greedy is NOT correct AND zero stochastic
      rollouts are correct.  No correct response was observed under the
      sampling budget.
    * **other**: remaining boundary cases (e.g. greedy=None).
    """
    if greedy_correct is None:
        return SupportState.OTHER

    if greedy_correct or correct_count / max(K, 1) >= 0.5:
        return SupportState.EXPOSED

    if correct_count == 0:
        return SupportState.NO_CORRECT_OBSERVED

    if 0 < correct_count < K / 2:
        return SupportState.CORRECT_TAIL

    return SupportState.OTHER

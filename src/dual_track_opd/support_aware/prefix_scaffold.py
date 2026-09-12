"""STP-OPD scaffold primitives (CPU-testable, no model I/O).

Implements the regionalized training contract from the support-transition
prefix OPD plan (docs/two_track_next_experiment_handoff_20260806.md, §4.4):

    L = lambda_P * L_prefix_FKL
      + lambda_D * L_suffix_teacher_RKL_K1
      + lambda_R * L_suffix_task_GRPO

Each region is masked and normalized independently so prefix length cannot
implicitly change the loss weight.  The scaffold branch fixes the verified
answer-free teacher prefix as context; the unscaffolded branch has no prefix.
All loss math returns tensors and preserves autograd (handoff
docs/cc_causal_state_to_stp_handoff_20260813.md §4.1): empty regions produce a
zero tensor with a live grad_fn so gradients stay zero instead of NaN, and no
trainable loss value is ever converted to Python float here.  This module
contains no training loop and no model loading.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


DEFAULT_SCHEDULE: tuple[tuple[int, int, float], ...] = (
    (1, 20, 0.75),
    (21, 40, 0.50),
    (41, 60, 0.25),
)


@dataclass(frozen=True)
class RegionTerms:
    """Region-normalized loss terms; zero tensors mean no masked tokens."""

    prefix_fkl: torch.Tensor
    suffix_rkl: torch.Tensor
    suffix_pg: torch.Tensor


def region_masks(
    response_length: int,
    prefix_length: int,
    *,
    scaffolded: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (prefix_mask, suffix_mask) over response token positions.

    Scaffolded: prefix occupies positions [0, prefix_length), suffix the rest.
    Unscaffolded: prefix mask is empty, suffix mask covers the whole response.
    """

    if response_length <= 0:
        raise ValueError("response_length must be positive")
    if prefix_length < 0 or prefix_length > response_length:
        raise ValueError("prefix_length must lie in [0, response_length]")
    positions = torch.arange(response_length)
    if scaffolded:
        prefix_mask = positions < prefix_length
        suffix_mask = positions >= prefix_length
    else:
        prefix_mask = torch.zeros(response_length, dtype=torch.bool)
        suffix_mask = torch.ones(response_length, dtype=torch.bool)
    return prefix_mask, suffix_mask


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean over masked positions, preserving autograd.

    An empty mask returns ``values.sum() * 0.0``: a zero tensor whose grad_fn
    keeps the autograd graph intact (gradients are zero, never NaN).  Never
    converts the result to a Python float.
    """

    if values.shape != mask.shape:
        raise ValueError(f"values {tuple(values.shape)} != mask {tuple(mask.shape)}")
    count = int(mask.sum())
    if count == 0:
        return values.sum() * 0.0
    return values[mask].mean()


def composed_loss(
    prefix_fkl: torch.Tensor,
    suffix_rkl: torch.Tensor,
    suffix_pg: torch.Tensor,
    prefix_mask: torch.Tensor,
    suffix_mask: torch.Tensor,
    *,
    lambda_prefix: float = 1.0,
    lambda_distill: float = 1.0,
    lambda_task: float = 1.0,
) -> tuple[torch.Tensor, RegionTerms]:
    """Regionalized STP-OPD objective with independent per-region means.

    Returns a differentiable scalar tensor.  Regions are disjoint by contract
    (see ``region_masks``/``validate_masks``) and each is normalized by its own
    masked token count, so prefix length cannot rescale the suffix terms.
    """

    prefix_term = masked_mean(prefix_fkl, prefix_mask)
    distill_term = masked_mean(suffix_rkl, suffix_mask)
    task_term = masked_mean(suffix_pg, suffix_mask)
    total = (
        lambda_prefix * prefix_term
        + lambda_distill * distill_term
        + lambda_task * task_term
    )
    return total, RegionTerms(
        prefix_fkl=prefix_term,
        suffix_rkl=distill_term,
        suffix_pg=task_term,
    )


def validate_masks(
    prefix_mask: torch.Tensor,
    suffix_mask: torch.Tensor,
    *,
    require_coverage: bool = True,
) -> None:
    """Fail-fast checks: same shape, disjoint; optionally full coverage."""

    if prefix_mask.shape != suffix_mask.shape or prefix_mask.dtype != torch.bool:
        raise ValueError("masks must be same-shape boolean tensors")
    if bool((prefix_mask & suffix_mask).any()):
        raise ValueError("prefix and suffix masks overlap")
    if require_coverage and not bool((prefix_mask | suffix_mask).all()):
        raise ValueError("prefix+suffix masks do not cover the response")


def scaffold_fraction(step: int, schedule: Sequence[tuple[int, int, float]] = DEFAULT_SCHEDULE) -> float:
    """Fraction of prompts scaffolded at this step (0 when outside the schedule)."""

    if step <= 0:
        return 0.0
    for start, end, fraction in schedule:
        if start <= step <= end:
            return float(fraction)
    return 0.0


def assign_scaffold(
    prompt_ids: Sequence[str],
    *,
    step: int,
    seed: int,
    schedule: Sequence[tuple[int, int, float]] = DEFAULT_SCHEDULE,
) -> dict[str, bool]:
    """Deterministic per-prompt scaffold flag for one step."""

    fraction = scaffold_fraction(step, schedule)
    rng = random.Random(f"stp-scaffold-step{step}-seed{seed}")
    return {
        str(prompt_id): rng.random() < fraction
        for prompt_id in prompt_ids
    }


def paired_batch(
    prompt_ids: Sequence[str],
    *,
    step: int,
    seed: int,
    schedule: Sequence[tuple[int, int, float]] = DEFAULT_SCHEDULE,
) -> list[dict[str, object]]:
    """Build one paired batch.

    Contract (handoff §4.2):
    - every prompt appears exactly twice, once per branch, so every comparison
      retains prompt pairing;
    - each record carries ``assigned_scaffolded`` from ``assign_scaffold`` so
      the training loop knows which branch realizes this step's scaffold share
      (the flags are used, not computed and dropped);
    - order is deterministic for a fixed (step, seed) pair.
    """

    flags = assign_scaffold(prompt_ids, step=step, seed=seed, schedule=schedule)
    records: list[dict[str, object]] = []
    for prompt_id in prompt_ids:
        records.append({
            "prompt_id": str(prompt_id),
            "scaffolded": True,
            "assigned_scaffolded": bool(flags[str(prompt_id)]),
        })
        records.append({
            "prompt_id": str(prompt_id),
            "scaffolded": False,
            "assigned_scaffolded": bool(flags[str(prompt_id)]),
        })
    rng = random.Random(f"stp-paired-batch-step{step}-seed{seed}")
    rng.shuffle(records)
    return records


def realized_scaffold_share(records: Sequence[Mapping[str, object]]) -> float:
    """Fraction of paired prompts whose assigned branch is scaffolded."""

    assigned: dict[str, bool] = {}
    for record in records:
        prompt_id = str(record["prompt_id"])
        assigned[prompt_id] = bool(record["assigned_scaffolded"])
    if not assigned:
        return 0.0
    return sum(assigned.values()) / len(assigned)


def hybrid_prefix_ids(
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
) -> tuple[int, ...]:
    """Exact token ids of the scaffolded hybrid response (prefix + suffix)."""

    return tuple(int(value) for value in prefix_ids) + tuple(int(value) for value in suffix_ids)

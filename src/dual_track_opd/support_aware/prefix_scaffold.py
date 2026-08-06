"""STP-OPD scaffold primitives (CPU-testable, no model I/O).

Implements the regionalized training contract from the support-transition
prefix OPD plan (docs/two_track_next_experiment_handoff_20260806.md, §4.4):

    L = lambda_P * L_prefix_FKL
      + lambda_D * L_suffix_teacher_RKL_K1
      + lambda_R * L_suffix_task_GRPO

Each region is masked and normalized independently so prefix length cannot
implicitly change the loss weight.  The scaffold branch fixes the verified
answer-free teacher prefix as context; the unscaffolded branch has no prefix.
This module contains no training loop and no model loading.
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
    """Region-normalized loss terms; None means the region had no masked tokens."""

    prefix_fkl: float | None
    suffix_rkl: float | None
    suffix_pg: float | None


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
) -> float | None:
    """Mean over masked positions; None when the mask selects nothing."""

    if values.shape != mask.shape:
        raise ValueError(f"values {tuple(values.shape)} != mask {tuple(mask.shape)}")
    count = int(mask.sum())
    if count == 0:
        return None
    return float(values[mask].mean())


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
) -> tuple[float, RegionTerms]:
    """Regionalized STP-OPD objective with independent per-region means."""

    prefix_term = masked_mean(prefix_fkl, prefix_mask)
    distill_term = masked_mean(suffix_rkl, suffix_mask)
    task_term = masked_mean(suffix_pg, suffix_mask)
    total = (
        lambda_prefix * (prefix_term or 0.0)
        + lambda_distill * (distill_term or 0.0)
        + lambda_task * (task_term or 0.0)
    )
    return float(total), RegionTerms(
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
    """Build one batch where every prompt appears in both scaffolded and
    unscaffolded branches (paired by prompt), with a deterministic shuffle."""

    flags = assign_scaffold(prompt_ids, step=step, seed=seed, schedule=schedule)
    records: list[dict[str, object]] = []
    for prompt_id in prompt_ids:
        records.append({"prompt_id": str(prompt_id), "scaffolded": True})
        records.append({"prompt_id": str(prompt_id), "scaffolded": False})
    rng = random.Random(f"stp-paired-batch-step{step}-seed{seed}")
    rng.shuffle(records)
    return records


def hybrid_prefix_ids(
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
) -> tuple[int, ...]:
    """Exact token ids of the scaffolded hybrid response (prefix + suffix)."""

    return tuple(int(value) for value in prefix_ids) + tuple(int(value) for value in suffix_ids)

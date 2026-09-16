"""Rollout-level NLL-TailOPD weighting.

V1 intentionally changes only the relative weight of sibling student rollouts.
It does not add correctness routing, token-level weighting, or teacher gating.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class TailRolloutWeights:
    """Detached NLL-TailOPD rollout weights.

    ``weights`` sums to one within each prompt group. ``scales`` is ``K * w``
    and has mean one within each complete group, so uniform scores exactly
    recover the unweighted OPD gradient scale.
    """

    weights: torch.Tensor
    scales: torch.Tensor
    scores: torch.Tensor
    zscores: torch.Tensor
    group_sizes: torch.Tensor
    group_ids: tuple[Hashable, ...]


def _as_group_ids(group_ids: Sequence[Hashable] | torch.Tensor) -> tuple[Hashable, ...]:
    if isinstance(group_ids, torch.Tensor):
        if group_ids.ndim != 1:
            raise ValueError(f"group_ids must be one-dimensional, got shape {tuple(group_ids.shape)}")
        values = group_ids.detach().cpu().tolist()
    else:
        values = list(group_ids)
    if len(values) == 0:
        return ()
    if any(isinstance(value, torch.Tensor) and value.ndim != 0 for value in values):
        raise ValueError("group_ids entries must be scalar values")
    return tuple(value.item() if isinstance(value, torch.Tensor) else value for value in values)


def _nested_to_padded(value: torch.Tensor, padding_value: float) -> torch.Tensor:
    if getattr(value, "is_nested", False):
        return value.to_padded_tensor(padding_value)
    return value


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return 0.0
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = x_centered.pow(2).sum().sqrt() * y_centered.pow(2).sum().sqrt()
    if denominator.item() == 0.0:
        return 0.0
    return (x_centered * y_centered).sum().div(denominator).item()


def compute_nll_tail_rollout_weights(
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: Sequence[Hashable] | torch.Tensor,
    *,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> TailRolloutWeights:
    """Compute sequence-normalized NLL and within-prompt softmax weights.

    Args:
        old_log_probs: Old-policy log probabilities with shape ``[B, T]``.
            Values outside the response mask are ignored.
        response_mask: Response mask with shape ``[B, T]``.
        group_ids: One prompt identifier per rollout. Sibling rollouts need not
            be contiguous.
        temperature: Softmax temperature ``tau``. Must be positive.
        eps: Numerical epsilon used for z-score normalization and the near-zero
            group standard-deviation test.

    Returns:
        Detached rollout weights, ``K*w`` scales, NLL scores, z-scores, group
        sizes, and the input group identifiers.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")

    old_log_probs = _nested_to_padded(old_log_probs.detach(), 0.0).float()
    response_mask = _nested_to_padded(response_mask.detach(), False).bool()
    ids = _as_group_ids(group_ids)

    if old_log_probs.ndim != 2 or response_mask.ndim != 2:
        raise ValueError(
            "old_log_probs and response_mask must both have shape [B, T], got "
            f"{tuple(old_log_probs.shape)} and {tuple(response_mask.shape)}"
        )
    if old_log_probs.shape != response_mask.shape:
        raise ValueError(
            f"old_log_probs shape {tuple(old_log_probs.shape)} does not match "
            f"response_mask shape {tuple(response_mask.shape)}"
        )
    batch_size = old_log_probs.shape[0]
    if len(ids) != batch_size:
        raise ValueError(f"group_ids length {len(ids)} does not match batch size {batch_size}")

    if batch_size == 0:
        empty = old_log_probs.new_zeros((0,), dtype=torch.float32)
        return TailRolloutWeights(
            weights=empty,
            scales=empty.clone(),
            scores=empty.clone(),
            zscores=empty.clone(),
            group_sizes=empty.clone(),
            group_ids=(),
        )

    lengths = response_mask.sum(dim=-1).to(old_log_probs.dtype)
    if torch.any(lengths <= 0):
        empty_rows = torch.nonzero(lengths <= 0).flatten().detach().cpu().tolist()
        raise ValueError(f"response_mask has zero valid tokens for rollouts {empty_rows}")

    token_log_prob_sum = (old_log_probs * response_mask.to(old_log_probs.dtype)).sum(dim=-1)
    scores = (-token_log_prob_sum / lengths).detach()
    if not torch.isfinite(scores).all():
        raise ValueError("NLL scores contain NaN or Inf values")

    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, group_id in enumerate(ids):
        groups[group_id].append(index)

    weights = torch.zeros_like(scores)
    zscores = torch.zeros_like(scores)
    group_sizes = torch.zeros_like(scores)
    for indices in groups.values():
        index_tensor = torch.as_tensor(indices, device=scores.device, dtype=torch.long)
        group_scores = scores.index_select(0, index_tensor)
        group_size = len(indices)
        group_sizes.index_fill_(0, index_tensor, float(group_size))

        mean = group_scores.mean()
        std = group_scores.var(unbiased=False).sqrt()
        if std.item() <= eps:
            group_weights = torch.full_like(group_scores, 1.0 / group_size)
            group_zscores = torch.zeros_like(group_scores)
        else:
            group_zscores = (group_scores - mean) / (std + eps)
            group_weights = torch.softmax(group_zscores / temperature, dim=0)

        weights.index_copy_(0, index_tensor, group_weights)
        zscores.index_copy_(0, index_tensor, group_zscores)

    scales = (weights * group_sizes).detach()
    result = TailRolloutWeights(
        weights=weights.detach(),
        scales=scales.detach(),
        scores=scores.detach(),
        zscores=zscores.detach(),
        group_sizes=group_sizes.detach(),
        group_ids=ids,
    )
    assert not result.weights.requires_grad
    assert not result.scales.requires_grad
    return result


def compute_diagnostics(
    result: TailRolloutWeights,
    response_mask: torch.Tensor,
    *,
    prefix: str = "tail_opd",
    eps: float = 1e-6,
) -> dict[str, float]:
    """Return compact training diagnostics for a complete global batch."""
    if result.scores.numel() == 0:
        raise ValueError("cannot compute TailOPD diagnostics for an empty batch")

    scores = result.scores.detach().float()
    weights = result.weights.detach().float()
    scales = result.scales.detach().float()
    lengths = response_mask.detach().sum(dim=-1).float()
    if lengths.shape[0] != scores.shape[0]:
        raise ValueError("response_mask batch size does not match TailOPD scores")

    group_count = 0
    near_zero_std = 0
    grouped: Mapping[Hashable, list[int]] = defaultdict(list)
    for index, group_id in enumerate(result.group_ids):
        grouped[group_id].append(index)
    for indices in grouped.values():
        group_scores = scores[indices]
        std = group_scores.var(unbiased=False).sqrt().item()
        group_count += 1
        if std <= eps:
            near_zero_std += 1

    quantiles = torch.quantile(scores, torch.tensor([0.1, 0.5, 0.9], device=scores.device))
    effective_rollouts = sum((1.0 / weights[indices].pow(2).sum()).item() for indices in grouped.values())
    effective_rollouts /= group_count

    diagnostics = {
        f"{prefix}/nll_mean": scores.mean().item(),
        f"{prefix}/nll_std": scores.std(unbiased=False).item() if scores.numel() > 1 else 0.0,
        f"{prefix}/nll_p10": quantiles[0].item(),
        f"{prefix}/nll_p50": quantiles[1].item(),
        f"{prefix}/nll_p90": quantiles[2].item(),
        f"{prefix}/weight_min": weights.min().item(),
        f"{prefix}/weight_max": weights.max().item(),
        f"{prefix}/weight_entropy": (-(weights * weights.clamp_min(1e-12).log()).sum()).item(),
        f"{prefix}/effective_rollouts": effective_rollouts,
        f"{prefix}/near_zero_std_group_fraction": near_zero_std / group_count,
        f"{prefix}/nll_length_correlation": _pearson(scores, lengths),
        f"{prefix}/scale_mean": scales.mean().item(),
        f"{prefix}/scale_max": scales.max().item(),
        f"{prefix}/group_size_mean": float(group_count and sum(len(v) for v in grouped.values()) / group_count),
    }
    return diagnostics

"""Framework-neutral implementation of the VA-OPD weighting equations.

The equations follow arXiv:2605.21924, Equations (2)-(6).  This module only
constructs fixed weights and masks.  The backend remains responsible for the
reverse-KL estimator and gradient calculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Sequence

import torch

VA_TOP_FRACTION = 0.20
VA_LAMBDA = 0.50
VA_TAU = 1.0


@dataclass(frozen=True)
class VAWeightingResult:
    """All fixed tensors needed to apply the grouped VA-OPD objective."""

    visual_advantage: torch.Tensor
    rollout_mean_va: torch.Tensor
    rollout_weights: torch.Tensor
    high_mask: torch.Tensor
    low_mask: torch.Tensor
    token_weights: torch.Tensor
    prompt_group_count: int
    degenerate_sequence_count: int


def compute_rollout_weights(
    visual_advantage: torch.Tensor,
    *,
    response_mask: torch.Tensor,
    prompt_ids: Sequence[Hashable],
    expected_rollouts: int | None = None,
    tau: float = VA_TAU,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[Hashable, list[int]]]:
    """Compute Equation (3)-(4) weights within each sibling-rollout group.

    The paper's worked example uses the population standard deviation.  Using
    PyTorch's default unbiased sample standard deviation does not reproduce the
    reported ``[0.656, 0.206, 0.095, 0.044]`` weights.
    """

    visual_advantage, response_mask = _validate_token_tensors(visual_advantage, response_mask)
    if tau <= 0:
        raise ValueError("tau must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    groups = _prompt_groups(prompt_ids, visual_advantage.shape[0])
    _validate_group_sizes(groups, expected_rollouts)

    valid_counts = response_mask.sum(dim=1)
    if (valid_counts == 0).any():
        rows = torch.nonzero(valid_counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"VA-OPD rows contain no valid response tokens: {rows}")
    summaries = (visual_advantage.float() * response_mask).sum(dim=1) / valid_counts.float()

    weights = torch.zeros_like(summaries, dtype=torch.float32)
    for indices in groups.values():
        idx = torch.as_tensor(indices, dtype=torch.long, device=visual_advantage.device)
        scores = summaries[idx]
        if len(indices) == 1:
            weights[idx] = 1.0
            continue
        sigma = scores.std(unbiased=False)
        if not torch.isfinite(sigma) or sigma <= eps:
            weights[idx] = 1.0 / float(len(indices))
            continue
        z_scores = (scores - scores.mean()) / (sigma + eps)
        weights[idx] = torch.softmax(z_scores / tau, dim=0)

    return weights, summaries, groups


def split_high_low_va(
    visual_advantage: torch.Tensor,
    *,
    response_mask: torch.Tensor,
    top_fraction: float = VA_TOP_FRACTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split valid tokens into the top-VA fraction and its complement."""

    visual_advantage, response_mask = _validate_token_tensors(visual_advantage, response_mask)
    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must be in (0, 1)")

    high = torch.zeros_like(response_mask)
    for row in range(visual_advantage.shape[0]):
        valid = torch.nonzero(response_mask[row], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        count = min(valid.numel(), max(1, int(math.ceil(valid.numel() * top_fraction))))
        positions = torch.topk(visual_advantage[row, valid], k=count, largest=True, sorted=False).indices
        high[row, valid[positions]] = True
    return high, response_mask & ~high


def build_va_token_weights(
    teacher_full_log_probs: torch.Tensor,
    teacher_degraded_log_probs: torch.Tensor,
    *,
    response_mask: torch.Tensor,
    prompt_ids: Sequence[Hashable],
    expected_rollouts: int = 4,
    top_fraction: float = VA_TOP_FRACTION,
    lambda_high: float = VA_LAMBDA,
    tau: float = VA_TAU,
    eps: float = 1e-8,
) -> VAWeightingResult:
    """Build exact per-token multipliers for a native verl PG reverse-KL loss.

    With ``actor.loss_agg_mode=seq-mean-token-sum``, multiplying the sampled
    reverse-KL estimator by ``token_weights`` yields Equation (6), averaged over
    prompts rather than over variable-length tokens.
    """

    if teacher_full_log_probs.shape != teacher_degraded_log_probs.shape:
        raise ValueError("full and degraded teacher log-probs must have identical shapes")
    if expected_rollouts < 2:
        raise ValueError("faithful VA-OPD requires expected_rollouts >= 2")
    if not 0.0 <= lambda_high <= 1.0:
        raise ValueError("lambda_high must be in [0, 1]")
    if not torch.isfinite(teacher_full_log_probs).all() or not torch.isfinite(teacher_degraded_log_probs).all():
        raise ValueError("teacher log-probs must be finite before VA weighting")

    response_mask = response_mask.to(device=teacher_full_log_probs.device, dtype=torch.bool)
    _validate_token_tensors(teacher_full_log_probs, response_mask)
    visual_advantage = (teacher_full_log_probs.float() - teacher_degraded_log_probs.float()).clamp_min(0.0)
    visual_advantage = visual_advantage * response_mask

    rollout_weights, rollout_mean_va, groups = compute_rollout_weights(
        visual_advantage,
        response_mask=response_mask,
        prompt_ids=prompt_ids,
        expected_rollouts=expected_rollouts,
        tau=tau,
        eps=eps,
    )
    high, low = split_high_low_va(
        visual_advantage,
        response_mask=response_mask,
        top_fraction=top_fraction,
    )

    token_weights = torch.zeros_like(visual_advantage, dtype=torch.float32)
    degenerate = 0
    for row in range(visual_advantage.shape[0]):
        high_count = int(high[row].sum().item())
        low_count = int(low[row].sum().item())
        group_size = len(groups[prompt_ids[row]])
        rollout_scale = float(group_size) * rollout_weights[row]
        if high_count and low_count:
            token_weights[row, high[row]] = rollout_scale * lambda_high / float(high_count)
            token_weights[row, low[row]] = rollout_scale * (1.0 - lambda_high) / float(low_count)
        elif high_count:
            # Equation (5) is undefined for a one-token response.  Preserve the
            # rollout's total mass instead of silently halving the update.
            degenerate += 1
            token_weights[row, high[row]] = rollout_scale / float(high_count)
        else:
            raise ValueError(f"row {row} contains no high-VA tokens")

    return VAWeightingResult(
        visual_advantage=visual_advantage,
        rollout_mean_va=rollout_mean_va,
        rollout_weights=rollout_weights,
        high_mask=high,
        low_mask=low,
        token_weights=token_weights,
        prompt_group_count=len(groups),
        degenerate_sequence_count=degenerate,
    )


def _validate_token_tensors(values: torch.Tensor, response_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2:
        raise ValueError("token values must have shape [batch, response_length]")
    if response_mask.shape != values.shape:
        raise ValueError("response_mask must match token-value shape")
    return values, response_mask.to(device=values.device, dtype=torch.bool)


def _prompt_groups(prompt_ids: Sequence[Hashable], batch_size: int) -> dict[Hashable, list[int]]:
    if len(prompt_ids) != batch_size:
        raise ValueError("prompt_ids must contain one stable group id per rollout")
    groups: dict[Hashable, list[int]] = {}
    for row, prompt_id in enumerate(prompt_ids):
        try:
            hash(prompt_id)
        except TypeError as exc:
            raise TypeError(f"prompt id at row {row} is not hashable") from exc
        groups.setdefault(prompt_id, []).append(row)
    return groups


def _validate_group_sizes(groups: dict[Hashable, list[int]], expected_rollouts: int | None) -> None:
    if expected_rollouts is None:
        return
    if expected_rollouts < 1:
        raise ValueError("expected_rollouts must be positive")
    invalid = {str(key): len(rows) for key, rows in groups.items() if len(rows) != expected_rollouts}
    if invalid:
        raise ValueError(
            f"VA-OPD requires exactly {expected_rollouts} sibling rollouts per prompt; got {invalid}"
        )

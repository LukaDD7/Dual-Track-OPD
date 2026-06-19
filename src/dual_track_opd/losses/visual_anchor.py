"""Visual-anchor token weighting heuristics.

This module contains the first functional proxy for visual-dependence weighting:
select high-scoring token positions and upweight their KL contribution. Later
versions should replace this with VLM-specific visual-dependence metrics such as
image counterfactual sensitivity, teacher disagreement under visual ablation, or
multimodal attribution signals.
"""

from __future__ import annotations

import math

import torch


def build_visual_anchor_weights(
    token_scores: torch.Tensor,
    mask: torch.Tensor | None = None,
    topk_ratio: float = 0.1,
    high_weight: float = 2.0,
    low_weight: float = 1.0,
) -> torch.Tensor:
    """Assign ``high_weight`` to top-k scored valid tokens and ``low_weight`` elsewhere."""

    if token_scores.ndim != 2:
        raise ValueError(
            "token_scores must have shape [batch, seq]; "
            f"got {tuple(token_scores.shape)}"
        )
    if not 0 < topk_ratio <= 1:
        raise ValueError("topk_ratio must be in (0, 1]")

    scores = token_scores.float()
    if mask is None:
        valid = torch.ones_like(scores, dtype=torch.bool)
    else:
        if mask.shape != scores.shape:
            raise ValueError(f"mask must have shape {tuple(scores.shape)}; got {tuple(mask.shape)}")
        valid = mask.bool()

    weights = torch.zeros_like(scores)
    for batch_idx in range(scores.shape[0]):
        valid_idx = torch.nonzero(valid[batch_idx], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        weights[batch_idx, valid_idx] = low_weight
        k = max(1, math.ceil(valid_idx.numel() * topk_ratio))
        valid_scores = scores[batch_idx, valid_idx]
        top_local = torch.topk(valid_scores, k=k).indices
        top_idx = valid_idx[top_local]
        weights[batch_idx, top_idx] = high_weight
    return weights


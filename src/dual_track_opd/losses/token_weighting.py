"""Utilities for token-level distillation weights."""

from __future__ import annotations

import torch


def normalize_token_weights(
    weights: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize weights so valid positions have mean weight 1."""

    if weights.ndim != 2:
        raise ValueError(f"weights must have shape [batch, seq]; got {tuple(weights.shape)}")
    weights = weights.float()

    if mask is None:
        valid = torch.ones_like(weights)
    else:
        if mask.shape != weights.shape:
            raise ValueError(f"mask must have shape {tuple(weights.shape)}; got {tuple(mask.shape)}")
        valid = mask.to(dtype=weights.dtype)

    masked = weights * valid
    total = masked.sum(dim=1, keepdim=True)
    count = valid.sum(dim=1, keepdim=True)
    scale = count / total.clamp_min(eps)
    normalized = masked * scale
    return torch.where(count > 0, normalized, torch.zeros_like(normalized))


def combine_weights(
    *weights: torch.Tensor | None,
    mode: str = "sum",
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine token weights with either sum or product."""

    tensors = [w.float() for w in weights if w is not None]
    if not tensors:
        raise ValueError("at least one weights tensor is required")
    if mode not in {"sum", "product"}:
        raise ValueError("mode must be 'sum' or 'product'")

    shape = tensors[0].shape
    if len(shape) != 2:
        raise ValueError(f"weights must have shape [batch, seq]; got {tuple(shape)}")
    for tensor in tensors[1:]:
        if tensor.shape != shape:
            raise ValueError(f"all weights must share shape {tuple(shape)}; got {tuple(tensor.shape)}")

    if mode == "sum":
        combined = torch.zeros_like(tensors[0])
        for tensor in tensors:
            combined = combined + tensor
    else:
        combined = torch.ones_like(tensors[0])
        for tensor in tensors:
            combined = combined * tensor

    if mask is not None:
        if mask.shape != shape:
            raise ValueError(f"mask must have shape {tuple(shape)}; got {tuple(mask.shape)}")
        combined = combined * mask.to(dtype=combined.dtype)
    return combined


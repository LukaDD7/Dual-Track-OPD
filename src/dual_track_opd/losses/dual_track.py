"""Composable dual-track OPD loss."""

from __future__ import annotations

import torch

from .opd_kl import masked_weighted_kl
from .token_weighting import combine_weights


def dual_track_opd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    text_weights: torch.Tensor | None = None,
    vision_weights: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    alpha_text: float = 1.0,
    alpha_vision: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Combine text and vision weighting signals and compute masked KL."""

    token_shape = student_logits.shape[:2]
    weighted_terms: list[torch.Tensor] = []

    if text_weights is not None:
        if text_weights.shape != token_shape:
            raise ValueError(f"text_weights must have shape {token_shape}; got {tuple(text_weights.shape)}")
        weighted_terms.append(text_weights.float() * alpha_text)
    if vision_weights is not None:
        if vision_weights.shape != token_shape:
            raise ValueError(
                f"vision_weights must have shape {token_shape}; got {tuple(vision_weights.shape)}"
            )
        weighted_terms.append(vision_weights.float() * alpha_vision)

    combined = None
    if weighted_terms:
        combined = combine_weights(*weighted_terms, mode="sum", mask=mask)

    return masked_weighted_kl(
        student_logits,
        teacher_logits,
        mask=mask,
        weights=combined,
        temperature=temperature,
        normalize_weights=True,
    )


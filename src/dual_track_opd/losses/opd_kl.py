"""Token-level KL utilities for on-policy distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> None:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have the same shape; "
            f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    if student_logits.ndim != 3:
        raise ValueError(
            "logits must have shape [batch, seq, vocab]; "
            f"got {tuple(student_logits.shape)}"
        )


def _validate_token_tensor(name: str, value: torch.Tensor, target_shape: tuple[int, int]) -> None:
    if value.shape != target_shape:
        raise ValueError(f"{name} must have shape {target_shape}; got {tuple(value.shape)}")


def kl_divergence_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "none",
) -> torch.Tensor:
    """Compute teacher-to-student KL per token from logits.

    Inputs must have shape ``[batch, seq, vocab]``. With ``reduction="none"``,
    the result has shape ``[batch, seq]``.
    """

    _validate_logits(student_logits, teacher_logits)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    kl = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1)
    kl = kl * (temperature**2)

    if reduction == "none":
        return kl
    if reduction == "mean":
        return kl.mean()
    return kl.sum()


def masked_weighted_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    temperature: float = 1.0,
    normalize_weights: bool = True,
) -> torch.Tensor:
    """Return a scalar masked, weighted token-level KL loss."""

    _validate_logits(student_logits, teacher_logits)
    token_shape = student_logits.shape[:2]

    kl = kl_divergence_from_logits(
        student_logits,
        teacher_logits,
        temperature=temperature,
        reduction="none",
    )

    effective = torch.ones_like(kl)
    if mask is not None:
        _validate_token_tensor("mask", mask, token_shape)
        effective = effective * mask.to(dtype=kl.dtype)
    if weights is not None:
        _validate_token_tensor("weights", weights, token_shape)
        effective = effective * weights.to(dtype=kl.dtype)

    weighted_sum = torch.sum(kl * effective)
    if normalize_weights:
        denom = torch.sum(effective).clamp_min(torch.finfo(kl.dtype).eps)
    elif mask is not None:
        denom = mask.to(dtype=kl.dtype).sum().clamp_min(torch.finfo(kl.dtype).eps)
    else:
        denom = torch.tensor(kl.numel(), device=kl.device, dtype=kl.dtype)
    return weighted_sum / denom


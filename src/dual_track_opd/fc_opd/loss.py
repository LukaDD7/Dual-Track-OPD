"""Sparse top-k distillation loss for failure-calibrated OPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .conditions import Condition
from .signal_decomposer import TeacherTopK


@dataclass(frozen=True)
class FCOPDLossConfig:
    renormalize_topk: bool = True
    include_tail: bool = True
    reduction: str = "mean"
    eps: float = 1e-8
    alignment_strategy: str = "none"
    token_weights: torch.Tensor | None = None
    chunk_weights: Mapping[str, float] | None = None
    condition_pair_weights: Mapping[str, float] | None = None


def _validate_student_and_teacher(student_logits: torch.Tensor, teacher: TeacherTopK) -> None:
    teacher.validate()
    if student_logits.ndim != 3:
        raise ValueError("student_logits must have shape [batch, seq, vocab]")
    if not torch.isfinite(student_logits).all():
        raise ValueError("student_logits contains NaN or Inf")
    if student_logits.shape[:2] != teacher.token_ids.shape[:2]:
        raise ValueError("student logits and teacher scores must share batch and sequence dimensions")
    if torch.any(teacher.token_ids < 0) or torch.any(teacher.token_ids >= student_logits.shape[-1]):
        raise ValueError("teacher token ID is outside the student vocabulary")


def _coalesce_teacher(
    token_ids: torch.Tensor,
    log_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    unique_ids, inverse = torch.unique(token_ids, sorted=False, return_inverse=True)
    masses = torch.zeros(unique_ids.shape, dtype=log_probs.dtype, device=log_probs.device)
    masses.scatter_add_(0, inverse, log_probs.exp())
    return unique_ids, masses


def sparse_forward_kl(
    student_logits: torch.Tensor,
    teacher: TeacherTopK,
    *,
    config: FCOPDLossConfig | None = None,
) -> torch.Tensor:
    """Compute response-token KL over teacher top-k support and optional tail.

    Returns a tensor with shape ``[batch, seq]``. Duplicate teacher token IDs are
    coalesced before the KL is evaluated.
    """

    config = config or FCOPDLossConfig()
    _validate_student_and_teacher(student_logits, teacher)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    losses = torch.zeros(student_logits.shape[:2], dtype=torch.float32, device=student_logits.device)

    for batch_index in range(losses.shape[0]):
        for token_index in range(losses.shape[1]):
            ids, topk_mass = _coalesce_teacher(
                teacher.token_ids[batch_index, token_index],
                teacher.log_probs[batch_index, token_index],
            )
            tail_mass = (
                teacher.tail_log_prob[batch_index, token_index].exp()
                if config.include_tail and teacher.tail_log_prob is not None
                else torch.zeros((), dtype=topk_mass.dtype, device=topk_mass.device)
            )
            total_mass = topk_mass.sum() + tail_mass
            if total_mass <= 0 or not torch.isfinite(total_mass):
                raise ValueError("teacher probability mass must be finite and positive")

            if config.renormalize_topk or tail_mass > 0:
                topk_mass = topk_mass / total_mass
                tail_mass = tail_mass / total_mass

            teacher_log_mass = topk_mass.clamp_min(config.eps).log()
            student_selected_log_probs = student_log_probs[batch_index, token_index, ids]
            # Clamp student log-probs to prevent single-token explosion when the
            # student assigns near-zero mass to a teacher top-k token.  A floor
            # of -15 nats (≈ 3e-7 probability) bounds the per-token KL at ~15.
            student_selected_log_probs = student_selected_log_probs.clamp_min(-15.0)
            token_loss = torch.sum(topk_mass * (teacher_log_mass - student_selected_log_probs))

            if tail_mass > 0:
                selected_student_mass = student_selected_log_probs.exp().sum()
                student_tail_mass = (1.0 - selected_student_mass).clamp_min(config.eps)
                student_tail_log = student_tail_mass.log().clamp_min(-15.0)
                token_loss = token_loss + tail_mass * (tail_mass.clamp_min(config.eps).log() - student_tail_log)
            losses[batch_index, token_index] = token_loss
    return losses


def compute_fc_opd_loss(
    student_logits: torch.Tensor,
    teacher_scores: Mapping[Condition | str, TeacherTopK],
    chunk_masks: Mapping[str, torch.Tensor],
    condition_weights: Mapping[Condition | str, torch.Tensor],
    response_mask: torch.Tensor,
    config: FCOPDLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Aggregate sparse per-condition KL using detached top-1 router weights."""

    config = config or FCOPDLossConfig()
    if response_mask.shape != student_logits.shape[:2]:
        raise ValueError("response_mask must match student batch and sequence dimensions")
    for chunk_name, chunk_mask in chunk_masks.items():
        if chunk_mask.shape != response_mask.shape:
            raise ValueError(f"{chunk_name} mask must have shape {tuple(response_mask.shape)}")

    normalized_scores = {Condition(key): value for key, value in teacher_scores.items()}
    normalized_weights = {Condition(key): value for key, value in condition_weights.items()}
    total_per_token = torch.zeros_like(response_mask, dtype=torch.float32)
    metrics: dict[str, torch.Tensor] = {}

    for condition, weights in normalized_weights.items():
        if weights.shape != response_mask.shape:
            raise ValueError(f"weights for {condition} must have shape {tuple(response_mask.shape)}")
        if torch.any(weights < 0):
            raise ValueError("condition weights must be non-negative")
        if condition not in normalized_scores:
            if torch.any(weights > 0):
                raise ValueError(f"missing teacher scores for selected condition: {condition}")
            continue
        per_token = sparse_forward_kl(student_logits, normalized_scores[condition], config=config)
        selected = weights.detach().float() * response_mask.float()
        total_per_token = total_per_token + per_token * selected
        denominator = selected.sum().clamp_min(1.0)
        metrics[f"loss/{condition.value}"] = (per_token * selected).sum() / denominator
        metrics[f"selection/{condition.value}"] = selected.sum() / response_mask.float().sum().clamp_min(1.0)

    combined_weight = sum(weight.detach().float() for weight in normalized_weights.values()) * response_mask.float()
    if torch.any(combined_weight > 1.0 + 1e-6):
        raise ValueError("condition weights sum to more than one")

    if config.reduction == "none":
        loss = total_per_token
    elif config.reduction == "sum":
        loss = total_per_token.sum()
    elif config.reduction == "mean":
        loss = total_per_token.sum() / combined_weight.sum().clamp_min(1.0)
    else:
        raise ValueError("reduction must be one of: none, mean, sum")
    for chunk_name, chunk_mask in chunk_masks.items():
        selected = chunk_mask.float() * response_mask.float()
        metrics[f"loss/chunk/{chunk_name}"] = (
            (total_per_token * selected).sum() / selected.sum().clamp_min(1.0)
        )
    metrics["loss/total"] = loss.detach() if loss.ndim == 0 else loss.detach().mean()
    return loss, metrics

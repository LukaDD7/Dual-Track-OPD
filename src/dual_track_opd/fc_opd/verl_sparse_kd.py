"""Tensor helpers for the thin verl FC-OPD actor patch.

The functions in this module intentionally operate on plain tensors so the
backend patch can stay small: verl transports the tensors, while this project
owns the sparse-KD math and validation policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VerlSparseKDOutput:
    per_token_loss: torch.Tensor
    active_weight: torch.Tensor
    metrics: dict[str, Any] | None = None


def compute_verl_sparse_topk_kd(
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    condition_weights: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    teacher_tail_log_prob: torch.Tensor | None = None,
    renormalize_topk: bool = True,
    include_tail: bool = True,
    eps: float = 1e-8,
) -> VerlSparseKDOutput:
    """Compute weighted sparse forward KL for verl response logits.

    Args:
        student_logits: Current actor logits with shape ``[B, T, V]``.
        teacher_topk_indices: Teacher token ids with shape ``[B, C, T, K]``.
        teacher_topk_log_probs: Teacher top-k log-probs with shape
            ``[B, C, T, K]``.
        condition_weights: Detached FC-OPD routing weights with shape
            ``[B, C, T]``.
        response_mask: Valid response-token mask with shape ``[B, T]``.

    Returns:
        ``per_token_loss`` is already weighted by condition weights and has
        shape ``[B, T]``. Divide its sum by ``active_weight.sum()`` to obtain
        the batch auxiliary loss.
    """

    _validate_shapes(
        student_logits=student_logits,
        teacher_topk_indices=teacher_topk_indices,
        teacher_topk_log_probs=teacher_topk_log_probs,
        condition_weights=condition_weights,
        response_mask=response_mask,
        teacher_tail_log_prob=teacher_tail_log_prob,
    )
    if not torch.isfinite(student_logits).all():
        raise ValueError("student_logits contains NaN or Inf")
    if not torch.isfinite(teacher_topk_log_probs).all():
        raise ValueError("teacher_topk_log_probs contains NaN or Inf")
    if teacher_tail_log_prob is not None and not torch.isfinite(teacher_tail_log_prob).all():
        raise ValueError("teacher_tail_log_prob contains NaN or Inf")
    if torch.any(condition_weights < 0):
        raise ValueError("condition_weights must be non-negative")
    vocab_size = student_logits.shape[-1]
    if torch.any(teacher_topk_indices < 0) or torch.any(teacher_topk_indices >= vocab_size):
        raise ValueError("teacher_topk_indices contains an id outside the student vocabulary")

    teacher_topk_indices = teacher_topk_indices.long()
    condition_weights = condition_weights.detach().float()
    response_mask_f = response_mask.detach().float()

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    expanded_student = student_log_probs.unsqueeze(1).expand(
        -1,
        teacher_topk_indices.shape[1],
        -1,
        -1,
    )
    student_selected = torch.gather(expanded_student, dim=-1, index=teacher_topk_indices)

    topk_mass = teacher_topk_log_probs.float().exp()
    tail_mass = None
    if include_tail and teacher_tail_log_prob is not None:
        tail_mass = teacher_tail_log_prob.float().exp()
    total_mass = topk_mass.sum(dim=-1)
    if tail_mass is not None:
        total_mass = total_mass + tail_mass
    if torch.any(total_mass <= 0) or not torch.isfinite(total_mass).all():
        raise ValueError("teacher probability mass must be finite and positive")

    if renormalize_topk or tail_mass is not None:
        topk_mass = topk_mass / total_mass.unsqueeze(-1).clamp_min(eps)
        if tail_mass is not None:
            tail_mass = tail_mass / total_mass.clamp_min(eps)

    teacher_log_mass = topk_mass.clamp_min(eps).log()
    # Clamp student log-probs to avoid -inf when mass is near zero.
    _student_sel = student_selected.clamp_min(-15.0)
    per_condition = torch.sum(topk_mass * (teacher_log_mass - _student_sel), dim=-1)

    if tail_mass is not None:
        selected_student_mass = _student_sel.exp().sum(dim=-1)
        student_tail_mass = (1.0 - selected_student_mass).clamp_min(eps)
        per_condition = per_condition + tail_mass * (tail_mass.clamp_min(eps).log() - student_tail_mass.log().clamp_min(-15.0))

    active_weight = condition_weights.sum(dim=1) * response_mask_f
    per_token_loss = torch.sum(per_condition * condition_weights, dim=1) * response_mask_f
    return VerlSparseKDOutput(per_token_loss=per_token_loss, active_weight=active_weight)


def compute_verl_sparse_reverse_kl(
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    condition_weights: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    teacher_tail_log_prob: torch.Tensor | None = None,
    renormalize_topk: bool = True,
    include_tail: bool = True,
    eps: float = 1e-8,
) -> VerlSparseKDOutput:
    """Compute weighted sparse **reverse** KL for verl response logits.

    Reverse KL = KL(P_student || P_teacher) is **mode-seeking**: the student is
    penalised where **it** places probability mass, ignoring teacher modes that
    the student does not cover.  This is preferred for 32B→4B distillation
    because the smaller model cannot cover all teacher modes and attempting to
    do so (forward KL / mode-covering) introduces noise.

    The student distribution is normalised over the teacher's top-K support
    plus a tail bucket so that the KL is well-defined.
    """
    _validate_shapes(
        student_logits=student_logits,
        teacher_topk_indices=teacher_topk_indices,
        teacher_topk_log_probs=teacher_topk_log_probs,
        condition_weights=condition_weights,
        response_mask=response_mask,
        teacher_tail_log_prob=teacher_tail_log_prob,
    )
    if not torch.isfinite(student_logits).all():
        raise ValueError("student_logits contains NaN or Inf")
    if not torch.isfinite(teacher_topk_log_probs).all():
        raise ValueError("teacher_topk_log_probs contains NaN or Inf")
    if teacher_tail_log_prob is not None and not torch.isfinite(teacher_tail_log_prob).all():
        raise ValueError("teacher_tail_log_prob contains NaN or Inf")
    if torch.any(condition_weights < 0):
        raise ValueError("condition_weights must be non-negative")

    teacher_topk_indices = teacher_topk_indices.long()
    condition_weights = condition_weights.detach().float()
    response_mask_f = response_mask.detach().float()

    # ── Student distribution over teacher top-K + tail ──────────────────
    student_probs = torch.softmax(student_logits.float(), dim=-1)  # [B, T, V]
    expanded_student = student_probs.unsqueeze(1).expand(
        -1, teacher_topk_indices.shape[1], -1, -1
    )
    student_topk_probs = torch.gather(expanded_student, dim=-1, index=teacher_topk_indices)  # [B,C,T,K]
    student_topk_sum = student_topk_probs.sum(dim=-1)  # [B,C,T]
    student_tail = (1.0 - student_topk_sum).clamp_min(eps)  # [B,C,T]
    student_total = student_topk_sum + student_tail  # [B,C,T]
    student_topk_norm = student_topk_probs / student_total.unsqueeze(-1).clamp_min(eps)
    student_tail_norm = student_tail / student_total.clamp_min(eps)

    # ── Teacher distribution (same normalisation as forward KL) ─────────
    topk_mass = teacher_topk_log_probs.float().exp()
    tail_mass = None
    if include_tail and teacher_tail_log_prob is not None:
        tail_mass = teacher_tail_log_prob.float().exp()
    total_mass = topk_mass.sum(dim=-1)
    if tail_mass is not None:
        total_mass = total_mass + tail_mass
    if torch.any(total_mass <= 0) or not torch.isfinite(total_mass).all():
        raise ValueError("teacher probability mass must be finite and positive")
    teacher_topk_norm = topk_mass / total_mass.unsqueeze(-1).clamp_min(eps)
    teacher_log_norm = teacher_topk_norm.clamp_min(eps).log()
    if tail_mass is not None:
        teacher_tail_norm = tail_mass / total_mass.clamp_min(eps)
        teacher_tail_log = teacher_tail_norm.clamp_min(eps).log()

    # ── Reverse KL: sum(P_s * (log P_s - log P_t)) ─────────────────────
    # Clamp student log-probs to prevent explosion when student places
    # near-zero mass on a teacher top-k token (or vice versa: high mass
    # on a token the teacher assigns near-0 probability).  A floor of
    # -15 nats bounds the per-token contribution to ~15 nats.
    _student_log = student_topk_norm.clamp_min(eps).log().clamp_min(-15.0)
    _log_ratio = _student_log - teacher_log_norm
    per_condition = torch.sum(student_topk_norm * _log_ratio, dim=-1)
    if tail_mass is not None:
        _student_tail_log = student_tail_norm.clamp_min(eps).log().clamp_min(-15.0)
        per_condition = per_condition + student_tail_norm * (_student_tail_log - teacher_tail_log)

    active_weight = condition_weights.sum(dim=1) * response_mask_f
    per_token_loss = torch.sum(per_condition * condition_weights, dim=1) * response_mask_f
    return VerlSparseKDOutput(per_token_loss=per_token_loss, active_weight=active_weight)


def _validate_shapes(
    *,
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    condition_weights: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_tail_log_prob: torch.Tensor | None,
) -> None:
    if student_logits.ndim != 3:
        raise ValueError("student_logits must have shape [B, T, V]")
    if teacher_topk_indices.ndim != 4:
        raise ValueError("teacher_topk_indices must have shape [B, C, T, K]")
    if teacher_topk_log_probs.shape != teacher_topk_indices.shape:
        raise ValueError("teacher_topk_log_probs must match teacher_topk_indices")
    batch, conditions, seq_len, _ = teacher_topk_indices.shape
    if student_logits.shape[:2] != (batch, seq_len):
        raise ValueError("student logits must match teacher batch and response length")
    if condition_weights.shape != (batch, conditions, seq_len):
        raise ValueError("condition_weights must have shape [B, C, T]")
    if response_mask.shape != (batch, seq_len):
        raise ValueError("response_mask must have shape [B, T]")
    if teacher_tail_log_prob is not None and teacher_tail_log_prob.shape != (batch, conditions, seq_len):
        raise ValueError("teacher_tail_log_prob must have shape [B, C, T]")

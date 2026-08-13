"""Token-level STP-OPD regional objective (CPU-testable, autograd-safe).

Implements the plan's §4.4 / handoff §4.1 contract:

    L = lambda_P * L_prefix_FKL
      + lambda_D * L_suffix_teacher_RKL_K1
      + lambda_R * L_suffix_task_GRPO

- ``L_prefix_FKL`` is the teacher-forced correct-prefix token CE, normalized by
  the prefix's own valid-token count;
- ``L_suffix_teacher_RKL_K1`` is the sampled-token reverse KL between the
  online student and the teacher scored on the same suffix state;
- ``L_suffix_task_GRPO`` applies task advantages only on the student-sampled
  suffix; fixed teacher-prefix tokens are excluded by the suffix mask and never
  enter the likelihood ratio;
- regions are disjoint, each normalized independently, empty regions contribute
  exactly zero gradient (never NaN), and all math returns tensors.

No model loading or training loop here; feed logits/ids/advantages directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .prefix_scaffold import masked_mean


def _log_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1)


def prefix_fkl_ce(
    student_logits: torch.Tensor,
    prefix_ids: torch.Tensor,
    prefix_mask: torch.Tensor,
) -> torch.Tensor:
    """Teacher-forced correct-prefix token CE (FKL region), masked mean."""

    if student_logits.ndim != 3:
        raise ValueError(f"student_logits must be [batch, seq, vocab]; got {tuple(student_logits.shape)}")
    token_nll = -_log_probs(student_logits).gather(
        -1, prefix_ids.unsqueeze(-1)
    ).squeeze(-1)
    return masked_mean(token_nll, prefix_mask)


def suffix_rkl_k1(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    suffix_mask: torch.Tensor,
) -> torch.Tensor:
    """Sampled-token reverse KL: KL(student || teacher) on the suffix region."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student {tuple(student_logits.shape)} != teacher {tuple(teacher_logits.shape)}"
        )
    student_logp = _log_probs(student_logits)
    teacher_logp = _log_probs(teacher_logits)
    rkl = torch.sum(
        student_logp.exp() * (student_logp - teacher_logp),
        dim=-1,
    )
    return masked_mean(rkl, suffix_mask)


def suffix_task_grpo(
    student_logits: torch.Tensor,
    sampled_ids: torch.Tensor,
    advantages: torch.Tensor,
    suffix_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood x advantage on the student-sampled suffix."""

    if student_logits.ndim != 3:
        raise ValueError(f"student_logits must be [batch, seq, vocab]; got {tuple(student_logits.shape)}")
    token_logp = _log_probs(student_logits).gather(
        -1, sampled_ids.unsqueeze(-1)
    ).squeeze(-1)
    if token_logp.shape != advantages.shape:
        raise ValueError(
            f"advantages {tuple(advantages.shape)} must match token grid {tuple(token_logp.shape)}"
        )
    return masked_mean(-token_logp * advantages, suffix_mask)


def stp_opd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    prefix_ids: torch.Tensor,
    sampled_ids: torch.Tensor,
    advantages: torch.Tensor,
    prefix_mask: torch.Tensor,
    suffix_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    lambda_prefix: float = 1.0,
    lambda_distill: float = 1.0,
    lambda_task: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compose the three regional terms with disjoint, independently
    normalized masks.  ``valid_mask`` optionally excludes misaligned
    teacher tokens from every region."""

    effective_prefix = prefix_mask
    effective_suffix = suffix_mask
    if valid_mask is not None:
        if valid_mask.shape != prefix_mask.shape:
            raise ValueError("valid_mask must match mask shape")
        effective_prefix = prefix_mask & valid_mask
        effective_suffix = suffix_mask & valid_mask
    prefix_term = prefix_fkl_ce(student_logits, prefix_ids, effective_prefix)
    distill_term = suffix_rkl_k1(student_logits, teacher_logits, effective_suffix)
    task_term = suffix_task_grpo(student_logits, sampled_ids, advantages, effective_suffix)
    total = (
        lambda_prefix * prefix_term
        + lambda_distill * distill_term
        + lambda_task * task_term
    )
    return total, {
        "prefix_fkl": prefix_term,
        "suffix_rkl": distill_term,
        "suffix_pg": task_term,
    }

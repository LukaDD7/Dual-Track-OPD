"""Actor-side FC/VA-OPD loss helpers for the thin verl patch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .conditions import Condition
from .signal_decomposer import TeacherTopK
from .va_opd_loss import compute_rollout_va_weights, compute_va_opd_loss
from .verl_sparse_kd import (
    VerlSparseKDOutput,
    compute_verl_sparse_reverse_kl,
    compute_verl_sparse_topk_kd,
)


def has_fc_opd_tensors(batch: Mapping[str, Any]) -> bool:
    """Return true when a verl micro-batch carries FC/VA-OPD teacher tensors."""
    required = ("fc_teacher_topk_indices", "fc_teacher_topk_log_probs", "fc_condition_weights")
    present = [key in batch for key in required]
    if any(present) and not all(present):
        missing = [key for key, is_present in zip(required, present, strict=True) if not is_present]
        raise RuntimeError(f"incomplete FC-OPD tensor set; missing: {missing}")
    return all(present)


def fc_opd_batch_denominator(batch: Mapping[str, Any], response_mask: torch.Tensor) -> torch.Tensor:
    """Global normalizer for one actor mini-batch.

    VA-OPD rollout weights sum to one per prompt sibling group, so their sum is
    the number of prompts represented in the mini-batch.  Legacy FC-OPD uses
    token/condition active weights.
    """
    if "fc_rollout_weights" in batch:
        return batch["fc_rollout_weights"].float().sum().clamp_min(1.0)
    condition_weights = batch["fc_condition_weights"].float()
    active = condition_weights.sum(dim=1) * response_mask.float()
    return active.sum().clamp_min(1.0)


def compute_verl_fc_opd_actor_loss(
    *,
    student_logits: torch.Tensor,
    batch: Mapping[str, Any],
    response_mask: torch.Tensor,
    config: Any,
) -> VerlSparseKDOutput:
    """Compute the actor auxiliary distillation loss payload.

    For ``loss_mode=va_opd`` this is the paper-faithful VA-OPD grouped reverse
    KL over full-image teacher scores, reweighted by full-vs-degraded VA.
    Other modes preserve the older FC-OPD sparse KD paths.
    """
    fc_config = _fc_config(config)
    mode = _loss_mode(batch, fc_config)
    if mode == "va_opd":
        return _compute_va_opd_actor_loss(
            student_logits=student_logits,
            batch=batch,
            response_mask=response_mask,
            fc_config=fc_config,
        )

    compute_kd = compute_verl_sparse_reverse_kl if mode == "reverse" else compute_verl_sparse_topk_kd
    return compute_kd(
        student_logits=student_logits,
        teacher_topk_indices=batch["fc_teacher_topk_indices"],
        teacher_topk_log_probs=batch["fc_teacher_topk_log_probs"],
        teacher_tail_log_prob=batch.get("fc_teacher_tail_log_prob", None),
        condition_weights=batch["fc_condition_weights"],
        response_mask=response_mask,
        renormalize_topk=bool(fc_config.get("renormalize_topk", True)),
        include_tail=bool(fc_config.get("include_tail", True)),
        eps=float(fc_config.get("eps", 1e-8)),
    )


def _compute_va_opd_actor_loss(
    *,
    student_logits: torch.Tensor,
    batch: Mapping[str, Any],
    response_mask: torch.Tensor,
    fc_config: Mapping[str, Any],
) -> VerlSparseKDOutput:
    sampled_log_probs = batch.get("fc_teacher_sampled_log_probs", None)
    if sampled_log_probs is None:
        raise RuntimeError("VA-OPD requires fc_teacher_sampled_log_probs for exact token VA")
    full_idx = _condition_position(batch, Condition.FULL)
    degraded_idx = _condition_position(batch, Condition.DEGRADED)

    tail = batch.get("fc_teacher_tail_log_prob", None)
    teacher_valid = batch.get("fc_teacher_valid_mask", None)
    teacher_full = TeacherTopK(
        token_ids=batch["fc_teacher_topk_indices"][:, full_idx],
        log_probs=batch["fc_teacher_topk_log_probs"][:, full_idx],
        tail_log_prob=None if tail is None else tail[:, full_idx],
        sampled_log_probs=sampled_log_probs[:, full_idx],
        valid_mask=None if teacher_valid is None else teacher_valid[:, full_idx],
    )
    teacher_degraded = TeacherTopK(
        token_ids=batch["fc_teacher_topk_indices"][:, degraded_idx],
        log_probs=batch["fc_teacher_topk_log_probs"][:, degraded_idx],
        tail_log_prob=None if tail is None else tail[:, degraded_idx],
        sampled_log_probs=sampled_log_probs[:, degraded_idx],
        valid_mask=None if teacher_valid is None else teacher_valid[:, degraded_idx],
    )

    rollout_weights = batch.get("fc_rollout_weights", None)
    prompt_ids = _prompt_ids(batch) if rollout_weights is None else None
    if rollout_weights is None:
        va_pos = (sampled_log_probs[:, full_idx] - sampled_log_probs[:, degraded_idx]).clamp_min(0.0)
        teacher_mask = _teacher_pair_valid_mask(teacher_full, teacher_degraded, response_mask)
        rollout_weights = compute_rollout_va_weights(
            va_pos,
            response_mask=response_mask.bool() & teacher_mask,
            prompt_ids=prompt_ids,
            tau=float(fc_config.get("tau_rollout", 1.0)),
        )

    result = compute_va_opd_loss(
        student_logits=student_logits,
        teacher_full=teacher_full,
        teacher_degraded=teacher_degraded,
        sampled_token_ids=batch["responses"],
        response_mask=response_mask,
        prompt_ids=prompt_ids,
        rollout_weights=rollout_weights,
        top_q=float(fc_config.get("va_top_q", 0.20)),
        tau_rollout=float(fc_config.get("tau_rollout", 1.0)),
        lambda_high=float(fc_config.get("va_lambda", 0.50)),
        renormalize_topk=bool(fc_config.get("renormalize_topk", True)),
        include_tail=bool(fc_config.get("include_tail", True)),
        eps=float(fc_config.get("eps", 1e-8)),
    )
    return VerlSparseKDOutput(
        per_token_loss=_put_row_values_on_first_valid_token(result.per_rollout_loss, response_mask),
        active_weight=_put_row_values_on_first_valid_token(result.rollout_weights.detach(), response_mask),
        metrics=result.metrics,
    )


def _put_row_values_on_first_valid_token(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    carrier = torch.zeros_like(response_mask, dtype=torch.float32)
    valid = response_mask.bool()
    if valid.numel() == 0:
        return carrier
    first = valid.float().argmax(dim=1)
    rows = torch.arange(valid.shape[0], device=valid.device)
    has_valid = valid.any(dim=1)
    carrier[rows[has_valid], first[has_valid]] = values.to(carrier.device, dtype=torch.float32)[has_valid]
    return carrier


def _teacher_pair_valid_mask(
    teacher_full: TeacherTopK,
    teacher_degraded: TeacherTopK,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    mask = torch.ones_like(response_mask, dtype=torch.bool)
    if teacher_full.valid_mask is not None:
        mask = mask & teacher_full.valid_mask.to(device=response_mask.device, dtype=torch.bool)
    if teacher_degraded.valid_mask is not None:
        mask = mask & teacher_degraded.valid_mask.to(device=response_mask.device, dtype=torch.bool)
    return mask


def _condition_position(batch: Mapping[str, Any], condition: Condition) -> int:
    ids = batch.get("fc_condition_ids", None)
    if ids is None:
        fallback = {Condition.FULL: 0, Condition.DEGRADED: 1}
        return fallback[condition]
    ids_tensor = torch.as_tensor(ids, device=batch["fc_teacher_topk_indices"].device)
    if ids_tensor.ndim == 2:
        ids_tensor = ids_tensor[0]
    target = 0 if condition == Condition.FULL else 1
    matches = torch.nonzero(ids_tensor.long() == target, as_tuple=False).flatten()
    if matches.numel() == 0:
        raise RuntimeError(f"VA-OPD condition set is missing {condition.value}")
    return int(matches[0].item())


def _loss_mode(batch: Mapping[str, Any], fc_config: Mapping[str, Any]) -> str:
    raw = batch.get("fc_opd_loss_mode", fc_config.get("loss_mode", "forward"))
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().flatten()[0].item()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        raw = raw[0]
    elif not isinstance(raw, (str, bytes)) and hasattr(raw, "__len__"):
        raw = raw[0]
    return str(raw)


def _prompt_ids(batch: Mapping[str, Any]) -> Sequence[int | str] | None:
    raw = batch.get("fc_prompt_ids", None)
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        return [int(item) for item in raw.detach().cpu().flatten().tolist()]
    return list(raw)


def _fc_config(config: Any) -> Mapping[str, Any]:
    getter = getattr(config, "get", None)
    fc_config = getter("fc_opd", {}) if callable(getter) else getattr(config, "fc_opd", {})
    return fc_config or {}

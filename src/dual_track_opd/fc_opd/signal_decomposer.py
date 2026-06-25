"""Top-k teacher distribution validation and condition-signal decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .conditions import Condition


@dataclass(frozen=True)
class TeacherTopK:
    token_ids: torch.Tensor
    log_probs: torch.Tensor
    tail_log_prob: torch.Tensor | None = None
    entropy: torch.Tensor | None = None

    def validate(self) -> None:
        if self.token_ids.ndim != 3:
            raise ValueError("token_ids must have shape [batch, seq, topk]")
        if self.log_probs.shape != self.token_ids.shape:
            raise ValueError("log_probs must have the same shape as token_ids")
        if self.token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must use an integer dtype")
        if self.token_ids.device != self.log_probs.device:
            raise ValueError("token_ids and log_probs must be on the same device")
        if not torch.isfinite(self.log_probs).all():
            raise ValueError("log_probs contains NaN or Inf")
        if self.tail_log_prob is not None:
            if self.tail_log_prob.shape != self.token_ids.shape[:2]:
                raise ValueError("tail_log_prob must have shape [batch, seq]")
            if not torch.isfinite(self.tail_log_prob).all():
                raise ValueError("tail_log_prob contains NaN or Inf")
        if self.entropy is not None:
            if self.entropy.shape != self.token_ids.shape[:2]:
                raise ValueError("entropy must have shape [batch, seq]")
            if not torch.isfinite(self.entropy).all():
                raise ValueError("entropy contains NaN or Inf")


def _coalesced_distribution(
    token_ids: torch.Tensor,
    log_probs: torch.Tensor,
    tail_log_prob: torch.Tensor | None,
) -> dict[int | str, torch.Tensor]:
    masses: dict[int | str, torch.Tensor] = {}
    for token_id, log_prob in zip(token_ids.tolist(), log_probs, strict=True):
        key = int(token_id)
        mass = log_prob.exp()
        masses[key] = masses.get(key, torch.zeros_like(mass)) + mass
    if tail_log_prob is not None:
        masses["__tail__"] = tail_log_prob.exp()
    total = torch.stack(list(masses.values())).sum().clamp_min(torch.finfo(log_probs.dtype).tiny)
    return {key: value / total for key, value in masses.items()}


def jensen_shannon_topk(left: TeacherTopK, right: TeacherTopK) -> torch.Tensor:
    """Compute token-wise JSD over the union of two top-k supports and a tail bucket."""

    left.validate()
    right.validate()
    if left.token_ids.shape[:2] != right.token_ids.shape[:2]:
        raise ValueError("left and right must share batch and sequence dimensions")

    result = torch.zeros(left.token_ids.shape[:2], dtype=torch.float32, device=left.log_probs.device)
    for batch_index in range(result.shape[0]):
        for token_index in range(result.shape[1]):
            left_tail = None if left.tail_log_prob is None else left.tail_log_prob[batch_index, token_index]
            right_tail = None if right.tail_log_prob is None else right.tail_log_prob[batch_index, token_index]
            p = _coalesced_distribution(
                left.token_ids[batch_index, token_index],
                left.log_probs[batch_index, token_index],
                left_tail,
            )
            q = _coalesced_distribution(
                right.token_ids[batch_index, token_index],
                right.log_probs[batch_index, token_index],
                right_tail,
            )
            keys = p.keys() | q.keys()
            js = torch.zeros((), dtype=torch.float32, device=result.device)
            for key in keys:
                p_value = p.get(key, torch.zeros((), device=result.device))
                q_value = q.get(key, torch.zeros((), device=result.device))
                midpoint = 0.5 * (p_value + q_value)
                if p_value > 0:
                    js = js + 0.5 * p_value * (p_value.log() - midpoint.log())
                if q_value > 0:
                    js = js + 0.5 * q_value * (q_value.log() - midpoint.log())
            result[batch_index, token_index] = js
    return result


def sampled_token_log_prob(scores: TeacherTopK, sampled_token_ids: torch.Tensor) -> torch.Tensor:
    """Gather sampled-token log-probability, using tail mass when absent from top-k."""

    scores.validate()
    if sampled_token_ids.shape != scores.token_ids.shape[:2]:
        raise ValueError("sampled_token_ids must have shape [batch, seq]")
    matches = scores.token_ids == sampled_token_ids.unsqueeze(-1)
    found = matches.any(dim=-1)
    gathered = scores.log_probs.masked_fill(~matches, -torch.inf).amax(dim=-1)
    if scores.tail_log_prob is None and not found.all():
        raise ValueError("sampled token missing from top-k and no tail_log_prob was provided")
    if scores.tail_log_prob is not None:
        gathered = torch.where(found, gathered, scores.tail_log_prob)
    return gathered


def compute_condition_signals(
    teacher_scores: Mapping[Condition | str, TeacherTopK],
    sampled_token_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the descriptive FC-OPD condition signals available in a score set."""

    normalized = {Condition(key): value for key, value in teacher_scores.items()}
    signals: dict[str, torch.Tensor] = {}

    pairs = (
        ("visual_detail", Condition.FULL, Condition.BLUR),
        ("task_extraction", Condition.TASK, Condition.FREE),
        ("fact_gap", Condition.FACT, Condition.TASK),
    )
    for name, left, right in pairs:
        if left in normalized and right in normalized:
            signals[name] = jensen_shannon_topk(normalized[left], normalized[right])
            if sampled_token_ids is not None:
                signals[f"{name}_sampled_logprob_delta"] = sampled_token_log_prob(
                    normalized[left], sampled_token_ids
                ) - sampled_token_log_prob(normalized[right], sampled_token_ids)
    return signals

"""Grouped KL weighting for Visual Grounding Gap OPD."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from .chunk_parser import CANONICAL_CHUNK_NAMES
from .hidden_state_visual_focus import compute_teacher_student_visual_gap, rank_normalize_scores
from .loss import FCOPDLossConfig, sparse_forward_kl
from .signal_decomposer import TeacherTopK, sampled_token_log_prob


DEFAULT_VERIFIER_CHUNK_GATES: dict[str, dict[str, float]] = {
    "correct": {
        "visible_evidence": 0.25,
        "diagram_inference": 0.25,
        "reasoning": 0.10,
        "answer": 0.00,
    },
    "wrong_but_format_valid": {
        "visible_evidence": 1.00,
        "diagram_inference": 1.00,
        "reasoning": 0.75,
        "answer": 0.50,
    },
    "malformed": {
        "visible_evidence": 0.25,
        "diagram_inference": 0.10,
        "reasoning": 0.00,
        "answer": 0.00,
    },
    "unknown": {
        "visible_evidence": 1.00,
        "diagram_inference": 1.00,
        "reasoning": 1.00,
        "answer": 1.00,
    },
}


@dataclass(frozen=True)
class VisualGroundingGapLossConfig:
    top_q: float = 0.20
    tau_rollout: float = 1.0
    tau_chunk: float = 0.15
    min_high_tokens: int = 1
    alpha_low: float = 0.25
    lambda_gap: float = 1.0
    include_tail: bool = True
    renormalize_topk: bool = True
    eps: float = 1e-8
    chunk_names: tuple[str, ...] = CANONICAL_CHUNK_NAMES
    verifier_chunk_gates: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: DEFAULT_VERIFIER_CHUNK_GATES
    )

    def __post_init__(self) -> None:
        if not 0 < self.top_q <= 1:
            raise ValueError("top_q must be in (0, 1]")
        if self.tau_rollout <= 0 or self.tau_chunk <= 0:
            raise ValueError("temperature values must be positive")
        if self.min_high_tokens < 1:
            raise ValueError("min_high_tokens must be at least 1")
        if self.alpha_low < 0 or self.lambda_gap < 0:
            raise ValueError("loss weights must be non-negative")


@dataclass(frozen=True)
class ChunkGapGates:
    gate: dict[str, torch.Tensor]
    chunk_gap: dict[str, torch.Tensor]
    chunk_gap_rank: dict[str, torch.Tensor]


@dataclass(frozen=True)
class VisualGroundingGapLossResult:
    loss: torch.Tensor
    token_mean_loss: torch.Tensor
    active_weight_normalized_loss: torch.Tensor
    per_token_kl: torch.Tensor
    va_raw: torch.Tensor
    va_pos: torch.Tensor
    rollout_weights: torch.Tensor
    teacher_vfs_rank: torch.Tensor
    student_vfs_rank: torch.Tensor
    gap_raw: torch.Tensor
    gap_pos: torch.Tensor
    chunk_gap_gates: ChunkGapGates
    metrics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class VisualGroundingGapWeightResult:
    token_weights: torch.Tensor
    high_va_mask: torch.Tensor
    low_va_mask: torch.Tensor
    va_raw: torch.Tensor
    va_pos: torch.Tensor
    rollout_weights: torch.Tensor
    teacher_vfs_rank: torch.Tensor
    student_vfs_rank: torch.Tensor
    gap_raw: torch.Tensor
    gap_pos: torch.Tensor
    chunk_gap_gates: ChunkGapGates


def compute_va_raw(
    teacher_full: TeacherTopK,
    teacher_degraded: TeacherTopK,
    sampled_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Signed teacher visual advantage for sampled rollout tokens."""

    return sampled_token_log_prob(teacher_full, sampled_token_ids) - sampled_token_log_prob(
        teacher_degraded, sampled_token_ids
    )


def compute_rollout_va_weights(
    va_pos: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
    prompt_ids: Sequence[Any] | torch.Tensor | None = None,
    top_q: float = 0.20,
    tau_rollout: float = 1.0,
) -> torch.Tensor:
    """Compute sibling-rollout VA softmax weights that sum to K per prompt."""

    if va_pos.ndim != 2:
        raise ValueError("va_pos must have shape [batch, seq]")
    mask = _mask_or_ones(va_pos, response_mask)
    summaries = torch.tensor(
        [_top_mean(va_pos[i, mask[i]], top_q=top_q) for i in range(va_pos.shape[0])],
        dtype=torch.float32,
        device=va_pos.device,
    )
    groups = _prompt_groups(prompt_ids, va_pos.shape[0])
    weights = torch.ones_like(summaries)
    for _, indices in groups.items():
        idx = torch.tensor(indices, dtype=torch.long, device=va_pos.device)
        group_scores = summaries[idx]
        group_weights = torch.softmax(group_scores / tau_rollout, dim=0) * float(len(indices))
        weights[idx] = group_weights
    return weights


def split_high_low_va_groups(
    va_pos: torch.Tensor,
    chunk_mask: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
    top_q: float = 0.20,
    min_high_tokens: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split each row/chunk into high-VA and low-VA token masks."""

    if va_pos.shape != chunk_mask.shape:
        raise ValueError("va_pos and chunk_mask must have matching shape")
    mask = chunk_mask.bool() & _mask_or_ones(va_pos, response_mask)
    high = torch.zeros_like(mask, dtype=torch.bool)
    for batch_index in range(va_pos.shape[0]):
        idx = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        count = max(min_high_tokens, int(math.ceil(float(idx.numel()) * top_q)))
        count = min(count, int(idx.numel()))
        _, order = torch.topk(va_pos[batch_index, idx], k=count, largest=True, sorted=False)
        high[batch_index, idx[order]] = True
    low = mask & ~high
    return high, low


def compute_chunk_gap_gate(
    gap_pos: torch.Tensor,
    chunk_masks: Mapping[str, torch.Tensor],
    *,
    response_mask: torch.Tensor | None = None,
    top_q: float = 0.20,
    tau_chunk: float = 0.15,
    chunk_names: Sequence[str] = CANONICAL_CHUNK_NAMES,
) -> ChunkGapGates:
    """Aggregate student grounding gaps by chunk and convert ranks to gates."""

    if gap_pos.ndim != 2:
        raise ValueError("gap_pos must have shape [batch, seq]")
    response = _mask_or_ones(gap_pos, response_mask)
    chunk_gap: dict[str, torch.Tensor] = {}
    active = []
    values = []
    for name in chunk_names:
        mask = _chunk_mask(chunk_masks, name, gap_pos).bool() & response
        summary = torch.tensor(
            [_top_mean(gap_pos[i, mask[i]], top_q=top_q) for i in range(gap_pos.shape[0])],
            dtype=torch.float32,
            device=gap_pos.device,
        )
        chunk_gap[name] = summary
        values.append(summary)
        active.append(mask.any(dim=1))
    stacked = torch.stack(values, dim=1)
    active_mask = torch.stack(active, dim=1)
    ranks = rank_normalize_scores(stacked, active_mask)
    gates_tensor = torch.sigmoid((ranks - 0.5) / tau_chunk)
    gates_tensor = torch.where(active_mask, gates_tensor, torch.zeros_like(gates_tensor))
    return ChunkGapGates(
        gate={name: gates_tensor[:, index] for index, name in enumerate(chunk_names)},
        chunk_gap=chunk_gap,
        chunk_gap_rank={name: ranks[:, index] for index, name in enumerate(chunk_names)},
    )


def verifier_outcome_class(verifier: Mapping[str, Any] | str | None) -> str:
    if verifier is None:
        return "unknown"
    if isinstance(verifier, str):
        return verifier if verifier in DEFAULT_VERIFIER_CHUNK_GATES else "unknown"
    if bool(verifier.get("malformed", False)):
        return "malformed"
    if verifier.get("correct") is True:
        return "correct"
    if bool(verifier.get("format_valid", False)):
        return "wrong_but_format_valid"
    return "malformed"


def build_verifier_learning_value_gate(
    verifier: Mapping[str, Any] | str | None,
    *,
    gates_by_outcome: Mapping[str, Mapping[str, float]] = DEFAULT_VERIFIER_CHUNK_GATES,
) -> dict[str, Any]:
    """Return the fixed learning-value gates from the VGG-OPD plan."""

    outcome = verifier_outcome_class(verifier)
    gates = dict(gates_by_outcome.get(outcome, gates_by_outcome["unknown"]))
    return {
        "outcome_class": outcome,
        "gate_policy": "vgg_opd_fixed_v1",
        "chunk_gates": gates,
    }


def compute_visual_grounding_gap_token_weights(
    va_raw: torch.Tensor,
    teacher_vfs: torch.Tensor,
    student_vfs: torch.Tensor,
    chunk_masks: Mapping[str, torch.Tensor],
    *,
    response_mask: torch.Tensor | None = None,
    prompt_ids: Sequence[Any] | torch.Tensor | None = None,
    rollout_weights_override: torch.Tensor | None = None,
    verifier_outcomes: Sequence[Mapping[str, Any] | str | None] | None = None,
    inputs_are_ranked: bool = False,
    config: VisualGroundingGapLossConfig | None = None,
) -> VisualGroundingGapWeightResult:
    """Compute detached VGG-OPD token weights without requiring student logits."""

    config = config or VisualGroundingGapLossConfig()
    if va_raw.shape != teacher_vfs.shape or va_raw.shape != student_vfs.shape:
        raise ValueError("va_raw, teacher_vfs, and student_vfs must share shape")
    response = _mask_or_ones(va_raw, response_mask)
    va_pos = va_raw.clamp_min(0.0)
    if rollout_weights_override is None:
        rollout_weights = compute_rollout_va_weights(
            va_pos,
            response_mask=response,
            prompt_ids=prompt_ids,
            top_q=config.top_q,
            tau_rollout=config.tau_rollout,
        )
    else:
        rollout_weights = rollout_weights_override.to(va_raw.device).float()
        if rollout_weights.shape != (va_raw.shape[0],):
            raise ValueError("rollout_weights_override must have shape [batch]")
    gap = compute_teacher_student_visual_gap(
        teacher_vfs,
        student_vfs,
        mask=response,
        inputs_are_ranked=inputs_are_ranked,
    )
    chunk_gates = compute_chunk_gap_gate(
        gap.gap_pos,
        chunk_masks,
        response_mask=response,
        top_q=config.top_q,
        tau_chunk=config.tau_chunk,
        chunk_names=config.chunk_names,
    )
    verifier_outcomes = verifier_outcomes or [None] * va_raw.shape[0]
    if len(verifier_outcomes) != va_raw.shape[0]:
        raise ValueError("verifier_outcomes must have one entry per batch row")

    token_weights = torch.zeros_like(va_raw, dtype=torch.float32)
    high_all = torch.zeros_like(response, dtype=torch.bool)
    low_all = torch.zeros_like(response, dtype=torch.bool)
    for batch_index in range(va_raw.shape[0]):
        verifier_gate = build_verifier_learning_value_gate(
            verifier_outcomes[batch_index],
            gates_by_outcome=config.verifier_chunk_gates,
        )
        gate_by_chunk = verifier_gate["chunk_gates"]
        for chunk_name in config.chunk_names:
            chunk_mask = _chunk_mask(chunk_masks, chunk_name, va_raw).bool() & response
            row_chunk_mask = chunk_mask[batch_index : batch_index + 1]
            if not row_chunk_mask.any():
                continue
            high, low = split_high_low_va_groups(
                va_pos[batch_index : batch_index + 1],
                row_chunk_mask,
                response_mask=response[batch_index : batch_index + 1],
                top_q=config.top_q,
                min_high_tokens=config.min_high_tokens,
            )
            base_weight = (
                rollout_weights[batch_index].float()
                * float(gate_by_chunk.get(chunk_name, 1.0))
            )
            if base_weight <= 0:
                continue
            alpha_high = 1.0 + config.lambda_gap * chunk_gates.gate[chunk_name][batch_index]
            token_weights[batch_index : batch_index + 1] = torch.where(
                high,
                base_weight * alpha_high,
                token_weights[batch_index : batch_index + 1],
            )
            token_weights[batch_index : batch_index + 1] = torch.where(
                low,
                base_weight * config.alpha_low,
                token_weights[batch_index : batch_index + 1],
            )
            high_all[batch_index : batch_index + 1] |= high
            low_all[batch_index : batch_index + 1] |= low
    token_weights = torch.where(response, token_weights, torch.zeros_like(token_weights))
    return VisualGroundingGapWeightResult(
        token_weights=token_weights,
        high_va_mask=high_all,
        low_va_mask=low_all,
        va_raw=va_raw,
        va_pos=va_pos,
        rollout_weights=rollout_weights,
        teacher_vfs_rank=gap.teacher_rank,
        student_vfs_rank=gap.student_rank,
        gap_raw=gap.gap_raw,
        gap_pos=gap.gap_pos,
        chunk_gap_gates=chunk_gates,
    )


def compute_grouped_sparse_kl_loss(
    student_logits: torch.Tensor,
    teacher_full: TeacherTopK,
    teacher_degraded: TeacherTopK,
    sampled_token_ids: torch.Tensor,
    teacher_vfs: torch.Tensor,
    student_vfs: torch.Tensor,
    chunk_masks: Mapping[str, torch.Tensor],
    *,
    response_mask: torch.Tensor | None = None,
    prompt_ids: Sequence[Any] | torch.Tensor | None = None,
    verifier_outcomes: Sequence[Mapping[str, Any] | str | None] | None = None,
    inputs_are_ranked: bool = False,
    config: VisualGroundingGapLossConfig | None = None,
) -> VisualGroundingGapLossResult:
    """Compute VGG-OPD grouped KL against the full-image teacher distribution."""

    config = config or VisualGroundingGapLossConfig()
    response = _mask_or_ones(sampled_token_ids.float(), response_mask)
    per_token_kl = sparse_forward_kl(
        student_logits,
        teacher_full,
        config=FCOPDLossConfig(
            renormalize_topk=config.renormalize_topk,
            include_tail=config.include_tail,
            reduction="none",
            eps=config.eps,
        ),
    )
    va_raw = compute_va_raw(teacher_full, teacher_degraded, sampled_token_ids)
    va_pos = va_raw.clamp_min(0.0)
    rollout_weights = compute_rollout_va_weights(
        va_pos,
        response_mask=response,
        prompt_ids=prompt_ids,
        top_q=config.top_q,
        tau_rollout=config.tau_rollout,
    )
    gap = compute_teacher_student_visual_gap(
        teacher_vfs,
        student_vfs,
        mask=response,
        inputs_are_ranked=inputs_are_ranked,
    )
    chunk_gates = compute_chunk_gap_gate(
        gap.gap_pos,
        chunk_masks,
        response_mask=response,
        top_q=config.top_q,
        tau_chunk=config.tau_chunk,
        chunk_names=config.chunk_names,
    )
    verifier_outcomes = verifier_outcomes or [None] * sampled_token_ids.shape[0]
    if len(verifier_outcomes) != sampled_token_ids.shape[0]:
        raise ValueError("verifier_outcomes must have one entry per batch row")

    numerator = torch.zeros((), dtype=torch.float32, device=student_logits.device)
    denominator = torch.zeros((), dtype=torch.float32, device=student_logits.device)
    weighted_token_sum = torch.zeros((), dtype=torch.float32, device=student_logits.device)
    weighted_token_count = torch.zeros((), dtype=torch.float32, device=student_logits.device)
    metrics: dict[str, torch.Tensor] = {}

    for batch_index in range(sampled_token_ids.shape[0]):
        verifier_gate = build_verifier_learning_value_gate(
            verifier_outcomes[batch_index],
            gates_by_outcome=config.verifier_chunk_gates,
        )
        gate_by_chunk = verifier_gate["chunk_gates"]
        for chunk_name in config.chunk_names:
            chunk_mask = _chunk_mask(chunk_masks, chunk_name, sampled_token_ids.float()).bool() & response
            row_chunk_mask = chunk_mask[batch_index : batch_index + 1]
            if not row_chunk_mask.any():
                continue
            high, low = split_high_low_va_groups(
                va_pos[batch_index : batch_index + 1],
                row_chunk_mask,
                response_mask=response[batch_index : batch_index + 1],
                top_q=config.top_q,
                min_high_tokens=config.min_high_tokens,
            )
            base_weight = (
                rollout_weights[batch_index].float()
                * float(gate_by_chunk.get(chunk_name, 1.0))
            )
            if base_weight <= 0:
                continue
            high_loss = _masked_mean(per_token_kl[batch_index : batch_index + 1], high)
            low_loss = _masked_mean(per_token_kl[batch_index : batch_index + 1], low)
            alpha_high = 1.0 + config.lambda_gap * chunk_gates.gate[chunk_name][batch_index]
            chunk_loss = base_weight * (alpha_high * high_loss + config.alpha_low * low_loss)
            numerator = numerator + chunk_loss
            denominator = denominator + base_weight
            token_weights = torch.zeros_like(row_chunk_mask, dtype=torch.float32)
            token_weights = torch.where(high, base_weight * alpha_high, token_weights)
            token_weights = torch.where(low, base_weight * config.alpha_low, token_weights)
            weighted_token_sum = weighted_token_sum + (per_token_kl[batch_index : batch_index + 1] * token_weights).sum()
            weighted_token_count = weighted_token_count + token_weights.sum()

    active_weight_normalized = numerator / denominator.clamp_min(config.eps)
    token_mean = weighted_token_sum / weighted_token_count.clamp_min(config.eps)
    metrics["loss/active_weight_normalized"] = active_weight_normalized.detach()
    metrics["loss/token_mean"] = token_mean.detach()
    metrics["weight/active_denominator"] = denominator.detach()
    metrics["va/raw_mean"] = _masked_mean(va_raw, response).detach()
    metrics["va/pos_sparsity"] = ((va_pos <= 0) & response).float().sum() / response.float().sum().clamp_min(1.0)
    metrics["gap/pos_mean"] = _masked_mean(gap.gap_pos, response).detach()
    metrics["rollout_weight/mean"] = rollout_weights.mean().detach()

    return VisualGroundingGapLossResult(
        loss=active_weight_normalized,
        token_mean_loss=token_mean,
        active_weight_normalized_loss=active_weight_normalized,
        per_token_kl=per_token_kl,
        va_raw=va_raw,
        va_pos=va_pos,
        rollout_weights=rollout_weights,
        teacher_vfs_rank=gap.teacher_rank,
        student_vfs_rank=gap.student_rank,
        gap_raw=gap.gap_raw,
        gap_pos=gap.gap_pos,
        chunk_gap_gates=chunk_gates,
        metrics=metrics,
    )


def _top_mean(values: torch.Tensor, *, top_q: float) -> float:
    if values.numel() == 0:
        return 0.0
    count = max(1, int(math.ceil(float(values.numel()) * top_q)))
    top = torch.topk(values.float(), k=min(count, int(values.numel())), largest=True).values
    return float(top.mean().item())


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask.bool()]
    if selected.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=values.device)
    return selected.float().mean()


def _mask_or_ones(reference: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.ones(reference.shape[:2], dtype=torch.bool, device=reference.device)
    if mask.shape != reference.shape[:2]:
        raise ValueError("mask shape must match [batch, seq]")
    return mask.to(reference.device).bool()


def _chunk_mask(
    chunk_masks: Mapping[str, torch.Tensor],
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    if name not in chunk_masks:
        return torch.zeros(reference.shape[:2], dtype=torch.bool, device=reference.device)
    mask = chunk_masks[name].to(reference.device)
    if mask.shape != reference.shape[:2]:
        raise ValueError(f"{name} chunk mask shape must match [batch, seq]")
    return mask.bool()


def _prompt_groups(prompt_ids: Sequence[Any] | torch.Tensor | None, batch: int) -> dict[Any, list[int]]:
    if prompt_ids is None:
        labels: list[Any] = ["__all__"] * batch
    elif isinstance(prompt_ids, torch.Tensor):
        if prompt_ids.numel() != batch:
            raise ValueError("prompt_ids must have one item per batch row")
        labels = [int(item) for item in prompt_ids.reshape(-1).tolist()]
    else:
        labels = list(prompt_ids)
        if len(labels) != batch:
            raise ValueError("prompt_ids must have one item per batch row")
    groups: dict[Any, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    return groups

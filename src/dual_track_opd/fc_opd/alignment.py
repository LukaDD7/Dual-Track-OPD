"""No-op-first alignment hooks for future failure-calibrated FC-OPD training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class AlignmentWeights:
    strategy: str
    available: bool
    token_weights: tuple[float, ...] | None = None
    chunk_weights: dict[str, float] | None = None
    condition_pair_weights: dict[str, float] | None = None
    warnings: tuple[str, ...] = ()
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "available": self.available,
            "notes": self.notes,
            "visual_signal_key": "visual_detail",
            "task_signal_key": "task_extraction",
            "outcome_signal_key": "is_correct",
            "token_weights": None if self.token_weights is None else list(self.token_weights),
            "chunk_weights": self.chunk_weights,
            "failure_calibrated_weights": self.condition_pair_weights,
            "warnings": list(self.warnings),
        }


@dataclass
class RolloutGroupStats:
    rollout_group_uid: str
    num_success: int = 0
    num_failure: int = 0
    rollout_ids: list[int] = field(default_factory=list)

    @property
    def has_success_failure_contrast(self) -> bool:
        return self.num_success > 0 and self.num_failure > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_group_uid": self.rollout_group_uid,
            "num_success": self.num_success,
            "num_failure": self.num_failure,
            "has_success_failure_contrast": self.has_success_failure_contrast,
            "rollout_ids": self.rollout_ids,
        }


def default_outcome_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build post-hoc outcome metadata without changing prompts/evidence."""

    answer = record.get("answer") or record.get("gold") or record.get("answer_metadata")
    source = None
    if record.get("answer") or record.get("gold"):
        source = "dataset_answer_field"
    elif record.get("answer_metadata"):
        source = "answer_metadata"
    return {
        "ground_truth_available": answer is not None and str(answer).strip() != "",
        "ground_truth_source": source,
        "ground_truth_value": None if answer is None else str(answer),
        "extracted_answer": None,
        "extraction_method": "not_computed",
        "is_correct": None,
        "correctness_used_for_prompt": False,
        "correctness_used_for_evidence_generation": False,
        "correctness_available_for_alignment": False,
    }


def compute_rollout_group_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute success/failure contrast metadata for fake or real K-rollout groups."""

    grouped: dict[str, RolloutGroupStats] = {}
    for record in records:
        group_uid = str(record.get("rollout_group_uid") or record.get("prompt_sample_uid") or "")
        if not group_uid:
            continue
        stats = grouped.setdefault(group_uid, RolloutGroupStats(rollout_group_uid=group_uid))
        if "rollout_id" in record:
            stats.rollout_ids.append(int(record["rollout_id"]))
        outcome = record.get("outcome_metadata")
        is_correct = None
        if isinstance(outcome, Mapping):
            is_correct = outcome.get("is_correct")
        if is_correct is True:
            stats.num_success += 1
        elif is_correct is False:
            stats.num_failure += 1
    return {key: value.to_dict() for key, value in grouped.items()}


def compute_alignment_weights(
    record: Mapping[str, Any],
    strategy: str = "none",
) -> AlignmentWeights:
    """Return alignment weights for one record.

    The default and unavailable strategies intentionally return no-op weights so
    existing 2C/4C FC-OPD loss behavior is unchanged.
    """

    response_ids = record.get("response_token_ids", ())
    length = len(response_ids) if isinstance(response_ids, Sequence) else 0
    ones = tuple(1.0 for _ in range(length))
    if strategy in {"none", "placeholder_noop"}:
        return AlignmentWeights(
            strategy=strategy,
            available=True,
            token_weights=ones,
            notes="No-op alignment; FC-OPD loss behavior is unchanged.",
        )
    if strategy == "visual_task_signal_only":
        signals = record.get("condition_signals")
        if not isinstance(signals, Mapping):
            return AlignmentWeights(
                strategy=strategy,
                available=False,
                token_weights=ones,
                warnings=("condition_signals_missing",),
            )
        visual = signals.get("visual_detail")
        task = signals.get("task_extraction")
        if not isinstance(visual, Sequence) or not isinstance(task, Sequence):
            return AlignmentWeights(
                strategy=strategy,
                available=False,
                token_weights=ones,
                warnings=("visual_or_task_signal_missing",),
            )
        weights = []
        for index in range(length):
            value = 1.0
            if index < len(visual):
                value += max(0.0, float(visual[index]))
            if index < len(task):
                value += max(0.0, float(task[index]))
            weights.append(value)
        return AlignmentWeights(strategy=strategy, available=True, token_weights=tuple(weights))
    if strategy in {"correctness_contrastive", "success_failure_rollout_contrastive"}:
        outcome = record.get("outcome_metadata")
        if not isinstance(outcome, Mapping) or outcome.get("is_correct") is None:
            return AlignmentWeights(
                strategy=strategy,
                available=False,
                token_weights=ones,
                warnings=("outcome_metadata_unavailable",),
            )
        return AlignmentWeights(
            strategy=strategy,
            available=False,
            token_weights=ones,
            warnings=("strategy_schema_present_but_not_implemented",),
        )
    return AlignmentWeights(
        strategy=strategy,
        available=False,
        token_weights=ones,
        warnings=("unknown_alignment_strategy",),
    )


def apply_alignment_weights(
    loss_per_token: torch.Tensor,
    weights: AlignmentWeights | Mapping[str, Any] | torch.Tensor | Sequence[float] | None,
) -> torch.Tensor:
    """Apply token weights to a per-token loss tensor.

    Returns a weighted mean. ``None`` or unavailable/no-op weights leave the
    result equal to ``loss_per_token.mean()``.
    """

    if loss_per_token.numel() == 0:
        raise ValueError("loss_per_token must be non-empty")
    if weights is None:
        return loss_per_token.mean()
    raw: Any = weights
    available = True
    if isinstance(weights, AlignmentWeights):
        raw = weights.token_weights
        available = weights.available
    elif isinstance(weights, Mapping):
        raw = weights.get("token_weights")
        available = bool(weights.get("available", True))
    if raw is None or not available:
        return loss_per_token.mean()
    if isinstance(raw, torch.Tensor):
        weight_tensor = raw.to(loss_per_token.device, dtype=loss_per_token.dtype)
    else:
        weight_tensor = torch.tensor(list(raw), dtype=loss_per_token.dtype, device=loss_per_token.device)
    if weight_tensor.shape != loss_per_token.shape:
        try:
            weight_tensor = weight_tensor.reshape(loss_per_token.shape)
        except RuntimeError as exc:
            raise ValueError("alignment token weights must match loss_per_token shape") from exc
    if torch.any(weight_tensor < 0):
        raise ValueError("alignment token weights must be non-negative")
    return (loss_per_token * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)

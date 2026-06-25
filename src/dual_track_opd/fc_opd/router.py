"""Router 0 and Router 1 for sparse condition selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch

from .conditions import Condition


@dataclass(frozen=True)
class RouterConfig:
    mode: str = "chunk"
    single_condition: Condition = Condition.FULL
    chunk_condition: Mapping[str, Condition] = field(
        default_factory=lambda: {
            "visual_evidence": Condition.TASK,
            "reasoning": Condition.FULL,
            "answer": Condition.FULL,
        }
    )
    invalid_format_condition: Condition = Condition.FULL


def _validate_masks(chunk_masks: Mapping[str, torch.Tensor], response_mask: torch.Tensor) -> None:
    expected = {"visual_evidence", "reasoning", "answer"}
    if set(chunk_masks) != expected:
        raise ValueError(f"chunk_masks must contain exactly {sorted(expected)}")
    for name, mask in chunk_masks.items():
        if mask.shape != response_mask.shape:
            raise ValueError(f"{name} mask must have shape {tuple(response_mask.shape)}")
    total = sum(mask.bool().int() for mask in chunk_masks.values())
    if torch.any(total > 1):
        raise ValueError("chunk masks must not overlap")


def route_condition_weights(
    signals: Mapping[str, torch.Tensor],
    chunk_masks: Mapping[str, torch.Tensor],
    router_config: RouterConfig,
    *,
    response_mask: torch.Tensor,
    available_conditions: Sequence[Condition | str] | None = None,
    format_valid: torch.Tensor | None = None,
) -> dict[Condition, torch.Tensor]:
    """Return detached top-1 condition weights for Router 0 or Router 1.

    ``signals`` is accepted for API stability but intentionally unused by these
    first deterministic routers.
    """

    del signals
    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [batch, seq]")
    _validate_masks(chunk_masks, response_mask)
    available = (
        set(Condition(condition) for condition in available_conditions)
        if available_conditions is not None
        else set(Condition)
    )
    if format_valid is not None and format_valid.shape != response_mask.shape[:1]:
        raise ValueError("format_valid must have shape [batch]")

    weights = {
        condition: torch.zeros_like(response_mask, dtype=torch.float32)
        for condition in available
    }
    valid_tokens = response_mask.bool()

    if router_config.mode == "single":
        condition = router_config.single_condition
        if condition not in available:
            raise ValueError(f"selected condition is unavailable: {condition}")
        weights[condition][valid_tokens] = 1.0
    elif router_config.mode == "chunk":
        for chunk_name, mask in chunk_masks.items():
            condition = router_config.chunk_condition.get(chunk_name)
            if condition is None:
                raise ValueError(f"no condition configured for chunk: {chunk_name}")
            if condition not in available:
                raise ValueError(f"condition {condition} for {chunk_name} is unavailable")
            weights[condition][mask.bool() & valid_tokens] = 1.0

        covered = torch.zeros_like(valid_tokens)
        for weight in weights.values():
            covered |= weight.bool()
        uncovered = valid_tokens & ~covered
        if format_valid is not None:
            invalid_tokens = (~format_valid.bool()).unsqueeze(-1).expand_as(valid_tokens) & valid_tokens
            uncovered |= invalid_tokens
            for condition in weights:
                weights[condition][invalid_tokens] = 0.0
        fallback_condition = router_config.invalid_format_condition
        if torch.any(uncovered):
            if fallback_condition not in available:
                raise ValueError(f"fallback condition is unavailable: {fallback_condition}")
            weights[fallback_condition][uncovered] = 1.0
    else:
        raise ValueError("router mode must be 'single' or 'chunk'")

    weight_sum = sum(weights.values())
    if torch.any(weight_sum > 1.0 + 1e-6):
        raise ValueError("router assigned more than one condition to a token")
    if not torch.equal(weight_sum.bool(), valid_tokens):
        raise ValueError("router did not assign exactly one condition to every response token")
    return {condition: weight.detach() for condition, weight in weights.items()}

"""Router 0 and Router 1 for sparse condition selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch

from .conditions import Condition

DEFAULT_CHUNK_CONDITION_ROUTING: dict[str, tuple[Condition, ...]] = {
    "visible_evidence": (Condition.TASK_VISIBLE, Condition.FULL, Condition.FREE),
    "diagram_inference": (Condition.TASK_INFER, Condition.FULL),
    "reasoning": (Condition.TASK_INFER, Condition.TASK_SOLVE),
    "answer": (Condition.TASK_SOLVE,),
}
LEGACY_CHUNK_ALIASES = {"visual_evidence": "visible_evidence"}
CONDITION_FALLBACKS: dict[Condition, tuple[Condition, ...]] = {
    Condition.TASK: (Condition.TASK_VISIBLE,),
    Condition.TASK_VISIBLE: (Condition.TASK,),
    Condition.TASK_INFER: (Condition.TASK, Condition.FULL),
    Condition.TASK_SOLVE: (Condition.FULL,),
}


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
    chunk_condition_routing: Mapping[str, Sequence[Condition | str]] = field(
        default_factory=lambda: DEFAULT_CHUNK_CONDITION_ROUTING
    )
    delta_threshold: float = 0.0
    entropy_max: float | None = None
    topk_mass_min: float | None = None
    max_conditions_per_token: int = 2
    normalize_weights_per_token: bool = True


def _normalize_chunk_masks(
    chunk_masks: Mapping[str, torch.Tensor],
    response_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for name, mask in chunk_masks.items():
        canonical = LEGACY_CHUNK_ALIASES.get(name, name)
        if canonical in normalized:
            normalized[canonical] = normalized[canonical].bool() | mask.bool()
        else:
            normalized[canonical] = mask.bool()
    if "visible_evidence" not in normalized and "visual_evidence" in chunk_masks:
        normalized["visible_evidence"] = chunk_masks["visual_evidence"].bool()
    for name in ("visible_evidence", "diagram_inference", "reasoning", "answer"):
        normalized.setdefault(name, torch.zeros_like(response_mask, dtype=torch.bool))
    return normalized


def _validate_masks(chunk_masks: Mapping[str, torch.Tensor], response_mask: torch.Tensor) -> None:
    expected = {"visible_evidence", "diagram_inference", "reasoning", "answer"}
    if set(chunk_masks) != expected:
        raise ValueError(f"chunk_masks must contain exactly {sorted(expected)}")
    for name, mask in chunk_masks.items():
        if mask.shape != response_mask.shape:
            raise ValueError(f"{name} mask must have shape {tuple(response_mask.shape)}")
    total = sum(mask.bool().int() for mask in chunk_masks.values())
    if torch.any(total > 1):
        raise ValueError("chunk masks must not overlap")


def _available_condition(condition: Condition, available: set[Condition]) -> Condition | None:
    if condition in available:
        return condition
    for fallback in CONDITION_FALLBACKS.get(condition, ()):
        if fallback in available:
            return fallback
    return None


def _chunk_delta_signal(chunk_name: str, signals: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    names = {
        "visible_evidence": ("visual_detail_delta", "visual_detail"),
        "diagram_inference": ("diagram_infer_delta", "diagram_infer"),
        "reasoning": ("solve_delta", "diagram_infer_delta", "diagram_infer"),
        "answer": ("solve_delta",),
    }.get(chunk_name, ())
    for name in names:
        signal = signals.get(name)
        if signal is not None:
            return signal
    return None


def _entropy_gate(
    signals: Mapping[str, torch.Tensor],
    *,
    threshold: float | None,
    shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    if threshold is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    entropy = signals.get("teacher_entropy")
    if entropy is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    return entropy <= threshold


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
    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [batch, seq]")
    chunk_masks = _normalize_chunk_masks(chunk_masks, response_mask)
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

    if router_config.max_conditions_per_token < 1:
        raise ValueError("max_conditions_per_token must be at least 1")

    if router_config.mode == "single":
        condition = router_config.single_condition
        if condition not in available:
            raise ValueError(f"selected condition is unavailable: {condition}")
        weights[condition][valid_tokens] = 1.0
    elif router_config.mode == "uniform_all_conditions":
        if not available:
            raise ValueError("uniform_all_conditions requires at least one available condition")
        value = 1.0 / len(available)
        for condition in available:
            weights[condition][valid_tokens] = value
    elif router_config.mode == "chunk":
        for chunk_name, mask in chunk_masks.items():
            condition = router_config.chunk_condition.get(chunk_name) or router_config.chunk_condition.get("visual_evidence")
            if condition is None:
                raise ValueError(f"no condition configured for chunk: {chunk_name}")
            condition = _available_condition(Condition(condition), available)
            if condition is None:
                raise ValueError(f"condition for {chunk_name} is unavailable")
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
    elif router_config.mode == "chunk_gated":
        entropy_ok = _entropy_gate(
            signals,
            threshold=router_config.entropy_max,
            shape=response_mask.shape,
            device=response_mask.device,
        )
        for chunk_name, mask in chunk_masks.items():
            candidate_raw = router_config.chunk_condition_routing.get(chunk_name, ())
            candidates: list[Condition] = []
            for condition_like in candidate_raw:
                condition = _available_condition(Condition(condition_like), available)
                if condition is not None and condition not in candidates:
                    candidates.append(condition)
            if not candidates:
                continue

            token_mask = mask.bool() & valid_tokens & entropy_ok
            delta = _chunk_delta_signal(chunk_name, signals)
            if delta is not None:
                if delta.shape != response_mask.shape:
                    raise ValueError(f"{chunk_name} delta signal must have shape {tuple(response_mask.shape)}")
                token_mask = token_mask & (delta >= router_config.delta_threshold)
            if not torch.any(token_mask):
                continue

            selected = candidates[: router_config.max_conditions_per_token]
            value = 1.0 / len(selected) if router_config.normalize_weights_per_token else 1.0
            for condition in selected:
                weights[condition][token_mask] = value

        covered = torch.zeros_like(valid_tokens)
        for weight in weights.values():
            covered |= weight.bool()
        uncovered = valid_tokens & ~covered
        if torch.any(uncovered):
            fallback_condition = _available_condition(router_config.invalid_format_condition, available)
            if fallback_condition is None:
                raise ValueError(f"fallback condition is unavailable: {router_config.invalid_format_condition}")
            weights[fallback_condition][uncovered] = 1.0
    else:
        raise ValueError("router mode must be 'single', 'chunk', 'chunk_gated', or 'uniform_all_conditions'")

    weight_sum = sum(weights.values())
    max_allowed = 1.0 + 1e-6 if router_config.normalize_weights_per_token else router_config.max_conditions_per_token + 1e-6
    if torch.any(weight_sum > max_allowed):
        raise ValueError("router assigned more than one condition to a token")
    if not torch.equal(weight_sum.bool(), valid_tokens):
        raise ValueError("router did not assign at least one condition to every response token")
    return {condition: weight.detach() for condition, weight in weights.items()}

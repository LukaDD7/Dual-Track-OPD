"""Router 0 and Router 1 for sparse condition selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch

from .conditions import CAPABILITY_CONTRASTS, Condition

DEFAULT_CHUNK_CONDITION_ROUTING: dict[str, tuple[Condition, ...]] = {
    "visible_evidence": (Condition.TASK_VISIBLE, Condition.FULL, Condition.FREE),
    "diagram_inference": (Condition.TASK_INFER, Condition.FULL),
    "reasoning": (Condition.TASK_INFER, Condition.TASK_SOLVE),
    "answer": (Condition.TASK_SOLVE,),
}
CONTRASTIVE_CHUNK_CONDITION_ROUTING: dict[str, tuple[Condition, ...]] = {
    "visible_evidence": (Condition.FULL, Condition.DEGRADED, Condition.TASK_VISIBLE, Condition.FREE),
    "diagram_inference": (Condition.TASK_INFER, Condition.TASK_VISIBLE, Condition.FULL, Condition.DEGRADED),
    "reasoning": (Condition.TASK_SOLVE, Condition.TASK_INFER),
    "answer": (Condition.TASK_SOLVE, Condition.TASK_INFER),
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


CAPABILITY_POSITIVE_CONDITION: dict[str, Condition] = {
    name: positive for name, (positive, _negative) in CAPABILITY_CONTRASTS.items()
}


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
    elif router_config.mode == "student_deficit_chunk_gated":
        capability_scores = signals.get("capability_scores") if isinstance(signals, Mapping) else None
        if isinstance(capability_scores, Mapping) and capability_scores:
            capability_weights: list[tuple[str, Condition, torch.Tensor]] = []
            for capability, block in capability_scores.items():
                if not isinstance(block, Mapping) or block.get("valid") is not True:
                    continue
                condition = _available_condition(
                    Condition(block.get("positive", CAPABILITY_POSITIVE_CONDITION.get(str(capability), Condition.FULL))),
                    available,
                )
                if condition is None:
                    continue
                raw = block.get("final_token_weight", [])
                tensor = torch.tensor(raw, dtype=torch.float32, device=response_mask.device).reshape(1, -1)
                if tensor.shape != response_mask.shape:
                    raise ValueError(f"{capability} final_token_weight must have shape {tuple(response_mask.shape)}")
                capability_weights.append((str(capability), condition, tensor.clamp_min(0.0)))
            if capability_weights:
                combined = {condition: torch.zeros_like(response_mask, dtype=torch.float32) for condition in available}
                for _capability, condition, tensor in capability_weights:
                    combined[condition] = combined[condition] + tensor * valid_tokens.float()
                weight_sum = sum(combined.values())
                nonzero = weight_sum > 0
                if router_config.normalize_weights_per_token and torch.any(nonzero):
                    for condition in combined:
                        combined[condition] = torch.where(
                            nonzero,
                            combined[condition] / weight_sum.clamp_min(1e-12),
                            combined[condition],
                        )
                uncovered = valid_tokens & ~nonzero
                if torch.any(uncovered):
                    fallback_condition = _available_condition(router_config.invalid_format_condition, available)
                    if fallback_condition is None:
                        raise ValueError(f"fallback condition is unavailable: {router_config.invalid_format_condition}")
                    combined[fallback_condition][uncovered] = 1.0
                weights = combined
            else:
                fallback = RouterConfig(
                    mode="chunk_gated_contrastive",
                    invalid_format_condition=router_config.invalid_format_condition,
                    delta_threshold=router_config.delta_threshold,
                    entropy_max=router_config.entropy_max,
                    topk_mass_min=router_config.topk_mass_min,
                    max_conditions_per_token=max(router_config.max_conditions_per_token, 4),
                    normalize_weights_per_token=router_config.normalize_weights_per_token,
                )
                return route_condition_weights(
                    signals,
                    chunk_masks,
                    fallback,
                    response_mask=response_mask,
                    available_conditions=available_conditions,
                    format_valid=format_valid,
                )
        else:
            fallback = RouterConfig(
                mode="chunk_gated_contrastive",
                invalid_format_condition=router_config.invalid_format_condition,
                delta_threshold=router_config.delta_threshold,
                entropy_max=router_config.entropy_max,
                topk_mass_min=router_config.topk_mass_min,
                max_conditions_per_token=max(router_config.max_conditions_per_token, 4),
                normalize_weights_per_token=router_config.normalize_weights_per_token,
            )
            return route_condition_weights(
                signals,
                chunk_masks,
                fallback,
                response_mask=response_mask,
                available_conditions=available_conditions,
                format_valid=format_valid,
            )
    elif router_config.mode in {"chunk_gated", "chunk_gated_primary", "chunk_gated_contrastive"}:
        entropy_ok = _entropy_gate(
            signals,
            threshold=router_config.entropy_max,
            shape=response_mask.shape,
            device=response_mask.device,
        )
        for chunk_name, mask in chunk_masks.items():
            if router_config.mode == "chunk_gated_contrastive":
                candidate_raw = CONTRASTIVE_CHUNK_CONDITION_ROUTING.get(chunk_name, ())
            else:
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
        raise ValueError(
            "router mode must be 'single', 'chunk', 'chunk_gated', "
            "'chunk_gated_primary', 'chunk_gated_contrastive', "
            "'student_deficit_chunk_gated', or 'uniform_all_conditions'"
        )

    weight_sum = sum(weights.values())
    max_allowed = 1.0 + 1e-6 if router_config.normalize_weights_per_token else router_config.max_conditions_per_token + 1e-6
    if torch.any(weight_sum > max_allowed):
        raise ValueError("router assigned more than one condition to a token")
    if not torch.equal(weight_sum.bool(), valid_tokens):
        raise ValueError("router did not assign at least one condition to every response token")
    return {condition: weight.detach() for condition, weight in weights.items()}

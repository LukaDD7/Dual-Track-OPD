"""Numerically stable visual counterfactual signals and candidate selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class JSDivergenceStatistics:
    js: torch.Tensor
    entropy_left: torch.Tensor
    entropy_right: torch.Tensor


@dataclass(frozen=True)
class CandidateSelectionConfig:
    windows: tuple[int, ...] = (8, 16, 32, 64)
    topk_high: int = 1
    topk_drop: int = 2
    fixed_positions: tuple[float, ...] = (0.2, 0.5, 0.8)
    min_candidates: int = 3
    max_candidates: int = 6
    nms_radius: int = 16
    snap_radius: int = 12
    output_half_width: int = 8

    def validate(self) -> None:
        if not self.windows or any(value <= 0 for value in self.windows):
            raise ValueError("candidate windows must be positive")
        if self.min_candidates <= 0 or self.max_candidates < self.min_candidates:
            raise ValueError("invalid candidate count bounds")
        if self.topk_high < 0 or self.topk_drop < 0:
            raise ValueError("top-k values must be non-negative")
        if self.nms_radius < 0 or self.snap_radius < 0 or self.output_half_width <= 0:
            raise ValueError("candidate radii must be non-negative")
        if any(not 0.0 < value < 1.0 for value in self.fixed_positions):
            raise ValueError("fixed candidate positions must lie in (0, 1)")


@dataclass(frozen=True)
class SelectedCandidate:
    start: int
    end: int
    anchor: int
    relative_position: float
    sources: tuple[str, ...]
    snapped: bool
    local_value: float
    drop_value: float


def full_vocab_js_statistics(
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
) -> JSDivergenceStatistics:
    """Return per-position JS and entropies without materializing a mixture.

    Inputs may be ``[T, V]`` or have arbitrary leading dimensions.  All
    probability arithmetic is promoted to fp32.  The returned tensors omit the
    vocabulary dimension and stay on the input device.
    """

    if left_logits.shape != right_logits.shape:
        raise ValueError("counterfactual logits must have identical shapes")
    if left_logits.ndim < 2 or left_logits.shape[-1] <= 1:
        raise ValueError("logits must have a non-trivial vocabulary dimension")
    if not torch.isfinite(left_logits).all() or not torch.isfinite(right_logits).all():
        raise ValueError("counterfactual logits contain NaN or Inf")

    left_logp = torch.log_softmax(left_logits.float(), dim=-1)
    right_logp = torch.log_softmax(right_logits.float(), dim=-1)
    log_mix = torch.logaddexp(left_logp, right_logp) - math.log(2.0)
    left_prob = left_logp.exp()
    right_prob = right_logp.exp()
    js = 0.5 * (
        (left_prob * (left_logp - log_mix)).sum(dim=-1)
        + (right_prob * (right_logp - log_mix)).sum(dim=-1)
    )
    # Tiny negative values can appear after fp32 cancellation even though JS
    # is non-negative analytically.
    js = js.clamp_min(0.0)
    entropy_left = (-(left_prob * left_logp).sum(dim=-1)).clamp_min(0.0)
    entropy_right = (-(right_prob * right_logp).sum(dim=-1)).clamp_min(0.0)
    return JSDivergenceStatistics(
        js=js,
        entropy_left=entropy_left,
        entropy_right=entropy_right,
    )


def concatenate_chunk_statistics(
    chunk_pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> JSDivergenceStatistics:
    """Compute JS online over sequence chunks and concatenate scalar outputs."""

    js_parts: list[torch.Tensor] = []
    left_parts: list[torch.Tensor] = []
    right_parts: list[torch.Tensor] = []
    for left, right in chunk_pairs:
        stats = full_vocab_js_statistics(left, right)
        js_parts.append(stats.js.detach().cpu())
        left_parts.append(stats.entropy_left.detach().cpu())
        right_parts.append(stats.entropy_right.detach().cpu())
    if not js_parts:
        raise ValueError("at least one chunk pair is required")
    return JSDivergenceStatistics(
        js=torch.cat(js_parts, dim=-1),
        entropy_left=torch.cat(left_parts, dim=-1),
        entropy_right=torch.cat(right_parts, dim=-1),
    )


def trailing_mean(values: Sequence[float], width: int) -> list[float]:
    if width <= 0:
        raise ValueError("width must be positive")
    if not values:
        return []
    prefix = [0.0]
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("signal values must be finite")
        prefix.append(prefix[-1] + value)
    result = []
    for end in range(1, len(values) + 1):
        start = max(0, end - width)
        result.append((prefix[end] - prefix[start]) / (end - start))
    return result


def transition_drop(values: Sequence[float], width: int) -> list[float]:
    """Pre-window mean minus post-window mean at every token anchor."""

    if width <= 0:
        raise ValueError("width must be positive")
    if not values:
        return []
    result: list[float] = []
    for anchor in range(len(values)):
        before = values[max(0, anchor - width) : anchor]
        after = values[anchor : min(len(values), anchor + width)]
        if not before or not after:
            result.append(0.0)
        else:
            result.append(float(fmean(before) - fmean(after)))
    return result


def multiscale_curves(
    values: Sequence[float],
    windows: Sequence[int],
) -> dict[str, list[float]]:
    if not values:
        raise ValueError("visual signal must be non-empty")
    widths = tuple(sorted(set(int(value) for value in windows)))
    if not widths or any(value <= 0 for value in widths):
        raise ValueError("multi-scale windows must be positive")
    local_by_width = [trailing_mean(values, width) for width in widths]
    drop_by_width = [transition_drop(values, width) for width in widths]
    local = [max(curve[index] for curve in local_by_width) for index in range(len(values))]
    drop = [max(curve[index] for curve in drop_by_width) for index in range(len(values))]
    return {
        "local": local,
        "drop": drop,
        **{f"local_w{width}": curve for width, curve in zip(widths, local_by_width, strict=True)},
        **{f"drop_w{width}": curve for width, curve in zip(widths, drop_by_width, strict=True)},
    }


def _top_indices(values: Sequence[float], count: int) -> list[int]:
    if count <= 0:
        return []
    return sorted(range(len(values)), key=lambda index: (-float(values[index]), index))[:count]


def _boundary_positions(decoded_tokens: Sequence[str]) -> list[int]:
    boundaries = []
    for index, token in enumerate(decoded_tokens):
        stripped = str(token).rstrip()
        if "\n" in str(token) or stripped.endswith((".", "?", "!", ";", ":", "。", "？", "！")):
            # The next token is the natural start of a new local state.
            boundaries.append(min(index + 1, len(decoded_tokens) - 1))
    return sorted(set(boundaries))


def _snap(anchor: int, boundaries: Sequence[int], radius: int) -> tuple[int, bool]:
    candidates = [value for value in boundaries if abs(value - anchor) <= radius]
    if not candidates:
        return anchor, False
    snapped = min(candidates, key=lambda value: (abs(value - anchor), value))
    return snapped, snapped != anchor


def _window_mean(values: Sequence[float], anchor: int, half_width: int) -> float:
    start = max(0, anchor - half_width)
    end = min(len(values), anchor + half_width + 1)
    return float(fmean(values[start:end]))


def select_visual_candidates(
    values: Sequence[float],
    *,
    decoded_tokens: Sequence[str] | None = None,
    config: CandidateSelectionConfig = CandidateSelectionConfig(),
) -> list[SelectedCandidate]:
    """Select 3-6 diverse transition candidates with source-aware NMS."""

    config.validate()
    signal = [float(value) for value in values]
    if len(signal) < 2 or any(not math.isfinite(value) or value < 0 for value in signal):
        raise ValueError("visual JS must contain at least two finite non-negative values")
    if decoded_tokens is not None and len(decoded_tokens) != len(signal):
        raise ValueError("decoded token count must match the JS trajectory")

    curves = multiscale_curves(signal, config.windows)
    proposed: dict[int, dict[str, object]] = {}

    def add(anchor: int, source: str, score: float, *, priority: int) -> None:
        anchor = min(max(int(anchor), 0), len(signal) - 1)
        item = proposed.setdefault(anchor, {"sources": set(), "score": -math.inf, "priority": priority})
        item["sources"].add(source)  # type: ignore[union-attr]
        item["score"] = max(float(item["score"]), float(score))
        item["priority"] = min(int(item["priority"]), priority)

    for index in _top_indices(curves["local"], config.topk_high):
        add(index, "high_visual_dependence", curves["local"][index], priority=0)
    for index in _top_indices(curves["drop"], config.topk_drop):
        add(index, "visual_dependence_drop", curves["drop"][index], priority=0)

    # A sustained-low candidate is distinct from the maximum instantaneous
    # drop: it rewards high preceding dependence and low following dependence.
    sustained_scores = []
    width = min(max(config.windows), max(1, len(signal) // 3))
    for anchor in range(len(signal)):
        before = signal[max(0, anchor - width) : anchor]
        after = signal[anchor : min(len(signal), anchor + width)]
        score = 0.0 if not before or not after else float(fmean(before) - fmean(after))
        sustained_scores.append(score)
    if sustained_scores:
        index = _top_indices(sustained_scores, 1)[0]
        add(index, "post_visual_low", sustained_scores[index], priority=1)

    fixed_anchors = []
    for fraction in config.fixed_positions:
        index = min(len(signal) - 1, max(0, int(round(fraction * (len(signal) - 1)))))
        fixed_anchors.append(index)
        add(index, f"fixed_{fraction:.2f}", curves["local"][index], priority=2)

    ordered = sorted(
        proposed,
        key=lambda index: (
            int(proposed[index]["priority"]),
            -float(proposed[index]["score"]),
            index,
        ),
    )
    kept: list[int] = []
    for anchor in ordered:
        nearby = next((value for value in kept if abs(value - anchor) <= config.nms_radius), None)
        if nearby is not None:
            proposed[nearby]["sources"].update(proposed[anchor]["sources"])  # type: ignore[union-attr]
            continue
        kept.append(anchor)
        if len(kept) >= config.max_candidates:
            break

    # NMS can collapse a short response to too few candidates.  Fill from the
    # fixed controls with a relaxed radius, then from evenly spaced anchors.
    relaxed_radius = max(1, config.nms_radius // 2)
    fillers = fixed_anchors + [
        int(round(index * (len(signal) - 1) / max(config.min_candidates - 1, 1)))
        for index in range(config.min_candidates)
    ]
    for anchor in fillers:
        if len(kept) >= config.min_candidates:
            break
        if any(abs(value - anchor) <= relaxed_radius for value in kept):
            continue
        if anchor not in proposed:
            proposed[anchor] = {"sources": {"coverage_control"}, "score": 0.0, "priority": 3}
        kept.append(anchor)

    boundaries = _boundary_positions(decoded_tokens or ())
    selected: list[SelectedCandidate] = []
    occupied: set[int] = set()
    for original_anchor in kept[: config.max_candidates]:
        anchor, snapped = _snap(original_anchor, boundaries, config.snap_radius)
        if anchor in occupied:
            anchor = original_anchor
            snapped = False
        occupied.add(anchor)
        start = max(0, anchor - config.output_half_width)
        end = min(len(signal), anchor + config.output_half_width + 1)
        selected.append(SelectedCandidate(
            start=start,
            end=end,
            anchor=anchor,
            relative_position=float(anchor / max(len(signal) - 1, 1)),
            sources=tuple(sorted(str(value) for value in proposed[original_anchor]["sources"])),
            snapped=snapped,
            local_value=_window_mean(curves["local"], original_anchor, config.output_half_width),
            drop_value=float(curves["drop"][original_anchor]),
        ))
    return sorted(selected, key=lambda value: value.anchor)


def token_signal_rows(
    *,
    token_ids: Sequence[int],
    js_full_degraded: Sequence[float],
    js_full_null: Sequence[float],
    entropy_full: Sequence[float],
    entropy_degraded: Sequence[float],
    entropy_null: Sequence[float],
) -> list[dict[str, float | int]]:
    arrays: Mapping[str, Sequence[float] | Sequence[int]] = {
        "token_ids": token_ids,
        "js_full_degraded": js_full_degraded,
        "js_full_null": js_full_null,
        "entropy_full": entropy_full,
        "entropy_degraded": entropy_degraded,
        "entropy_null": entropy_null,
    }
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all token-signal arrays must have equal length")
    rows = []
    for index in range(len(token_ids)):
        row: dict[str, float | int] = {
            "position": index,
            "token_id": int(token_ids[index]),
            "js_full_degraded": float(js_full_degraded[index]),
            "js_full_null": float(js_full_null[index]),
            "entropy_full": float(entropy_full[index]),
            "entropy_degraded": float(entropy_degraded[index]),
            "entropy_null": float(entropy_null[index]),
        }
        if any(not math.isfinite(float(value)) for key, value in row.items() if key not in {"position", "token_id"}):
            raise ValueError("token signals contain NaN or Inf")
        rows.append(row)
    return rows

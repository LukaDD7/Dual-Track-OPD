"""Outcome-grounded aggregation for causal state probes.

This module contains no model I/O.  It turns verifier outcomes and teacher-path
support into actionability labels that can be tested on CPU before any training
operator is connected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence

from .causal_schema import ContinuationEstimate
from .prefix_intervention import posterior_lift_probability


@dataclass(frozen=True)
class ActionabilityThresholds:
    min_mean_gain: float = 0.20
    min_posterior_probability: float = 0.90
    max_low_gain: float = 0.10

    def validate(self) -> None:
        if self.min_mean_gain < 0 or self.max_low_gain < 0:
            raise ValueError("gain thresholds must be non-negative")
        if self.max_low_gain > self.min_mean_gain:
            raise ValueError("max_low_gain cannot exceed min_mean_gain")
        if not 0.0 <= self.min_posterior_probability <= 1.0:
            raise ValueError("posterior threshold must lie in [0, 1]")


@dataclass(frozen=True)
class ProbeLift:
    treatment: str
    control: str
    mean_lift: float
    posterior_probability: float
    treatment_pass_rate: float
    control_pass_rate: float

    @property
    def positive(self) -> bool:
        return self.mean_lift > 0.0 and self.posterior_probability > 0.5


@dataclass(frozen=True)
class ReachabilityBarrier:
    start: int
    end: int
    anchor: int
    mean_nll: float
    peak_nll: float
    relative_position: float


def compare_estimates(
    treatment: ContinuationEstimate,
    control: ContinuationEstimate,
    *,
    seed: int,
    draws: int = 20_000,
) -> ProbeLift:
    treatment.validate()
    control.validate()
    if treatment.n != control.n:
        raise ValueError("posterior comparison currently requires matched K")
    mean, probability = posterior_lift_probability(
        treatment.n_correct,
        control.n_correct,
        treatment.n,
        seed=seed,
        draws=draws,
    )
    return ProbeLift(
        treatment=treatment.condition,
        control=control.condition,
        mean_lift=mean,
        posterior_probability=probability,
        treatment_pass_rate=treatment.pass_rate,
        control_pass_rate=control.pass_rate,
    )


def visual_dependence_gains(
    estimates: Sequence[ContinuationEstimate],
) -> tuple[float | None, float | None]:
    by_condition = {estimate.condition: estimate for estimate in estimates}
    full = by_condition.get("full")
    degraded = by_condition.get("degraded")
    null = by_condition.get("null")
    fine = None if full is None or degraded is None else full.pass_rate - degraded.pass_rate
    overall = None if full is None or null is None else full.pass_rate - null.pass_rate
    return fine, overall


def classify_actionability(
    *,
    relay_gain: float | None,
    transport_gain: float | None,
    relay_probability: float | None = None,
    transport_probability: float | None = None,
    thresholds: ActionabilityThresholds = ActionabilityThresholds(),
) -> str:
    """Route a state without equating intervention failure with incapacity."""

    thresholds.validate()

    def high(gain: float | None, probability: float | None) -> bool:
        if gain is None:
            return False
        probability = 1.0 if probability is None else probability
        return gain >= thresholds.min_mean_gain and probability >= thresholds.min_posterior_probability

    def low(gain: float | None) -> bool:
        return gain is not None and gain <= thresholds.max_low_gain

    if high(relay_gain, relay_probability):
        return "on_policy_repairable"
    if low(relay_gain) and high(transport_gain, transport_probability):
        return "transportable_low_reachability"
    if low(relay_gain) and low(transport_gain):
        return "unresolved_under_current_intervention"
    return "insufficient_evidence"


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of no values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    location = probability * (len(ordered) - 1)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def detect_reachability_barriers(
    student_token_nll: Sequence[float],
    *,
    window: int = 32,
    top_k: int = 3,
    quantile_threshold: float = 0.80,
    nms_radius: int | None = None,
) -> list[ReachabilityBarrier]:
    """Find local high-NLL windows on a verified teacher trajectory."""

    nll = [float(value) for value in student_token_nll]
    if not nll or any(not math.isfinite(value) or value < 0 for value in nll):
        raise ValueError("teacher-path NLL must be finite and non-negative")
    if window <= 0 or top_k <= 0:
        raise ValueError("window and top_k must be positive")
    radius = max(1, window // 2) if nms_radius is None else max(0, nms_radius)
    scores = []
    for start in range(len(nll)):
        end = min(len(nll), start + window)
        scores.append(float(fmean(nll[start:end])))
    threshold = _quantile(scores, quantile_threshold)
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    selected: list[ReachabilityBarrier] = []
    for start in order:
        end = min(len(nll), start + window)
        anchor = (start + end - 1) // 2
        if scores[start] < threshold:
            break
        if any(abs(barrier.anchor - anchor) <= radius for barrier in selected):
            continue
        selected.append(ReachabilityBarrier(
            start=start,
            end=end,
            anchor=anchor,
            mean_nll=scores[start],
            peak_nll=max(nll[start:end]),
            relative_position=float(anchor / max(len(nll) - 1, 1)),
        ))
        if len(selected) >= top_k:
            break
    return sorted(selected, key=lambda value: value.anchor)


def summarize_actionability(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    relay_values: list[float] = []
    transport_values: list[float] = []
    for row in rows:
        state = str(row.get("state_class") or "unclassified")
        counts[state] = counts.get(state, 0) + 1
        relay = row.get("relay_gain")
        transport = row.get("transport_gain")
        if relay is not None and math.isfinite(float(relay)):
            relay_values.append(float(relay))
        if transport is not None and math.isfinite(float(transport)):
            transport_values.append(float(transport))
    return {
        "candidate_count": len(rows),
        "state_counts": dict(sorted(counts.items())),
        "mean_relay_gain": None if not relay_values else float(fmean(relay_values)),
        "mean_transport_gain": None if not transport_values else float(fmean(transport_values)),
    }

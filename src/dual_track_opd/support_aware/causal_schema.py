"""Shared records for causal state localization diagnostics.

The schema deliberately stores scalar/short-vector diagnostics only.  Full
vocabulary logits, model weights, images, and raw rollout corpora remain
outside the record so Phase-I outputs stay small enough to audit and merge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "causal-state-probe-v1"


def _require_finite(name: str, value: float | None) -> None:
    if value is None:
        return
    import math

    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite when present")


@dataclass(frozen=True)
class StudentSupport:
    """Observed native support with an explicit finite-budget posterior."""

    n_rollouts: int
    n_correct: int
    pass_rate: float
    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    posterior_prior: str = "jeffreys_beta_half_half"

    @classmethod
    def from_counts(cls, n_correct: int, n_rollouts: int) -> "StudentSupport":
        if n_rollouts <= 0 or not 0 <= n_correct <= n_rollouts:
            raise ValueError("support counts must satisfy 0 <= correct <= rollouts")
        alpha = float(n_correct) + 0.5
        beta = float(n_rollouts - n_correct) + 0.5
        return cls(
            n_rollouts=n_rollouts,
            n_correct=n_correct,
            pass_rate=float(n_correct / n_rollouts),
            posterior_alpha=alpha,
            posterior_beta=beta,
            posterior_mean=float(alpha / (alpha + beta)),
        )

    def validate(self) -> None:
        if self.n_rollouts <= 0 or not 0 <= self.n_correct <= self.n_rollouts:
            raise ValueError("invalid student-support counts")
        if not 0.0 <= self.pass_rate <= 1.0 or not 0.0 <= self.posterior_mean <= 1.0:
            raise ValueError("student-support probabilities must lie in [0, 1]")


@dataclass(frozen=True)
class ContinuationEstimate:
    """Verifier-backed success estimate for one fixed-prefix intervention."""

    condition: str
    n: int
    n_correct: int
    n_malformed: int
    pass_rate: float
    seeds: tuple[int, ...] = ()
    response_hashes: tuple[str, ...] = ()
    # Review P0-2: split the historical opaque `n_malformed` counter into
    # auditable reason counts.  Legacy shard JSON (written before these fields
    # existed) simply omits them; new records carry the full split.
    n_student_continuations_generated: int = 0
    n_wrong_format_valid: int = 0
    n_no_answer_marker: int = 0
    n_truncated: int = 0
    n_relay_answer_leakage: int = 0
    n_generation_error: int = 0

    def validate(self) -> None:
        if self.n <= 0 or not 0 <= self.n_correct <= self.n:
            raise ValueError("invalid continuation counts")
        if not 0 <= self.n_malformed <= self.n:
            raise ValueError("invalid malformed continuation count")
        if not 0.0 <= self.pass_rate <= 1.0:
            raise ValueError("continuation pass_rate must lie in [0, 1]")
        if self.seeds and len(self.seeds) != self.n:
            raise ValueError("continuation seed count must equal n")
        if self.response_hashes and len(self.response_hashes) != self.n:
            raise ValueError("continuation response-hash count must equal n")
        if self.n_student_continuations_generated:
            if (
                self.n_correct
                + self.n_wrong_format_valid
                + self.n_no_answer_marker
                + self.n_truncated
                + self.n_generation_error
                != self.n_student_continuations_generated
            ):
                raise ValueError(
                    "continuation counts inconsistent with generated continuations"
                )
            if self.n_relay_answer_leakage + self.n_student_continuations_generated != self.n:
                raise ValueError(
                    "relay-leakage count plus generated count must equal n"
                )
        elif (
            self.n_wrong_format_valid
            or self.n_no_answer_marker
            or self.n_truncated
            or self.n_generation_error
        ):
            raise ValueError(
                "reason counts set without n_student_continuations_generated"
            )
        elif self.n_relay_answer_leakage and self.n_relay_answer_leakage != self.n:
            # With zero generated continuations, every trial must be censored by
            # relay answer leakage (e.g. a teacher relay that always exposes the
            # gold answer).
            raise ValueError(
                "relay-answer-leakage count must equal n when no continuation "
                "was generated"
            )


@dataclass(frozen=True)
class CandidateWindow:
    """One candidate transition zone and all attached causal measurements."""

    candidate_id: str
    start: int
    end: int
    anchor: int
    relative_position: float
    sources: tuple[str, ...]
    snapped: bool = False
    student_js_full_degraded: float | None = None
    student_js_full_null: float | None = None
    js_drop_full_degraded: float | None = None
    js_drop_full_null: float | None = None
    entropy_full: float | None = None
    entropy_degraded: float | None = None
    entropy_null: float | None = None
    visual_continuations: tuple[ContinuationEstimate, ...] = ()
    visual_fine_gain: float | None = None
    visual_all_gain: float | None = None
    relay_continuations: Mapping[str, ContinuationEstimate] = field(default_factory=dict)
    relay_gain_by_length: Mapping[str, float] = field(default_factory=dict)
    relay_probability_by_length: Mapping[str, float] = field(default_factory=dict)
    unaided_continuation: ContinuationEstimate | None = None
    transport_continuation: ContinuationEstimate | None = None
    wrong_prefix_continuation: ContinuationEstimate | None = None
    transport_gain: float | None = None
    transport_probability: float | None = None
    transport_vs_wrong_gain: float | None = None
    transport_vs_wrong_probability: float | None = None
    wrong_prefix_gain: float | None = None
    answer_leakage_continuation: ContinuationEstimate | None = None
    answer_leakage: float | None = None
    student_teacher_overlap: float | None = None
    state_class: str | None = None
    notes: tuple[str, ...] = ()

    def validate(self, trajectory_length: int | None = None) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not 0 <= self.start <= self.anchor < self.end:
            raise ValueError("candidate indices must satisfy start <= anchor < end")
        if trajectory_length is not None and self.end > trajectory_length:
            raise ValueError("candidate window exceeds trajectory length")
        if not 0.0 <= self.relative_position <= 1.0:
            raise ValueError("relative_position must lie in [0, 1]")
        if not self.sources:
            raise ValueError("candidate sources must be non-empty")
        for name in (
            "student_js_full_degraded",
            "student_js_full_null",
            "js_drop_full_degraded",
            "js_drop_full_null",
            "entropy_full",
            "entropy_degraded",
            "entropy_null",
            "visual_fine_gain",
            "visual_all_gain",
            "transport_gain",
            "transport_probability",
            "transport_vs_wrong_gain",
            "transport_vs_wrong_probability",
            "wrong_prefix_gain",
            "answer_leakage",
            "student_teacher_overlap",
        ):
            _require_finite(name, getattr(self, name))
        for key, value in self.relay_gain_by_length.items():
            _require_finite(f"relay_gain_by_length[{key}]", float(value))
        for key, estimate in self.relay_continuations.items():
            if not key:
                raise ValueError("relay continuation keys must be non-empty")
            estimate.validate()
        for key, value in self.relay_probability_by_length.items():
            _require_finite(f"relay_probability_by_length[{key}]", float(value))
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("relay posterior probabilities must lie in [0, 1]")
        if self.transport_probability is not None and not 0.0 <= self.transport_probability <= 1.0:
            raise ValueError("transport_probability must lie in [0, 1]")
        if self.transport_vs_wrong_probability is not None and not 0.0 <= self.transport_vs_wrong_probability <= 1.0:
            raise ValueError("transport_vs_wrong_probability must lie in [0, 1]")
        for estimate in (
            self.unaided_continuation,
            self.transport_continuation,
            self.wrong_prefix_continuation,
            self.answer_leakage_continuation,
        ):
            if estimate is not None:
                estimate.validate()
        for estimate in self.visual_continuations:
            estimate.validate()


@dataclass(frozen=True)
class CausalStateRecord:
    """Unified per-trajectory Phase-I record."""

    prompt_id: str
    dataset: str
    question: str
    image_path: str | None
    image_sha256: str | None
    ground_truth: Any
    student_support: StudentSupport
    trajectory_id: str
    trajectory_correct: bool | None
    trajectory_token_ids: tuple[int, ...]
    trajectory_token_hash: str
    trajectory_text: str
    teacher_trajectories: tuple[Mapping[str, Any], ...]
    token_signals: tuple[Mapping[str, Any], ...]
    candidate_windows: tuple[CandidateWindow, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        if not self.prompt_id or not self.question or not self.trajectory_id:
            raise ValueError("prompt_id, question, and trajectory_id are required")
        if not self.trajectory_token_ids:
            raise ValueError("trajectory_token_ids must be non-empty")
        if not self.trajectory_token_hash:
            raise ValueError("trajectory_token_hash must be non-empty")
        self.student_support.validate()
        for candidate in self.candidate_windows:
            candidate.validate(len(self.trajectory_token_ids))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def continuation_estimate(
    condition: str,
    correctness: Sequence[bool | None],
    *,
    seeds: Sequence[int] = (),
    response_hashes: Sequence[str] = (),
    reasons: Sequence[str] = (),
) -> ContinuationEstimate:
    if not correctness:
        raise ValueError("at least one continuation outcome is required")
    correct_count = sum(value is True for value in correctness)
    if reasons:
        if len(reasons) != len(correctness):
            raise ValueError("reasons length must equal correctness length")
        generated = 0
        n_wrong_format_valid = 0
        n_no_answer_marker = 0
        n_truncated = 0
        n_relay_answer_leakage = 0
        n_generation_error = 0
        for reason in reasons:
            if reason == "correct":
                continue
            if reason == "wrong_format_valid":
                n_wrong_format_valid += 1
            elif reason == "no_answer_marker":
                n_no_answer_marker += 1
            elif reason == "truncated":
                n_truncated += 1
            elif reason == "relay_answer_leakage":
                n_relay_answer_leakage += 1
            elif reason == "generation_error":
                n_generation_error += 1
            else:
                raise ValueError(f"unknown continuation reason: {reason!r}")
        generated = len(correctness) - n_relay_answer_leakage
    else:
        generated = 0
        n_wrong_format_valid = 0
        n_no_answer_marker = 0
        n_truncated = 0
        n_relay_answer_leakage = 0
        n_generation_error = 0
    estimate = ContinuationEstimate(
        condition=str(condition),
        n=len(correctness),
        n_correct=correct_count,
        n_malformed=sum(value is None for value in correctness),
        pass_rate=float(correct_count / len(correctness)),
        seeds=tuple(int(value) for value in seeds),
        response_hashes=tuple(str(value) for value in response_hashes),
        n_student_continuations_generated=generated,
        n_wrong_format_valid=n_wrong_format_valid,
        n_no_answer_marker=n_no_answer_marker,
        n_truncated=n_truncated,
        n_relay_answer_leakage=n_relay_answer_leakage,
        n_generation_error=n_generation_error,
    )
    estimate.validate()
    return estimate

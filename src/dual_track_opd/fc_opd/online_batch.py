"""Online FC-OPD batch assembly for on-policy training hooks.

This module is deliberately independent of ``third_party/verl``. It consumes
the current student rollout tokens produced inside a training step, immediately
asks teacher/student forced scorers for those same tokens, and returns loss-ready
FC-OPD tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from .chunk_parser import DecodeTokenizer, parse_response_chunks
from .conditions import Condition, ConditionInputs
from .four_condition_offline_builder import (
    build_verifier_learning_value_gate,
    compute_student_deficit_capability_scores,
    grouped_loss_plan,
)
from .loss import FCOPDLossConfig, compute_fc_opd_loss
from .router import RouterConfig, route_condition_weights
from .signal_decomposer import TeacherTopK, sampled_token_log_prob
from .verifier import verify_geometry3k_response


DEFAULT_ONLINE_CONDITIONS: tuple[Condition, ...] = (
    Condition.FULL,
    Condition.DEGRADED,
    Condition.FREE,
    Condition.TASK_VISIBLE,
    Condition.TASK_INFER,
    Condition.TASK_SOLVE,
)
STALE_OFFLINE_FIELD_NAMES = frozenset(
    {
        "offline_score_jsonl",
        "offline_score_path",
        "teacher_condition_scores",
        "condition_scores",
        "teacher_scores",
        "precomputed_teacher_scores",
        "precomputed_rollout",
        "score_jsonl_row",
    }
)


@dataclass(frozen=True)
class OnlineFCOPDConfig:
    conditions: tuple[Condition, ...] = DEFAULT_ONLINE_CONDITIONS
    router_config: RouterConfig = field(default_factory=lambda: RouterConfig(mode="student_deficit_chunk_gated"))
    loss_config: FCOPDLossConfig = field(default_factory=FCOPDLossConfig)
    capability_margin: float = 0.0
    max_capabilities_per_token: int = 2
    reject_stale_offline_fields: bool = True
    grouped_loss_schema: str = "capability_chunk_v1"
    compute_hook_loss: bool = True
    skip_routing: bool = False  # VA-OPD: skip chunk parsing, capability scores, student scoring


@dataclass(frozen=True)
class OnlineFCOPDSample:
    sample_uid: str
    question: str
    condition_inputs: ConditionInputs
    rollout_token_ids: tuple[int, ...]
    rollout_text: str
    prompt: Any | None = None
    images: Any | None = None
    choices: tuple[str, ...] = ()
    answer_metadata: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OnlineStudentScores:
    """Current-student forced scores for the current rollout."""

    loss_logits: torch.Tensor | None
    condition_log_probs: Mapping[Condition | str, Sequence[float] | torch.Tensor]


@dataclass(frozen=True)
class OnlineFCOPDSampleOutput:
    sample_uid: str
    response_token_ids: torch.Tensor
    teacher_scores: Mapping[Condition, TeacherTopK]
    student_scores: Mapping[str, dict[str, Any]]
    capability_scores: Mapping[str, dict[str, Any]]
    verifier: Mapping[str, Any]
    verifier_learning_value_gate: Mapping[str, Any]
    condition_weights: Mapping[Condition, torch.Tensor]
    chunk_masks: Mapping[str, torch.Tensor]
    grouped_loss_tensors: Mapping[str, torch.Tensor]
    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class OnlineFCOPDBatchOutput:
    samples: tuple[OnlineFCOPDSampleOutput, ...]
    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor]


class TeacherScorer(Protocol):
    def __call__(
        self,
        sample: OnlineFCOPDSample,
        conditions: Sequence[Condition],
    ) -> Mapping[Condition | str, TeacherTopK]: ...


class StudentForcedScorer(Protocol):
    def __call__(
        self,
        sample: OnlineFCOPDSample,
        conditions: Sequence[Condition],
    ) -> OnlineStudentScores: ...


VerifierFn = Callable[[OnlineFCOPDSample], Mapping[str, Any]]


def compute_online_fc_opd_batch(
    samples: Sequence[OnlineFCOPDSample],
    *,
    tokenizer: DecodeTokenizer,
    teacher_scorer: TeacherScorer,
    student_scorer: StudentForcedScorer,
    verifier: VerifierFn | None = None,
    config: OnlineFCOPDConfig | None = None,
    pre_scored_students: Sequence[OnlineStudentScores] | None = None,
) -> OnlineFCOPDBatchOutput:
    """Compute online FC-OPD losses for fresh runtime rollouts.

    The function never reads offline score JSONL data. Teacher and student
    scorers are called with ``sample.rollout_token_ids`` from the current
    training step.

    When *pre_scored_students* is provided (one per sample), the pre-batch
    student-scoring loop is skipped entirely — this is the preferred path
    for production because it mirrors how the teacher is pre-batched.
    """

    if not samples:
        raise ValueError("at least one online sample is required")
    config = config or OnlineFCOPDConfig()
    for sample in samples:
        _reject_stale_offline_fields(sample, config)
    if config.skip_routing:
        all_student_scores = [None] * len(samples)
    elif pre_scored_students is not None:
        all_student_scores = list(pre_scored_students)
    else:
        # Pre-batch student scoring — split into sub-batches of 8 to avoid OOM.
        _STUDENT_BATCH_MAX = 8
        all_student_scores = []
        for _start in range(0, len(samples), _STUDENT_BATCH_MAX):
            chunk = list(samples)[_start:_start + _STUDENT_BATCH_MAX]
            try:
                chunk_scores = student_scorer(chunk, config.conditions)
                all_student_scores.extend(chunk_scores if isinstance(chunk_scores, list) else [chunk_scores])
            except (AttributeError, TypeError, NotImplementedError, RuntimeError):
                import logging as _logging2
                _logging2.getLogger(__name__).warning(
                    "student scorer batch call failed, falling back to per-sample scoring",
                    exc_info=True,
                )
                all_student_scores = None
                break
    if all_student_scores is not None and len(all_student_scores) != len(samples):
        all_student_scores = None

    sample_outputs = [
        _compute_online_sample(
            sample,
            tokenizer=tokenizer,
            teacher_scorer=teacher_scorer,
            student_scorer=student_scorer,
            verifier=verifier,
            config=config,
            pre_scored_student=all_student_scores[idx] if all_student_scores else None,
        )
        for idx, sample in enumerate(samples)
    ]
    losses = [output.loss for output in sample_outputs]
    total_loss = torch.stack([loss.reshape(()) for loss in losses]).mean()
    metrics = _merge_metrics(sample_outputs, total_loss)
    return OnlineFCOPDBatchOutput(samples=tuple(sample_outputs), loss=total_loss, metrics=metrics)


def _compute_online_sample(
    sample: OnlineFCOPDSample,
    *,
    tokenizer: DecodeTokenizer,
    teacher_scorer: TeacherScorer,
    student_scorer: StudentForcedScorer,
    verifier: VerifierFn | None,
    config: OnlineFCOPDConfig,
    pre_scored_student: OnlineStudentScores | None = None,
) -> OnlineFCOPDSampleOutput:
    _reject_stale_offline_fields(sample, config)
    if not sample.rollout_token_ids:
        raise ValueError(f"{sample.sample_uid}: rollout_token_ids must be non-empty")

    response_token_ids = torch.tensor([sample.rollout_token_ids], dtype=torch.long)
    T = len(sample.rollout_token_ids)

    if config.skip_routing:
        # ── VA-OPD fast path: no XML parsing, no student scorer, no routing ──
        chunk_masks = {}
        response_mask = torch.ones_like(response_token_ids, dtype=torch.bool)
        teacher_scores = _normalize_teacher_scores(teacher_scorer(sample, config.conditions))
        _validate_teacher_scores(sample, teacher_scores, response_token_ids)
        target_device = response_token_ids.device
        teacher_scores = {c: _topk_to_device(s, target_device) for c, s in teacher_scores.items()}
        response_ids_device = response_token_ids.to(target_device)
        response_mask = response_mask.to(target_device)
        # Uniform condition weights: all response tokens weighted equally
        C = len(config.conditions)
        condition_weights = {
            Condition(c): torch.ones(1, T, device=target_device)
            for c in config.conditions
        }
        loss = torch.zeros((), dtype=torch.float32, device=target_device)
        metrics = {}
        verifier_result = {}
        verifier_gate = {}
        capability_scores = {}
    else:
        chunks = parse_response_chunks(sample.rollout_token_ids, sample.rollout_text, tokenizer)
        chunk_masks = {
            "visible_evidence": chunks.visible_evidence_mask.unsqueeze(0),
            "diagram_inference": chunks.diagram_inference_mask.unsqueeze(0),
            "reasoning": chunks.reasoning_mask.unsqueeze(0),
            "answer": chunks.answer_mask.unsqueeze(0),
        }
        response_mask = torch.ones_like(response_token_ids, dtype=torch.bool)
        format_valid = torch.tensor([chunks.format_valid], dtype=torch.bool)

        teacher_scores = _normalize_teacher_scores(teacher_scorer(sample, config.conditions))
        _validate_teacher_scores(sample, teacher_scores, response_token_ids)
        student = pre_scored_student if pre_scored_student is not None else student_scorer(sample, config.conditions)
        loss_logits = (
            _normalize_loss_logits(sample, student.loss_logits, len(sample.rollout_token_ids))
            if student.loss_logits is not None
            else None
        )
        if config.compute_hook_loss and loss_logits is None:
            raise ValueError(f"{sample.sample_uid}: loss_logits are required when compute_hook_loss=true")

        target_device = loss_logits.device if loss_logits is not None else response_token_ids.device
        teacher_scores = {c: _topk_to_device(s, target_device) for c, s in teacher_scores.items()}
        response_ids_device = response_token_ids.to(target_device)
        response_mask = response_mask.to(target_device)
        chunk_masks = {name: mask.to(target_device) for name, mask in chunk_masks.items()}

        verifier_result = dict(verifier(sample) if verifier is not None else _default_verifier(sample))
        verifier_gate = build_verifier_learning_value_gate(verifier_result)
        capability_scores = compute_student_deficit_capability_scores(
            teacher_condition_scores=_actual_logprob_blocks_from_teacher(teacher_scores, response_ids_device),
            student_condition_scores=_student_logprob_blocks(student.condition_log_probs, len(sample.rollout_token_ids)),
            chunk_spans=_chunk_spans_for_deficit(chunks),
            verifier_learning_value_gate=verifier_gate,
            margin=config.capability_margin,
            max_capabilities_per_token=config.max_capabilities_per_token,
            enable_student_deficit_gate=True,
        )
        condition_weights = route_condition_weights(
            signals={"capability_scores": capability_scores},
            chunk_masks=chunk_masks,
            router_config=config.router_config,
            response_mask=response_mask,
            available_conditions=tuple(teacher_scores),
            format_valid=format_valid.to(target_device),
        )
        condition_weights = _apply_verifier_token_gate(
            condition_weights=condition_weights,
            chunk_masks=chunk_masks,
            verifier_learning_value_gate=verifier_gate,
            response_mask=response_mask,
        )
        if loss_logits is None:
            loss = torch.zeros((), dtype=torch.float32, device=target_device)
            metrics = {}
        else:
            loss, metrics = compute_fc_opd_loss(
                loss_logits,
                teacher_scores,
                chunk_masks,
                condition_weights,
                response_mask,
                config.loss_config,
            )
    grouped = _grouped_loss_tensors(
        capability_scores=capability_scores,
        response_mask=response_mask,
        condition_weights=condition_weights,
        schema=config.grouped_loss_schema,
    )
    if config.skip_routing:
        student_scores = {}
    else:
        student_scores = _student_logprob_blocks(student.condition_log_probs, len(sample.rollout_token_ids))
    return OnlineFCOPDSampleOutput(
        sample_uid=sample.sample_uid,
        response_token_ids=response_ids_device,
        teacher_scores=teacher_scores,
        student_scores=student_scores,
        capability_scores=capability_scores,
        verifier=verifier_result,
        verifier_learning_value_gate=verifier_gate,
        condition_weights=condition_weights,
        chunk_masks=chunk_masks,
        grouped_loss_tensors=grouped,
        loss=loss,
        metrics=metrics,
    )


def _reject_stale_offline_fields(sample: OnlineFCOPDSample, config: OnlineFCOPDConfig) -> None:
    if not config.reject_stale_offline_fields:
        return
    stale = sorted(STALE_OFFLINE_FIELD_NAMES & set(sample.metadata))
    if stale:
        raise ValueError(
            f"{sample.sample_uid}: online FC-OPD must not consume stale offline fields: {', '.join(stale)}"
        )


def _normalize_teacher_scores(scores: Mapping[Condition | str, TeacherTopK]) -> dict[Condition, TeacherTopK]:
    return {Condition(condition): score for condition, score in scores.items()}


def _validate_teacher_scores(
    sample: OnlineFCOPDSample,
    teacher_scores: Mapping[Condition, TeacherTopK],
    response_token_ids: torch.Tensor,
) -> None:
    """Validate and align teacher score shapes to rollout length.

    Qwen3-VL 32B and 4B tokenizers differ slightly; the teacher may produce
    slightly more or fewer tokens than the student rollout.  Instead of
    crashing we trim (teacher longer) or pad (teacher shorter) and emit a
    warning so we can track how often this occurs.
    """
    import logging as _logging

    _logger = _logging.getLogger(__name__)
    if not teacher_scores:
        raise ValueError(f"{sample.sample_uid}: teacher_scorer returned no conditions")
    for condition, score in list(teacher_scores.items()):
        score.validate()
        score_len = score.token_ids.shape[1]
        rollout_len = response_token_ids.shape[1]
        if score_len == rollout_len:
            continue
        # Align to the rollout length.
        if score_len > rollout_len:
            _logger.warning(
                "%s/%s: trimming teacher scores (%d → %d tokens).  "
                "This is expected to be rare (<1%% of steps).",
                sample.sample_uid, condition.value, score_len, rollout_len,
            )
            teacher_scores[condition] = _slice_teacher_topk(score, rollout_len)
        else:
            _logger.warning(
                "%s/%s: padding teacher scores (%d → %d tokens).  "
                "Positions beyond teacher length get zero quality weight.",
                sample.sample_uid, condition.value, score_len, rollout_len,
            )
            teacher_scores[condition] = _pad_teacher_topk(score, rollout_len)


def _slice_teacher_topk(score: TeacherTopK, target_len: int) -> TeacherTopK:
    """Trim teacher scores to exactly *target_len* response positions."""
    return TeacherTopK(
        token_ids=score.token_ids[:, :target_len, :],
        log_probs=score.log_probs[:, :target_len, :],
        tail_log_prob=score.tail_log_prob[:, :target_len] if score.tail_log_prob is not None else None,
        entropy=score.entropy[:, :target_len] if score.entropy is not None else None,
        sampled_log_probs=(
            score.sampled_log_probs[:, :target_len] if score.sampled_log_probs is not None else None
        ),
    )


def _pad_teacher_topk(score: TeacherTopK, target_len: int) -> TeacherTopK:
    """Pad teacher scores to *target_len* with neutral (zero-quality) values."""
    import torch

    cur_len = score.token_ids.shape[1]
    pad_len = target_len - cur_len
    device = score.token_ids.device
    _k = score.token_ids.shape[-1]
    # top-K token IDs: zero-filled (will not receive quality weight)
    pad_ids = torch.zeros(1, pad_len, _k, dtype=score.token_ids.dtype, device=device)
    # top-K log_probs: large negative → quality ≈ 0
    pad_log = torch.full((1, pad_len, _k), -1e10, dtype=score.log_probs.dtype, device=device)
    pad_tail = (
        torch.zeros(1, pad_len, dtype=score.tail_log_prob.dtype, device=device)
        if score.tail_log_prob is not None else None
    )
    pad_ent = (
        torch.zeros(1, pad_len, dtype=score.entropy.dtype, device=device)
        if score.entropy is not None else None
    )
    pad_slp = (
        torch.full((1, pad_len), -30.0, dtype=score.sampled_log_probs.dtype, device=device)
        if score.sampled_log_probs is not None else None
    )
    return TeacherTopK(
        token_ids=torch.cat([score.token_ids, pad_ids], dim=1),
        log_probs=torch.cat([score.log_probs, pad_log], dim=1),
        tail_log_prob=torch.cat([score.tail_log_prob, pad_tail], dim=1) if pad_tail is not None else None,
        entropy=torch.cat([score.entropy, pad_ent], dim=1) if pad_ent is not None else None,
        sampled_log_probs=(
            torch.cat([score.sampled_log_probs, pad_slp], dim=1)
            if score.sampled_log_probs is not None else None
        ),
    )


def _normalize_loss_logits(sample: OnlineFCOPDSample, logits: torch.Tensor, seq_len: int) -> torch.Tensor:
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != seq_len:
        raise ValueError(f"{sample.sample_uid}: loss_logits must have shape [1, {seq_len}, vocab]")
    if not torch.isfinite(logits).all():
        raise ValueError(f"{sample.sample_uid}: loss_logits contains NaN or Inf")
    return logits


def _topk_to_device(score: TeacherTopK, device: torch.device) -> TeacherTopK:
    return TeacherTopK(
        token_ids=score.token_ids.to(device),
        log_probs=score.log_probs.to(device),
        tail_log_prob=None if score.tail_log_prob is None else score.tail_log_prob.to(device),
        entropy=None if score.entropy is None else score.entropy.to(device),
    )


def _actual_logprob_blocks_from_teacher(
    scores: Mapping[Condition, TeacherTopK],
    sampled_token_ids: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {}
    for condition, score in scores.items():
        actual = sampled_token_log_prob(score, sampled_token_ids).detach().cpu().reshape(-1).tolist()
        blocks[condition.value] = {
            "actual_token_log_probs": [float(value) for value in actual],
            "token_ids": score.token_ids.detach().cpu()[0].tolist(),
            "log_probs": score.log_probs.detach().cpu()[0].tolist(),
        }
    return blocks


def _student_logprob_blocks(
    condition_log_probs: Mapping[Condition | str, Sequence[float] | torch.Tensor],
    seq_len: int,
) -> dict[str, dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {}
    for condition, values in condition_log_probs.items():
        tensor = values.detach().cpu().float().reshape(-1) if isinstance(values, torch.Tensor) else torch.tensor(list(values), dtype=torch.float32)
        if tensor.numel() != seq_len:
            raise ValueError(f"{Condition(condition).value}: student log-prob length {tensor.numel()} != {seq_len}")
        blocks[Condition(condition).value] = {"actual_token_log_probs": [float(value) for value in tensor.tolist()]}
    return blocks


def _chunk_spans_for_deficit(chunks: Any) -> dict[str, Any]:
    spans: dict[str, Any] = {"chunk_labels": tuple(chunks.chunk_labels)}
    for name, span in chunks.chunk_spans.items():
        spans[name] = [tuple(span)]
    return spans


def _default_verifier(sample: OnlineFCOPDSample) -> Mapping[str, Any]:
    if sample.choices or sample.answer_metadata is not None:
        return verify_geometry3k_response(
            question=sample.question,
            choices=sample.choices,
            response_text=sample.rollout_text,
            answer_metadata=sample.answer_metadata,
        )
    return {
        "answer_extracted": None,
        "gold_answer": None,
        "correct": None,
        "format_valid": False,
        "malformed": True,
        "reward": 0.0,
        "verifier_source": "online_default_malformed",
    }


def _grouped_loss_tensors(
    *,
    capability_scores: Mapping[str, Mapping[str, Any]],
    response_mask: torch.Tensor,
    condition_weights: Mapping[Condition, torch.Tensor],
    schema: str,
) -> dict[str, torch.Tensor]:
    plan = grouped_loss_plan(schema)
    if not plan:
        return {}
    device = response_mask.device
    condition_sum = sum(weight.detach().float() for weight in condition_weights.values())
    grouped: dict[str, torch.Tensor] = {}
    for group, capabilities in plan.items():
        values = torch.zeros_like(response_mask, dtype=torch.float32, device=device)
        for capability in capabilities:
            block = capability_scores.get(capability)
            if not isinstance(block, Mapping):
                continue
            raw = block.get("final_token_weight", [])
            if len(raw) != response_mask.shape[1]:
                continue
            values = values + torch.tensor(raw, dtype=torch.float32, device=device).reshape(1, -1)
        grouped[group] = values * condition_sum * response_mask.float()
    return grouped


def _apply_verifier_token_gate(
    *,
    condition_weights: Mapping[Condition, torch.Tensor],
    chunk_masks: Mapping[str, torch.Tensor],
    verifier_learning_value_gate: Mapping[str, Any],
    response_mask: torch.Tensor,
) -> dict[Condition, torch.Tensor]:
    chunk_gates = verifier_learning_value_gate.get("chunk_gates", {})
    gate = torch.zeros_like(response_mask, dtype=torch.float32)
    for chunk_name, mask in chunk_masks.items():
        gate = torch.where(
            mask.bool(),
            torch.full_like(gate, float(chunk_gates.get(chunk_name, 1.0))),
            gate,
        )
    gate = gate * response_mask.float()
    return {condition: weight.float() * gate for condition, weight in condition_weights.items()}


def _merge_metrics(
    sample_outputs: Sequence[OnlineFCOPDSampleOutput],
    total_loss: torch.Tensor,
) -> dict[str, torch.Tensor]:
    merged: dict[str, list[torch.Tensor]] = {}
    for output in sample_outputs:
        for key, value in output.metrics.items():
            merged.setdefault(key, []).append(value.detach().reshape(()))
    metrics = {key: torch.stack(values).mean() for key, values in merged.items()}
    metrics["online/loss"] = total_loss.detach()
    metrics["online/num_samples"] = torch.tensor(float(len(sample_outputs)))
    return metrics

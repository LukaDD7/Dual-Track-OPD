"""Diagnostic-only clean-data 4C FC-OPD offline score builder.

This builder is useful for prompt, verifier, scorer, and row-level diagnostics.
It must not be treated as the final FC-OPD training data path because its
rollouts and teacher scores are precomputed and become stale after the student
policy updates.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import torch

from .alignment import compute_alignment_weights, compute_rollout_group_stats, default_outcome_metadata
from .chunk_parser import parse_response_chunks
from .conditions import (
    CAPABILITY_CONTRASTS,
    CHUNK_CAPABILITY_COMPATIBILITY,
    CONDITION_SETS as CONDITION_SET_NAMES,
    Condition,
    ConditionInputs,
    ImageInput,
)
from .dataset_signal_audit import compute_pairwise_kd_gradient_cosines, hash_text, hash_token_ids, materialize_gaussian_blur
from .degradation import DEGRADED_MODES, degraded_transform, materialize_degraded_image as materialize_degraded_image_file
from .evidence_generation import validate_evidence_row
from .geometry3k_adapter import default_degraded_image_dir, load_geometry3k_records
from .offline_loss import offline_record_to_tensors
from .offline_scoring import DEFAULT_TEACHER_URL, _serialize_chunks, _serialize_topk
from .signal_decomposer import compute_condition_signals
from .signal_decomposer import sampled_token_log_prob
from .student_rollout_signal_audit import (
    DEFAULT_TEACHER_URL as _DEFAULT_TEACHER_URL,
    HFQwenStudentRolloutGenerator,
    ROLLOUT_RESPONSE_FORMATS,
    StudentRolloutAuditConfig,
    StudentRolloutGenerator,
    build_rollout_prompt,
    response_format_note,
    rollout_seed,
)
from .teacher_client import TeacherClient, score_teacher_conditions
from .teacher_protocol import tokenizer_fingerprint
from .verifier import verify_geometry3k_response

CONDITION_SETS: dict[str, tuple[Condition, ...]] = {
    name: tuple(Condition(item) for item in values) for name, values in CONDITION_SET_NAMES.items()
}
FOUR_CLEAN_CONDITIONS: tuple[Condition, ...] = (
    Condition.FULL,
    Condition.DEGRADED,
    Condition.FREE,
    Condition.TASK,
)
DEFAULT_STUDENT_MODEL = "hf:$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct"
VERIFIER_GATE_RESIDUAL_FLOOR = 0.05
VERIFIER_GATE_CHUNK_MAX: dict[str, float] = {
    "visible_evidence": 1.00,
    "diagram_inference": 0.80,
    "reasoning": 0.60,
    "answer": 0.40,
}
CHUNK_CAPABILITY_COMPATIBILITY_FLOOR = 0.05
VERIFIER_LEARNING_VALUE_CHUNK_GATES: dict[str, dict[str, float]] = {
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
class FourConditionOfflineBuilderConfig:
    dataset: Path
    evidence_cache: Path
    output_jsonl: Path
    summary_json: Path
    dataset_type: str = "geometry3k"
    source_dataset: str = "geometry3k"
    student_model_path: str = DEFAULT_STUDENT_MODEL
    teacher_url: str = DEFAULT_TEACHER_URL
    limit: int | None = None
    start_index: int = 0
    end_index: int | None = None
    rollouts_per_prompt: int = 4
    rollout_response_format: str = "fc_opd_structured"
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    degraded_mode: str = "lowres_10pct_nearest"
    degraded_dir: str | None = None
    blur_sigma: float = 2.0
    resume: bool = False
    skip_existing: bool = False
    condition_set: str = "4c-legacy"
    enable_student_condition_scoring: bool = False
    student_deficit_gate: bool = False
    verifier_gate: str = "none"
    strict_condition_validation: bool = False
    capability_margin: float = 0.0
    max_capabilities_per_token: int = 2
    routing_mode: str = "chunk_gated_contrastive"
    grouped_loss_schema: str = "none"
    allow_empty_output: bool = False
    debug_first_n: int = 0
    max_rollout_attempts: int = 3

    def __post_init__(self) -> None:
        if self.dataset_type != "geometry3k":
            raise ValueError("clean-data 4C builder currently supports Geometry3K first")
        if self.rollout_response_format not in ROLLOUT_RESPONSE_FORMATS:
            raise ValueError(f"rollout_response_format must be one of {ROLLOUT_RESPONSE_FORMATS}")
        if self.degraded_mode not in DEGRADED_MODES:
            raise ValueError(f"degraded_mode must be one of {DEGRADED_MODES}")
        if self.condition_set not in CONDITION_SETS:
            raise ValueError(f"condition_set must be one of {sorted(CONDITION_SETS)}")
        if self.verifier_gate not in {"none", "geometry3k_verifier"}:
            raise ValueError("verifier_gate must be none or geometry3k_verifier")
        if self.max_capabilities_per_token < 1:
            raise ValueError("max_capabilities_per_token must be at least 1")
        if self.debug_first_n < 0:
            raise ValueError("debug_first_n must be non-negative")
        if self.max_rollout_attempts <= 0:
            raise ValueError("max_rollout_attempts must be positive")


@dataclass
class FourConditionOfflineBuilderResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class OfflineBuilderDiagnostics:
    selected_records: int = 0
    evidence_rows_loaded: int = 0
    evidence_rows_matched: int = 0
    rollout_attempt_count: int = 0
    rollout_success_count: int = 0
    rollout_empty_count: int = 0
    rollout_exception_count: int = 0
    chunk_parse_failure_count: int = 0
    teacher_score_exception_count: int = 0
    student_score_exception_count: int = 0
    verifier_exception_count: int = 0
    row_write_count: int = 0
    rollout_exhausted_count: int = 0
    rollout_retry_count: int = 0
    evidence_exhausted_records: int = 0
    skipped_records_by_reason: Counter[str] = field(default_factory=Counter)
    skipped_examples_top20: list[dict[str, str]] = field(default_factory=list)

    def add_skip(self, *, sample_uid: object, stage: str, reason: str) -> None:
        reason_text = _compact_reason(reason)
        self.skipped_records_by_reason[f"{stage}:{reason_text}"] += 1
        if len(self.skipped_examples_top20) < 20:
            self.skipped_examples_top20.append(
                {
                    "sample_uid": str(sample_uid),
                    "stage": stage,
                    "reason": reason_text,
                }
            )

    def to_summary(self) -> dict[str, Any]:
        return {
            "selected_records": self.selected_records,
            "evidence_rows_loaded": self.evidence_rows_loaded,
            "evidence_rows_matched": self.evidence_rows_matched,
            "rollout_attempt_count": self.rollout_attempt_count,
            "rollout_success_count": self.rollout_success_count,
            "rollout_empty_count": self.rollout_empty_count,
            "rollout_exception_count": self.rollout_exception_count,
            "chunk_parse_failure_count": self.chunk_parse_failure_count,
            "teacher_score_exception_count": self.teacher_score_exception_count,
            "student_score_exception_count": self.student_score_exception_count,
            "verifier_exception_count": self.verifier_exception_count,
            "row_write_count": self.row_write_count,
            "rollout_exhausted_count": self.rollout_exhausted_count,
            "rollout_retry_count": self.rollout_retry_count,
            "evidence_exhausted_records": self.evidence_exhausted_records,
            "skipped_records_by_reason": dict(self.skipped_records_by_reason),
            "skipped_examples_top20": list(self.skipped_examples_top20),
        }


def run_four_condition_offline_builder(
    config: FourConditionOfflineBuilderConfig,
    *,
    rollout_generator: StudentRolloutGenerator | None = None,
    teacher_client: TeacherClient | None = None,
) -> FourConditionOfflineBuilderResult:
    if config.skip_existing and config.output_jsonl.is_file() and config.summary_json.is_file():
        return FourConditionOfflineBuilderResult(
            rows=_read_jsonl(config.output_jsonl),
            summary=json.loads(config.summary_json.read_text(encoding="utf-8")),
        )
    records = _select(load_geometry3k_records(config.dataset, source_dataset=config.source_dataset), config)
    evidence_rows = _read_jsonl(config.evidence_cache)
    evidence = {str(row["sample_uid"]): row for row in evidence_rows}
    diagnostics = OfflineBuilderDiagnostics(
        selected_records=len(records),
        evidence_rows_loaded=len(evidence_rows),
    )
    rollout_generator = rollout_generator or HFQwenStudentRolloutGenerator(_rollout_config(config))
    tokenizer_hash = tokenizer_fingerprint(rollout_generator.tokenizer)
    teacher_client = teacher_client or TeacherClient(config.teacher_url, expected_tokenizer_hash=tokenizer_hash)
    existing = _read_jsonl(config.output_jsonl) if config.resume else []
    existing_uids = {str(row.get("rollout_uid", "")) for row in existing}
    new_rows: list[dict[str, Any]] = []
    mode = "a" if config.resume and config.output_jsonl.is_file() else "w"
    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.output_jsonl.open(mode, encoding="utf-8") as handle:
        for record_index, record in enumerate(records):
            sample_uid = str(record.get("sample_uid", "unknown"))
            _debug(config, record_index, f"selected record uid={sample_uid}")
            evidence_row = evidence.get(str(record["sample_uid"]))
            if evidence_row is None:
                diagnostics.add_skip(
                    sample_uid=sample_uid,
                    stage="evidence_match",
                    reason="missing evidence cache row",
                )
                _debug(config, record_index, "evidence matched=false")
                continue
            diagnostics.evidence_rows_matched += 1
            _debug(config, record_index, "evidence matched=true")
            if evidence_row.get("task_infer_exhausted_retries"):
                diagnostics.evidence_exhausted_records += 1
            errors = validate_evidence_row(evidence_row)
            if config.strict_condition_validation and evidence_row.get("task_infer_class") in {"solve_like", "unusable"}:
                errors.append(f"task_infer_{evidence_row.get('task_infer_class')}")
            if errors:
                diagnostics.add_skip(
                    sample_uid=sample_uid,
                    stage="evidence_validation",
                    reason="; ".join(str(error) for error in errors[:3]),
                )
                continue
            try:
                built_rows = _build_rows(
                    record,
                    evidence_row,
                    config,
                    rollout_generator,
                    teacher_client,
                    tokenizer_hash,
                    diagnostics=diagnostics,
                    record_index=record_index,
                )
            except Exception as exc:  # noqa: BLE001
                diagnostics.add_skip(
                    sample_uid=sample_uid,
                    stage="record_setup",
                    reason=_exception_reason(exc),
                )
                _debug(config, record_index, f"record setup exception reason={_exception_reason(exc)}")
                continue
            for row in built_rows:
                if row["rollout_uid"] in existing_uids:
                    diagnostics.add_skip(
                        sample_uid=row["rollout_uid"],
                        stage="resume",
                        reason="rollout already exists",
                    )
                    continue
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                new_rows.append(row)
                diagnostics.row_write_count += 1
                _debug(config, record_index, f"row written rollout_uid={row['rollout_uid']}")
    rows = [*existing, *new_rows]
    _attach_group_stats(rows)
    if rows:
        config.output_jsonl.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
    summary = summarize_rows(rows, config=config, teacher_client=teacher_client, diagnostics=diagnostics)
    config.summary_json.parent.mkdir(parents=True, exist_ok=True)
    config.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if diagnostics.selected_records > 0 and diagnostics.row_write_count == 0 and not config.allow_empty_output:
        raise RuntimeError(
            "offline builder selected records but wrote zero rows; "
            f"summary written to {config.summary_json}"
        )
    return FourConditionOfflineBuilderResult(rows=rows, summary=summary)


def _build_rows(
    record: Mapping[str, Any],
    evidence_row: Mapping[str, Any],
    config: FourConditionOfflineBuilderConfig,
    rollout_generator: StudentRolloutGenerator,
    teacher_client: TeacherClient,
    tokenizer_hash: str,
    *,
    diagnostics: OfflineBuilderDiagnostics,
    record_index: int,
) -> list[dict[str, Any]]:
    image_path = str(record["image_path"])
    degraded_path = materialize_degraded_image(image_path, config)
    condition_inputs = ConditionInputs(
        full_image=ImageInput(path=image_path),
        degraded_image=ImageInput(path=degraded_path, transform=_degraded_transform(config)),
        free_caption=str(evidence_row["free_caption"]),
        task_evidence=str(evidence_row.get("task_evidence") or evidence_row.get("task_visible_evidence")),
        task_visible_evidence=str(evidence_row.get("task_visible_evidence") or evidence_row["task_evidence"]),
        task_infer_evidence=(
            None if evidence_row.get("task_infer_evidence") is None else str(evidence_row.get("task_infer_evidence"))
        ),
        task_solve_evidence=(
            None if evidence_row.get("task_solve_evidence") is None else str(evidence_row.get("task_solve_evidence"))
        ),
    )
    condition_inputs.validate(require_paths=False)
    question = str(record["question"])
    prompt = build_rollout_prompt(question, response_format=config.rollout_response_format)
    source_index = int(record["source_index"])
    group_uid = f"{record['sample_uid']}:rollout-group"
    rows: list[dict[str, Any]] = []
    conditions = CONDITION_SETS[config.condition_set]
    for rollout_id in range(config.rollouts_per_prompt):
        rollout_uid = f"{record['sample_uid']}:rollout-{rollout_id}"
        response_text = ""
        token_ids: tuple[int, ...] = ()
        chunk_masks = None
        seed = rollout_seed(base_seed=config.seed, source_index=source_index, rollout_id=rollout_id)
        rollout_fallback = False
        for attempt in range(config.max_rollout_attempts):
            attempt_seed = seed + attempt * 100_000
            diagnostics.rollout_attempt_count += 1
            if attempt > 0:
                diagnostics.rollout_retry_count += 1
            try:
                candidate_text = rollout_generator.generate(
                    question=question,
                    image_path=image_path,
                    prompt_text=prompt.text,
                    seed=attempt_seed,
                )
            except Exception as exc:  # noqa: BLE001
                diagnostics.rollout_exception_count += 1
                diagnostics.add_skip(
                    sample_uid=rollout_uid,
                    stage="student_rollout",
                    reason=f"attempt={attempt + 1}: {_exception_reason(exc)}",
                )
                _debug(config, record_index, f"rollout exception rollout_id={rollout_id} attempt={attempt + 1} reason={_exception_reason(exc)}")
                continue
            _debug(config, record_index, f"rollout text length={len(candidate_text)} rollout_id={rollout_id} attempt={attempt + 1}")
            if not candidate_text.strip():
                diagnostics.rollout_empty_count += 1
                diagnostics.add_skip(
                    sample_uid=rollout_uid,
                    stage="student_rollout",
                    reason=f"attempt={attempt + 1}: empty rollout text",
                )
                continue
            candidate_token_ids = tuple(int(item) for item in rollout_generator.tokenizer.encode(candidate_text))
            candidate_chunks = parse_response_chunks(
                candidate_token_ids,
                candidate_text,
                rollout_generator.tokenizer,
                fallback="all_reasoning",
            )
            if not candidate_chunks.format_valid:
                diagnostics.chunk_parse_failure_count += 1
                diagnostics.add_skip(
                    sample_uid=rollout_uid,
                    stage="chunk_parse",
                    reason=f"attempt={attempt + 1}: {'; '.join(candidate_chunks.errors[:3])}",
                )
                _debug(config, record_index, f"chunk parse valid=false rollout_id={rollout_id} attempt={attempt + 1}")
                continue
            response_text = candidate_text
            token_ids = candidate_token_ids
            chunk_masks = candidate_chunks
            seed = attempt_seed
            diagnostics.rollout_success_count += 1
            _debug(config, record_index, f"chunk parse valid=true rollout_id={rollout_id} attempt={attempt + 1}")
            break
        if chunk_masks is None:
            diagnostics.rollout_exhausted_count += 1
            diagnostics.add_skip(
                sample_uid=rollout_uid,
                stage="rollout_exhausted",
                reason=f"exhausted {config.max_rollout_attempts} attempts; writing malformed fallback row",
            )
            response_text = _fallback_malformed_response(rollout_uid)
            token_ids = tuple(int(item) for item in rollout_generator.tokenizer.encode(response_text))
            chunk_masks = parse_response_chunks(token_ids, response_text, rollout_generator.tokenizer, fallback="all_reasoning")
            if not chunk_masks.format_valid:
                diagnostics.chunk_parse_failure_count += 1
            rollout_fallback = True
            _debug(config, record_index, f"rollout exhausted; fallback row=true rollout_id={rollout_id}")
        try:
            teacher_scores = score_teacher_conditions(
                token_ids,
                question,
                condition_inputs,
                conditions,
                teacher_client,
                response_text=response_text,
                request_prefix=rollout_uid,
            )
        except Exception as exc:  # noqa: BLE001
            diagnostics.teacher_score_exception_count += 1
            diagnostics.add_skip(
                sample_uid=rollout_uid,
                stage="teacher_score",
                reason=_exception_reason(exc),
            )
            _debug(config, record_index, f"teacher score ok=false rollout_id={rollout_id} reason={_exception_reason(exc)}")
            continue
        _debug(config, record_index, f"teacher score ok=true rollout_id={rollout_id}")
        sampled_ids = torch.tensor([list(token_ids)], dtype=torch.int64)
        signals = compute_condition_signals(teacher_scores, sampled_token_ids=sampled_ids)
        condition_scores = {
            condition.value: _serialize_condition_score(
                teacher_scores[condition],
                token_ids,
                top_k=teacher_client.metadata.top_k,
            )
            for condition in conditions
        }
        try:
            student_scores_raw = _score_student_conditions(
                rollout_generator=rollout_generator,
                token_ids=token_ids,
                question=question,
                condition_inputs=condition_inputs,
                conditions=conditions,
                response_text=response_text,
                top_k=teacher_client.metadata.top_k,
                enabled=config.enable_student_condition_scoring,
            )
        except Exception as exc:  # noqa: BLE001
            diagnostics.student_score_exception_count += 1
            diagnostics.add_skip(
                sample_uid=rollout_uid,
                stage="student_score",
                reason=_exception_reason(exc),
            )
            _debug(config, record_index, f"student score ok=false rollout_id={rollout_id} reason={_exception_reason(exc)}")
            continue
        _debug(config, record_index, f"student score ok=true rollout_id={rollout_id}")
        student_score_source = (
            student_scores_raw
            if student_scores_raw
            else teacher_scores
            if config.enable_student_condition_scoring
            else {}
        )
        student_condition_scores = {
            condition.value: _serialize_condition_score(
                student_score_source[condition],
                token_ids,
                top_k=teacher_client.metadata.top_k,
                actual_override=(
                    [0.0 for _ in token_ids]
                    if config.enable_student_condition_scoring and not _has_student_condition_scorer(rollout_generator)
                    else None
                ),
            )
            for condition in conditions
            if condition in student_score_source
        }
        try:
            verifier = (
                verify_geometry3k_response(
                    question=question,
                    choices=list(record.get("choices", [])),
                    response_text=response_text,
                    answer_metadata=record.get("answer_metadata") or record.get("answer") or record.get("gold"),
                )
                if config.verifier_gate == "geometry3k_verifier"
                else None
            )
        except Exception as exc:  # noqa: BLE001
            diagnostics.verifier_exception_count += 1
            diagnostics.add_skip(
                sample_uid=rollout_uid,
                stage="verifier",
                reason=_exception_reason(exc),
            )
            _debug(config, record_index, f"verifier outcome=exception rollout_id={rollout_id} reason={_exception_reason(exc)}")
            continue
        _debug(config, record_index, f"verifier outcome={verifier_outcome_class(verifier)} rollout_id={rollout_id}")
        verifier_learning_value_gate = build_verifier_learning_value_gate(verifier)
        capability_scores = compute_student_deficit_capability_scores(
            teacher_condition_scores=condition_scores,
            student_condition_scores=student_condition_scores,
            chunk_spans=_serialize_chunks(chunk_masks),
            verifier_learning_value_gate=verifier_learning_value_gate,
            margin=config.capability_margin,
            max_capabilities_per_token=config.max_capabilities_per_token,
            task_infer_solve_like=bool(evidence_row.get("task_infer_solve_like")),
            enable_student_deficit_gate=config.student_deficit_gate,
        )
        outcome = default_outcome_metadata(record)
        if verifier is not None:
            outcome = {
                **outcome,
                "verifier": verifier,
                "correct": verifier["correct"],
                "format_valid": verifier["format_valid"],
                "reward": verifier["reward"],
            }
        row = {
            "sample_uid": rollout_uid,
            "prompt_sample_uid": str(record["sample_uid"]),
            "rollout_group_uid": group_uid,
            "sibling_rollout_ids": [idx for idx in range(config.rollouts_per_prompt) if idx != rollout_id],
            "source_dataset": config.source_dataset,
            "source_index": source_index,
            "rollout_id": rollout_id,
            "rollout_uid": rollout_uid,
            "question": question,
            "clean_question_text": str(record.get("clean_question_text", question)),
            "choices": list(record.get("choices", [])),
            "original_image_path": image_path,
            "image_path": image_path,
            "degraded_image_path": degraded_path,
            "degraded_mode": config.degraded_mode,
            "free_caption": evidence_row["free_caption"],
            "task_evidence": evidence_row["task_evidence"],
            "task_visible_evidence": evidence_row.get("task_visible_evidence", evidence_row["task_evidence"]),
            "task_infer_evidence": evidence_row.get("task_infer_evidence"),
            "task_solve_evidence": evidence_row.get("task_solve_evidence"),
            "task_infer_class": evidence_row.get("task_infer_class"),
            "task_infer_solve_like": bool(evidence_row.get("task_infer_solve_like")),
            "condition_validation_warnings": list(evidence_row.get("condition_validation_warnings", [])),
            "evidence_cache_uid": evidence_row["sample_uid"],
            "response_source": "fallback_malformed_rollout" if rollout_fallback else "student_rollout",
            "rollout_fallback_malformed": rollout_fallback,
            "rollout_attempts": config.max_rollout_attempts if rollout_fallback else max(1, (seed - rollout_seed(base_seed=config.seed, source_index=source_index, rollout_id=rollout_id)) // 100_000 + 1),
            "response_text": response_text,
            "response_token_ids": list(token_ids),
            "response_token_count": len(token_ids),
            "response_text_hash": hash_text(response_text),
            "response_token_hash": hash_token_ids(token_ids),
            "tokenizer_hash": tokenizer_hash,
            "prompt_hash": hash_text(question),
            "rollout_prompt": prompt.text,
            "rollout_prompt_hash": hash_text(prompt.text),
            "generation_seed": seed,
            "student_generation_metadata": {
                "student_model_path": config.student_model_path,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_new_tokens": config.max_new_tokens,
                "device": config.device,
                "dtype": config.dtype,
                "rollout_response_format": config.rollout_response_format,
                "rollout_response_format_note": response_format_note(config.rollout_response_format),
            },
            "teacher_metadata": {
                "teacher_url": config.teacher_url,
                "teacher_model_id": teacher_client.metadata.model_id,
                "tokenizer_hash": teacher_client.metadata.tokenizer_hash,
                "top_k": teacher_client.metadata.top_k,
                "protocol_version": teacher_client.metadata.protocol_version,
                "git_revision": teacher_client.metadata.git_revision,
            },
            "evidence_generation_metadata": evidence_row.get("generator_metadata", {}),
            "condition_set_name": config.condition_set,
            "conditions": [condition.value for condition in conditions],
            "condition_inputs": condition_inputs.to_dict(),
            "condition_scores": condition_scores,
            "student_condition_scores": student_condition_scores,
            "capability_contrasts": {
                name: {"positive": positive.value, "negative": negative.value}
                for name, (positive, negative) in CAPABILITY_CONTRASTS.items()
            },
            "capability_scores": capability_scores,
            "verifier": verifier,
            "verifier_learning_value_gate": verifier_learning_value_gate,
            "grouped_loss_plan": grouped_loss_plan(config.grouped_loss_schema),
            "routing_mode": config.routing_mode,
            "grouped_loss_schema": config.grouped_loss_schema,
            "condition_signals": {name: [float(item) for item in signal[0].tolist()] for name, signal in signals.items()},
            "condition_signal_summary": {name: _stats([float(item) for item in signal[0].tolist()]) for name, signal in signals.items()},
            "gradient_cosines": compute_pairwise_kd_gradient_cosines(teacher_scores),
            "chunk_spans": _serialize_chunks(chunk_masks),
            "outcome_metadata": outcome,
            "alignment_targets": compute_alignment_weights({"response_token_ids": token_ids}, strategy="none").to_record(),
            "success_group_stats": None,
            "leakage_warnings": list(evidence_row.get("leakage_warnings", [])),
            "errors": [],
        }
        rows.append(row)
    return rows


def materialize_degraded_image(image_path: str, config: FourConditionOfflineBuilderConfig) -> str:
    return materialize_degraded_image_file(
        image_path,
        mode=config.degraded_mode,
        degraded_dir=config.degraded_dir,
        blur_sigma=config.blur_sigma,
    )


def _degraded_transform(config: FourConditionOfflineBuilderConfig) -> dict[str, Any]:
    return degraded_transform(config.degraded_mode, blur_sigma=config.blur_sigma)


def _serialize_condition_score(
    scores: Any,
    response_token_ids: Sequence[int],
    *,
    top_k: int,
    actual_override: Sequence[float] | None = None,
) -> dict[str, Any]:
    block = {**_serialize_topk(scores), "top_k": top_k}
    if actual_override is None:
        sampled = torch.tensor([list(response_token_ids)], dtype=torch.int64, device=scores.token_ids.device)
        actual = sampled_token_log_prob(scores, sampled)[0].detach().float().cpu().tolist()
    else:
        actual = [float(value) for value in actual_override]
    block["actual_token_log_probs"] = actual
    return block


def _has_student_condition_scorer(rollout_generator: StudentRolloutGenerator) -> bool:
    return callable(getattr(rollout_generator, "score_conditions", None))


def _score_student_conditions(
    *,
    rollout_generator: StudentRolloutGenerator,
    token_ids: Sequence[int],
    question: str,
    condition_inputs: ConditionInputs,
    conditions: Sequence[Condition],
    response_text: str,
    top_k: int,
    enabled: bool,
) -> dict[Condition, Any]:
    if not enabled:
        return {}
    scorer = getattr(rollout_generator, "score_conditions", None)
    if callable(scorer):
        return scorer(
            response_token_ids=token_ids,
            question=question,
            condition_inputs=condition_inputs,
            conditions=conditions,
            response_text=response_text,
            top_k=top_k,
        )
    return {}


def _fallback_malformed_response(rollout_uid: str) -> str:
    del rollout_uid
    return (
        "<visible_evidence>\n"
        "Fallback malformed rollout emitted after repeated empty or invalid student rollouts.\n"
        "</visible_evidence>\n"
        "No final answer block is available.\n"
    )


def verifier_outcome_class(verifier: Mapping[str, Any] | None) -> str:
    if verifier is None:
        return "unknown"
    if bool(verifier.get("malformed", False)):
        return "malformed"
    if verifier.get("correct") is True:
        return "correct"
    if bool(verifier.get("format_valid", False)):
        return "wrong_but_format_valid"
    return "malformed"


def build_verifier_learning_value_gate(verifier: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build continuous per-chunk learning-value weights from verifier reward."""
    outcome_class = verifier_outcome_class(verifier)
    if verifier is None:
        reward = None
        gates = {chunk: 1.0 for chunk in VERIFIER_GATE_CHUNK_MAX}
    else:
        reward = _clamp01(float(verifier.get("reward", 0.0)))
        learning_value = 1.0 - reward
        gates = {
            chunk: VERIFIER_GATE_RESIDUAL_FLOOR + (chunk_max - VERIFIER_GATE_RESIDUAL_FLOOR) * learning_value
            for chunk, chunk_max in VERIFIER_GATE_CHUNK_MAX.items()
        }
    return {
        "outcome_class": outcome_class,
        "gate_policy": "reward_smoothed_v2",
        "correct": None if verifier is None else verifier.get("correct"),
        "format_valid": True if verifier is None else bool(verifier.get("format_valid", False)),
        "reward": reward,
        "chunk_gates": gates,
    }


def grouped_loss_plan(schema: str) -> dict[str, list[str]]:
    if schema != "capability_chunk_v1":
        return {}
    return {
        "visible": ["visual_detail", "evidence_selection"],
        "infer": ["visual_text_inference"],
        "solve": ["solving"],
    }


def compute_student_deficit_capability_scores(
    *,
    teacher_condition_scores: Mapping[str, Mapping[str, Any]],
    student_condition_scores: Mapping[str, Mapping[str, Any]],
    chunk_spans: Mapping[str, Any],
    verifier_learning_value_gate: Mapping[str, Any],
    margin: float = 0.0,
    max_capabilities_per_token: int = 2,
    task_infer_solve_like: bool = False,
    enable_student_deficit_gate: bool = True,
) -> dict[str, dict[str, Any]]:
    if not teacher_condition_scores:
        return {}
    length = _score_length(next(iter(teacher_condition_scores.values())))
    chunk_labels = _chunk_labels(chunk_spans, length)
    raw_weights: dict[str, list[float]] = {}
    output: dict[str, dict[str, Any]] = {}

    for capability, (positive, negative) in CAPABILITY_CONTRASTS.items():
        pos = positive.value
        neg = negative.value
        valid = pos in teacher_condition_scores and neg in teacher_condition_scores
        invalid_reason = None
        effective_negative = neg
        if capability == "visual_text_inference" and task_infer_solve_like:
            valid = False
            invalid_reason = "task_infer_solve_like"
        if capability == "solving" and task_infer_solve_like and "task_visible" in teacher_condition_scores:
            effective_negative = "task_visible"
            valid = pos in teacher_condition_scores
        if not valid:
            output[capability] = _invalid_capability(length, pos, effective_negative, invalid_reason or "missing_condition")
            raw_weights[capability] = [0.0] * length
            continue

        t_pos = _actual_log_probs(teacher_condition_scores[pos], length)
        t_neg = _actual_log_probs(teacher_condition_scores[effective_negative], length)
        if enable_student_deficit_gate and pos in student_condition_scores and effective_negative in student_condition_scores:
            s_pos = _actual_log_probs(student_condition_scores[pos], length)
            s_neg = _actual_log_probs(student_condition_scores[effective_negative], length)
        else:
            s_pos = [0.0] * length
            s_neg = [0.0] * length
        teacher_delta = [a - b for a, b in zip(t_pos, t_neg, strict=True)]
        student_delta = [a - b for a, b in zip(s_pos, s_neg, strict=True)]
        teacher_attribution = [max(0.0, value) for value in teacher_delta]
        student_deficit = [
            max(0.0, t_value - s_value - float(margin))
            for t_value, s_value in zip(teacher_delta, student_delta, strict=True)
        ]
        chunk_compatibility_prior = [
            float(CHUNK_CAPABILITY_COMPATIBILITY.get(label, {}).get(capability, 0.0))
            for label in chunk_labels
        ]
        chunk_compatibility = [
            _soft_chunk_capability_compatibility(label, capability)
            for label in chunk_labels
        ]
        chunk_gates = (
            verifier_learning_value_gate.get("chunk_gates", {})
            if isinstance(verifier_learning_value_gate, Mapping)
            else {}
        )
        verifier_learning_value_weights = [float(chunk_gates.get(label, 1.0)) for label in chunk_labels]
        final = [
            attr * deficit * compat * gate
            for attr, deficit, compat, gate in zip(
                teacher_attribution,
                student_deficit,
                chunk_compatibility,
                verifier_learning_value_weights,
                strict=True,
            )
        ]
        raw_weights[capability] = final
        output[capability] = {
            "positive": pos,
            "negative": effective_negative,
            "teacher_delta": teacher_delta,
            "student_delta": student_delta,
            "teacher_attribution": teacher_attribution,
            "student_deficit": student_deficit,
            "chunk_compatibility_prior": chunk_compatibility_prior,
            "chunk_compatibility": chunk_compatibility,
            "verifier_learning_value_gate": verifier_learning_value_weights,
            "final_token_weight": final,
            "valid": True,
        }

    if max_capabilities_per_token > 0 and raw_weights:
        kept = {capability: [0.0] * length for capability in raw_weights}
        for index in range(length):
            ranked = sorted(
                ((capability, weights[index]) for capability, weights in raw_weights.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            for capability, value in ranked[:max_capabilities_per_token]:
                kept[capability][index] = float(value)
        for capability, weights in kept.items():
            if capability in output:
                output[capability]["final_token_weight"] = weights
    return output


def _invalid_capability(length: int, positive: str, negative: str, reason: str) -> dict[str, Any]:
    return {
        "positive": positive,
        "negative": negative,
        "teacher_delta": [0.0] * length,
        "student_delta": [0.0] * length,
        "teacher_attribution": [0.0] * length,
        "student_deficit": [0.0] * length,
        "chunk_compatibility_prior": [0.0] * length,
        "chunk_compatibility": [0.0] * length,
        "verifier_learning_value_gate": [0.0] * length,
        "final_token_weight": [0.0] * length,
        "valid": False,
        "invalid_reason": reason,
    }


def _score_length(block: Mapping[str, Any]) -> int:
    if isinstance(block.get("actual_token_log_probs"), Sequence):
        return len(block["actual_token_log_probs"])
    return len(block.get("token_ids", []))


def _soft_chunk_capability_compatibility(label: str, capability: str) -> float:
    prior = float(CHUNK_CAPABILITY_COMPATIBILITY.get(label, {}).get(capability, 0.0))
    prior = _clamp01(prior)
    return CHUNK_CAPABILITY_COMPATIBILITY_FLOOR + (1.0 - CHUNK_CAPABILITY_COMPATIBILITY_FLOOR) * prior


def _clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _actual_log_probs(block: Mapping[str, Any], length: int) -> list[float]:
    values = block.get("actual_token_log_probs")
    if isinstance(values, Sequence) and len(values) == length:
        return [float(value) for value in values]
    token_log_probs = block.get("log_probs", [])
    return [float(row[0]) if row else 0.0 for row in token_log_probs][:length]


def _chunk_labels(chunk_spans: Mapping[str, Any], length: int) -> list[str]:
    labels = chunk_spans.get("chunk_labels")
    if isinstance(labels, Sequence) and len(labels) == length:
        return [str(label) for label in labels]
    output = ["reasoning"] * length
    for chunk in ("visible_evidence", "diagram_inference", "reasoning", "answer", "visual_evidence"):
        canonical = "visible_evidence" if chunk == "visual_evidence" else chunk
        spans = chunk_spans.get(chunk, [])
        if not isinstance(spans, Sequence):
            continue
        for span in spans:
            if not isinstance(span, Sequence) or len(span) != 2:
                continue
            start, end = int(span[0]), int(span[1])
            for index in range(max(0, start), min(length, end)):
                output[index] = canonical
    return output


def validate_four_condition_rows(path: str | Path, *, condition_set: str | None = None) -> dict[str, Any]:
    rows = _read_jsonl(path)
    errors: list[str] = []
    for row in rows:
        uid = str(row.get("sample_uid", "unknown"))
        expected = (
            [condition.value for condition in CONDITION_SETS[condition_set]]
            if condition_set is not None
            else row.get("conditions")
        )
        if row.get("conditions") != expected:
            errors.append(f"{uid}: conditions mismatch")
        if row.get("degraded_mode") not in set(DEGRADED_MODES):
            errors.append(f"{uid}: invalid degraded_mode")
        if row.get("leakage_warnings"):
            errors.append(f"{uid}: leakage warnings present")
        try:
            tensors = offline_record_to_tensors(row)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{uid}: tensor conversion failed: {exc}")
            continue
        if tensors.seq_len != len(row.get("response_token_ids", [])):
            errors.append(f"{uid}: T mismatch")
        teacher_meta = row.get("teacher_metadata", {})
        if isinstance(teacher_meta, Mapping) and row.get("tokenizer_hash") != teacher_meta.get("tokenizer_hash"):
            errors.append(f"{uid}: tokenizer hash mismatch")
        evidence_meta = row.get("evidence_generation_metadata")
        if not isinstance(evidence_meta, Mapping):
            errors.append(f"{uid}: missing evidence_generation_metadata")
        chunk_spans = row.get("chunk_spans", {})
        if isinstance(chunk_spans, Mapping) and chunk_spans.get("format_valid") is not True:
            chunk_errors = chunk_spans.get("errors", [])
            errors.append(f"{uid}: chunk parse failed: {chunk_errors}")
        delta_errors = _row_delta_validation_errors(row)
        errors.extend(f"{uid}: {error}" for error in delta_errors)
    return {"num_rows": len(rows), "valid": bool(rows) and not errors, "errors": errors}


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: FourConditionOfflineBuilderConfig,
    teacher_client: TeacherClient,
    diagnostics: OfflineBuilderDiagnostics | None = None,
) -> dict[str, Any]:
    lengths = [int(row.get("response_token_count", 0)) for row in rows]
    conditions = [condition.value for condition in CONDITION_SETS[config.condition_set]]
    parse_success = [
        bool(row.get("chunk_spans", {}).get("format_valid", False))
        for row in rows
    ]
    rows_with_chunk_parse_failure = [
        str(row.get("sample_uid", "unknown"))
        for row in rows
        if row.get("chunk_spans", {}).get("format_valid") is not True
    ]
    chunk_counts: Counter[str] = Counter()
    for row in rows:
        token_counts = row.get("chunk_spans", {}).get("token_counts", {})
        if isinstance(token_counts, Mapping):
            for key, value in token_counts.items():
                chunk_counts[str(key)] += int(value)
    delta_report = _summarize_delta_fields(rows)
    capability_report = _summarize_capability_fields(rows)
    verifier_outcome_counts = _verifier_outcome_counts(rows)
    gate_weight_by_outcome = _gate_weight_by_outcome(rows)
    validation_errors = _summary_validation_errors(rows, delta_report)
    validation_valid = bool(rows) and not validation_errors
    summary = {
        "source_dataset": config.source_dataset,
        "condition_set_name": config.condition_set,
        "conditions": conditions,
        "degraded_mode": config.degraded_mode,
        "num_rows": len(rows),
        "num_prompts": len({row.get("prompt_sample_uid") for row in rows}),
        "rollouts_per_prompt": config.rollouts_per_prompt,
        "mean_response_tokens": None if not lengths else sum(lengths) / len(lengths),
        "chunk_parse_success_rate": None if not parse_success else sum(parse_success) / len(parse_success),
        "chunk_token_counts": dict(chunk_counts),
        "condition_score_success_rate": {condition: 1.0 for condition in conditions},
        "student_condition_score_success_rate": _student_condition_success_rate(rows, conditions),
        "delta_means": delta_report["delta_means"],
        "capability_delta_means_teacher": capability_report["teacher_delta_means"],
        "capability_delta_means_student": capability_report["student_delta_means"],
        "capability_deficit_means": capability_report["deficit_means"],
        "capability_final_weight_sums": capability_report["final_weight_sums"],
        "capability_nonzero_token_counts": capability_report["nonzero_token_counts"],
        "capability_by_chunk_weight_sums": capability_report["by_chunk_weight_sums"],
        "verifier_outcome_counts": verifier_outcome_counts,
        "gate_weight_by_outcome": gate_weight_by_outcome,
        "capability_weight_by_outcome": capability_report["by_outcome_weight_sums"],
        "capability_weight_by_chunk_and_outcome": capability_report["by_chunk_and_outcome_weight_sums"],
        "correct_rollout_opd_weight_sum": capability_report["rollout_weight_sums"]["correct"],
        "wrong_valid_rollout_opd_weight_sum": capability_report["rollout_weight_sums"]["wrong_but_format_valid"],
        "malformed_rollout_opd_weight_sum": capability_report["rollout_weight_sums"]["malformed"],
        "verifier_accuracy_over_rollouts": _verifier_accuracy(rows),
        "task_infer_class_counts": dict(Counter(str(row.get("task_infer_class", "missing")) for row in rows)),
        "rows_with_invalid_capability": capability_report["rows_with_invalid_capability"],
        "degraded_source_image_count": 0,
        "degraded_image_output_root": str(default_degraded_image_dir()),
        "grouped_loss_ready": bool(rows) and all(isinstance(row.get("capability_scores"), Mapping) for row in rows),
        "visual_detail_delta_count": delta_report["counts"]["visual_detail_delta"],
        "visual_detail_delta_invalid_count": delta_report["invalid_counts"]["visual_detail_delta"],
        "delta_counts": delta_report["counts"],
        "delta_invalid_counts": delta_report["invalid_counts"],
        "gate_ready_fields_available": bool(rows),
        "validation_valid": validation_valid,
        "validation_error_count": len(validation_errors),
        "validation_errors_top10": validation_errors[:10],
        "rows_with_chunk_parse_failure": rows_with_chunk_parse_failure,
        "rows_with_invalid_delta": delta_report["rows_with_invalid_delta"],
        "response_length_p50": _quantile(lengths, 0.5),
        "response_length_p90": _quantile(lengths, 0.9),
        "max_new_tokens_recommendation": _max_new_tokens_recommendation(config),
        "unique_response_text_hash_count": len({row.get("response_text_hash") for row in rows}),
        "response_source_counts": dict(Counter(str(row.get("response_source", "unknown")) for row in rows)),
        "teacher_model_id": teacher_client.metadata.model_id,
        "tokenizer_hash": rows[0].get("tokenizer_hash") if rows else "",
        "output_jsonl": str(config.output_jsonl),
        "summary_json": str(config.summary_json),
        "red_box_contaminated": False,
        "not_main_experiment": False,
    }
    if diagnostics is not None:
        summary.update(diagnostics.to_summary())
    return summary


DELTA_SIGNAL_NAMES = ("visual_detail_delta", "task_selection_delta", "diagram_infer_delta", "solve_delta")


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) else None


def _row_delta_validation_errors(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    row_conditions = set(str(condition) for condition in row.get("conditions", []))
    required_pairs = {
        "visual_detail_delta": ("full", "degraded"),
        "task_selection_delta": ("task_visible", "free"),
        "diagram_infer_delta": ("task_infer", "task_visible"),
        "solve_delta": ("task_solve", "task_infer"),
    }
    signal_summary = row.get("condition_signal_summary", {})
    for name, pair in required_pairs.items():
        if not set(pair).issubset(row_conditions):
            continue
        block = signal_summary.get(name) if isinstance(signal_summary, Mapping) else None
        if not isinstance(block, Mapping):
            errors.append(f"{name} missing")
            continue
        if _finite_float(block.get("mean")) is None:
            errors.append(f"{name} invalid mean")
    return errors


def _summarize_delta_fields(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {name: [] for name in DELTA_SIGNAL_NAMES}
    invalid_counts: dict[str, int] = {name: 0 for name in DELTA_SIGNAL_NAMES}
    rows_with_invalid_delta: list[str] = []

    for row in rows:
        row_invalid = False
        row_conditions = set(str(condition) for condition in row.get("conditions", []))
        required = {
            "visual_detail_delta": {"full", "degraded"}.issubset(row_conditions),
            "task_selection_delta": {"task_visible", "free"}.issubset(row_conditions),
            "diagram_infer_delta": {"task_infer", "task_visible"}.issubset(row_conditions),
            "solve_delta": {"task_solve", "task_infer"}.issubset(row_conditions),
        }
        signal_summary = row.get("condition_signal_summary", {})
        for name in DELTA_SIGNAL_NAMES:
            if not required[name]:
                continue
            block = signal_summary.get(name) if isinstance(signal_summary, Mapping) else None
            mean_value = _finite_float(block.get("mean") if isinstance(block, Mapping) else None)
            if mean_value is None:
                invalid_counts[name] += 1
                row_invalid = True
            else:
                values[name].append(mean_value)
        if row_invalid:
            rows_with_invalid_delta.append(str(row.get("sample_uid", "unknown")))

    delta_means = {
        name: (None if not signal_values else float(mean(signal_values)))
        for name, signal_values in values.items()
    }
    return {
        "delta_means": delta_means,
        "counts": {name: len(signal_values) for name, signal_values in values.items()},
        "invalid_counts": invalid_counts,
        "rows_with_invalid_delta": rows_with_invalid_delta,
    }


def _student_condition_success_rate(rows: Sequence[Mapping[str, Any]], conditions: Sequence[str]) -> dict[str, float | None]:
    if not rows:
        return {condition: None for condition in conditions}
    output: dict[str, float | None] = {}
    for condition in conditions:
        output[condition] = sum(
            1
            for row in rows
            if isinstance(row.get("student_condition_scores"), Mapping)
            and condition in row.get("student_condition_scores", {})
        ) / len(rows)
    return output


def _summarize_capability_fields(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    teacher_values: dict[str, list[float]] = {name: [] for name in CAPABILITY_CONTRASTS}
    student_values: dict[str, list[float]] = {name: [] for name in CAPABILITY_CONTRASTS}
    deficit_values: dict[str, list[float]] = {name: [] for name in CAPABILITY_CONTRASTS}
    final_sums: dict[str, float] = {name: 0.0 for name in CAPABILITY_CONTRASTS}
    nonzero_counts: dict[str, int] = {name: 0 for name in CAPABILITY_CONTRASTS}
    by_chunk: dict[str, dict[str, float]] = {name: {} for name in CAPABILITY_CONTRASTS}
    by_outcome: dict[str, dict[str, float]] = {name: {} for name in CAPABILITY_CONTRASTS}
    by_chunk_and_outcome: dict[str, dict[str, dict[str, float]]] = {name: {} for name in CAPABILITY_CONTRASTS}
    rollout_weight_sums: dict[str, float] = {
        "correct": 0.0,
        "wrong_but_format_valid": 0.0,
        "malformed": 0.0,
        "unknown": 0.0,
    }
    rows_with_invalid: list[str] = []

    for row in rows:
        capability_scores = row.get("capability_scores", {})
        if not isinstance(capability_scores, Mapping):
            continue
        labels = _chunk_labels(row.get("chunk_spans", {}) if isinstance(row.get("chunk_spans"), Mapping) else {}, int(row.get("response_token_count", 0)))
        outcome_class = _row_verifier_outcome_class(row)
        row_invalid = False
        row_weight_sum = 0.0
        for capability in CAPABILITY_CONTRASTS:
            block = capability_scores.get(capability)
            if not isinstance(block, Mapping):
                continue
            if block.get("valid") is not True:
                row_invalid = True
            teacher_values[capability].extend(_finite_values(block.get("teacher_delta", [])))
            student_values[capability].extend(_finite_values(block.get("student_delta", [])))
            deficit_values[capability].extend(_finite_values(block.get("student_deficit", [])))
            weights = _finite_values(block.get("final_token_weight", []))
            final_sums[capability] += float(sum(weights))
            row_weight_sum += float(sum(weights))
            nonzero_counts[capability] += sum(1 for value in weights if value > 0)
            by_outcome[capability][outcome_class] = by_outcome[capability].get(outcome_class, 0.0) + float(sum(weights))
            chunk_report = by_chunk[capability]
            outcome_chunk_report = by_chunk_and_outcome[capability].setdefault(outcome_class, {})
            for label, value in zip(labels, weights, strict=False):
                chunk_report[label] = chunk_report.get(label, 0.0) + float(value)
                outcome_chunk_report[label] = outcome_chunk_report.get(label, 0.0) + float(value)
        if row_invalid:
            rows_with_invalid.append(str(row.get("sample_uid", "unknown")))
        rollout_weight_sums[outcome_class] = rollout_weight_sums.get(outcome_class, 0.0) + row_weight_sum

    return {
        "teacher_delta_means": _mean_map(teacher_values),
        "student_delta_means": _mean_map(student_values),
        "deficit_means": _mean_map(deficit_values),
        "final_weight_sums": final_sums,
        "nonzero_token_counts": nonzero_counts,
        "by_chunk_weight_sums": by_chunk,
        "by_outcome_weight_sums": by_outcome,
        "by_chunk_and_outcome_weight_sums": by_chunk_and_outcome,
        "rollout_weight_sums": rollout_weight_sums,
        "rows_with_invalid_capability": rows_with_invalid,
    }


def _verifier_outcome_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"correct": 0, "wrong_but_format_valid": 0, "malformed": 0, "unknown": 0}
    for row in rows:
        counts[_row_verifier_outcome_class(row)] += 1
    return counts


def _row_verifier_outcome_class(row: Mapping[str, Any]) -> str:
    gate = row.get("verifier_learning_value_gate")
    if isinstance(gate, Mapping) and str(gate.get("outcome_class", "")) in VERIFIER_LEARNING_VALUE_CHUNK_GATES:
        return str(gate["outcome_class"])
    verifier = row.get("verifier")
    return verifier_outcome_class(verifier if isinstance(verifier, Mapping) else None)


def _gate_weight_by_outcome(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {
        outcome: {chunk: 0.0 for chunk in ("visible_evidence", "diagram_inference", "reasoning", "answer")}
        for outcome in VERIFIER_LEARNING_VALUE_CHUNK_GATES
    }
    for row in rows:
        outcome_class = _row_verifier_outcome_class(row)
        length = int(row.get("response_token_count", 0))
        labels = _chunk_labels(
            row.get("chunk_spans", {}) if isinstance(row.get("chunk_spans"), Mapping) else {},
            length,
        )
        gate = row.get("verifier_learning_value_gate")
        chunk_gates = gate.get("chunk_gates", {}) if isinstance(gate, Mapping) else {}
        for label in labels:
            if label in output[outcome_class]:
                output[outcome_class][label] += float(chunk_gates.get(label, 1.0))
    return output


def _verifier_accuracy(rows: Sequence[Mapping[str, Any]]) -> float | None:
    verified = [row.get("verifier") for row in rows if isinstance(row.get("verifier"), Mapping)]
    usable = [block for block in verified if block.get("correct") is not None]
    if not usable:
        return None
    return sum(1 for block in usable if block.get("correct") is True) / len(usable)


def _finite_values(values: object) -> list[float]:
    if not isinstance(values, Sequence):
        return []
    output = []
    for value in values:
        finite = _finite_float(value)
        if finite is not None:
            output.append(finite)
    return output


def _mean_map(values: Mapping[str, Sequence[float]]) -> dict[str, float | None]:
    return {key: None if not vals else float(mean(vals)) for key, vals in values.items()}


def _exception_reason(exc: Exception) -> str:
    return _compact_reason(f"{type(exc).__name__}: {exc}")


def _compact_reason(reason: str, *, limit: int = 240) -> str:
    compact = " ".join(str(reason).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _debug(config: FourConditionOfflineBuilderConfig, record_index: int, message: str) -> None:
    if config.debug_first_n > 0 and record_index < config.debug_first_n:
        print(f"[fc-opd-builder-debug] record_index={record_index} {message}", flush=True)


def _summary_validation_errors(rows: Sequence[Mapping[str, Any]], delta_report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not rows:
        errors.append("no rows produced")
        return errors
    for row in rows:
        uid = str(row.get("sample_uid", "unknown"))
        chunk_spans = row.get("chunk_spans", {})
        if isinstance(chunk_spans, Mapping) and chunk_spans.get("format_valid") is not True:
            errors.append(f"{uid}: chunk parse failed: {chunk_spans.get('errors', [])}")
        elif not isinstance(chunk_spans, Mapping):
            errors.append(f"{uid}: missing chunk_spans")
        for error in _row_delta_validation_errors(row):
            errors.append(f"{uid}: {error}")
    for name, invalid_count in delta_report["invalid_counts"].items():
        if int(invalid_count) > 0:
            errors.append(f"{name}: invalid_count={invalid_count}")
    return errors


def _max_new_tokens_recommendation(config: FourConditionOfflineBuilderConfig) -> str | None:
    if config.rollout_response_format == "fc_opd_structured_v2" and config.condition_set == "6c-solve" and config.max_new_tokens < 768:
        return "For Geometry3K 6C fc_opd_structured_v2 smoke, prefer --max-new-tokens >= 768 unless using a deliberately concise model prompt."
    return None


def _attach_group_stats(rows: list[dict[str, Any]]) -> None:
    stats = compute_rollout_group_stats(rows)
    for row in rows:
        row["success_group_stats"] = stats.get(str(row.get("rollout_group_uid")))


def _rollout_config(config: FourConditionOfflineBuilderConfig) -> StudentRolloutAuditConfig:
    return StudentRolloutAuditConfig(
        dataset=config.dataset,
        dataset_type="generic_jsonl",
        source_dataset=config.source_dataset,
        student_model_path=config.student_model_path,
        teacher_url=config.teacher_url,
        limit=config.limit or 1,
        conditions=CONDITION_SETS[config.condition_set],
        rollouts_per_prompt=config.rollouts_per_prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
        seed=config.seed,
        device=config.device,
        dtype=config.dtype,
        rollout_response_format=config.rollout_response_format,
    )


def _select(records: Sequence[dict[str, Any]], config: FourConditionOfflineBuilderConfig) -> list[dict[str, Any]]:
    end = config.end_index if config.end_index is not None else len(records)
    selected = list(records[config.start_index : end])
    return selected[: config.limit] if config.limit is not None else selected


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    if not cleaned:
        return {"mean": None, "p50": None, "p90": None}
    return {"mean": float(mean(cleaned)), "p50": _quantile(cleaned, 0.5), "p90": _quantile(cleaned, 0.9)}


def _quantile(values: Sequence[float | int], fraction: float) -> float | None:
    cleaned = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    index = fraction * (len(cleaned) - 1)
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return cleaned[lo]
    weight = index - lo
    return cleaned[lo] * (1 - weight) + cleaned[hi] * weight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evidence-cache", type=Path, required=True)
    parser.add_argument("--dataset-type", default="geometry3k")
    parser.add_argument("--source-dataset", default="geometry3k")
    parser.add_argument("--student-model-path", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--teacher-url", default=_DEFAULT_TEACHER_URL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--rollout-response-format", choices=ROLLOUT_RESPONSE_FORMATS, default="fc_opd_structured")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--degraded-mode", choices=DEGRADED_MODES, default="lowres_10pct_nearest")
    parser.add_argument("--degraded-dir")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--condition-set", choices=tuple(CONDITION_SETS), default="4c-clean")
    parser.add_argument("--enable-student-condition-scoring", action="store_true")
    parser.add_argument("--student-deficit-gate", action="store_true")
    parser.add_argument("--verifier-gate", choices=("none", "geometry3k_verifier"), default="none")
    parser.add_argument(
        "--outcome-gate",
        choices=("none", "geometry3k_verifier"),
        dest="verifier_gate",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--strict-condition-validation", action="store_true")
    parser.add_argument("--capability-margin", type=float, default=0.0)
    parser.add_argument("--max-capabilities-per-token", type=int, default=2)
    parser.add_argument(
        "--routing-mode",
        choices=(
            "uniform_all_conditions",
            "chunk_gated_primary",
            "chunk_gated_contrastive",
            "student_deficit_chunk_gated",
        ),
        default="chunk_gated_contrastive",
    )
    parser.add_argument("--grouped-loss-schema", choices=("none", "capability_chunk_v1"), default="none")
    parser.add_argument("--allow-empty-output", action="store_true")
    parser.add_argument("--debug-first-n", type=int, default=0)
    parser.add_argument("--max-rollout-attempts", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        report = validate_four_condition_rows(args.output_jsonl, condition_set=args.condition_set)
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 1
    result = run_four_condition_offline_builder(
        FourConditionOfflineBuilderConfig(
            dataset=args.dataset,
            evidence_cache=args.evidence_cache,
            dataset_type=args.dataset_type,
            source_dataset=args.source_dataset,
            student_model_path=args.student_model_path,
            teacher_url=args.teacher_url,
            limit=args.limit,
            start_index=args.start_index,
            end_index=args.end_index,
            rollouts_per_prompt=args.rollouts_per_prompt,
            rollout_response_format=args.rollout_response_format,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
            degraded_mode=args.degraded_mode,
            degraded_dir=args.degraded_dir,
            output_jsonl=args.output_jsonl,
            summary_json=args.summary_json,
            resume=args.resume,
            skip_existing=args.skip_existing,
            condition_set=args.condition_set,
            enable_student_condition_scoring=args.enable_student_condition_scoring,
            student_deficit_gate=args.student_deficit_gate,
            verifier_gate=args.verifier_gate,
            strict_condition_validation=args.strict_condition_validation,
            capability_margin=args.capability_margin,
            max_capabilities_per_token=args.max_capabilities_per_token,
            routing_mode=args.routing_mode,
            grouped_loss_schema=args.grouped_loss_schema,
            allow_empty_output=args.allow_empty_output,
            debug_first_n=args.debug_first_n,
            max_rollout_attempts=args.max_rollout_attempts,
        )
    )
    validation = validate_four_condition_rows(args.output_jsonl, condition_set=args.condition_set)
    print(json.dumps({**result.summary, "validation_valid": validation["valid"]}, indent=2))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

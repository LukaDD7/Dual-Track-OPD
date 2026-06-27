"""Offline FC-OPD scoring dataset builder.

This module turns a prepared Vision-OPD-style dataset plus student responses
into a self-describing offline-score dataset by querying a running teacher
service. It deliberately stays model-free: no student model is loaded, and the
only tokenizer it needs is one that can ``encode``/``decode`` token IDs and
expose its vocabulary for fingerprinting. A byte-level tokenizer is bundled so
that protocol-smoke datasets and tests run without any model weights.

The heavy lifting (top-k scoring, signal decomposition, chunk parsing) is reused
from the existing ``dual_track_opd.fc_opd`` primitives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

import torch

from .chunk_parser import parse_response_chunks
from .conditions import Condition, ConditionInputs, ImageInput
from .signal_decomposer import TeacherTopK, compute_condition_signals
from .teacher_client import TeacherClient, score_teacher_conditions
from .teacher_protocol import tokenizer_fingerprint

DEFAULT_CONDITIONS: tuple[Condition, ...] = (
    Condition.FULL,
    Condition.BLUR,
    Condition.FREE,
    Condition.TASK,
)
DEFAULT_TEACHER_URL = "http://127.0.0.1:18080"
OFFLINE_SCORE_SUBDIR = Path("fc_opd") / "offline_scores"


class OfflineScoringTokenizer(Protocol):
    """Minimal tokenizer surface required for offline scoring."""

    def encode(self, text: str, **kwargs: object) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], **kwargs: object) -> str: ...

    def get_vocab(self) -> Mapping[str, int]: ...


class ByteTokenizer:
    """Deterministic byte-level tokenizer with an exact decode round-trip.

    Each UTF-8 byte maps to its own token ID. This makes the tokenizer fully
    reproducible and model-free, which keeps protocol-smoke datasets and unit
    tests independent of any downloaded weights.
    """

    special_tokens_map: dict[str, str] = {}

    def encode(self, text: str, **kwargs: object) -> list[int]:
        del kwargs
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int], **kwargs: object) -> str:
        del kwargs
        return bytes(int(token_id) for token_id in token_ids).decode("utf-8", errors="replace")

    def get_vocab(self) -> dict[str, int]:
        return {chr(index): index for index in range(256)}

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    @property
    def vocab_size(self) -> int:
        return 256


@dataclass(frozen=True)
class OfflineScoringConfig:
    """Configuration for building one offline-score dataset."""

    source_dataset: str
    conditions: tuple[Condition, ...] = DEFAULT_CONDITIONS
    blur_sigma: float = 2.0
    data_root: str | None = None
    degraded_dir: str | None = None
    require_paths: bool = False
    chunk_fallback: str = "all_reasoning"
    smoke_caption: str = (
        "Protocol-smoke placeholder caption: a single scene with one foreground object."
    )
    smoke_task_evidence: str = (
        "Protocol-smoke placeholder evidence: the question-relevant region shows one object."
    )

    def __post_init__(self) -> None:
        if not self.source_dataset.strip():
            raise ValueError("source_dataset must be a non-empty string")
        if not self.conditions:
            raise ValueError("at least one condition is required")
        if self.blur_sigma <= 0:
            raise ValueError("blur_sigma must be positive")


# ---------------------------------------------------------------------------
# Dataset and student-response loading
# ---------------------------------------------------------------------------


def load_vision_opd_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a Vision-OPD-style dataset from a JSON array or JSONL file."""

    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("top-level JSON must be a list of records")
        return [_as_mapping(record, index) for index, record in enumerate(data)]
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        records.append(_as_mapping(json.loads(line), line_number))
    return records


def load_student_responses(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a student response JSONL keyed by its resolvable sample identifier."""

    path = Path(path)
    mapping: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, Mapping):
            raise ValueError("each student response line must be a JSON object")
        key = _student_response_key(record)
        mapping[key] = dict(record)
    return mapping


def _as_mapping(record: object, index: int) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"record at position {index} must be a JSON object")
    return dict(record)


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def extract_question(record: Mapping[str, Any]) -> str:
    question = _first_present(record, ("query", "question", "prompt", "instruction"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("record is missing a non-empty query/question field")
    return question.strip()


def extract_answer(record: Mapping[str, Any]) -> str | None:
    answer = _first_present(record, ("response", "answer", "label", "target"))
    if answer is None:
        return None
    return str(answer)


def extract_image_paths(record: Mapping[str, Any], data_root: str | None) -> list[str]:
    raw = _first_present(record, ("images", "image", "image_path", "image_paths"))
    if raw is None:
        raise ValueError("record is missing an images/image/image_path field")
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, Sequence):
        candidates = [str(item) for item in raw]
    else:
        raise ValueError("images field must be a string or list of strings")
    if not candidates:
        raise ValueError("record contains no image paths")
    return [_resolve_path(path, data_root) for path in candidates]


def extract_source_index(record: Mapping[str, Any], fallback: int) -> int:
    value = _first_present(record, ("source_index", "index", "idx"))
    if value is None:
        return fallback
    return int(value)


def extract_question_id(record: Mapping[str, Any], fallback: int) -> str:
    value = _first_present(record, ("question_id", "qid", "id", "uid", "index"))
    if value is None:
        return str(fallback)
    return str(value)


def _student_response_key(record: Mapping[str, Any]) -> str:
    value = _first_present(record, ("sample_uid", "question_id", "qid", "id", "uid", "index"))
    if value is None:
        raise ValueError("student response is missing a sample_uid/question_id/index field")
    return str(value)


def _resolve_path(path: str, data_root: str | None) -> str:
    candidate = Path(path).expanduser()
    if data_root is not None and not candidate.is_absolute():
        candidate = Path(data_root).expanduser() / candidate
    return str(candidate)


# ---------------------------------------------------------------------------
# Condition inputs and response construction
# ---------------------------------------------------------------------------


def derive_degraded_path(full_path: str, sigma: float, degraded_dir: str | None) -> str:
    """Derive a deterministic prepared blurred-image path without creating it."""

    source = Path(full_path)
    stem = source.stem
    suffix = source.suffix or ".png"
    sigma_tag = f"{sigma:g}".replace(".", "_")
    name = f"{stem}.gaussian_blur_s{sigma_tag}{suffix}"
    target_dir = Path(degraded_dir).expanduser() if degraded_dir else source.parent
    return str(target_dir / name)


def build_condition_inputs_for_record(
    record: Mapping[str, Any],
    config: OfflineScoringConfig,
) -> ConditionInputs:
    """Build validated condition inputs for one Vision-OPD record.

    If the record already carries an explicit ``condition_inputs`` mapping it is
    used verbatim (only its blur transform is reused). Otherwise condition inputs
    are synthesised from the record's images and any caption/task-evidence
    fields, falling back to protocol-smoke placeholders.
    """

    image_paths = extract_image_paths(record, config.data_root)
    full_path = image_paths[0]

    raw_inputs = record.get("condition_inputs")
    if isinstance(raw_inputs, Mapping):
        return _condition_inputs_from_mapping(raw_inputs, config)

    transform = {"type": "gaussian_blur", "sigma": float(config.blur_sigma)}
    degraded_path = derive_degraded_path(full_path, config.blur_sigma, config.degraded_dir)

    caption = _coalesce_text(record, ("free_caption", "caption"), config.smoke_caption)
    evidence = _coalesce_text(
        record, ("task_evidence", "task_extraction", "evidence"), config.smoke_task_evidence
    )
    visible = _coalesce_text(record, ("task_visible_evidence", "task_visible"), evidence)
    infer = _first_present(record, ("task_infer_evidence", "task_infer"))
    solve = _first_present(record, ("task_solve_evidence", "task_solve"))

    inputs = ConditionInputs(
        full_image=ImageInput(path=full_path),
        degraded_image=ImageInput(path=degraded_path, transform=transform),
        free_caption=caption,
        task_evidence=evidence,
        task_visible_evidence=visible,
        task_infer_evidence=None if infer is None else str(infer),
        task_solve_evidence=None if solve is None else str(solve),
    )
    inputs.validate(require_paths=config.require_paths)
    return inputs


def _condition_inputs_from_mapping(
    raw_inputs: Mapping[str, Any],
    config: OfflineScoringConfig,
) -> ConditionInputs:
    full = raw_inputs["full_image"]
    degraded = raw_inputs["degraded_image"]
    if not isinstance(full, Mapping) or not isinstance(degraded, Mapping):
        raise ValueError("condition_inputs image entries must be mappings")
    transform = degraded.get("transform") or {"type": "gaussian_blur", "sigma": float(config.blur_sigma)}
    inputs = ConditionInputs(
        full_image=ImageInput(path=_resolve_path(str(full["path"]), config.data_root)),
        degraded_image=ImageInput(
            path=_resolve_path(str(degraded["path"]), config.data_root),
            transform=dict(transform),
        ),
        free_caption=str(raw_inputs["free_caption"]),
        task_evidence=str(raw_inputs["task_evidence"]),
        task_visible_evidence=(
            None if raw_inputs.get("task_visible_evidence") is None else str(raw_inputs["task_visible_evidence"])
        ),
        task_infer_evidence=(
            None if raw_inputs.get("task_infer_evidence") is None else str(raw_inputs["task_infer_evidence"])
        ),
        task_solve_evidence=(
            None if raw_inputs.get("task_solve_evidence") is None else str(raw_inputs["task_solve_evidence"])
        ),
        verified_facts=(
            None if raw_inputs.get("verified_facts") is None else str(raw_inputs["verified_facts"])
        ),
        verified_facts_source=(
            None
            if raw_inputs.get("verified_facts_source") is None
            else str(raw_inputs["verified_facts_source"])
        ),
    )
    inputs.validate(require_paths=config.require_paths)
    return inputs


def _coalesce_text(record: Mapping[str, Any], keys: Sequence[str], default: str) -> str:
    value = _first_present(record, keys)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def build_protocol_smoke_response(record: Mapping[str, Any]) -> str:
    """Construct a structured student response for protocol-smoke datasets."""

    answer = extract_answer(record)
    answer_text = answer.strip() if answer and answer.strip() else "unknown"
    return (
        "<visual_evidence>\n"
        "One question-relevant object is visible in the image.\n"
        "</visual_evidence>\n"
        "<reasoning>\n"
        "The visible evidence is sufficient to determine the answer.\n"
        "</reasoning>\n"
        "<answer>\n"
        f"{answer_text}\n"
        "</answer>"
    )


@dataclass(frozen=True)
class StudentResponse:
    text: str
    token_ids: tuple[int, ...]


def resolve_student_response(
    record: Mapping[str, Any],
    *,
    mode: str,
    tokenizer: OfflineScoringTokenizer,
    student_responses: Mapping[str, Mapping[str, Any]] | None,
    question_id: str,
) -> StudentResponse:
    """Resolve the student response text and token IDs for one record."""

    if mode == "protocol_smoke":
        text = build_protocol_smoke_response(record)
        return StudentResponse(text=text, token_ids=tuple(tokenizer.encode(text)))

    if mode != "student":
        raise ValueError(f"unknown response mode: {mode}")
    if not student_responses:
        raise ValueError("student mode requires a student response mapping")

    entry = student_responses.get(question_id)
    if entry is None:
        raise KeyError(f"no student response found for sample {question_id}")

    text = _first_present(entry, ("response_text", "response", "text", "answer"))
    if not isinstance(text, str) or not text:
        raise ValueError(f"student response for {question_id} has no usable text")

    raw_token_ids = _first_present(entry, ("response_token_ids", "token_ids", "tokens"))
    if raw_token_ids is None:
        token_ids = tuple(tokenizer.encode(text))
    else:
        token_ids = tuple(int(item) for item in raw_token_ids)
    if not token_ids:
        raise ValueError(f"student response for {question_id} produced no token IDs")
    return StudentResponse(text=text, token_ids=token_ids)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _mask_to_spans(mask: torch.Tensor) -> list[list[int]]:
    spans: list[list[int]] = []
    start: int | None = None
    values = mask.tolist()
    for index, flag in enumerate(values):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append([start, index])
            start = None
    if start is not None:
        spans.append([start, len(values)])
    return spans


def _serialize_topk(scores: TeacherTopK) -> dict[str, Any]:
    scores.validate()
    token_ids = scores.token_ids[0]
    log_probs = scores.log_probs[0]
    return {
        "token_ids": token_ids.tolist(),
        "log_probs": log_probs.tolist(),
        "tail_log_prob": None if scores.tail_log_prob is None else scores.tail_log_prob[0].tolist(),
        "entropy": None if scores.entropy is None else scores.entropy[0].tolist(),
    }


def _serialize_chunks(chunk_masks: Any) -> dict[str, Any]:
    return {
        "visible_evidence": _mask_to_spans(chunk_masks.visible_evidence_mask),
        "visual_evidence": _mask_to_spans(chunk_masks.visual_evidence_mask),
        "diagram_inference": _mask_to_spans(chunk_masks.diagram_inference_mask),
        "reasoning": _mask_to_spans(chunk_masks.reasoning_mask),
        "answer": _mask_to_spans(chunk_masks.answer_mask),
        "format_valid": chunk_masks.format_valid,
        "errors": list(chunk_masks.errors),
        "fallback": chunk_masks.fallback,
        "token_counts": chunk_masks.token_counts,
        "chunk_labels": list(chunk_masks.chunk_labels),
    }


# ---------------------------------------------------------------------------
# Per-record scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OfflineScoreRecord:
    payload: dict[str, Any]

    def to_json_line(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False, separators=(",", ":"))


def score_record(
    record: Mapping[str, Any],
    *,
    position: int,
    config: OfflineScoringConfig,
    tokenizer: OfflineScoringTokenizer,
    teacher_client: TeacherClient,
    tokenizer_hash: str,
    mode: str,
    student_responses: Mapping[str, Mapping[str, Any]] | None,
) -> OfflineScoreRecord:
    """Score one Vision-OPD record under all configured teacher conditions."""

    question = extract_question(record)
    image_paths = extract_image_paths(record, config.data_root)
    source_index = extract_source_index(record, position)
    question_id = extract_question_id(record, position)
    sample_uid = f"{config.source_dataset}:{question_id}"

    condition_inputs = build_condition_inputs_for_record(record, config)
    student = resolve_student_response(
        record,
        mode=mode,
        tokenizer=tokenizer,
        student_responses=student_responses,
        question_id=question_id,
    )

    chunk_masks = parse_response_chunks(
        student.token_ids,
        student.text,
        tokenizer,
        fallback=config.chunk_fallback,
    )

    teacher_scores = score_teacher_conditions(
        student.token_ids,
        question,
        condition_inputs,
        config.conditions,
        teacher_client,
        response_text=student.text,
        request_prefix=sample_uid,
    )
    signals = compute_condition_signals(teacher_scores)

    payload = {
        "sample_uid": sample_uid,
        "source_dataset": config.source_dataset,
        "source_index": source_index,
        "question_id": question_id,
        "question": question,
        "image_paths": image_paths,
        "condition_inputs": condition_inputs.to_dict(),
        "response_text": student.text,
        "response_token_ids": list(student.token_ids),
        "chunk_spans": _serialize_chunks(chunk_masks),
        "tokenizer_hash": tokenizer_hash,
        "protocol_version": teacher_client.metadata.protocol_version,
        "condition_scores": {
            condition.value: _serialize_topk(teacher_scores[condition])
            for condition in config.conditions
            if condition in teacher_scores
        },
        "condition_signals": {
            name: signal[0].tolist() for name, signal in signals.items()
        },
        "metadata": {
            "top_k": teacher_client.metadata.top_k,
            "blur_sigma": float(config.blur_sigma),
            "teacher_model_id": teacher_client.metadata.model_id,
            "teacher_git_revision": teacher_client.metadata.git_revision,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response_mode": mode,
        },
    }
    return OfflineScoreRecord(payload=payload)


# ---------------------------------------------------------------------------
# Dataset-level orchestration
# ---------------------------------------------------------------------------


@dataclass
class OfflineScoringResult:
    records: list[OfflineScoreRecord] = field(default_factory=list)
    jsonl_path: Path | None = None
    parquet_path: Path | None = None


def iter_offline_scores(
    records: Iterable[Mapping[str, Any]],
    *,
    config: OfflineScoringConfig,
    tokenizer: OfflineScoringTokenizer,
    teacher_client: TeacherClient,
    mode: str,
    student_responses: Mapping[str, Mapping[str, Any]] | None = None,
    limit: int | None = None,
) -> Iterator[OfflineScoreRecord]:
    """Yield offline-score records for each input record up to ``limit``."""

    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    for position, record in enumerate(records):
        if limit is not None and position >= limit:
            return
        yield score_record(
            record,
            position=position,
            config=config,
            tokenizer=tokenizer,
            teacher_client=teacher_client,
            tokenizer_hash=tokenizer_hash,
            mode=mode,
            student_responses=student_responses,
        )


def default_output_dir(output_root: str | Path) -> Path:
    return Path(output_root).expanduser() / OFFLINE_SCORE_SUBDIR


def write_offline_scores(
    records: Iterable[OfflineScoreRecord],
    *,
    output_dir: str | Path,
    filename: str,
    write_parquet: bool = False,
) -> OfflineScoringResult:
    """Write offline-score records to JSONL and optionally Parquet."""

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{filename}.jsonl"

    materialized = list(records)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in materialized:
            handle.write(record.to_json_line())
            handle.write("\n")

    result = OfflineScoringResult(records=materialized, jsonl_path=jsonl_path)
    if write_parquet:
        result.parquet_path = _write_parquet(materialized, output_dir / f"{filename}.parquet")
    return result


def _write_parquet(records: Sequence[OfflineScoreRecord], path: Path) -> Path:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("writing Parquet requires pandas and pyarrow") from exc

    frame = pd.DataFrame(
        {"sample_uid": record.payload["sample_uid"], "payload": record.to_json_line()}
        for record in records
    )
    frame.to_parquet(path)
    return path


# ---------------------------------------------------------------------------
# Synthetic VStar-like fixtures for smoke runs and tests
# ---------------------------------------------------------------------------


def make_smoke_dataset(num_samples: int = 16, *, dataset_name: str = "vstar") -> list[dict[str, Any]]:
    """Build a deterministic Vision-OPD-style dataset for smoke runs.

    The records mimic the VStar high-resolution VQA layout (images + query +
    answer + question_id) without referencing any real data files.
    """

    questions = (
        "What color is the small object near the center?",
        "How many people are visible in the scene?",
        "What text appears on the sign?",
        "Which direction is the arrow pointing?",
    )
    answers = ("red", "two", "exit", "left")
    records: list[dict[str, Any]] = []
    for index in range(num_samples):
        choice = index % len(questions)
        records.append(
            {
                "question_id": f"{dataset_name}-{index:04d}",
                "index": index,
                "images": [f"images/{dataset_name}/{index:04d}.jpg"],
                "query": questions[choice],
                "answer": answers[choice],
            }
        )
    return records

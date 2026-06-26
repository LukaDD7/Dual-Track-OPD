"""Trainable Vision-OPD 2C FC-OPD offline score builder.

This module converts the validated structured student-rollout audit path into
offline score JSONL rows that can be consumed by the existing FC-OPD offline
loss and real-student smoke pipeline. It keeps Vision-OPD crop/bbox images as
metadata only; the scored conditions use the original/global image and its
deterministic gaussian-blur counterpart.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .chunk_parser import parse_response_chunks
from .conditions import Condition
from .dataset_adapters import load_normalized_records
from .dataset_signal_audit import (
    build_audit_condition_inputs,
    compute_pairwise_kd_gradient_cosines,
    detect_task_evidence_leakage,
    hash_text,
    hash_token_ids,
    materialize_gaussian_blur,
)
from .offline_loss import offline_record_to_tensors
from .offline_scoring import DEFAULT_TEACHER_URL, _serialize_chunks, _serialize_topk, derive_degraded_path
from .signal_decomposer import compute_condition_signals
from .student_rollout_signal_audit import (
    HFQwenStudentRolloutGenerator,
    ROLLOUT_RESPONSE_FORMATS,
    StudentRolloutGenerator,
    build_rollout_prompt,
    response_format_note,
    rollout_seed,
)
from .teacher_client import TeacherClient, score_teacher_conditions
from .teacher_protocol import tokenizer_fingerprint
from ..utils.git_state import get_git_state

TWO_CONDITIONS: tuple[Condition, ...] = (Condition.FULL, Condition.BLUR)
DEFAULT_STUDENT_MODEL = "hf:$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct"
CROP_BBOX_POLICY = "metadata_only_default_no_crop_condition"


@dataclass(frozen=True)
class VisionOPD2COfflineScoreConfig:
    dataset: Path
    output_jsonl: Path
    summary_json: Path
    dataset_type: str = "vision_opd_parquet"
    source_dataset: str = "vision-opd-6k"
    student_model_path: str = DEFAULT_STUDENT_MODEL
    teacher_url: str = DEFAULT_TEACHER_URL
    conditions: tuple[Condition, ...] = TWO_CONDITIONS
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
    blur_sigma: float = 2.0
    materialize_degraded_images: bool = False
    degraded_dir: str | None = None
    skip_existing: bool = False
    resume: bool = False
    shard_id: int | None = None
    num_shards: int | None = None
    min_response_tokens_for_warning: int = 16
    high_signal_threshold: float = 0.01
    high_cosine_threshold: float = 0.98

    def __post_init__(self) -> None:
        if self.dataset_type != "vision_opd_parquet":
            valid = {"vision_opd_parquet", "vision_opd_json", "generic_jsonl", "auto"}
            if self.dataset_type not in valid:
                raise ValueError(f"dataset_type must be one of {sorted(valid)}")
        if self.conditions != TWO_CONDITIONS:
            raise ValueError("Vision-OPD 2C offline builder currently requires conditions full,blur")
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if self.end_index is not None and self.end_index < self.start_index:
            raise ValueError("end_index must be >= start_index")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided")
        if self.rollouts_per_prompt <= 0:
            raise ValueError("rollouts_per_prompt must be positive")
        if self.rollout_response_format not in ROLLOUT_RESPONSE_FORMATS:
            raise ValueError(f"rollout_response_format must be one of {ROLLOUT_RESPONSE_FORMATS}")
        if self.blur_sigma <= 0:
            raise ValueError("blur_sigma must be positive")
        if (self.shard_id is None) != (self.num_shards is None):
            raise ValueError("shard_id and num_shards must be provided together")
        if self.num_shards is not None:
            if self.num_shards <= 0:
                raise ValueError("num_shards must be positive")
            if self.shard_id is None or not 0 <= self.shard_id < self.num_shards:
                raise ValueError("shard_id must be in [0, num_shards)")


@dataclass
class VisionOPD2COfflineScoreResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    output_jsonl: Path | None = None
    summary_json: Path | None = None
    skipped_existing: bool = False


def run_vision_opd_2c_offline_score_builder(
    config: VisionOPD2COfflineScoreConfig,
    *,
    rollout_generator: StudentRolloutGenerator | None = None,
    teacher_client: TeacherClient | None = None,
) -> VisionOPD2COfflineScoreResult:
    """Build and write trainable 2C offline score rows."""

    if config.skip_existing and config.output_jsonl.is_file() and config.summary_json.is_file():
        summary = json.loads(config.summary_json.read_text(encoding="utf-8"))
        return VisionOPD2COfflineScoreResult(
            summary=summary,
            output_jsonl=config.output_jsonl,
            summary_json=config.summary_json,
            skipped_existing=True,
        )

    normalized = load_normalized_records(
        config.dataset,
        config.dataset_type,
        source_dataset=config.source_dataset,
    )
    selected = _select_records(normalized, config)

    rollout_generator = rollout_generator or HFQwenStudentRolloutGenerator(_rollout_config(config))
    tokenizer = rollout_generator.tokenizer
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    teacher_client = teacher_client or TeacherClient(
        config.teacher_url,
        expected_tokenizer_hash=tokenizer_hash,
    )

    existing_rows: list[dict[str, Any]] = []
    existing_rollout_uids: set[str] = set()
    if config.resume and config.output_jsonl.is_file():
        existing_rows = _read_jsonl(config.output_jsonl)
        existing_rollout_uids = {str(row.get("rollout_uid", "")) for row in existing_rows}

    new_rows: list[dict[str, Any]] = []
    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if config.resume and config.output_jsonl.is_file() else "w"
    with config.output_jsonl.open(mode, encoding="utf-8") as handle:
        for prompt_index, record in selected:
            for row in _build_rows_for_record(
                record,
                prompt_index=prompt_index,
                config=config,
                rollout_generator=rollout_generator,
                tokenizer_hash=tokenizer_hash,
                teacher_client=teacher_client,
            ):
                if row["rollout_uid"] in existing_rollout_uids:
                    continue
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                new_rows.append(row)

    rows = [*existing_rows, *new_rows]
    summary = summarize_vision_opd_2c_rows(
        rows,
        config=config,
        num_prompts=len(selected),
        teacher_client=teacher_client,
    )
    config.summary_json.parent.mkdir(parents=True, exist_ok=True)
    config.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return VisionOPD2COfflineScoreResult(
        rows=rows,
        summary=summary,
        output_jsonl=config.output_jsonl,
        summary_json=config.summary_json,
    )


def _build_rows_for_record(
    record: Mapping[str, Any],
    *,
    prompt_index: int,
    config: VisionOPD2COfflineScoreConfig,
    rollout_generator: StudentRolloutGenerator,
    tokenizer_hash: str,
    teacher_client: TeacherClient,
) -> list[dict[str, Any]]:
    base_sample_uid = str(record.get("sample_uid") or f"{config.source_dataset}:{prompt_index}")
    source_index = int(record.get("source_index", prompt_index))
    question_id = str(record.get("question_id", source_index))
    question = str(record.get("question") or record.get("query") or "").strip()
    image_path = str(record.get("image_path") or "")
    degraded_image_path = derive_degraded_path(image_path, config.blur_sigma, config.degraded_dir)
    if config.materialize_degraded_images and image_path:
        materialize_gaussian_blur(image_path, degraded_image_path, config.blur_sigma)

    prompt = build_rollout_prompt(question, response_format=config.rollout_response_format)
    condition_inputs = build_audit_condition_inputs(
        record,
        question=question,
        answer=None,
        full_image_path=image_path,
        degraded_image_path=degraded_image_path,
        blur_sigma=config.blur_sigma,
        task_evidence_mode="none",
    )
    answer = record.get("answer") or record.get("gold") or record.get("answer_metadata")
    leakage_warnings = detect_task_evidence_leakage(
        answer=None if answer is None else str(answer),
        task_evidence=condition_inputs.task_evidence,
    )

    rows: list[dict[str, Any]] = []
    for rollout_id in range(config.rollouts_per_prompt):
        generation_seed = rollout_seed(
            base_seed=config.seed,
            source_index=source_index,
            rollout_id=rollout_id,
        )
        response_text = rollout_generator.generate(
            question=question,
            image_path=image_path,
            prompt_text=prompt.text,
            seed=generation_seed,
        )
        token_ids = tuple(int(item) for item in rollout_generator.tokenizer.encode(response_text))
        chunk_masks = parse_response_chunks(
            token_ids,
            response_text,
            rollout_generator.tokenizer,
            fallback="all_reasoning",
        )
        rollout_uid = f"{base_sample_uid}:rollout-{rollout_id}"
        teacher_scores = score_teacher_conditions(
            token_ids,
            question,
            condition_inputs,
            config.conditions,
            teacher_client,
            response_text=response_text,
            request_prefix=rollout_uid,
        )
        signals = compute_condition_signals(teacher_scores)
        gradient_cosines = compute_pairwise_kd_gradient_cosines(teacher_scores)
        condition_signal_values = {
            name: [float(item) for item in signal[0].tolist()]
            for name, signal in signals.items()
        }

        row = {
            "sample_uid": rollout_uid,
            "prompt_sample_uid": base_sample_uid,
            "source_dataset": config.source_dataset,
            "source_index": source_index,
            "question_id": question_id,
            "rollout_id": rollout_id,
            "rollout_uid": rollout_uid,
            "question": question,
            "clean_question_text": question,
            "image_path": image_path,
            "degraded_image_path": degraded_image_path,
            "image_exists": Path(image_path).expanduser().is_file() if image_path else False,
            "degraded_image_exists": (
                Path(degraded_image_path).expanduser().is_file() if degraded_image_path else False
            ),
            "bbox_image_path": str(record.get("bbox_image_path") or ""),
            "bbox_image_paths": list(record.get("bbox_image_paths") or []),
            "bbox_image_exists": bool(record.get("bbox_image_exists", False)),
            "crop_bbox_policy": CROP_BBOX_POLICY,
            "bbox_metadata_unused_by_default": True,
            "response_source": "student_rollout",
            "response_text": response_text,
            "response_token_ids": list(token_ids),
            "response_token_count": len(token_ids),
            "response_length": len(token_ids),
            "response_text_hash": hash_text(response_text),
            "response_token_hash": hash_token_ids(token_ids),
            "tokenizer_hash": tokenizer_hash,
            "prompt_hash": hash_text(question),
            "rollout_prompt": prompt.text,
            "rollout_prompt_hash": hash_text(prompt.text),
            "generation_seed": generation_seed,
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
                "model_id": teacher_client.metadata.model_id,
                "protocol_version": teacher_client.metadata.protocol_version,
                "tokenizer_hash": teacher_client.metadata.tokenizer_hash,
                "top_k": teacher_client.metadata.top_k,
                "git_revision": teacher_client.metadata.git_revision,
            },
            "teacher_model_id": teacher_client.metadata.model_id,
            "protocol_version": teacher_client.metadata.protocol_version,
            "conditions": [condition.value for condition in config.conditions],
            "condition_inputs": condition_inputs.to_dict(),
            "condition_scores": {
                condition.value: _serialize_topk(teacher_scores[condition])
                for condition in config.conditions
            },
            "condition_signals": condition_signal_values,
            "condition_signal_summary": {
                name: _series_stats(values) for name, values in condition_signal_values.items()
            },
            "gradient_cosines": gradient_cosines,
            "chunk_spans": _serialize_chunks(chunk_masks),
            "leakage_warnings": leakage_warnings,
            "errors": [],
            "metadata": {
                "crop_bbox_policy": CROP_BBOX_POLICY,
                "bbox_metadata_unused_by_default": True,
                "blur_sigma": float(config.blur_sigma),
                "source_builder": "vision_opd_2c_offline_score_builder",
                "response_source": "student_rollout",
            },
        }
        rows.append(row)
    return rows


def summarize_vision_opd_2c_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: VisionOPD2COfflineScoreConfig,
    num_prompts: int,
    teacher_client: TeacherClient | None = None,
) -> dict[str, Any]:
    response_lengths = [int(row.get("response_token_count", row.get("response_length", 0))) for row in rows]
    response_text_hashes = [str(row.get("response_text_hash", "")) for row in rows]
    response_token_hashes = [str(row.get("response_token_hash", "")) for row in rows]
    response_source_counts = Counter(str(row.get("response_source", "unknown")) for row in rows)
    duplicate_rate, unique_per_prompt, identical_count = _rollout_diversity(rows)
    visual_values = _collect_signal(rows, "visual_detail")
    teacher_errors = sum(_has_error(row, "teacher_error:") for row in rows)
    student_errors = sum(_has_error(row, "student_generation_error:") for row in rows)
    expected_rows = num_prompts * config.rollouts_per_prompt
    actual_rows = len(rows)
    output_size = config.output_jsonl.stat().st_size if config.output_jsonl.is_file() else None
    git = _safe_git_state()
    tokenizer_hash = str(rows[0].get("tokenizer_hash", "")) if rows else ""
    teacher_model_id = (
        teacher_client.metadata.model_id
        if teacher_client is not None
        else str(rows[0].get("teacher_model_id", ""))
        if rows
        else ""
    )
    visual_stats = _series_stats(visual_values)
    gradient_cosines = _summarize_cosines(rows)
    return {
        "source_dataset": config.source_dataset,
        "dataset_path": str(config.dataset),
        "dataset_type": config.dataset_type,
        "num_prompts": num_prompts,
        "rollouts_per_prompt": config.rollouts_per_prompt,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "num_samples_scored": actual_rows - teacher_errors,
        "teacher_error_rate": _rate(teacher_errors, actual_rows),
        "student_generation_error_rate": _rate(student_errors, actual_rows),
        "image_missing_rate": _rate(sum(not bool(row.get("image_exists")) for row in rows), actual_rows),
        "degraded_image_missing_rate": _rate(
            sum(not bool(row.get("degraded_image_exists")) for row in rows), actual_rows
        ),
        "mean_response_tokens": _mean_or_none(response_lengths),
        "response_length_p50": visual_p50(response_lengths),
        "response_length_p90": visual_p90(response_lengths),
        "unique_response_text_hash_count": len({value for value in response_text_hashes if value}),
        "unique_response_token_hash_count": len({value for value in response_token_hashes if value}),
        "unique_response_per_prompt_mean": unique_per_prompt,
        "duplicate_rollout_rate": duplicate_rate,
        "all_rollouts_identical_per_prompt_count": identical_count,
        "short_response_rate": _rate(
            sum(length < config.min_response_tokens_for_warning for length in response_lengths),
            len(response_lengths),
        ),
        "response_source_counts": dict(response_source_counts),
        "mean_full_vs_blur_divergence": visual_stats["mean"],
        "visual_detail_signal": visual_stats,
        "high_visual_signal_token_ratio": _high_ratio(visual_values, config.high_signal_threshold),
        "condition_collapse": {
            "visual_detail_collapsed": (
                visual_stats["mean"] is not None
                and float(visual_stats["mean"]) < config.high_signal_threshold
            ),
            "high_condition_gradient_cosine": any(
                value is not None and abs(float(value)) >= config.high_cosine_threshold
                for value in gradient_cosines.values()
            ),
        },
        "gradient_cosines": gradient_cosines,
        "leakage_warnings": [
            {"sample_uid": str(row.get("sample_uid", "")), "warnings": row.get("leakage_warnings")}
            for row in rows
            if row.get("leakage_warnings")
        ],
        "conditions": [condition.value for condition in config.conditions],
        "tokenizer_hash": tokenizer_hash,
        "teacher_model_id": teacher_model_id,
        "student_model_path": config.student_model_path,
        "output_jsonl": str(config.output_jsonl),
        "summary_json": str(config.summary_json),
        "output_file_size_bytes": output_size,
        "git": git,
        "builder_config": {
            "start_index": config.start_index,
            "end_index": config.end_index,
            "limit": config.limit,
            "shard_id": config.shard_id,
            "num_shards": config.num_shards,
            "rollout_response_format": config.rollout_response_format,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
            "seed": config.seed,
            "blur_sigma": config.blur_sigma,
            "crop_bbox_policy": CROP_BBOX_POLICY,
        },
    }


def validate_vision_opd_2c_offline_scores(path: str | Path) -> dict[str, Any]:
    """Validate trainability and schema compatibility for the 2C JSONL."""

    rows = _read_jsonl(path)
    errors: list[str] = []
    for index, row in enumerate(rows):
        prefix = str(row.get("sample_uid") or index)
        try:
            offline_record_to_tensors(row)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{prefix}: offline tensor conversion failed: {exc}")
            continue
        if list(row.get("conditions", [])) != ["full", "blur"]:
            errors.append(f"{prefix}: conditions must be ['full', 'blur']")
        if row.get("crop_bbox_policy") != CROP_BBOX_POLICY:
            errors.append(f"{prefix}: crop bbox policy is not metadata-only")
        if row.get("bbox_metadata_unused_by_default") is not True:
            errors.append(f"{prefix}: bbox metadata is not marked unused by default")
        if row.get("leakage_warnings"):
            errors.append(f"{prefix}: leakage warnings are present")
        teacher_meta = row.get("teacher_metadata")
        if not isinstance(teacher_meta, Mapping):
            errors.append(f"{prefix}: missing teacher_metadata")
        elif row.get("tokenizer_hash") != teacher_meta.get("tokenizer_hash"):
            errors.append(f"{prefix}: tokenizer_hash does not match teacher metadata")
        response_len = len(row.get("response_token_ids", []))
        scores = row.get("condition_scores", {})
        if not isinstance(scores, Mapping):
            errors.append(f"{prefix}: condition_scores is missing")
            continue
        for condition in ("full", "blur"):
            block = scores.get(condition)
            if not isinstance(block, Mapping):
                errors.append(f"{prefix}: missing {condition} scores")
                continue
            seq_len = len(block.get("token_ids", []))
            if seq_len != response_len:
                errors.append(f"{prefix}: {condition} T={seq_len} != response length {response_len}")
            if len(block.get("log_probs", [])) != response_len:
                errors.append(f"{prefix}: {condition} log_probs length mismatch")
            tail = block.get("tail_log_prob")
            entropy = block.get("entropy")
            if tail is not None and len(tail) != response_len:
                errors.append(f"{prefix}: {condition} tail length mismatch")
            if entropy is not None and len(entropy) != response_len:
                errors.append(f"{prefix}: {condition} entropy length mismatch")
        condition_inputs = row.get("condition_inputs")
        if isinstance(condition_inputs, Mapping):
            full = condition_inputs.get("full_image", {})
            degraded = condition_inputs.get("degraded_image", {})
            if isinstance(full, Mapping) and full.get("path") != row.get("image_path"):
                errors.append(f"{prefix}: full condition does not use image_path")
            if isinstance(degraded, Mapping) and degraded.get("path") != row.get("degraded_image_path"):
                errors.append(f"{prefix}: blur condition does not use degraded_image_path")
            bbox_paths = {str(row.get("bbox_image_path") or ""), *map(str, row.get("bbox_image_paths", []))}
            used_paths = {str(full.get("path", "")), str(degraded.get("path", ""))}
            if "" in bbox_paths:
                bbox_paths.remove("")
            if bbox_paths & used_paths:
                errors.append(f"{prefix}: bbox/crop image is used by a default condition")
        else:
            errors.append(f"{prefix}: missing condition_inputs")
    return {
        "path": str(path),
        "num_rows": len(rows),
        "valid": bool(rows) and not errors,
        "errors": errors,
    }


def _select_records(
    records: Sequence[Mapping[str, Any]],
    config: VisionOPD2COfflineScoreConfig,
) -> list[tuple[int, Mapping[str, Any]]]:
    end = config.end_index if config.end_index is not None else len(records)
    selected: list[tuple[int, Mapping[str, Any]]] = []
    for index in range(config.start_index, min(end, len(records))):
        if config.num_shards is not None and index % config.num_shards != config.shard_id:
            continue
        selected.append((index, records[index]))
        if config.limit is not None and len(selected) >= config.limit:
            break
    return selected


def _rollout_config(config: VisionOPD2COfflineScoreConfig) -> Any:
    from .student_rollout_signal_audit import StudentRolloutAuditConfig

    return StudentRolloutAuditConfig(
        dataset=config.dataset,
        dataset_type=config.dataset_type,
        source_dataset=config.source_dataset,
        limit=config.limit or 1,
        conditions=config.conditions,
        teacher_url=config.teacher_url,
        student_model_path=config.student_model_path,
        rollouts_per_prompt=config.rollouts_per_prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
        seed=config.seed,
        device=config.device,
        dtype=config.dtype,
        blur_sigma=config.blur_sigma,
        degraded_dir=config.degraded_dir,
        materialize_degraded_images=config.materialize_degraded_images,
        rollout_response_format=config.rollout_response_format,
        min_response_tokens_for_warning=config.min_response_tokens_for_warning,
    )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path).expanduser()
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row in {path} is not an object")
            rows.append(dict(value))
    return rows


def _rollout_diversity(rows: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None, int]:
    by_prompt: dict[str, list[str]] = {}
    for row in rows:
        by_prompt.setdefault(str(row.get("prompt_sample_uid", row.get("sample_uid", ""))), []).append(
            str(row.get("response_text_hash", ""))
        )
    if not by_prompt:
        return None, None, 0
    duplicate_count = 0
    total = 0
    unique_counts: list[int] = []
    identical_count = 0
    for hashes in by_prompt.values():
        counts = Counter(hashes)
        duplicate_count += sum(max(0, count - 1) for count in counts.values())
        total += len(hashes)
        unique_counts.append(len(set(hashes)))
        if len(hashes) > 1 and len(set(hashes)) == 1:
            identical_count += 1
    return (
        None if total == 0 else duplicate_count / total,
        None if not unique_counts else sum(unique_counts) / len(unique_counts),
        identical_count,
    )


def _collect_signal(rows: Sequence[Mapping[str, Any]], name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        signal_map = row.get("condition_signals", {})
        if isinstance(signal_map, Mapping) and name in signal_map:
            values.extend(float(item) for item in signal_map[name])
    return values


def _series_stats(values: Sequence[float]) -> dict[str, float | None]:
    cleaned = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not cleaned:
        return {"mean": None, "p50": None, "p90": None}
    return {
        "mean": float(mean(cleaned)),
        "p50": _quantile(cleaned, 0.5),
        "p90": _quantile(cleaned, 0.9),
    }


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(sorted_values[lower])
    weight = index - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def visual_p50(values: Sequence[int]) -> float | None:
    stats = _series_stats([float(value) for value in values])
    return stats["p50"]


def visual_p90(values: Sequence[int]) -> float | None:
    stats = _series_stats([float(value) for value in values])
    return stats["p90"]


def _mean_or_none(values: Iterable[float]) -> float | None:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    return float(mean(cleaned)) if cleaned else None


def _rate(count: int, total: int) -> float | None:
    if total == 0:
        return None
    return float(count) / float(total)


def _high_ratio(values: Sequence[float], threshold: float) -> float | None:
    if not values:
        return None
    return sum(float(value) >= threshold for value in values) / len(values)


def _has_error(row: Mapping[str, Any], prefix: str) -> bool:
    return any(str(error).startswith(prefix) for error in row.get("errors", []))


def _summarize_cosines(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    keys = sorted({str(key) for row in rows for key in dict(row.get("gradient_cosines", {})).keys()})
    return {
        key: _mean_or_none(
            float(dict(row.get("gradient_cosines", {}))[key])
            for row in rows
            if key in dict(row.get("gradient_cosines", {}))
        )
        for key in keys
    }


def _safe_git_state() -> dict[str, Any]:
    try:
        state = get_git_state(Path.cwd())
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    if hasattr(state, "to_dict"):
        return state.to_dict()
    if isinstance(state, Mapping):
        return dict(state)
    return {"value": str(state)}


def _parse_conditions(value: str) -> tuple[Condition, ...]:
    conditions = tuple(Condition(item.strip()) for item in value.split(",") if item.strip())
    if conditions != TWO_CONDITIONS:
        raise argparse.ArgumentTypeError("this builder requires --conditions full,blur")
    return conditions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-type", default="vision_opd_parquet")
    parser.add_argument("--source-dataset", default="vision-opd-6k")
    parser.add_argument("--student-model-path", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--teacher-url", default=DEFAULT_TEACHER_URL)
    parser.add_argument("--conditions", type=_parse_conditions, default=TWO_CONDITIONS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--rollout-response-format", choices=ROLLOUT_RESPONSE_FORMATS, default="fc_opd_structured")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--blur-sigma", type=float, default=2.0)
    parser.add_argument("--materialize-degraded-images", action="store_true")
    parser.add_argument("--degraded-dir", default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        report = validate_vision_opd_2c_offline_scores(args.output_jsonl)
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 1

    result = run_vision_opd_2c_offline_score_builder(
        VisionOPD2COfflineScoreConfig(
            dataset=args.dataset,
            dataset_type=args.dataset_type,
            source_dataset=args.source_dataset,
            student_model_path=args.student_model_path,
            teacher_url=args.teacher_url,
            conditions=args.conditions,
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
            blur_sigma=args.blur_sigma,
            materialize_degraded_images=args.materialize_degraded_images,
            degraded_dir=args.degraded_dir,
            output_jsonl=args.output_jsonl,
            summary_json=args.summary_json,
            skip_existing=args.skip_existing,
            resume=args.resume,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
    )
    validation = validate_vision_opd_2c_offline_scores(args.output_jsonl)
    print(f"wrote {result.summary.get('actual_rows', 0)}/{result.summary.get('expected_rows', 0)} trainable rows")
    print(f"jsonl: {result.output_jsonl}")
    print(f"summary: {result.summary_json}")
    print(json.dumps({"validation_valid": validation["valid"], "validation_errors": validation["errors"][:5]}))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

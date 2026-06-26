"""Dataset signal audit before selecting FC-OPD training data.

The audit runs the same offline teacher-scoring protocol used by FC-OPD, but
with dataset-selection outputs: compact per-sample JSONL plus aggregate signal
summaries. It does not load a student model, does not import verl, and can run
in ``--dry-run`` mode to validate dataset fields before a teacher is available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import torch

from .conditions import Condition, ConditionInputs, ImageInput
from .dataset_adapters import load_normalized_records, load_raw_records, task_evidence_mode_label
from .offline_scoring import (
    DEFAULT_CONDITIONS,
    DEFAULT_TEACHER_URL,
    ByteTokenizer,
    OfflineScoringTokenizer,
    derive_degraded_path,
    extract_answer,
    extract_image_paths,
    extract_question,
    extract_question_id,
    extract_source_index,
)
from .signal_decomposer import TeacherTopK, compute_condition_signals
from .teacher_client import TeacherClient
from .teacher_protocol import tokenizer_fingerprint

DEFAULT_TOKENIZER = "hf:$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct"
DEFAULT_LIMIT = 128
HIGH_SIGNAL_THRESHOLD = 0.01


@dataclass(frozen=True)
class DatasetAuditConfig:
    dataset: Path
    dataset_type: str = "auto"
    source_dataset: str = "candidate"
    limit: int = DEFAULT_LIMIT
    teacher_url: str = DEFAULT_TEACHER_URL
    tokenizer: str = DEFAULT_TOKENIZER
    conditions: tuple[Condition, ...] = DEFAULT_CONDITIONS
    blur_sigma: float = 2.0
    degraded_dir: str | None = None
    materialize_degraded_images: bool = False
    output_dir: Path | None = None
    skip_existing: bool = False
    dry_run: bool = False
    task_evidence_mode: str = "none"
    high_signal_threshold: float = HIGH_SIGNAL_THRESHOLD
    high_cosine_threshold: float = 0.98

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.blur_sigma <= 0:
            raise ValueError("blur_sigma must be positive")
        if not self.conditions:
            raise ValueError("at least one condition is required")
        if self.task_evidence_mode not in {
            "none",
            "free_caption",
            "question_conditioned_caption",
            "oracle_answer",
        }:
            raise ValueError("unsupported task_evidence_mode")
        valid_types = {"vision_opd_json", "vision_opd_parquet", "generic_jsonl", "auto"}
        if self.dataset_type not in valid_types:
            raise ValueError(f"dataset_type must be one of {sorted(valid_types)}")


@dataclass
class DatasetAuditResult:
    samples: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    jsonl_path: Path | None = None
    summary_json_path: Path | None = None
    summary_tsv_path: Path | None = None
    summary_md_path: Path | None = None
    skipped_existing: bool = False


def load_candidate_records(path: str | Path, dataset_type: str = "auto") -> list[dict[str, Any]]:
    """Load a candidate dataset for signal auditing."""

    return load_raw_records(path, dataset_type)


def build_tokenizer(spec: str) -> OfflineScoringTokenizer:
    """Build the audit tokenizer from ``byte`` or ``hf:<name-or-path>``."""

    spec = os.path.expandvars(spec)
    if spec == "byte":
        return ByteTokenizer()
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer  # lazy import keeps dry tests light

        return AutoTokenizer.from_pretrained(spec[len("hf:") :])
    raise ValueError("tokenizer must be 'byte' or 'hf:<name-or-path>'")


def run_dataset_signal_audit(
    config: DatasetAuditConfig,
    *,
    tokenizer: OfflineScoringTokenizer | None = None,
    teacher_client: TeacherClient | None = None,
) -> DatasetAuditResult:
    """Run the dataset-selection signal audit and return in-memory outputs."""

    output_dir = config.output_dir
    paths = _output_paths(output_dir, config.source_dataset) if output_dir else {}
    if config.skip_existing and paths and all(path.is_file() for path in paths.values()):
        return DatasetAuditResult(
            jsonl_path=paths["jsonl"],
            summary_json_path=paths["summary_json"],
            summary_tsv_path=paths["summary_tsv"],
            summary_md_path=paths["summary_md"],
            skipped_existing=True,
        )

    records = load_normalized_records(
        config.dataset,
        config.dataset_type,
        source_dataset=config.source_dataset,
    )
    requested = min(config.limit, len(records))
    tokenizer = tokenizer or build_tokenizer(config.tokenizer)
    tokenizer_hash = tokenizer_fingerprint(tokenizer)

    if config.dry_run:
        teacher_client = None
        teacher_model_id = "dry_run"
    else:
        teacher_client = teacher_client or TeacherClient(
            config.teacher_url,
            expected_tokenizer_hash=tokenizer_hash,
        )
        teacher_model_id = teacher_client.metadata.model_id

    samples = [
        audit_record(
            record,
            position=position,
            config=config,
            tokenizer=tokenizer,
            tokenizer_hash=tokenizer_hash,
            teacher_client=teacher_client,
            teacher_model_id=teacher_model_id,
        )
        for position, record in enumerate(records[:requested])
    ]
    summary = summarize_audit_samples(samples, config=config, num_loaded=len(records))
    result = DatasetAuditResult(samples=samples, summary=summary)
    if output_dir:
        _write_audit_outputs(result, paths)
    return result


def audit_record(
    record: Mapping[str, Any],
    *,
    position: int,
    config: DatasetAuditConfig,
    tokenizer: OfflineScoringTokenizer,
    tokenizer_hash: str,
    teacher_client: TeacherClient | None,
    teacher_model_id: str,
) -> dict[str, Any]:
    """Audit one sample, keeping errors local to the per-sample record."""

    errors: list[str] = []
    leakage_warnings: list[str] = []
    condition_scores: dict[Condition, TeacherTopK] = {}
    condition_signal_values: dict[str, list[float]] = {}
    gradient_cosines: dict[str, float] = {}

    source_index = _safe_extract_source_index(record, position)
    question_id = _safe_extract_question_id(record, position)
    sample_uid = f"{config.source_dataset}:{question_id}"
    question = ""
    answer: str | None = None
    image_path = ""
    degraded_image_path = ""
    prompt_length: int | None = None
    response_token_ids: tuple[int, ...] = ()

    question = str(record.get("question") or record.get("query") or "")
    if not question.strip():
        try:
            question = extract_question(record)
        except Exception as exc:  # noqa: BLE001 - audit must keep moving
            errors.append(f"question_error: {exc}")

    answer_value = record.get("answer") or record.get("gold")
    answer = None if answer_value is None else str(answer_value)
    if answer is None:
        try:
            answer = extract_answer(record)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"answer_error: {exc}")

    try:
        image_paths = extract_image_paths(record, None)
        image_path = image_paths[0]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"image_error: {exc}")

    if image_path:
        degraded_image_path = derive_degraded_path(
            image_path,
            config.blur_sigma,
            config.degraded_dir,
        )
        if config.materialize_degraded_images:
            try:
                materialize_gaussian_blur(image_path, degraded_image_path, config.blur_sigma)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"degraded_image_error: {exc}")

    response_text = build_audit_response(record, answer=answer)
    response_token_ids = tuple(int(item) for item in tokenizer.encode(response_text))
    if question:
        prompt_length = len(tokenizer.encode(question))

    condition_inputs: ConditionInputs | None = None
    if question and image_path and response_token_ids:
        condition_inputs = build_audit_condition_inputs(
            record,
            question=question,
            answer=answer,
            full_image_path=image_path,
            degraded_image_path=degraded_image_path,
            blur_sigma=config.blur_sigma,
            task_evidence_mode=config.task_evidence_mode,
        )
        leakage_warnings.extend(
            detect_task_evidence_leakage(
                answer=answer,
                task_evidence=condition_inputs.task_evidence,
            )
        )
        if config.task_evidence_mode == "oracle_answer":
            leakage_warnings.append("oracle_mode_enabled")

    if teacher_client is not None and condition_inputs is not None:
        try:
            from .teacher_client import score_teacher_conditions

            condition_scores = score_teacher_conditions(
                response_token_ids,
                question,
                condition_inputs,
                config.conditions,
                teacher_client,
                response_text=response_text,
                request_prefix=sample_uid,
            )
            signals = compute_condition_signals(condition_scores)
            condition_signal_values = {
                name: [float(item) for item in signal[0].tolist()]
                for name, signal in signals.items()
            }
            gradient_cosines = compute_pairwise_kd_gradient_cosines(condition_scores)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"teacher_error: {exc}")

    score_availability = {
        condition.value: condition in condition_scores for condition in config.conditions
    }
    entropy_means = {
        condition.value: _tensor_mean(scores.entropy)
        for condition, scores in condition_scores.items()
        if scores.entropy is not None
    }

    return {
        "sample_uid": sample_uid,
        "source_dataset": config.source_dataset,
        "source_index": source_index,
        "question_id": question_id,
        "image_path": image_path,
        "degraded_image_path": degraded_image_path,
        "bbox_image_path": str(record.get("bbox_image_path") or ""),
        "bbox_image_paths": list(record.get("bbox_image_paths") or []),
        "bbox_image_exists": bool(record.get("bbox_image_exists", False)),
        "image_exists": Path(image_path).expanduser().is_file() if image_path else False,
        "degraded_image_exists": (
            Path(degraded_image_path).expanduser().is_file() if degraded_image_path else False
        ),
        "question": question,
        "answer_available": bool(answer and str(answer).strip()),
        "response_length": len(response_token_ids),
        "prompt_length": prompt_length,
        "tokenizer_hash": tokenizer_hash,
        "teacher_model_id": teacher_model_id,
        "condition_score_available": score_availability,
        "condition_entropy_mean": entropy_means,
        "condition_signals": condition_signal_values,
        "condition_signal_summary": {
            name: _series_stats(values) for name, values in condition_signal_values.items()
        },
        "gradient_cosines": gradient_cosines,
        "leakage_warnings": leakage_warnings,
        "errors": errors,
        "metadata": {
            "crop_bbox_policy": "metadata_only_default_no_crop_condition",
            "task_evidence_mode": config.task_evidence_mode,
            "task_evidence_mode_label": task_evidence_mode_label(config.task_evidence_mode),
        },
    }


def build_audit_response(record: Mapping[str, Any], *, answer: str | None) -> str:
    """Resolve a fixed response string for teacher scoring during the audit."""

    for key in ("response_text", "student_response", "response", "completion"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    answer_text = answer.strip() if isinstance(answer, str) and answer.strip() else "unavailable"
    return (
        "<visual_evidence>\n"
        "The image should be inspected for the question-relevant visual details.\n"
        "</visual_evidence>\n"
        "<reasoning>\n"
        "Use the visual evidence and the question to derive the final answer.\n"
        "</reasoning>\n"
        "<answer>\n"
        f"{answer_text}\n"
        "</answer>"
    )


def build_audit_condition_inputs(
    record: Mapping[str, Any],
    *,
    question: str,
    answer: str | None,
    full_image_path: str,
    degraded_image_path: str,
    blur_sigma: float,
    task_evidence_mode: str = "none",
) -> ConditionInputs:
    """Build non-oracle condition inputs for audit scoring."""

    free_caption = _coalesce_text(
        record,
        ("free_caption", "caption", "image_caption"),
        "Audit caption placeholder: image-only evidence is available, but no gold answer is injected.",
    )
    task_evidence = build_task_evidence(
        record,
        question=question,
        answer=answer,
        mode=task_evidence_mode,
        free_caption=free_caption,
    )
    inputs = ConditionInputs(
        full_image=ImageInput(path=full_image_path),
        degraded_image=ImageInput(
            path=degraded_image_path,
            transform={"type": "gaussian_blur", "sigma": float(blur_sigma)},
        ),
        free_caption=free_caption,
        task_evidence=task_evidence,
    )
    inputs.validate(require_paths=False)
    return inputs


def build_task_evidence(
    record: Mapping[str, Any],
    *,
    question: str,
    answer: str | None,
    mode: str,
    free_caption: str,
) -> str:
    if mode == "none":
        return "No task-specific evidence is provided in this non-oracle audit."
    if mode == "free_caption":
        return free_caption
    if mode == "question_conditioned_caption":
        evidence = _coalesce_text(
            record,
            ("task_evidence", "task_extraction", "evidence"),
            "",
        )
        if evidence:
            return evidence
        return f"Question-conditioned evidence request for audit only: {question}"
    if mode == "oracle_answer":
        answer_text = answer.strip() if isinstance(answer, str) and answer.strip() else "unknown"
        return f"ORACLE UPPER-BOUND evidence. Gold answer: {answer_text}"
    raise ValueError(f"unsupported task_evidence_mode: {mode}")


def materialize_gaussian_blur(source_path: str, target_path: str, sigma: float) -> None:
    """Create a deterministic blurred image artifact for the blur condition."""

    from PIL import Image, ImageFilter

    source = Path(source_path).expanduser()
    target = Path(target_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"source image does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.filter(ImageFilter.GaussianBlur(radius=float(sigma))).save(target)


def detect_task_evidence_leakage(*, answer: str | None, task_evidence: str) -> list[str]:
    """Warn if task evidence appears to contain the gold answer."""

    if not isinstance(answer, str) or not answer.strip():
        return []
    normalized_answer = _normalize_for_leakage(answer)
    normalized_evidence = _normalize_for_leakage(task_evidence)
    if len(normalized_answer) >= 3 and normalized_answer in normalized_evidence:
        return ["task_evidence_contains_answer"]
    return []


def summarize_audit_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    config: DatasetAuditConfig,
    num_loaded: int,
) -> dict[str, Any]:
    """Aggregate per-sample audit output into training-set selection metrics."""

    requested = len(samples)
    teacher_error_count = sum(_has_error(sample, "teacher_error:") for sample in samples)
    scored = [
        sample for sample in samples
        if sample.get("condition_score_available")
        and all(bool(value) for value in dict(sample["condition_score_available"]).values())
    ]
    visual_values = _collect_signal(samples, "visual_detail")
    task_values = _collect_signal(samples, "task_extraction")
    entropy_by_condition = {
        condition.value: _mean_or_none(
            float(dict(sample.get("condition_entropy_mean", {}))[condition.value])
            for sample in samples
            if condition.value in dict(sample.get("condition_entropy_mean", {}))
        )
        for condition in config.conditions
    }
    gradient_cosines = _summarize_cosines(samples)
    high_cosines = {
        key: value
        for key, value in gradient_cosines.items()
        if value is not None and abs(float(value)) >= config.high_cosine_threshold
    }
    leakage = [
        {
            "sample_uid": str(sample.get("sample_uid", "")),
            "warnings": list(sample.get("leakage_warnings", [])),
        }
        for sample in samples
        if sample.get("leakage_warnings")
    ]
    visual_stats = _series_stats(visual_values)
    task_stats = _series_stats(task_values)
    collapse = {
        "visual_detail_collapsed": (
            visual_stats["mean"] is not None
            and float(visual_stats["mean"]) < config.high_signal_threshold
        ),
        "task_extraction_collapsed": (
            task_stats["mean"] is not None
            and float(task_stats["mean"]) < config.high_signal_threshold
        ),
        "missing_full_or_blur_scores": not visual_values and not config.dry_run,
        "missing_task_or_free_scores": not task_values and not config.dry_run,
        "high_condition_gradient_cosine": bool(high_cosines),
        "high_condition_gradient_cosines": high_cosines,
    }
    return {
        "source_dataset": config.source_dataset,
        "dataset_path": str(config.dataset),
        "dataset_type": config.dataset_type,
        "num_samples_loaded": num_loaded,
        "num_samples_requested": requested,
        "num_samples_scored": len(scored),
        "image_missing_rate": _rate(
            sum(not bool(sample.get("image_exists")) for sample in samples),
            requested,
        ),
        "degraded_image_missing_rate": _rate(
            sum(not bool(sample.get("degraded_image_exists")) for sample in samples),
            requested,
        ),
        "teacher_error_rate": _rate(teacher_error_count, requested),
        "mean_response_tokens": _mean_or_none(
            int(sample.get("response_length", 0)) for sample in samples
        ),
        "mean_entropy": entropy_by_condition,
        "mean_full_vs_blur_divergence": visual_stats["mean"],
        "mean_task_vs_free_divergence": task_stats["mean"],
        "visual_detail_signal": visual_stats,
        "task_extraction_signal": task_stats,
        "high_visual_signal_token_ratio": _high_ratio(
            visual_values, config.high_signal_threshold
        ),
        "high_task_signal_token_ratio": _high_ratio(task_values, config.high_signal_threshold),
        "condition_collapse": collapse,
        "leakage_warnings": leakage,
        "gradient_cosines": gradient_cosines,
        "gradient_cosine_diagnostic": {
            "label": "condition redundancy diagnostic",
            "is_ideal_alignment": False,
            "note": (
                "Pairwise condition KD-gradient cosine detects redundancy/collapse; "
                "it is not success-conditioned ideal-gradient alignment."
            ),
            "high_cosine_threshold": config.high_cosine_threshold,
        },
        "decision_rule": [
            "1. Vision-OPD-6K for the first fair objective comparison.",
            "2. Geometry3K for VA-OPD-style visual-math comparison.",
            "3. ViRL39K or mixed visual reasoning data only after non-collapsed signal is confirmed.",
        ],
        "dry_run": config.dry_run,
        "conditions": [condition.value for condition in config.conditions],
        "high_signal_threshold": config.high_signal_threshold,
    }


def compute_pairwise_kd_gradient_cosines(
    scores: Mapping[Condition, TeacherTopK],
) -> dict[str, float]:
    """Approximate condition-gradient redundancy under a fixed top-k support.

    This is not ideal-gradient alignment. It compares ``p_student - p_teacher``
    vectors using a uniform synthetic student distribution over the union of the
    two top-k supports plus a tail bucket.
    """

    pairs = (
        (Condition.FULL, Condition.BLUR),
        (Condition.FULL, Condition.FREE),
        (Condition.FULL, Condition.TASK),
        (Condition.BLUR, Condition.TASK),
    )
    output: dict[str, float] = {}
    for left, right in pairs:
        if left not in scores or right not in scores:
            continue
        output[f"cos_g_{left.value}_{right.value}"] = _mean_gradient_cosine(
            scores[left],
            scores[right],
        )
    return output


def _mean_gradient_cosine(left: TeacherTopK, right: TeacherTopK) -> float:
    left.validate()
    right.validate()
    seq_len = left.token_ids.shape[1]
    values: list[float] = []
    for index in range(seq_len):
        p = _topk_distribution_at(left, index)
        q = _topk_distribution_at(right, index)
        keys = sorted(p.keys() | q.keys(), key=str)
        uniform = 1.0 / len(keys)
        g_left = torch.tensor([uniform - p.get(key, 0.0) for key in keys], dtype=torch.float32)
        g_right = torch.tensor([uniform - q.get(key, 0.0) for key in keys], dtype=torch.float32)
        denom = g_left.norm() * g_right.norm()
        if denom > 0:
            values.append(float(torch.dot(g_left, g_right).div(denom).item()))
    return float(mean(values)) if values else 0.0


def _topk_distribution_at(scores: TeacherTopK, token_index: int) -> dict[int | str, float]:
    ids = scores.token_ids[0, token_index].tolist()
    log_probs = scores.log_probs[0, token_index].tolist()
    masses: dict[int | str, float] = {}
    for token_id, log_prob in zip(ids, log_probs, strict=True):
        masses[int(token_id)] = masses.get(int(token_id), 0.0) + math.exp(float(log_prob))
    if scores.tail_log_prob is not None:
        masses["__tail__"] = math.exp(float(scores.tail_log_prob[0, token_index].item()))
    total = sum(masses.values()) or 1.0
    return {key: value / total for key, value in masses.items()}


def _resolve_dataset_type(path: Path, dataset_type: str) -> str:
    if dataset_type != "auto":
        return dataset_type
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "vision_opd_parquet"
    if suffix == ".jsonl":
        return "generic_jsonl"
    return "vision_opd_json"


def _parse_conditions(value: str) -> tuple[Condition, ...]:
    conditions = tuple(Condition(item.strip()) for item in value.split(",") if item.strip())
    if not conditions:
        raise argparse.ArgumentTypeError("at least one condition is required")
    return conditions


def _coalesce_text(record: Mapping[str, Any], keys: Sequence[str], default: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _safe_extract_source_index(record: Mapping[str, Any], position: int) -> int:
    try:
        return extract_source_index(record, position)
    except Exception:  # noqa: BLE001
        return position


def _safe_extract_question_id(record: Mapping[str, Any], position: int) -> str:
    try:
        return extract_question_id(record, position)
    except Exception:  # noqa: BLE001
        return str(position)


def _normalize_for_leakage(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _tensor_mean(tensor: torch.Tensor | None) -> float | None:
    if tensor is None or tensor.numel() == 0:
        return None
    return float(tensor.float().mean().item())


def _collect_signal(samples: Sequence[Mapping[str, Any]], name: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        signal_map = sample.get("condition_signals", {})
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
    if not sorted_values:
        raise ValueError("cannot compute quantile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(sorted_values[lower])
    weight = index - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


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


def _has_error(sample: Mapping[str, Any], prefix: str) -> bool:
    return any(str(error).startswith(prefix) for error in sample.get("errors", []))


def _summarize_cosines(samples: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    keys = sorted(
        {
            str(key)
            for sample in samples
            for key in dict(sample.get("gradient_cosines", {})).keys()
        }
    )
    return {
        key: _mean_or_none(
            float(dict(sample.get("gradient_cosines", {}))[key])
            for sample in samples
            if key in dict(sample.get("gradient_cosines", {}))
        )
        for key in keys
    }


def _output_paths(output_dir: Path | str | None, source_dataset: str) -> dict[str, Path]:
    if output_dir is None:
        raise ValueError("output_dir is required")
    root = Path(output_dir).expanduser()
    stem = f"{source_dataset}_dataset_signal_audit"
    return {
        "jsonl": root / f"{stem}.jsonl",
        "summary_json": root / f"{stem}_summary.json",
        "summary_tsv": root / f"{stem}_summary.tsv",
        "summary_md": root / f"{stem}_summary.md",
    }


def _write_audit_outputs(result: DatasetAuditResult, paths: Mapping[str, Path]) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    with paths["jsonl"].open("w", encoding="utf-8") as handle:
        for sample in result.samples:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    paths["summary_json"].write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["summary_tsv"].write_text(_summary_tsv(result.summary), encoding="utf-8")
    paths["summary_md"].write_text(_summary_markdown(result.summary), encoding="utf-8")
    result.jsonl_path = paths["jsonl"]
    result.summary_json_path = paths["summary_json"]
    result.summary_tsv_path = paths["summary_tsv"]
    result.summary_md_path = paths["summary_md"]


def _summary_tsv(summary: Mapping[str, Any]) -> str:
    rows = [
        ("metric", "value"),
        ("source_dataset", summary.get("source_dataset")),
        ("num_samples_requested", summary.get("num_samples_requested")),
        ("num_samples_scored", summary.get("num_samples_scored")),
        ("image_missing_rate", summary.get("image_missing_rate")),
        ("degraded_image_missing_rate", summary.get("degraded_image_missing_rate")),
        ("teacher_error_rate", summary.get("teacher_error_rate")),
        ("mean_response_tokens", summary.get("mean_response_tokens")),
        ("mean_full_vs_blur_divergence", summary.get("mean_full_vs_blur_divergence")),
        ("mean_task_vs_free_divergence", summary.get("mean_task_vs_free_divergence")),
        (
            "high_visual_signal_token_ratio",
            summary.get("high_visual_signal_token_ratio"),
        ),
        ("high_task_signal_token_ratio", summary.get("high_task_signal_token_ratio")),
    ]
    return "\n".join(f"{key}\t{value}" for key, value in rows) + "\n"


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    visual = dict(summary.get("visual_detail_signal", {}))
    task = dict(summary.get("task_extraction_signal", {}))
    collapse = dict(summary.get("condition_collapse", {}))
    return (
        f"# FC-OPD Dataset Signal Audit: {summary.get('source_dataset')}\n\n"
        f"- Samples requested: {summary.get('num_samples_requested')}\n"
        f"- Samples scored: {summary.get('num_samples_scored')}\n"
        f"- Image missing rate: {summary.get('image_missing_rate')}\n"
        f"- Degraded image missing rate: {summary.get('degraded_image_missing_rate')}\n"
        f"- Teacher error rate: {summary.get('teacher_error_rate')}\n"
        f"- Mean response tokens: {summary.get('mean_response_tokens')}\n\n"
        "## Signal\n\n"
        f"- Visual detail mean / p50 / p90: {visual.get('mean')} / "
        f"{visual.get('p50')} / {visual.get('p90')}\n"
        f"- Task extraction mean / p50 / p90: {task.get('mean')} / "
        f"{task.get('p50')} / {task.get('p90')}\n"
        f"- High visual signal token ratio: {summary.get('high_visual_signal_token_ratio')}\n"
        f"- High task signal token ratio: {summary.get('high_task_signal_token_ratio')}\n\n"
        "## Collapse Indicators\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in collapse.items())
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--dataset-type",
        choices=("vision_opd_json", "vision_opd_parquet", "generic_jsonl", "auto"),
        default="auto",
    )
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--teacher-url", default=DEFAULT_TEACHER_URL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--conditions", type=_parse_conditions, default=DEFAULT_CONDITIONS)
    parser.add_argument("--blur-sigma", type=float, default=2.0)
    parser.add_argument("--degraded-dir")
    parser.add_argument("--materialize-degraded-images", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--task-evidence-mode",
        choices=("none", "free_caption", "question_conditioned_caption", "oracle_answer"),
        default="none",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DatasetAuditConfig(
        dataset=args.dataset,
        dataset_type=args.dataset_type,
        source_dataset=args.source_dataset,
        limit=args.limit,
        teacher_url=args.teacher_url,
        tokenizer=args.tokenizer,
        conditions=args.conditions,
        blur_sigma=args.blur_sigma,
        degraded_dir=args.degraded_dir,
        materialize_degraded_images=args.materialize_degraded_images,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        task_evidence_mode=args.task_evidence_mode,
    )
    result = run_dataset_signal_audit(config)
    if result.skipped_existing:
        print(f"dataset signal audit already exists under {args.output_dir}")
        return 0
    print(
        "wrote dataset signal audit: "
        f"{result.jsonl_path}, {result.summary_json_path}, "
        f"{result.summary_tsv_path}, {result.summary_md_path}"
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

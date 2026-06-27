"""Clean-data 4C FC-OPD offline score builder."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .alignment import compute_alignment_weights, compute_rollout_group_stats, default_outcome_metadata
from .chunk_parser import parse_response_chunks
from .conditions import Condition, ConditionInputs, ImageInput
from .dataset_signal_audit import compute_pairwise_kd_gradient_cosines, hash_text, hash_token_ids, materialize_gaussian_blur
from .evidence_generation import validate_evidence_row
from .geometry3k_adapter import load_geometry3k_records
from .offline_loss import offline_record_to_tensors
from .offline_scoring import DEFAULT_TEACHER_URL, _serialize_chunks, _serialize_topk
from .signal_decomposer import compute_condition_signals
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

CONDITION_SETS: dict[str, tuple[Condition, ...]] = {
    "4c-legacy": (Condition.FULL, Condition.DEGRADED, Condition.FREE, Condition.TASK),
    "4c-clean": (Condition.FULL, Condition.DEGRADED, Condition.FREE, Condition.TASK_VISIBLE),
    "5c-infer": (
        Condition.FULL,
        Condition.DEGRADED,
        Condition.FREE,
        Condition.TASK_VISIBLE,
        Condition.TASK_INFER,
    ),
    "6c-solve": (
        Condition.FULL,
        Condition.DEGRADED,
        Condition.FREE,
        Condition.TASK_VISIBLE,
        Condition.TASK_INFER,
        Condition.TASK_SOLVE,
    ),
}
FOUR_CLEAN_CONDITIONS: tuple[Condition, ...] = (
    Condition.FULL,
    Condition.DEGRADED,
    Condition.FREE,
    Condition.TASK,
)
DEFAULT_STUDENT_MODEL = "hf:$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct"


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

    def __post_init__(self) -> None:
        if self.dataset_type != "geometry3k":
            raise ValueError("clean-data 4C builder currently supports Geometry3K first")
        if self.rollout_response_format not in ROLLOUT_RESPONSE_FORMATS:
            raise ValueError(f"rollout_response_format must be one of {ROLLOUT_RESPONSE_FORMATS}")
        if self.degraded_mode not in {"lowres_10pct_nearest", "gaussian_blur_s2"}:
            raise ValueError("degraded_mode must be lowres_10pct_nearest or gaussian_blur_s2")
        if self.condition_set not in CONDITION_SETS:
            raise ValueError(f"condition_set must be one of {sorted(CONDITION_SETS)}")


@dataclass
class FourConditionOfflineBuilderResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


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
    evidence = {str(row["sample_uid"]): row for row in _read_jsonl(config.evidence_cache)}
    rollout_generator = rollout_generator or HFQwenStudentRolloutGenerator(_rollout_config(config))
    tokenizer_hash = tokenizer_fingerprint(rollout_generator.tokenizer)
    teacher_client = teacher_client or TeacherClient(config.teacher_url, expected_tokenizer_hash=tokenizer_hash)
    existing = _read_jsonl(config.output_jsonl) if config.resume else []
    existing_uids = {str(row.get("rollout_uid", "")) for row in existing}
    new_rows: list[dict[str, Any]] = []
    mode = "a" if config.resume and config.output_jsonl.is_file() else "w"
    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.output_jsonl.open(mode, encoding="utf-8") as handle:
        for record in records:
            evidence_row = evidence.get(str(record["sample_uid"]))
            if evidence_row is None:
                raise ValueError(f"missing evidence cache row for {record['sample_uid']}")
            errors = validate_evidence_row(evidence_row)
            if errors:
                raise ValueError(f"evidence cache validation failed for {record['sample_uid']}: {errors}")
            for row in _build_rows(record, evidence_row, config, rollout_generator, teacher_client, tokenizer_hash):
                if row["rollout_uid"] in existing_uids:
                    continue
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                new_rows.append(row)
    rows = [*existing, *new_rows]
    _attach_group_stats(rows)
    if rows:
        config.output_jsonl.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
    summary = summarize_rows(rows, config=config, teacher_client=teacher_client)
    config.summary_json.parent.mkdir(parents=True, exist_ok=True)
    config.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return FourConditionOfflineBuilderResult(rows=rows, summary=summary)


def _build_rows(
    record: Mapping[str, Any],
    evidence_row: Mapping[str, Any],
    config: FourConditionOfflineBuilderConfig,
    rollout_generator: StudentRolloutGenerator,
    teacher_client: TeacherClient,
    tokenizer_hash: str,
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
        seed = rollout_seed(base_seed=config.seed, source_index=source_index, rollout_id=rollout_id)
        response_text = rollout_generator.generate(question=question, image_path=image_path, prompt_text=prompt.text, seed=seed)
        token_ids = tuple(int(item) for item in rollout_generator.tokenizer.encode(response_text))
        chunk_masks = parse_response_chunks(token_ids, response_text, rollout_generator.tokenizer, fallback="all_reasoning")
        rollout_uid = f"{record['sample_uid']}:rollout-{rollout_id}"
        teacher_scores = score_teacher_conditions(
            token_ids,
            question,
            condition_inputs,
            conditions,
            teacher_client,
            response_text=response_text,
            request_prefix=rollout_uid,
        )
        signals = compute_condition_signals(teacher_scores)
        condition_scores = {
            condition.value: {**_serialize_topk(teacher_scores[condition]), "top_k": teacher_client.metadata.top_k}
            for condition in conditions
        }
        outcome = default_outcome_metadata(record)
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
            "image_path": image_path,
            "degraded_image_path": degraded_path,
            "degraded_mode": config.degraded_mode,
            "free_caption": evidence_row["free_caption"],
            "task_evidence": evidence_row["task_evidence"],
            "task_visible_evidence": evidence_row.get("task_visible_evidence", evidence_row["task_evidence"]),
            "task_infer_evidence": evidence_row.get("task_infer_evidence"),
            "task_solve_evidence": evidence_row.get("task_solve_evidence"),
            "evidence_cache_uid": evidence_row["sample_uid"],
            "response_source": "student_rollout",
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
    source = Path(image_path).expanduser()
    suffix = source.suffix or ".png"
    degraded_dir = Path(config.degraded_dir).expanduser() if config.degraded_dir else source.parent
    if config.degraded_mode == "gaussian_blur_s2":
        target = degraded_dir / f"{source.stem}.gaussian_blur_s2{suffix}"
        materialize_gaussian_blur(str(source), str(target), config.blur_sigma)
        return str(target)
    target = degraded_dir / f"{source.stem}.lowres_10pct_nearest{suffix}"
    if target.is_file():
        return str(target)
    from PIL import Image

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        original = image.convert("RGB")
        width, height = original.size
        low = original.resize((max(1, width // 10), max(1, height // 10)), Image.Resampling.NEAREST)
        low.resize((width, height), Image.Resampling.NEAREST).save(target)
    return str(target)


def _degraded_transform(config: FourConditionOfflineBuilderConfig) -> dict[str, Any]:
    if config.degraded_mode == "gaussian_blur_s2":
        return {"type": "gaussian_blur", "sigma": float(config.blur_sigma), "degraded_mode": config.degraded_mode}
    return {"type": "lowres_nearest", "scale": 0.1, "degraded_mode": config.degraded_mode}


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
        if row.get("degraded_mode") not in {"lowres_10pct_nearest", "gaussian_blur_s2"}:
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
    return {"num_rows": len(rows), "valid": bool(rows) and not errors, "errors": errors}


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: FourConditionOfflineBuilderConfig,
    teacher_client: TeacherClient,
) -> dict[str, Any]:
    lengths = [int(row.get("response_token_count", 0)) for row in rows]
    conditions = [condition.value for condition in CONDITION_SETS[config.condition_set]]
    parse_success = [
        bool(row.get("chunk_spans", {}).get("format_valid", False))
        for row in rows
    ]
    chunk_counts: Counter[str] = Counter()
    delta_means: dict[str, float | None] = {}
    for row in rows:
        token_counts = row.get("chunk_spans", {}).get("token_counts", {})
        if isinstance(token_counts, Mapping):
            for key, value in token_counts.items():
                chunk_counts[str(key)] += int(value)
        signal_summary = row.get("condition_signal_summary", {})
        if isinstance(signal_summary, Mapping):
            for name in ("visual_detail_delta", "task_selection_delta", "diagram_infer_delta", "solve_delta"):
                block = signal_summary.get(name)
                if isinstance(block, Mapping) and block.get("mean") is not None:
                    delta_means.setdefault(name, 0.0)
    for name in ("visual_detail_delta", "task_selection_delta", "diagram_infer_delta", "solve_delta"):
        values = []
        for row in rows:
            block = row.get("condition_signal_summary", {}).get(name, {})
            if isinstance(block, Mapping) and block.get("mean") is not None:
                values.append(float(block["mean"]))
        delta_means[name] = None if not values else float(mean(values))
    return {
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
        "delta_means": delta_means,
        "gate_ready_fields_available": bool(rows),
        "response_length_p50": _quantile(lengths, 0.5),
        "response_length_p90": _quantile(lengths, 0.9),
        "unique_response_text_hash_count": len({row.get("response_text_hash") for row in rows}),
        "response_source_counts": dict(Counter(str(row.get("response_source", "unknown")) for row in rows)),
        "teacher_model_id": teacher_client.metadata.model_id,
        "tokenizer_hash": rows[0].get("tokenizer_hash") if rows else "",
        "output_jsonl": str(config.output_jsonl),
        "summary_json": str(config.summary_json),
        "red_box_contaminated": False,
        "not_main_experiment": False,
    }


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
    if not values:
        return {"mean": None, "p50": None, "p90": None}
    return {"mean": float(mean(values)), "p50": _quantile(values, 0.5), "p90": _quantile(values, 0.9)}


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
    parser.add_argument("--degraded-mode", choices=("lowres_10pct_nearest", "gaussian_blur_s2"), default="lowres_10pct_nearest")
    parser.add_argument("--degraded-dir")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--condition-set", choices=tuple(CONDITION_SETS), default="4c-clean")
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
        )
    )
    validation = validate_four_condition_rows(args.output_jsonl, condition_set=args.condition_set)
    print(json.dumps({**result.summary, "validation_valid": validation["valid"]}, indent=2))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

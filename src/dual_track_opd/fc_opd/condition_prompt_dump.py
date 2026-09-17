"""Dump exact FC-OPD condition prompts for manual audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .conditions import Condition, ConditionInputs, ImageInput
from .dataset_adapters import (
    discover_vision_opd_train_parquet,
    load_normalized_records,
    task_evidence_mode_label,
)
from .dataset_signal_audit import materialize_gaussian_blur
from .offline_scoring import DEFAULT_CONDITIONS, derive_degraded_path
from .teacher_prompts import render_teacher_prompt


TASK_EVIDENCE_MODES = (
    "none",
    "free_caption",
    "question_conditioned_caption",
    "oracle_answer",
)


@dataclass(frozen=True)
class PromptDumpConfig:
    dataset: Path
    dataset_type: str = "auto"
    source_dataset: str = "candidate"
    limit: int = 3
    conditions: tuple[Condition, ...] = DEFAULT_CONDITIONS
    blur_sigma: float = 2.0
    degraded_dir: str | None = None
    materialize_degraded_images: bool = False
    task_evidence_mode: str = "none"
    output: Path = Path("fc_opd_condition_prompts.md")
    include_images_as_paths: bool = False

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.blur_sigma <= 0:
            raise ValueError("blur_sigma must be positive")
        if self.task_evidence_mode not in TASK_EVIDENCE_MODES:
            raise ValueError(f"task_evidence_mode must be one of {TASK_EVIDENCE_MODES}")


@dataclass(frozen=True)
class PromptDumpResult:
    records: list[dict[str, Any]]
    markdown_path: Path
    jsonl_path: Path


def dump_condition_prompts(config: PromptDumpConfig) -> PromptDumpResult:
    records = load_normalized_records(
        config.dataset,
        config.dataset_type,
        source_dataset=config.source_dataset,
    )
    dumped = [
        dump_prompt_record(record, config=config)
        for record in records[: config.limit]
    ]
    markdown_path, jsonl_path = _output_paths(config.output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(dumped, config), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in dumped:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return PromptDumpResult(records=dumped, markdown_path=markdown_path, jsonl_path=jsonl_path)


def dump_prompt_record(record: Mapping[str, Any], *, config: PromptDumpConfig) -> dict[str, Any]:
    question = str(record.get("question") or "")
    answer = _optional_text(record.get("answer") or record.get("gold"))
    image_path = str(record.get("image_path") or "")
    degraded_image_path = derive_degraded_path(image_path, config.blur_sigma, config.degraded_dir)
    if config.materialize_degraded_images and image_path:
        materialize_gaussian_blur(image_path, degraded_image_path, config.blur_sigma)

    free_caption = _coalesce_text(
        record,
        ("free_caption", "caption", "image_caption"),
        "Prompt audit placeholder: weak image-only evidence, no gold answer.",
    )
    task_evidence = build_prompt_task_evidence(
        record,
        question=question,
        answer=answer,
        free_caption=free_caption,
        mode=config.task_evidence_mode,
    )
    inputs = ConditionInputs(
        full_image=ImageInput(path=image_path),
        degraded_image=ImageInput(
            path=degraded_image_path,
            transform={"type": "gaussian_blur", "sigma": float(config.blur_sigma)},
        ),
        free_caption=free_caption,
        task_evidence=task_evidence,
    )
    inputs.validate(require_paths=False)

    rendered: dict[str, Any] = {}
    for condition in config.conditions:
        prompt = render_teacher_prompt(condition, question, inputs)
        rendered[condition.value] = {
            "messages": prompt.messages,
            "image_paths": prompt.image_paths,
            "text": _rendered_text(prompt.messages),
        }

    leakage_flags = detect_prompt_leakage(
        task_evidence=task_evidence,
        answer=answer,
        options=[str(item) for item in record.get("options", [])],
        oracle_mode=config.task_evidence_mode == "oracle_answer",
    )
    warnings = []
    if config.task_evidence_mode != "oracle_answer":
        warnings.append("answer/gold is not allowed in non-oracle task_evidence")
    if config.task_evidence_mode == "oracle_answer":
        warnings.append("oracle_answer mode is an oracle/upper-bound construction")

    return {
        "sample_uid": str(record.get("sample_uid") or ""),
        "source_dataset": config.source_dataset,
        "source_index": int(record.get("source_index", 0)),
        "image_path": image_path,
        "degraded_image_path": degraded_image_path,
        "bbox_image_path": str(record.get("bbox_image_path") or ""),
        "bbox_image_paths": list(record.get("bbox_image_paths") or []),
        "bbox_image_exists": bool(record.get("bbox_image_exists", False)),
        "question": question,
        "answer": answer,
        "gold": answer,
        "non_oracle_warning": warnings,
        "rendered_prompts": rendered,
        "condition_metadata": {
            "conditions": [condition.value for condition in config.conditions],
            "blur_sigma": float(config.blur_sigma),
            "task_evidence_mode": config.task_evidence_mode,
            "task_evidence_mode_label": task_evidence_mode_label(config.task_evidence_mode),
            "crop_bbox_policy": "metadata_only_default_no_crop_condition",
            "include_images_as_paths": config.include_images_as_paths,
        },
        "leakage_flags": leakage_flags,
    }


def build_prompt_task_evidence(
    record: Mapping[str, Any],
    *,
    question: str,
    answer: str | None,
    free_caption: str,
    mode: str,
) -> str:
    if mode == "none":
        return "No task-specific evidence is provided in this non-oracle prompt audit."
    if mode == "free_caption":
        return free_caption
    if mode == "question_conditioned_caption":
        evidence = _coalesce_text(record, ("task_evidence", "task_extraction", "evidence"), "")
        if evidence:
            return evidence
        return f"Question-conditioned evidence request for prompt audit: {question}"
    if mode == "oracle_answer":
        answer_text = answer.strip() if answer else "unknown"
        return f"ORACLE UPPER-BOUND evidence. Gold answer: {answer_text}"
    raise ValueError(f"unsupported task_evidence_mode: {mode}")


def detect_prompt_leakage(
    *,
    task_evidence: str,
    answer: str | None,
    options: Sequence[str],
    oracle_mode: bool,
) -> dict[str, bool]:
    normalized_evidence = _normalize(task_evidence)
    normalized_answer = _normalize(answer or "")
    contains_answer = len(normalized_answer) >= 3 and normalized_answer in normalized_evidence
    option_letter = ""
    if answer is not None and len(answer.strip()) == 1 and answer.strip().isalpha():
        option_letter = answer.strip().lower()
    contains_option_letter = bool(
        option_letter and f"goldanswer{option_letter}" in normalized_evidence
    )
    gold_texts = []
    if answer:
        gold_texts.append(answer)
    gold_texts.extend(options)
    contains_gold_text = any(
        len(_normalize(text)) >= 3 and _normalize(text) in normalized_evidence
        for text in gold_texts
    )
    return {
        "task_evidence_contains_answer": contains_answer,
        "task_evidence_contains_option_letter": contains_option_letter,
        "task_evidence_contains_gold_text": contains_gold_text,
        "oracle_mode_enabled": oracle_mode,
    }


def _render_markdown(records: Sequence[Mapping[str, Any]], config: PromptDumpConfig) -> str:
    lines = [
        "# FC-OPD Condition Prompt Audit",
        "",
        f"- Source dataset: {config.source_dataset}",
        f"- Task evidence mode: {config.task_evidence_mode}",
        "- Default crop/bbox policy: metadata only; no crop condition is used in 4C.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['sample_uid']}",
                "",
                f"- Source index: {record['source_index']}",
                f"- Image: {record['image_path']}",
                f"- Degraded image: {record['degraded_image_path']}",
                f"- Crop/bbox image metadata: {record['bbox_image_path'] or 'None'}",
                f"- Crop/bbox image exists: {record['bbox_image_exists']}",
                f"- Question: {record['question']}",
                f"- Answer/gold: {record['answer']}",
                f"- Leakage flags: `{json.dumps(record['leakage_flags'], sort_keys=True)}`",
                "",
            ]
        )
        for condition, rendered in dict(record["rendered_prompts"]).items():
            lines.append(f"### `{condition}`")
            if config.include_images_as_paths:
                lines.append(f"- Image paths: {list(rendered['image_paths'])}")
            lines.extend(["", "```text", str(rendered["text"]), "```", ""])
    return "\n".join(lines)


def _rendered_text(messages: Sequence[Mapping[str, Any]]) -> str:
    chunks: list[str] = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, Sequence) and not isinstance(content, str | bytes):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                elif isinstance(item, Mapping) and item.get("type") == "image":
                    chunks.append(f"[image: {item.get('image')}]")
    return "\n".join(chunk for chunk in chunks if chunk)


def _output_paths(output: Path) -> tuple[Path, Path]:
    output = Path(output).expanduser()
    if output.suffix == ".jsonl":
        return output.with_suffix(".md"), output
    if output.suffix == ".md":
        return output, output.with_suffix(".jsonl")
    return output / "fc_opd_condition_prompts.md", output / "fc_opd_condition_prompts.jsonl"


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


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--dataset-type",
        choices=("auto", "vision_opd_json", "vision_opd_parquet", "generic_jsonl"),
        default="auto",
    )
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--conditions", type=_parse_conditions, default=DEFAULT_CONDITIONS)
    parser.add_argument("--blur-sigma", type=float, default=2.0)
    parser.add_argument("--degraded-dir")
    parser.add_argument("--materialize-degraded-images", action="store_true")
    parser.add_argument(
        "--task-evidence-mode",
        choices=TASK_EVIDENCE_MODES,
        default="none",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-images-as-paths", action="store_true")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="used only to discover Vision-OPD train.parquet when --dataset is omitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = args.dataset
    if dataset is None:
        dataset = discover_vision_opd_train_parquet(args.project_root)
        if dataset is None:
            raise SystemExit("--dataset is required when Vision-OPD train.parquet is not discoverable")
    result = dump_condition_prompts(
        PromptDumpConfig(
            dataset=dataset,
            dataset_type=args.dataset_type,
            source_dataset=args.source_dataset,
            limit=args.limit,
            conditions=args.conditions,
            blur_sigma=args.blur_sigma,
            degraded_dir=args.degraded_dir,
            materialize_degraded_images=args.materialize_degraded_images,
            task_evidence_mode=args.task_evidence_mode,
            output=args.output,
            include_images_as_paths=args.include_images_as_paths,
        )
    )
    print(f"wrote condition prompt dump: {result.markdown_path}, {result.jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Geometry3K adapter for clean-data FC-OPD 4C experiments."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset_adapters import normalize_prompt_text, resolve_dataset_path

QUESTION_KEYS = ("compact_text", "problem_text", "question", "problem", "query", "prompt", "text")
IMAGE_KEYS = ("image", "image_path", "diagram", "diagram_path", "images", "img")
CHOICE_KEYS = ("choices", "compact_choices", "options", "answer_choices", "candidates")
ANSWER_KEYS = ("answer", "label", "target", "gold", "ground_truth")
OFFICIAL_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
OFFICIAL_SPLITS = ("train", "val", "test")
GENERATED_DEGRADED_MARKERS = (
    ".lowres_",
    ".lowres-",
    ".gaussian_blur",
    ".degraded",
    ".blurred",
)
PREFERRED_OFFICIAL_IMAGE_STEMS = ("img_diagram", "diagram")


def load_geometry3k_records(path: str | Path, *, source_dataset: str = "geometry3k") -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    raw_records = _load_raw(path)
    return [
        normalize_geometry3k_record(
            record,
            source_dataset=source_dataset,
            source_index=index,
            dataset_path=path,
        )
        for index, record in enumerate(raw_records)
    ]


def inspect_geometry3k_dataset(
    path: str | Path,
    *,
    source_dataset: str = "geometry3k",
    examples: int = 3,
) -> dict[str, Any]:
    records = load_geometry3k_records(path, source_dataset=source_dataset)
    split_counts = Counter(str(record.get("split") or "unknown") for record in records)
    missing_image_count = sum(1 for record in records if not record.get("image_exists"))
    return {
        "dataset": str(Path(path).expanduser()),
        "source_dataset": source_dataset,
        "num_records": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "missing_image_count": missing_image_count,
        "sample_examples": [
            {
                "sample_uid": record.get("sample_uid"),
                "source_index": record.get("source_index"),
                "split": record.get("split"),
                "question": _preview(str(record.get("question", ""))),
                "choices": record.get("choices", []),
                "image_path": record.get("image_path"),
                "image_exists": record.get("image_exists"),
                "data_json_path": record.get("data_json_path"),
                "logic_form_json_path": record.get("logic_form_json_path"),
            }
            for record in records[: max(0, examples)]
        ],
    }


def normalize_geometry3k_record(
    record: Mapping[str, Any],
    *,
    source_dataset: str,
    source_index: int,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = None if dataset_path is None else Path(dataset_path).expanduser()
    is_official_dir_record = bool(record.get("_geometry3k_official_dir_record"))
    question_id = str(_first(record, ("id", "uid", "question_id", "qid", "index")) or source_index)
    question = _official_question(record) if is_official_dir_record else normalize_prompt_text(_first(record, QUESTION_KEYS)).strip()
    if not question:
        raise ValueError(f"Geometry3K record {source_index} is missing a question")
    choices = _extract_choices(record)
    if choices and not is_official_dir_record and not _question_contains_choices(question, choices):
        question = "\n\n".join([question, *choices])
    image_raw = record.get("_image_path") or _first(record, IMAGE_KEYS)
    image_path = resolve_dataset_path(_pathlike(image_raw), dataset_path)
    answer = _answer(record)
    split = _string_or_none(_first(record, ("data_type", "split", "_split")))
    sample_uid = (
        question_id
        if question_id.startswith(f"{source_dataset}:")
        else f"{source_dataset}:{split}:{question_id}"
        if is_official_dir_record and split
        else f"{source_dataset}:{question_id}"
    )
    logic_form = record.get("_logic_form_json") if isinstance(record.get("_logic_form_json"), Mapping) else {}
    data_json_path = _string_or_none(record.get("_data_json_path"))
    logic_form_json_path = _string_or_none(record.get("_logic_form_json_path"))
    image_exists = Path(image_path).expanduser().is_file() if image_path else False
    return {
        "sample_uid": sample_uid,
        "source_dataset": source_dataset,
        "source_index": int(_first(record, ("source_index", "idx", "index")) or source_index),
        "question_id": question_id,
        "split": split,
        "question": question,
        "clean_question_text": question,
        "choices": choices,
        "image_path": image_path,
        "image_exists": image_exists,
        "images": [image_path] if image_path else [],
        "answer": answer,
        "gold": answer,
        "answer_metadata": answer,
        "answer_source": "metadata_only_posthoc" if answer is not None else None,
        "answer_available": answer is not None,
        "data_json_path": data_json_path,
        "logic_form_json_path": logic_form_json_path,
        "problem_type_graph": record.get("problem_type_graph"),
        "problem_type_goal": record.get("problem_type_goal"),
        "text_logic_form": logic_form.get("text_logic_form"),
        "dissolved_text_logic_form": logic_form.get("dissolved_text_logic_form"),
        "diagram_logic_form": logic_form.get("diagram_logic_form"),
        "line_instances": logic_form.get("line_instances"),
        "point_positions": logic_form.get("point_positions"),
        "circle_instances": logic_form.get("circle_instances"),
        "outcome_metadata": {
            "answer": answer,
            "answer_source": "data_json.answer" if is_official_dir_record and answer is not None else "metadata_only_posthoc" if answer is not None else None,
            "correctness_used_for_prompt": False,
            "correctness_used_for_evidence_generation": False,
        },
        "no_gold_field_used": True,
        "bbox_image_path": "",
        "bbox_image_paths": [],
        "bbox_image_exists": False,
        "metadata": {
            "adapter": "geometry3k",
            "clean_data_policy": "no_red_box_no_crop",
            "gold_answer_metadata_only": True,
            "official_directory_record": is_official_dir_record,
            "split": split,
            "image_exists": image_exists,
            "data_json_path": data_json_path,
            "logic_form_json_path": logic_form_json_path,
            "problem_type_graph": record.get("problem_type_graph"),
            "problem_type_goal": record.get("problem_type_goal"),
            "logic_form": {
                "text_logic_form": logic_form.get("text_logic_form"),
                "dissolved_text_logic_form": logic_form.get("dissolved_text_logic_form"),
                "diagram_logic_form": logic_form.get("diagram_logic_form"),
                "line_instances": logic_form.get("line_instances"),
                "point_positions": logic_form.get("point_positions"),
                "circle_instances": logic_form.get("circle_instances"),
            },
            "raw_keys": sorted(str(key) for key in record.keys()),
        },
    }


def _load_raw(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        return _load_official_directory(path)
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("reading parquet requires pandas and pyarrow") from exc
        return [dict(item) for item in pd.read_parquet(path).to_dict(orient="records")]
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("top-level Geometry3K JSON must be a list")
        return [dict(item) for item in data]
    return [dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def _load_official_directory(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for data_json_path in sorted(path.rglob("data.json")):
        sample_dir = data_json_path.parent
        data = _read_json_object(data_json_path)
        logic_form_json_path = sample_dir / "logic_form.json"
        logic_form = _read_json_object(logic_form_json_path) if logic_form_json_path.is_file() else {}
        image_path = _find_sample_image(sample_dir)
        data.setdefault("id", sample_dir.name)
        data["_geometry3k_official_dir_record"] = True
        data["_sample_dir"] = str(sample_dir.resolve(strict=False))
        data["_data_json_path"] = str(data_json_path.resolve(strict=False))
        data["_logic_form_json_path"] = str(logic_form_json_path.resolve(strict=False)) if logic_form_json_path.is_file() else ""
        data["_logic_form_json"] = logic_form
        data["_image_path"] = str(image_path.resolve(strict=False)) if image_path is not None else ""
        data["_split"] = _infer_split(data_json_path)
        records.append(data)
    return records


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(data)


def _find_sample_image(sample_dir: Path) -> Path | None:
    candidates = [
        item
        for item in sample_dir.iterdir()
        if item.is_file() and item.suffix.lower() in OFFICIAL_IMAGE_SUFFIXES
    ]
    clean_candidates = sorted(item for item in candidates if not is_generated_degraded_image_path(item))
    for stem in PREFERRED_OFFICIAL_IMAGE_STEMS:
        for suffix in OFFICIAL_IMAGE_SUFFIXES:
            preferred = sample_dir / f"{stem}{suffix}"
            if preferred in clean_candidates:
                return preferred
    return clean_candidates[0] if clean_candidates else None


def is_generated_degraded_image_path(path: str | Path) -> bool:
    """True when an image path looks like a generated degraded/cache artifact."""

    name = Path(path).name.lower()
    return any(marker in name for marker in GENERATED_DEGRADED_MARKERS)


def default_degraded_image_dir() -> Path:
    """Default degraded-image cache outside source dataset directories."""

    output_root = os.environ.get("DTOPD_OUTPUT_ROOT")
    if output_root:
        return Path(output_root).expanduser() / "fc_opd" / "degraded_images"
    return Path.cwd() / "artifacts" / "fc_opd" / "degraded_images"


def _infer_split(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    for part in reversed(parts):
        if part in OFFICIAL_SPLITS:
            return part
    return None


def _official_question(record: Mapping[str, Any]) -> str:
    compact = normalize_prompt_text(record.get("compact_text")).strip()
    if compact:
        return compact
    return normalize_prompt_text(record.get("problem_text") or _first(record, QUESTION_KEYS)).strip()


def _first(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _pathlike(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("path", "image", "image_path", "diagram_path"):
            if value.get(key):
                return str(value[key])
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return _pathlike(value[0]) if value else None
    return str(value)


def _extract_choices(record: Mapping[str, Any]) -> list[str]:
    raw = _first(record, CHOICE_KEYS)
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [f"{key}. {value}" for key, value in raw.items()]
    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
        output = []
        for index, value in enumerate(raw):
            text = str(value).strip()
            if not text:
                continue
            prefix = chr(ord("A") + index)
            output.append(text if text[:2].upper() == f"{prefix}." else f"{prefix}. {text}")
        return output
    return [str(raw)]


def _question_contains_choices(question: str, choices: Sequence[str]) -> bool:
    normalized = question.lower()
    return all(choice.split(".", 1)[-1].strip().lower() in normalized for choice in choices[:2])


def _answer(record: Mapping[str, Any]) -> str | None:
    value = _first(record, ANSWER_KEYS)
    if value is None:
        value = _nested(record, ("extra_info", "answer")) or _nested(record, ("reward_model", "ground_truth"))
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _nested(record: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = record
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _preview(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."

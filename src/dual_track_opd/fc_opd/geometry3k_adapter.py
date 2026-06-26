"""Geometry3K adapter for clean-data FC-OPD 4C experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset_adapters import normalize_prompt_text, resolve_dataset_path

QUESTION_KEYS = ("question", "problem", "query", "prompt", "text")
IMAGE_KEYS = ("image", "image_path", "diagram", "diagram_path", "images", "img")
CHOICE_KEYS = ("choices", "options", "answer_choices", "candidates")
ANSWER_KEYS = ("answer", "label", "target", "gold", "ground_truth")


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


def normalize_geometry3k_record(
    record: Mapping[str, Any],
    *,
    source_dataset: str,
    source_index: int,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = None if dataset_path is None else Path(dataset_path).expanduser()
    question_id = str(_first(record, ("id", "uid", "question_id", "qid", "index")) or source_index)
    question = normalize_prompt_text(_first(record, QUESTION_KEYS)).strip()
    if not question:
        raise ValueError(f"Geometry3K record {source_index} is missing a question")
    choices = _extract_choices(record)
    if choices and not _question_contains_choices(question, choices):
        question = "\n\n".join([question, *choices])
    image_raw = _first(record, IMAGE_KEYS)
    image_path = resolve_dataset_path(_pathlike(image_raw), dataset_path)
    answer = _answer(record)
    sample_uid = (
        question_id
        if question_id.startswith(f"{source_dataset}:")
        else f"{source_dataset}:{question_id}"
    )
    return {
        "sample_uid": sample_uid,
        "source_dataset": source_dataset,
        "source_index": int(_first(record, ("source_index", "idx", "index")) or source_index),
        "question_id": question_id,
        "question": question,
        "clean_question_text": question,
        "choices": choices,
        "image_path": image_path,
        "images": [image_path] if image_path else [],
        "answer": answer,
        "gold": answer,
        "answer_metadata": answer,
        "answer_source": "metadata_only_posthoc" if answer is not None else None,
        "answer_available": answer is not None,
        "bbox_image_path": "",
        "bbox_image_paths": [],
        "bbox_image_exists": False,
        "metadata": {
            "adapter": "geometry3k",
            "clean_data_policy": "no_red_box_no_crop",
            "gold_answer_metadata_only": True,
            "raw_keys": sorted(str(key) for key in record.keys()),
        },
    }


def _load_raw(path: Path) -> list[dict[str, Any]]:
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

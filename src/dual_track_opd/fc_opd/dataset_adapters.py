"""Dataset adapters for FC-OPD pre-training audits.

The adapters normalize candidate VLM datasets into a small common schema used by
prompt dumps and dataset-signal audits. Vision-OPD crop / bbox images are kept
as metadata by default and are never silently mapped into the default 4C
conditions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


QUESTION_KEYS = ("query", "question", "prompt", "instruction", "problem", "text")
GLOBAL_IMAGE_KEYS = (
    "image_path",
    "image",
    "images",
    "image_paths",
    "global_image_path",
    "global_image",
    "original_image_path",
    "original_image",
    "full_image_path",
    "img_path",
    "img",
)
CROP_IMAGE_KEYS = (
    "bbox_image_path",
    "bbox_image",
    "crop_image_path",
    "crop_image",
    "cropped_image_path",
    "roi_image_path",
    "local_image_path",
    "patch_image_path",
)
ANSWER_KEYS = (
    "answer",
    "gold",
    "gold_answer",
    "reference_answer",
    "label",
    "target",
    "response",
    "chosen",
)
OPTION_KEYS = ("options", "choices", "candidates")


def load_raw_records(path: str | Path, dataset_type: str = "auto") -> list[dict[str, Any]]:
    """Load JSON, JSONL, or Parquet records without schema normalization."""

    path = Path(path).expanduser()
    resolved_type = resolve_dataset_type(path, dataset_type)
    if resolved_type == "vision_opd_parquet":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional HPC dependency
            raise RuntimeError("reading parquet requires pandas and pyarrow") from exc
        return [_json_safe_record(record) for record in pd.read_parquet(path).to_dict(orient="records")]
    if resolved_type in {"vision_opd_json", "generic_jsonl"}:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        if text[0] == "[":
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("top-level JSON must be a list of records")
            return [_json_safe_record(record) for record in data]
        return [
            _json_safe_record(json.loads(line))
            for line in text.splitlines()
            if line.strip()
        ]
    raise ValueError(f"unsupported dataset type: {resolved_type}")


def load_normalized_records(
    path: str | Path,
    dataset_type: str = "auto",
    *,
    source_dataset: str,
) -> list[dict[str, Any]]:
    """Load and normalize candidate records for FC-OPD audit tools."""

    path = Path(path).expanduser()
    raw_records = load_raw_records(path, dataset_type)
    return [
        normalize_record(
            record,
            source_dataset=source_dataset,
            source_index=index,
            dataset_path=path,
        )
        for index, record in enumerate(raw_records)
    ]


def normalize_record(
    record: Mapping[str, Any],
    *,
    source_dataset: str,
    source_index: int,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize a raw dataset row into the shared audit schema."""

    dataset_path = None if dataset_path is None else Path(dataset_path).expanduser()
    question_id = _first_text(record, ("sample_uid", "question_id", "qid", "id", "uid", "index"))
    if question_id is None:
        question_id = str(source_index)
    sample_uid = (
        question_id
        if str(question_id).startswith(f"{source_dataset}:")
        else f"{source_dataset}:{question_id}"
    )
    question = _first_text(record, QUESTION_KEYS) or ""
    image_raw = _first_value(record, GLOBAL_IMAGE_KEYS)
    crop_raw = _first_value(record, CROP_IMAGE_KEYS)
    image_path = resolve_dataset_path(_pathlike_from_value(image_raw), dataset_path)
    bbox_image_path = resolve_dataset_path(_pathlike_from_value(crop_raw), dataset_path)
    answer = _first_text(record, ANSWER_KEYS)
    free_caption = _first_text(record, ("free_caption", "caption", "image_caption")) or ""
    task_evidence = _first_text(record, ("task_evidence", "task_extraction", "evidence")) or ""
    options = _extract_options(record)

    normalized = {
        "sample_uid": sample_uid,
        "question_id": str(question_id),
        "source_dataset": source_dataset,
        "source_index": int(_first_text(record, ("source_index", "idx", "index")) or source_index),
        "question": question,
        "query": question,
        "image_path": image_path,
        "images": [image_path] if image_path else [],
        "bbox_image_path": bbox_image_path,
        "answer": answer,
        "gold": answer,
        "free_caption": free_caption,
        "task_evidence": task_evidence,
        "options": options,
        "metadata": {
            "adapter": "vision_opd_6k_or_generic",
            "crop_bbox_policy": "metadata_only_default_no_crop_condition",
            "bbox_image_path": bbox_image_path,
            "raw_keys": sorted(str(key) for key in record.keys()),
        },
    }
    return normalized


def discover_vision_opd_train_parquet(project_root: str | Path) -> Path | None:
    """Find a prepared Vision-OPD train parquet under ``third_party/Vision-OPD``."""

    root = Path(project_root).expanduser()
    preferred = root / "third_party" / "Vision-OPD" / "data" / "train.parquet"
    if preferred.is_file():
        return preferred
    candidates = sorted((root / "third_party").glob("**/train.parquet"))
    for candidate in candidates:
        if "vision" in str(candidate).lower() and "opd" in str(candidate).lower():
            return candidate
    return None


def resolve_dataset_type(path: Path, dataset_type: str) -> str:
    if dataset_type != "auto":
        return dataset_type
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "vision_opd_parquet"
    if suffix == ".jsonl":
        return "generic_jsonl"
    return "vision_opd_json"


def resolve_dataset_path(value: str | None, dataset_path: Path | None) -> str:
    """Resolve dataset-relative paths using common Vision-OPD layouts."""

    if value is None or not value.strip():
        return ""
    expanded = Path(os.path.expandvars(value)).expanduser()
    if expanded.is_absolute():
        return str(expanded)

    candidates: list[Path] = []
    if dataset_path is not None:
        dataset_parent = dataset_path.parent
        candidates.extend(
            [
                dataset_parent / expanded,
                dataset_parent.parent / expanded,
                dataset_parent.parent / "data" / expanded,
                dataset_parent.parent / "images" / expanded,
            ]
        )
    cwd = Path.cwd()
    candidates.extend([cwd / expanded, cwd / "third_party" / "Vision-OPD" / expanded])

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve(strict=False))
    return str(candidates[0].resolve(strict=False) if candidates else expanded)


def _first_value(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            value = record[key]
            if not _is_empty(value):
                return value
    return None


def _first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    value = _first_value(record, keys)
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, bytes):
        return None
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if not value:
            return None
        return str(value[0]).strip() or None
    return str(value).strip() or None


def _pathlike_from_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Mapping):
        for key in ("path", "image_path", "filename", "file_name"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
        return None
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for item in value:
            path = _pathlike_from_value(item)
            if path is not None:
                return path
        return None
    return None


def _extract_options(record: Mapping[str, Any]) -> list[str]:
    raw = _first_value(record, OPTION_KEYS)
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [f"{key}. {value}" for key, value in raw.items()]
    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
        return [str(item) for item in raw]
    return [str(raw)]


def _is_empty(value: Any) -> bool:
    try:
        import pandas as pd

        if pd.isna(value):
            return True
    except Exception:  # noqa: BLE001
        pass
    return isinstance(value, str) and not value.strip()


def _json_safe_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("each dataset record must be a mapping")
    output: dict[str, Any] = {}
    for key, value in record.items():
        output[str(key)] = _json_safe_value(value)
    return output


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_safe_value(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe_value(value.tolist())
        except Exception:  # noqa: BLE001
            return str(value)
    if hasattr(value, "item"):
        try:
            return _json_safe_value(value.item())
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)

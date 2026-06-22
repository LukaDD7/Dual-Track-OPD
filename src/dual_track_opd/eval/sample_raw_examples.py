"""Sample raw VLM responses for qualitative inspection."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .score_raw_responses import (
    _first_present,
    _stringify,
    get_finish_reason,
    get_ground_truths,
    get_prediction,
    get_row_id,
    load_manifest,
)

log = logging.getLogger(__name__)

QUESTION_KEYS = (
    "question",
    "query",
    "prompt",
    "problem",
    "instruction",
    "input",
    "user_prompt",
)
OPTION_KEYS = ("options", "choices_text", "choices", "candidates")
REASONING_KEYS = (
    "reasoning",
    "rationale",
    "explanation",
    "solution",
    "cot",
    "chain_of_thought",
    "analysis",
)
IMAGE_KEYS = (
    "image",
    "image_path",
    "image_paths",
    "image_url",
    "image_urls",
    "images",
)
IMAGE_REF_KEY_HINTS = ("image", "img", "path", "url", "file")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _select_indices(count: int, samples_per_dataset: int, strategy: str) -> list[int]:
    if count <= 0 or samples_per_dataset <= 0:
        return []
    if strategy == "first":
        return list(range(min(count, samples_per_dataset)))
    if samples_per_dataset == 1:
        return [0]
    step = (count - 1) / (samples_per_dataset - 1)
    return sorted({round(i * step) for i in range(samples_per_dataset)})


def _field(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    value = _first_present(item, keys)
    return _stringify(value) if value is not None else ""


def _recursive_first_field(value: Any, keys: tuple[str, ...]) -> Any:
    """Find the first non-empty value for keys, including nested records."""

    if isinstance(value, dict):
        direct = _first_present(value, keys)
        if direct not in (None, ""):
            return direct
        for nested in value.values():
            found = _recursive_first_field(nested, keys)
            if found not in (None, ""):
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _recursive_first_field(nested, keys)
            if found not in (None, ""):
                return found
    return None


def _looks_like_image_path(value: str) -> bool:
    lowered = value.lower()
    return any(
        lowered.endswith(suffix)
        for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")
    )


def _looks_like_image_ref(value: str) -> bool:
    lowered = value.lower()
    return (
        _looks_like_image_path(value)
        or lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("s3://")
        or lowered.startswith("/")
    )


def collect_image_refs(obj: Any) -> list[str]:
    """Recursively collect image path or URL references from nested records."""

    refs: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and _looks_like_image_ref(value) and value not in seen:
            seen.add(value)
            refs.append(value)
        elif isinstance(value, list):
            for item in value:
                add(item)
        elif isinstance(value, dict):
            if "url" in value:
                add(value["url"])
            for item in value.values():
                add(item)

    def walk(value: Any, key_name: str = "") -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                lowered_key = key.lower()
                if key in IMAGE_KEYS or any(hint in lowered_key for hint in IMAGE_REF_KEY_HINTS):
                    add(inner)
                walk(inner, key)
        elif isinstance(value, list):
            for item in value:
                walk(item, key_name)
        elif isinstance(value, str):
            lowered_key = key_name.lower()
            if _looks_like_image_ref(value) and (
                _looks_like_image_path(value)
                or any(hint in lowered_key for hint in IMAGE_REF_KEY_HINTS)
            ):
                add(value)

    walk(obj)
    return refs


def _metadata_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("metadata", "original_metadata"):
        if key in item:
            metadata[key] = item[key]
    return metadata


def _sha256_prefix(data: bytes, n: int = 8) -> str:
    return hashlib.sha256(data).hexdigest()[:n]


def _is_image_bytes(data: bytes) -> bool:
    return data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:4] == b"RIFF"


def _save_image_bytes(data: bytes, out_dir: Path, prefix: str) -> str:
    if data[:3] == b"\xff\xd8\xff":
        ext = ".jpg"
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        ext = ".png"
    elif data[:4] == b"RIFF":
        ext = ".webp"
    else:
        ext = ".bin"
    name = f"{prefix}{ext}"
    (out_dir / name).write_bytes(data)
    return name


def _load_parquet_table(path: Path) -> Any:
    try:
        import pyarrow.parquet as pq

        return pq.read_table(str(path))
    except Exception:
        return None


def _gqa_resolve_image(meta: dict[str, Any], dataset_root: Path) -> bytes | None:
    image_id = meta.get("imageId")
    if not image_id:
        return None
    split = meta.get("split", "val_balanced")
    images_dir = dataset_root / "GQA" / f"{split}_images"
    if not images_dir.is_dir():
        return None

    cache_key = str(images_dir)
    if not hasattr(_gqa_resolve_image, "_cache"):
        _gqa_resolve_image._cache = {}
    if cache_key not in _gqa_resolve_image._cache:
        idx: dict[str, bytes] = {}
        for pq_file in sorted(images_dir.glob("*.parquet")):
            table = _load_parquet_table(pq_file)
            if table is None or "id" not in table.column_names:
                continue
            id_col = table.column("id")
            img_col = table.column("image") if "image" in table.column_names else None
            if img_col is None:
                continue
            for i in range(len(id_col)):
                raw_id = str(id_col[i].as_py())
                img = img_col[i].as_py()
                if isinstance(img, dict) and img.get("bytes"):
                    idx[raw_id] = img["bytes"]
        _gqa_resolve_image._cache[cache_key] = idx
        log.info("GQA index built: %d images from %s", len(idx), cache_key)

    return _gqa_resolve_image._cache[cache_key].get(str(image_id))


def _parquet_resolve_by_column(
    dataset_root: Path,
    subpath: str,
    column_name: str,
    match_value: Any,
    image_column: str = "image",
    file_glob: str = "*.parquet",
) -> bytes | None:
    data_dir = dataset_root / subpath
    if not data_dir.is_dir():
        return None
    for pq_file in sorted(data_dir.glob(file_glob)):
        table = _load_parquet_table(pq_file)
        if table is None:
            continue
        if column_name not in table.column_names:
            continue
        col = table.column(column_name)
        img_col_name = image_column if image_column in table.column_names else None
        bytes_col_name = "bytes" if "bytes" in table.column_names else None
        for i in range(len(col)):
            if str(col[i].as_py()) == str(match_value):
                if img_col_name:
                    val = table.column(img_col_name)[i].as_py()
                    if isinstance(val, dict) and val.get("bytes"):
                        return val["bytes"]
                    if isinstance(val, str) and _is_image_bytes(base64.b64decode(val)):
                        return base64.b64decode(val)
                    if isinstance(val, str):
                        img_path = data_dir / val
                        if img_path.is_file():
                            return img_path.read_bytes()
                if bytes_col_name:
                    val = table.column(bytes_col_name)[i].as_py()
                    if isinstance(val, bytes) and _is_image_bytes(val):
                        return val
    return None


def _mmvet_resolve_image(sample_id: str, dataset_root: Path) -> bytes | None:
    data_dir = dataset_root / "MMVet" / "data"
    if not data_dir.is_dir():
        return None
    for pq_file in sorted(data_dir.glob("*.parquet")):
        table = _load_parquet_table(pq_file)
        if table is None:
            continue
        if "question_id" not in table.column_names:
            continue
        qid_col = table.column("question_id")
        for i in range(len(qid_col)):
            if str(qid_col[i].as_py()) == str(sample_id):
                if "image" in table.column_names:
                    val = table.column("image")[i].as_py()
                    if isinstance(val, dict) and val.get("bytes"):
                        return val["bytes"]
                if "bytes" in table.column_names:
                    val = table.column("bytes")[i].as_py()
                    if isinstance(val, bytes) and _is_image_bytes(val):
                        return val
                path_col = "path" if "path" in table.column_names else None
                if path_col:
                    val = table.column(path_col)[i].as_py()
                    if isinstance(val, str) and val:
                        img_path = data_dir / val
                        if img_path.is_file():
                            return img_path.read_bytes()
    return None


def _mm_bench_resolve_image(sample_id: str, dataset_root: Path) -> bytes | None:
    data_dir = dataset_root / "MMBench" / "data"
    if not data_dir.is_dir():
        return None
    for pq_file in sorted(data_dir.glob("*.parquet")):
        table = _load_parquet_table(pq_file)
        if table is None:
            continue
        if "index" not in table.column_names:
            continue
        idx_col = table.column("index")
        for i in range(len(idx_col)):
            if str(idx_col[i].as_py()) == str(sample_id):
                if "image" in table.column_names:
                    val = table.column("image")[i].as_py()
                    if isinstance(val, str) and val:
                        try:
                            decoded = base64.b64decode(val)
                            if _is_image_bytes(decoded):
                                return decoded
                        except Exception:
                            pass
    return None


def _viewspatial_resolve_image(
    sample_id: str, dataset_root: Path, raw_item: dict[str, Any]
) -> str | None:
    vs_dir = dataset_root / "ViewSpatial-Bench"
    json_path = vs_dir / "ViewSpatial-Bench.json"
    if not json_path.is_file():
        return None
    try:
        entries = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        idx = int(sample_id)
    except (ValueError, TypeError):
        return None
    if idx >= len(entries):
        return None
    entry = entries[idx]
    image_paths = entry.get("image_path", [])
    if not image_paths:
        return None
    rel = image_paths[0] if isinstance(image_paths, list) else image_paths
    full = dataset_root / rel
    if full.is_file():
        return str(full)
    full2 = vs_dir / rel
    if full2.is_file():
        return str(full2)
    return None


def _mv_math_resolve_image(
    sample_id: str, dataset_root: Path, raw_item: dict[str, Any]
) -> list[str]:
    mv_dir = dataset_root / "MV-MATH"
    json_path = mv_dir / "MV-MATH.json"
    if not json_path.is_file():
        return []
    try:
        entries = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    try:
        pid = int(sample_id)
    except (ValueError, TypeError):
        return []
    for entry in entries:
        if entry.get("problem_id") == pid:
            imgs = entry.get("input_image", [])
            result = []
            for rel in imgs:
                full = mv_dir / "images" / "images" / rel
                if full.is_file():
                    result.append(str(full))
            return result
    return []


def _mindcube_resolve_image(
    sample_id: str, dataset_root: Path, raw_item: dict[str, Any]
) -> list[str]:
    mc_dir = dataset_root / "MindCube" / "data" / "data" / "raw"
    jsonl_path = mc_dir / "MindCube.jsonl"
    if not jsonl_path.is_file():
        return []
    base_dir = mc_dir.parent.parent
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("id") == sample_id:
                imgs = item.get("images", [])
                result = []
                for rel in imgs:
                    if isinstance(rel, str):
                        full = base_dir / rel
                        if full.is_file():
                            result.append(str(full))
                return result
    return []


def _sciqa_resolve_image(sample_id: str, dataset_root: Path) -> bytes | None:
    data_dir = dataset_root / "ScienceQA-IMG" / "data"
    if not data_dir.is_dir():
        return None
    for pq_file in sorted(data_dir.glob("*.parquet")):
        table = _load_parquet_table(pq_file)
        if table is None:
            continue
        try:
            idx = int(sample_id)
        except (ValueError, TypeError):
            continue
        if idx >= table.num_rows:
            continue
        if "image" in table.column_names:
            val = table.column("image")[idx].as_py()
            if isinstance(val, dict) and val.get("bytes"):
                return val["bytes"]
            if isinstance(val, bytes) and _is_image_bytes(val):
                return val
        if "bytes" in table.column_names:
            val = table.column("bytes")[idx].as_py()
            if isinstance(val, bytes) and _is_image_bytes(val):
                return val
    return None


def _remi_resolve_image(sample_id: str, dataset_root: Path) -> list[bytes]:
    data_dir = dataset_root / "ReMI"
    result = []
    for pq_file in sorted(data_dir.glob("*.parquet")):
        table = _load_parquet_table(pq_file)
        if table is None:
            continue
        img_cols = [c for c in table.column_names if c.startswith("image_")]
        if not img_cols:
            continue
        try:
            idx = int(sample_id)
        except (ValueError, TypeError):
            continue
        if idx < table.num_rows:
            for col_name in img_cols:
                val = table.column(col_name)[idx].as_py()
                if isinstance(val, dict) and val.get("bytes"):
                    result.append(val["bytes"])
                elif isinstance(val, bytes) and _is_image_bytes(val):
                    result.append(val)
            if result:
                return result
    return result


def _dyna_math_resolve_image(sample_id: str, dataset_root: Path) -> bytes | None:
    data_dir = dataset_root / "DynaMath_Sample" / "data"
    if not data_dir.is_dir():
        return None
    for pq_file in sorted(data_dir.glob("*.parquet")):
        table = _load_parquet_table(pq_file)
        if table is None:
            continue
        if "id" not in table.column_names:
            continue
        id_col = table.column("id")
        for i in range(len(id_col)):
            if str(id_col[i].as_py()) == str(sample_id):
                if "decoded_image" in table.column_names:
                    val = table.column("decoded_image")[i].as_py()
                    if isinstance(val, bytes) and len(val) > 100:
                        return val
                if "image" in table.column_names:
                    val = table.column("image")[i].as_py()
                    if isinstance(val, str) and val:
                        full = data_dir.parent / val
                        if full.is_file():
                            return full.read_bytes()
                    if isinstance(val, dict):
                        if val.get("bytes") and isinstance(val["bytes"], bytes):
                            return val["bytes"]
    return None


def _vqav2_resolve_image(sample_id: str, dataset_root: Path) -> bytes | None:
    data_dir = dataset_root / "VQAv2" / "data"
    if not data_dir.is_dir():
        return None
    for pq_file in sorted(data_dir.glob("*.parquet")):
        table = _load_parquet_table(pq_file)
        if table is None or "question_id" not in table.column_names:
            continue
        qid_col = table.column("question_id")
        img_col = table.column("image") if "image" in table.column_names else None
        if img_col is None:
            continue
        for i in range(len(qid_col)):
            if str(qid_col[i].as_py()) == str(sample_id):
                img = img_col[i].as_py()
                if isinstance(img, dict) and img.get("bytes"):
                    return img["bytes"]
                return None
    return None


def _mathverse_resolve_image(sample_id: str, dataset_root: Path) -> bytes | None:
    data_dir = dataset_root / "MathVerse"
    for pq_file in sorted(data_dir.glob("*.parquet")):
        table = _load_parquet_table(pq_file)
        if table is None:
            continue
        if "sample_index" not in table.column_names:
            continue
        idx_col = table.column("sample_index")
        for i in range(len(idx_col)):
            if str(idx_col[i].as_py()) == str(sample_id):
                if "image" in table.column_names:
                    val = table.column("image")[i].as_py()
                    if isinstance(val, dict):
                        if val.get("bytes") and isinstance(val["bytes"], bytes):
                            return val["bytes"]
                        if val.get("path"):
                            full = data_dir / val["path"]
                            if full.is_file():
                                return full.read_bytes()
                    if isinstance(val, str) and val:
                        try:
                            decoded = base64.b64decode(val)
                            if _is_image_bytes(decoded):
                                return decoded
                        except Exception:
                            pass
    return None


def _mathvista_resolve_image(sample_id: str, dataset_root: Path) -> bytes | None:
    data_dir = dataset_root / "MathVista" / "data"
    if not data_dir.is_dir():
        return None
    for pq_file in sorted(data_dir.glob("*.parquet")):
        table = _load_parquet_table(pq_file)
        if table is None:
            continue
        pid_col_name = "pid" if "pid" in table.column_names else None
        if not pid_col_name:
            continue
        col = table.column(pid_col_name)
        for i in range(len(col)):
            if str(col[i].as_py()) == str(sample_id):
                if "decoded_image" in table.column_names:
                    val = table.column("decoded_image")[i].as_py()
                    if isinstance(val, dict) and val.get("bytes"):
                        return val["bytes"]
                    if isinstance(val, bytes) and len(val) > 100:
                        return val
                if "image" in table.column_names:
                    val = table.column("image")[i].as_py()
                    if isinstance(val, dict):
                        if val.get("bytes") and isinstance(val["bytes"], bytes):
                            return val["bytes"]
                        if val.get("path"):
                            full = data_dir / val["path"]
                            if full.is_file():
                                return full.read_bytes()
                    if isinstance(val, str) and val:
                        full = data_dir / val
                        if full.is_file():
                            return full.read_bytes()
                if "bytes" in table.column_names:
                    val = table.column("bytes")[i].as_py()
                    if isinstance(val, bytes) and _is_image_bytes(val):
                        return val
    return None
    for pq_file in sorted(data_dir.glob("*.parquet")):
        table = _load_parquet_table(pq_file)
        if table is None:
            continue
        pid_col = "pid" if "pid" in table.column_names else None
        if not pid_col:
            continue
        col = table.column(pid_col)
        for i in range(len(col)):
            if str(col[i].as_py()) == str(sample_id):
                if "image" in table.column_names:
                    val = table.column("image")[i].as_py()
                    if isinstance(val, dict):
                        if val.get("bytes") and isinstance(val["bytes"], bytes):
                            return val["bytes"]
                        if val.get("path"):
                            full = data_dir / val["path"]
                            if full.is_file():
                                return full.read_bytes()
                if "bytes" in table.column_names:
                    val = table.column("bytes")[i].as_py()
                    if isinstance(val, bytes) and _is_image_bytes(val):
                        return val
    return None


def resolve_images_from_dataset(
    dataset: str,
    sample_id: str,
    meta: dict[str, Any],
    dataset_root: Path | None,
    raw_item: dict[str, Any] | None = None,
) -> list[str]:
    """Resolve image paths from dataset sources for a given sample.

    Returns a list of absolute file paths to extracted/copied images.
    Empty list means no images could be resolved.
    """
    if dataset_root is None:
        return []

    out_dir = dataset_root.parent / "sampled_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{dataset}_{sample_id}"

    raw = raw_item or {}

    if dataset == "GQA":
        data = _gqa_resolve_image(meta, dataset_root)
        if data:
            name = _save_image_bytes(data, out_dir, prefix)
            return [str(out_dir / name)]

    if dataset == "MMBench":
        data = _mm_bench_resolve_image(sample_id, dataset_root)
        if data:
            name = _save_image_bytes(data, out_dir, prefix)
            return [str(out_dir / name)]

    if dataset == "MMVet":
        data = _mmvet_resolve_image(sample_id, dataset_root)
        if data:
            name = _save_image_bytes(data, out_dir, prefix)
            return [str(out_dir / name)]

    if dataset == "MathVista":
        data = _mathvista_resolve_image(sample_id, dataset_root)
        if data:
            name = _save_image_bytes(data, out_dir, prefix)
            return [str(out_dir / name)]

    if dataset == "VQAv2":
        data = _vqav2_resolve_image(sample_id, dataset_root)
        if data:
            name = _save_image_bytes(data, out_dir, prefix)
            return [str(out_dir / name)]

    if dataset == "MathVerse":
        data = _mathverse_resolve_image(sample_id, dataset_root)
        if data:
            name = _save_image_bytes(data, out_dir, prefix)
            return [str(out_dir / name)]

    if dataset == "ScienceQA-IMG":
        data = _sciqa_resolve_image(sample_id, dataset_root)
        if data:
            name = _save_image_bytes(data, out_dir, prefix)
            return [str(out_dir / name)]

    if dataset == "DynaMath_Sample":
        data = _dyna_math_resolve_image(sample_id, dataset_root)
        if data:
            name = _save_image_bytes(data, out_dir, prefix)
            return [str(out_dir / name)]

    if dataset == "ReMI":
        images = _remi_resolve_image(sample_id, dataset_root)
        result = []
        for i, data in enumerate(images):
            name = _save_image_bytes(data, out_dir, f"{prefix}_{i}")
            result.append(str(out_dir / name))
        return result

    if dataset == "MMSI-Bench":
        data_dir = dataset_root / "MMSI-Bench"
        pq_file = data_dir / "MMSI_Bench.parquet"
        if pq_file.is_file():
            table = _load_parquet_table(pq_file)
            if table is not None and "id" in table.column_names:
                id_col = table.column("id")
                for i in range(len(id_col)):
                    if str(id_col[i].as_py()) == str(sample_id):
                        if "images" in table.column_names:
                            imgs = table.column("images")[i].as_py()
                            if isinstance(imgs, list):
                                result = []
                                for j, img_bytes in enumerate(imgs):
                                    if isinstance(img_bytes, bytes) and _is_image_bytes(img_bytes):
                                        name = _save_image_bytes(img_bytes, out_dir, f"{prefix}_{j}")
                                        result.append(str(out_dir / name))
                                return result

    if dataset == "ViewSpatial-Bench":
        path = _viewspatial_resolve_image(sample_id, dataset_root, raw)
        if path:
            return [path]

    if dataset == "MV-MATH":
        return _mv_math_resolve_image(sample_id, dataset_root, raw)

    if dataset == "MindCube-Bench":
        return _mindcube_resolve_image(sample_id, dataset_root, raw)

    if dataset in ("BLINK", "MMMU_Pro_10", "MMMU_Pro_4"):
        file_path = meta.get("file", "")
        if file_path and Path(file_path).is_file():
            table = _load_parquet_table(Path(file_path))
            if table is not None and "image" in table.column_names:
                for i in range(table.num_rows):
                    val = table.column("image")[i].as_py()
                    if isinstance(val, dict) and val.get("bytes"):
                        name = _save_image_bytes(val["bytes"], out_dir, f"{prefix}_{i}")
                        return [str(out_dir / name)]

    return []


def sample_raw_examples(
    raw_dir: str | Path,
    manifest: str | Path,
    samples_per_dataset: int = 2,
    strategy: str = "first",
    dataset_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    entries = load_manifest(manifest)
    raw_dir = Path(raw_dir)
    ds_root = Path(dataset_root) if dataset_root else None
    samples: list[dict[str, Any]] = []

    for entry in entries:
        raw_path = raw_dir / entry.file
        rows = _read_jsonl(raw_path)
        for index in _select_indices(len(rows), samples_per_dataset, strategy):
            item = rows[index]
            image_paths = collect_image_refs(item)

            if not image_paths and ds_root is not None:
                meta = item.get("meta", {})
                sample_id = item.get("sample_id", str(index + 1))
                resolved = resolve_images_from_dataset(
                    dataset=entry.dataset,
                    sample_id=sample_id,
                    meta=meta,
                    dataset_root=ds_root,
                    raw_item=item,
                )
                if resolved:
                    image_paths = resolved
                    log.info(
                        "resolved %d image(s) for %s sample_id=%s",
                        len(resolved),
                        entry.dataset,
                        sample_id,
                    )

            samples.append(
                {
                    "dataset": entry.dataset,
                    "file": entry.file,
                    "raw_response_file": str(raw_path),
                    "raw_response_row_index": index + 1,
                    "row_index": index + 1,
                    "row_id": get_row_id(item, index + 1),
                    "scoring_type": entry.scoring_type,
                    "question": _field(item, QUESTION_KEYS),
                    "options": _field(item, OPTION_KEYS),
                    "image": image_paths[0] if image_paths else "",
                    "image_paths": image_paths,
                    "prediction": get_prediction(item),
                    "reasoning": _field(item, REASONING_KEYS),
                    "ground_truths": get_ground_truths(item),
                    "finish_reason": get_finish_reason(item),
                    "error": _stringify(item.get("error")),
                    "original_metadata": _metadata_snapshot(item),
                    "raw_keys": sorted(item.keys()),
                }
            )
    return samples


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_markdown(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qwen3-VL-8B Baseline Raw Example Samples",
        "",
        "These examples are sampled from raw responses for qualitative inspection. Do not commit large raw JSONL files.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['dataset']} / {row['row_id']}",
                "",
                f"- file: `{row['file']}`",
                f"- row_index: `{row['row_index']}`",
                f"- scoring_type: `{row['scoring_type']}`",
                f"- finish_reason: `{row['finish_reason']}`",
                f"- image: `{row['image']}`",
                f"- image_paths: `{json.dumps(row.get('image_paths', []), ensure_ascii=False)}`",
                f"- raw_response_file: `{row.get('raw_response_file', row['file'])}`",
                f"- raw_response_row_index: `{row.get('raw_response_row_index', row['row_index'])}`",
                "",
                "**Question**",
                "",
                row["question"] or "(not found in recognized fields)",
                "",
                "**Options**",
                "",
                row["options"] or "(not found in recognized fields)",
                "",
                "**Prediction**",
                "",
                row["prediction"] or "(empty)",
                "",
                "**Reasoning / Explanation**",
                "",
                row["reasoning"] or "(not found in recognized fields)",
                "",
                "**Ground Truths**",
                "",
                json.dumps(row["ground_truths"], ensure_ascii=False),
                "",
                "**Original Metadata**",
                "",
                json.dumps(row.get("original_metadata", {}), ensure_ascii=False, sort_keys=True),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw JSONL files.")
    parser.add_argument("--manifest", required=True, help="Preferred raw-file manifest JSONL.")
    parser.add_argument("--out-jsonl", required=True, help="Output sampled examples JSONL.")
    parser.add_argument("--out-md", required=True, help="Output sampled examples Markdown.")
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--strategy", choices=["first", "even"], default="first")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Root directory containing dataset parquets/files for image resolution.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    rows = sample_raw_examples(
        raw_dir=args.raw_dir,
        manifest=args.manifest,
        samples_per_dataset=args.samples_per_dataset,
        strategy=args.strategy,
        dataset_root=args.dataset_root,
    )
    write_jsonl(args.out_jsonl, rows)
    write_markdown(args.out_md, rows)


if __name__ == "__main__":
    main()

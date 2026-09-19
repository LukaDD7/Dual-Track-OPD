#!/usr/bin/env python3
"""Build a VA-OPD-ready ViRL39K parquet with full/degraded image metadata.

The paper's counterfactual image is generated deterministically as:

1. bilinear downsample each spatial dimension to 10%;
2. nearest-neighbor upsample back to the original size.

The output preserves multi-image samples.  Each image gets its own prepared
degraded counterpart and the native VA-OPD teacher pass replaces all images.
By default the validation monitor overlaps training data so the train split
remains the full convertible ViRL39K pool.  Pass ``--holdout-val`` to make the
monitor disjoint for a controlled ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from PIL import Image

from dual_track_opd.fc_opd.degradation import precomputed_degraded_transform


DEFAULT_SOURCE = Path(
    "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/"
    "dataset/ViRL39K/39Krelease.parquet"
)
DEFAULT_IMAGE_ROOT = Path(
    "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/ViRL39K"
)
DEFAULT_OUTPUT = Path(
    "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/"
    "fc-opd-storage/outputs/fc_opd/virl39k_va_opd/train.parquet"
)
SCHEMA_VERSION = "virl39k-qwen3vl-va-opd-v1"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-output", type=Path)
    parser.add_argument(
        "--asset-dir",
        type=Path,
        help="Default: sibling assets directory next to the train parquet",
    )
    parser.add_argument("--val-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-val", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Debug row limit; 0 means all")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _unbox(answer: Any) -> str:
    text = str(answer or "").strip()
    match = re.fullmatch(r"\\boxed\{(.*)\}", text)
    return match.group(1).strip() if match else text


def _gt_type(answer: str) -> str:
    if re.fullmatch(r"\(?[A-Ea-e]\)?\.?", answer):
        return "letter"
    if re.fullmatch(r"(yes|no|true|false)", answer.lower()):
        return "yesno"
    if re.fullmatch(r"-?\d+(\.\d+)?", answer):
        return "pure_number"
    if re.search(r"\d", answer):
        return "has_number"
    return "other"


def _choices(question: str) -> list[str]:
    choices: list[str] = []
    for line in question.splitlines():
        stripped = line.strip()
        if re.match(r"^\(?[A-E][.)]\s+", stripped):
            choices.append(stripped)
    return choices


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned[:180] or "sample"


def _materialize_degraded(
    source: Path,
    target: Path,
) -> Path:
    """Apply the exact VA-OPD paper degradation and return the target path."""
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as handle:
        image = handle.convert("RGB")
        width, height = image.size
        low_width = max(1, int(round(width * 0.1)))
        low_height = max(1, int(round(height * 0.1)))
        low = image.resize((low_width, low_height), Image.Resampling.BILINEAR)
        degraded = low.resize((width, height), Image.Resampling.NEAREST)
        degraded.save(target)
    with Image.open(source) as full, Image.open(target) as degraded:
        if full.size != degraded.size:
            raise RuntimeError(
                f"degraded dimension drift: full={full.size}, degraded={degraded.size}, path={target}"
            )
    return target


def _build_row(
    row: dict[str, Any],
    *,
    ordinal: int,
    image_root: Path,
    degraded_dir: Path,
) -> dict[str, Any] | None:
    question = str(row.get("question") or "").strip()
    answer = _unbox(row.get("answer"))
    if not question or not answer:
        return None
    if _gt_type(answer) == "other":
        return None

    image_paths = row.get("image")
    if image_paths is None:
        image_paths = []
    elif hasattr(image_paths, "tolist"):
        image_paths = image_paths.tolist()
    elif not isinstance(image_paths, (list, tuple)):
        image_paths = [image_paths]
    if not isinstance(image_paths, (list, tuple)) or not image_paths:
        return None
    full_paths = [image_root / str(path) for path in image_paths]
    if any(not path.is_file() for path in full_paths):
        return None

    placeholders = question.count("<image>")
    if placeholders > len(full_paths):
        return None
    if placeholders < len(full_paths):
        question = "<image>" * (len(full_paths) - placeholders) + question

    qid = str(row.get("qid") or f"ordinal-{ordinal:06d}")
    stem = _safe_stem(qid)
    full_entries = []
    degraded_entries = []
    images = []
    for image_index, full_path in enumerate(full_paths):
        degraded_path = degraded_dir / f"{stem}_{image_index:02d}.lowres_10pct_nearest.png"
        _materialize_degraded(full_path, degraded_path)
        full_entries.append({"path": str(full_path)})
        degraded_entries.append(
            {
                "path": str(degraded_path),
                "transform": precomputed_degraded_transform("lowres_10pct_nearest"),
            }
        )
        images.append({"bytes": full_path.read_bytes(), "path": str(full_path)})

    condition_inputs = {
        "full_images": full_entries,
        "degraded_images": degraded_entries,
    }
    source = str(row.get("source") or "ViRL39K")
    extra_info = {
        "question": question,
        "choices": _choices(question),
        "answer": answer,
        "source": source,
        "gt_source": "answer_unboxed",
        "gt_type": _gt_type(answer),
        "original_answer": str(row.get("answer") or ""),
        "condition_inputs": condition_inputs,
        "sample_uid": f"ViRL39K:{qid}",
    }
    return {
        "data_source": "ViRL39K",
        "prompt": [{"role": "user", "content": question}],
        "images": images,
        "ability": "reasoning",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": extra_info,
        "question": question,
        "condition_inputs": condition_inputs,
        "choices": extra_info["choices"],
        "answer": answer,
        "sample_uid": extra_info["sample_uid"],
    }


def _stratified_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("source") or "ViRL39K")].append(row)
    target = min(count, len(rows))
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for source in sorted(groups):
        group = groups[source]
        quota = int(len(group) * target / len(rows))
        selected.extend(rng.sample(group, min(quota, len(group))))
    remaining = target - len(selected)
    if remaining:
        selected_uids = {row["sample_uid"] for row in selected}
        pool = [row for row in rows if row["sample_uid"] not in selected_uids]
        selected.extend(rng.sample(pool, min(remaining, len(pool))))
    return selected


def _manifest_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_virl39k_va_opd_parquet(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    val_output = None if args.val_output is None else args.val_output.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output.with_suffix(".manifest.json")
    )
    asset_dir = (
        args.asset_dir.expanduser().resolve()
        if args.asset_dir is not None
        else output.parent / "assets"
    )
    degraded_dir = asset_dir / "degraded_lowres_10pct_nearest"

    if not source.is_file():
        raise FileNotFoundError(f"ViRL39K source parquet not found: {source}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"ViRL39K image root not found: {image_root}")
    for target in (output, val_output, manifest_path):
        if target is not None and target.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {target}; pass --overwrite")
    if args.val_rows < 0:
        raise ValueError("--val-rows must be non-negative")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if val_output is not None and args.val_rows == 0:
        raise ValueError("a validation output requires --val-rows > 0")

    output.parent.mkdir(parents=True, exist_ok=True)
    if val_output is not None:
        val_output.parent.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)

    source_frame = pd.read_parquet(source)
    if args.limit:
        source_frame = source_frame.iloc[: args.limit]
    if source_frame.empty:
        raise ValueError("ViRL39K source contains no rows")

    rows: list[dict[str, Any]] = []
    skipped = Counter()
    seen_uids: set[str] = set()
    for ordinal, (_, source_row) in enumerate(source_frame.iterrows()):
        row = _build_row(
            source_row.to_dict(),
            ordinal=ordinal,
            image_root=image_root,
            degraded_dir=degraded_dir,
        )
        if row is None:
            skipped["invalid_or_filtered"] += 1
            continue
        if row["sample_uid"] in seen_uids:
            skipped["duplicate_qid"] += 1
            continue
        seen_uids.add(row["sample_uid"])
        rows.append(row)
        if (len(rows) % 1000) == 0:
            print(f"prepared {len(rows)} rows", flush=True)

    if not rows:
        raise ValueError("no convertible VA-OPD rows")
    val_rows = _stratified_sample(rows, args.val_rows, args.seed)
    if args.holdout_val:
        val_uids = {row["sample_uid"] for row in val_rows}
        train_rows = [row for row in rows if row["sample_uid"] not in val_uids]
    else:
        train_rows = rows

    train_tmp = output.with_name(output.name + ".tmp")
    pd.DataFrame(train_rows).to_parquet(train_tmp, index=False)
    os.replace(train_tmp, output)
    if val_output is not None:
        val_tmp = val_output.with_name(val_output.name + ".tmp")
        pd.DataFrame(val_rows).to_parquet(val_tmp, index=False)
        os.replace(val_tmp, val_output)

    source_stat = source.stat()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "image_root": str(image_root),
        "output": str(output),
        "val_output": None if val_output is None else str(val_output),
        "asset_dir": str(asset_dir),
        "source_rows": len(source_frame),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "holdout_val": bool(args.holdout_val),
        "limit": args.limit,
        "seed": args.seed,
        "degradation": "lowres_10pct_nearest",
        "degradation_transform": {
            "scale": 0.1,
            "downsample": "bilinear",
            "upsample": "nearest",
            "restore_original_dimensions": True,
        },
        "skipped_rows": dict(skipped),
        "sample_uid_first": train_rows[0]["sample_uid"],
        "sample_uid_last": train_rows[-1]["sample_uid"],
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    build_virl39k_va_opd_parquet(_parse_args(argv))


if __name__ == "__main__":
    main()

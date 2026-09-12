#!/usr/bin/env python3
"""Build a reproducible Geometry3K parquet for online Qwen3-VL GKD/VA-OPD."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from dual_track_opd.fc_opd.degradation import (
    materialize_degraded_image,
    precomputed_degraded_transform,
)


DEFAULT_SOURCE = Path(
    "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/"
    "dataset/geometry3k/data/train-00000-of-00001.parquet"
)
DEFAULT_OUTPUT = Path(
    "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/"
    "fc-opd-storage/outputs/fc_opd/geometry3k_gkd/train.parquet"
)
SCHEMA_VERSION = "geometry3k-qwen3vl-gkd-v1"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-output", type=Path)
    parser.add_argument(
        "--asset-dir",
        type=Path,
        help="Extracted original/degraded images (default: sibling assets directory)",
    )
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument(
        "--holdout-val",
        action="store_true",
        help="Remove validation rows from train instead of retaining the historical full train set",
    )
    parser.add_argument("--limit", type=int, default=0, help="Debug-only source row limit; 0 means all rows")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _first_image_bytes(row: dict[str, Any]) -> bytes | None:
    images = row.get("images")
    if images is None:
        images = []
    if isinstance(images, np.ndarray):
        images = images.tolist()
    if not isinstance(images, (list, tuple)) or not images:
        return None
    image = images[0]
    if isinstance(image, dict):
        image = image.get("bytes")
    if isinstance(image, memoryview):
        image = image.tobytes()
    if isinstance(image, bytearray):
        image = bytes(image)
    return image if isinstance(image, bytes) and image else None


def _question_choices(problem: Any) -> tuple[str, list[str]]:
    text = str(problem or "").replace("<image>", "").strip()
    question_lines: list[str] = []
    choices: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^[A-D](?:[.\s)])", stripped):
            choices.append(stripped)
        elif stripped:
            question_lines.append(stripped)
    question = "\n".join(question_lines).strip() or text
    return question, choices


def _answer_label(raw: Any, choices: Sequence[str]) -> str:
    answer = str(raw or "").strip().upper()
    if re.fullmatch(r"[A-D]", answer):
        return answer
    if answer.isdigit():
        index = int(answer)
        # The parquet used on the cluster stores choice indices as zero-based.
        # Preserve the legacy fallback for a one-based terminal index.
        if 0 <= index < len(choices):
            return chr(ord("A") + index)
        if 1 <= index <= len(choices):
            return chr(ord("A") + index - 1)
    return answer


def _build_row(
    row: dict[str, Any],
    *,
    source_index: int,
    image_dir: Path,
    degraded_dir: Path,
) -> dict[str, Any]:
    image_bytes = _first_image_bytes(row)
    if image_bytes is None:
        raise ValueError(f"source row {source_index} has no usable image bytes")

    image_path = image_dir / f"{source_index:06d}.png"
    image_path.write_bytes(image_bytes)
    degraded_path = Path(
        materialize_degraded_image(
            str(image_path),
            mode="lowres_10pct_nearest",
            degraded_dir=str(degraded_dir),
        )
    )
    if not degraded_path.is_file():
        raise RuntimeError(f"failed to materialize degraded image for row {source_index}")

    question, choices = _question_choices(row.get("problem"))
    answer = _answer_label(row.get("answer"), choices)
    canonical_question = question
    if choices:
        canonical_question += "\n\nChoices: " + " ".join(choices)

    condition_inputs = {
        "full_image": {"path": str(image_path)},
        "degraded_image": {
            "path": str(degraded_path),
            "transform": precomputed_degraded_transform("lowres_10pct_nearest"),
        },
        # These fields are retained for the later FC-OPD condition router.  The
        # vanilla GKD baseline consumes only full_image.
        "free_caption": "A geometry diagram with labeled points and lines.",
        "task_evidence": "Points, lines, and angles are labeled in the diagram.",
        "task_visible_evidence": "The diagram shows labeled geometric elements.",
        "task_infer_evidence": "Geometric relationships can be inferred from the labels.",
        "task_solve_evidence": "Apply geometric theorems using the labeled elements.",
    }
    sample_uid = f"geometry3k:{source_index}"
    return {
        "data_source": "geometry3k",
        "prompt": [{"role": "user", "content": f"<image>\n{canonical_question}"}],
        "images": [{"bytes": image_bytes, "path": str(image_path)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": source_index,
            "question": canonical_question,
            "choices": choices,
            "answer": answer,
            "condition_inputs": condition_inputs,
            "sample_uid": sample_uid,
        },
        "question": canonical_question,
        "condition_inputs": condition_inputs,
        "choices": choices,
        "answer": answer,
        "sample_uid": sample_uid,
    }


def _manifest_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_geometry3k_parquet(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.expanduser().resolve()
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
    image_dir = asset_dir / "images"
    degraded_dir = asset_dir / "degraded_lowres_10pct_nearest"

    if not source.is_file():
        raise FileNotFoundError(f"Geometry3K source parquet not found: {source}")
    for target in (output, val_output, manifest_path):
        if target is not None and target.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {target}; pass --overwrite")
    if args.val_size < 0:
        raise ValueError("--val-size must be non-negative")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")

    output.parent.mkdir(parents=True, exist_ok=True)
    if val_output is not None:
        val_output.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_parquet(source)
    if args.limit:
        source_df = source_df.iloc[: args.limit]
    if len(source_df) < 1:
        raise ValueError("Geometry3K source contains no rows")

    rows = []
    for ordinal, (_, source_row) in enumerate(source_df.iterrows()):
        raw_index = source_row.get("index", ordinal)
        source_index = ordinal if pd.isna(raw_index) else int(raw_index)
        rows.append(
            _build_row(
                source_row.to_dict(),
                source_index=source_index,
                image_dir=image_dir,
                degraded_dir=degraded_dir,
            )
        )
        if (ordinal + 1) % 200 == 0:
            print(f"processed {ordinal + 1}/{len(source_df)}", flush=True)

    val_size = min(args.val_size, max(0, len(rows) - 1)) if val_output is not None else 0
    val_rows = rows[-val_size:] if val_size else []
    train_rows = rows[:-val_size] if (args.holdout_val and val_size) else rows
    pd.DataFrame(train_rows).to_parquet(output, index=False)
    if val_output is not None:
        pd.DataFrame(val_rows).to_parquet(val_output, index=False)

    source_stat = source.stat()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "output": str(output),
        "val_output": None if val_output is None else str(val_output),
        "asset_dir": str(asset_dir),
        "source_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "holdout_val": bool(args.holdout_val),
        "limit": args.limit,
        "sample_uid_first": train_rows[0]["sample_uid"],
        "sample_uid_last": train_rows[-1]["sample_uid"],
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    build_geometry3k_parquet(_parse_args(argv))


if __name__ == "__main__":
    main()

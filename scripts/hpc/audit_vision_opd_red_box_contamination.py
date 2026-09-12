#!/usr/bin/env python
"""Audit Vision-OPD images for baked-in red bounding-box cues."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dual_track_opd.fc_opd.dataset_adapters import load_normalized_records  # noqa: E402


@dataclass(frozen=True)
class RedStats:
    width: int
    height: int
    red_pixels: int
    strong_red_pixels: int
    red_ratio: float
    strong_red_ratio: float
    red_component_count: int
    largest_red_component: int
    suspected: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "red_pixels": self.red_pixels,
            "strong_red_pixels": self.strong_red_pixels,
            "red_ratio": self.red_ratio,
            "strong_red_ratio": self.strong_red_ratio,
            "red_component_count": self.red_component_count,
            "largest_red_component": self.largest_red_component,
            "red_box_suspected": self.suspected,
            "error": self.error,
        }


def image_red_stats(path: str | Path, *, threshold: float = 0.002) -> RedStats:
    from PIL import Image

    path = Path(path).expanduser()
    if not path.is_file():
        return RedStats(0, 0, 0, 0, 0.0, 0.0, 0, 0, False, f"missing image: {path}")
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            pixels = list(rgb.getdata())
    except Exception as exc:  # noqa: BLE001
        return RedStats(0, 0, 0, 0, 0.0, 0.0, 0, 0, False, f"image read error: {exc}")

    red_mask = []
    strong_mask = []
    for r, g, b in pixels:
        is_red = r >= 150 and r > g * 1.35 and r > b * 1.35
        is_strong = r >= 180 and g <= 90 and b <= 90 and r > g * 1.8 and r > b * 1.8
        red_mask.append(is_red)
        strong_mask.append(is_strong)
    total = max(1, width * height)
    components = _component_sizes(strong_mask, width, height)
    strong_count = sum(strong_mask)
    red_count = sum(red_mask)
    strong_ratio = strong_count / total
    red_ratio = red_count / total
    suspected = strong_ratio >= threshold or red_ratio >= threshold * 2.0 or bool(
        components and max(components) >= max(20, int(total * threshold * 0.25))
    )
    return RedStats(
        width=width,
        height=height,
        red_pixels=red_count,
        strong_red_pixels=strong_count,
        red_ratio=red_ratio,
        strong_red_ratio=strong_ratio,
        red_component_count=len(components),
        largest_red_component=max(components) if components else 0,
        suspected=suspected,
    )


def _component_sizes(mask: Sequence[bool], width: int, height: int) -> list[int]:
    seen = bytearray(width * height)
    sizes: list[int] = []
    for start, flag in enumerate(mask):
        if not flag or seen[start]:
            continue
        stack = [start]
        seen[start] = 1
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            x = current % width
            y = current // width
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                idx = ny * width + nx
                if mask[idx] and not seen[idx]:
                    seen[idx] = 1
                    stack.append(idx)
        sizes.append(size)
    return sizes


def audit_records(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int | None,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = records if limit is None else records[:limit]
    for record in selected:
        image_path = str(record.get("image_path") or "")
        bbox_paths = [str(path) for path in record.get("bbox_image_paths", [])]
        full_stats = image_red_stats(image_path, threshold=threshold)
        bbox_stats = [image_red_stats(path, threshold=threshold).to_dict() for path in bbox_paths]
        rows.append(
            {
                "sample_uid": str(record.get("sample_uid", "")),
                "source_dataset": str(record.get("source_dataset", "vision-opd-6k")),
                "source_index": int(record.get("source_index", len(rows))),
                "question_id": str(record.get("question_id", "")),
                "question": str(record.get("question", "")),
                "image_path": image_path,
                "bbox_image_paths": bbox_paths,
                "full_image": full_stats.to_dict(),
                "bbox_images": bbox_stats,
                "full_image_red_box_suspected": full_stats.suspected,
                "bbox_image_red_box_suspected": any(bool(item["red_box_suspected"]) for item in bbox_stats),
            }
        )
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], *, threshold: float) -> dict[str, Any]:
    total = len(rows)
    full_suspected = sum(bool(row.get("full_image_red_box_suspected")) for row in rows)
    bbox_suspected = sum(bool(row.get("bbox_image_red_box_suspected")) for row in rows)
    ranked = sorted(
        rows,
        key=lambda row: float(dict(row.get("full_image", {})).get("strong_red_ratio", 0.0)),
        reverse=True,
    )[:10]
    conclusion = (
        "red_box_likely_baked_into_full_images"
        if total and full_suspected / total >= 0.05
        else "no_dataset_wide_full_image_red_box_signal_detected"
    )
    return {
        "num_samples_checked": total,
        "red_threshold": threshold,
        "full_image_red_box_suspected_rate": None if total == 0 else full_suspected / total,
        "bbox_image_red_box_suspected_rate": None if total == 0 else bbox_suspected / total,
        "examples_with_highest_red_ratio": [
            {
                "sample_uid": row.get("sample_uid"),
                "image_path": row.get("image_path"),
                "red_ratio": dict(row.get("full_image", {})).get("red_ratio"),
                "strong_red_ratio": dict(row.get("full_image", {})).get("strong_red_ratio"),
            }
            for row in ranked
        ],
        "likely_conclusion": conclusion,
        "red_box_contamination_suspected": conclusion == "red_box_likely_baked_into_full_images",
    }


def write_contact_sheet(rows: Sequence[Mapping[str, Any]], path: Path, *, max_items: int = 32) -> None:
    from PIL import Image, ImageDraw

    thumbs = []
    for row in rows[:max_items]:
        image_path = Path(str(row.get("image_path", ""))).expanduser()
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((220, 220))
            canvas = Image.new("RGB", (240, 270), "white")
            canvas.paste(thumb, ((240 - thumb.width) // 2, 5))
            draw = ImageDraw.Draw(canvas)
            stats = dict(row.get("full_image", {}))
            draw.text((8, 230), str(row.get("sample_uid", ""))[:32], fill=(0, 0, 0))
            draw.text((8, 246), f"strong={float(stats.get('strong_red_ratio', 0.0)):.4f}", fill=(180, 0, 0))
            thumbs.append(canvas)
    if not thumbs:
        return
    cols = 4
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 240, rows_count * 270), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 240, (index // cols) * 270))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def export_examples(rows: Sequence[Mapping[str, Any]], output_dir: Path, *, count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        rows,
        key=lambda row: float(dict(row.get("full_image", {})).get("strong_red_ratio", 0.0)),
        reverse=True,
    )[:count]
    for index, row in enumerate(ranked):
        source = Path(str(row.get("image_path", ""))).expanduser()
        if source.is_file():
            target = output_dir / f"{index:03d}_{str(row.get('sample_uid', 'sample')).replace(':', '_')}{source.suffix}"
            shutil.copy2(source, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-type", default="vision_opd_parquet")
    parser.add_argument("--source-dataset", default="vision-opd-6k")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--red-threshold", type=float, default=0.002)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--examples-dir", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_normalized_records(
        args.dataset,
        args.dataset_type,
        source_dataset=args.source_dataset,
    )
    rows = audit_records(records, limit=args.limit, threshold=args.red_threshold)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = summarize(rows, threshold=args.red_threshold)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_contact_sheet(rows, args.contact_sheet, max_items=args.num_examples)
    export_examples(rows, args.examples_dir, count=args.num_examples)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

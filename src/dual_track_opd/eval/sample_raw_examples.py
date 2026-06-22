"""Sample raw VLM responses for qualitative inspection."""

from __future__ import annotations

import argparse
import json
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
IMAGE_KEYS = ("image", "image_path", "image_paths", "images")
PATH_HINTS = ("image", "img", "path", "file")


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


def _recursive_image_path(value: Any) -> str:
    found = _recursive_first_field(value, IMAGE_KEYS)
    if found not in (None, ""):
        return _stringify(found)

    candidates: list[str] = []

    def collect(nested: Any, key_name: str = "") -> None:
        if isinstance(nested, dict):
            for key, inner in nested.items():
                collect(inner, key)
        elif isinstance(nested, list):
            for inner in nested:
                collect(inner, key_name)
        elif isinstance(nested, str):
            lowered_key = key_name.lower()
            if _looks_like_image_path(nested) or any(hint in lowered_key for hint in PATH_HINTS):
                if _looks_like_image_path(nested):
                    candidates.append(nested)

    collect(value)
    return candidates[0] if candidates else ""


def sample_raw_examples(
    raw_dir: str | Path,
    manifest: str | Path,
    samples_per_dataset: int = 2,
    strategy: str = "first",
) -> list[dict[str, Any]]:
    entries = load_manifest(manifest)
    raw_dir = Path(raw_dir)
    samples: list[dict[str, Any]] = []

    for entry in entries:
        raw_path = raw_dir / entry.file
        rows = _read_jsonl(raw_path)
        for index in _select_indices(len(rows), samples_per_dataset, strategy):
            item = rows[index]
            samples.append(
                {
                    "dataset": entry.dataset,
                    "file": entry.file,
                    "row_index": index + 1,
                    "row_id": get_row_id(item, index + 1),
                    "scoring_type": entry.scoring_type,
                    "question": _field(item, QUESTION_KEYS),
                    "options": _field(item, OPTION_KEYS),
                    "image": _recursive_image_path(item),
                    "prediction": get_prediction(item),
                    "reasoning": _field(item, REASONING_KEYS),
                    "ground_truths": get_ground_truths(item),
                    "finish_reason": get_finish_reason(item),
                    "error": _stringify(item.get("error")),
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
    args = parser.parse_args()

    rows = sample_raw_examples(
        raw_dir=args.raw_dir,
        manifest=args.manifest,
        samples_per_dataset=args.samples_per_dataset,
        strategy=args.strategy,
    )
    write_jsonl(args.out_jsonl, rows)
    write_markdown(args.out_md, rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Inspect Visual Grounding Gap OPD diagnostic JSONL files."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


TAG_OR_PUNCT = set("<>/_.:,;()[]{}+-=*|\\\"' \n\t")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    rows = _read_jsonl(args.jsonl)
    summary = inspect_rows(rows, top_n=args.top_n)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def inspect_rows(rows: list[Mapping[str, Any]], *, top_n: int = 20) -> dict[str, Any]:
    va_raw = []
    va_pos = []
    gap_pos = []
    weights = []
    by_chunk: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    weighted_tokens = []
    warnings = []

    for row in rows:
        block = row.get("visual_grounding_gap_opd")
        if not isinstance(block, Mapping):
            warnings.append({"sample_uid": row.get("sample_uid"), "warning": "missing_vgg_opd_block"})
            continue
        labels = list(row.get("chunk_labels") or _labels_from_spans(row, len(block.get("token_weights", []))))
        tokens = list(row.get("response_tokens") or row.get("response_token_texts") or [])
        for index, weight in enumerate(block.get("token_weights", [])):
            weight = float(weight)
            weights.append(weight)
            chunk = labels[index] if index < len(labels) else "outside"
            by_chunk[chunk]["weight"].append(weight)
            for key, target in (("va_raw", va_raw), ("va_pos", va_pos), ("gap_pos", gap_pos)):
                values = block.get(key, [])
                if index < len(values):
                    value = float(values[index])
                    target.append(value)
                    by_chunk[chunk][key].append(value)
            token_text = str(tokens[index]) if index < len(tokens) else ""
            if weight > 0:
                weighted_tokens.append(
                    {
                        "sample_uid": row.get("sample_uid", row.get("rollout_uid")),
                        "token_index": index,
                        "chunk": chunk,
                        "token": token_text,
                        "weight": weight,
                        "va_raw": _nth(block.get("va_raw", []), index),
                        "gap_pos": _nth(block.get("gap_pos", []), index),
                    }
                )

    weighted_tokens.sort(key=lambda item: item["weight"], reverse=True)
    top = weighted_tokens[:top_n]
    suspicious = [
        item for item in top if item["token"] and all(ch in TAG_OR_PUNCT for ch in item["token"])
    ]
    if top and len(suspicious) / len(top) >= 0.5:
        warnings.append(
            {
                "warning": "top_weighted_tokens_are_mostly_markup_or_punctuation",
                "fraction": len(suspicious) / len(top),
            }
        )

    return {
        "rows": len(rows),
        "distributions": {
            "va_raw": _stats(va_raw),
            "va_pos": _stats(va_pos),
            "gap_pos": _stats(gap_pos),
            "token_weight": _stats(weights),
        },
        "by_chunk": {
            chunk: {metric: _stats(values) for metric, values in metrics.items()}
            for chunk, metrics in sorted(by_chunk.items())
        },
        "top_weighted_tokens": top,
        "chunk_counts": dict(Counter(item["chunk"] for item in weighted_tokens)),
        "warnings": warnings,
    }


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    values = list(values)
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": sorted_values[0],
        "p50": sorted_values[len(sorted_values) // 2],
        "p90": sorted_values[int(0.9 * (len(sorted_values) - 1))],
        "max": sorted_values[-1],
    }


def _labels_from_spans(row: Mapping[str, Any], length: int) -> list[str]:
    labels = ["outside"] * length
    spans = row.get("chunk_spans", {})
    if not isinstance(spans, Mapping):
        return labels
    for chunk, value in spans.items():
        if chunk == "format_valid":
            continue
        for start, end in _spans(value):
            for index in range(max(0, start), min(length, end)):
                labels[index] = str(chunk)
    return labels


def _spans(value: Any) -> list[tuple[int, int]]:
    if isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) for item in value):
        return [(int(value[0]), int(value[1]))]
    if isinstance(value, list):
        return [(int(item[0]), int(item[1])) for item in value if isinstance(item, list) and len(item) == 2]
    return []


def _nth(values: Any, index: int) -> float | None:
    if isinstance(values, list) and index < len(values):
        return float(values[index])
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()

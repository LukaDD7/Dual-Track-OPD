"""Git-safe scalar reports for causal-state probe outputs."""

from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return None if not finite else float(fmean(finite))


def flatten_candidates(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for candidate in record.get("candidate_windows") or ():
            relay = candidate.get("relay_gain_by_length") or {}
            relay_probability = candidate.get("relay_probability_by_length") or {}
            visual = {
                str(row.get("condition")): row
                for row in candidate.get("visual_continuations") or ()
            }
            rows.append({
                "prompt_id": record.get("prompt_id"),
                "trajectory_id": record.get("trajectory_id"),
                "trajectory_correct": record.get("trajectory_correct"),
                "support_n_correct": (record.get("student_support") or {}).get("n_correct"),
                "support_n_rollouts": (record.get("student_support") or {}).get("n_rollouts"),
                "candidate_id": candidate.get("candidate_id"),
                "start": candidate.get("start"),
                "end": candidate.get("end"),
                "anchor": candidate.get("anchor"),
                "relative_position": candidate.get("relative_position"),
                "sources": "|".join(candidate.get("sources") or ()),
                "student_js_full_degraded": candidate.get("student_js_full_degraded"),
                "student_js_full_null": candidate.get("student_js_full_null"),
                "js_drop_full_degraded": candidate.get("js_drop_full_degraded"),
                "js_drop_full_null": candidate.get("js_drop_full_null"),
                "visual_full_success": (visual.get("full") or {}).get("pass_rate"),
                "visual_degraded_success": (visual.get("degraded") or {}).get("pass_rate"),
                "visual_null_success": (visual.get("null") or {}).get("pass_rate"),
                "visual_fine_gain": candidate.get("visual_fine_gain"),
                "visual_all_gain": candidate.get("visual_all_gain"),
                "relay_gain_l32": relay.get("32"),
                "relay_gain_l64": relay.get("64"),
                "relay_gain_l128": relay.get("128"),
                "relay_probability_l32": relay_probability.get("32"),
                "relay_probability_l64": relay_probability.get("64"),
                "relay_probability_l128": relay_probability.get("128"),
                "unaided_success": (candidate.get("unaided_continuation") or {}).get("pass_rate"),
                "transport_success": (candidate.get("transport_continuation") or {}).get("pass_rate"),
                "wrong_prefix_success": (candidate.get("wrong_prefix_continuation") or {}).get("pass_rate"),
                "transport_gain": candidate.get("transport_gain"),
                "transport_probability": candidate.get("transport_probability"),
                "transport_vs_wrong_gain": candidate.get("transport_vs_wrong_gain"),
                "transport_vs_wrong_probability": candidate.get("transport_vs_wrong_probability"),
                "wrong_prefix_gain": candidate.get("wrong_prefix_gain"),
                "answer_leakage": candidate.get("answer_leakage"),
                "student_teacher_overlap": candidate.get("student_teacher_overlap"),
                "state_class": candidate.get("state_class"),
                "notes": "|".join(candidate.get("notes") or ()),
            })
    return rows


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_rows = flatten_candidates(records)
    prompt_ids = {str(record.get("prompt_id")) for record in records}
    outcomes = Counter(
        "correct" if record.get("trajectory_correct") is True else
        "wrong" if record.get("trajectory_correct") is False else "unknown"
        for record in records
    )
    states = Counter(str(row.get("state_class") or "unclassified") for row in candidate_rows)
    source_counts: Counter[str] = Counter()
    for row in candidate_rows:
        source_counts.update(value for value in str(row.get("sources") or "").split("|") if value)

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in candidate_rows if row.get(key) is not None]

    return {
        "schema_version": "causal-state-probe-summary-v1",
        "record_count": len(records),
        "prompt_count": len(prompt_ids),
        "candidate_count": len(candidate_rows),
        "trajectory_outcome_counts": dict(sorted(outcomes.items())),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "state_class_counts": dict(sorted(states.items())),
        "means": {
            "student_js_full_degraded": _mean(values("student_js_full_degraded")),
            "student_js_full_null": _mean(values("student_js_full_null")),
            "visual_fine_gain": _mean(values("visual_fine_gain")),
            "visual_all_gain": _mean(values("visual_all_gain")),
            "transport_gain": _mean(values("transport_gain")),
            "answer_leakage": _mean(values("answer_leakage")),
        },
    }


def write_reports(output_dir: str | Path, records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "causal_state_records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")

    rows = flatten_candidates(records)
    csv_path = output / "candidate_windows.csv"
    fieldnames = list(rows[0]) if rows else ["prompt_id", "trajectory_id", "candidate_id"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output / "summary.json"
    _write_json(summary_path, summarize_records(records))
    svg_path = output / "trajectory_overview.svg"
    svg_path.write_text(render_trajectory_svg(records), encoding="utf-8")
    return {
        "records_jsonl": str(records_path),
        "candidate_csv": str(csv_path),
        "summary_json": str(summary_path),
        "overview_svg": str(svg_path),
    }


def render_trajectory_svg(
    records: Sequence[Mapping[str, Any]],
    *,
    max_records: int = 12,
    width: int = 1200,
    row_height: int = 150,
) -> str:
    shown = list(records[:max_records])
    height = max(120, 70 + row_height * max(len(shown), 1))
    left, right = 180, width - 30
    plot_width = right - left
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:ui-monospace,monospace;font-size:12px}.axis{stroke:#94a3b8}.deg{fill:none;stroke:#2563eb;stroke-width:2}.nul{fill:none;stroke:#dc2626;stroke-width:2}.cand{stroke:#16a34a;stroke-width:1;stroke-dasharray:3 3}</style>',
        '<text x="20" y="24" font-size="16">Student visual JS along fixed trajectories</text>',
        '<text x="20" y="44" fill="#2563eb">blue: full vs degraded</text>',
        '<text x="260" y="44" fill="#dc2626">red: full vs null</text>',
    ]
    if not shown:
        parts.append('<text x="20" y="90">No completed records.</text>')
    for row_index, record in enumerate(shown):
        top = 65 + row_index * row_height
        signals = list(record.get("token_signals") or ())
        degraded = [float(value.get("js_full_degraded") or 0.0) for value in signals]
        null = [float(value.get("js_full_null") or 0.0) for value in signals]
        maximum = max(degraded + null + [1e-9])
        bottom = top + row_height - 30
        parts.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
        label = html.escape(f"{record.get('prompt_id')} / {record.get('trajectory_id')}")
        parts.append(f'<text x="10" y="{top + 18}">{label}</text>')

        def points(values: Sequence[float]) -> str:
            if not values:
                return ""
            return " ".join(
                f"{left + plot_width * index / max(len(values) - 1, 1):.1f},{bottom - 100 * value / maximum:.1f}"
                for index, value in enumerate(values)
            )

        if degraded:
            parts.append(f'<polyline class="deg" points="{points(degraded)}"/>')
        if null:
            parts.append(f'<polyline class="nul" points="{points(null)}"/>')
        for candidate in record.get("candidate_windows") or ():
            relative = float(candidate.get("relative_position") or 0.0)
            x = left + plot_width * relative
            parts.append(f'<line class="cand" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>')
    parts.append("</svg>\n")
    return "".join(parts)

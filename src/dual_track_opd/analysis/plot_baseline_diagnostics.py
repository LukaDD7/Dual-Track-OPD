"""Render lightweight SVG diagnostics for baseline score tables."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


TIER_COLORS = {
    "high_coverage_deterministic_mcq": "#2f6f9f",
    "rough_exact_match_diagnostic": "#78a641",
    "low_coverage_diagnostic": "#c9792b",
    "needs_judge": "#7a7a7a",
}


def _float_or_none(value: str) -> float | None:
    return None if value in ("", None) else float(value)


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render_svg(rows: list[dict[str, str]], title: str) -> str:
    width = 1500
    height = 760
    margin_left = 90
    margin_right = 60
    margin_top = 80
    margin_bottom = 190
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    slot = plot_width / max(len(rows), 1)
    bar_width = min(54, slot * 0.62)

    def x_center(index: int) -> float:
        return margin_left + slot * index + slot / 2

    def y(value: float) -> float:
        return margin_top + plot_height * (1 - value)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Helvetica, Arial, sans-serif; fill: #202124; }",
        ".title { font-size: 28px; font-weight: 700; }",
        ".subtitle { font-size: 16px; fill: #5f6368; }",
        ".axis { stroke: #3c4043; stroke-width: 1.2; }",
        ".grid { stroke: #dadce0; stroke-width: 1; }",
        ".tick { font-size: 13px; fill: #5f6368; }",
        ".label { font-size: 13px; }",
        ".legend { font-size: 14px; fill: #3c4043; }",
        ".note { font-size: 13px; fill: #5f6368; }",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="38" class="title">{html.escape(title)}</text>',
        f'<text x="{margin_left}" y="62" class="subtitle">Bars show parser-conditional accuracy; black dots show scoring coverage. Internal diagnostic, not official benchmark accuracy.</text>',
    ]

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = y(tick)
        parts.append(
            f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{margin_left - 12}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{int(tick * 100)}%</text>'
        )

    parts.append(
        f'<line x1="{margin_left}" x2="{margin_left}" y1="{margin_top}" y2="{margin_top + plot_height}" class="axis"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{margin_top + plot_height}" y2="{margin_top + plot_height}" class="axis"/>'
    )

    for index, row in enumerate(rows):
        cx = x_center(index)
        accuracy = _float_or_none(row["accuracy"])
        coverage = float(row["coverage"])
        color = TIER_COLORS.get(row["interpretation_tier"], "#5f6368")
        if accuracy is not None:
            bar_h = plot_height * accuracy
            parts.append(
                f'<rect x="{cx - bar_width / 2:.1f}" y="{margin_top + plot_height - bar_h:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.92"/>'
            )
            parts.append(
                f'<text x="{cx:.1f}" y="{margin_top + plot_height - bar_h - 6:.1f}" text-anchor="middle" class="tick">{accuracy * 100:.0f}</text>'
            )
        else:
            parts.append(
                f'<line x1="{cx - bar_width / 2:.1f}" x2="{cx + bar_width / 2:.1f}" y1="{margin_top + plot_height - 4:.1f}" y2="{margin_top + plot_height - 4:.1f}" stroke="{color}" stroke-width="4"/>'
            )
            parts.append(
                f'<text x="{cx:.1f}" y="{margin_top + plot_height - 14:.1f}" text-anchor="middle" class="tick">n/a</text>'
            )

        cy = y(coverage)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="#111111" stroke="white" stroke-width="1.4"/>'
        )
        parts.append(
            f'<text transform="translate({cx:.1f},{height - margin_bottom + 26}) rotate(42)" text-anchor="start" class="label">{html.escape(row["dataset"])}</text>'
        )

    legend_x = margin_left
    legend_y = height - 74
    legend_items = [
        ("high_coverage_deterministic_mcq", "High-coverage MCQ diagnostic"),
        ("rough_exact_match_diagnostic", "Rough exact-match diagnostic"),
        ("low_coverage_diagnostic", "Low-coverage diagnostic"),
        ("needs_judge", "Needs judge"),
    ]
    for offset, (tier, label) in enumerate(legend_items):
        x = legend_x + offset * 325
        parts.append(f'<rect x="{x}" y="{legend_y}" width="16" height="16" fill="{TIER_COLORS[tier]}"/>')
        parts.append(f'<text x="{x + 24}" y="{legend_y + 13}" class="legend">{html.escape(label)}</text>')
    parts.append(f'<circle cx="{legend_x}" cy="{legend_y + 42}" r="5.5" fill="#111111"/>')
    parts.append(f'<text x="{legend_x + 24}" y="{legend_y + 47}" class="legend">Coverage marker</text>')
    parts.append(
        f'<text x="{width - margin_right}" y="{height - 18}" text-anchor="end" class="note">Generated from qwen3vl8b_baseline_score_diagnostics.csv</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, help="Input diagnostics CSV.")
    parser.add_argument("--out-svg", required=True, help="Output SVG path.")
    parser.add_argument(
        "--title",
        default="Qwen3-VL-8B Baseline Diagnostic Scores",
        help="Figure title.",
    )
    args = parser.parse_args()

    rows = load_rows(args.diagnostics)
    Path(args.out_svg).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_svg).write_text(render_svg(rows, args.title), encoding="utf-8")


if __name__ == "__main__":
    main()


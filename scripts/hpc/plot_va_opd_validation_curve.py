#!/usr/bin/env python
"""Render a VA-OPD validation-history JSON file as a small dependency-free SVG."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


def load_scores(path: Path) -> Dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_scores = payload.get("scores", {})
    if not isinstance(raw_scores, dict):
        raise ValueError("history JSON must contain a scores object")
    scores: Dict[int, float] = {}
    for raw_step, raw_score in raw_scores.items():
        step = int(raw_step)
        score = float(raw_score)
        if step < 0 or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid validation point: step={step}, score={score}")
        scores[step] = score
    if not scores:
        raise ValueError("history JSON contains no validation points")
    return scores


def trailing_mean(values: List[float], window: int) -> List[float]:
    result: List[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(statistics.mean(values[start : index + 1]))
    return result


def svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "start") -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" '
        f'font-size="{size}" fill="#24292f" text-anchor="{anchor}">{escaped}</text>'
    )


def render(
    scores: Dict[int, float],
    output: Path,
    *,
    run_id: str,
    width: int,
    height: int,
    rolling_window: int,
) -> dict:
    steps = sorted(scores)
    values = [scores[step] for step in steps]
    rolled = trailing_mean(values, rolling_window)
    best_step, best_score = max(scores.items(), key=lambda item: (item[1], item[0]))
    initial_score = scores[min(scores)]
    latest_step = max(scores)
    latest_score = scores[latest_step]

    left, right = 76.0, float(width - 30)
    top, bottom = 88.0, float(height - 58)
    y_low = min(0.56, min(values) - 0.005)
    y_high = max(0.67, max(values) + 0.005)

    def xy(step: int, score: float) -> Tuple[float, float]:
        x_ratio = step / latest_step if latest_step else 0.0
        y_ratio = (score - y_low) / (y_high - y_low)
        return left + x_ratio * (right - left), bottom - y_ratio * (bottom - top)

    raw_points = [xy(step, scores[step]) for step in steps]
    rolling_points = [xy(step, rolled[i]) for i, step in enumerate(steps)]
    raw_path = " ".join(f"{x:.1f},{y:.1f}" for x, y in raw_points)
    rolling_path = " ".join(f"{x:.1f},{y:.1f}" for x, y in rolling_points)

    lines: List[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
    )
    lines.append("<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>")
    lines.append(svg_text(30, 34, "VA-OPD ViRL39K validation curve", size=20))
    lines.append(
        svg_text(
            30,
            58,
            f"run: {run_id}  |  points: {len(scores)}  |  "
            f"best: step {best_step} = {best_score:.3f}  |  "
            f"initial: {initial_score:.3f}  |  gain: +{(best_score-initial_score)*100:.1f} points",
            size=12,
        )
    )

    for tick in range(int(y_low * 100), int(y_high * 100) + 1):
        score = tick / 100
        if score < y_low or score > y_high:
            continue
        _, y = xy(0, score)
        lines.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" '
            'stroke="#d0d7de" stroke-width="1"/>'
        )
        lines.append(svg_text(left - 10, y + 4, f"{score:.2f}", anchor="end", size=11))

    for step in range(0, latest_step + 1, 200):
        x, _ = xy(step, y_low)
        lines.append(
            f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
            'stroke="#d0d7de" stroke-width="1"/>'
        )
        lines.append(svg_text(x, bottom + 24, str(step), anchor="middle", size=11))

    initial_x, initial_y = xy(0, initial_score)
    best_x, best_y = xy(best_step, best_score)
    latest_x, latest_y = xy(latest_step, latest_score)
    lines.append(
        f'<line x1="{left:.1f}" y1="{initial_y:.1f}" x2="{right:.1f}" '
        f'y2="{initial_y:.1f}" stroke="#0969da" stroke-width="1.5" stroke-dasharray="6 5"/>'
    )
    lines.append(
        f'<line x1="{left:.1f}" y1="{best_y:.1f}" x2="{right:.1f}" '
        f'y2="{best_y:.1f}" stroke="#cf222e" stroke-width="1.5" stroke-dasharray="6 5"/>'
    )
    lines.append(
        f'<polyline points="{raw_path}" fill="none" stroke="#57ab5a" stroke-width="2" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
    )
    lines.append(
        f'<polyline points="{rolling_path}" fill="none" stroke="#bc4c00" stroke-width="3.5" '
        'stroke-linejoin="round" stroke-linecap="round" opacity="0.85"/>'
    )
    for x, y in raw_points:
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="#24292f"/>')

    lines.append(f'<circle cx="{best_x:.1f}" cy="{best_y:.1f}" r="6" fill="#cf222e"/>')
    lines.append(
        svg_text(best_x + 10, best_y - 10, f"best {best_step}: {best_score:.3f}", size=12)
    )
    lines.append(f'<circle cx="{latest_x:.1f}" cy="{latest_y:.1f}" r="6" fill="#8250df"/>')
    lines.append(
        svg_text(latest_x - 10, latest_y + 22, f"latest {latest_step}: {latest_score:.3f}", anchor="end", size=12)
    )
    lines.append(
        svg_text(initial_x + 8, initial_y - 8, f"initial: {initial_score:.3f}", size=12)
    )
    lines.append(svg_text(left, top - 16, "validation score", size=11))
    lines.append(svg_text((left + right) / 2, height - 16, "optimizer step", anchor="middle", size=11))
    lines.append(
        svg_text(left + 12, top + 10, "raw (green) and 5-point trailing mean (orange)", size=11)
    )
    lines.append("</svg>\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return {
        "points": len(scores),
        "mean": statistics.mean(values),
        "initial": initial_score,
        "best_step": best_step,
        "best_score": best_score,
        "latest_step": latest_step,
        "latest_score": latest_score,
        "absolute_gain": best_score - initial_score,
        "relative_gain": best_score / initial_score - 1.0,
        "output": str(output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="VA-OPD run")
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=430)
    parser.add_argument("--rolling-window", type=int, default=5)
    args = parser.parse_args()
    if args.width < 500 or args.height < 250:
        parser.error("--width and --height must be at least 500x250")
    if args.rolling_window < 1:
        parser.error("--rolling-window must be positive")
    return args


def main() -> None:
    args = parse_args()
    scores = load_scores(args.history)
    summary = render(
        scores,
        args.output,
        run_id=args.run_id,
        width=args.width,
        height=args.height,
        rolling_window=args.rolling_window,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

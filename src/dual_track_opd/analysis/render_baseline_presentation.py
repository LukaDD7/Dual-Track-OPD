"""Render a compact HTML briefing for baseline diagnostics."""

from __future__ import annotations

import argparse
import csv
import html
import re
from pathlib import Path


def _pct(value: str) -> str:
    return "n/a" if value == "" else f"{float(value) * 100:.1f}%"


def load_diagnostics(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_sample_markdown(path: str | Path, limit: int = 8) -> list[dict[str, str]]:
    text = Path(path).read_text(encoding="utf-8")
    chunks = re.split(r"\n## ", text)
    cards: list[dict[str, str]] = []
    for chunk in chunks[1:]:
        lines = chunk.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:])
        card = {"title": title}
        for key in ["file", "row_index", "scoring_type", "finish_reason", "image"]:
            match = re.search(rf"^- {key}: `([^`]*)`", body, flags=re.MULTILINE)
            card[key] = match.group(1) if match else ""
        for label, target in [
            ("Question", "question"),
            ("Options", "options"),
            ("Prediction", "prediction"),
            ("Reasoning / Explanation", "reasoning"),
            ("Ground Truths", "ground_truths"),
        ]:
            match = re.search(
                rf"\*\*{re.escape(label)}\*\*\n\n(.*?)(?=\n\n\*\*|\Z)",
                body,
                flags=re.DOTALL,
            )
            card[target] = match.group(1).strip() if match else ""
        cards.append(card)
        if len(cards) >= limit:
            break
    return cards


def render_html(
    diagnostics: list[dict[str, str]],
    sample_cards: list[dict[str, str]],
    svg_path: str | Path,
) -> str:
    total_examples = sum(int(row["n"]) for row in diagnostics)
    scored_examples = sum(int(row["scored_n"]) for row in diagnostics)
    correct = sum(int(float(row["accuracy"]) * int(row["scored_n"])) for row in diagnostics if row["accuracy"])
    unparsed = sum(int(row["unparsed_rows"]) for row in diagnostics)
    length_rows = sum(int(row["length_rows"]) for row in diagnostics)
    weighted_acc = correct / scored_examples if scored_examples else 0.0
    coverage = scored_examples / total_examples if total_examples else 0.0

    high_coverage = [
        row
        for row in diagnostics
        if row["interpretation_tier"] in {
            "high_coverage_deterministic_mcq",
            "rough_exact_match_diagnostic",
        }
    ]
    low_coverage = [
        row
        for row in diagnostics
        if row["interpretation_tier"] == "low_coverage_diagnostic"
    ]

    svg_text = Path(svg_path).read_text(encoding="utf-8") if Path(svg_path).exists() else ""

    rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['dataset'])}</td>"
        f"<td>{html.escape(row['interpretation_tier'])}</td>"
        f"<td>{_pct(row['coverage'])}</td>"
        f"<td>{_pct(row['accuracy'])}</td>"
        f"<td>{html.escape(row['unparsed_rows'])}</td>"
        f"<td>{html.escape(row['length_rows'])}</td>"
        "</tr>"
        for row in diagnostics
    )

    cards_html = []
    for card in sample_cards:
        image = card.get("image", "")
        image_html = ""
        if image and image != "(not found in recognized fields)":
            escaped_image = html.escape(image)
            image_html = f'<img src="{escaped_image}" alt="sample image" />'
        cards_html.append(
            f"""
            <section class="card">
              <h3>{html.escape(card['title'])}</h3>
              <p class="meta">type={html.escape(card.get('scoring_type', ''))} · finish={html.escape(card.get('finish_reason', ''))}</p>
              {image_html}
              <h4>Question</h4>
              <pre>{html.escape(card.get('question', ''))}</pre>
              <h4>Prediction</h4>
              <pre>{html.escape(card.get('prediction', ''))}</pre>
              <h4>Ground Truth</h4>
              <pre>{html.escape(card.get('ground_truths', ''))}</pre>
              <h4>Reasoning / Explanation</h4>
              <pre>{html.escape(card.get('reasoning', ''))}</pre>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Qwen3-VL-8B Baseline Briefing</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; color: #202124; }}
    h1 {{ font-size: 34px; margin-bottom: 6px; }}
    h2 {{ margin-top: 36px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
    .subtitle {{ color: #5f6368; margin-top: 0; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 12px; margin: 24px 0; }}
    .metric {{ border: 1px solid #ddd; border-radius: 8px; padding: 14px; background: #fafafa; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e6e6e6; padding: 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    .figure svg {{ max-width: 100%; height: auto; }}
    .cards {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; background: white; }}
    .card img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; margin: 8px 0; }}
    .meta {{ color: #5f6368; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f6f8fa; padding: 10px; border-radius: 6px; }}
    .note {{ color: #5f6368; }}
  </style>
</head>
<body>
  <h1>Qwen3-VL-8B Baseline 内部诊断汇报</h1>
  <p class="subtitle">Conservative deterministic scorer. Internal diagnostic, not official benchmark metric.</p>
  <div class="metrics">
    <div class="metric">Datasets<strong>{len(diagnostics)}</strong></div>
    <div class="metric">Examples<strong>{total_examples:,}</strong></div>
    <div class="metric">Scored / Coverage<strong>{scored_examples:,} / {coverage * 100:.1f}%</strong></div>
    <div class="metric">Parser-cond. Acc<strong>{weighted_acc * 100:.1f}%</strong></div>
    <div class="metric">Unparsed / Length<strong>{unparsed:,} / {length_rows:,}</strong></div>
  </div>

  <h2>结果趋势图</h2>
  <p class="note">柱子是 parser-conditional accuracy，黑点是 coverage。低覆盖数据集的 accuracy 只能作解析成功样本上的诊断。</p>
  <div class="figure">{svg_text}</div>

  <h2>数据集概览</h2>
  <table>
    <thead><tr><th>Dataset</th><th>Tier</th><th>Coverage</th><th>Accuracy</th><th>Unparsed</th><th>Length</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>

  <h2>适合汇报的内部趋势</h2>
  <ul>
    <li>高覆盖任务：{html.escape(", ".join(row["dataset"] for row in high_coverage))}</li>
    <li>低覆盖任务：{html.escape(", ".join(row["dataset"] for row in low_coverage))}</li>
    <li>MMVet 需要 judge，不进入 deterministic aggregate。</li>
  </ul>

  <h2>VLMEvalKit vs 当前 scorer</h2>
  <p>VLMEvalKit 是完整 VLM benchmark 评测框架，覆盖数据准备、推理、后处理、官方或社区 metric，以及 MMBench 等任务的 LLM-based answer extraction / CircularEval。当前 scorer 只做 post-hoc 保守解析，输出 coverage、unparsed 和 parser-conditional accuracy，因此适合内部审计，不适合作为 paper official metric。</p>

  <h2>样题与模型输出</h2>
  <p class="note">以下样例来自 sampled raw responses。若 image 字段为空，说明当前 raw JSONL 未暴露可识别图片路径；可用增强后的抽样脚本在 CPU 实例重跑。</p>
  <div class="cards">{"".join(cards_html)}</div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--samples-md", required=True)
    parser.add_argument("--figure-svg", required=True)
    parser.add_argument("--out-html", required=True)
    parser.add_argument("--sample-limit", type=int, default=8)
    args = parser.parse_args()

    html_text = render_html(
        diagnostics=load_diagnostics(args.diagnostics),
        sample_cards=parse_sample_markdown(args.samples_md, limit=args.sample_limit),
        svg_path=args.figure_svg,
    )
    Path(args.out_html).write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    main()


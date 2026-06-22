import csv
from pathlib import Path

from dual_track_opd.analysis.render_baseline_presentation import (
    load_diagnostics,
    parse_sample_markdown,
    render_html,
)


def test_render_html_briefing(tmp_path: Path):
    diagnostics = tmp_path / "diag.csv"
    with diagnostics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "scoring_type",
                "interpretation_tier",
                "n",
                "scored_n",
                "coverage",
                "accuracy",
                "unparsed_rows",
                "unparsed_rate",
                "length_rows",
                "length_rate",
                "needs_judge",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "GQA",
                "scoring_type": "normalized_exact",
                "interpretation_tier": "rough_exact_match_diagnostic",
                "n": "10",
                "scored_n": "10",
                "coverage": "1.0",
                "accuracy": "0.7",
                "unparsed_rows": "0",
                "unparsed_rate": "0",
                "length_rows": "0",
                "length_rate": "0",
                "needs_judge": "false",
            }
        )
    samples = tmp_path / "samples.md"
    samples.write_text(
        """# Samples

## GQA / 1

- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: `/tmp/image.png`

**Question**

What is this bird called?

**Prediction**

cockatoo

**Ground Truths**

["parrot"]
""",
        encoding="utf-8",
    )
    svg = tmp_path / "plot.svg"
    svg.write_text("<svg></svg>", encoding="utf-8")
    cards = parse_sample_markdown(samples, limit=1)
    html = render_html(load_diagnostics(diagnostics), cards, svg)
    assert "Qwen3-VL-8B Baseline" in html
    assert "cockatoo" in html
    assert "/tmp/image.png" in html

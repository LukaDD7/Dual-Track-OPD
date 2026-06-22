import csv
from pathlib import Path

from dual_track_opd.analysis.plot_baseline_diagnostics import load_rows, render_svg


def test_render_svg_contains_dataset_and_legend(tmp_path: Path):
    path = tmp_path / "diagnostics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
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
                "dataset": "MMBench",
                "scoring_type": "mcq",
                "interpretation_tier": "high_coverage_deterministic_mcq",
                "n": "10",
                "scored_n": "10",
                "coverage": "1.0",
                "accuracy": "0.9",
                "unparsed_rows": "0",
                "unparsed_rate": "0",
                "length_rows": "0",
                "length_rate": "0",
                "needs_judge": "false",
            }
        )
    svg = render_svg(load_rows(path), "Test Plot")
    assert "<svg" in svg
    assert "MMBench" in svg
    assert "Coverage marker" in svg


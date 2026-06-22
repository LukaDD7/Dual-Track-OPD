import csv
from pathlib import Path

from dual_track_opd.analysis.analyze_baseline_scores import analyze_scores, render_markdown


def test_analyze_scores_classifies_rows(tmp_path: Path):
    scores = tmp_path / "scores.csv"
    audit = tmp_path / "audit.csv"
    with scores.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "file",
                "scoring_type",
                "n",
                "scored_n",
                "correct",
                "accuracy",
                "errors",
                "length_rows",
                "unparsed_rows",
                "needs_judge",
                "parser_version",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "GoodMCQ",
                "file": "a.jsonl",
                "scoring_type": "mcq",
                "n": "10",
                "scored_n": "10",
                "correct": "8",
                "accuracy": "0.8",
                "errors": "0",
                "length_rows": "0",
                "unparsed_rows": "0",
                "needs_judge": "false",
                "parser_version": "test",
                "notes": "",
            }
        )
        writer.writerow(
            {
                "dataset": "LowCoverage",
                "file": "b.jsonl",
                "scoring_type": "numeric_exact",
                "n": "10",
                "scored_n": "1",
                "correct": "1",
                "accuracy": "1.0",
                "errors": "0",
                "length_rows": "2",
                "unparsed_rows": "7",
                "needs_judge": "false",
                "parser_version": "test",
                "notes": "",
            }
        )
        writer.writerow(
            {
                "dataset": "Judge",
                "file": "c.jsonl",
                "scoring_type": "needs_judge",
                "n": "5",
                "scored_n": "0",
                "correct": "0",
                "accuracy": "",
                "errors": "0",
                "length_rows": "0",
                "unparsed_rows": "0",
                "needs_judge": "true",
                "parser_version": "test",
                "notes": "",
            }
        )

    with audit.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "file",
                "row_id",
                "prediction",
                "parsed_prediction",
                "ground_truth",
                "parsed_ground_truth",
                "correct",
                "finish_reason",
                "error",
                "audit_reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "LowCoverage",
                "file": "b.jsonl",
                "row_id": "1",
                "prediction": "",
                "parsed_prediction": "",
                "ground_truth": "",
                "parsed_ground_truth": "",
                "correct": "",
                "finish_reason": "stop",
                "error": "",
                "audit_reason": "unparsed",
            }
        )

    analysis = analyze_scores(scores, audit)
    tiers = {row["dataset"]: row["interpretation_tier"] for row in analysis["scores"]}
    assert tiers["GoodMCQ"] == "high_coverage_deterministic_mcq"
    assert tiers["LowCoverage"] == "low_coverage_diagnostic"
    assert tiers["Judge"] == "needs_judge"
    assert "LowCoverage" in render_markdown(analysis)

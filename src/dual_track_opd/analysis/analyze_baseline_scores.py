"""Analyze conservative baseline scoring outputs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LOW_COVERAGE_THRESHOLD = 0.8
HIGH_COVERAGE_THRESHOLD = 0.95


def _to_int(value: str) -> int:
    return int(value) if value not in ("", None) else 0


def _to_float(value: str) -> float | None:
    return None if value in ("", None) else float(value)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def classify_score_row(row: dict[str, str]) -> str:
    """Classify how to interpret one dataset's deterministic score."""

    if row["needs_judge"].lower() == "true" or row["scoring_type"] == "needs_judge":
        return "needs_judge"

    coverage = _to_int(row["scored_n"]) / max(_to_int(row["n"]), 1)
    scoring_type = row["scoring_type"]
    if coverage < LOW_COVERAGE_THRESHOLD:
        return "low_coverage_diagnostic"
    if scoring_type in {"normalized_exact", "numeric_exact"}:
        return "rough_exact_match_diagnostic"
    if coverage >= HIGH_COVERAGE_THRESHOLD and scoring_type == "mcq":
        return "high_coverage_deterministic_mcq"
    return "deterministic_diagnostic"


def analyze_scores(
    scores_path: str | Path,
    audit_path: str | Path,
) -> dict[str, Any]:
    scores = load_csv(scores_path)
    audit = load_csv(audit_path)

    enriched: list[dict[str, Any]] = []
    by_tier: Counter[str] = Counter()
    scoring_types: Counter[str] = Counter()
    totals = Counter()

    for row in scores:
        n = _to_int(row["n"])
        scored_n = _to_int(row["scored_n"])
        correct = _to_int(row["correct"])
        errors = _to_int(row["errors"])
        length_rows = _to_int(row["length_rows"])
        unparsed_rows = _to_int(row["unparsed_rows"])
        accuracy = _to_float(row["accuracy"])
        coverage = scored_n / n if n else 0.0
        unparsed_rate = unparsed_rows / n if n else 0.0
        length_rate = length_rows / n if n else 0.0
        tier = classify_score_row(row)

        by_tier[tier] += 1
        scoring_types[row["scoring_type"]] += 1
        totals["n"] += n
        totals["scored_n"] += scored_n
        totals["correct"] += correct
        totals["errors"] += errors
        totals["length_rows"] += length_rows
        totals["unparsed_rows"] += unparsed_rows

        enriched.append(
            {
                **row,
                "n_int": n,
                "scored_n_int": scored_n,
                "correct_int": correct,
                "accuracy_float": accuracy,
                "coverage": coverage,
                "unparsed_rate": unparsed_rate,
                "length_rate": length_rate,
                "interpretation_tier": tier,
            }
        )

    audit_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in audit:
        audit_counts[row["dataset"]][row["audit_reason"]] += 1

    deterministic_rows = [
        row
        for row in enriched
        if row["interpretation_tier"] not in {"needs_judge", "low_coverage_diagnostic"}
    ]
    low_coverage_rows = [
        row for row in enriched if row["interpretation_tier"] == "low_coverage_diagnostic"
    ]
    needs_judge_rows = [row for row in enriched if row["interpretation_tier"] == "needs_judge"]

    weighted_accuracy = (
        totals["correct"] / totals["scored_n"] if totals["scored_n"] else None
    )
    macro_accuracy_values = [
        row["accuracy_float"] for row in deterministic_rows if row["accuracy_float"] is not None
    ]
    macro_accuracy = (
        sum(macro_accuracy_values) / len(macro_accuracy_values)
        if macro_accuracy_values
        else None
    )

    return {
        "scores": enriched,
        "audit_counts": audit_counts,
        "by_tier": by_tier,
        "scoring_types": scoring_types,
        "totals": totals,
        "weighted_accuracy": weighted_accuracy,
        "macro_accuracy": macro_accuracy,
        "deterministic_rows": deterministic_rows,
        "low_coverage_rows": low_coverage_rows,
        "needs_judge_rows": needs_judge_rows,
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    totals = analysis["totals"]
    lines = [
        "# Qwen3-VL-8B Baseline Diagnostic Analysis",
        "",
        "This report summarizes the conservative deterministic baseline scorer outputs. It is an internal diagnostic analysis, not an official benchmark table.",
        "",
        "## Overall",
        "",
        f"- Datasets: {len(analysis['scores'])}",
        f"- Total examples: {totals['n']}",
        f"- Scored examples: {totals['scored_n']} ({_pct(totals['scored_n'] / totals['n'] if totals['n'] else None)})",
        f"- Correct among scored examples: {totals['correct']}",
        f"- Parser-conditional weighted accuracy: {_pct(analysis['weighted_accuracy'])}",
        f"- Macro accuracy over non-low-coverage deterministic rows: {_pct(analysis['macro_accuracy'])}",
        f"- Unparsed rows: {totals['unparsed_rows']} ({_pct(totals['unparsed_rows'] / totals['n'] if totals['n'] else None)})",
        f"- Length rows: {totals['length_rows']} ({_pct(totals['length_rows'] / totals['n'] if totals['n'] else None)})",
        f"- Error rows: {totals['errors']}",
        "",
        "## Interpretation Tiers",
        "",
    ]

    for tier, count in sorted(analysis["by_tier"].items()):
        lines.append(f"- `{tier}`: {count}")

    lines.extend(
        [
            "",
            "## Dataset Table",
            "",
            "| Dataset | Type | Tier | n | scored_n | Coverage | Accuracy | Unparsed | Length |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["scores"]:
        lines.append(
            "| {dataset} | {scoring_type} | {tier} | {n} | {scored} | {coverage} | {accuracy} | {unparsed} | {length} |".format(
                dataset=row["dataset"],
                scoring_type=row["scoring_type"],
                tier=row["interpretation_tier"],
                n=row["n_int"],
                scored=row["scored_n_int"],
                coverage=_pct(row["coverage"]),
                accuracy=_pct(row["accuracy_float"]),
                unparsed=row["unparsed_rows"],
                length=row["length_rows"],
            )
        )

    lines.extend(["", "## Reliable Internal Diagnostics", ""])
    if analysis["deterministic_rows"]:
        for row in analysis["deterministic_rows"]:
            lines.append(
                f"- `{row['dataset']}`: {row['scoring_type']}, coverage {_pct(row['coverage'])}, parser-conditional accuracy {_pct(row['accuracy_float'])}."
            )
    else:
        lines.append("- None.")

    lines.extend(["", "## Low-Coverage Or Needs-Judge Results", ""])
    for row in analysis["low_coverage_rows"]:
        lines.append(
            f"- `{row['dataset']}`: coverage {_pct(row['coverage'])}; interpret only as a parser-conditional diagnostic."
        )
    for row in analysis["needs_judge_rows"]:
        lines.append(f"- `{row['dataset']}`: requires judge-based evaluation.")

    lines.extend(["", "## Audit Sample Counts", ""])
    for dataset in sorted(analysis["audit_counts"]):
        counts = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(analysis["audit_counts"][dataset].items())
        )
        lines.append(f"- `{dataset}`: {counts}")

    lines.extend(
        [
            "",
            "## Paper-Ready Implication",
            "",
            "Use this report to prioritize evaluator integration and failure analysis. Do not cite these numbers as official benchmark metrics. Paper-facing tables should use official or community-standard evaluators and document evaluator versions, prompts, judge models, and answer extraction settings.",
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "scoring_type": row["scoring_type"],
                    "interpretation_tier": row["interpretation_tier"],
                    "n": row["n"],
                    "scored_n": row["scored_n"],
                    "coverage": f"{row['coverage']:.6f}",
                    "accuracy": row["accuracy"],
                    "unparsed_rows": row["unparsed_rows"],
                    "unparsed_rate": f"{row['unparsed_rate']:.6f}",
                    "length_rows": row["length_rows"],
                    "length_rate": f"{row['length_rate']:.6f}",
                    "needs_judge": row["needs_judge"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="Dataset-level scores CSV.")
    parser.add_argument("--audit", required=True, help="Failure audit CSV.")
    parser.add_argument("--out-report", required=True, help="Output Markdown report.")
    parser.add_argument("--out-analysis-csv", help="Optional derived analysis CSV.")
    args = parser.parse_args()

    analysis = analyze_scores(args.scores, args.audit)
    Path(args.out_report).write_text(render_markdown(analysis), encoding="utf-8")
    if args.out_analysis_csv:
        write_analysis_csv(args.out_analysis_csv, analysis["scores"])


if __name__ == "__main__":
    main()

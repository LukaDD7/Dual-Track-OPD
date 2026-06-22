# Qwen3-VL-8B Baseline Metric Notes

This note describes how to interpret the first-pass scoring outputs:

- `reports/tables/qwen3vl8b_baseline_raw_summary.csv`
- `reports/tables/qwen3vl8b_baseline_scores.csv`
- `reports/tables/qwen3vl8b_baseline_failure_audit.csv`

The scorer is intentionally conservative and deterministic. It is useful for internal analysis, failure triage, and regression tracking, but it is not a substitute for official benchmark evaluation.

## Current Scorer

- CLI: `python -m dual_track_opd.eval.score_raw_responses`
- Parser version: `conservative_v2` after the MathVerse MCQ manifest fix.
- MCQ scoring: explicit option labels only.
- Short-answer scoring: normalized exact match.
- Numeric scoring: one explicit integer/decimal only.
- Judge tasks: marked `needs_judge` and excluded from deterministic scoring.

## Coverage Summary

| Dataset | Scoring type | n | scored_n | Coverage | Accuracy | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| BLINK | mcq | 1124 | 562 | 50.0% | 78.47% | Partial deterministic score; hidden labels and unparsed rows limit interpretation. |
| DynaMath_Sample | numeric_exact | 501 | 285 | 56.9% | 25.96% | Rough numeric exact signal only. |
| GQA | normalized_exact | 5000 | 5000 | 100.0% | 70.56% | Useful diagnostic; official GQA evaluator still required for paper reporting. |
| MMBench | mcq | 4329 | 4327 | 100.0% | 89.72% | Strong deterministic diagnostic; official VLMEvalKit/CircularEval still required for paper reporting. |
| MMMU_Pro_10 | mcq | 1730 | 309 | 17.9% | 30.10% | Low parser coverage; do not report as dataset accuracy. |
| MMMU_Pro_4 | mcq | 1730 | 657 | 38.0% | 52.21% | Low parser coverage; do not report as dataset accuracy. |
| MMSI-Bench | mcq | 1000 | 1000 | 100.0% | 32.40% | Deterministic diagnostic. |
| MMVet | needs_judge | 218 | 0 | 0.0% | n/a | Requires judge-based evaluation. |
| MV-MATH | numeric_exact | 2009 | 23 | 1.1% | 78.26% | Extremely low coverage; only confirms a tiny set of clearly numeric answers. |
| MathVerse | mcq | 3940 | pending rerun | pending rerun | pending rerun | Manifest fixed from numeric exact to MCQ option-letter exact; rerun scorer before using this number. |
| MathVista | normalized_exact | 1000 | 975 | 97.5% | 60.92% | Rough diagnostic; use official MathVista path for paper reporting. |
| MindCube-Bench | mcq | 21154 | 21154 | 100.0% | 33.46% | Deterministic diagnostic. |
| ReMI | normalized_exact | 2600 | 2457 | 94.5% | 23.77% | Rough normalized exact diagnostic. |
| ScienceQA-IMG | mcq | 2097 | 2096 | 100.0% | 89.79% | Strong deterministic diagnostic. |
| VQAv2 | normalized_exact | 5000 | 4996 | 99.9% | 72.22% | Diagnostic only; official VQA soft-agreement accuracy required for paper reporting. |
| ViewSpatial-Bench | mcq | 5712 | 5712 | 100.0% | 39.69% | Deterministic diagnostic. |

## Main Caveats

1. Low coverage means the reported accuracy is conditional on successful parsing.
2. VQAv2 official accuracy uses human-answer agreement, not one-answer exact match.
3. GQA official reporting includes additional diagnostic metrics beyond normalized exact match.
4. MMBench official evaluation uses a more capable choice extraction path and CircularEval.
5. MathVerse contains option-letter rows in the preferred raw run; `conservative_v2` scores those as MCQ, but official/community evaluation is still required for paper reporting.
6. MMVet requires judge-based evaluation.

## Paper-Ready Next Steps

1. Run official or community-standard evaluators for paper tables.
2. Preserve raw predictions outside Git and record their immutable paths.
3. Record evaluator commits, judge model details, prompts, and configs.
4. Use this deterministic table as an internal audit appendix or regression signal, not as leaderboard-comparable accuracy.

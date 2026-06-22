# Qwen3-VL-8B Baseline Diagnostic Analysis

This report summarizes the conservative deterministic baseline scorer outputs. It is an internal diagnostic analysis, not an official benchmark table.

## Overall

- Datasets: 16
- Total examples: 59144
- Scored examples: 50200 (84.9%)
- Correct among scored examples: 24907
- Parser-conditional weighted accuracy: 49.6%
- Macro accuracy over non-low-coverage deterministic rows: 56.9%
- Unparsed rows: 8233 (13.9%)
- Length rows: 493 (0.8%)
- Error rows: 0

## Interpretation Tiers

- `high_coverage_deterministic_mcq`: 5
- `low_coverage_diagnostic`: 6
- `needs_judge`: 1
- `rough_exact_match_diagnostic`: 4

## Dataset Table

| Dataset | Type | Tier | n | scored_n | Coverage | Accuracy | Unparsed | Length |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BLINK | mcq | low_coverage_diagnostic | 1124 | 562 | 50.0% | 78.5% | 562 | 0 |
| DynaMath_Sample | numeric_exact | low_coverage_diagnostic | 501 | 285 | 56.9% | 26.0% | 216 | 0 |
| GQA | normalized_exact | rough_exact_match_diagnostic | 5000 | 5000 | 100.0% | 70.6% | 0 | 0 |
| MMBench | mcq | high_coverage_deterministic_mcq | 4329 | 4327 | 100.0% | 89.7% | 0 | 2 |
| MMMU_Pro_10 | mcq | low_coverage_diagnostic | 1730 | 309 | 17.9% | 30.1% | 1416 | 5 |
| MMMU_Pro_4 | mcq | low_coverage_diagnostic | 1730 | 657 | 38.0% | 52.2% | 1065 | 8 |
| MMSI-Bench | mcq | high_coverage_deterministic_mcq | 1000 | 1000 | 100.0% | 32.4% | 0 | 0 |
| MMVet | needs_judge | needs_judge | 218 | 0 | 0.0% | n/a | 0 | 0 |
| MV-MATH | numeric_exact | low_coverage_diagnostic | 2009 | 23 | 1.1% | 78.3% | 1622 | 364 |
| MathVerse | numeric_exact | low_coverage_diagnostic | 3940 | 647 | 16.4% | 29.5% | 3179 | 114 |
| MathVista | normalized_exact | rough_exact_match_diagnostic | 1000 | 975 | 97.5% | 60.9% | 25 | 0 |
| MindCube-Bench | mcq | high_coverage_deterministic_mcq | 21154 | 21154 | 100.0% | 33.5% | 0 | 0 |
| ReMI | normalized_exact | rough_exact_match_diagnostic | 2600 | 2457 | 94.5% | 23.8% | 143 | 0 |
| ScienceQA-IMG | mcq | high_coverage_deterministic_mcq | 2097 | 2096 | 100.0% | 89.8% | 1 | 0 |
| VQAv2 | normalized_exact | rough_exact_match_diagnostic | 5000 | 4996 | 99.9% | 72.2% | 4 | 0 |
| ViewSpatial-Bench | mcq | high_coverage_deterministic_mcq | 5712 | 5712 | 100.0% | 39.7% | 0 | 0 |

## Reliable Internal Diagnostics

- `GQA`: normalized_exact, coverage 100.0%, parser-conditional accuracy 70.6%.
- `MMBench`: mcq, coverage 100.0%, parser-conditional accuracy 89.7%.
- `MMSI-Bench`: mcq, coverage 100.0%, parser-conditional accuracy 32.4%.
- `MathVista`: normalized_exact, coverage 97.5%, parser-conditional accuracy 60.9%.
- `MindCube-Bench`: mcq, coverage 100.0%, parser-conditional accuracy 33.5%.
- `ReMI`: normalized_exact, coverage 94.5%, parser-conditional accuracy 23.8%.
- `ScienceQA-IMG`: mcq, coverage 100.0%, parser-conditional accuracy 89.8%.
- `VQAv2`: normalized_exact, coverage 99.9%, parser-conditional accuracy 72.2%.
- `ViewSpatial-Bench`: mcq, coverage 100.0%, parser-conditional accuracy 39.7%.

## Low-Coverage Or Needs-Judge Results

- `BLINK`: coverage 50.0%; interpret only as a parser-conditional diagnostic.
- `DynaMath_Sample`: coverage 56.9%; interpret only as a parser-conditional diagnostic.
- `MMMU_Pro_10`: coverage 17.9%; interpret only as a parser-conditional diagnostic.
- `MMMU_Pro_4`: coverage 38.0%; interpret only as a parser-conditional diagnostic.
- `MV-MATH`: coverage 1.1%; interpret only as a parser-conditional diagnostic.
- `MathVerse`: coverage 16.4%; interpret only as a parser-conditional diagnostic.
- `MMVet`: requires judge-based evaluation.

## Audit Sample Counts

- `BLINK`: incorrect=29, unparsed=171
- `DynaMath_Sample`: incorrect=98, unparsed=102
- `GQA`: incorrect=200
- `MMBench`: incorrect=199, length=1
- `MMMU_Pro_10`: incorrect=22, length=2, unparsed=176
- `MMMU_Pro_4`: incorrect=57, length=2, unparsed=141
- `MMSI-Bench`: incorrect=200
- `MMVet`: needs_judge=200
- `MV-MATH`: unparsed=200
- `MathVerse`: unparsed=200
- `MathVista`: incorrect=190, unparsed=10
- `MindCube-Bench`: incorrect=200
- `ReMI`: incorrect=186, unparsed=14
- `ScienceQA-IMG`: incorrect=199, unparsed=1
- `VQAv2`: incorrect=200
- `ViewSpatial-Bench`: incorrect=200

## Paper-Ready Implication

Use this report to prioritize evaluator integration and failure analysis. Do not cite these numbers as official benchmark metrics. Paper-facing tables should use official or community-standard evaluators and document evaluator versions, prompts, judge models, and answer extraction settings.

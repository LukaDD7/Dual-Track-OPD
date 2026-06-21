# Evaluation Plan

This project separates official benchmark reporting from conservative internal diagnostics. The distinction is mandatory for paper tables and reproducibility.

## Metric Tiers

### Tier 1: Official Or Leaderboard-Comparable Metrics

Use the benchmark's official evaluator, or the evaluation toolkit explicitly recommended by the benchmark maintainers. Record evaluator repository, commit, configuration, judge model if any, prompt version, and raw prediction path.

Tier 1 metrics are the only metrics suitable for main paper benchmark tables.

### Tier 2: Community-Standard Reimplementation

Use a widely adopted evaluator such as VLMEvalKit or lmms-eval when it is the de facto benchmark path and the official code is unavailable, impractical, or explicitly replaced by the community standard. Record package version or commit and all post-processing settings.

Tier 2 metrics may appear in paper tables only when clearly labeled and justified.

### Tier 3: Internal Deterministic Diagnostics

Use small, auditable scripts in this repository for conservative failure analysis, parser coverage checks, and rough regression tracking. These metrics must not be described as official benchmark accuracy.

Current Tier 3 scorer:

- `dual_track_opd.eval.score_raw_responses`
- `parser_version=conservative_v1`
- outputs `coverage`, `unparsed_rows`, `length_rows`, and `needs_judge`

## Benchmark Notes

### GQA

GQA is a visual reasoning and compositional QA benchmark. Official reporting is not limited to a single normalized exact-match number: standard GQA evaluation includes overall accuracy and more diagnostic metrics such as consistency, validity, plausibility, distribution, and type-wise scores.

For paper reporting:

- Use official GQA evaluation where possible.
- Report at least official accuracy.
- Treat `conservative_v1` normalized exact match as an internal diagnostic only.

References:

- https://cs.stanford.edu/people/dorarad/gqa/evaluate.html
- https://openaccess.thecvf.com/content_CVPR_2019/papers/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.pdf

### VQAv2

VQAv2 official accuracy uses human-answer agreement over 10 annotations with answer normalization. A single normalized exact match against one stored answer is not the official metric.

For paper reporting:

- Use the official VQA evaluator or a faithful implementation of the VQA accuracy rule.
- Preserve all human reference answers needed by the evaluator.
- Treat `conservative_v1` normalized exact match as an internal diagnostic only.

References:

- https://visualqa.org/evaluation.html

### MMBench

MMBench is multiple-choice, but official evaluation handles free-form VLM outputs using an LLM-based choice extractor and CircularEval. Simple regex choice extraction is useful for auditing but is not leaderboard-comparable.

For paper reporting:

- Use VLMEvalKit or the current official MMBench evaluation path.
- Record whether CircularEval is enabled and which helper model extracts choices.
- Do not compare `conservative_v1` MCQ accuracy directly against official MMBench numbers.

References:

- https://github.com/open-compass/MMBench
- https://arxiv.org/html/2307.06281v4
- https://arxiv.org/html/2407.11691v2

### MathVerse

MathVerse includes both multiple-choice and free-form visual math problems. It is not purely numeric-output evaluation. The benchmark emphasizes judge-assisted chain-of-thought evaluation for fine-grained reasoning analysis.

For paper reporting:

- Use MathVerse's official path or lmms-eval integration where appropriate.
- Record judge model, prompt, and whether outcome accuracy or CoT step scoring is reported.
- Treat numeric exact match from `conservative_v1` as a low-coverage internal signal only.

References:

- https://github.com/ZrrSkywalker/MathVerse
- https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01270.pdf

### MathVista

MathVista evaluates mathematical reasoning in visual contexts across diverse task types. It should be reported with the official MathVista evaluation path or a community-standard implementation, not a generic normalized exact-match script.

For paper reporting:

- Use the official MathVista repository/evaluator or a documented community-standard path.
- Record any answer extraction, execution, or judge settings.
- Treat `conservative_v1` normalized exact match as rough internal tracking.

References:

- https://github.com/lupantech/MathVista
- https://mathvista.github.io/

### MMVet

MMVet is open-ended and requires judge-based evaluation. It should not be included in deterministic exact-match aggregates.

For paper reporting:

- Use the official or community-standard judge evaluation.
- Record judge model, prompt, sampling settings, and evaluator version.
- Keep `needs_judge=true` in deterministic diagnostic tables.

## Paper Table Requirements

Every paper-facing benchmark row must include:

- benchmark name and split
- metric name exactly as reported
- evaluator source and commit/version
- raw prediction path outside Git
- config path and repo commit
- model checkpoint path
- whether a judge model was used
- judge model name, endpoint/provider, prompt version, and temperature if applicable
- answer extraction strategy
- any excluded examples or failures

## Internal Diagnostic Reporting

Internal deterministic tables should include:

- `n`
- `scored_n`
- `coverage = scored_n / n`
- `correct`
- `accuracy = correct / scored_n`
- `errors`
- `length_rows`
- `unparsed_rows`
- `needs_judge`
- `parser_version`

When coverage is low, the accuracy should be interpreted as parser-conditional accuracy, not dataset accuracy.


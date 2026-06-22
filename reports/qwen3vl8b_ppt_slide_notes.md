# Qwen3-VL-8B Baseline PPT Notes

This note is PPT-oriented. It summarizes four representative benchmarks for discussion:
MathVista, MathVerse, MMVet, and ReMI.

## One-Slide Message

Qwen3-VL-8B already has usable short-answer behavior on some mixed visual reasoning tasks, but the failure modes are not a single "VLM cannot see" story. The current examples split into three signal types:

- Text-dominant or format-friendly tasks: the model can often answer directly when the visual evidence is simple and the target answer is short.
- Visual-grounding bottlenecks: failures often require reading values, geometry, charts, object attributes, or multi-image relations from the figure.
- Evaluation-protocol bottlenecks: some datasets need official extraction/judging; conservative exact match can under-report performance, especially when the dataset is actually multiple-choice or open-ended.

Research implication: before Dual-Track OPD, we should separate language reasoning signal from visual-evidence signal. Otherwise OPD may mostly distill teacher language priors while diluting the sparse tokens that actually depend on the image.

## Selected Benchmarks

| Dataset | Role in PPT | Current diagnostic result | Reliability | Main takeaway |
| --- | --- | ---: | --- | --- |
| MathVista | Mixed visual math and chart/scene QA | 594/975, 60.9%; coverage 97.5% | Rough but useful | Good for showing short-answer visual math. Failures expose numeric reasoning, chart reading, and visual attribute grounding. |
| MathVerse | Geometry/math diagrams, often MCQ | Pending rerun after MCQ scorer fix | Usable after rerun | Sampled rows are clean option letters. The manifest now scores MathVerse as MCQ exact instead of numeric exact. |
| MMVet | Open-ended VLM ability, judge-needed | 218 samples, deterministic score excluded | Needs judge | Good qualitative examples: brief answers can be semantically correct, so exact match is inappropriate. |
| ReMI | Multi-image reasoning, charts, rules, IQ-style tasks | 584/2457, 23.8%; coverage 94.5% | Rough but informative | More diagnostic for visual reasoning failures: multi-image alignment, graph reading, and relational reasoning are weak. |

## Representative Examples

### MathVista: Visual Math With Short Answers

Use one correct and one incorrect example.

- Incorrect physics/diagram example: `sampled_images/MathVista_1.png`
  - Question: spring compression from mass, velocity, and spring constant.
  - Prediction: `0.023`
  - Ground truth: `1.2`
  - Interpretation: model gives a plausible numeric answer, but likely misses unit conversion/rounding target or the expected output scale. This is a good example where "answer-only" evaluation cannot tell whether the failure is visual reading, formula selection, or arithmetic.

- Correct scene/counting example: `sampled_images/MathVista_1000.png`
  - Question: subtract object groups and count remaining objects.
  - Prediction: `9`
  - Ground truth: `9`
  - Interpretation: when the task has a direct visual grounding path and the answer format is simple, the baseline can work.

Slide conclusion: MathVista is useful as a first rough benchmark, but for research claims we need per-type breakdown: chart reading, geometry, counting, physical reasoning, OCR-like reading, and pure arithmetic.

### MathVerse: Geometry MCQ After Scorer Fix

Use this as the second math benchmark after rerunning the scorer on the server.

- Correct MCQ example: `sampled_images/MathVerse_1314.png`
  - Question: equilateral triangle with side expressions.
  - Prediction: `B`
  - Ground truth: `B`
  - Current scorer label after fix: `mcq`
  - Interpretation: the raw output is a clean option letter. This should be counted by the corrected scorer.

- Incorrect MCQ example: `sampled_images/MathVerse_2627.png`
  - Prediction: `D`
  - Ground truth: `F`
  - Interpretation: after MCQ scoring, residual errors can reveal real geometry/coordinate reasoning errors instead of parser artifacts.

Slide conclusion: MathVerse is now a good candidate for the second math slide, but only use the rerun MCQ result, not the old numeric-exact result.

### MMVet: Open-Ended Semantic Correctness

Use this to explain why VLMEvalKit or judge-based scoring matters.

- Semantically correct short answer: `sampled_images/MMVet_v1_72.png`
  - Question: name the dish.
  - Prediction: `Pad Thai`
  - Ground truth: `pad thai`
  - Interpretation: deterministic exact match could handle this simple case after normalization, but the dataset contains broader open-ended answers.

- Rationale answer: `sampled_images/MMVet_v1_145.jpg`
  - Question: what kind of school, and give rationale.
  - Prediction: private school with rationale based on uniform.
  - Ground truth: private school, because of formal suit/uniform.
  - Interpretation: semantically aligned, but not suitable for strict exact match. A judge or official rubric is required.

Slide conclusion: MMVet is valuable for qualitative capability demonstration and judge-based evaluation, not for the conservative deterministic aggregate.

### ReMI: Multi-Image and Relational Reasoning

Use one math/graph failure and one IQ-style example.

- Multi-image graph/math failure: `sampled_images/ReMI_0_0.png` and `sampled_images/ReMI_0_1.png`
  - Question: compute the right-limit from the graph of `g(x)`.
  - Prediction: `2`
  - Ground truth: `1`
  - Interpretation: likely visual value-reading or image-to-symbol grounding failure. This is exactly the kind of token-level visual evidence signal OPD should emphasize.

- IQ-style multi-image correct example: `sampled_images/ReMI_1733_0.png` plus option images `sampled_images/ReMI_1733_1.png` to `sampled_images/ReMI_1733_4.png`
  - Prediction: `D`
  - Ground truth: `D`
  - Interpretation: the model can solve some structured visual pattern tasks, so failure is not uniformly "no visual reasoning"; it depends on how cleanly visual structure maps to answer choice.

Slide conclusion: ReMI is a useful probe for "what visual signal should the student learn": compare teacher/student probabilities under full image, masked/degraded image, and text-only prompts.

## Failure Modes To Report

1. Evaluator mismatch
   - MathVerse exposed a first-pass scorer bug: sampled rows were MCQ option letters, while the manifest previously used numeric exact.
   - This has been fixed in code; rerun the server-side analysis before using MathVerse numbers in slides.

2. Answer-only prompt limits
   - Many prompts ask the model to return only the final answer.
   - Therefore "Reasoning / Explanation not found" is expected and does not prove the model lacked reasoning.
   - It does mean we cannot diagnose internal reasoning from this raw output alone.

3. Numeric exact-match brittleness
   - MathVista and ReMI use normalized exact match.
   - This is fine for conservative internal diagnostics, but it does not handle tolerance, equivalent forms, units, or multi-answer semantics.

4. Visual grounding failures
   - Examples include reading graph limits, extracting chart values, counting object subsets, interpreting geometry diagrams, and comparing multiple images.
   - These are the strongest candidates for visual-specific OPD signals.

5. Open-ended semantic scoring
   - MMVet examples can be correct in meaning while differing from the reference string.
   - Judge-based scoring or official evaluator is required.

## Research Direction For Next Week

Core research question:

What part of VLM benchmark improvement comes from better language-side reasoning, and what part comes from better use of visual evidence?

Operational hypothesis:

For many VLM tasks, the teacher's useful signal is sparse. Only a subset of answer tokens is truly image-dependent. Standard OPD may over-distill language priors and underweight the visual evidence tokens.

Two-week plan:

- Reproduce VA-OPD-style visual advantage: compare teacher confidence under original image versus degraded/masked image, and identify tokens whose probability drops without visual evidence.
- Reproduce pure-text OPD/VLM distillation baseline: test whether text-only teacher reasoning improves VLM performance even without extra visual supervision.
- Run ablations on the four selected datasets:
  - image-only/full prompt vs text-only prompt
  - full image vs masked/degraded image
  - answer-only prompt vs rationale prompt
  - official/VLMEvalKit-style scoring vs conservative internal scoring

Expected discussion point:

If pure-text OPD improves MathVista/MMVet while visual-advantage weighting improves ReMI/geometry/chart tasks, then Dual-Track OPD has a clean motivation: one track distills language reasoning, the other track selectively distills visual-grounded evidence.

# Evaluation results snapshot — 2026-09-12

This snapshot updates the persisted Project15 v1 offline-recovery results after
the final 4-GPU batch completed.

Raw outputs, caches, and model weights remain outside Git.

## Newly completed Project15 v1 results

| Arm | Benchmark | Score |
|---|---|---:|
| Base | VQAv2 | 0.8182 |
| PTD-PO r4 step390 | VQAv2 | 0.8157 |
| TailSFT | MathVista | 66.80 |
| TailSFT | MMVet | 56.7257 |

## MMVet judge retry fix

The first TailSFT MMVet attempt completed inference but stalled in
postprocessing. The upstream MMVet judge loop could retry indefinitely when
the judge response parsed as a float outside `[0, 1]`, because that branch did
not advance the retry state. This caused tens of thousands of judge calls with
no progress.

The runtime fix now:

- limits each sample to four judge attempts;
- extracts the first valid 0-1 score with a regex;
- records 0 only after the retry budget is exhausted;
- preserves the aggregate schema.

Patch:

`patches/lmms-eval/0001-mmvet-bound-judge-retry.patch`

Commit:

`3f66bcc fix(lmms-eval): bound MMVet judge retries`

The TailSFT MMVet result above was produced after this fix.

## Current Project15 v1 status

| Arm | Status |
|---|---|
| Base | 15/15 complete |
| PTD-PO r4 step390 | 15/15 complete |
| TailSFT | 14/15 complete; VQAv2 remains |

Remaining:

1. TailSFT VQAv2
2. PTD-PO CharXiv reasoning rescore
3. Final merged Project15 summary and v1/v2 comparison

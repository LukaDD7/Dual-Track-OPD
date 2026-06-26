# Geometry3K 4C Plan

Geometry3K is now the first clean-data FC-OPD target. Vision-OPD remains
infrastructure validation or red-box-contaminated ablation only.

## Sequence

1. Build Geometry3K evidence cache smoke, `LIMIT=8`.
2. Run Geometry3K 4C signal/offline builder smoke, `LIMIT=8` or `16`,
   `ROLLOUTS_PER_PROMPT=2` or `4`.
3. Validate raw teacher top-k scores for `full,degraded,free,task`.
4. Run real Qwen3-VL-4B 4C min-train smoke with `--condition-set 4c-clean`.
5. Only after these pass, consider larger Geometry3K runs.

## Condition Definitions

- `full`: clean original diagram + question; no red box, no crop, no answer.
- `degraded`: clean degraded diagram + question; default
  `lowres_10pct_nearest`.
- `free`: cached diagram caption + question; no image at forced-scoring time.
- `task`: cached question-conditioned diagram evidence + question; no image at
  forced-scoring time and no final answer.

ViRL39K is the next target, but the adapter remains a placeholder until the
actual data schema is inspected.

# FC-OPD 4C Evidence Generation

Clean-data 4C uses generated evidence caches. `free` and `task` are not
placeholders.

Condition evidence:

- `free`: image-only caption/evidence generated without question, choices, or
  answer.
- `task`: question-conditioned visual evidence generated from image + question
  + choices, with final answer and option letters forbidden.

Build a Geometry3K evidence cache smoke:

```bash
DATASET=/path/to/geometry3k.json \
DTOPD_OUTPUT_ROOT=/path/to/outputs \
LIMIT=8 \
scripts/hpc/build_fc_opd_4c_evidence_cache.sh
```

Rows include prompt hashes, image hash, generator metadata,
`no_gold_field_used=true`, answer/final-answer forbid flags, leakage warnings,
and local errors. The default template generator is for pipeline validation;
set `GENERATOR_MODEL_PATH` for local Qwen3-VL evidence generation. The teacher
forced-scoring service is unchanged.

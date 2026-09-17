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
DATASET=/path/to/Geometry3K_official/unzipped \
DTOPD_OUTPUT_ROOT=/path/to/outputs \
LIMIT=8 \
scripts/hpc/build_fc_opd_4c_evidence_cache.sh
```

`DATASET` may be a legacy Geometry3K JSON/JSONL file or the official
directory-style root. For official Geometry3K, the adapter recursively loads
sample directories containing `data.json`, optional `logic_form.json`, and a
same-directory `.png`/`.jpg`/`.jpeg` diagram.

Probe the dataset shape without generating evidence:

```bash
set -o pipefail
DATASET=/path/to/Geometry3K_official/unzipped \
DRY_RUN_INSPECT=1 \
scripts/hpc/build_fc_opd_4c_evidence_cache.sh 2>&1 | tee geometry3k_probe.log
```

Keep `set -o pipefail` when piping through `tee`; otherwise Python adapter
failures can be hidden by `tee` exiting successfully.

Rows include prompt hashes, image hash, generator metadata,
`no_gold_field_used=true`, answer/final-answer forbid flags, leakage warnings,
and local errors. The default template generator is for pipeline validation;
set `GENERATOR_MODEL_PATH` for local Qwen3-VL evidence generation. The teacher
forced-scoring service is unchanged.

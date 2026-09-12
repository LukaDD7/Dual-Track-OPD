# FC-OPD Condition Prompt Audit

Use the condition prompt dump before training to inspect exactly what the teacher
receives under each FC-OPD condition. This is a pre-training audit tool only: it
does not call the teacher service, does not modify `third_party/verl`, and does
not start training.

## Command

```bash
DATASET="$PROJECT_ROOT/third_party/Vision-OPD/data/train.parquet" \
SOURCE_DATASET=vision-opd-6k \
bash scripts/hpc/dump_fc_opd_condition_prompts.sh
```

Equivalent Python entrypoint:

```bash
python scripts/hpc/dump_fc_opd_condition_prompts.py \
  --dataset "$PROJECT_ROOT/third_party/Vision-OPD/data/train.parquet" \
  --dataset-type vision_opd_parquet \
  --source-dataset vision-opd-6k \
  --limit 3 \
  --conditions full,blur,free,task \
  --blur-sigma 2.0 \
  --task-evidence-mode none \
  --output "$DTOPD_OUTPUT_ROOT/fc_opd/condition_prompt_audit/vision_opd6k.md" \
  --include-images-as-paths
```

The tool writes both Markdown and JSONL. If `--output` ends in `.md`, the JSONL
uses the same stem with `.jsonl`.

Prompt dumps do not establish OPD rollout signal. They are a prompt/path audit
for condition construction.

## Task Evidence Modes

- `none`: default non-oracle mode. Task evidence says that no task-specific
  evidence is provided. It must not contain the answer. This is a
  non-informative placeholder / audit mode, not final 4C training evidence.
- `free_caption`: reuses the weak/free caption as task evidence.
- `question_conditioned_caption`: uses an existing `task_evidence` /
  `task_extraction` / `evidence` field if present, otherwise a question-only
  audit placeholder.
- `oracle_answer`: injects the gold answer. This is only for oracle or
  upper-bound audits and always emits oracle/leakage warnings.

The VStar16 protocol smoke used benchmark reference answers as task evidence
only to exercise the protocol. That is not a valid default training
construction.

For first real Vision-OPD training, prefer FC-OPD-2C (`full,blur`) unless and
until non-oracle free/task evidence generators are implemented and audited.

## Default 4C Image Policy

For Vision-OPD-6K, the default FC-OPD-4C prompt construction is:

- `full`: original/global image + question.
- `blur`: degraded original/global image + question.
- `free`: weak/free caption/evidence + question.
- `task`: task evidence text + question.

Crop or bbox images are preserved as metadata only. They are not silently mapped
to `task`. A crop condition should be added only as an explicit ablation, for
example a future `--condition-set full,blur,free,task,crop` or
`--use-crop-condition`, and should be labeled as a privileged visual condition.

## Output Fields

Each JSONL record includes:

- `sample_uid`, `source_dataset`, `source_index`;
- original image path and degraded image path;
- crop/bbox image path if present;
- question and answer/gold field if present;
- a warning that answer/gold is not allowed in non-oracle task evidence;
- rendered `full`, `blur`, `free`, and `task` condition prompts;
- condition metadata;
- leakage flags:
  - `task_evidence_contains_answer`
  - `task_evidence_contains_option_letter`
  - `task_evidence_contains_gold_text`
  - `oracle_mode_enabled`

## Recommended HPC Validation

```bash
bash scripts/hpc/run_fc_opd_vision_opd_adapter_smoke.sh
```

```bash
python scripts/hpc/dump_fc_opd_condition_prompts.py \
  --dataset "$PROJECT_ROOT/third_party/Vision-OPD/data/train.parquet" \
  --dataset-type vision_opd_parquet \
  --source-dataset vision-opd-6k \
  --limit 3 \
  --conditions full,blur,free,task \
  --blur-sigma 2.0 \
  --task-evidence-mode none \
  --materialize-degraded-images \
  --output "$DTOPD_OUTPUT_ROOT/fc_opd/prompt_audit/vision_opd6k_first3/prompts.md" \
  --include-images-as-paths
```

Expected: degraded images are created, question text is clean, no Python dict
repr appears in rendered prompts, leakage flags remain false, and bbox/crop
metadata is preserved but unused by default 4C.

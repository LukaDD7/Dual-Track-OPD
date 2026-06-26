# FC-OPD Dataset Selection And Signal Audit

Before starting FC-OPD training, run a small offline scoring audit over each
candidate training set. The goal is to check whether the dataset actually
produces useful condition signal under `full`, `blur`, `free`, and `task`, not to
train or tune the student.

Do not modify `third_party/verl` for this stage, and do not start actual
training from the audit outputs.

## Response Provenance Levels

There are three audit levels, and their summaries must not be mixed:

- `fixed_audit_response`: protocol/path/condition audit only. It uses a fixed
  non-gold XML response and can show whether prompts, paths, teacher scoring, and
  condition divergences are non-collapsed. It is not student-rollout evidence.
- `dataset_target`: diagnostic target scoring only. It may use preserved answer
  metadata and must not be called OPD or student rollout.
- `student_rollout`: formal OPD-compatible signal audit. It samples K student
  responses first, freezes those response token IDs, then teacher-force scores
  each rollout under the requested conditions.

Every dataset audit JSONL row records `response_source`, response hashes, token
counts, and a provenance note. Summaries report response-source counts, unique
response hash counts, response length distribution, and an
`all_responses_identical` warning when applicable.

## Candidate Order

Recommended decision order:

1. Vision-OPD-6K prepared train JSON / Parquet for the first fair objective
   comparison with the Vision-OPD baseline. This is the first recommended
   training dataset.
2. Geometry3K, if available, for a VA-OPD-style visual-math comparison.
3. ViRL39K, or mixed visual reasoning data, only after the audit confirms
   non-collapsed condition signal.

Optional later candidates should stay out of training mixtures until their
signal audit is done: MathVista / MathVerse-like training splits, MV-MATH,
ScienceQA-IMG, BLINK, ViewSpatial-Bench, MindCube, and MMMU_Pro.

## Leakage Rule

For real training, `task_evidence` must not simply contain the gold answer unless
the run is explicitly marked as an oracle or upper-bound run.

The previous VStar16 smoke used benchmark reference answers as task evidence
only as a protocol smoke. That construction is not valid default training-data
construction.

The audit emits `task_evidence_contains_answer` warnings when it can detect that
the resolved task evidence includes the answer string.

## Vision-OPD-6K Default Adapter Policy

The default FC-OPD-4C setup for Vision-OPD-6K is no-crop:

- `full`: original/global image + question.
- `blur`: degraded original/global image + question.
- `free`: weak/free caption/evidence + question.
- `task`: question-conditioned text evidence + question.

Crop or bbox images are preserved as metadata only by the adapter. They are not
silently mapped to `task`, because that would introduce a privileged visual
condition into the default comparison.

Crop/bbox should be used only in an explicit ablation, for example a future
`--use-crop-condition` or `--condition-set full,blur,free,task,crop`, and should
be marked as a privileged visual condition.

`task_evidence_mode=none` is safe for prompt/path audits but deliberately
non-informative. It should not be recommended as final 4C training evidence.
For first real Vision-OPD training, use FC-OPD-2C (`full,blur`) or wait for
audited non-oracle free/task evidence generators before running FC-OPD-4C.

## Audit Command

Start the teacher service separately, then run:

```bash
DATASET="$DTOPD_DATA_ROOT/vision_opd_6k/train.parquet" \
SOURCE_DATASET=vision_opd_6k \
DATASET_TYPE=vision_opd_parquet \
bash scripts/hpc/run_fc_opd_dataset_signal_audit.sh
```

Useful environment overrides:

```bash
LIMIT=128
TEACHER_URL=http://127.0.0.1:18080
TOKENIZER="hf:${DTOPD_MODEL_ROOT}/Qwen3-VL-4B-Instruct"
CONDITIONS=full,blur,free,task
BLUR_SIGMA=2.0
DEGRADED_DIR="$DTOPD_OUTPUT_ROOT/fc_opd/degraded_images/vision_opd_6k"
MATERIALIZE_DEGRADED_IMAGES=1
SKIP_EXISTING=1
DRY_RUN=1
```

For a schema-only dry run on a machine without model tokenizers, set
`TOKENIZER=byte`; real audit runs should use the same HF tokenizer hash expected
by the teacher service.

For the dedicated Vision-OPD adapter smoke, run:

```bash
PROJECT_ROOT="$PROJECT_ROOT" \
DATASET="$PROJECT_ROOT/third_party/Vision-OPD/data/train.parquet" \
bash scripts/hpc/run_fc_opd_vision_opd_adapter_smoke.sh
```

This writes normalized records, a 3-sample prompt dump, and an 8-sample dataset
audit dry-run without requiring a teacher service.

The first path-readiness dry-run should materialize degraded images:

```bash
python scripts/hpc/run_fc_opd_dataset_signal_audit.py \
  --dataset "$PROJECT_ROOT/third_party/Vision-OPD/data/train.parquet" \
  --dataset-type vision_opd_parquet \
  --source-dataset vision-opd-6k \
  --limit 8 \
  --conditions full,blur,free,task \
  --blur-sigma 2.0 \
  --task-evidence-mode none \
  --response-source fixed_audit_response \
  --materialize-degraded-images \
  --output-dir "$DTOPD_OUTPUT_ROOT/fc_opd/dataset_audit/vision_opd6k_dryrun_8" \
  --dry-run
```

Expected for the first Vision-OPD samples after materialization:
`image_missing_rate=0.0`, `degraded_image_missing_rate=0.0`, no leakage
warnings, clean question text, and bbox/crop metadata preserved if present.

For the first OPD-compatible rollout smoke, start with:

```bash
DATASET="$PROJECT_ROOT/third_party/Vision-OPD/data/train.parquet" \
LIMIT=4 \
ROLLOUTS_PER_PROMPT=2 \
CONDITIONS=full,blur \
bash scripts/hpc/run_fc_opd_student_rollout_signal_audit.sh
```

Then scale to `LIMIT=16` and `ROLLOUTS_PER_PROMPT=4` once the smoke passes.

The Python entrypoint exposes the same flags directly:

```bash
python scripts/hpc/run_fc_opd_dataset_signal_audit.py \
  --dataset "$DTOPD_DATA_ROOT/vision_opd_6k/train.parquet" \
  --dataset-type vision_opd_parquet \
  --source-dataset vision_opd_6k \
  --limit 128 \
  --teacher-url http://127.0.0.1:18080 \
  --tokenizer "hf:${DTOPD_MODEL_ROOT}/Qwen3-VL-4B-Instruct" \
  --conditions full,blur,free,task \
  --blur-sigma 2.0 \
  --output-dir "$DTOPD_OUTPUT_ROOT/fc_opd/dataset_signal_audit/vision_opd_6k"
```

## Supported Inputs

`--dataset-type` may be:

- `vision_opd_json`: JSON array or JSONL with Vision-OPD-style fields.
- `vision_opd_parquet`: Parquet read through pandas / pyarrow.
- `generic_jsonl`: one JSON object per line with flexible image and question
  field names.
- `auto`: `.parquet` -> Parquet, `.jsonl` -> generic JSONL, otherwise JSON.

Flexible field names include:

- question: `query`, `question`, `prompt`, `instruction`;
- image: `images`, `image`, `image_path`, `image_paths`;
- crop/bbox metadata: `bbox_images`, `bbox_image_path`, `crop_images`,
  `crop_image_path`;
- answer: `response`, `answer`, `label`, `target`;
- evidence: `task_evidence`, `task_extraction`, `evidence`;
- caption: `free_caption`, `caption`, `image_caption`.

## Outputs

The audit writes four files under `--output-dir`:

- `<source>_dataset_signal_audit.jsonl`
- `<source>_dataset_signal_audit_summary.json`
- `<source>_dataset_signal_audit_summary.tsv`
- `<source>_dataset_signal_audit_summary.md`

Each per-sample JSONL row includes:

- `sample_uid`, `source_dataset`, `source_index`;
- image path and degraded image path;
- question and answer availability flag;
- response length `T` and prompt length if known;
- response source, response text/token hashes, response tokens or token count,
  and response provenance note;
- tokenizer hash and teacher model ID;
- condition score availability;
- condition entropy means;
- condition signal arrays and mean / p50 / p90 summaries;
- optional pairwise KD-gradient cosine diagnostics;
- leakage warnings and local errors.

The summary includes:

- requested and scored sample counts;
- image and degraded-image missing rates;
- teacher error rate;
- mean response tokens;
- response source counts;
- unique response text/token hash counts;
- response length mean / p50 / p90 and top-10 counter;
- all-responses-identical and response provenance warnings;
- mean entropy per condition;
- mean `full` vs `blur` divergence;
- mean `task` vs `free` divergence;
- visual detail signal mean / p50 / p90;
- task extraction signal mean / p50 / p90;
- high visual and task signal token ratios;
- condition-collapse indicators;
- leakage warnings.

The optional gradient-cosine diagnostic compares condition-induced KD gradients
under a fixed synthetic student distribution:

- `cos(g_full, g_blur)`
- `cos(g_full, g_free)`
- `cos(g_full, g_task)`
- `cos(g_blur, g_task)`

This is labeled as a condition redundancy diagnostic. It is useful for detecting
condition redundancy or collapse, but it is not ideal-gradient alignment and
should not be interpreted as
`Align_c(u) = cos(g_c^KD(u), g_u^ideal(u))`.

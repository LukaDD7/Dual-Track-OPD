# FC-OPD Dataset Selection And Signal Audit

Before starting FC-OPD training, run a small offline scoring audit over each
candidate training set. The goal is to check whether the dataset actually
produces useful condition signal under the selected condition set, not to train
or tune the student.

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

Recommended decision order after the Vision-OPD red-box finding:

1. Geometry3K for the first clean-data VA-OPD-style visual-math comparison.
2. ViRL39K after its schema is inspected and an adapter is implemented.
3. Mixed visual reasoning data only after the audit confirms
   non-collapsed condition signal.

Vision-OPD-6K is no longer a main-data candidate unless clean no-red-box source
images are recovered. Current Vision-OPD outputs are infrastructure validation
or red-box-contaminated / localization-cued ablation only.

Optional later candidates should stay out of training mixtures until their
signal audit is done: MathVista / MathVerse-like training splits, MV-MATH,
ScienceQA-IMG, BLINK, ViewSpatial-Bench, MindCube, and MMMU_Pro.

## Leakage Rule

For real training, `task_evidence` must not contain the gold answer unless the
run is explicitly marked as an oracle or upper-bound run.

The previous VStar16 smoke used benchmark reference answers as task evidence
only as a protocol smoke. That construction is not valid default training-data
construction.

The audit emits `task_evidence_contains_answer` warnings when it can detect that
the resolved task evidence includes the answer string.

## Vision-OPD-6K Default Adapter Policy

Vision-OPD is frozen as main experimental data because red-box localization
cues are suspected or confirmed in local images. Run:

```bash
DATASET=/path/to/Vision-OPD/train.parquet \
DTOPD_OUTPUT_ROOT=/path/to/outputs \
scripts/hpc/audit_vision_opd_red_box_contamination.sh
```

The current Vision-OPD Gaussian-blur degraded image is generated from the loaded
full image. If the loaded full image contains a red box, the degraded image also
contains that red box.

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
Do not run Vision-OPD full 2C or full 4C as main experiments. Use
`--allow-red-box-contaminated-images` only for an explicit localization-cued
ablation; summaries must mark the run as not main evidence.

## Clean-Data Geometry3K 4C Path

The clean-data condition set is `4c_full_degraded_free_task`:

- `full`: clean original image + question.
- `degraded`: default `lowres_10pct_nearest` image + question.
- `free`: generated image-only evidence + question.
- `task`: generated question-conditioned visual evidence + question.

Build the evidence cache first:

```bash
DATASET=/path/to/geometry3k.json \
DTOPD_OUTPUT_ROOT=/path/to/outputs \
LIMIT=8 \
scripts/hpc/build_fc_opd_4c_evidence_cache.sh
```

Then build trainable 4C scores:

```bash
DATASET=/path/to/geometry3k.json \
EVIDENCE_CACHE=/path/to/evidence_cache.jsonl \
DTOPD_OUTPUT_ROOT=/path/to/outputs \
LIMIT=8 \
ROLLOUTS_PER_PROMPT=4 \
scripts/hpc/build_fc_opd_geometry3k_4c_offline_scores.sh
```

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
ROLLOUT_RESPONSE_FORMAT=fc_opd_structured \
bash scripts/hpc/run_fc_opd_student_rollout_signal_audit.sh
```

Then scale to `LIMIT=16` and `ROLLOUTS_PER_PROMPT=4` once the smoke passes.
Use `ROLLOUT_RESPONSE_FORMAT=answer_only` only as a mechanical generation and
teacher-scoring smoke. On Vision-OPD it usually produces 3-4 token answers,
which is too short for token-level FC-OPD training and often collapses
full-vs-blur signal.

For FC-OPD training, prefer `ROLLOUT_RESPONSE_FORMAT=fc_opd_structured`, which
asks the student to emit `<visual_evidence>`, `<reasoning>`, and `<answer>`
spans without injecting the gold answer. Inspect `duplicate_rollout_rate`,
`unique_response_per_prompt_mean`, and `short_response_rate`; `K=4` only helps
when same-prompt rollouts are not duplicates.

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

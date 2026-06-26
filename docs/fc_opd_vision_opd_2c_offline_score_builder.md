# Vision-OPD-6K 2C Offline Score Builder

This builder turns the validated structured student-rollout audit path into a
trainable FC-OPD offline score JSONL for Vision-OPD infrastructure validation.
It does not touch `third_party/verl`, and it does not start full training.

Important: Vision-OPD is currently red-box-contaminated / localization-cued in
the local data. Builder outputs are valid for infrastructure validation and
explicit contaminated ablations, not main clean-data experimental evidence.

## Scope

- Dataset: Vision-OPD-6K `train.parquet`
- Student rollout model: Qwen3-VL-4B-Instruct
- Teacher service: Qwen3-VL-32B-Instruct at `http://127.0.0.1:18080`
- Conditions: `full,blur`
- Rollout format: `fc_opd_structured`
- Default crop policy: no crop; original/global image is used
- Bbox/crop image paths are preserved only as metadata

Each output JSONL row is one prompt-rollout pair. The row includes the structured
student response, exact response token IDs, raw teacher top-k tensors under
`condition_scores.full` and `condition_scores.blur`, chunk spans, condition
signals, gradient-cosine diagnostics, tokenizer/teacher metadata, and image/bbox
provenance fields.

## Build Commands

Very small trainable smoke:

```bash
DATASET=/path/to/Vision-OPD-6K/train.parquet \
DTOPD_OUTPUT_ROOT=/path/to/outputs \
scripts/hpc/build_fc_opd_vision_opd_2c_offline_scores.sh limit4_k2
```

This writes under:

```text
$DTOPD_OUTPUT_ROOT/fc_opd/offline_scores/vision_opd6k_2c_structured_limit4_k2/
```

Next smoke:

```bash
DATASET=/path/to/Vision-OPD-6K/train.parquet \
DTOPD_OUTPUT_ROOT=/path/to/outputs \
scripts/hpc/build_fc_opd_vision_opd_2c_offline_scores.sh limit16_k4
```

This writes under:

```text
$DTOPD_OUTPUT_ROOT/fc_opd/offline_scores/vision_opd6k_2c_structured_limit16_k4/
```

## Validation

The builder validates the JSONL after writing. To validate an existing file:

```bash
DATASET=/path/to/Vision-OPD-6K/train.parquet \
DTOPD_OUTPUT_ROOT=/path/to/outputs \
OUTPUT_JSONL=/path/to/vision_opd6k_2c_offline_scores.jsonl \
SUMMARY_JSON=/path/to/vision_opd6k_2c_offline_scores_summary.json \
scripts/hpc/build_fc_opd_vision_opd_2c_offline_scores.sh validate
```

Validation checks that each row converts through the existing FC-OPD offline
record tensor schema, response length matches teacher score sequence length for
both conditions, tokenizer hashes match teacher metadata, conditions are exactly
`full,blur`, bbox/crop images are not used by default, and leakage warnings are
absent.

## Real-Student Min-Train Smoke

After building a trainable JSONL, run the existing real-student optimizer-step
path in explicit 2C mode:

```bash
DATASET=/path/to/Vision-OPD-6K/train.parquet \
DTOPD_OUTPUT_ROOT=/path/to/outputs \
scripts/hpc/build_fc_opd_vision_opd_2c_offline_scores.sh real-min-train
```

The smoke loads Qwen3-VL-4B-Instruct, reads 1-2 offline score rows, reconstructs
the prompt plus structured response, slices response logits, computes the FC-OPD
2C loss from full/blur teacher scores, runs `backward`, and steps a small
trainable parameter subset. It verifies finite loss, finite nonzero gradients,
and an actual parameter update.

## Important Fields

The summary JSON includes prompt/rollout counts, teacher and student generation
error rates, image/degraded-image missing rates, response length and diversity
diagnostics, full-vs-blur signal statistics, high visual signal ratio, gradient
cosines, tokenizer hash, teacher model ID, student model path, output path/file
size, and best-effort git commit/dirty status.

Full-ish Vision-OPD runs are guarded by default because red-box contamination is
suspected. Use `--allow-red-box-contaminated-images` only for an explicit
localization-cued ablation; the summary then records
`red_box_contaminated=true`, `localization_cued_ablation=true`, and
`not_main_experiment=true`.

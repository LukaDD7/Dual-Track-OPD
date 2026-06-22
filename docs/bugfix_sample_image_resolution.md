# Bugfix: sample_raw_examples Image Resolution (2026-06-22)

## Problem

`python -m dual_track_opd.eval.sample_raw_examples` produced markdown output where 12 of 16 datasets had empty `- image:` fields. Only BLINK and MMMU_Pro entries resolved because their `meta.file` happened to be an absolute filesystem path (picked up by `collect_image_refs`).

## Root Cause

The eval pipeline loads images from dataset parquets at inference time, encodes them, and passes them to the VLM. The raw JSONL responses only record `num_images` (a count), never the image paths or bytes. `collect_image_refs` recursively searches the raw JSONL dict for anything resembling a file path or URL, but for most datasets no such field exists.

Additionally, the original manifest (`qwen3vl8b_ppt_selected_manifest.jsonl`) only contained 5 of 16 datasets.

## Errors Encountered

### 1. `ModuleNotFoundError: No module named 'dual_track_opd'`

The `tools` conda env did not have the package installed. Ran `pip install -e ".[dev]"` in `dtopd-dev` env to resolve. The `dtopd-dev` env also lacked `pyarrow`, required for parquet reading.

```bash
pip install pyarrow   # in dtopd-dev env
```

### 2. OOM when building VQAv2 image index

VQAv2 has 383 parquet files (~1.2M images). An initial implementation loaded all image bytes into a dict in memory, which exceeded available RAM. Reverted to per-file linear search with early termination.

### 3. PyArrow `FileNotFoundError` for dataset parquets

Several datasets use non-standard parquet filenames (e.g., `testmini-00000-of-00001-725687bf7a18d64b.parquet`). The code already uses `glob("*.parquet")` so this resolved naturally, but hardcoded filenames in early testing failed.

## Key Fixes

### New `--dataset-root` CLI argument

Added to `sample_raw_examples.py` to enable dataset-aware image resolution:

```bash
python -m dual_track_opd.eval.sample_raw_examples \
  --raw-dir .../raw_responses \
  --manifest reports/qwen3vl8b_ppt_selected_manifest.jsonl \
  --dataset-root .../dataset \
  --samples-per-dataset 4 --strategy even \
  --out-jsonl ... --out-md ...
```

### Per-dataset image resolvers

Each dataset stores images differently. The fix adds dataset-specific resolvers:

| Dataset | Image source | Resolution method |
|---------|-------------|-------------------|
| BLINK | `meta.file` → parquet `image` column | Existing `collect_image_refs` |
| MMMU_Pro | `meta.file` → parquet `image` column | Existing `collect_image_refs` |
| MindCube-Bench | `meta.file` → JSONL `images` field (relative paths) | `_mindcow_resolve_image` |
| GQA | `meta.imageId` → `val_balanced_images/*.parquet` by `id` column, `image.bytes` | Indexed dict (cached) |
| MMBench | `sample_id` → parquet `index` column, `image` column (base64 str) | `_mm_bench_resolve_image` |
| MathVista | `sample_id` → parquet `pid` column, `decoded_image.bytes` | `_mathvista_resolve_image` |
| VQAv2 | `sample_id` → 383 parquets by `question_id`, `image.bytes` | Per-file search |
| MMVet | `sample_id` → parquet `question_id`, `image.bytes` | `_mmvet_resolve_image` |
| MathVerse | `sample_id` → parquet `sample_index`, `image.bytes` | `_mathverse_resolve_image` |
| DynaMath_Sample | `sample_id` → parquet `id`, `decoded_image` bytes or `image` path | `_dyna_math_resolve_image` |
| MMSI-Bench | `sample_id` → parquet `id`, `images` list of bytes | `_msi_resolve_image` |
| MV-MATH | `sample_id` → JSON `problem_id`, `input_image` relative paths | `_mv_math_resolve_image` |
| ReMI | `sample_id` → parquet row index, `image_1..6` dict bytes | `_remi_resolve_image` |
| ScienceQA-IMG | `sample_id` → parquet row index, `image.bytes` | `_sciqa_resolve_image` |
| ViewSpatial-Bench | `sample_id` → JSON `image_path` relative paths | `_viewspatial_resolve_image` |

### Image extraction

Resolved images are saved to `sampled_images/` (sibling of `dataset/`) with format `{Dataset}_{sample_id}.{ext}`. The markdown `image:` field points to these extracted files.

### Manifest expanded

Original manifest had 5 entries. Expanded to all 16 datasets with correct `scoring_type` from the diagnostic CSV.

## Verification

```bash
# All 16 datasets resolve images
python3 -c "..."  # → 16/16 datasets with 4/4 images each

# 50 image files extracted
ls sampled_images/ | wc -l  # → 50

# Tests pass
pytest -q  # → 20 passed
```

## Files Changed

- `src/dual_track_opd/eval/sample_raw_examples.py` — image resolution logic, `--dataset-root` arg
- `reports/qwen3vl8b_ppt_selected_manifest.jsonl` — expanded to 16 datasets
- `reports/qwen3vl8b_ppt_selected_examples.md` — regenerated with resolved image paths

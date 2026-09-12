# FC-OPD Clean-Data 4C Offline Score Builder

The clean-data 4C builder writes trainable offline score JSONL for:

```text
4c_full_degraded_free_task = full, degraded, free, task
```

Default degraded mode is `lowres_10pct_nearest`: downsample to 10% width/height
and upsample back to original size with nearest neighbor. `gaussian_blur_s2` is
available only as an ablation.

Build Geometry3K 4C scores after evidence cache generation:

```bash
DATASET=/path/to/Geometry3K_official/unzipped \
EVIDENCE_CACHE=/path/to/evidence_cache.jsonl \
DTOPD_OUTPUT_ROOT=/path/to/outputs \
LIMIT=8 \
ROLLOUTS_PER_PROMPT=4 \
scripts/hpc/build_fc_opd_geometry3k_4c_offline_scores.sh
```

Each row stores one structured student rollout, exact response token IDs, raw
teacher top-k scores for all four conditions, condition signals, gradient
cosines, chunk spans, evidence metadata, post-hoc outcome metadata, and no-op
alignment targets. Gold answers may appear only in post-hoc outcome metadata,
never in rollout prompts, evidence generation, or teacher condition inputs.

Validate an existing file:

```bash
python scripts/hpc/build_fc_opd_geometry3k_4c_offline_scores.py \
  --dataset /path/to/Geometry3K_official/unzipped \
  --evidence-cache /path/to/evidence_cache.jsonl \
  --output-jsonl /path/to/geometry3k_4c_offline_scores.jsonl \
  --summary-json /path/to/summary.json \
  --validate-only
```

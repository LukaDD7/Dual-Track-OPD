# Verifier-Gated Student-Deficit FC-OPD HPC Smoke

This note records the first-version Geometry3K 6C verifier-gated student-deficit
offline scoring path. It intentionally avoids full training and does not require
modifying `third_party/verl`.

## 1. Evidence Cache

```bash
export G3K_EVID32_DIR="$DTOPD_OUTPUT_ROOT/fc_opd/evidence_cache/geometry3k_6c_limit32_clean_vsd_v1"
rm -rf "$G3K_EVID32_DIR"
mkdir -p "$G3K_EVID32_DIR"

CUDA_VISIBLE_DEVICES=0 \
python scripts/hpc/build_fc_opd_4c_evidence_cache.py \
  --dataset "$GEOMETRY3K_DATASET" \
  --dataset-type geometry3k \
  --source-dataset geometry3k \
  --generator-model-path "$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct" \
  --condition-set 6c-solve \
  --include-prompts-in-output \
  --limit 32 \
  --seed 42 \
  --max-evidence-attempts 3 \
  --temperature 0.2 \
  --top-p 0.9 \
  --max-new-tokens 384 \
  --output-jsonl "$G3K_EVID32_DIR/geometry3k_6c_evidence_cache.jsonl" \
  --summary-json "$G3K_EVID32_DIR/geometry3k_6c_evidence_cache_summary.json"
```

## 2. Offline Scores

```bash
export G3K_VSD32_DIR="$DTOPD_OUTPUT_ROOT/fc_opd/offline_scores/geometry3k_6c_limit32_k2_vsd_v1"
rm -rf "$G3K_VSD32_DIR"
mkdir -p "$G3K_VSD32_DIR"

CUDA_VISIBLE_DEVICES=0 \
python scripts/hpc/build_fc_opd_geometry3k_4c_offline_scores.py \
  --dataset "$GEOMETRY3K_DATASET" \
  --dataset-type geometry3k \
  --source-dataset geometry3k \
  --evidence-cache "$G3K_EVID32_DIR/geometry3k_6c_evidence_cache.jsonl" \
  --student-model-path "hf:$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct" \
  --teacher-url http://127.0.0.1:18080 \
  --condition-set 6c-solve \
  --enable-student-condition-scoring \
  --student-deficit-gate \
  --verifier-gate geometry3k_verifier \
  --routing-mode student_deficit_chunk_gated \
  --grouped-loss-schema capability_chunk_v1 \
  --limit 32 \
  --rollouts-per-prompt 2 \
  --rollout-response-format fc_opd_structured_v2 \
  --temperature 0.5 \
  --top-p 0.9 \
  --max-new-tokens 768 \
  --seed 42 \
  --max-rollout-attempts 3 \
  --device cuda \
  --dtype bfloat16 \
  --degraded-mode lowres_50_bilinear_nearest \
  --output-jsonl "$G3K_VSD32_DIR/geometry3k_6c_vsd_offline_scores.jsonl" \
  --summary-json "$G3K_VSD32_DIR/geometry3k_6c_vsd_offline_scores_summary.json"
```

Expected summary fields:

- `student_condition_score_success_rate` near `1.0` for all 6C conditions
- finite `capability_deficit_means`
- nonzero `capability_final_weight_sums`
- populated `verifier_outcome_counts`
- nonzero `wrong_valid_rollout_opd_weight_sum` when wrong format-valid rollouts exist
- `grouped_loss_ready: true`
- `degraded_source_image_count: 0`

## 3. Real-Student Min-Train Smoke

```bash
CUDA_VISIBLE_DEVICES=1 \
SCORES="$G3K_VSD32_DIR/geometry3k_6c_vsd_offline_scores.jsonl" \
FC_OPD_STUDENT_MODEL="$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct" \
FC_OPD_CONDITION_SET=6c-solve \
FC_OPD_ROUTING_MODE=student_deficit_chunk_gated \
FC_OPD_EXPECT_CAPABILITY_SCORES=1 \
FC_OPD_REAL_STUDENT_LIMIT=32 \
FC_OPD_REAL_STUDENT_STEPS=1 \
bash scripts/hpc/run_fc_opd_geometry3k_4c_real_student_min_train_smoke.sh
```

Acceptance checks:

- `passed=true`
- `all_loss_finite=true`
- `all_grads_finite=true`
- `every_step_updates_params=true`
- `consumed_capabilities` is non-empty
- `capability_weight_sums` is not all zero

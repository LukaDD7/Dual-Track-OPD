# Support-Aware SFT-RL-OPD — Step 1 Diagnostic Status

**Date**: 2026-08-01
**Author**: LukaDD7
**Handoff to**: codex

---

## Overview

Step 1 of the Support-Aware OPD runbook: frozen-policy diagnostic on Geometry3K.
Goal: measure whether the 32B teacher's forced log-prob scores can rank 4B student
rollouts within a prompt, especially for "correct-tail" prompts where the student
occasionally gets the right answer but not greedily.

## Current State

### Completed: Initial full diagnostic run (512 tokens, flawed)

- **Run ID**: `diag_full_20260801_013648`
- **Config**: `configs/experiment/support_aware_geometry3k_pilot.yaml`
- **All 7/7 acceptance gates PASSED**
- **Summary**: `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/support_aware_opd/diag_full_20260801_013648/summary.json`

Key findings:
- 128 prompts × 9 rollouts (1 greedy + 8 stochastic, T=0.7)
- Greedy accuracy: 22.7% (Qwen3-VL-4B on Geometry3K)
- Support states: 35 exposed / 22 correct_tail / 69 no_correct / 2 other
- teacher_gap AUC = 0.665, teacher_mean_logp AUC = 0.763
- Correct-tail Rank@1 = 0.409 (vs random 0.125 baseline)
- Correct-tail MRR = 0.592

**Critical flaw discovered**: `max_new_tokens=512` truncates **71.4%** of responses.
Non-truncated accuracy is 71.8%, truncated accuracy is 3.6% — the model is much
more capable than the summary suggests. The current AUC/Rank metrics are
confounded by truncation (short→correct, long→truncated→wrong).

### Pending: Re-run with 2048 tokens + batch scoring

- `max_new_tokens` increased to **2048** in config
- Teacher scoring now uses **batch HTTP** (all 9 rollouts per prompt in 1 call)
  via `score_teacher_conditions_multi_sample`
- Student scoring now uses **batch forward pass** (padded concatenation)
- **Must re-run before proceeding to Step 2**

### Fixed: batch student scoring crashed on the first prompt (2026-08-01)

The 2026-08-01 10:55 re-run died on prompt 1/128 inside
`StudentScorer.score_batch` with:

```
ValueError: Image features and image tokens do not match, tokens: 630, features: 70
```

Root cause: `score_batch` padded `n` (prompt+response) rows into one
`input_ids` tensor but re-used the **single-image** tensors from the prompt-only
encoding (`pixel_values`, `image_grid_thw`, `mm_token_type_ids`).  Qwen3-VL
counts image-pad tokens across the whole batch (n × 70 = 630) and compares them
with the one image's vision features (70).  The unpadded `mm_token_type_ids`
would also have broken M-RoPE next.

Fix: encode all rows in **one** processor call
(`processor(text=full_texts, images=[image] * n, padding=True)`), so every row
carries its own per-image features/grid and a padded `mm_token_type_ids`.
Verified on CPU that batched rows tokenize identically to the per-item
encodings.  Regression tests in `tests/test_support_scorer_batch.py`
(CPU-only, no LLM weights).

## Architecture

```
src/dual_track_opd/support_aware/
├── __init__.py
├── diagnostic.py       # Main pipeline (~1100 lines)
├── reporter.py         # Output writers (JSONL, summary, manifest)
├── scorer.py           # TeacherScorer + StudentScorer (+ batch variants)
├── support_state.py    # SupportState enum + classify_support_state()
└── verifier.py         # Geometry3K answer extraction + verification
```

### Key classes

| Class | Role |
|-------|------|
| `DiagnosticConfig` | YAML-driven config with env var expansion |
| `TeacherScorer` | Wraps `TeacherClient` from FC-OPD; `score()` + `score_batch()` |
| `StudentScorer` | Loads Qwen3-VL-4B locally; `score()` + `score_batch()` |
| `SupportState` | `exposed`, `correct_tail`, `no_correct_observed`, `other` |

### Data flow

```
Geometry3K parquet → 128 prompts (SHA256-sorted deterministic selection)
  → Generate (4B student): 1 greedy + 8 stochastic (T=0.7)
  → Batch teacher score (32B, HTTP)
  → Batch student score (4B, local forward)
  → teacher_gap = teacher_mean_logp - student_mean_logp
  → Support state classification
  → Gate checks → summary.json
```

### Scoring details

- **teacher_gap** = teacher_mean_logp − student_mean_logp
  - Positive: teacher assigns higher probability than student
  - Negative: student is more confident (overconfident)
- **teacher_mean_logp**: teacher's forced log-prob, averaged across response tokens
- **student_mean_logp**: student's forced log-prob, same computation
- Both use **exact token alignment** — same tokenizer (Qwen2.5, verified hash match)

### Support state classification

```
greedy_correct=True                        → exposed
greedy_correct=False, correct_count ≥ K/2 → exposed
greedy_correct=False, 0 < correct < K/2   → correct_tail
greedy_correct=False, correct_count = 0   → no_correct_observed
greedy_correct=None                       → other
```

## GPU Setup

```
GPU 0: Teacher service (Qwen3-VL-32B-Instruct, ~64GB)
GPU 1-7: Student model (Qwen3-VL-4B-Instruct, device_map="auto")
Conda env: vision-opd-cu128 (Python 3.12, torch 2.10.0+cu128)
```

### Teacher startup
```bash
conda activate vision-opd-cu128
export FC_OPD_TEACHER_MODEL=${DTOPD_MODEL_ROOT}/Qwen3-VL-32B-Instruct
CUDA_VISIBLE_DEVICES=0 bash scripts/hpc/start_fc_teacher.sh
```

### Diagnostic run
```bash
conda activate vision-opd-cu128
bash scripts/hpc/run_support_aware_diagnostic.sh \
    --config configs/experiment/support_aware_geometry3k_pilot.yaml \
    --mode full
```

Or with nohup:
```bash
nohup bash scripts/hpc/run_support_aware_diagnostic.sh \
    --config configs/experiment/support_aware_geometry3k_pilot.yaml \
    --mode full \
    > /inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/support_aware_opd/diag_full_nohup.log 2>&1 &
```

### Important: editable install required
```bash
pip install --no-deps -e .
```
The `vision-opd-cu128` env may not have `dual-track-opd` installed. Always verify:
```bash
python -c "import dual_track_opd; print(dual_track_opd.__file__)"
```

## Incremental Saving

The pipeline now saves incrementally:
- `rollouts.jsonl` — appended after each rollout is scored
- `prompt_support_summary.jsonl` — appended after each prompt
- `_partial_summary.json` — written after each prompt (progress + error tail)
- Final files are overwritten by reporter functions when the run completes

If the GPU instance is reclaimed mid-run, partial data is on disk.

### Resuming a partial run

The pipeline now supports `--resume RUN_ID` (also passable through
`run_support_aware_diagnostic.sh --resume RUN_ID`).  On resume it reloads
`rollouts.jsonl` / `prompt_support_summary.jsonl` from the existing run dir,
skips prompts that already have a completed summary, and continues appending
to the same files.  The prompt selection is deterministic (SHA256(sample_uid)),
so the same config regenerates only the missing prompts with identical seeds.

```bash
# After the old process is dead, start the teacher again, then:
bash scripts/hpc/run_support_aware_diagnostic.sh \
    --config configs/experiment/support_aware_geometry3k_pilot.yaml \
    --mode full \
    --resume diag_full_20260801_123045
```

Do not start a resumed run while the original process is still alive — both
would append to the same files.

## Run Output Directory

```
{DTOPD_OUTPUT_ROOT}/support_aware_opd/diag_{mode}_{timestamp}/
├── rollouts.jsonl                  # 1152 lines (128 × 9)
├── prompt_support_summary.jsonl    # 128 lines
├── selected_prompts.jsonl          # Prompt metadata
├── summary.json                    # Aggregate metrics + gates
├── run_manifest.json               # Git info, timing
├── resolved_config.yaml            # Fully resolved config
├── _partial_summary.json           # Present during run, removed on completion
└── logs/                           # Empty (reserved)
```

## Analysis / Gate Framework

After the run completes, read `summary.json`:

### Threshold gates (must all pass)
- `tokenizer_alignment` — tokenizer hash must match
- `zero_missing_images` — all images loaded
- `finite_scores` — no NaN scores
- `malformed_rate_ok` — < 10% malformed answers
- `duplicate_rate_ok` — < 25% duplicate rollouts

### Signal gates (indicate Step 2 viability)
- `sufficient_correct_tails` — ≥ 20 correct_tail prompts
- `teacher_gap_auc` — AUC > 0.60 for correct/wrong discrimination
- `correct_tail_rank1_above_random` — Rank@1 > 1/K (0.125 for K=8)

### Key diagnostic metrics
- **correct_tail Rank@1**: teacher_gap ranking accuracy within a prompt
- **correct_tail MRR**: mean reciprocal rank of correct responses
- **AUC comparison**: teacher_gap vs teacher_mean_logp vs student_mean_logp
- **Response length percentiles**: check truncation
- **teacher_gap distribution by support state**: exposed vs correct_tail vs no_correct

## Response Truncation Analysis

The 512-token run revealed severe truncation:

| | Truncated (≥512) | Non-truncated |
|---|---|---|
| Fraction | 71.4% | 28.6% |
| Accuracy | 3.6% | 71.8% |

All non-truncated responses end with "Answer: X" format, confirming natural completion.
Community standard for Geometry3K: 2048-4096 tokens (evaluation), 512-2048 (training).

**Recommendation**: Use 2048 for the diagnostic re-run. If truncation remains >15%,
increase to 4096. Check `_partial_summary.json` during the run to monitor.

## Known Issues Fixed (this session)

1. **`CUDA_VISIBLE_DEVICES` unbound variable** in run script (`set -u` + optional var)
2. **No incremental saving** — all data was lost on mid-run kill
3. **`non_finite_count` never incremented** — Python int immutability bug
4. **Per-rollout `run_id` generated fresh** — each rollout had different run_id
5. **Teacher scoring serial** — 9 HTTP calls per prompt → now 1 batch call
6. **Student scoring serial** — 9 forward passes per prompt → now 1 batch pass
7. **`max_new_tokens=512`** — changed to 2048
8. **Batch student scoring crashed ("Image features and image tokens do not
   match")** — padded multi-row `input_ids` with single-image tensors; fixed by
   one batched processor call so each row has its own image tensors + padded
   `mm_token_type_ids`

## 2026-08-02: 2048-token full run finished, gate FAILED, measurement invalid

`diag_full_20260801_163119` (commit 5fd8f1a) completed 128/128 with two defects:

1. **Teacher batch scoring silently dropped the exact student prompt.** The
   batch path built 3-tuple samples with `prompt=None`, so the teacher backend
   fell back to `render_teacher_prompt()` (`"Question:\n{question}"`), while the
   student actually saw the `_PROMPT_TEMPLATE` with the step-by-step instruction.
   `prompt_texts` was accepted by `TeacherScorer.score_batch` but never used.
   Result: for 331 identical response texts shared with the 512-token run,
   `teacher_mean_logp` differed by mean abs 0.18 (corr 0.86) while
   `student_mean_logp` differed by 0.0012 (corr 0.999).  The batch teacher
   scores were also length-invariant (corr with response length ≈ 0.01), so
   `teacher_gap` became dominated by the student's length penalty, inverting
   `teacher_gap` AUC to 0.309 (vs 0.665 on the per-item 512 run).
   Cross-check: `diag_full_20260801_123045` (same batch path, older instance)
   reproduced 163119's scores bit-for-bit (corr 1.0), so the batch path is
   deterministic — it was just conditioning the teacher on the wrong prompt.
   Fixed: `TeacherScorer.score_batch` now forwards the exact rollout messages
   (image + `prompt_texts[i]`), matching the per-item `score()` path.  Regression
   test: `test_teacher_score_batch_preserves_exact_student_prompt`.

2. **Truncation at 2048 tokens is still 44%** (507/1152 rollouts), with 5.1%
   accuracy on truncated vs 69.5% on non-truncated responses — the same confound
   as the 512-token run, just weaker.  Per the runbook threshold (>15% → bump),
   `max_new_tokens` is now **4096** in
   `configs/experiment/support_aware_geometry3k_pilot.yaml`.

Because both defects invalidate the pooled AUC/rank metrics, the 163119
`teacher_gap_auc` gate failure is **not** evidence that the Step-1 signal is
absent.  A fresh full run (new run id — do NOT `--resume` 163119, its teacher
scores are invalid) is required before a Step 2 go/no-go decision.

## Step 2 Prerequisites

Before starting Step 2 (50-step micro-training pilot):
1. ✅ Step 1 code complete and debugged
2. ⏳ Re-run full diagnostic with 2048 tokens (must be done on GPU instance)
3. ⏳ Verify signal gates pass with the new token limit
4. ⏳ Analyze clean (non-truncated) teacher_gap distributions

## Tests

```bash
pytest tests/test_support_diagnostic.py -q
# 41 tests covering: extract_answer, normalize_gold_answer, verify_answer,
# support_state classification boundaries
```

## Relevant Files for Step 2

- `configs/experiment/support_aware_geometry3k_pilot.yaml` — experiment config
- `src/dual_track_opd/support_aware/` — all Step 1 modules
- `scripts/hpc/run_support_aware_diagnostic.sh` — HPC runner
- `scripts/hpc/start_fc_teacher.sh` — teacher service launcher

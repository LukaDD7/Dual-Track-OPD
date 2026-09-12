# Support-aware exact-token diagnostic: implemented delivery (2026-08-03)

## Outcome and decision

The correctness fix described in
`docs/support_aware_diagnostic_fix_spec_20260803.md` is implemented on branch
`codex/va-opd`.

The correct treatment of `skip_special_tokens=True` is **not** to turn it off
everywhere. Keep it only for a human-readable/verifier view, while preserving
the raw IDs returned by `model.generate()` as the sole scoring action sequence:

```text
raw generation IDs ───────────────┬── student forced scoring
                                  ├── teacher forced scoring
                                  └── token hash / exact-ID assertions

raw IDs --decode(skip_special_tokens=True)--> display/verifier text only
raw IDs --decode(skip_special_tokens=False)-> audit text only
```

This is required because decode followed by encode is not an identity:
decoding can remove EOS/special tokens, normalize spaces, and change BPE
boundary behavior. Re-tokenizing the display string therefore changes the
event whose teacher/student likelihood is being compared. A 4B and a 32B model
may still be token-aligned if their tokenizer fingerprints are identical;
different parameter counts are not a reason to repair or translate IDs.

If tokenizer fingerprints differ, token-level OPD must fail before scoring.
A future heterogeneous-tokenizer experiment must be separately named and use a
sequence-level objective; it cannot be reported as token-level OPD.

## Exact implementation locations

Line numbers below are against this delivery commit; symbol names are the
stable locator after later edits.

| Concern | Implemented location | Behavior |
| --- | --- | --- |
| Online FC-OPD cross-prompt leakage and right-padding window | `src/dual_track_opd/fc_opd/student_scorer.py:123` (`_score_batched`), `:198` (`_score_same_prompt_chunk`) | Groups by canonical rendered messages + image paths; batches only identical contexts; restores original order; appends raw IDs; slices logits from `prompt_width - 1`; drops stale prompt-only `position_ids` |
| Support diagnostic student exact-ID scoring | `src/dual_track_opd/support_aware/scorer.py:364` (`score`), `:387` (`score_batch`) | Display text is metadata only; prompt/image are encoded once, raw response IDs are appended and right-padded, masks are extended, and original IDs are gathered |
| No second 4B load | `src/dual_track_opd/support_aware/scorer.py:286` and `src/dual_track_opd/support_aware/diagnostic.py:1280` | `StudentScorer` accepts the already loaded generation model/processor |
| Teacher tokenizer preflight and exact response IDs | `src/dual_track_opd/support_aware/scorer.py:31-47`, `src/dual_track_opd/fc_opd/teacher_client.py:65-88` | Student fingerprint is passed into `TeacherClient`; any teacher-returned ID difference raises `TeacherServiceError` |
| Lossless generation representation | `src/dual_track_opd/support_aware/diagnostic.py:279` (`GenerationRecord`), `:304` (`generation_record_from_token_ids`), `:361` (`generate_response`) | Stores raw IDs/hash, display/raw decodes, prompt-token hash, finish reason, terminal ID, and content mask |
| Hard generated/student/teacher identity | `src/dual_track_opd/support_aware/diagnostic.py:457` and `:486` | Requires exact IDs, hashes, masks, and token counts before a gap is computed |
| EOS/truncation policy | `src/dual_track_opd/support_aware/diagnostic.py:304-359` | Primary score excludes only an observed final configured EOS/stop token; all-token and terminal scores remain recorded; length truncation is explicit |
| Canonical conservative verifier | `src/dual_track_opd/support_aware/verifier.py:38` | Requires explicit `Answer`/`final answer`, XML answer, or boxed marker; no last-number fallback |
| Prompt-level statistics and gates | `src/dual_track_opd/support_aware/diagnostic.py:713`, `:1904`, `:2042` | Macro within-prompt pairwise AUC, tie credit, prompt bootstrap CI, `c_i/K` Rank@1 baseline, exact random MRR, finish/length strata, exact-ID/prompt-hash/completeness/truncation gates |
| Strict merge | `src/dual_track_opd/support_aware/diagnostic.py:909` | Rejects missing/duplicate/overlapping/wrong-order cohorts, interval gaps, config/model/tokenizer/git/dataset/selection/prompt/scoring-policy mismatches; writes through a temporary directory then atomically renames |
| Old-run immutable migration | `src/dual_track_opd/support_aware/rescore.py:204` | Validates source cohort and raw hashes, always recomputes both scorers and verifier from raw IDs, records provenance, verifies source hashes before publish, writes only a new directory |
| Resume safety | `src/dual_track_opd/support_aware/diagnostic.py:681` | Refuses to mix historical text-retokenized rows with new exact-ID rows |
| Frozen legacy prompt for current cohort | `configs/experiment/support_aware_geometry3k_pilot.yaml:43` | Existing 4096-token cohort remains `legacy_answer`; structured-v2 requires a new run ID |

## What Claude Code should do on the server

### 1. Pull and verify the commit

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
git fetch origin
git switch codex/va-opd
git pull --ff-only origin codex/va-opd

python -m py_compile \
  src/dual_track_opd/fc_opd/student_scorer.py \
  src/dual_track_opd/fc_opd/teacher_client.py \
  src/dual_track_opd/support_aware/scorer.py \
  src/dual_track_opd/support_aware/diagnostic.py \
  src/dual_track_opd/support_aware/rescore.py

pytest -q \
  tests/test_support_diagnostic.py \
  tests/test_support_scorer_batch.py \
  tests/test_support_diagnostic_stats.py \
  tests/test_support_diagnostic_rescore.py \
  tests/test_support_diagnostic_resume.py \
  tests/test_support_diagnostic_shards.py \
  tests/fc_opd/test_student_scorer_batch_alignment.py \
  tests/fc_opd/test_teacher_service.py
```

Expected targeted result from the delivery machine: `68 passed, 3 skipped`.

### 2. Do not resume old scored rows into the fixed pipeline

The old 4096-token shard directories contain a valuable raw rollout cohort but
their student scores were produced from re-tokenized display text. Do not use
`--resume` to mix them with fixed rows. Do not regenerate the already complete
cohort merely because display decode removed special tokens.

Instead, run append-only rescoring for every completed shard. The source and
output must be different directories, and the output must not already exist:

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export DTOPD_MODEL_ROOT="$DTOPD_ROOT/models"
export DTOPD_OUTPUT_ROOT="$DTOPD_ROOT/fc-opd-storage/outputs"
export PROJECT="$DTOPD_ROOT/projects/Dual-Track-OPD"

cd "$PROJECT"

# Keep the existing 32B teacher service running on its assigned GPU/port.
# Put the 4B rescorer on the student GPU via CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES=1 python -m dual_track_opd.support_aware.rescore \
  --source-run "$DTOPD_OUTPUT_ROOT/support_aware_opd/<OLD_SHARD_RUN_ID>" \
  --output-dir "$DTOPD_OUTPUT_ROOT/support_aware_opd/<OLD_SHARD_RUN_ID>.rescored-exact-v1" \
  --student-model "$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct" \
  --teacher-url http://127.0.0.1:18080 \
  --device cuda \
  --dtype bfloat16 \
  --bootstrap-seed 42 \
  --bootstrap-resamples 10000
```

Exit code `0` means all gates passed. Exit code `2` means the immutable output
was produced but at least one scientific/protocol gate failed; inspect
`summary.json` rather than deleting the result. Exit code `1` means migration
itself failed and no final output directory should have been published.

For historical rows without `response_token_hash` or `prompt_token_hash`, the
new manifest records them as `backfilled`; it never claims the hashes were
original evidence. Student and teacher scores are always recomputed because
the old student score is invalid and the old teacher record does not prove all
prompt/token/protocol invariants.

### 3. Merge only rescored shards

The strict merge intentionally rejects old shard artifacts that lack exact-ID
evidence. After every expected slice has a rescored output:

```bash
python -m dual_track_opd.support_aware.diagnostic \
  --merge-shards \
    "$DTOPD_OUTPUT_ROOT/support_aware_opd/<SHARD_0>.rescored-exact-v1" \
    "$DTOPD_OUTPUT_ROOT/support_aware_opd/<SHARD_1>.rescored-exact-v1" \
    "$DTOPD_OUTPUT_ROOT/support_aware_opd/<SHARD_2>.rescored-exact-v1" \
    "$DTOPD_OUTPUT_ROOT/support_aware_opd/<SHARD_3>.rescored-exact-v1" \
  --merge-output \
    "$DTOPD_OUTPUT_ROOT/support_aware_opd/<MERGED_RUN_ID>.rescored-exact-v1" \
  --merge-mode full
```

If merge rejects the inputs, do not manually deduplicate or edit JSONL. The
exception reports the missing/extra keys or mismatched invariant. Fix/rerun the
specific shard so the union is exactly one greedy plus `K` stochastic records
for every deterministic selected UID.

### 4. Required real-model smoke before using the result in the report

CPU tests prove construction and accounting, not Qwen3-VL backend behavior.
Run one fresh prompt with the real 4B/32B stack and verify in its rollout row:

```text
exact_token_alignment == true
response_token_hash == student_scored_token_hash == teacher_scored_token_hash
teacher_response_mask == student_response_mask == [true] * response_token_count
prompt_token_hash is non-empty
teacher_scored_token_count == student_scored_token_count == response_token_count
```

Also compare the batch scorer against serial scoring for the same raw IDs
within numerical tolerance. This one-prompt GPU smoke remains a server action;
it cannot be established by the local CPU doubles.

## How to interpret the new summary

Use `correct_tail_rank_metrics.within_prompt_auc` as the primary diagnostic,
not pooled rollout AUC. A defensible full-run positive result requires all of:

- tokenizer and exact-token identity gates pass;
- prompt-token hashes and complete shard coverage are 100%;
- no non-finite scores or missing images;
- truncation rate is at most 0.15;
- at least 20 eligible correct-tail prompts;
- macro within-prompt AUC point estimate is at least 0.60 and its 95% prompt
  bootstrap lower bound exceeds 0.50;
- Rank@1 lift over the prompt-specific `c_i/K` random baseline has a 95% lower
  bound above zero.

The canonical verifier may increase the malformed rate because an unmarked
last number is now correctly treated as ambiguous. This is a measurement fix,
not a regression to hide. Results are also stratified by `stop` versus
`length`, with content-only, all-token, and terminal likelihoods separated.

Do not start the three-arm RL pilot until the protocol gates and real-model
smoke pass. Per repository policy, the main FC-OPD training path must retain the
online student scorer; a teacher-only router is an explicitly named ablation.

## Local verification note

The targeted suite passes (`68 passed, 3 skipped`). The repository-wide suite
reports `359 passed, 5 skipped, 1 failed`; the sole failure is the pre-existing
environment-contract parser treating the valid constraint `cmake<4.0` as if
every line must contain `==`. It is unrelated to the files in this delivery and
was deliberately not folded into this research-correctness commit.

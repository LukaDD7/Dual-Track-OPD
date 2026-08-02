# Support-aware diagnostic correctness fix specification (2026-08-03)

## Purpose

This is an implementation handoff for Claude Code. It fixes the frozen-policy
support-aware diagnostic without changing its research question. The primary
invariant is:

> Student and teacher must force-score the exact response token IDs sampled by
> the student, under an auditable prompt, mask, and terminal-token policy.

Do not regenerate the current 4096-token rollouts merely because the display
text was decoded with `skip_special_tokens=True`. The raw generated token IDs
are already stored and can be re-scored. Regenerate only if the stored records
fail the integrity checks below or if a new prompt protocol is intentionally
being evaluated.

## 0. Baseline, evidence, and implementation map

### 0.1 Code baseline and line-number convention

This handoff was written against repository commit `3515c67` on branch
`codex/va-opd`. All line numbers below refer to that commit. Use the named
symbol as the stable locator if later commits move a line.

The repository's own experiment runbook already requires the same invariant:

- `docs/support_aware_sft_rl_opd_experiment_runbook.md:119-124` defines the
  frozen diagnostic as the prerequisite to the micro-training pilot.
- `docs/support_aware_sft_rl_opd_experiment_runbook.md:163-171` explicitly
  requires teacher and student to force-score the exact same response IDs.
- `docs/support_aware_sft_rl_opd_experiment_runbook.md:128-145` says to reuse
  the existing structured prompt, teacher scorer, student scorer, and canonical
  Geometry3K verifier rather than reimplementing them.

The required teacher-gap statistic is conceptually

```text
gap(y) = (1 / |M|) * sum_{t in M} [
    log p_teacher(y_t | x, y_<t)
  - log p_student(y_t | x, y_<t)
]
```

where `y` is the exact student-sampled token sequence and `M` is an explicitly
recorded content/all-token mask. If decoded text is re-tokenized, `y`, its
length, its boundaries, and possibly its terminal token change. The two terms
then no longer measure the same event, so their subtraction is not a valid
token-level teacher gap.

### 0.2 Ordered implementation map for Claude Code

| Priority | File and current lines | Current behavior/root cause | Required replacement |
| --- | --- | --- | --- |
| P0-1 | `src/dual_track_opd/fc_opd/student_scorer.py:122-207` | `_score_batched` repeats `chunk[0]` prompt/image across all rows (`:146-163`) even though the verl hook supplies the whole multi-prompt batch; it also right-pads responses (`:167-178`) but slices `outs.logits[i, -rlen:, :]` (`:196-205`) | Stably group by exact rendered prompt/image/condition fingerprint, batch only within a group, restore original order, and slice from the prompt boundary |
| P0-2 | `src/dual_track_opd/support_aware/scorer.py:316-409` | Serial scorer receives raw IDs but ignores them after the empty check; `chat_text + response_text` is re-tokenized at `:362-397` | Process prompt/image only; append the supplied raw IDs; gather those exact IDs |
| P0-3 | `src/dual_track_opd/support_aware/scorer.py:411-513` | Batch scorer uses raw IDs only as empty flags (`:457-470`) and scores re-tokenized display text | Build a shared prompt encoding, append right-padded raw IDs plus masks, and return scored IDs/hash |
| P0-4 | `src/dual_track_opd/support_aware/diagnostic.py:245-305` | `generate_response` returns an ambiguous `(text, ids)` tuple and records no prompt-token hash, stop reason, raw decode, or content mask | Return a typed generation record with raw/display representations and terminal metadata |
| P0-5 | `src/dual_track_opd/support_aware/diagnostic.py:845-925` | Teacher/student results are subtracted after only finite checks; there is no token-ID/mask identity assertion | Refuse to compute a gap until generated, student-scored, and teacher-scored IDs/masks match position by position |
| P0-6 | `src/dual_track_opd/fc_opd/teacher_client.py:65-88` | Client explicitly skips `response.token_ids == request.response_token_ids` at `:80-87` based on a stale claim that the teacher may “repair” 32B-vs-4B IDs | Delete the exception and raise `TeacherServiceError` on any token-ID difference; tokenizer-size compatibility must be established before the request |
| P0-7 | `src/dual_track_opd/support_aware/scorer.py:29-65`; `teacher_client.py:27-43` | Support wrapper does not pass the student's known tokenizer hash into `TeacherClient(expected_tokenizer_hash=...)` | Add `expected_tokenizer_hash` to `TeacherScorerConfig` and fail during construction/preflight |
| P0-8 | `src/dual_track_opd/support_aware/verifier.py:42-65` | Missing `Answer:` falls back to scanning the entire response and taking a number | Route through `fc_opd.verifier.verify_geometry3k_response` and the conservative extractor at `fc_opd/answer_extraction.py:15-38` |
| P0-9 | `src/dual_track_opd/support_aware/diagnostic.py:803-842` | Greedy response is verified at `:812`, but malformed count is incremented only for stochastic responses at `:833-837` | Count malformed for every rollout through one shared record-building helper |
| P0-10 | New CLI path in `support_aware/diagnostic.py` or a small `support_aware/rescore.py` | Existing 4096 outputs have no safe migration path | Add append-only `--rescore-existing`; never mutate original JSONL |
| P1-1 | `src/dual_track_opd/support_aware/diagnostic.py:431-444`, `:1246-1329` | Primary AUC pools rollouts across prompts; Rank@1 has no prompt-specific random baseline or CI | Add macro within-prompt pairwise AUC, prompt bootstrap, `c_i/K` Rank@1 baseline, and random MRR baseline |
| P1-2 | `src/dual_track_opd/support_aware/diagnostic.py:1332-1398`; `configs/experiment/support_aware_geometry3k_pilot.yaml:53-62` | Gates are hardcoded; configured `correct_tail_rank1_above_random` is never executed; no truncation/exact-ID/completeness gate | Parse gates into config and make every declared gate executable and CI-aware |
| P1-3 | `src/dual_track_opd/support_aware/diagnostic.py:515-636` | Merge silently discards duplicate rollout/UID keys (`:544-565`), conflates student/teacher hashes (`:624-625`), and writes fake git metadata (`:629-630`) | Fail closed on overlap/incompleteness/config mismatch and preserve per-side metadata |
| P1-4 | `src/dual_track_opd/support_aware/diagnostic.py:718-735` | Student model is loaded once for generation and a second time inside `StudentScorer` despite the comment saying it is the same model | Allow `StudentScorer` dependency injection of the live model/processor; retain standalone loading only as an explicit fallback |
| P1-5 | `src/dual_track_opd/support_aware/diagnostic.py:60-66`; `fc_opd/student_rollout_signal_audit.py:56-80,498-511` | Diagnostic duplicates a legacy `Answer:` prompt instead of using the canonical structured prompt factory | Add `prompt_version`/`rollout_response_format` to config and call `build_rollout_prompt` |

Do not change the essential exact-ID behavior in
`fc_opd/teacher_transformers.py:250-321`: it already concatenates the request's
raw response IDs, recomputes multimodal positions, directly indexes the same
IDs, and returns them. Its round-trip check at `:148-179` is correctly only a
warning. The student path should be made symmetric with it. In contrast, do
change the stale permissive client check in `teacher_client.py:80-87`: the
service response must return exactly the requested IDs.

### 0.3 Recommended commit order

To keep review and rollback simple, implement in four commits:

1. `fix online student batch prompt and response alignment`
2. `fix exact-token support diagnostic scoring`
3. `add immutable diagnostic rescoring and canonical verification`
4. `add prompt-level statistics and strict merge gates`

Do not combine a new training experiment with these correctness changes.

### 0.4 Concrete target data structures

Replace the `(response_text, response_token_ids)` tuple from
`generate_response` with a frozen dataclass similar to:

```python
@dataclass(frozen=True)
class GenerationRecord:
    response_token_ids_raw: tuple[int, ...]
    response_text_display: str
    response_text_raw: str
    response_token_hash: str
    prompt_token_hash: str
    finish_reason: Literal["stop", "length", "unknown"]
    terminal_token_id: int | None
    content_mask: tuple[bool, ...]
```

Use a deterministic integer-sequence hash, not `str(tuple(ids))`:

```python
def hash_token_ids(ids: Sequence[int]) -> str:
    payload = json.dumps([int(x) for x in ids], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
```

Extend `StudentScorer.ScoreResult` at
`support_aware/scorer.py:289-294` to include:

```python
scored_token_ids: tuple[int, ...]
scored_token_hash: str
response_mask: tuple[bool, ...]
sum_logp_all: float
mean_logp_all: float
sum_logp_content: float
mean_logp_content: float
terminal_logp: float | None
```

Make `TeacherClient.score` compare `TeacherScoreResponse.token_ids` against the
corresponding request before converting to `TeacherTopK`. Then make the support
teacher wrapper return the validated scored IDs/hash in its local
`ScoreResult`, rather than discarding the alignment evidence.

### 0.5 Concrete exact-ID batch construction

For a same-prompt group, construct the batch from prompt tokens plus raw
response IDs. The implementation should be equivalent to:

```python
prompt_enc = processor(
    text=[chat_text] * batch_size,
    images=[image] * batch_size,
    return_tensors="pt",
    padding=True,
)
prompt_ids = prompt_enc["input_ids"]
prompt_width = prompt_ids.shape[1]

response_lens = torch.tensor([len(ids) for ids in response_ids_list])
max_response_len = int(response_lens.max())
response_mask = (
    torch.arange(max_response_len)[None, :] < response_lens[:, None]
)
response_ids = torch.full(
    (batch_size, max_response_len),
    fill_value=pad_token_id,
    dtype=prompt_ids.dtype,
)
for row, ids in enumerate(response_ids_list):
    response_ids[row, :len(ids)] = torch.tensor(ids, dtype=prompt_ids.dtype)

model_inputs["input_ids"] = torch.cat([prompt_ids, response_ids], dim=1)
model_inputs["attention_mask"] = torch.cat(
    [prompt_enc["attention_mask"], response_mask.to(prompt_enc["attention_mask"].dtype)],
    dim=1,
)
```

Extend `token_type_ids` and `mm_token_type_ids` with zeros as in
`teacher_transformers.py:275-289`. Recompute position IDs when required by the
Qwen3-VL backend. For row `i`, the logits predicting its raw response are:

```python
rlen = len(response_ids_list[i])
response_logits = logits[i, prompt_width - 1 : prompt_width - 1 + rlen]
sampled_logp = log_softmax(response_logits.float(), -1).gather(
    -1,
    raw_response_ids[:, None],
).squeeze(-1)
```

Do not use the pad-filled portion in either gather or a returned mask. Assert
that the gathered length is exactly `rlen`.

### 0.6 Concrete online scorer grouping fix

`verl_post_rollout_hook.py:50-65` builds one `OnlineFCOPDSample` per batch row,
so `_score_batched` must assume rows may have different prompts. Use a stable
grouping key such as:

```python
key = hash_json({
    "condition": condition.value,
    "rendered_messages": rendered.messages,
    "image_paths": rendered.image_paths,
    "question": sample.question,
    "condition_inputs": sample.condition_inputs.to_dict(),
})
```

Create `(original_index, sample)` groups, use the exact-ID batch construction
only within each group, and assign each result back to `results[original_index]`.
The grouping hash is for partitioning/audit only; never rely on a hash collision
for semantic equality without also comparing the canonical payload.

This approach is preferred over padding unrelated multimodal prompts together
for the deadline fix: each prompt normally has several rollout samples, so it
retains the useful K-response batch dimension while minimizing Qwen3-VL
M-RoPE/image-placeholder risk.

## 1. Correct handling of decoded text and token IDs

### 1.1 Keep two representations with different responsibilities

Generation must return/store a structured record containing at least:

- `response_token_ids_raw`: exact IDs returned by `model.generate`, after the
  prompt boundary and before any decode/re-encode operation.
- `response_text_display`: decode of the raw IDs with
  `skip_special_tokens=True, clean_up_tokenization_spaces=False`; use only for
  human inspection and answer verification.
- `response_text_raw`: optional audit decode with
  `skip_special_tokens=False, clean_up_tokenization_spaces=False`.
- `response_token_hash`: SHA-256 of the raw integer sequence.
- `prompt_token_hash`: SHA-256 of the exact student prompt input IDs used for
  generation.
- `finish_reason`: `stop`, `length`, or `unknown`.
- `terminal_token_id`: the final stop/EOS token when present, otherwise null.
- `content_token_count` and `raw_token_count`.

Decoded text is not an invertible representation of a generated token
sequence. Special tokens can disappear, whitespace can normalize, and BPE
tokens can merge differently at a concatenation boundary. Therefore neither
student nor teacher scoring may call `tokenizer.encode(response_text_display)`
to construct the scored suffix.

### 1.2 Student forced scoring

Refactor both `StudentScorer.score` and `StudentScorer.score_batch` in
`src/dual_track_opd/support_aware/scorer.py`:

1. Render and process the prompt/image only.
2. Take `response_token_ids_raw` directly from the generation record.
3. Right-pad response IDs for batching and concatenate them to the prompt
   `input_ids`.
4. Extend `attention_mask` with the true response mask; padded positions must
   be false.
5. Extend `token_type_ids` / `mm_token_type_ids` exactly as the teacher scorer
   does. Recompute multimodal position IDs if the model backend requires it.
6. Extract logits from `[prompt_len - 1 : prompt_len - 1 + response_len]` for
   each row. Do not use `-response_len:` in a right-padded batch.
7. Gather log probabilities using the original response IDs.
8. Return the scored token IDs/hash as part of `ScoreResult` so the caller can
   assert identity, rather than checking token counts alone.

Factor the prompt-plus-response construction into one tested helper and use it
from both serial and batch paths. The exact-ID append pattern in
`fc_opd/teacher_transformers.py` can be reused; do not create a third
text-concatenation implementation.

Audit and fix the online `fc_opd/student_scorer.py` batch path in the same
change. It currently right-pads responses to `max_R` but slices
`outs.logits[i, -rlen:, :]`; for rows shorter than `max_R`, that selects padded
tail positions. Use the prompt boundary window
`[prompt_len - 1 : prompt_len - 1 + rlen]`, and add a mixed-response-length
batch-versus-serial regression test. This matters to the main online-student
FC-OPD path, not only the frozen diagnostic.

The same online batch path also renders the prompt and opens the image from
`chunk[0]`, then repeats them for every row. The verl post-rollout hook passes
samples from the whole rollout batch, so a chunk can contain different
questions/images. Do not score those rows under the first sample's context.
For the least risky fix, stably group samples by an exact rendered-prompt and
condition-input fingerprint, batch only rows within the same prompt/image
group, and restore results to their original batch order. This retains useful
K-rollout batching without introducing ragged multimodal prompt packing. Add a
two-prompt/two-image regression test whose outputs differ by prompt so prompt
leakage cannot pass unnoticed.

### 1.3 Teacher forced scoring

Keep the teacher behavior that warns on display-text round-trip drift but
force-scores the original student IDs. Add a hard caller-side assertion for
every record:

```text
teacher.token_ids == student.scored_token_ids == response_token_ids_raw
```

Also require equal response masks and equal token counts before computing any
teacher-student gap.

### 1.4 Tokenizer compatibility policy

Different parameter counts do not imply different tokenizers. Qwen models of
different sizes may share the same vocabulary and special-token mapping, in
which case exact token-ID scoring is the correct alignment.

For token-level OPD, require before generation:

- identical tokenizer fingerprints (vocabulary, added vocabulary, and special
  token mapping);
- identical response vocabulary size and token-ID semantics;
- separately recorded chat-template hashes and prompt-token hashes.

In the support diagnostic, compute the student fingerprint from the already
loaded rollout processor before constructing the teacher wrapper:

```python
student_hash = tokenizer_fingerprint(processor.tokenizer)
teacher = TeacherScorer(TeacherScorerConfig(
    base_url=config.teacher_url,
    expected_tokenizer_hash=student_hash,
))
```

Pass that value to `TeacherClient(expected_tokenizer_hash=...)`. Keep the
existing explicit preflight printout for the manifest, but do not rely on a
late string comparison after scorers have already been constructed.

If the student and teacher tokenizer fingerprints differ, fail fast for this
diagnostic and for token-level OPD. Decode/re-encode is not a valid repair: the
models no longer share the same token action space, so per-token KL and a
teacher-minus-student mean-token log-prob gap are not mathematically aligned.

If heterogeneous-tokenizer teachers are needed later, implement a separately
named sequence-level diagnostic using the same visible text and byte-normalized
sequence log likelihood, or use verified teacher generations as SFT/DPO/reward
guidance. Do not report that mode as token-level OPD.

## 2. Terminal tokens and truncation

The primary diagnostic score must be content-only, with termination reported
separately:

- `student/teacher_sum_logp_content`
- `student/teacher_mean_logp_content`
- `student/teacher_sum_logp_all`
- `student/teacher_mean_logp_all`
- `student/teacher_terminal_logp` when a terminal token exists
- content and all-token teacher gaps

Remove only a final token that is explicitly one of the configured generation
stop/EOS IDs from the content mask. Never remove arbitrary special tokens from
the middle of the response. Keep the raw all-token score for auditability.

Derive `finish_reason=stop` from an observed configured stop/EOS token and
`finish_reason=length` when the generation reaches `max_new_tokens` without a
stop token. Report metrics for all, stopped, and length-truncated strata. If
the truncated rate exceeds 15%, mark the scientific signal gates exploratory
rather than passed.

For future generation, use the canonical concise structured prompt and, if
supported by the backend, stop after the complete `</answer>` tag while still
retaining the actual generated stop sequence.

## 3. Verifier and prompt contract

Do not maintain a second permissive Geometry3K verifier.

- Route correctness through
  `fc_opd.verifier.verify_geometry3k_response` and
  `fc_opd.answer_extraction.extract_final_answer_candidate`.
- An unmarked reasoning trace containing numbers must be malformed, not parsed
  by taking its last number.
- Record correctness and format validity as separate fields.
- Count malformed greedy and stochastic rollouts consistently.
- Centralize future prompt creation through
  `fc_opd.student_rollout_signal_audit.build_rollout_prompt` with an explicit
  `rollout_response_format` and `prompt_version`; remove the duplicated prompt
  literal from the diagnostic.

Historical `Answer: ...` rollouts can be re-verified with the canonical
extractor without regeneration. Mark their prompt version as legacy; do not
claim that they satisfy the four-block `fc_opd_structured_v2` format.

## 4. Statistical outputs and acceptance gates

### 4.1 Primary signal metric

Replace pooled rollout AUC as the primary gate with macro within-prompt
pairwise AUC. For every prompt containing at least one correct and one wrong
stochastic rollout:

```text
AUC_i = mean(1[g_correct > g_wrong] + 0.5 * 1[g_correct == g_wrong])
```

Report the macro mean across prompts and a 95% confidence interval from a
seeded cluster bootstrap over prompts. Keep pooled AUC as a secondary
descriptive metric only.

### 4.2 Ranking baselines

For a prompt with `c_i` correct rollouts among `K`, the random Rank@1 baseline
is `c_i / K`, not `1 / K`. Report:

- observed Rank@1;
- mean prompt-specific random Rank@1;
- Rank@1 lift and bootstrap CI;
- observed MRR and its prompt-specific random-permutation baseline.

The exact random MRR baseline for that prompt can be computed without Monte
Carlo. If `R` is the rank of the first correct item under a random permutation:

```text
P(R = r) = C(K-r, c_i-1) / C(K, c_i),  r = 1 ... K-c_i+1
E[MRR_i] = sum_r P(R = r) / r
```

Use a seeded prompt bootstrap (`seed=42`, 10,000 resamples) over eligible
prompt-level rows, not over individual rollouts. Recommended full-mode gates:

```text
eligible mixed/correct-tail prompts >= 20
macro within-prompt AUC point estimate >= 0.60
macro within-prompt AUC 95% CI lower bound > 0.50
Rank@1 lift 95% CI lower bound > 0.0
exact-token identity rate == 1.0
truncation rate <= 0.15
shard completeness == 1.0
```

Store the bootstrap seed, resample count, eligible prompt count, point
estimate, lower bound, and upper bound in `summary.json`.

Make the configured Rank@1 gate executable; do not leave it as an unused YAML
field. A gate passes only if its lower bootstrap confidence bound exceeds the
configured random baseline/lift threshold.

### 4.3 Required strata and protocol gates

Report every primary metric for:

- all responses;
- `finish_reason=stop`;
- `finish_reason=length`;
- correct-tail prompts;
- length bins or a length-controlled sensitivity analysis.

Add hard protocol gates for exact token identity, response-mask identity,
prompt-token hash availability, tokenizer compatibility, complete shard
coverage, finite scores, and truncation rate. Protocol-gate failure must make
the signal result invalid rather than merely attach a warning.

## 5. Strict shard merge and reproducibility

Change `merge_shard_runs` from defensive silent deduplication to strict
validation:

- require `resolved_config.yaml`, `run_manifest.json`,
  `selected_prompts.jsonl`, `rollouts.jsonl`, and
  `prompt_support_summary.jsonl` from every shard;
- require identical dataset content hash, selected UID manifest hash, K,
  model IDs/revisions, tokenizer fingerprints, prompt version, scoring policy,
  and git commit;
- require disjoint shard UID sets and exactly one greedy plus K stochastic
  records per completed UID;
- require the union of UIDs to equal the expected selected UID set;
- reject duplicate rollout keys instead of silently discarding them;
- preserve separate student and teacher tokenizer hashes;
- preserve the real git commit/dirty state rather than writing `merged`/dirty;
- never emit passed signal gates for a partial merge.

Hash the source dataset file/content in addition to the selected UID list.

Perform validation before creating the output directory. Build the expected
rollout-key set as exactly

```text
{(uid, True, 0)} union {(uid, False, rollout_id) for rollout_id in 1..K}
```

for every expected UID. Compare sets and report missing and extra keys in the
exception. Do not “repair” a merge. Write the merged artifact only after every
check succeeds, using a temporary directory plus atomic rename if practical.

## 6. Migration of existing 4096-token outputs

Implement a `--rescore-existing RUN_DIR --output-dir NEW_DIR` path that is
append-only and never overwrites raw rollout data:

1. Validate raw response IDs and their hashes for all records.
2. Reconstruct the exact prompt with the recorded prompt version and assert
   its token hash when available.
3. Re-run the fixed student scorer on the raw IDs.
4. Reuse teacher scores only when their returned token IDs exactly match the
   stored raw IDs and their prompt/tokenizer metadata pass; otherwise re-score
   teacher too.
5. Re-run the canonical verifier.
6. Recompute content/all-token metrics, finish reasons, support states,
   within-prompt statistics, bootstrap intervals, and gates.
7. Write a migration manifest containing source run path/hash, old and new git
   commits, resolved config, model revisions, and reasons for every reused or
   recomputed field.

This is cheaper and scientifically cleaner than generating a new sample set,
because it preserves the exact rollout cohort while changing only the faulty
measurement layer.

Expected invocation shape:

```bash
python -m dual_track_opd.support_aware.rescore \
  --source-run /path/to/original-4096-run \
  --output-dir /path/to/original-4096-run.rescored-exact-v1 \
  --student-model /path/to/Qwen3-VL-4B-Instruct \
  --teacher-url http://127.0.0.1:18080 \
  --bootstrap-seed 42 \
  --bootstrap-resamples 10000
```

The new `run_manifest.json` must include:

```text
source_run_path
source_rollouts_sha256
source_run_manifest_sha256
source_git_commit / source_git_dirty
rescore_git_commit / rescore_git_dirty
student_model_id / revision / tokenizer_hash
teacher_model_id / revision / tokenizer_hash / protocol_version
prompt_version and whether prompt_token_hash was original or backfilled
verifier_version
terminal_token_policy
fields_reused
fields_recomputed
```

Existing historical rows do not contain all of these fields. Missing evidence
must not be invented: label it `unavailable`/`backfilled` and re-score the
teacher if exact prompt, returned-token, tokenizer, and protocol metadata
cannot all be proven. Student scores must always be recomputed because the
current student implementation used display-text re-tokenization.

## 7. Required regression tests

Add tests that fail on the current implementation:

1. `response_text_display` loses EOS, but both scorers still receive and score
   the exact original IDs.
2. Deliberately make `tokenizer.encode(response_text_display)` differ from the
   raw IDs and assert that model inputs use the raw IDs.
3. Batch responses with different lengths and assert correct right-padding,
   masks, prompt-boundary logits windows, and token gathers in both the
   support-aware scorer and `fc_opd/student_scorer.py`.
4. Batch at least two different questions/images and assert each row uses its
   own rendered context; then assert batch and serial scorer equality on the
   same exact IDs.
5. Assert tokenizer mismatch fails before signal computation.
6. Assert teacher/student/scored/generated token IDs and masks are identical.
7. Assert content-only score excludes only the terminal stop token and that
   all-token score retains it.
8. Assert unmarked truncated reasoning with a matching last number is
   malformed.
9. Assert malformed greedy responses are counted.
10. Assert prompt-macro AUC, `c_i/K` Rank@1 baseline, seeded prompt bootstrap,
    and CI-based gates on synthetic examples.
11. Assert incomplete, overlapping, or config-mismatched shards cannot merge.
12. Assert re-scoring is append-only and records source/output manifests.

Put the tests in these concrete locations so reviewers do not have to infer
coverage:

| Test file | Required coverage |
| --- | --- |
| `tests/fc_opd/test_student_scorer_batch_alignment.py` (new) | two different prompts/images in one caller batch; mixed response lengths; stable regrouping; original result order; serial/batch equality |
| `tests/fc_opd/test_teacher_service.py` | service response with one changed token ID raises; identical IDs pass; tokenizer mismatch fails before scoring |
| `tests/test_support_scorer_batch.py` | display-text round-trip differs from raw IDs; scorer input/gather still uses raw IDs; content/all/EOS masks |
| `tests/test_support_diagnostic.py` | canonical verifier rejects an unmarked final number; greedy malformed count; finish-reason classification |
| `tests/test_support_diagnostic_stats.py` (new) | macro prompt AUC, ties, bootstrap reproducibility, `c_i/K`, exact random MRR, CI gates |
| `tests/test_support_diagnostic_shards.py` | overlap, missing UID, missing rollout, K/config/model/tokenizer/git mismatch all fail closed |
| `tests/test_support_diagnostic_rescore.py` (new) | source output remains byte-identical; new manifest has source hashes and recomputation provenance |

Use a non-uniform stub model whose logits depend on both prompt token IDs and
position. A uniform-logit stub cannot detect prompt leakage or an off-by-padding
logit window because every gathered log-prob is identical.

Run the targeted tests in the server Torch environment, then the full suite:

```bash
pytest -q tests/test_support_diagnostic.py \
  tests/test_support_scorer_batch.py \
  tests/test_support_diagnostic_stats.py \
  tests/test_support_diagnostic_rescore.py \
  tests/test_support_diagnostic_resume.py \
  tests/test_support_diagnostic_shards.py \
  tests/fc_opd/test_student_scorer_batch_alignment.py \
  tests/fc_opd/test_teacher_service.py
pytest -q
```

## 8. Completion criteria

The fix is complete only when:

- all new and existing tests pass;
- a 1-prompt smoke proves native generation IDs equal student-scored and
  teacher-scored IDs position by position;
- batch and serial scoring agree within numerical tolerance;
- a re-scored 4096 run is written to a new immutable output directory;
- the report includes exact-token, verifier, truncation, within-prompt CI, and
  merge-completeness gates;
- old 512/2048 results remain explicitly marked invalid and are not silently
  replaced.

Do not start the three-arm RL pilot until these protocol gates pass. The pilot
must retain the online student scorer; a teacher-only router is an ablation,
not the main FC-OPD path.

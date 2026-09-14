# Causal State Probe Pre-check Review — 2026-08-11

## Scope and verdict

This review covers commit `20ca9d2` and, in particular, the semantics of
`precheck_causal_state_probe.py`, continuation `malformed` accounting, the
actionability estimates, and the single-forward fixed-trajectory optimization.

The basic fixed-prefix probe design and response-logit indexing are sound, and
the relevant isolated tests pass (`14 passed`).  The current pre-check should
not yet be used as final evidence for state-class or strongest-candidate claims,
however.  Its reported relay "coverage" is an all-clean K=8 group rate rather
than continuation-level coverage, `n_malformed` conflates distinct outcomes,
and the advertised valid-only subset does not require a clean control estimate.

No full GPU rerun is required before fixing the offline report.  Most coverage
diagnostics can be recomputed from the existing trajectory JSON.  A small,
targeted replay is required only to split historical malformed causes that were
not retained in the current schema.

## P0: Relay coverage is not continuation coverage

The pre-check increments `relay_coverage[L]["valid"]` only when
`estimate.n_malformed == 0`.  With K=8, one malformed continuation makes the
entire estimate non-valid even when the other seven continuations are usable.
Consequently, the current value is the probability of an **8/8 all-clean
estimate**, not the fraction of usable continuations.

The compounding is large:

| per-continuation malformed rate | expected K=8 all-clean rate |
|---:|---:|
| 10.0% | 43.0% |
| 14.3% | 29.1% |
| 20.2% | 16.4% |

Required report changes:

1. Rename the current metric to `all_clean_estimate_rate`.
2. Add `continuation_valid_rate = sum(n - n_malformed) / sum(n)`.
3. Add the distribution of `n_malformed` over 0..K, by relay length and anchor
   position bucket.
4. Keep `fully_malformed` (K/K) separate from `with_any_malformed` (1..K).
5. Do not describe `with_any_malformed` estimates as unusable.

Acceptance criterion: a synthetic collection in which every K=8 estimate has
exactly one malformed continuation must report 87.5% continuation coverage,
0% all-clean estimate rate, and 0% fully-malformed estimate rate.

## P0: `n_malformed` conflates censoring and output format failure

`continuation_estimate()` defines every `correctness is None` outcome as
malformed.  `teacher_relay_estimate()` also appends `None` when the teacher relay
prefix exposes an extractable answer, without generating a student
continuation.  The same counter can therefore include:

- relay answer leakage (student continuation was never generated);
- no explicit final-answer marker;
- length truncation before a final-answer marker;
- another verifier-unparseable response.

These outcomes have different causal meanings.  In particular, a relay that
solves too early is not evidence that the student's continuation is malformed.
The current label
`relay_l{L}_answer_leakage_or_malformed` acknowledges the ambiguity but does not
preserve the counts needed to resolve it.

Required schema/runtime changes:

- store per-estimate reason counts at minimum:
  `n_correct`, `n_wrong_format_valid`, `n_no_answer_marker`, `n_truncated`,
  `n_relay_answer_leakage`, and `n_generation_error`;
- store `n_student_continuations_generated` explicitly;
- retain per-rollout `finish_reason`, verifier extraction status, seed, and
  response hash; raw text may remain outside Git under an auditable raw-output
  path;
- reserve `malformed` for a generated, verifier-unparseable continuation rather
  than using it for intervention censoring.

Historical shard JSON cannot fully recover this split because it stores hashes
and aggregate counts, not per-rollout reasons.  Replay a small stratified sample
of high-`n_malformed` candidates with the saved seeds and verify response hashes
before deciding whether a full rerun is necessary.

## P0: Main gains count malformed/leakage as failures

`compare_estimates()` uses `n_correct` and the original K for its Jeffreys
posterior.  Thus malformed continuations and relay-leakage samples enter the
main relay gain as failures; they are not excluded.  This is a defensible
intent-to-treat-style operational lower bound, but it is not a conditional
estimate of student repair after an answer-free relay.

Longer relay lengths have more opportunity to expose the final answer, so the
current estimator can systematically penalize L=64/L=128 and confound relay
reachability with the availability of an answer-free teacher segment.

Required analysis changes:

1. Preserve the current all-outcomes estimate, clearly named `itt_lower_bound`.
2. Report relay answer-leakage probability as its own outcome.
3. Report a descriptive answer-free/generated-continuation estimate using its
   actual effective denominator.  Label it conditional/sensitivity analysis;
   do not treat post-treatment filtering as an unbiased causal estimand.
4. Require both treatment and control denominators/reason counts in every
   comparison table.
5. Stratify results by relay length and anchor position instead of interpreting
   a single pooled malformed number.

State classification and `strong_relay_examples` should continue to use the
predeclared conservative estimator only if the report explicitly labels it as
such and shows that the result is not driven by leakage/malformed imbalance.

## P1: The advertised valid-only subset is not pair-valid

`relay_valid_only_by_length` checks only the relay treatment's
`n_malformed == 0`.  It does not check the corresponding `baseline_full`
estimate.  Cross-length "fully valid" consistency has the same limitation.
A clean treatment can therefore be called fully valid even when its control
contains malformed outcomes.

Required change: persist or expose baseline reason counts in the pre-check and
define pair-clean coverage as both treatment and matched control being clean.
Report treatment-clean, control-clean, and pair-clean rates separately.

## P1: Pre-check validation is shallower than its verdict claims

`validate()` checks top-level schema version, duplicate trajectory IDs, and
finiteness in token signals/top-level candidate scalars.  It does not validate:

- shard manifests, expected work IDs, or completion;
- nested continuation probabilities/count identities;
- nested non-finite values;
- expected K, seeds, or response-hash counts;
- candidate bounds and duplicate candidate IDs;
- config/provenance consistency across shards;
- old-implementation versus new-implementation unit membership.

Therefore `No schema/NaN/duplicate issues found` overstates what was checked.
Either reconstruct and call the schema validators, or rename the message to the
specific checks performed.  Before final merge, compare every manifest's
expected work IDs with actual trajectory IDs and verify shared config,
tokenizer, input hashes, backend version, repo commit, and dirty status.

The generated title is also hard-coded to "64-sample" even though the script
accepts arbitrary partial or complete inputs; derive the count dynamically.

## P1: Single-forward memory is not unchanged

The causal indexing of the optimization is correct, but the peak-memory claim
is not.  With a roughly 151k-token vocabulary and a 4096-token trajectory, one
bf16 full-response logit tensor is about 1.16 GiB:

- visual statistics retain full/degraded/null logits together: roughly
  3.5 GiB of response logits before JS working memory;
- teacher-path statistics retain student and moved teacher logits together on
  the student GPU: roughly 2.3 GiB before working memory.

The previous `logits_to_keep` path retained roughly 64-token chunks (about
19 MiB each), although its last backbone forward still processed the full
prefix.  Full-prefix hidden-state compute and retained vocabulary logits are
different memory costs.

Required action:

- correct the documentation and record measured `torch.cuda.max_memory_*` on a
  long trajectory;
- if headroom is insufficient, run the backbone once and apply the LM head to
  hidden states in vocabulary-logit chunks, or otherwise offload/release
  condition logits without retaining all full-vocabulary tensors together;
- add a long-trajectory GPU memory smoke before resuming all shards.

## P1: Old and new scoring implementations are not numerically identical

The toy causal-LM equality tests pass and establish correct indexing.  They do
not establish numerical identity for Qwen bf16 kernels across different
sequence lengths.  The run report itself records a maximum logit difference of
about 0.53 for positions compared under different forward lengths.  The old
implementation scores early chunks with shorter forward shapes; the new one
scores all positions in the maximum-length shape.

This is likely shape-dependent bf16 kernel noise rather than causal leakage,
but candidate ranking and threshold/NMS selection are nonlinear.  It is
therefore unsafe to claim exact equivalence or silently mix old and new units.

Required action:

- add a real-model old-vs-new end-to-end comparison over token JS curves,
  selected anchors, and state inputs, not only logits at equal sequence length;
- record an implementation/version field per trajectory result;
- either rerun old units with the new implementation or report old/new strata
  and demonstrate that candidate selection is stable before merging.

## Re-analysis order

1. Fix the offline coverage labels and pair-valid accounting.
2. Recompute existing results without rerunning GPUs.
3. Replay a small stratified malformed sample to identify reason proportions.
4. Decide whether historical units need regeneration.
5. Validate long-trajectory GPU memory.
6. Only then publish final state-class and actionable-candidate conclusions.

Until those checks pass, visual JS curves and teacher-path NLL/barrier summaries
remain useful descriptive diagnostics, but relay/transport gains and derived
state classes should be treated as provisional.

# User–ChatGPT Update + Phase 0–6 Audit → Claude Code Execution Brief

Date: 2026-08-15

Current evidence commit: `40ac71dc5a99c64118d6d57439f52829e828977b`

This document is the self-contained translation of the user's latest research
convergence with ChatGPT, reconciled with Claude Code's Phase 0–6 handback.
Claude Code does not need access to the external conversation.

## 0. Authority, immediate decision, and non-goals

The user's research decision is the source of scope.  Existing code and the
Phase 0–6 handback are evidence, not authority for the next experiment.

Effective immediately:

- keep STP canary, four-arm training, SFT, FKL training, and RL paused;
- preserve all existing STP and reachability-proxy implementation;
- do not launch the draft Phase-7 native-conversion command in the Phase 0–6
  handback;
- first correct the proxy evaluation and build the single-pass symmetric
  Top-100 cache specified here;
- do not add or change any training objective during this task;
- stop at the handoff-viability proxy gate and report evidence to the user.

The Phase 0–6 work is not discarded.  Its data integrity, rescue expansion,
GPU scorer, strict joins, and reproducibility plumbing are reusable.  Only its
research interpretation and next-step gate change.

## 1. Phase 0–6 disposition

| Component | Disposition | Reason |
|---|---|---|
| NFS preflight, hashes, manifests | PASS / reuse | Inputs and provenance were verified. |
| Frozen 88-prompt expansion | PASS / reuse | Selection stayed frozen and thresholds were not moved. |
| 52-prompt combined gold, 194 horizon rows | PASS / reuse | This is the current behavioral-gold dataset. |
| 27 scaffoldable prompts | useful but below target | The preregistered target was 30–50 positives; use for evaluator/cache development, then top up gold if needed. |
| Forced-forward scorer | PASS / extend | It already produces one student and one teacher forward per trace, but the persisted cache is asymmetric. |
| Prompt-level NLL AUROC 0.873 | exploratory association only | The evaluated prompt row is selected using the test label; it is not a deployable held-out prediction. |
| Within-prompt horizon result | NO-GO stands | No old scalar reliably identifies the minimal sufficient handoff horizon. |
| Draft Phase-7 training/conversion | NOT AUTHORIZED | The central handoff-location proxy has not passed. |

### 1.1 Mandatory evaluation correction

The current evaluator must not be used to claim prompt-level GO.

In `_prompt_level_scores`, a positive prompt is represented by its
gold-revealed minimal positive horizon, while a negative prompt is represented
by its maximum horizon.  `evaluate_proxy_heldout` repeats the same
label-conditioned row selection on held-out prompts.  A deployment-time proxy
does not know which horizon is the minimal gold-positive horizon, so this is
target leakage through evaluation-row construction.

The cumulative NLL/FKL features also grow with horizon.  Selecting earlier
rows for positives and terminal rows for negatives can create discrimination
even when the scalar does not identify handoff viability.  The reported 0.87
therefore cannot authorize selection or training.

Required regression tests:

- the held-out evaluator never reads `rescue_gold` to select a feature row;
- all `(prompt_id, horizon)` rows from a test prompt are scored before labels
  are used for metrics;
- splits are always by prompt, never by horizon row;
- tied scores use a tie-correct AUROC implementation;
- a synthetic null feature remains at chance and cannot gain AUROC from
  gold-conditioned row selection;
- thresholds and model fitting use training prompts only.

## 2. The research is now two questions, in order

### Question 1 — Where can the student take over?

For prompt `x` and verified teacher prefix ending at horizon `h`, define:

```text
q_x(h) = P_S(R=1 | image, question, verified teacher prefix y^T_<=h)
Y_x,h  = the stored preregistered rescue label for (x,h)
h*     = the first confirmed horizon with Y_x,h = 1
```

The expensive causal gold is **handoff viability**: after transporting the
frozen student to this off-policy teacher state, can it execute the remaining
solution itself?

This is not the same as asking whether the student could natively generate the
teacher prefix.  A prefix can be extremely unlikely under the student and
still be an excellent handoff state once supplied for free.

The working decomposition is:

```text
handoff viability = progress/sufficiency
                  + state compatibility
                  + residual executability
```

### Question 2 — Why can the student not reach `h*` itself?

Only after Question 1 has a reliable proxy, examine the teacher path before
`h*` to locate a native reachability barrier `[a*, h*]`.  Cumulative student
NLL and cumulative Top-100 FKL belong primarily here.

If a barrier is identified, the later intervention is Top-100 FKL only on
`[a*, h*]`, followed by a fresh no-prefix student rollout.  RL is discussed
only if that no-prefix rollout enters a mixed-support frontier.

Do not use Question-2 cumulative barriers as if they had already solved
Question 1.

## 3. Registered proxy families for Question 1

These features may be combined only through the preregistered nested models in
§6.  This is structured mechanism testing, not an unrestricted feature search.

### 3.1 Progress / state sufficiency

Primary:

```text
r_h = h / T
```

Audit-only sensitivity: remaining-length ratio `(T-h)/T`.  Position remains
the mandatory zero-cost baseline.

### 3.2 Endpoint state compatibility

For `K in {16, 100}` as primary values:

```text
O_h^K = |TopK_S(h) intersect TopK_T(h)| / K

C_h^K = sum_{v in TopK_S(h)} p_T(v)
        # teacher mass on student support; core compatibility feature

M_h^K = sum_{v in TopK_T(h)} p_S(v)
        # student mass on teacher support

D_h^K = endpoint Top-K FKL with a tail bucket
```

Report `K={4,64}` only as sensitivity from the same Top-100 cache.  Do not run
separate model forwards for different K.

### 3.3 Short-horizon residual executability

For future windows `W in {32,64}` after candidate handoff `h`:

```text
NLL_takeoff(h,W) = mean_{j=h+1}^{h+W} -log p_S(y^T_j | teacher-forced context)

FKL_takeoff(h,W) = mean_{j=h+1}^{h+W}
                   D_FKL_top100+tail(p_T^j || p_S^j)
```

Clip deterministically at the trace end and persist the actual token count.
These are cheap teacher-forced corridor-compatibility features, not substitutes
for rollout gold.

Compatibility-mass window averages may be derived from the same cache as
secondary features.

### 3.4 Handoff transition contrast

For the same W, compute past and future window means and their contrast:

```text
Delta_handoff(h,W) = FKL_past(h,W) - FKL_takeoff(h,W)
```

A large positive value means the teacher path was difficult before `h` but the
post-handoff corridor becomes student-compatible.  Register the analogous NLL
contrast as a hard-path sensitivity, not an additional primary model family.

### 3.5 Visual incremental features

If the existing visual counterfactual scorer can provide them without a new
large experimental branch, retain endpoint and future-window visual JS only as
M4 incremental features.  They are mechanism probes and must not block the
generic handoff study.

### 3.6 Negative-control baselines

Report teacher entropy, student entropy, sampled-token log-ratio, Top1–Top2
margin, and raw/truncated disagreement where available.  Their purpose is to
test that generic uncertainty/disagreement is not equivalent to handoff
viability.  Do not expand this list post hoc after viewing results.

## 4. Existing cache: what can be salvaged and what is missing

The current `proxy_token_rows.jsonl` stores:

- teacher Top-100 IDs and log-probabilities;
- student log-probabilities gathered at the teacher Top-100 IDs;
- teacher and student tail mass on the teacher support;
- sampled teacher-token student NLL;
- per-token teacher-support Top-100+tail FKL.

Without any GPU work, it can already derive:

- position;
- `M_h^K` for `K={4,16,64,100}`;
- endpoint and past/future-window NLL/FKL;
- NLL/FKL handoff contrasts;
- cumulative barriers for later Question-2 analysis.

It cannot reconstruct:

- student Top-100 IDs or the overlap `O_h^K`;
- teacher probability mass at student Top-K IDs, `C_h^K`;
- exact student entropy or student Top1–Top2 margin;
- optional visual-JS features.

Do a CPU-only corrected analysis from the existing cache first.  Then extend
the exporter once, and rescore once, rather than repeatedly occupying H200s.

## 5. Single-pass symmetric Top-100 cache extension

For every response position, persist outside Git:

```text
position_index
teacher_top100_ids
teacher_top100_logprobs
student_logp_at_teacher_top100_ids
teacher_tail_on_teacher_support
student_tail_on_teacher_support

student_top100_ids
student_top100_logprobs
teacher_logp_at_student_top100_ids
student_tail_on_student_support
teacher_tail_on_student_support

sampled_teacher_token_student_nll
fkl_top100_tail_on_teacher_support
teacher_entropy
student_entropy
teacher_top1_top2_margin
student_top1_top2_margin
```

The cross-gather `teacher_logp_at_student_top100_ids` is required: storing the
two models' separate Top-100 lists alone is insufficient to compute teacher
mass on student support when a student token lies outside the teacher Top-100.

Requirements:

- one student and one teacher forced forward per trace;
- derive every K slice from the cached Top-100 tensors;
- keep full arrays and raw outputs outside Git;
- record schema version, checkpoint/tokenizer IDs, hashes, resolved config,
  backend version, repo commit/dirty state, and source-manifest hashes;
- no autograd, optimizer, loss, router, Ray training, or checkpoint write;
- preserve resumability and existing strict trace/hash checks.

Add unit tests for symmetric cross-gathers, overlap, both directional support
masses, K slicing, endpoint alignment, window clipping, contrasts, entropy and
margin equivalence, deterministic output, and cache-schema rejection.

## 6. Correct evaluation and nested models

### 6.1 Primary unit of prediction

The primary dataset is one observed candidate row per `(prompt_id, horizon)`:

```text
X_x,h -> Y_x,h
```

All rows for a prompt stay in the same split.  Evaluate raw features and a
within-prompt residualized version as separate, preregistered views.  The
residualized view removes between-problem difficulty but does not replace raw
deployment metrics.

The old prompt-level task may remain secondary only with a deployment-valid,
label-free aggregation fixed in advance, for example `max_h p(Y_x,h=1)`.  It
must never select a row using `Y` or `h*`.

### 6.2 Preregistered nested models

```text
M0 Position only:
   [r_h]

M1 Compatibility only:
   [C_h^16, M_h^16, O_h^16]

M2 Takeoff only:
   [FKL_takeoff(h,32), FKL_takeoff(h,64)]

M3 Mechanistic combination:
   [r_h, C_h^16, FKL_takeoff(h,64), Delta_handoff(h,64)]

M4 Visual increment:
   M3 + endpoint/future visual JS, only if already available
```

First report every single-feature AUROC/AUPRC.  Fit regularized logistic
regression for M0–M4.  A monotonic GAM or small monotonic GBDT is allowed only
as a preregistered second-layer sensitivity.  Do not use an MLP or neural
router.

Predeclare directions:

```text
r_h up             -> viability up
C_h and M_h up     -> viability up
O_h up             -> viability up
FKL_takeoff down   -> viability up
```

### 6.3 Splitting, uncertainty, and metrics

- group exclusively by `prompt_id`;
- use repeated stratified grouped folds because one 21-prompt test split is
  too unstable for model selection;
- keep one frozen final prompt holdout if sample size after gold top-up permits;
- perform 2,000-resample cluster bootstrap by prompt, carrying all candidate
  rows and prompt multiplicity together;
- fit scalers, thresholds, residualization means, and models on training data
  only;
- report AUROC, AUPRC versus prevalence, and paired delta versus M0;
- report within-prompt `h*` recall@1, +/- one bin, absolute bin distance, and
  pairwise ordering;
- report results by rare-success / no-correct-observed / mixed-support stratum.

## 7. Execution order

### Phase A — CPU-only evaluator repair and cache salvage

1. Freeze a new analysis config and cache schema version.
2. Replace label-conditioned prompt-row selection with prefix-level grouped
   evaluation.
3. Add leakage/null/tie/group-split regression tests.
4. Derive all currently available endpoint/window/contrast features from the
   existing token cache.
5. Re-run M0, available M1 components, M2, and M3-lite without GPU work.
6. Hand back corrected metrics explicitly labeled as partial because `C_h` and
   overlap are not yet cached.

### Phase B — one symmetric-cache scoring pass

1. Implement and test §5 only.
2. Two-trace smoke on the already verified short/long traces.
3. Rescore the frozen combined cohort once.
4. Validate cache completeness, mass conservation, hashes, and deterministic
   CPU derivation before releasing GPUs.

### Phase C — full registered proxy evaluation

1. Materialize a versioned feature table outside Git.
2. Run single-feature analysis and M0–M4 exactly as registered.
3. Run grouped repeated validation and prompt-cluster bootstrap.
4. Report predictive metrics, within-prompt horizon metrics, calibration
   thresholds, strata, failure cases, and paired deltas versus position.

### Phase D — gold top-up only if required

The current 27 positive prompts are below the original 30–50 target.  Freeze
the extractor and evaluator before adding gold.  If confidence intervals or
fold class balance remain unstable, extend the existing frozen-pool protocol
without changing rescue thresholds, aiming for at least 40 confirmed positive
prompts plus negative controls.  Keep `teacher_trace_unavailable`,
`wrong_control_unavailable`, and no-wrong outcomes as separate skip reasons;
never relabel them as negatives.

Do not perform a new broad K=32 support run merely to increase sample size.

## 8. Decision gate

Question 1 is GO only if all of the following hold on prompt-disjoint evidence:

- at least one compatibility/executability model has AUROC confidence above
  chance and AUPRC above prevalence;
- it improves on or justifies itself against position under paired
  prompt-cluster bootstrap;
- it gives useful, stable within-prompt `h*` ranking across folds and strata;
- its threshold/model is calibrated without test-label row selection;
- the result is not driven by one horizon, trace length, or one prompt stratum.

If only prompt-level discrimination works but `h*` localization remains poor,
return NO-GO for training and collect/refine handoff gold.  Do not replace the
failed horizon gate with a conservative fixed horizon and proceed as if the
research question were solved.

If GO, stop and ask for user approval before starting Question 2.  The next
proposal may then:

1. estimate `h*` with the selected handoff proxy;
2. use past cumulative NLL/FKL and support change points to estimate `a*`;
3. distill Top-100 FKL only on `[a*, h*]`;
4. run fresh no-prefix student `G=8` rollouts;
5. measure `0/G -> 0<k<G` or stronger mixed-support conversion;
6. discuss RL only after stable conversion.

## 9. Required Claude Code handback

Return one concise handback containing:

- branch/commit/dirty status and every new commit;
- exact code/config/test surface changed;
- proof that the old label-conditioned evaluator is no longer used;
- existing-cache partial results and symmetric-cache full results separately;
- prompt/horizon counts, labels, skips, strata, and all source hashes;
- M0–M4 held-out metrics with prompt-cluster uncertainty;
- within-prompt `h*` metrics and failure cases;
- a clear GO/NO-GO for Question 1;
- confirmation that no loss, training, checkpoint update, or Phase-7 launch
  occurred;
- the next proposed command and resource estimate, marked **not launched**.

Raw token arrays, feature tables, rescue JSONL, model outputs, and caches stay
outside Git.  Only code, configs, tests, compact summaries, and this execution
contract belong in the repository.

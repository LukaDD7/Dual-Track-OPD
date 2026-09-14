# User–ChatGPT Research Convergence → Claude Code Execution Brief

Date: 2026-08-14

> **Superseded for future execution.** This brief produced the completed
> Phase 0–6 evidence handback, but its exactly-three-scalars restriction and
> proxy decision protocol are superseded by
> `docs/cc_handoff_viability_proxy_adjustment_20260815.md`.  Keep this file as
> provenance; do not use it to authorize Phase 7 or training.

Source: the user's research decision, fully translated into this document.
This specification is self-contained; Claude Code does not need access to any
external conversation.

Repository context only: branch `codex/va-opd`, implementation commit
`489d61a0162950bd341f23e2efdac536a8bc3d7b`

## 0. How Claude Code must use this brief

The research direction below is a user decision reached with ChatGPT.  It is
not a conclusion delegated to Claude Code and is not derived from the amount
of STP code already implemented.

Claude Code is the execution and evidence layer.  It must:

- accept the converged research question and experiment order below;
- use existing repository work only to establish what can be reused;
- verify data and implementation claims instead of inferring them;
- implement the smallest missing offline measurement path;
- run the registered experiment and return evidence;
- stop at the stated gate and let the user make the next research decision.

Claude Code must not:

- reopen whether STP canary/four-arm training should run first;
- defend an old handoff because implementation already exists;
- expand the proxy set, add a loss, or turn the task into a framework rewrite;
- treat its own previous plan, commit message, or documentation as research
  authority.

The older `docs/cc_causal_state_to_stp_handoff_20260813.md` remains useful as
implementation history and evidence inventory only.  Its launch direction is
superseded by the user's decision recorded here.

## 1. The research question and four-step route are already converged

The immediate research problem is not to optimize STP-OPD.  It is to discover
a cheap signal for whether a low-native-support student can continue
successfully after being transported onto a verified teacher trajectory.

The research object is the latent **reachability barrier** and the
**minimal sufficient scaffold** `h*`.  It is not whether a prefix should be
30% or 40%, and this phase does not jointly solve where distillation should
start.  First determine the minimum endpoint the teacher must reach before the
student can continue successfully.

The methodological template is fixed:

```text
small expensive causal experiment
    -> functional rescue/reachability gold
    -> cheapest predictive proxy
    -> large-scale method uses only the proxy
```

### 1.1 Why the expensive diagnostic comes before the cheap proxy

The motivating pattern is a PW-OPSD-style workflow: first use a small, costly
intervention to obtain a functional label, then test whether a very cheap
feature predicts that label.  The costly intervention is not repeated online
during training.  It is used once to discover structure.

Applied here:

```text
expensive label:
    teacher-prefix transport -> frozen student continuation -> final verifier

cheap candidate:
    position OR cumulative student NLL OR cumulative Top-100 FKL+tail

deployment:
    use only the selected cheap scalar to choose the scaffold/horizon
```

Do not assume the most complex feature must win.  Position is deliberately a
strong baseline because reasoning trajectories have phase structure.  If it
matches or beats NLL/FKL, that is a valid and preferable result.

### 1.2 What the three proxy candidates mean

- **Cumulative student NLL** asks whether the student can reproduce the one
  exact path sampled by the teacher.  It is a path-imitation difficulty.
- **Cumulative Top-100 FKL+tail** asks whether the student's local support
  overlaps the teacher's set of plausible next tokens.  It is a support-space
  mismatch and is theoretically closer to reachability when several reasoning
  continuations are valid.
- **Position** asks whether trajectory phase alone predicts when the student
  can take over.  It is the zero-cost baseline that the other two must justify
  exceeding.

The hard-NLL and soft-FKL signals are not interchangeable.  A teacher may
sample token A while both teacher and student assign high probability to a
semantically equivalent token B.  NLL on A penalizes that path difference;
Top-100 FKL can recognize the broader support overlap.

### 1.3 Why SFT, RL, and new losses are out of scope now

SFT trains the student on teacher states and can move correct trajectories into
student support, but it does not establish which states are the minimum needed
for reachability.  Outcome RL trains on current student states but gives little
or no useful relative signal on native `0/G` prompt groups.  Adding either now
would mix proxy discovery with policy change and destroy the frozen-student
causal label.

Teacher-sampled token CE is related to a Monte-Carlo forward-KL estimator, but
selected correct teacher traces are a filtered demonstration distribution,
not an unbiased sample from the raw teacher policy.  This is another reason to
measure hard NLL and soft Top-100 FKL separately before choosing an objective.

Therefore the student and teacher checkpoints remain frozen throughout this
brief.  The first allowed weight update is the separately approved experiment
after proxy selection.

The four research steps are:

1. Use existing causal/rescue evidence to make the proxy plumbing work and
   align position, cumulative student NLL, and cumulative Top-100 FKL+tail
   with teacher-prefix transport gold.  Do not train.
2. Reuse the existing 256-prompt support pool to expand causal rescue gold;
   do not launch another broad K=32 support-sampling run.
3. Select among exactly the three scalar proxies using prompt-level and
   within-prompt held-out evaluation.  Prefer the simplest signal that works.
4. Only after the proxy succeeds, propose one clean non-RL native-frontier
   conversion experiment.  Training remains behind explicit user approval.

Current decision:

- **PAUSE** the STP canary, the four-arm mechanics pilot, and all training.
- Preserve commit `489d61a` and all STP implementation work.  Do not delete,
  revert, or redesign it during this task.
- Do not add a loss, change the router, or launch RL/SFT/FKL training.
- Run only data verification, forced-forward scoring, offline export, causal
  rescue expansion, and scalar proxy evaluation described here.
- Stop and report the proxy result before proposing any training command.

The risk being controlled is premature optimization of the wrong objective.
Prefix rescue is a real causal effect; the previous localization statistics did
not predict it.  The next information-gain step is therefore proxy discovery.

## 2. Four execution confirmations required from Claude Code

These are implementation/evidence confirmations, not four open research
questions and not an invitation to choose a different direction.

| # | Required confirmation | Current evidence | Required CC action |
|---|---|---|---|
| 1 | STP canary/four-arm training is paused while its implementation is preserved. | User decision is fixed; `489d61a` is present on `codex/va-opd`. | Confirm no training process is launched; do not delete, revert, or extend STP during this task. |
| 2 | The 64-prompt causal outputs and 256-prompt support outputs remain intact and readable. | Canonical paths are documented; the Mac checkout cannot verify `/inspire`. | Verify files, counts, hashes, and manifests on the HPC/NFS host before implementation. |
| 3 | One forced forward over a verified correct teacher trace can export every quantity needed by NLL and Top-100+tail. | Repository components expose the needed logits/top-k/tail pieces, but no single causal-proxy artifact currently exports them. | Prove the component contract, then add only the missing inference exporter.  Do not add a training loss. |
| 4 | Existing rescue records can produce a strict offline `proxy_study.csv`. | Join appears feasible but is conditional on NFS integrity and exact trace identity.  The CSV does not yet exist. | Build the 12-prompt plumbing artifact first; expand causal gold only after its coverage checks pass. |

Do not turn any conditional answer above into an unqualified “done”.  The
required handback must distinguish verified facts, newly produced evidence,
and remaining blockers.

## 3. Canonical evidence and NFS preflight

Use this root:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/support_aware_opd
```

Required inputs:

| Evidence | Canonical path | Minimum required files/coverage |
|---|---|---|
| 256-prompt K=8 support pool | `diag_full_20260802_256_merged.rescored-exact-v1/` | `summary.json`, `rollouts.jsonl` with 2304 rows, `prompt_support_summary.jsonl` with 256 prompts, `frontier_analysis/frontier_prompts.jsonl` |
| 64-prompt K=32 confirmation | `diag_full_k32_20260804_merged/` | `k32_validation.json` with `valid=true`, 64 prompts, `rollouts.jsonl` with 2112 rows, support summary, cohort/provenance references |
| 64-prompt causal probe | `causal_state_probe_20260808_merged/` | `summary.json` with 64 prompts / 101 trajectories, `causal_state_records.jsonl`, `candidate_windows.csv`; exact coverage 101/101 |
| Verified teacher trajectories | `proposal_feasibility_20260805_merged/` | `retained_proposals.jsonl`; exact token IDs/hash, `correct=true`, retained trace identity |
| Existing causal rescue gold | `prefix_intervention_20260806_merged/` | `rescue_comparisons.jsonl`, `minimal_rescue_prefixes.jsonl`, `intervention_units.jsonl`; 12 prompts, 7 preregistered rescue positives |

Run this preflight on the HPC host before implementation:

```bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/support_aware_opd

test -r "$ROOT/diag_full_20260802_256_merged.rescored-exact-v1/summary.json"
test -r "$ROOT/diag_full_k32_20260804_merged/k32_validation.json"
test -r "$ROOT/causal_state_probe_20260808_merged/causal_state_records.jsonl"
test -r "$ROOT/proposal_feasibility_20260805_merged/retained_proposals.jsonl"
test -r "$ROOT/prefix_intervention_20260806_merged/rescue_comparisons.jsonl"
test -r "$ROOT/prefix_intervention_20260806_merged/minimal_rescue_prefixes.jsonl"
test -r "$ROOT/prefix_intervention_20260806_merged/intervention_units.jsonl"
```

Then record, without copying raw outputs into Git:

- absolute resolved paths;
- file sizes, line counts, and SHA256;
- source manifests, repo commit/dirty status, model IDs, tokenizer hash, and
  dataset manifest hashes;
- exact prompt/trajectory/work-ID coverage and duplicate/missing join keys;
- whether the documented hashes in the dated result reports still match.

Any missing file, hash mismatch, duplicate key, or tokenizer mismatch is a
hard stop.  Do not regenerate data silently and do not substitute a similarly
named directory.

## 4. Gold label: only teacher-prefix transport

The primary causal gold is `G_transport`:

```text
Place the frozen student after an answer-free verified teacher prefix.
Can the student continue into a verifier-confirmed correct outcome?
```

For prompt `x`, define horizon `h` as the first `h` response tokens of the
verified teacher trajectory.  Then:

```text
q_x(h) = P_student(R=1 | verified teacher tokens y^T_<h)
```

For existing Experiment-B records, define exactly:

```text
rescue_gold := bool(rescue_row["meets_preregistered_rescue_rule"])
h*          := the first increasing horizon whose rescue_gold is true
```

The stored rule already requires teacher-vs-wrong and teacher-vs-unaided
posterior mean/probability thresholds.  Do not recompute it with a new
threshold after seeing proxy values.  `minimal_rescue_prefixes.jsonl` must
reproduce the same `h*` rows.

Keep the other variables in secondary roles:

- `D_V`: optional mechanism/incremental analysis only;
- `G_relay`: future on-policy intervention work only;
- previous gap/OT localization: recorded negative evidence, not a primary
  proxy and not a blocker for this task.

## 5. Exactly three first-round proxies

Compare only these scalar signals:

1. relative position;
2. cumulative student NLL on the verified teacher path;
3. cumulative teacher-Top-100 forward KL with one tail bucket.

No MLP, router, feature fusion, visual-JS primary score, entropy grid, or new
learned classifier is authorized.

For a verified teacher trajectory of response length `T`, score every teacher
token under the same exact rendered multimodal prompt and prefix state.

### 5.1 Position

At horizon `h`:

```text
position(h) = min(h, T) / T
```

### 5.2 Cumulative student NLL

```text
nll_t = -log p_S(y^T_t | image, question, y^T_<t)
cum_student_nll(h) = sum_{t < min(h,T)} nll_t
```

Use exact teacher response token IDs.  Never decode and re-tokenize the trace.

### 5.3 Cumulative Top-100 FKL + tail

At token `t`, let `K_t` be the teacher Top-100 token IDs, `q_i` the teacher
probabilities, and `p_i` the student probabilities gathered at the same IDs.

```text
q_tail = 1 - sum_i q_i
p_tail = 1 - sum_i p_i

d_t = sum_i q_i * (log q_i - log p_i)
    + q_tail * (log q_tail - log p_tail)

cum_top100_fkl_tail(h) = sum_{t < min(h,T)} d_t
```

This is a coarse Top-100-plus-one-tail-bucket KL, not exact full-vocabulary
KL.  Preserve that name in reports.  Clamp only for numerical safety and
record epsilon; do not renormalize away either tail.  Assert that each model's
Top-100 plus tail mass sums to one within tolerance.

## 6. Minimal scorer change: inference/export only

Existing components already establish feasibility:

- `fc_opd.teacher_transformers.TransformersTeacherScorer` accepts arbitrary
  `top_k`, forced-scores exact response IDs, and returns Top-K ids/log-probs,
  sampled-token log-probs, and teacher tail log-prob;
- `fc_opd.teacher_service` exposes `--top-k`, so `--top-k 100` is supported;
- `fc_opd.student_scorer.StudentScorer` produces response-position logits and
  exact teacher-token log-probs;
- `support_aware.causal_runtime.teacher_path_support_statistics` already
  performs one forced forward per model on verified teacher paths, but it
  currently exports only student NLL, full-vocab JS, and Top-K overlap.

Therefore implement the smallest reusable offline exporter under
`src/dual_track_opd/support_aware/`.  Prefer extending the one-forward causal
runtime with a new explicit return schema over routing through the training
hook.  Requirements:

- one student forward and one teacher forward per teacher trajectory, subject
  only to an explicit OOM-safe chunk fallback;
- exact prompt IDs, response IDs, tokenizer hash, and response hash checks;
- configurable `top_k`, fixed to 100 in the study config;
- emit per-token scalar/audit rows outside Git, then aggregate horizons;
- no autograd, optimizer, actor update, Ray training, or loss registration;
- do not alter the online student scorer guardrail in the main FC-OPD path.

The expected small repository surface is:

```text
src/dual_track_opd/support_aware/reachability_proxy.py
configs/diagnostics/reachability_proxy.yaml
scripts/hpc/run_reachability_proxy.sh
tests/support_aware/test_reachability_proxy.py
```

`reachability_proxy.py` should expose pure functions for the per-token coarse
KL, cumulative horizon aggregation, strict joins, and CSV construction, plus a
thin CLI with three phases:

```text
preflight   CPU-only input/hash/schema/join validation
score       frozen student+teacher forced-forward scoring
analyze     CPU-only CSV construction and scalar proxy evaluation
```

The launcher contains environment/path orchestration only.  The YAML carries
all input/output paths, models, `top_k=100`, horizons, epsilon, and provenance
settings.  Do not hard-code NFS paths inside Python research logic.

Add focused tests for:

- causal shift alignment at the first and last response tokens;
- teacher Top-100 IDs/log-probs and both mass-conservation checks;
- student gather on teacher IDs and student tail mass;
- sampled teacher-token NLL equivalence with direct full-vocab log-softmax;
- the coarse KL formula against a small synthetic full-vocabulary reference;
- exact horizon cumulative sums, short trajectories, and duplicate/missing
  join rejection;
- deterministic CSV ordering and manifest hashes.

## 7. First artifact: plumbing only on existing rescue records

Join keys must be explicit and one-to-one:

```text
prompt_id       := sample_uid
teacher_trace_id := stable ID from retained_proposals
horizon          := the Experiment-B teacher-prefix horizon
```

If the retained proposal lacks a prebuilt trace ID, construct and persist a
stable value from `sample_uid`, `proposal_id`, and `response_token_hash`.  Do
not use row order as identity.

Match Experiment B's teacher-trace selection exactly: among correct rows with
`retained_for_fkl=true`, select the lowest numeric `reachability_rank` for each
`sample_uid`.  The same `response_token_hash` must appear in the intervention
record.  The 64-prompt causal records may repeat that teacher path across
multiple student trajectories; deduplicate only by the explicit prompt,
proposal, and response-hash identity, never by array equality or row order.

Write raw/scalar output outside Git, for example:

```text
$DTOPD_OUTPUT_ROOT/support_aware_opd/reachability_proxy_20260814/
  proxy_token_rows.jsonl
  proxy_study.csv
  proxy_analysis.json
  resolved_config.yaml
  run_manifest.json
```

`proxy_study.csv` must contain at least exactly these primary columns:

```text
prompt_id
teacher_trace_id
horizon
rescue_gold
position
cum_student_nll
cum_top100_fkl_tail
```

Audit columns may be added after them, including teacher trace length/hash,
prefix hash, token count, rescue counts, and source file hashes.  Keep raw
Top-100 arrays in the outside-Git token JSONL, not in the CSV and not in Git.

Acceptance for this first artifact:

- all 12 existing rescue prompts join exactly once to the intended retained
  teacher trace;
- every non-skipped `(prompt,horizon)` rescue row appears exactly once;
- 7 preregistered rescue-positive prompts and their stored minimal horizons
  are reproduced without changing thresholds;
- every scalar is finite; horizons never exceed the available teacher trace;
- position and both cumulative values equal direct recomputation;
- the manifest records repo/backend/model/data/config provenance and all input
  hashes.

This 12-prompt artifact validates plumbing only.  Seven positive prompts are
not enough to select a proxy or justify training.

## 8. Expand causal gold from the existing 256-prompt pool

Do not resample another large K=32 support pool.  Reuse the completed 256-prompt
K=8 pool and select approximately:

- all 37 `rare_success` prompts;
- 30–40 of the 85 `no_correct_observed` prompts;
- 20 of the 54 `mixed_support` prompts as controls.

Target roughly 90–100 prompts.  Freeze the prompt manifest, selection seed,
strata, and all hashes before generating or scoring rescue outcomes.

For each prompt:

1. obtain a verifier-confirmed correct teacher trajectory with exact token IDs;
2. reject answer-leaking prefixes using the existing leakage contract;
3. evaluate horizons `{64,128,256,512}` with student continuation `K=4`;
4. identify the first apparent transition interval;
5. use `K=8` only around that interval for confirmation.

The target is not an arbitrary total prompt count.  Continue only until the
frozen design yields at least 30–50 scaffoldable positive prompts or the
predeclared candidate pool is exhausted.  Do not alter thresholds to reach the
target.  Record negative and skipped prompts as evidence.

## 9. Proxy evaluation and decision

Evaluate each scalar separately.  Do not fuse them.

Prompt-level task:

```text
scaffoldable vs non-scaffoldable
metrics: AUROC and AUPRC with prompt bootstrap confidence intervals
```

Within-prompt task:

```text
predict the minimal sufficient horizon h*
metrics: pairwise ranking, recall@1 horizon, exact horizon match,
         +/- one horizon-bin match, and absolute bin distance
```

Any scalar calibration/threshold must be fit on training prompts and evaluated
on disjoint prompts.  Split by `prompt_id`, never by horizon row.  Fix score
orientation and evaluation code before reading final labels.  Report position
as the mandatory cheap baseline.

Proxy discovery is a GO only when at least one scalar shows stable held-out
discrimination above chance at prompt level and useful minimal-horizon ranking.
If several qualify, select the simplest one whose paired bootstrap evidence is
not worse than the alternatives.  If Top-100 FKL or NLL does not add reliable
value over position, choose position and say so.  If none qualifies, return a
NO-GO; do not rescue the result with post-hoc features.

## 10. End-to-end execution order and stop rules

Execute in this order.  Do not skip ahead because GPUs are available.

### Phase 0 — freeze scope

- Record branch, `HEAD`, remote tracking state, and dirty paths.
- Confirm no STP/RL/SFT/FKL training process will be launched.
- Do not clean, stash, commit, or overwrite unrelated user work.

### Phase 1 — CPU/NFS preflight

- Run every §3 file/readability/hash/schema/coverage check.
- Resolve the exact 12 Experiment-B prompts and their selected teacher trace.
- Produce a preflight report with zero model loading.
- **Stop** on any missing file, ambiguous trace, hash mismatch, or duplicate
  join.  Report the blocker; do not regenerate silently.

### Phase 2 — implement and unit-test the offline exporter

- Add only the four small repository surfaces in §6.
- Run the focused tests and the existing causal-runtime/scorer tests.
- Run `git diff --check` and shell syntax checks.
- **Stop** if direct-logit equivalence, token alignment, mass conservation, or
  deterministic aggregation fails.

Expected CLI shape after implementation:

```bash
$DTOPD_PYTHON -m dual_track_opd.support_aware.reachability_proxy \
  preflight --config configs/diagnostics/reachability_proxy.yaml

CUDA_VISIBLE_DEVICES=3,4 \
$DTOPD_PYTHON -m dual_track_opd.support_aware.reachability_proxy \
  score --config configs/diagnostics/reachability_proxy.yaml

$DTOPD_PYTHON -m dual_track_opd.support_aware.reachability_proxy \
  analyze --config configs/diagnostics/reachability_proxy.yaml
```

### Phase 3 — two-prompt real scorer smoke

Before the 12-prompt run, score one short and one long existing verified trace.
Use `geo3k:1141` and `geo3k:845` unless preflight proves either identity is no
longer the Experiment-B selected trace.

The smoke must prove:

- exact tokenizer/prompt/response hashes;
- one student and one teacher forced forward per trace, or a recorded OOM-safe
  fallback with identical scalar results;
- finite NLL and Top-100+tail KL at every valid token;
- teacher and student mass residuals within tolerance;
- direct recomputation of every requested horizon aggregate;
- peak allocated/reserved GPU memory and elapsed time recorded.

### Phase 4 — existing 12-prompt plumbing study

- Score the exact Experiment-B selected trace for all 12 prompts.
- Build `proxy_study.csv` and reproduce all rescue/minimal-horizon labels.
- Treat the output as plumbing evidence only; do not select a winner from seven
  positives.
- If coverage is exact, continue directly to the preregistered expansion.

### Phase 5 — expand causal rescue gold

- Freeze the 90–100 prompt manifest from the existing 256-prompt pool before
  new teacher or student continuations.
- Reuse the existing verified-proposal and prefix-intervention modules; extend
  them only for manifest-driven selection and adaptive `K=4 -> K=8` rescue.
- Freeze the teacher proposal budget and decoding contract before generation.
  A prompt with no verified correct teacher trace is
  `teacher_trace_unavailable`, not a non-scaffoldable negative.
- Use the same answer-leakage gate, wrong-prefix control, unaided control,
  verifier, posterior rescue rule, and horizon ordering as Experiment B.
- Store all raw generations and per-token arrays outside Git.
- Stop when the frozen pool is exhausted or 30–50 scaffoldable positives have
  been obtained.  Do not change thresholds to hit the target.

### Phase 6 — score and choose the proxy

- Forced-score the frozen verified teacher traces.
- Build the expanded horizon-level CSV with the same schema as Phase 4.
- Run the three separate held-out analyses in §9.
- Return a selected scalar or explicit NO-GO, then stop before all training.

## 11. Training remains behind a separate approval gate

After proxy selection, return the evidence and stop.  Do not run training.

The next experiment, only after explicit approval, is a clean non-RL test:

```text
native-low + predicted-scaffoldable prompts
        -> selective teacher-prefix Top-100 FKL
        -> fresh no-prefix student rollout, G=8
        -> measure native frontier conversion
```

Primary outcome:

```text
0/G -> 0 < k < G
or
1/G -> reliably higher mixed support
```

If native frontier conversion fails, do not run RL.  Only a stable conversion
result justifies a later comparison of RL-only, SFT-to-RL, and
selective-FKL-to-RL.

## 12. Required CC handback

Return one compact report with:

- branch, commit, and dirty status; do not include unrelated local changes;
- explicit confirmation that STP training stayed paused and `489d61a` was
  preserved;
- NFS path/readability, counts, hashes, and provenance results for every input;
- files changed and why;
- exact unit and synthetic test commands/results;
- real two-prompt scorer smoke command, runtime, GPU memory, output path, and
  hashes;
- `proxy_study.csv` path/hash, row coverage, missing/duplicate join report, and
  a short sample with no raw Top-100 arrays;
- expanded rescue-gold cohort counts by stratum, horizons, positives,
  negatives, skips, and budget;
- separate position/NLL/Top-100-FKL metrics with prompt-bootstrap intervals;
- selected proxy or explicit NO-GO, including whether it adds value over
  position;
- an explicit statement that no loss was added and no training was launched;
- the next proposed command and resource estimate, awaiting user approval.

Do not commit datasets, raw JSONL, logits, model weights, checkpoints, caches,
or experiment output directories.  Git may contain only code, configs, tests,
small aggregate summaries, and this handoff.

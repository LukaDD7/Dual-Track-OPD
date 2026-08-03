# Frontier-Operator Causal Experiment

Status: implementation contract for CPU handoff and GPU execution.  The
post-hoc frontier analyzer is executable now; the five training arms remain a
reviewed protocol and must not be presented as implemented.

Competitive evidence and the 22-paper high-resolution matrix are in
[`trek_competitive_landscape_and_fkl_opd.md`](./trek_competitive_landscape_and_fkl_opd.md).
In particular, use its occupancy-by-divergence four-quadrant analysis when
implementing the TREK, exact-token OPD, and FKL-to-OPD arms; do not describe
the comparison as KL direction alone.

## Executive decision

The immediate project should use three primary research questions.  The old
VLM-specific RQ4 is removed from the main causal chain because it adds a second
claim—localizing a visual decision fork—before the modality-agnostic bridge
mechanism is established.  Geometry3K and Qwen3-VL remain the testbed.  A small
image-null/degraded-image check is retained only as a guardrail.

The broad claim "distillation expands support and makes later RL more
efficient" is no longer novel by itself.  TREK (arXiv:2607.05339), ReGFT
(2603.01223), PACED (2603.11178), and the sparse-to-dense reward principle
(2605.12483) already cover major parts of it.  The defensible contribution is
instead:

> Under a fixed prompt state and recorded compute budget, which supervision
> operator—verified proposal forward-KL, exact-token on-policy distillation, or
> RL—moves the policy into a gradient-bearing RL frontier, and does that
> prompt-level transition predict observed early-RL efficiency?

This is an operator-by-state phase diagram plus mechanism evidence, not a new
support gate or a new OPD coefficient schedule.

## Research questions and falsifiable answers

### RQ1 — Operator by pre-bridge support state

Which operator creates the largest increase in the probability that a fresh
RL group is mixed, conditional on the prompt's pre-bridge support state?

- `no_correct_observed`: verified proposal forward-KL is expected to dominate
  pure exact-token OPD because student rollouts do not contain a successful
  mode for OPD to consolidate.
- `rare_success`: exact-token OPD may match or beat forward-KL when the teacher
  can locally rank or correct student-visited states.
- `mixed_support`: ordinary RL should be competitive because groups already
  contain reward variance.
- `all_correct_observed`: skip or very light consolidation should dominate in
  efficiency; additional bridge updates can reduce diversity without adding
  useful reward signal.

Falsification: if the same operator wins in every stratum after token and
teacher-compute accounting, a state-conditioned controller is unnecessary.

### RQ2 — Teacher rankability/compatibility as the OPD moderator

On prompts where both correct and incorrect student rollouts are observed, do
teacher-gap ranking and local support compatibility predict the benefit of
exact-token OPD relative to verified forward-KL?

The current run measures response-level teacher rankability.  A later token
diagnostic should add the teacher mass on the student's top-k token support,
following the "token teachability" distinction between learnable and
off-support disagreement.  Large disagreement alone is not sufficient.

Falsification: teacher-gap AUC/rank and local compatibility do not predict the
OPD-vs-FKL treatment difference, or the teacher cannot rank correct tails above
wrong modes.

### RQ3 — Frontier movement as a mediator of early-RL efficiency

Does bridge-induced change in predicted mixed-group probability explain the
number of actual mixed groups and the early-RL performance AUC?

For binary reward and an RL group of size `G`, define

```text
U_G(p) = 1 - p^G - (1-p)^G
```

This is the probability that a fresh group contains at least one success and
at least one failure.  It is an operational proxy for a group that can produce
non-zero group-relative advantage.  It is not proof that the gradient is
correct, sufficiently large, or beneficial.

Falsification occurs if any of the following hold:

- a bridge improves pass@1 but reduces `U_G` by moving prompts directly to an
  all-correct/low-diversity regime;
- predicted `U_G` does not calibrate to the observed mixed-group fraction;
- prompt-level delta `U_G` does not predict early reward gain or non-zero
  advantage tokens;
- the treatment effect remains unchanged after adding delta `U_G`, so the
  proposed mediator is not supported.

With one pilot seed, report this as a predictive mechanism check.  Use
"causal mediation" only after replicated randomized arms and sensitivity
analysis for treatment-induced confounders such as entropy and response length.

## Why VLM RQ4 is deferred

The old visual decision-fork question requires additional counterfactuals:
full image, degraded image, text-only/image-null, and ideally intervention at a
specific token or reasoning step.  It would test whether the bridge changes
visual dependence, not just whether it improves support.  That is scientifically
valuable, but it is not required to answer RQ1–RQ3 and is too broad for the
current report.

Use this staged rule:

1. Run the main bridge experiment with ordinary full-image Geometry3K prompts.
2. If an arm produces a clear frontier and early-RL effect, sample 16–32
   prompts and repeat scoring with the existing full/degraded conditions.
3. Treat "gain vanishes under image-null/degradation" as a sanity check that
   the result is not pure language leakage.
4. Promote visual localization to a separate RQ only if the operator treatment
   interacts strongly and reproducibly with visual dependence.

## Audit of the currently running k=9 diagnostic

The running job should not be restarted or semantically changed.

### Generation contract

- `diagnostic.py:1459–1481` produces one greedy response with temperature 0.
- `diagnostic.py:1483–1508` produces `K=8` stochastic responses with the config
  temperature (0.7) and top-p (0.95).
- The greedy outcome is an independent deterministic diagnostic.  It must not
  be counted as a ninth Bernoulli sample in the pass-rate posterior.
- `diagnostic.py:304–357` constructs three separate artifacts:
  `response_token_ids_raw`, display text decoded with
  `skip_special_tokens=True`, and raw audit text decoded with
  `skip_special_tokens=False`.  The helper explicitly forbids reconstructing
  action IDs from display text.  Verification may consume display text;
  scoring consumes `response_token_ids_raw`.  Therefore display decoding does
  not create the earlier exact-token alignment bug.

### Teacher and student scoring contract

- `diagnostic.py:1511–1528` submits the greedy plus eight stochastic raw token
  sequences in one teacher HTTP request.
- `support_aware/scorer.py:151–205` constructs exact-ID scoring requests under
  the same prompt/image history that the student saw.
- `fc_opd/teacher_client.py:139–203` flattens all sample-condition requests into
  one HTTP POST and then restores sample order.
- `fc_opd/teacher_transformers.py:433–444` deliberately calls `_score_one` per
  request.  This is one transport batch, not one `B=9` tensor forward.  The
  sequential backend avoids two previously observed corruptions: replicating
  the first prompt/image and slicing variable-length responses from padded
  tails.
- `diagnostic.py:1530–1537` student-scores the nine exact token sequences in one
  true batched forward.

Do not "optimize" teacher `_score_batched` before a fixture proves, for
heterogeneous response lengths and images, exact equality of response masks,
token hashes, top-k IDs, and log probabilities against `_score_one`.

## Phase 0 — Analyze the current run immediately

The new CPU-only analyzer consumes the finished run directory:

```bash
bash scripts/hpc/analyze_support_frontier.sh \
  /absolute/path/to/current_diagnostic_run \
  --group-size 8 \
  --mc-samples 20000 \
  --seed 42
```

It writes under `<run>/frontier_analysis/`:

- `frontier_summary.json`: posterior frontier mass, mean useful-group
  probability, state counts, and RL-ready count;
- `frontier_prompts.jsonl`: prompt-level posterior pass rate, 95% interval,
  expected `U_G`, interval, and RL-ready probability;
- `frontier_prompts.csv`: the same prompt records for plotting.

Implementation details are in `support_aware/frontier_analysis.py`:

- Jeffreys prior `Beta(0.5, 0.5)` avoids declaring true `p=0` from `0/8`;
- the posterior expectation is analytic:

```text
E[U_G(p)] = 1 - E[p^G] - E[(1-p)^G]
```

- deterministic Beta Monte Carlo is used only for intervals and the posterior
  probability of crossing the declared useful-group threshold;
- greedy correctness is copied for audit but excluded from all posterior
  calculations.

Observed strata for the `K=8` screening run are intentionally transparent:

| State | Observed count |
|---|---:|
| no correct observed | `c=0` |
| rare success | `c=1–2` |
| mixed support | `c=3–7` |
| all correct observed | `c=8` |

These labels are screening strata, not claims about true support.  Before
allocating expensive bridge arms, adaptively resample candidate prompts to
`K=32`, especially `c=0`, `c=1`, `c=2`, and `c=8` cases.  PACED itself notes
that `K=8` creates a hard-zero/granularity problem; the posterior reduces but
does not eliminate selection uncertainty.

## Phase 1 — Freeze a matched cohort

Create an immutable manifest keyed by `sample_uid`; never allow each arm to
resample a different prompt set.

Recommended pilot allocation:

- up to 16 prompts per confirmed stratum, 64 total;
- if a stratum has fewer than 16, include all and report the imbalance;
- reserve a disjoint held-out prompt set for transfer, rather than evaluating
  only on the bridge prompts;
- record diagnostic run ID, prompt hash, image hash, dataset manifest hash,
  checkpoint hash/path, `K`, decoding config, verifier version, and all seeds.

The current 128-prompt run is sufficient for a reportable screening figure.
It is not sufficient to assert a sharp threshold between true zero support and
rare success.

## Phase 2 — Five bridge arms

The contract is machine-readable in
`configs/experiment/frontier_operator_bridge_pilot.yaml`.

### A — No bridge

Start GRPO from the exact frozen student checkpoint.  This is the cold-RL
control.

### B — Verified proposal forward-KL (TREK-like baseline)

For every selected prompt:

1. Request four teacher proposals.
2. Verify outcomes; discard failures.
3. Compute trimmed length-normalized student NLL on verified proposals.
4. Retain the two most student-proximal proposals.
5. Run one fixed bridge budget of teacher-trajectory forward-KL/NLL.

This arm is mandatory.  Without it, the project cannot distinguish itself from
TREK and cannot test the claim that exact-token OPD is useful in a different
state regime.

### C — Exact-token OPD

Sample student trajectories, preserve raw response token IDs, teacher-score
those same IDs at the same prefixes, and optimize reverse-KL (or the repository's
declared sampled/top-k approximation).  Never decode and re-tokenize the
response for loss construction.

Run C on every stratum for causal comparison, even though the prior says it
will fail on `no_correct_observed`.  If C is only routed to favorable prompts,
operator efficacy and prompt selection become confounded.

### D — Forward-KL then exact-token OPD

Split the bridge optimizer-token budget 50/50 between B and C, recomputing
support at the midpoint.  This is the nearest baseline to PACED's
mode-coverage-then-consolidation schedule and the sparse-to-dense pipeline.

### E — State-routed bridge

Use the predeclared rule:

- no correct observed → verified proposal forward-KL;
- rare success → exact-token OPD only when teacher rankability/compatibility
  passes its gate, otherwise verified forward-KL;
- mixed support → no bridge; enter RL;
- all correct observed → skip.

E tests the proposed controller.  B–D are required to interpret it.

## Budget accounting

Equal optimizer steps are not equal compute.  Every arm must record:

- student rollout tokens;
- teacher proposal-generation tokens;
- teacher forced-scored tokens;
- verifier calls;
- student optimizer response tokens;
- wall-clock seconds;
- GPU-hours separated by student, teacher generation, teacher scoring, and RL.

Use identical bridge optimizer response tokens and optimizer steps as the main
training-budget match.  Report teacher/generation costs as separate axes rather
than hiding them in "one epoch".  For early RL, verifier calls are the primary
x-axis; generated tokens and wall-clock are secondary x-axes.

## Phase 3 — Fresh bridge-endpoint diagnostic

Do not reuse bridge-training rollouts.  From every arm, generate `K=32` fresh
stochastic responses on the matched prompts and held-out prompts with the same
temperature/top-p.  Run the analyzer, and compare any two matched runs with:

```bash
bash scripts/hpc/analyze_support_frontier.sh \
  /absolute/path/to/pre_bridge_run \
  --after-run /absolute/path/to/post_bridge_run \
  --group-size 8 \
  --mc-samples 20000 \
  --bootstrap-resamples 10000
```

This produces a transition matrix and prompt-level delta `U_G`.  Also report
pass@1, greedy accuracy, unique-response rate, duplicate rate, length,
truncation, entropy, and malformed rate.  Frontier mass without diversity and
format controls is not interpretable.

## Phase 4 — Matched early GRPO

From every arm checkpoint, run identical GRPO:

- group size `G=8`;
- identical prompt schedule and seeds;
- pilot horizon 100 optimizer steps;
- checkpoints at 0, 10, 25, 50, and 100;
- fixed verifier-call budget as the primary stopping axis.

Log these group-level observables on every step:

- mixed / all-zero / all-one group fractions;
- reward mean and reward standard deviation;
- count of non-zero-advantage response tokens;
- generated response tokens and length distribution;
- policy KL and entropy;
- prompt UID so realized mixed groups can be joined to predicted `U_G`.

Primary outcome: area under held-out accuracy versus verifier calls over the
first 100 steps.  Secondary outcomes: time/verifier calls to a predeclared
threshold, final step-100 accuracy, and accuracy versus generated tokens.

## Statistical analysis

The key unit for mechanism evidence is the prompt, not the five treatment-arm
means.  Fit or bootstrap a prompt-level model such as:

```text
observed_mixed_group_rate_i
  ~ prebridge_p_i + delta_U_i + arm + response_length_i + entropy_i

early_reward_gain_i
  ~ prebridge_p_i + delta_U_i + arm + response_length_i + entropy_i
```

For the confirmatory study, add seed as a random intercept or use a
seed-stratified bootstrap.  Report how the arm coefficient changes after
adding delta `U_G`, with a bootstrap interval.  Do not use a bar chart over
five arms as mediation evidence.

## Four figures for the next report

1. Current checkpoint: count/posterior mass by observed support stratum, with
   `K=8` uncertainty shown.
2. Teacher diagnostic: correct-vs-wrong teacher-gap AUC and correct-tail rank,
   split by response length/truncation.
3. Pilot bridge: pre→post transition heatmap plus delta posterior frontier mass
   for B/C/D/E.
4. Early RL: predicted `U_G` calibration against actual mixed-group fraction,
   and accuracy AUC versus verifier calls.

If training arms cannot finish before the report, figures 1–2 plus the
pre-registered five-arm protocol are still honest preliminary results.  Do not
replace absent bridge data with a method claim.

## Claude Code CPU→GPU handoff checklist

On the CPU/network instance:

```bash
git fetch origin
git switch codex/frontier-mass-terk
git pull --ff-only
python -m pip install -e '.[dev]'
pytest -q tests/test_support_frontier_analysis.py
```

Then give Claude Code this constrained task:

1. Read this document and the protocol YAML completely.
2. Run the analyzer on the finished diagnostic; do not edit its raw outputs.
3. Return the exact artifact paths and a JSON summary before implementing any
   training arm.
4. Audit the current verl launcher's objective and reward paths against arms
   A–E; list missing components by file/line.
5. Implement one arm per branch with a synthetic/unit fixture, then a 2-prompt
   smoke run, before launching the cohort.
6. Never modify `third_party/verl` directly; place minimal explained patches in
   `patches/verl/` and keep research logic under `src/dual_track_opd/`.
7. Require the same prompt manifest and budget ledger for every arm.
8. Stop on token-hash mismatch, UID mismatch, missing image, non-finite score,
   malformed-rate gate failure, or resume-config mismatch.

This order separates research decisions from framework plumbing.  Claude Code
should not be asked to invent the experimental definition while also debugging
the trainer.

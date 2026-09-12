# CC Handback: Reachability-Proxy Study, Phase 0–6 Evidence and Decision

Date: 2026-08-15

> **Evidence handback, not current launch authority.** The user's subsequent
> ChatGPT convergence separates handoff viability from native reachability and
> requires a corrected, prefix-level proxy evaluation.  The prompt-level GO
> and draft Phase-7 command below are therefore not approval to train.  Follow
> `docs/cc_handoff_viability_proxy_adjustment_20260815.md` next.

Execution brief: `docs/cc_reachability_proxy_convergence_handoff_20260814.md`
(user–ChatGPT convergence).  This handback reports the completed offline
reachability-proxy work and stops at the training gate.

## 1. Repository status

- Branch: `codex/va-opd`
- HEAD: `ed6c934d33867194158c44d1ede4f55e6a26b58b` (all Phase 0–6 code
  committed; unrelated pre-existing dirty paths left untouched)
- Remote tracking: `origin/codex/va-opd`
- STP canary / four-arm pilot / all training: **paused**; STP implementation
  preserved at `489d61a` and never deleted, reverted, or extended by this work.

## 2. NFS preflight (Phase 1)

All 14 canonical inputs under
`/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/support_aware_opd/`
were verified readable on the HPC host; SHA256, line counts, and coverage were
recorded in
`reachability_proxy_20260814/preflight_report.json`.  Verified facts:

- 256-pool: 2304 rollout rows / 256 prompts; prompt support 256; frontier 256
  (strata: rare_success 37, no_correct_observed 85, mixed_support 54).
- K32 confirmation: `valid=true`, 64 prompts, 2112 rollout rows.
- Causal probe: 64 prompts / 101 records (53 correct / 48 wrong), unique
  trajectory IDs.
- Retained teacher proposals: 22/22 correct and `retained_for_fkl`.
- Experiment-B rescue: 12 prompts / 46 rows; 7 preregistered positives.
- Fresh SHA256s matched every hash documented in the causal probe
  `run_manifest.json` (`cohort`, `k32_rollouts`, `k32_support_summary`,
  `k32_validation`, `retained_proposals`).

No missing file, hash mismatch, duplicate join key, or tokenizer mismatch was
observed; no data was regenerated.

## 3. Implementation surface (Phase 2)

Inference/export only; no loss, no autograd, no optimizer, no router change,
no training launch.

- `src/dual_track_opd/support_aware/reachability_proxy.py` — three-phase CLI
  (`preflight` / `score` / `analyze`), pure scalar math (cumulative student
  NLL, coarse teacher Top-100 FKL + one tail bucket, position), strict joins,
  deterministic CSV, descriptive + held-out evaluation, resumable and
  lock-protected scoring, multi-source combined analysis.
- `src/dual_track_opd/support_aware/reachability_expansion.py` — frozen
  88-prompt manifest from the 256 pool (28 rare_success + 40
  no_correct_observed + 20 mixed_support controls; 12 already-rescued prompts
  excluded), deterministic and hashed.
- `src/dual_track_opd/support_aware/reachability_wrong_controls.py` —
  student-only wrong-control rollout generation for prompts lacking a wrong
  control.
- `proposal_feasibility.py` / `prefix_intervention.py` — pool256+manifest
  selection, full-cohort parquet support, adaptive K=4→K=8 rescue with the
  stored Experiment-B rule, deterministic shard-manifest fingerprints.
- Configs: `configs/diagnostics/reachability_proxy{,_expansion}.yaml`,
  `configs/experiment/reachability_expansion_{proposals,intervention,
  intervention_b}.yaml`, `configs/experiment/reachability_wrong_controls.yaml`.
- Launchers: `scripts/hpc/{launch_reachability_expansion,
  launch_reachability_expansion_b,prescore_reachability_expansion}.sh` and
  shard scripts.

## 4. Tests

- `tests/support_aware/` suite: **83 passed** (includes coarse-KL vs exact
  full-vocab reference, NLL gather equivalence, mass conservation, horizon
  aggregation, strict join rejection, deterministic CSV, selection rules,
  adaptive stage mixing, held-out evaluation, combined analysis, wrong-control
  targeting).
- `git diff --check` clean; all launcher scripts pass `bash -n`.

## 5. Real scorer smoke (Phase 3)

`geo3k:1141` (532 tokens) and `geo3k:845` (2099 tokens) scored on GPU:

- one student + one teacher forward per trace (`logits_to_keep=True`), no
  OOM fallback triggered;
- exact tokenizer/prompt/response hashes verified against the Experiment-B
  selected traces;
- every token NLL and Top-100+tail KL finite; teacher/student mass residuals
  within tolerance;
- horizon aggregates exactly reproducible.

## 6. Proxy study artifacts (Phase 4)

- `reachability_proxy_20260814/` — 12-prompt plumbing: 21,139 token rows,
  `proxy_study.csv` 46 rows, 7/7 minimal horizons reproduced exactly
  (`minimal_horizons_reproduced=True`).
- `reachability_proxy_20260815/` — 33-prompt expansion: 76,179 token rows,
  `proxy_study.csv` 123 rows, 17/17 reproduced.
- `reachability_proxy_combined_final/proxy_study.csv` — pooled 52 prompts,
  **194 rows, SHA256 `941a83a0fe0d1d35…`**, every non-skipped rescue row joined
  exactly once, no duplicate/missing keys, 27 positive prompts.

## 7. Expanded causal gold (Phase 5)

- Frozen manifest: 88 prompts (rare_success 28, no_correct_observed 40,
  mixed_support 20), hash `30adc94f…`.
- Teacher proposals: 52/88 prompts obtained a verified correct trace; 36
  recorded `teacher_trace_unavailable`.
- Adaptive rescue (K=4 stage-1 at {64,128,256,512}, K=8 confirmation of the
  first transition interval; stored thresholds 0.20/0.90 unchanged):
  - 33 prompts from the 256-pool wrong controls → 17 positives;
  - wrong-control generation for the remaining 19 (16 unaided draws each;
    only 7 prompts produced any wrong rollout — those students are
    already strong) → 7 rescued prompts → 3 positives;
  - 24 prompts recorded `wrong_control_unavailable`, 12 no-wrong after
    generation.
- Pooled positives by stratum: rare_success 18 / no_correct_observed 6 /
  mixed_support 3 (of 29/11/12 pooled).  h* distribution: 64→12, 128→3,
  256→7, 512→5.
- Target of 30–50 scaffoldable positives was not reached (27); thresholds
  were not moved to hit it.  Negatives and skips are recorded as evidence.

## 8. Proxy evaluation (Phase 6, 52 prompts / 27 positives)

Descriptive (full set, prompt bootstrap CI95):

| scalar | AUROC (CI95) | AUPRC |
|---|---:|---:|
| position | 0.612 (0.45–0.76) | 0.684 |
| cum_student_nll | 0.874 (0.76–0.96) | 0.889 |
| cum_top100_fkl_tail | 0.858 (0.74–0.95) | 0.881 |

Held-out (seed 20260815, 31 train / 21 test prompts; thresholds fit on train
only):

| scalar | test AUROC | test AUPRC | within r@1 / ±1bin / dist (n=10) |
|---|---:|---:|---:|
| position | 0.618 | 0.668 | 0.50 / 0.70 / 0.90 |
| cum_student_nll | 0.873 | 0.870 | 0.00 / 0.50 / 1.50 |
| cum_top100_fkl_tail | 0.855 | 0.860 | 0.30 / 0.50 / 1.30 |

## 9. Decision (per §9)

**GO, prompt level:** cumulative student NLL (and its statistical twin,
teacher-Top-100 FKL+tail) shows stable held-out discrimination above chance
and clearly adds reliable value over position (position's descriptive CI
includes 0.5; held-out 0.618 vs 0.87/0.86).  Selected scalar:
**cum_student_nll** (simplest; FKL equivalent and reported alongside).

**NO-GO, within-prompt horizon ranking:** no scalar reliably predicts the
minimal sufficient horizon h* at this sample size (metrics are unstable and
flip across splits).  Deployment must not yet auto-select a per-prompt
horizon; use a conservative fixed horizon (e.g., the pooled h* median/mode)
until more gold is collected.

## 10. Training gate

No loss was added and no training was launched at any point.  The next
experiment — native-low + predicted-scaffoldable prompts → selective
teacher-prefix Top-100 FKL → fresh no-prefix student rollout (G=8) →
measure 0/G→0<k<G or 1/G→higher mixed support — requires explicit user
approval before execution.  RL remains out of scope unless that conversion
is stable.

## 11. Next proposed command and resource estimate (awaiting approval)

Draft (not launched): select ~30 predicted-scaffoldable prompts from the
frozen gold or a new native-low pool by held-out NLL threshold; forced-score
their verified teacher traces with the existing exporter; run G=8 fresh
no-prefix student rollouts; measure frontier conversion.

```bash
# Phase-7 (draft): native frontier conversion test
$DTOPD_PYTHON -m dual_track_opd.support_aware.reachability_proxy \
  score --config configs/diagnostics/reachability_proxy_native_conversion.yaml \
  --prompt-uids <predicted-scaffoldable-uids>
# conversion module/config to be added after approval
$DTOPD_PYTHON -m dual_track_opd.support_aware.native_conversion run \
  --config configs/experiment/reachability_native_conversion.yaml
```

Resource estimate: ~30 prompts × (one 32B teacher forward + G=8 4B student
rollouts) ≈ **2–4 GPU-hours on one teacher/student GPU pair**; the exact
protocol (selection seed, horizon policy, G, verifier, conversion metric)
will be frozen before launch and reported for approval.

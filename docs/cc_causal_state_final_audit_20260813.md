# Causal-State Probe Final Audit — 101/101 Units (2026-08-13)

Phase A evidence audit for `docs/cc_causal_state_to_stp_handoff_20260813.md`.
Raw outputs stay outside Git; this report records provenance, coverage, the
predeclared relay/transport views, and the evidence verdict.

## 1. Provenance

All four shards were produced by the same code identity and runtime:

- repo commit: `ab7d054374592888e43ba7056b247369d25ab35e`, **dirty worktree**
  for every shard (uncommitted local fixes at run time); the merged manifest
  records `git_dirty: true` plus the four `dirty_shards` paths explicitly.
- backend: `transformers-5.12.0`, `torch 2.11.0+cu129`, python `3.12.13`,
  CUDA runtime `12.9`, env `va-opd-qwen35-cu128`.
- tokenizer hash: `b5005feade74589def61d27a576ef874392651e2c92c48f26bbdab1c4cad0192`
  (student == teacher; matches K=32/proposal evidence).
- shard GPU pairs: s0=0,1 / s1=2,3 / s2=4,5 / s3=6,7; each shard ran one
  student (Qwen3-VL-4B-Instruct, `cuda:0`) + one teacher
  (Qwen3-VL-32B-Instruct, `cuda:1`).

Immutable input hashes (identical across all four shards):

| input | sha256 |
|---|---|
| k32 validation | `1616678c2346cdd2e23aea72cbb040997713b6d3404c8dc2ea6630146f498c8a` |
| k32 rollouts | `ab964d6d98cdd30042875175587f37944342596c2bf0985e62262eaf31665e47` |
| k32 support summary | `947fefecaab2501a903f1aa6519cf3a4b4fd0cb26ea709af3655b9a6fbca4768` |
| cohort | `8d0db6a05f07453b5d4eb80fd076c42e2f2b8ee73d90ca421002f51f30b12f16` |
| retained proposals | `7f2745099b4882d0b0254b323ad98324ec9f4cad64128a5aa01792caca42f7c2` |

## 2. Work-ID coverage

- expected work IDs: **101**
- present trajectory JSON: **101**
- missing: **0**; stray (present but not expected): **0**; duplicates: **0**
- shard split: s0=27, s1=28, s2=23, s3=23; every shard manifest
  `status=completed`.

## 3. Counts and state classes

- prompts: 64; trajectories: 101 (correct 53 / wrong 48); candidates: 447
  (fixed 0.20/0.50/0.80 x101 each, high_visual_dependence x101,
  post_visual_low x101, visual_dependence_drop 143).
- state classes: `insufficient_evidence 335 (74.9%)`,
  `on_policy_repairable 70 (15.7%)`,
  `transportable_low_reachability 17 (3.8%)`,
  `unresolved_under_current_intervention 25 (5.6%)`.

## 4. Relay results (L32/L64/L128, three predeclared views)

ITT lower bound (original K; malformed/leakage counted as failures):

| length | mean | median | p90 | max | positive |
|---|---:|---:|---:|---:|---:|
| L32 | +0.0103 | 0.0002 | 0.1125 | 0.7785 | 0.539 |
| L64 | −0.0010 | 0.0002 | 0.2207 | 0.7788 | 0.553 |
| L128 | −0.0380 | 0.0000 | 0.2210 | 0.8875 | 0.517 |

Cross-length sign consistency: all 447 candidates 0.383; pair-clean
(treatment + control both clean) n=98, 0.316.

Continuation-level coverage (n=447 estimates per length):

| length | all-clean estimate rate | continuation-valid rate | fully-malformed |
|---|---:|---:|---:|
| L32 | 0.4855 | 0.7027 | 50 |
| L64 | 0.4295 | 0.6625 | 59 |
| L128 | 0.3221 | 0.5861 | 78 |

`n_malformed` distribution (0..8): L32 `{0:217,1:45,2:27,3:16,4:18,5:25,6:24,7:25,8:50}`,
L64 `{0:192,1:36,2:27,3:35,4:26,5:23,6:22,7:27,8:59}`,
L128 `{0:144,1:45,2:35,3:30,4:27,5:20,6:33,7:35,8:78}`.

Pair-clean (treatment + matched control both clean):
L32 0.4139, L64 0.3736, L128 0.2617 (control-clean rate 0.5078).

Conditional sensitivity (answer-free, pair-clean):
L32 n=185 mean +0.0199; L64 n=167 mean +0.0254; L128 n=117 mean +0.0342.
Descriptive answer-free pass rate (effective denominator `n - n_malformed`):
L32 0.5583 (2513 continuations), L64 0.5732 (2369), L128 0.5768 (2096).

Anchor-position stratification shows the L128 late-position penalty:
late (rel>=2/3) L128 continuation-valid 0.3896, mean gain −0.2353 vs early
0.6292 / +0.0446 — consistent with longer relays exposing the answer.

## 5. Relay reason counts

All 101 records predate the reason-split schema (commit `ef3f2f3` added the
fields, but no record written before it carries them), so per-unit reason
counts are not recoverable.  A stratified replay of the worst candidates
(`scripts/hpc/replay_causal_malformed_sample.py`, 18 samples = 6 per relay
length, each with `n_malformed >= 4`) re-ran the deterministic saved seeds and
split 138 malformed continuations as:

| relay length | malformed | answer leakage | truncated | leakage share |
|---|---:|---:|---:|---:|
| L32 | 47 | 16 | 31 | 0.340 |
| L64 | 47 | 32 | 15 | 0.681 |
| L128 | 44 | 24 | 20 | 0.545 |
| total | 138 | 72 | 66 | 0.522 |

No `generation_error` and no `no_answer_marker` outcomes were observed; the
remaining 6 replayed continuations were format-valid (4 wrong, 2 correct).
Interpretation: historical `n_malformed` is dominated by **relay answer
leakage censoring** (52%, strongest at L64) and **truncation** (48%, strongest
at L32).  Both are intervention-side censoring, not student generation
failures, so the ITT lower bound is conservative in a systematic direction and
the conditional answer-free estimate is the more descriptive view.

Caveat: the replay sample was stratified over the high-`n_malformed` tail
(`n_malformed >= 4`), so the shares are tail-weighted, not an unbiased
estimate over all 447 candidates.  The legacy leakage-or-malformed note proxy
for all candidates: L32 230/447, L64 255/447, L128 303/447.

Decision: historical units do **not** need regeneration for the relay/transport
accounting; the offline report can carry the replay-derived reason split as a
documented sensitivity stratum instead of a per-unit `legacy_unknown`.

## 6. Transport

- transport gain: n=102, mean 0.2658, median 0.1108, positive 0.686.
- transport-vs-wrong gain: n=83, mean 0.2316, median 0.0011, positive 0.687.
- answer-leakage pass rate (candidate level): n=102, mean 0.4436, positive 0.461.
- wrong-prefix skips: `source_shorter_than_teacher_prefix` 19.

## 7. Strongest actionable candidates (predeclared thresholds)

`on_policy_repairable` (relay gain > 0.2, p > 0.85): top examples include
`geo3k:2008:rollout-4:candidate-2` (gain 0.888, p 1.0, rel 0.348),
`geo3k:855:rollout-2:candidate-0` (0.779, rel 0.019), and
`geo3k:1025:rollout-7:candidate-1` (0.778, rel 0.039).

`transportable_low_reachability` (transport gain > 0.2): 17 candidates; full
top-10 tables are in the precheck report.

## 8. Implementation strata

The runtime added `metadata.implementation_version="single_forward_v2"` after
these records were written, so **no record carries the field**.  Strata here
use file-mtime heuristic (cutoff `1786252400.0`): old 28 / new 73.  This is
labelled heuristic and is NOT stable provenance; candidate-selection stability
across strata is therefore **unverified**.  A code-version field will be
written by any future run.

## 9. GPU memory

`torch.cuda.max_memory_allocated/reserved` for the single-forward
implementation were **not measured** on this run (review P1-3).  The
single-forward retention estimate (~3.5 GiB visual response logits, ~2.3 GiB
teacher-path on student GPU) remains theoretical; this item is **unsupported**.

## 10. Raw-output paths and hashes (outside Git)

`/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/support_aware_opd/`

| artifact | sha256 (head) | size |
|---|---|---:|
| `causal_state_probe_20260808_merged/summary.json` | `6166e5c42df7c256` | 2180 |
| `causal_state_probe_20260808_merged/candidate_windows.csv` | `91afd42719feba45` | 263950 |
| `causal_state_probe_20260808_merged/causal_state_records.jsonl` | `d0925a2d649946f6` | 58167437 |
| `causal_state_probe_20260808_merged/trajectory_overview.svg` | `ac44d39d7c9a7f33` | 398521 |
| `causal_malformed_replay_20260813/replay_result.json` | (recorded in the run) | — |

Per-shard trajectory JSON: `causal_state_probe_20260808_s{0..3}/trajectory_results/*.json`
(27/28/23/23 files).

## 11. Evidence verdict

- **Final**: work-ID coverage (101/101, no missing/stray/duplicate), provenance
  hashes, counts, continuation-valid coverage, ITT lower-bound relay gains,
  transport gains, pair-clean conditional sensitivity, anchor stratification.
- **Provisional**: replay-derived reason shares (18-sample tail-weighted
  stratum; supports the leakage-vs-truncation split globally but not per unit),
  old/new strata stability (mtime heuristic only), state-class counts for
  candidates whose estimate is not pair-clean.
- **Unsupported**: GPU memory measurement (P1-3).

STP-OPD training readiness is **NOT** claimed by this audit; the handoff's own
P0 gates govern any pilot launch.

# CC Handoff: Causal-State Final Audit → STP-OPD Mechanics Pilot

> **Superseded by user research decision (2026-08-14).**  Preserve the
> implementation and historical evidence in this document, but do not execute
> its canary, four-arm pilot, or training phases.  Follow the user–ChatGPT
> convergence brief in
> `docs/cc_reachability_proxy_convergence_handoff_20260814.md`; this older
> Claude Code handoff is implementation history, not research direction.

Date: 2026-08-13

Starting commit: `ef3f2f3`

Branch: `codex/va-opd`

## 0. Mandate and launch decision

The 2026-08-08 causal-state probe reports 101/101 completed work units
(`s0=27`, `s1=28`, `s2=23`, `s3=23`) and a successful merge.  That establishes
output coverage, not readiness for an STP-OPD full training run.

Current decision:

- **NO-GO for STP-OPD full training.**
- **GO only for implementation work and the gated 7-prompt mechanics pilot**
  after all P0 gates below pass.
- Do not use a teacher-only router in the main method.  The online student
  scorer and teacher-vs-student sensitivity comparison are mandatory.
- Do not launch any training automatically merely because tests pass.  Report
  the completed P0 gates and proposed command first; obtain user approval for
  the GPU launch.

This handoff supersedes stale progress text in
`docs/causal_state_probe_run_report_20260808.md` that still says 30/101.  Do
not rewrite historical raw outputs or commit raw trajectory JSONL.

## 1. Required execution order

Complete these phases in order.  A failure in one phase blocks later phases.

1. Final 101-unit evidence audit and compact report.
2. Causal-probe resume/precheck correctness fixes.
3. Differentiable STP-OPD objective and real training integration.
4. CPU/unit/integration verification on a clean commit.
5. Per-arm real-image 4-step canary.
6. User-approved 7-prompt, four-arm, 60-step mechanics pilot.
7. Only if the registered P1 mechanics gate passes, propose cohort expansion.

## 2. Phase A — final 101-unit evidence audit

The final merged/precheck outputs currently live outside Git.  Add a compact,
reviewable report under `docs/`; do not add raw records, datasets, logits,
checkpoints, model files, or full output directories.

The report must include:

- exact repo commit and dirty state for all four input shards;
- backend/runtime versions and tokenizer hash;
- input manifest hashes and merged summary hash;
- exact work-ID coverage: expected, present, missing, stray, duplicates;
- prompt/trajectory/candidate counts and state-class counts;
- relay results at L32/L64/L128 under all three views:
  - ITT lower bound with original K;
  - continuation-valid coverage and malformed distribution;
  - treatment/control pair-clean conditional sensitivity;
- relay reason counts for new records and an explicit `legacy_unknown` stratum
  for old records that cannot supply reason-level evidence;
- transport and transport-vs-wrong results;
- strongest relay/transport candidates with predeclared thresholds;
- old/new implementation strata and an explicit statement about whether
  candidate selection/state classes are stable across strata;
- long-trajectory `torch.cuda.max_memory_allocated` and
  `torch.cuda.max_memory_reserved` for the single-forward implementation;
- raw-output paths outside Git plus SHA256 hashes;
- an explicit evidence verdict: which claims are final, provisional, or
  unsupported.

Use `metadata.implementation_version` when present.  File mtime may be used
only for legacy records and must be labelled heuristic.  Do not silently pool
legacy chunked-forward records with `single_forward_v2` records.

If the final compact report cannot be reconstructed from available outputs,
stop and report the missing paths/files.  Do not infer the final experimental
numbers from the commit message or the earlier 1-unit report.

## 3. Phase B — causal-probe correctness fixes

### 3.1 Stable work-slice assignment

Current code filters completed items and then partitions the shrinking pending
list.  That can change ownership if concurrent runners start at different
times or a sliced run is resumed after partial completion.

Required behavior:

- derive slice ownership from the immutable full selected input set or a
  stable hash of `_work_id(item)`;
- only then remove already completed work IDs;
- the same work ID must map to the same slice for every restart;
- union of slices must equal the immutable expected set;
- pairwise slice intersections must be empty;
- completion order or staggered runner start must not change ownership.

Add tests for initial launch, staggered launch, partial completion, and repeated
resume.  Do not rely only on partitioning one static pending list.

### 3.2 Manifest/precheck validation

Fix `scripts/hpc/precheck_causal_state_probe.py` so that:

- top-level `manifest["git_commit"]` is compared across shards;
- top-level `git_dirty`, schema version, and completion status are checked and
  surfaced explicitly;
- config normalization removes only shard/output/work-slice identity fields;
- all required immutable input hashes, tokenizer hash, backend version, and
  model identity are compared;
- exact work-ID coverage is required before a result is described as final;
- reason-count identities include consistency with `n_malformed`;
- `metadata.implementation_version` is used before mtime fallback;
- the report header and strongest-candidate thresholds match the actual code;
- validation failures produce a non-zero exit status.

Dirty historical shards may be merged only when their dirty provenance is
recorded and the code identity is explicitly reconciled.  Do not relabel a
dirty merge as clean.

### 3.3 Resume safety

Preserve the useful resume checks already present locally/upstream:

- refuse trajectory JSON files without a run manifest;
- reject duplicate, unexpected, schema-mismatched, or immutable-evidence-
  mismatched records;
- validate before overwriting `resolved_config.yaml` or the manifest;
- keep model identity and runtime compatibility checks;
- preserve atomic per-process temporary writes.

Resume compatibility with the historical `ab7d054 + dirty` run must remain an
explicit legacy policy, not an accidental disabling of all code-identity
checks.

## 4. Phase C — real STP-OPD training integration

`src/dual_track_opd/support_aware/prefix_scaffold.py` currently contains only
CPU-testable primitives.  It is not a trainable implementation.

### 4.1 Differentiable loss contract

Required objective:

```text
L = lambda_P * L_prefix_FKL
  + lambda_D * L_suffix_teacher_RKL_K1
  + lambda_R * L_suffix_task_GRPO
```

Requirements:

- `masked_mean` and the composed objective must return tensors and preserve
  autograd; never convert trainable loss tensors to Python `float`;
- prefix, suffix-distillation, and suffix-PG regions must be disjoint;
- each non-empty region is normalized by its own valid-token count;
- prefix length must not implicitly rescale another loss region;
- empty regions are handled without NaN and without silent gradient loss;
- invalid teacher-token alignment is excluded using the existing valid mask;
- fixed teacher-prefix tokens are excluded from PG likelihood ratios;
- suffix is sampled by the current online student;
- teacher and online student score the same exact hybrid token trajectory;
- tokenizer/hash mismatch fails before optimization;
- the main path retains the online student scorer.  A teacher-only route may
  exist only as an explicitly named ablation.

Add gradient tests proving that the intended prefix and suffix parameters
receive finite, non-zero gradients and masked regions receive none.

### 4.2 Scaffold schedule and paired batching

The registered scaffold schedule is:

- steps 1–20: 75%;
- steps 21–40: 50%;
- steps 41–60: 25%;
- after step 60: 0%.

`paired_batch()` currently computes assignment flags but does not use them.
Define one unambiguous batch contract and test it statistically and exactly.
Every comparison must retain prompt pairing while the realized scaffold share
follows the schedule.  Record per-step scaffolded/unscaffolded prompt and token
counts in metrics and the resolved manifest.

### 4.3 Required repository artifacts

Keep research logic under `src/dual_track_opd/` and backend changes minimal.
At minimum provide:

- a real STP-OPD integration module under
  `src/dual_track_opd/support_aware/`;
- `src/dual_track_opd/support_aware/support_transition_eval.py`;
- `configs/experiment/support_transition_prefix_opd_pilot.yaml`;
- `configs/experiment/support_transition_eval.yaml`;
- an HPC smoke/pilot launcher that contains orchestration only;
- the smallest necessary backend patch under `patches/verl/`, with rationale;
- tests for masks, gradients, normalization, exact-token identity, schedule,
  pairing, online student scoring, resume, and manifests.

Do not vendor or directly rewrite `third_party/verl/`.

## 5. Phase D — four matched arms

The mechanics pilot uses the existing seven rescue-positive rare-support
prompts.  All arms must use the same prompt manifest, seed grid, decoding
contract, optimizer steps, LR, task reward, validation protocol, and matched
generated-token budget.

| Arm | Prefix region | Suffix region |
|---|---|---|
| A0 no-prefix OPD | none | online teacher RKL/K1 + GRPO |
| A1 PrefixRL-style | context only; fully masked | GRPO |
| A2 prefix-FKL | FKL | GRPO |
| A3 STP-OPD | FKL | online teacher RKL/K1 + GRPO |

Do not change teacher, prompt, horizon, LR, batch size, or decode parameters
between arms.  Use the frozen verified shortest answer-free horizon per prompt.
Do not tune horizons inside this pilot.

Final evaluation must remove the teacher prefix and use fresh seed grids.  The
primary question is unconditional support transition, not conditioned
continuation quality.

## 6. P0 gates before any pilot launch

All items below must pass on a clean tracked commit:

- full resolved config and repo/backend/model/data provenance recorded;
- exact prefix/suffix token hash and tokenizer mapping tests pass;
- differentiable regional objective and gradient-isolation tests pass;
- online student scorer is exercised in the main A0/A3 paths;
- schedule and paired batching tests pass;
- resume/save/load preserves global step, scheduler, optimizer, data cursor,
  arm identity, and prompt manifest;
- no NaN/Inf in synthetic integration tests;
- dry-run command prints the four resolved arm commands without GPU/model
  access;
- real-image 4-step canary passes independently for A0, A1, A2, and A3;
- each canary writes manifest, metrics, raw-output path outside Git, and hashes;
- GPU memory is measured, and no launcher depends on untracked local patches.

If any P0 item fails, fix and rerun it.  Do not compensate by launching a
larger job.

## 7. P1 mechanics gate

After P0 and explicit user approval, run only the 7-prompt × 4-arm × 60-step
pilot.

The A3 arm passes the mechanics gate only if:

- at least 4/7 prompts improve in fresh no-prefix Jeffreys posterior mean;
- A3 minus A0 prompt-macro mean lift is at least `+0.10`;
- losses, gradients, entropy, rewards, valid-mask ratio, and response metrics
  remain finite;
- the result is not explained by different token budgets, truncation, missing
  answers, or scaffold exposure;
- exact resume/provenance checks pass;
- no-prefix gains are reported under original, degraded, shuffled, and
  no-image conditions.

Passing P1 authorizes a proposal for cohort expansion; it does not authorize a
full experiment automatically.  Failing P1 means diagnose the prefix-FKL,
suffix-RKL, and scaffold-removal mechanisms before spending more compute.

## 8. Verification commands

At minimum run:

```bash
git status --short
git diff --check
python -m compileall -q src scripts tests
pytest -q tests/support_aware/
pytest -q tests/fc_opd/
pytest -q
```

Also run shell syntax checks for every changed launcher and the project-specific
dry-run/preflight commands added by the implementation.

If a test cannot run locally because the macOS environment lacks research
dependencies, run it in the pinned HPC environment and record the exact Python
path, versions, command, exit code, and log path.  Do not report upstream or
historical test counts as verification of newly changed code.

## 9. Required CC handback

Return one compact report containing:

- commit(s), branch, and dirty status;
- files changed and why;
- final 101-unit evidence summary and hash/path provenance;
- tests and preflights with exact pass/fail counts;
- unresolved review findings, if any;
- the four resolved canary/pilot commands;
- per-arm canary results, output paths, and hashes;
- an explicit `P0 PASS` or `P0 FAIL`;
- if P0 passes, the proposed 7-prompt pilot command and expected resource/time
  budget, awaiting user approval;
- no raw data, model weights, checkpoints, raw JSONL, or secrets in Git.

Do not claim training readiness merely because the causal probe completed or
because a baseline VA/OPD canary passed.  STP-OPD readiness requires this
handoff's own P0 gates.

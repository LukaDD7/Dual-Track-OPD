# NLL-TailOPD v1 Backend Handoff

Date: 2026-09-18

Repository branch: `exp/nll-tailopd-v1`

Purpose: continue the NLL-TailOPD v1 normalization work on the active server
without treating local repository checks as proof of external VERL backend
reproducibility.

## Semantic contract

```text
old_log_probs [B,T]
  -> masked mean rollout NLL [B]
  -> per-uid group z-score
  -> per-group softmax weights
  -> detached K*w rollout scale [B]
  -> scale[:, None] * distillation loss [B,T]
  -> policy-gradient advantage and backward
```

V1 is rollout-level NLL weighting only. Do not add correctness routing,
token-level weighting, teacher gating, or a new distillation estimator.

Core invariants are recorded in
`configs/backend/verl_qwen35_v090_cu132.yaml`. In particular:

- weights sum to one within each uid group;
- uniform scores recover `weights=1/K`, `scale=1`, and Vanilla OPD loss;
- scores, weights, and scales are detached;
- padding does not affect rollout scores;
- every formal-run uid group contains exactly configured `ROLLOUT_N` siblings.

## VERIFIED LOCALLY

- `src/dual_track_opd/tail_opd/weights.py` implements masked rollout NLL,
  per-group normalization, softmax weighting, detached `K*w`, and diagnostics.
- `validate_fixed_group_size()` provides the fail-fast interface that the
  backend integration must call with configured `ROLLOUT_N`.
- Group isolation is tested with non-contiguous ids and an adversarial change
  to a different prompt group.
- Uniform recovery is tested through the broadcasted `[B,T]` distillation
  tensor, not only the scalar rollout scale.
- Zero valid tokens, non-finite valid scores, invalid temperature, invalid
  epsilon, single-rollout groups, and arbitrary core group sizes are covered.
- `tail_opd/weight_entropy` is the mean of per-group entropies and is invariant
  to changing the number of uniform K=4 prompt groups from 2 to 8.
- The local backend-contract verifier checks schema, repository-relative
  references, patch existence and SHA-256, runtime/config path consistency,
  and positive `training.rollout_n`.
- Targeted result: `18 passed`.
- `git diff --check`, Python compilation, and the contract verifier pass.

The stored TailOPD patch SHA-256 at handoff time is:

```text
2737079917d7ac3ee2bbbe4a99964400525058fc2ddad26018edf2bb95663c33
```

The full local test suite was not collected because the available environment
lacks optional dependencies `lmms_eval`, `mathruler`, and external `verl`.
This is not evidence that the full repository suite passes.

## REQUIRES SERVER VERIFICATION

Nominal upstream base:

```text
https://github.com/verl-project/verl.git
483b8a009ba3a97563edee3a19887e4862b8094a
```

Active backend expected by the experiment config:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/backends/verl-qwen35-v090-cu132
```

The server agent must verify:

1. current backend HEAD and full Git status;
2. every diff relative to nominal base `483b8a00`;
3. the ordered prerequisite patch stack, including PTD-related dependencies;
4. whether any dirty diff cannot be explained by a stored patch;
5. whether a pristine worktree plus the reconstructed patch stack reproduces
   the active backend exactly;
6. the fixed-sibling validation is wired to the actual configured
   `ROLLOUT_N` before TailOPD scales are accepted;
7. TailOPD enabled/disabled backend integration behavior;
8. the three-step GPU semantic smoke.

Do not reset, clean, checkout, or apply patches to the active backend during
the initial audit.

## UNKNOWN UNTIL AUDIT

- the active backend HEAD;
- the true ordered patch stack;
- whether `483b8a00 + nll_tailopd_v1.patch` is sufficient (it likely is not,
  because the patch context contains PTD-related code);
- whether there are unexplained manual backend changes;
- whether the current backend can be reproduced exactly from stored inputs;
- whether the previously reported smoke used the exact backend state now
  active on the server.

## Server execution checklist

### 1. Read-only active-worktree audit

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/backends/verl-qwen35-v090-cu132
git rev-parse HEAD
git status --short --branch
git diff --stat 483b8a009ba3a97563edee3a19887e4862b8094a
git diff 483b8a009ba3a97563edee3a19887e4862b8094a
```

Return an attribution table mapping every changed file/hunk to a stored patch
or to `UNEXPLAINED`.

### 2. Patch-stack reconstruction

Use a separate pristine worktree or clone. Do not mutate the active backend.
Record the exact patch order and SHA-256 of every patch. Compare the rebuilt
tree with the active backend and report any remaining diff.

Only after exact reconstruction is demonstrated should these contract fields
be updated:

```yaml
verification_status: VERIFIED_ON_SERVER
patch_stack:
  status: VERIFIED_ON_SERVER
  ordered_patches:
    - path: patches/verl/<first-confirmed-patch>.patch
      sha256: <64-character-lowercase-sha256>
    - path: patches/verl/nll_tailopd_v1.patch
      sha256: <64-character-lowercase-sha256>
  reconstruction_evidence:
    audit_report: docs/<checked-in-server-audit-report>.md
    audit_report_sha256: <64-character-lowercase-sha256>
    active_backend_head: <40-character-git-commit>
    active_backend_tree: <40-character-git-tree>
    reconstructed_backend_tree: <same-40-character-git-tree>
    clean_reconstruction_diff: true
```

The local verifier accepts both contract states. In
`TO_BE_CONFIRMED_ON_SERVER`, `ordered_patches` and
`reconstruction_evidence` must both be `null` and the server checks are
reported as `SKIP`. In `VERIFIED_ON_SERVER`, every ordered patch and the audit
report are SHA-256 checked, both tree SHAs must match, and the clean-diff flag
must be true.

### 3. Backend integration contract

Verify on both the legacy and v1 trainer path used by the active experiment:

- `uid`, `old_log_probs`, and `response_mask` refer to the same rollout order;
- actual uid group sizes equal configured `ROLLOUT_N`;
- scales are detached and broadcast as `[B,1]` over `[B,T]` losses;
- TailOPD scales enter the real distillation loss and policy-gradient path;
- `TAIL_OPD_ENABLED=false` is numerically identical to Vanilla OPD;
- missing/incomplete sibling groups fail before an optimizer update.

### 4. Three-step GPU semantic smoke

Passing means more than return code zero. Require all of:

```text
observed group_size == ROLLOUT_N
tail_opd/nll_std is finite and > 0 for a nonuniform batch
tail_opd/scale_mean ~= 1
weights and scales are finite and within expected bounds
distillation loss is finite
actor grad_norm is finite and > 0
three optimizer steps complete
expected checkpoint exists
```

Record the backend commit/dirty state, ordered patch hashes, full resolved
config, repo commit/dirty state, dataset manifest hash, model paths, raw log
path, checkpoint path, and summary metrics.

## Required server handback

Return these headings without converting unknowns into verified facts:

```text
BASE COMMIT
ACTIVE BACKEND HEAD
ACTIVE BACKEND STATUS
PATCH STACK IN ORDER
PATCH HASHES
UNEXPLAINED DIFFS
RECONSTRUCTION DIFF
BACKEND INTEGRATION RESULT
3-STEP SEMANTIC SMOKE RESULT
REPRODUCIBLE: YES/NO
REMAINING UNKNOWN
```

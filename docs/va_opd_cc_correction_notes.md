# VA-OPD Reproduction: Correction Notes for CC

Date: 2026-07-05

Purpose: align the GPU-side follow-up with a faithful VA-OPD reproduction, and
avoid spending time on old FC-OPD compatibility concerns that no longer drive
the current experiment.

Primary paper anchor: arXiv 2605.21924v1, Sections 3.1-3.3.

## Bottom Line

The current priority is **VA-OPD faithful reproduction**, not preserving or
debugging the old FC-OPD route. Any claim should be evaluated by whether it
affects this chain:

1. Student generates on-policy sibling rollouts under the full image and the
   canonical question.
2. Teacher forced-scores the same student rollout under full image and degraded
   image.
3. Token VA is computed from exact sampled-token teacher log-probs:
   `max(log p_T(full) - log p_T(degraded), 0)`.
4. Rollout weights are normalized within sibling rollouts of the same prompt:
   `sum_k w_k = 1`.
5. Actor optimizes grouped reverse KL against the full-image teacher target,
   with HighVA/LowVA token groups averaged separately.

## Corrections

### 1. "The 10-step GPU run proves VA-OPD is reproduced"

Verdict: **Not proven.**

Reason:

- The logged run was produced before the current correctness fixes.
- The log still prints `w^(k) = K·softmax(...), sums to K`, which is not the
  faithful rollout-weight normalization.
- The run still configured `StudentScorerClient`, although VA-OPD does not need
  a student forced scorer in the post-rollout hook.
- The teacher log still shows `batched B=8 forward`; that path was exactly one
  of the suspected VA corruption points because the old batched scoring path
  reused the first sample's prompt/image for the whole group.

Use that run only as evidence that an older training path could finish 10
steps. Do not use it as evidence that the faithful VA-OPD reproduction is
correct.

### 2. "Rollout weight sum should be approximately batch size"

Verdict: **Wrong for VA-OPD.**

Reason:

- VA-OPD normalizes rollout weights within the K sibling rollouts for the same
  prompt.
- Therefore, for each prompt group, `sum_k w_k = 1`.
- Over an actor mini-batch, `rollout_weight_sum` should be approximately the
  number of prompt groups represented in that mini-batch, not the number of
  rollout rows.

Current handling:

- `compute_rollout_va_weights()` keeps softmax weights summing to 1 per prompt.
- Comments and diagnostics that implied `K * softmax` or `sums to K` were
  corrected.

### 3. "VA-OPD dp_actor changes affect FC-OPD"

Verdict: **Not a blocker for the current track.**

Reason:

- The old FC-OPD route is not the active reproduction target.
- The only dp_actor question that matters now is whether `loss_mode=va_opd`
  receives the right tensors and computes the faithful VA-OPD objective.
- We should not complicate the VA-OPD patch just to preserve legacy FC-OPD
  behavior unless it demonstrably contaminates the VA-OPD path.

Current handling:

- Research logic remains in `src/dual_track_opd/fc_opd/verl_actor_loss.py`.
- `patches/verl/fc_opd_fsdp_actor_aux_kd.patch` stays a thin transport/logging
  patch for verl.
- Legacy forward/reverse sparse KD helpers are still available, but they are
  not the criterion for VA-OPD correctness.

### 4. "Metrics gap is the main issue"

Verdict: **Useful diagnostic issue, not the primary correctness blocker.**

Reason:

- CC correctly observed that the old actor path did not expose VA diagnostics
  such as `va/mean`, `va/sparsity`, and `rollout_weight_sum`.
- However, metrics visibility does not fix corrupted VA if teacher scoring,
  degraded image protocol, prompt equivalence, or rollout weighting are wrong.

Current handling:

- `VerlSparseKDOutput` now carries optional metrics.
- `verl_actor_loss.py` passes `compute_va_opd_loss().metrics` through.
- The verl actor patch merges those metrics into actor logs, e.g.
  `actor/fc_opd_va/mean`, `actor/fc_opd_va/sparsity`, and
  `actor/fc_opd_va_opd/rollout_weight_sum`.

### 5. "Submodule pointer update is harmless"

Verdict: **Problematic.**

Reason:

- The pulled branch moved `third_party/verl` to gitlink
  `386742ef4b2b11cbac9ac4e314996131418d6989`.
- A normal pull failed because the submodule remote could not provide that
  object.
- This violates the intended patch-based workflow for third-party code and
  makes the branch hard to reproduce.

Current handling:

- The submodule pointer was restored to the fetchable baseline.
- verl changes should be represented under `patches/verl/`, not by relying on
  an unavailable submodule commit.

## What CC Should Test Next

After pulling the latest `codex/va-opd`, apply the verl patches and run the
VA-only smoke:

```bash
git pull --ff-only
git apply --directory=third_party/verl patches/verl/fc_opd_ray_trainer_post_rollout_hook.patch
git apply --directory=third_party/verl patches/verl/fc_opd_fsdp_actor_aux_kd.patch
bash scripts/hpc/run_verl_fc_opd_smoke.sh --gpus 4 --steps 10
```

Expected log signals:

- `conditions: ['full', 'degraded']`
- no `StudentScorer` / `StudentScorerClient`
- no teacher `batched B=... forward`
- rollout formula says softmax weights sum to 1 per prompt group
- actor logs include:
  - `actor/fc_opd_va/mean`
  - `actor/fc_opd_va/sparsity`
  - `actor/fc_opd_va_opd/rollout_weight_sum`
  - `actor/fc_opd_va_opd/token_mean_loss`

If any of these fail, treat that as a VA-OPD reproduction blocker. If the only
failure is old FC-OPD compatibility, do not block the VA-OPD reproduction on it.

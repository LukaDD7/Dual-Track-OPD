# `verl` Patches

Keep minimal patch overlays for `verl` here. Each patch should explain:

- upstream commit
- reason for the patch
- how to apply it
- how to remove it when upstream support exists

## FC-OPD Online FSDP Patch Set

Upstream target:

- `verl v0.7.1`
- investigated commit: `bec9ef74768dd201881cd4e54cd0385e87caae27`

Patches:

- `fc_opd_ray_trainer_post_rollout_hook.patch`
  - Adds a configurable post-rollout hook in `RayPPOTrainer.fit` immediately
    after current student rollout responses are unioned into the batch and
    `response_mask` exists.
  - The hook FQN is read from `algorithm.fc_opd.post_rollout_hook`.
  - The hook must be project-owned code that attaches `fc_*` tensor fields to
    `DataProto.batch`; it must not read offline score JSONL.

- `fc_opd_fsdp_actor_aux_kd.patch`
  - Preserves FC-OPD tensor fields through `DataParallelPPOActor.update_policy`.
  - Computes sparse top-k forward KL/JSD-ready auxiliary loss from live actor
    logits in `_forward_micro_batch`.
  - Adds `actor/fc_opd_loss`, `actor/fc_opd_coef`, and active-weight metrics.
  - Uses `dual_track_opd.fc_opd.verl_sparse_kd.compute_verl_sparse_topk_kd`
    for the actual tensor math.

Expected tensor fields attached by the trainer hook:

```text
fc_teacher_topk_indices    [B, C, T, K] int64
fc_teacher_topk_log_probs  [B, C, T, K] float32
fc_teacher_tail_log_prob   [B, C, T]    float32, optional
fc_condition_weights       [B, C, T]    float32
fc_condition_ids           [C]          int64
```

Apply from inside the checked-out `third_party/verl` tree, or from repo root
with `--directory=third_party/verl`:

```bash
git apply --directory=third_party/verl patches/verl/fc_opd_ray_trainer_post_rollout_hook.patch
git apply --directory=third_party/verl patches/verl/fc_opd_fsdp_actor_aux_kd.patch
```

The actor patch intentionally fail-fasts for fused actor kernels because FC-OPD
requires live logits. Remove-padding without Ulysses sequence parallel is
covered; remove-padding plus Ulysses SP should be added only after a dedicated
alignment smoke.

Removal path: revert the two patches and remove `algorithm.fc_opd.*` from the
verl config. Project-side modules under `src/dual_track_opd/fc_opd/` remain
valid for diagnostics and non-verl smokes.

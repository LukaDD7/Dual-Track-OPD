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
  - Computes the actor-side auxiliary loss from live actor logits in
    `_forward_micro_batch`.
  - Supports the faithful VA-OPD path (`loss_mode=va_opd`): full/degraded
    teacher scores, exact sampled-token VA, rollout softmax weights that sum
    to 1 per prompt, and grouped reverse KL.
  - Treats `va_opd` as pure distillation in the actor update, replacing the
    zero-advantage PPO loss instead of multiplying by an auxiliary coefficient
    stored outside the actor config.
  - Adds `actor/fc_opd_loss`, `actor/fc_opd_coef`, denominator metrics, and
    VA diagnostics such as `actor/fc_opd_va/mean`.
  - Delegates tensor math to `dual_track_opd.fc_opd.verl_actor_loss` so the
    third-party patch stays thin.

Expected tensor fields attached by the trainer hook:

```text
fc_teacher_topk_indices    [B, C, T, K] int64
fc_teacher_topk_log_probs  [B, C, T, K] float32
fc_teacher_tail_log_prob   [B, C, T]    float32, optional
fc_teacher_sampled_log_probs [B, C, T]  float32, required for VA-OPD
fc_condition_weights       [B, C, T]    float32
fc_rollout_weights         [B]          float32, required for VA-OPD actor path
fc_condition_ids           [B, C]       int64, non-tensor batch
fc_opd_loss_mode           [B]          object/string, non-tensor batch
fc_prompt_ids              [B]          object/string, non-tensor batch
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

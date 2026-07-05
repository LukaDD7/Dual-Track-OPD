# VA-OPD grad_norm=0 bug: pure OPD detection lost in dp_actor patch rewrite

Date: 2026-07-05
Run: `va_opd_20260705_150640`, exit 0, 10 steps

## Symptoms

```
actor/fc_opd_loss:             15.79   ← loss computed correctly
actor/fc_opd_va_opd/loss:      15.79   ← same loss, confirmed
actor/fc_opd_coef:              0.0    ← BUG: should be 1.0 for pure distillation
actor/pg_loss:                   0.0    ← GRPO with zero advantages
actor/grad_norm:                 0.0    ← BUG: total loss = 0, no backward
```

## Root cause

The rewritten dp_actor patch (`patches/verl/fc_opd_fsdp_actor_aux_kd.patch`) removed the
`_pure_opd` detection that existed in the older inlined dp_actor code. The new code at
`third_party/verl/verl/workers/actor/dp_actor.py:740-749` reads:

```python
fc_opd_loss = fc_opd_loss_sum / denominator          # 15.79, correct
fc_opd_coef = _fc_opd_config_float(self.config, "loss_coef", 0.0)
#             ↑ reads self.config.fc_opd.loss_coef
#               BUT self.config is actor_rollout_ref.actor,
#               and loss_coef lives under algorithm.fc_opd.loss_coef
#               → returns default 0.0
policy_loss = policy_loss + (fc_opd_loss * fc_opd_coef / loss_scale_factor)
#                            ^^^^^^^^^^^^^^^^^^^^^^^^^^ = 0.0
```

The old inlined code (before the `verl_actor_loss.py` refactor) had:

```python
_pure_opd = _fc_opd_mode in ("vgg_opd", "va_opd")
if _pure_opd:
    policy_loss = fc_opd_loss       # ← correct: pure distillation, no coef
else:
    policy_loss = policy_loss + fc_opd_loss * fc_opd_coef / loss_scale_factor
```

## Why loss_coef=1.0 in hydra config doesn't reach the actor

The verl CLI passes `+algorithm.fc_opd.loss_coef=1.0`, which lands in the top-level
Hydra config under `algorithm → fc_opd → loss_coef`. But `self.config` inside
`DataParallelPPOActor` is the actor sub-config (`actor_rollout_ref.actor`), which does
NOT receive `algorithm.fc_opd`. So `self.config.get("fc_opd", {})` returns `{}` and
the default `0.0` is used.

## Fix (two options)

### Option A: Restore `_pure_opd` bypass (recommended, minimal)

In `third_party/verl/verl/workers/actor/dp_actor.py`, around line 740, add the
pure-OPD bypass before the coef multiplication:

```python
# ── Detect pure-OPD mode ──
_fc_opd_mode = "forward"
_ntb = getattr(mini_batch, "non_tensor_batch", None) or {}
if "fc_opd_loss_mode" in _ntb:
    _fc_opd_mode = str(_ntb["fc_opd_loss_mode"].flat[0])
_pure_opd = _fc_opd_mode in ("vgg_opd", "va_opd")

if fc_opd_loss_sum is not None:
    ...
    fc_opd_loss = fc_opd_loss_sum / denominator
    if _pure_opd:
        policy_loss = fc_opd_loss          # pure distillation, no GRPO, no coef
    else:
        fc_opd_coef = _fc_opd_config_float(self.config, "loss_coef", 0.0)
        policy_loss = policy_loss + (fc_opd_loss * fc_opd_coef / loss_scale_factor)
        micro_batch_metrics["actor/fc_opd_coef"] = fc_opd_coef
```

### Option B: Pipe fc_opd config into actor config

Add `+actor_rollout_ref.actor.fc_opd.loss_coef=1.0` to the run script. Less clean
because it requires the script to know about pure vs auxiliary mode.

## Verification

After fix, at step 10 expect:
- `actor/grad_norm > 0` (gradient flowing)
- `actor/fc_opd_loss` decreasing across steps
- `actor/fc_opd_coef` absent for pure OPD, or 1.0

## Secondary issue: VA ≈ 0 everywhere

```
actor/fc_opd_va/mean:      0.0
actor/fc_opd_va/sparsity:  1.0   (100% of tokens have VA ≤ 0)
```

The geometry3k dataset uses `degraded_image.transform.type = "lowres_nearest"` with
scale=0.1 on the SAME file as the full image. The nearest-neighbor 10%→100% round-trip
on line-art geometry diagrams preserves enough structure that the teacher VLM gives
nearly identical log-probs for both conditions.

This is a secondary concern — the grad_norm=0 bug must be fixed first. After that,
stronger degradation (e.g. `gaussian_blur_s2` or the new `lowres_bilinear_nearest`)
should produce non-zero VA, or re-materialize the geometry3k parquet with a different
degraded mode.

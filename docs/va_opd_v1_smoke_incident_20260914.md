# Native VA-OPD V1 smoke incident and Gate C result (2026-09-14)

## Outcome

The paper-primary Qwen3-VL-8B→2B VA-OPD 4-step smoke passed on a 4-GPU
2-actor + 2-teacher layout.

Evidence:

- Run: `fc-opd-storage/runs/va_opd_native/qwen3vl_geometry3k_native_va_opd_paper_paper_va_smoke_4gpu_fix6_20260914_081049`
- `result.json`: `passed=true`, `completed_steps=4`, `exit_code=0`
- Loss/gradient/entropy counts: 8 / 4 / 4 finite observations
  (distillation loss is logged twice per update)
- `last_distillation_loss=0.26045`
- `last_gradient_norm=15.16154`
- `last_va_mean=0.11639`
- `last_va_positive_ratio≈0.518`
- `last_va_high_token_ratio≈0.200`
- `max_group_weight_sum_error=1.19209e-7`
- final validation monitor reward: `0.24`

This is the first real-image VA-OPD smoke pass; the 2026-08-06 native canary
had passed only the OPD baseline, not the degraded-image VA path.

## Incident chain

The first paper-primary OPD smoke failed because the prepared paper dataset
used `data_source=geometry3k`, while verl's built-in reward table recognizes
`hiyouga/geometry3k`. The launcher had populated only the legacy
`custom_reward_function` config, so the resolved `reward.custom_reward_function.path`
remained `None`. The empty-key TransferQueue error that followed was a
downstream symptom of every reward failing.

Fix: pass both current and legacy reward overrides, causing
`smoke_reward.py::compute_score` to load directly.

The first VA smoke failed on image dimensions because the runtime full image
can be resized before the degraded teacher pass (for example, a persisted
640×415 pair can arrive as 640×416). The persisted full/degraded files were
correct and had passed preflight; the code was comparing the runtime image
against the persisted degraded image.

Fix: validate the persisted full/degraded pair, then nearest-resize the
prepared degraded image to the runtime full-image size when needed.

The next failure was `KeyError: va_opd_token_weights`. The first backend
integration computed weights only in the legacy V0 trainer, while the active
launcher uses `TaskRunnerV1` and TransferQueue-backed nested tensors.

Fix: add a V1 hook before sequence balancing. It retrieves the complete
sibling group, converts it to a padded `DataProto`, computes the shared VA
weights, and writes nested `va_opd_token_weights` and
`va_opd_visual_advantage` fields back to TransferQueue.

The V1 hook then reported missing `teacher_degraded_ids` and
`teacher_degraded_logprobs`. `AgentLoopOutput.as_dict()` promoted full-image
teacher fields to top-level output fields but left the degraded fields inside
`extra_fields`, so the V1 TransferQueue path dropped them.

Fix: promote degraded teacher fields in `as_dict()` as top-level fields.

Finally, exact teacher/student ID identity failed because V1 teacher tensors
may be full-sequence padded tensors with prompt-left padding and
response-right padding. Fixed-width final-response slicing therefore did not
always identify the response span.

Fix: search each teacher ID row for the exact contiguous student response
token subsequence, extract teacher values at those positions, and then enforce
the exact-ID gate. This keeps token identity strict without relying on a
layout-specific padding offset.

## Follow-up gates

1. Run the paired 50-step OPD and VA-OPD pilots on the same 4-GPU layout.
2. Monitor validation score, entropy, response clip ratio, VA mean/positive
   ratio, loss, gradient, and collapse behavior.
3. Do not launch the 5-epoch runs unless both pilots complete and remain
   numerically stable.

The 4-step entropy scale is high (`657.10` at step 4), but it is finite and
the smoke is too short to diagnose collapse. Treat it as a required pilot
monitoring item, not as evidence of failure.

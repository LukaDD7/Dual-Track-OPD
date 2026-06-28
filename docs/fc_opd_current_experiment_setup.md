# FC-OPD Current Experiment Setup

This document records the current verified FC-OPD setup after the real-student
optimizer-step smoke. It is a reproducibility checkpoint, not a verl integration
plan.

## Current Pipeline

The current FC-OPD pipeline is offline-teacher, fixed-response distillation:

1. A student response / rollout is fixed first.
2. The exact `response_token_ids` are fixed and carried through the pipeline.
3. Teacher condition scores are computed offline under multiple conditions.
4. Training reads the offline score JSONL; the teacher is not called inside
   student backward.
5. The real student forward reconstructs `prompt + response`.
6. Response-token logits are sliced as `[1, T, V]`, where `T` is the number of
   fixed response tokens.
7. FC-OPD loss compares student logits against stored teacher condition scores.
8. `backward()` and `optimizer.step()` update real student parameters.

This pipeline is now classified as diagnostics/prototype infrastructure only.
It is not the final FC-OPD training path, because the rollout tokens and teacher
condition scores are fixed before training and become stale after the student
policy changes. The training integration must instead follow online on-policy
distillation as described in `docs/fc_opd_online_verl_integration.md`.

The original dataset signal audit used `fixed_audit_response`, which is only a
protocol/path/condition audit. It is useful for checking full-vs-blur signal and
teacher scoring health, but it is not a student-rollout signal audit. Formal
OPD-compatible evidence requires `response_source=student_rollout`.

For Vision-OPD, the validated 2C student-rollout builder and real-student
optimizer-step smoke are now infrastructure validation only. The local
Vision-OPD images likely contain baked-in red bounding-box localization cues,
and any Gaussian-blur degraded image derived from those loaded images carries
the same cue. Do not treat current Vision-OPD outputs as main experimental
evidence.

The latest real student optimizer-step smoke verified that the tied Qwen3-VL-4B
`lm_head.weight` / `model.language_model.embed_tokens.weight` parameter receives
gradient and changes after Adam steps.

## Current Conditions

The legacy implemented pipeline uses four conditions:

- `full`: original image + question.
- `blur`: degraded image + question, currently Gaussian blur metadata with
  `sigma=2.0` unless overridden.
- `free`: generic image-only or weak caption/evidence.
- `task`: question-conditioned evidence/caption.

For Vision-OPD path and prompt audits, `task_evidence_mode=none` is a safe
non-informative placeholder mode. It is useful for verifying prompts and image
paths, but it is not final 4C training evidence. Vision-OPD full 2C and 4C runs
are frozen as main-data experiments unless clean no-red-box source images are
recovered.

The clean-data main path is now Geometry3K first, then ViRL39K after schema
inspection. Its condition set is named `4c_full_degraded_free_task`:

- `full`: clean original image + question.
- `degraded`: clean degraded image + question; default
  `lowres_10pct_nearest`.
- `free`: cached image-only caption/evidence + question, no image at
  forced-scoring time.
- `task`: cached question-conditioned evidence + question, no final answer and
  no gold-answer access.

Answer fields such as `reward_model.ground_truth` and `extra_info.answer` are
preserved as answer metadata for later evaluation, correctness estimation, and
alignment diagnostics. They must not be injected into default prompts or fixed
audit responses.

`fact` is reserved and is not implemented in the current real pipeline. It
should be used only for externally verified facts with provenance. Same-model
generated captions should not be treated as verified facts.

## Offline Teacher Caveat

Conditions are computed offline before gradient computation. The teacher service
is used only to build the offline score JSONL. The teacher is not called during
student backward or optimizer steps.

This separation is intentional: offline scoring records the teacher model ID,
tokenizer hash, top-k score tensors, response tokens, condition inputs, chunk
spans, and provenance before any student update.

## Alignment Status

The current implementation does not use explicit success-conditioned alignment:

```text
Align_c(u) = cos(g_c^KD(u), g_u^ideal(u))
```

That alignment requires `g_u^ideal`, which in turn requires success, reward,
correctness estimates, or multi-rollout success statistics. Those signals are
not available in the verified real-student pipeline.

Implemented today:

- condition decomposition (`full`, `blur`, `free`, `task`) and clean-data
  schema support for (`full`, `degraded`, `free`, `task`);
- offline top-k teacher score recording;
- chunk parsing for `visual_evidence`, `reasoning`, and `answer`;
- router-weighted FC-OPD distillation loss;
- synthetic loss/backward and optimizer-update smokes;
- real student logits/backward and optimizer-step smokes;
- no-op alignment hook schema for future correctness/success-failure
  calibration.

Not implemented yet:

- success-conditioned ideal-gradient estimation;
- verl trainer integration.

See `docs/fc_opd_rollout_and_alignment_plan.md` for the formal multi-rollout
plan. In short, `K=1` is smoke/debug only; formal OPD should use `K=4` by
default, with `K=8` as an ablation.

## Current Router Status

The default `RouterConfig()` in `src/dual_track_opd/fc_opd/router.py` maps:

- `visual_evidence -> task`
- `reasoning -> full`
- `answer -> full`
- invalid or uncovered tokens fall back to `full`

That router is a deterministic chunk router, not an alignment-aware router.

The offline loss and real-student smokes currently use
`FOUR_CONDITION_ROUTER` from `src/dual_track_opd/fc_opd/offline_loss.py`:

- `visual_evidence -> task`
- `reasoning -> free`
- `answer -> full`
- remaining tag / whitespace / uncovered tokens -> blur

`FOUR_CONDITION_ROUTER` is a smoke/default router designed to exercise all four
recorded conditions in the loss path. It should not be treated as the final
training router without a separate dataset-signal and training-quality decision.

## Verified Milestones

- Teacher service smoke.
- Real teacher offline score generation.
- Offline loss/backward smoke.
- Synthetic optimizer-update smoke.
- Real student logits/backward smoke.
- Real student optimizer-step smoke.

## Reproducibility Notes

Every experiment should continue to record:

- repo git commit and dirty status;
- backend commit or package version;
- full resolved config;
- dataset manifest hash;
- model checkpoint path;
- raw eval output path outside Git;
- summary metrics and notes.

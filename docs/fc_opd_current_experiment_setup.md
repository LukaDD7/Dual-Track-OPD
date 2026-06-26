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

The latest real student optimizer-step smoke verified that the tied Qwen3-VL-4B
`lm_head.weight` / `model.language_model.embed_tokens.weight` parameter receives
gradient and changes after Adam steps.

## Current Conditions

The implemented real pipeline uses four conditions:

- `full`: original image + question.
- `blur`: degraded image + question, currently Gaussian blur metadata with
  `sigma=2.0` unless overridden.
- `free`: generic image-only or weak caption/evidence.
- `task`: question-conditioned evidence/caption.

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

- condition decomposition (`full`, `blur`, `free`, `task`);
- offline top-k teacher score recording;
- chunk parsing for `visual_evidence`, `reasoning`, and `answer`;
- router-weighted FC-OPD distillation loss;
- synthetic loss/backward and optimizer-update smokes;
- real student logits/backward and optimizer-step smokes.

Not implemented yet:

- alignment-aware condition selection;
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

# FC-OPD Real Student-Logits Adapter Smoke

This stage is the **bridge from synthetic student logits to real student-model
logits**. It loads a real Qwen3-VL / Qwen3.5-VL student, runs a teacher-forced
multimodal forward on records from an offline-score JSONL, extracts the logits at
the exact response-token positions, feeds them into the existing FC-OPD
loss/router path, and verifies that `backward` reaches real model parameters.

It is **not** verl integration. No teacher service is required, and
`third_party/verl` is never imported. The transformers-specific code is imported
lazily, so the alignment and loss logic stay importable and CPU-testable.

Module: `src/dual_track_opd/fc_opd/real_student_smoke.py`
Scripts: `scripts/hpc/run_fc_opd_real_student_logits_smoke.{py,sh}`

## Token-position alignment

The teacher-forced sequence is `[prompt(P tokens), response(T tokens)]`. For a
causal LM the logits at position `i` predict token `i + 1`, so the logits that
predict the `T` response tokens are positions `P - 1 .. P + T - 2`:

```
response_logits = full_logits[:, P - 1 : P - 1 + T, :]   # shape [1, T, V]
```

This slicing lives in the pure, unit-tested `response_logit_slice` helper. The
exact `response_token_ids` from the offline record are appended to the processed
prompt, so the extracted positions correspond exactly to the scored tokens.

## What each record does

For every offline-score record the provider:

1. reads `question`, full image path, `response_text`, `response_token_ids`,
   `chunk_spans`, and `condition_scores`;
2. rebuilds the teacher-protocol multimodal chat prompt for the `full` condition
   via `render_teacher_prompt` and the processor chat template;
3. appends `response_text`'s token IDs for a teacher-forced forward;
4. runs the student forward and slices the response-position logits `[1, T, V]`;
5. feeds those logits into `route_condition_weights` -> `compute_fc_opd_loss`
   (the same path used by the synthetic-logits smoke), with the
   `FOUR_CONDITION_ROUTER` so all four conditions are exercised.

## Verification

`RealStudentResult.passed` requires:

- the student tokenizer fingerprint matches the offline `tokenizer_hash`;
- the response token length `T` matches the condition-score `T`;
- student logits have shape `[1, T, V]` with `V` covering the teacher token IDs;
- the decoded response matches the offline record (`--allow-retokenize-mismatch`
  downgrades this to a warning);
- the loss is finite and `backward` succeeds;
- at least one real model parameter has a finite, non-zero gradient;
- all four conditions (`full`, `blur`, `free`, `task`) are consumed.

No teacher service is contacted at any point.

## Running on HPC

```bash
bash scripts/hpc/run_fc_opd_real_student_logits_smoke.sh \
    "$DTOPD_OUTPUT_ROOT/fc_opd/offline_scores/vstar16_real_teacher/vstar_offline_scores.jsonl"
```

Or directly:

```bash
python scripts/hpc/run_fc_opd_real_student_logits_smoke.py \
    --scores "$DTOPD_OUTPUT_ROOT/fc_opd/offline_scores/vstar16_real_teacher/vstar_offline_scores.jsonl" \
    --model-path "$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct" \
    --limit 1 --device cuda --dtype bfloat16 --freeze-all-but-lm-head
```

The command prints one JSON object per record plus a summary, and exits non-zero
if any check fails.

### Inputs / flags

| flag | default | purpose |
| --- | --- | --- |
| `--scores` | (required) | offline-score JSONL |
| `--model-path` | `$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct` | student model (use Qwen3.5-VL-4B if available) |
| `--limit` | `1` | records to score (keep small for memory) |
| `--device` | `cuda` | forward device |
| `--dtype` | `bfloat16` | use `bfloat16` on H200 |
| `--freeze-all-but-lm-head` | off | freeze everything but the LM head to cut memory |
| `--max-prompt-length` | none | guard against oversized prompts |
| `--max-response-tokens` | none | guard against oversized responses |

## Memory safety

- Default `--limit 1` — a single forward per run.
- `--freeze-all-but-lm-head` keeps only the output head trainable; the
  finite/non-zero gradient check then targets `lm_head`.
- `bfloat16` for the H200 forward; the FC-OPD loss upcasts logits to float32
  internally, so the gradient remains well-conditioned.
- CPU tests use a tiny fake model (`tests/fc_opd/test_real_student_smoke.py`),
  so no weights are needed for CI.

## Tests

`tests/fc_opd/test_real_student_smoke.py` covers:

- `response_logit_slice` picks the correct next-token positions and rejects
  out-of-range slices;
- a fake provider's real parameter (`lm_head.weight`) receives a finite,
  non-zero gradient through the loss;
- shape / `T`-mismatch and tokenizer-hash-mismatch failures;
- decoded-text mismatch is rejected by default and relaxable;
- the four-condition consumption check.

# FC-OPD Offline Scoring

> Diagnostic only. Offline score JSONL and offline replay/min-train smokes are
> prompt/scoring/loss plumbing diagnostics. They must not be used as the final
> FC-OPD training input because precomputed student rollouts, teacher top-k
> scores, and router weights become stale as soon as the policy updates. The
> training path must be online and on-policy; see
> `docs/fc_opd_online_verl_integration.md`.

This stage turns a prepared Vision-OPD-style dataset plus student responses into
a self-describing **offline-score dataset**: for every student response it records
the teacher's top-k distribution under each FC-OPD condition (`full`, `blur`,
`free`, `task`), the decomposed condition signals, and the chunk spans of the
structured response.

It is deliberately **model-free**:

- The student model is **never** loaded.
- The teacher runs as a separate service (`dual_track_opd.fc_opd.teacher_service`);
  this stage only talks to it over HTTP.
- `third_party/verl` is **not** modified or imported.

All heavy lifting reuses existing primitives:
`ConditionInputs`, `TeacherClient` / `score_teacher_conditions`,
`compute_condition_signals`, and the chunk parser (`parse_response_chunks`).

## Inputs

1. **Vision-OPD dataset** — a JSON array or JSONL file. Each record is read with
   flexible field names:
   - image(s): `images` (list) / `image` / `image_path`
   - question: `query` / `question` / `prompt`
   - answer (optional): `response` / `answer` / `label`
   - identifier (optional): `question_id` / `id` / `index`
   - optional `condition_inputs`, `free_caption`, `task_evidence`
2. **Student responses** — required only in `--mode student`: a JSONL keyed by
   `sample_uid` / `question_id` / `index`. Each entry needs `response_text` and
   optionally `response_token_ids` (otherwise the text is re-tokenized).
   In `--mode protocol_smoke` the structured response is synthesised instead.
3. **Teacher service URL** — default `http://127.0.0.1:18080`.
4. **Condition configuration** — `--conditions full,blur,free,task`, `--blur-sigma`.

## Tokenizers

The stage needs a tokenizer only to encode/decode token IDs and to fingerprint
the vocabulary — not the student model.

- `--tokenizer byte` (default): the bundled `ByteTokenizer`, a deterministic
  byte-level tokenizer with an exact decode round-trip. No weights required.
- `--tokenizer hf:<name-or-path>`: a Hugging Face `AutoTokenizer` (loads only the
  tokenizer, never the model). The tokenizer fingerprint must match the teacher's
  `tokenizer_hash`.

## Output

Written under `$DTOPD_OUTPUT_ROOT/fc_opd/offline_scores/` (override with
`--output-dir`) as JSONL, and optionally Parquet (`--parquet`). Each record:

| field | description |
| --- | --- |
| `sample_uid` | `"{source_dataset}:{question_id}"` |
| `source_dataset`, `source_index`, `question_id` | provenance |
| `question` | the resolved query |
| `image_paths` | resolved full-image paths |
| `condition_inputs` | full/degraded image paths + blur transform, caption, evidence |
| `response_text`, `response_token_ids` | student response |
| `chunk_spans` | per-chunk token spans + `format_valid`, `errors`, `token_counts` |
| `tokenizer_hash` | student tokenizer fingerprint |
| `protocol_version` | teacher protocol version |
| `condition_scores[cond]` | `token_ids` `[T,K]`, `log_probs` `[T,K]`, `tail_log_prob` `[T]`, `entropy` `[T]` |
| `condition_signals` | `visual_detail` (full vs blur), `task_extraction` (task vs free), each `[T]` |
| `metadata` | `top_k`, `blur_sigma`, `teacher_model_id`, `created_at`, `response_mode` |

## Smoke run

A self-contained smoke run scores the first 16 synthetic VStar-style samples
against an **in-process** synthetic teacher (top-k 32). It needs no GPU, no
weights, and no external service:

```bash
bash scripts/hpc/run_fc_opd_offline_smoke.sh
```

Equivalently:

```bash
python scripts/hpc/build_fc_opd_offline_scores.py \
    --self-contained-smoke --limit 16 \
    --source-dataset vstar --mode protocol_smoke \
    --tokenizer byte --smoke-top-k 32 \
    --output-dir "$TMPDIR/fc_opd_offline_smoke"
```

The run verifies that every sample has all four conditions and that each
condition tensor has shape `[T, 32]` before exiting.

## Real run

Start the teacher service (see `docs/fc_opd_teacher_service.md`), then:

```bash
python scripts/hpc/build_fc_opd_offline_scores.py \
    --dataset "$DTOPD_DATA_ROOT/vstar_eval.json" \
    --mode student \
    --student-responses "$DTOPD_OUTPUT_ROOT/vstar_student.jsonl" \
    --teacher-url http://127.0.0.1:18080 \
    --tokenizer hf:Qwen/Qwen3-VL-8B-Instruct \
    --source-dataset vstar \
    --conditions full,blur,free,task --blur-sigma 2.0 \
    --verify-top-k 32 --parquet
```

The blurred-image path for each sample is derived deterministically
(`<stem>.gaussian_blur_s<sigma><ext>`) and only its transform metadata is
recorded — this stage never mutates source images.

## Offline loss / backward smoke

This is also diagnostic only. It verifies tensor plumbing and gradients over
recorded scores; it is not a replay-training recipe.

`dual_track_opd.fc_opd.offline_loss` closes the loop between the offline-score
dataset and the existing FC-OPD loss/router path **before** any trainer/verl
change. For each recorded payload it:

1. rebuilds the four condition `TeacherTopK` tensors (`token_ids [T,K]`,
   `log_probs [T,K]`, `tail_log_prob [T]`, `entropy [T]`);
2. parses the recorded chunk spans into `visual_evidence` / `reasoning` /
   `answer` masks aligned to `T`;
3. attaches synthetic student logits `[1, T, V]` with `requires_grad=True`
   (`V` defaults to `max teacher token id + 1`);
4. runs `route_condition_weights` -> `compute_fc_opd_loss` -> `backward`.

The router used here (`FOUR_CONDITION_ROUTER`) maps `visual_evidence→task`,
`reasoning→free`, `answer→full`, and the remaining tag/whitespace tokens fall
back to `blur`, so all four conditions are exercised in the loss.

It verifies: finite loss, present and finite student-logit gradients, chunk
masks aligned to the response token length, and that all four conditions are
consumed. No student model or teacher service is loaded.

```bash
# Against the real-teacher VStar smoke output on HPC:
bash scripts/hpc/run_fc_opd_offline_loss_smoke.sh \
    "$DTOPD_OUTPUT_ROOT/fc_opd/offline_scores/vstar16_real_teacher/vstar_offline_scores.jsonl"

# Or directly:
python scripts/hpc/run_fc_opd_offline_loss_smoke.py --scores <offline_scores.jsonl>
```

The command prints a JSON report and exits non-zero if any check fails.

## Minimal optimizer-update smoke

This smoke is diagnostic only. A decreasing synthetic replay loss is not
evidence that FC-OPD training is correctly on-policy.

`run_offline_min_train_smoke` (also in `dual_track_opd.fc_opd.offline_loss`) takes
the loss/backward smoke one step further: it attaches one synthetic
`student_logits` parameter per record and runs **Adam** for a few steps (default
10) against the recorded teacher scores. Router weights are fixed (they do not
depend on the student), so the loss is purely a function of the student logits
and the update should reduce it.

Per step it reports `loss`, `grad_norm`, `logits_delta_norm`, and
`consumed_conditions`, and verifies: finite loss and gradients every step, that
every `optimizer.step()` changes the logits (`logits_delta_norm > 0`), and that
all four conditions are consumed.

```bash
# Against the real-teacher VStar smoke output on HPC (10 Adam steps):
bash scripts/hpc/run_fc_opd_min_train_smoke.sh \
    "$DTOPD_OUTPUT_ROOT/fc_opd/offline_scores/vstar16_real_teacher/vstar_offline_scores.jsonl"

# Or directly, with a custom step count:
python scripts/hpc/run_fc_opd_min_train_smoke.py --scores <offline_scores.jsonl> --steps 10
```

The command prints one JSON object per step plus a JSON summary, and exits
non-zero if any check fails. It loads no student model and never imports
`third_party/verl`.

## Tests

`tests/fc_opd/test_offline_scoring.py` exercises the scoring pipeline against the
synthetic teacher: four-condition `[T,32]` shapes, chunk-span validity, student
vs protocol-smoke modes, and JSONL round-tripping.

`tests/fc_opd/test_offline_loss.py` exercises the loss/backward path on synthetic
offline-score fixtures: tensor/mask alignment, finite loss, gradient presence on
student logits only, four-condition consumption, and JSONL round-tripping.

`tests/fc_opd/test_offline_min_train.py` exercises the Adam optimizer-update
smoke: finite loss/gradients every step, logit updates every step, loss
decreasing over training, and four-condition consumption.

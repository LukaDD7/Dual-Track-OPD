# FC-OPD Teacher Service

The teacher scorer is a standalone synchronous HTTP sidecar. It does not import
or depend on verl.

Endpoints:

```text
GET  /health
GET  /metadata
POST /score
```

`/metadata` records the model ID, resolved model revision, tokenizer hash,
vocabulary size, top-k width, dtype, and protocol version.

## CPU protocol smoke

The synthetic backend validates serialization, batching, condition rendering,
top-k/tail normalization, determinism, and exact response-token transport:

```bash
python -m dual_track_opd.fc_opd.teacher_service \
  --backend synthetic \
  --host 127.0.0.1 \
  --port 18080 \
  --vocab-size 128 \
  --top-k 32
```

It is a protocol test only and must never be used as a training teacher.

## Qwen3-VL teacher

After the separate GPU environment is installed:

```bash
bash scripts/hpc/start_fc_teacher.sh
```

Relevant environment variables:

```text
FC_OPD_TEACHER_MODEL
FC_OPD_TEACHER_REVISION
FC_OPD_TEACHER_HOST
FC_OPD_TEACHER_PORT
FC_OPD_TEACHER_TOP_K
FC_OPD_TEACHER_DTYPE
FC_OPD_TEACHER_DEVICE
```

The first real run must use 4–8 samples and perform no optimizer step. It must
verify:

- student and teacher tokenizer hashes are equal;
- the server returns the exact supplied response token IDs;
- full, blur, free, and task conditions all score successfully;
- top-k and tail probability mass is finite and sums to one;
- repeated requests are deterministic;
- condition distributions are not all identical.

The Transformers backend is correctness-first and scores requests sequentially.
Batching by condition and sequence length should be enabled only after this GPU
probe passes.

## Hard failure policy

The service and client raise errors for:

- tokenizer hash mismatch;
- response retokenization mismatch when response text is supplied;
- token IDs outside the teacher vocabulary;
- duplicate top-k IDs;
- invalid or non-normalized probability mass;
- missing condition input;
- unavailable fact condition;
- server timeout or malformed response.

There is no fallback from a failed condition to the full-image condition.

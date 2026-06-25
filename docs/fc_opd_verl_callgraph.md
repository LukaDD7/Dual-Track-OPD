# Failure-Calibrated OPD: verl v0.7.1 Call Graph

Date: 2026-06-25

Backend:

- `verl v0.7.1`
- commit `bec9ef74768dd201881cd4e54cd0385e87caae27`
- FSDP actor + vLLM rollout target

## Executive finding

Official v0.7.1 already includes an asynchronous top-k on-policy distillation
recipe under `recipe/gkd/megatron/`. It provides useful teacher-service and
sparse-KL reference code, but its training engine is Megatron only.

For this project's FSDP target, the minimum integration is:

1. keep condition metadata in the outer project's dataset schema;
2. build chunk masks after the student response is finalized;
3. call a standalone teacher service from the trainer after rollout;
4. attach masks and teacher top-k tensors to `DataProto.batch`;
5. preserve those new tensor keys in the FSDP actor's fixed `select_keys`;
6. compute the auxiliary sparse KD loss while response logits still exist in
   `DataParallelPPOActor._forward_micro_batch`;
7. add the auxiliary loss to the existing actor loss and report metrics.

The rollout engine itself does not need modification for the first synchronous
full-only implementation.

## End-to-end call graph

```text
Parquet / JSONL row
  |
  v
verl.utils.dataset.rl_dataset.RLHFDataset.__getitem__
  - builds raw_prompt with image/video objects
  - retains arbitrary row metadata as non-tensors
  |
  v
collate_fn
  - tensors -> stacked torch.Tensor
  - other values -> NumPy object arrays
  |
  v
RayPPOTrainer.fit
  - DataProto.from_single_dict(batch_dict)
  - assigns uid
  - _get_gen_batch()
  - repeat(rollout.n)
  |
  v
AgentLoopManager.generate_sequences
  |
  v
AgentLoop._postprocess
  - responses
  - response_mask
  - input_ids / attention_mask / position_ids
  - multi_modal_inputs in non_tensor_batch
  - original non-tensor fields restored
  |
  v
RayPPOTrainer.fit
  - repeat source batch to rollout.n
  - union(generation output)
  - compute response_mask if absent
  - optional batch balancing/reordering
  - reward / old_log_probs / advantages
  |
  v
RayPPOTrainer._update_actor
  |
  v
ActorRolloutRefWorker.update_actor
  |
  v
DataParallelPPOActor.update_policy
  - fixed batch key selection
  - mini-batch and micro-batch split
  - _forward_micro_batch
  - policy loss dispatch
  - optional entropy/reference KL
  - backward and optimizer step
```

## Data and multimodal storage

### Dataset row

`RLHFDataset.__getitem__` returns the full row plus:

- `raw_prompt`
- `dummy_tensor`
- `index`
- `tools_kwargs`
- `interaction_kwargs`

The project can keep `condition_inputs` as a regular nested row field. The
collator converts it to a NumPy object array, and `DataProto.from_single_dict`
places NumPy arrays in `non_tensor_batch`.

Recommended ownership:

- outer repository: validate and construct condition metadata;
- verl: transport only;
- teacher service: render condition-specific teacher inputs.

Do not put image transforms or caption generation logic into
`RLHFDataset`.

### Rollout multimodal inputs

The agent loop converts the prompt's images into processor outputs and returns
them as:

```text
DataProto.non_tensor_batch["multi_modal_inputs"]
```

It finalizes these response-aligned tensors:

```text
responses
response_mask
input_ids
attention_mask
position_ids
```

These tensors are the authoritative basis for chunk parsing and teacher token
alignment.

## Response finalization and chunk masks

Best first insertion point:

```text
RayPPOTrainer.fit
after:
  batch = batch.repeat(...)
  batch = batch.union(gen_batch_output)
  response_mask is available
before:
  batch balancing
  old-log-prob computation
  actor update
```

At that point every source record is aligned with each generated response.
Build:

```text
fc_visual_evidence_mask  [B, T]
fc_reasoning_mask        [B, T]
fc_answer_mask           [B, T]
fc_format_valid          [B]
```

Store response-aligned masks as tensor fields in `DataProto.batch`. Tensor
fields are automatically repeated, indexed, split, concatenated, serialized,
and reordered with the rest of the batch.

Do not store token masks as Python objects in `non_tensor_batch`.

## Teacher scoring insertion

For the first synchronous implementation, call the standalone teacher client
at the same post-rollout insertion point, after responses and masks exist.

Inputs:

```text
condition_inputs from non_tensor_batch
responses
response_mask
the exact student response token IDs
condition identifier
```

Outputs to attach as tensors:

```text
fc_teacher_topk_indices   [B, T, K]  int32/int64
fc_teacher_topk_log_probs [B, T, K]  float32
fc_teacher_tail_log_prob  [B, T]     optional float32
fc_teacher_entropy        [B, T]     optional float32
fc_condition_id           [B, T]     compact integer routing result
```

Hard-assert that the teacher-scored response token IDs equal `responses`.
Never silently retokenize or substitute the full-image condition after a
condition failure.

The official `recipe/gkd/megatron/teacher/` code is a useful protocol reference:

- asynchronous client futures;
- micro-batch aggregation;
- top-k indices and log-probabilities;
- finite-value checks;
- vLLM prompt-log-prob extraction.

It should not be imported directly into project research logic. Implement the
stable project API under `src/dual_track_opd/` and keep any verl patch thin.

## Ray/DataProto transport

`DataProto` supports:

- tensor fields in a `TensorDict`;
- NumPy object arrays in `non_tensor_batch`;
- `repeat`, `select`, `select_idxs`, `split`, `concat`, `union`, serialization,
  and distributed object gathering.

This is sufficient for condition metadata and teacher tensors. The main risk is
not Ray transport; it is downstream key filtering.

`DataParallelPPOActor.update_policy` currently selects a fixed list:

```text
responses
response_mask
input_ids
attention_mask
position_ids
old_log_probs
advantages
optional ref_log_prob / rollout correction fields
```

Therefore new teacher tensors and chunk masks will be dropped unless the FSDP
actor selection list is extended. That is one required minimal verl patch.

## Actor log-probability and loss path

`RayPPOTrainer._compute_old_log_prob` calls the actor worker, which calls:

```text
DataParallelPPOActor.compute_log_prob
  -> _forward_micro_batch
  -> response-token sampled log_probs
```

During actor update:

```text
DataParallelPPOActor.update_policy
  -> _forward_micro_batch
  -> get_policy_loss_fn(loss_mode)
  -> registered PPO-style policy loss
  -> entropy/reference KL additions
  -> backward
```

The registered policy-loss API receives sampled-token log-probabilities, not
the full or teacher-top-k student distribution. A sparse forward KL cannot be
implemented correctly only by registering another standard policy loss.

The auxiliary KD term must be computed while response logits are still
available in `_forward_micro_batch`, or `_forward_micro_batch` must return the
student log-probabilities gathered at teacher top-k indices.

Recommended minimal design:

```text
_forward_micro_batch(model_inputs)
  - normal model forward
  - normal sampled-token log_probs
  - if fc_teacher_topk_* exists:
      compute response-aligned sparse KD per token
      return fc_kd_loss_per_token

update_policy()
  - aggregate fc_kd_loss_per_token with response/chunk masks
  - add fc_kd_coef * auxiliary loss to policy_loss
  - emit chunk/condition metrics
```

This avoids returning full-vocabulary logits from a micro-batch and keeps the
large tensor local to the actor forward pass.

## Minimal backend patch surface

Expected patch locations:

1. `verl/workers/actor/dp_actor.py`
   - retain project tensor fields in `select_keys`;
   - calculate sparse KD from live logits;
   - aggregate and log the auxiliary term.
2. a small trainer hook or subclass outside verl
   - post-rollout chunk parsing;
   - teacher request;
   - attach teacher tensors.
3. config extension
   - enable flag, coefficient, top-k, condition mode, mask fallback.

Avoid modifying:

- vLLM rollout internals;
- `RLHFDataset` for research-specific condition construction;
- `DataProto`;
- distributed worker dispatch;
- checkpoint code.

Keep the backend change as a documented patch under `patches/verl/`.

## First implementation sequence

1. Framework-independent CPU modules and synthetic tests.
2. Standalone teacher service/client with exact-token assertions.
3. Four-to-eight sample teacher scoring probe.
4. Thin trainer hook that attaches full-only teacher tensors.
5. FSDP actor auxiliary sparse-KL patch.
6. Single-GPU one-step backward smoke test.
7. Four-H200 two-step distributed smoke test.
8. Only then enable task-only and deterministic chunk routing.

## Risks to test explicitly

- dynamic batching reorders masks or teacher tensors incorrectly;
- remove-padding path misaligns response positions;
- teacher top-k contains duplicate IDs or invalid tail mass;
- top-k tensors are accidentally cast to bf16;
- teacher tensors are copied repeatedly through object arrays;
- invalid structured output leaves all semantic masks empty;
- auxiliary loss is applied to prompt or padding tokens;
- the actor's fused-kernel path does not expose logits needed by sparse KD;
- student/teacher tokenizers differ despite sharing a model family.

For the first smoke run, disable fused actor kernels if necessary and prove the
non-fused sparse-KL path before optimizing it.

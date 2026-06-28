# Online FC-OPD verl Integration

FC-OPD training must be on-policy. The rollout used for distillation is sampled
from the current student policy in the current training step, then teacher and
student forced scoring are computed immediately on that same response.

The offline score builders remain useful for diagnostics, prompt auditing, and
prototype signal checks. They must not be used as the main training input,
because precomputed rollouts and precomputed teacher weights become stale as
soon as the student policy changes.

## Current verl Extension Points

The current upstream GKD path lives under
`third_party/verl/recipe/gkd/megatron/`.

- Rollout source: `OnPolicyDistillTrainer._async_gen_next_batch` in
  `ray_trainer.py` syncs actor weights to the rollout worker, then calls
  `rollout_wg.async_generate_sequences(gen_batch)`. This is the correct place
  to attach FC-OPD metadata because `gen_batch_output` contains current
  on-policy `responses`, `input_ids`, and `attention_mask`.
- Teacher top-k scoring: `OnPolicyDistillTrainer._async_get_teacher_knowledge`
  calls `teacher_utils.get_teacher_knowledge(gen_batch_output, teacher_client)`.
  That helper extracts valid current sequence token ids and asks the GKD teacher
  server for `teacher_topk_logps` and `teacher_topk_indices`.
- Training batch merge: `fit()` unions the original batch, current rollout
  output, and teacher output before `actor_wg.update_actor(batch)`.
- GKD loss: `MegatronOnPolicyDistillActor.forward_backward_batch` in
  `megatron_workers.py` reads `teacher_topk_logps` and
  `teacher_topk_indices`, builds `calc_kl_mask` over response tokens, then calls
  `self.distill_loss_op(...)`.
- Distill operator: `megatron_distill_losses.py` already supports selectable
  vocab-parallel KL/RKL/KL_RKL/JSD losses.

No `third_party/verl` code is changed by this document. The first integration
step should keep FC-OPD logic in `src/dual_track_opd/fc_opd/online_batch.py` and
only patch verl after the batch schema is stable.

## Online FC-OPD Step

1. Sample current rollouts from the current student policy.

   The rollout source is only the live policy after the latest actor-to-rollout
   weight sync. Do not load response ids, rollout text, teacher top-k, router
   weights, or loss weights from offline score JSONL.

2. Parse chunks on the current response.

   Decode the current response token ids, run the structured chunk parser, and
   build masks for `visible_evidence`, `diagram_inference`, `reasoning`, and
   `answer`. Invalid chunks are still represented with fallback masks so the
   verifier gate can reduce their learning value.

3. Verify the current response.

   The verifier classifies each rollout as:

   - `correct`
   - `wrong_but_format_valid`
   - `malformed`

   The verifier produces `verifier_learning_value_gate`. It is an
   error-focused learning-value gate, not a correct-only filter. Wrong but
   format-valid rollouts receive the highest visual/diagram/reasoning weight;
   correct answer chunks receive zero or near-zero OPD weight.

4. Teacher forced scoring on the current rollout.

   For each selected FC-OPD condition, the teacher is forced to score the exact
   current response token ids. The request must include the current question,
   current condition evidence/cache, current images, current response text, and
   current response token ids.

5. Student forced scoring on the current rollout.

   The current student is also forced over the same current response token ids
   under the same condition prompts. These scores are used for the student
   deficit term:

   `student_deficit = max(0, teacher_delta - student_delta - margin)`

   This makes FC-OPD focus on capabilities where the teacher has condition
   sensitivity that the current student lacks.

6. Capability attribution and gated routing.

   Teacher condition contrasts produce capability attribution:

   - `visual_detail`: `full - degraded`
   - `evidence_selection`: `task_visible - free`
   - `visual_text_inference`: `task_infer - task_visible`
   - `solving`: `task_solve - task_infer`

   Capability weights are multiplied by chunk compatibility,
   `student_deficit`, and `verifier_learning_value_gate`, then routed with
   `student_deficit_chunk_gated`.

7. Grouped loss in the current batch.

   The actor loss consumes current student logits, immediate teacher top-k, and
   online condition weights. The loss can be sparse KL/JSD depending on the verl
   GKD distill operator. Grouping is by capability/chunk:

   - visible group: `visual_detail`, `evidence_selection`
   - infer group: `visual_text_inference`
   - solve group: `solving`

8. Backward in the same training step.

   The FC-OPD loss is added to the actor update for the current batch. A valid
   online smoke must generate fresh rollouts, score them immediately with the
   teacher, compute FC-OPD loss, run backward, take one optimizer step, and
   verify a nonzero parameter update.

## Minimal External Hook

`src/dual_track_opd/fc_opd/online_batch.py` provides the reusable hook outside
verl. It accepts:

- batch prompts/images through `OnlineFCOPDSample.prompt` and `.images`
- current rollout text and token ids through `.rollout_text` and
  `.rollout_token_ids`
- condition evidence/cache through `.condition_inputs`
- a teacher scorer callable
- a current-student forced scorer callable
- an optional verifier callable

It returns:

- per-token teacher top-k scores
- per-token student condition log-probs
- capability attribution and student deficit
- `verifier_learning_value_gate`
- condition weights and grouped loss-ready tensors
- a finite FC-OPD loss tensor suitable for a training step

`src/dual_track_opd/fc_opd/verl_integration.py` converts these online outputs
into the tensor-only contract expected by the thin verl patches:

```text
fc_teacher_topk_indices    [B, C, T, K]
fc_teacher_topk_log_probs  [B, C, T, K]
fc_teacher_tail_log_prob   [B, C, T] optional
fc_condition_weights       [B, C, T]
fc_condition_ids           [C]
```

`src/dual_track_opd/fc_opd/verl_sparse_kd.py` owns the sparse top-k KD tensor
math used by the actor patch. Keeping this math in the project package makes
the backend patch a transport-and-callsite layer rather than a research logic
fork.

## Patch Overlay

The first verl integration is stored as patches, not direct edits to
`third_party/verl`:

- `patches/verl/fc_opd_ray_trainer_post_rollout_hook.patch`
  adds `algorithm.fc_opd.post_rollout_hook` immediately after current rollout
  responses are unioned into `batch` and `response_mask` exists.
- `patches/verl/fc_opd_fsdp_actor_aux_kd.patch`
  preserves `fc_*` tensor fields through the FSDP actor `select_keys`, computes
  sparse KD from live response logits in `_forward_micro_batch`, and adds the
  weighted auxiliary loss to the PPO actor loss.

The project-owned post-rollout hook should:

1. decode current response token ids from `batch.batch["responses"]`;
2. parse structured FC-OPD chunks on those current responses;
3. run the verifier and build `verifier_learning_value_gate`;
4. call teacher forced scoring on the exact current response token ids;
5. compute student-deficit/routing weights;
6. attach the `fc_*` tensors listed above to `DataProto.batch`.

The hook must not read offline score JSONL or reuse precomputed teacher scores.

The actor patch deliberately fail-fasts when fused actor kernels hide logits.
Remove-padding without Ulysses sequence parallel is covered; remove-padding plus
Ulysses SP needs a separate alignment smoke before enabling.

## Online Smoke

`scripts/hpc/run_fc_opd_online_train_step_smoke.py` is the first online
training-step smoke. It reads an evidence cache row only for condition text and
image paths, then:

1. loads the current 4B student;
2. samples a fresh structured rollout from that current student;
3. immediately scores that rollout with the teacher service;
4. forced-scores the same rollout with the current student under FC-OPD
   conditions;
5. computes online FC-OPD loss through `online_batch.py`;
6. backpropagates and applies one optimizer step to a selected student
   parameter;
7. verifies the selected parameter changed.

It does not read offline score JSONL, precomputed rollout weights, or
precomputed teacher top-k.

Example:

```bash
python scripts/hpc/run_fc_opd_online_train_step_smoke.py \
  --evidence-jsonl "$DTOPD_OUTPUT_ROOT/fc_opd/evidence/geometry3k/evidence.jsonl" \
  --student-model "hf:$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct" \
  --teacher-url http://127.0.0.1:18080 \
  --row-index 0 \
  --output-json "$DTOPD_OUTPUT_ROOT/fc_opd/online_smoke/one_step.json"
```

## What Must Not Happen

- Do not train from offline score JSONL as the final FC-OPD path.
- Do not reuse precomputed teacher top-k or condition weights after rollout
  generation has moved to a later student checkpoint.
- Do not implement correct-only OPD.
- Do not treat Phase B offline score JSONL or Phase C replay smoke as the final
  training mechanism. They are diagnostics only.

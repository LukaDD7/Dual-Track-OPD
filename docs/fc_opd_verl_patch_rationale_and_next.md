# FC-OPD verl Integration: Patch Rationale & Missing Hook

Written 2026-06-28. Context: codex delivered the patch overlay (`993cf28`),
tests pass (195 passed, 2 skipped), patches apply cleanly. CPU follow-up
identified that the trainer-side hook implementation was missing. The first
hook implementation now lives in
`src/dual_track_opd/fc_opd/verl_post_rollout_hook.py`; it wires current rollout
tokens to online FC-OPD scoring and attaches `fc_*` tensors. A real GPU student
forced-scorer FQN/worker still needs to be supplied for full verl training.

## 1. Why Patches Instead of Directly Modifying `third_party/verl`

### 1.1 Git Submodule — Dirty State Problem

```
third_party/verl/  →  https://github.com/verl-project/verl.git  (v0.7.1, bec9ef74)
```

`third_party/verl` is a **git submodule**, not a vendored copy. If we edit files
inside it directly:

- `git diff` inside the submodule shows uncommitted changes
- The superproject records a dirty submodule hash
- `git submodule update --init` on another machine wipes our edits
- Every verl upgrade requires manually re-applying scattered diffs

Patches stored in `patches/verl/` are explicit, versioned, and re-applicable.
`git apply --check` verifies compatibility before touching the submodule.

### 1.2 Reproducibility — Clean Separation of Concern

A paper reviewer or collaborator should be able to:

```bash
git clone <this-repo>
git submodule update --init   # pristine verl v0.7.1
git apply patches/verl/*.patch  # exact 26+177 line contract
# train
```

The patches are not "fork of verl with FC-OPD baked in." They are a minimal
interface contract: two hook points, zero research logic. All math is in
`src/dual_track_opd/fc_opd/` — importable, testable, independent of verl.

### 1.3 Upstream Rebase Surface

verl v0.8+ has a core `distillation` module with `MultiTeacherModelManager`.
When we upgrade, the only things that need to change are the patches:

| Patch | Rebase risk |
|-------|-------------|
| `fc_opd_ray_trainer_post_rollout_hook.patch` | Low — inserts ~20 lines after `response_mask`, which is a stable checkpoint in the PPO loop |
| `fc_opd_fsdp_actor_aux_kd.patch` | Medium — `_forward_micro_batch` and `update_policy` evolve across versions, but the patch only reads `fc_*` keys and adds one loss term |

Direct edits scattered across 10 files would have unbounded rebase risk.

### 1.4 The Alternative: Forking verl

Forking verl would mean maintaining a parallel `verl` repo just for FC-OPD.
This is what HJSang/OPSD does — they forked verl and modified GKD recipe files
directly. The downside: their fork diverges from upstream, and merging upstream
bugfixes is manual.

Our patch approach is **lighter than a fork, heavier than nothing** — exactly
right for a research codebase that needs to track upstream.

## 2. What We Reuse From verl (We Do NOT Write Our Own Trainer)

This is the critical point. The patches are 26 + 177 lines because verl already
provides everything else. We are **not** writing our own distributed trainer,
FSDP actor, data pipeline, or optimizer.

### 2.1 Training Loop — `RayPPOTrainer.fit()`

```
third_party/verl/verl/trainer/ppo/ray_trainer.py  (1627 lines)
```

This is the main PPO training loop. It handles:

- Data loading and curriculum scheduling
- Rollout generation via `rollout_wg.async_generate_sequences()`
- Reward computation via `rm_wg.compute_rm_score()`
- Advantage estimation (GAE / GRPO / REINFORCE++)
- Old log-prob reference computation
- Actor update dispatch
- Multi-turn conversation stitching
- Checkpointing and metrics logging

Our hook inserts at **one point**: immediately after on-policy responses are
generated and `response_mask` exists, but before advantage computation. This
means FC-OPD teacher/student scoring happens on the **exact tokens the current
policy produced**, which is the definition of on-policy distillation.

**We do not write a training loop.** We inject one function call.

### 2.2 Actor Worker — `DataParallelPPOActor`

```
third_party/verl/verl/workers/actor/dp_actor.py  (676 lines)
```

This is the FSDP-based actor that handles:

- **FSDP sharding**: `self.actor_module` is an FSDP-wrapped model
- **Micro-batching**: `data.split(micro_batch_size)` or `prepare_dynamic_batch()`
- **Forward pass**: `_forward_micro_batch()` with two code paths:
  - **Remove-padding** (`use_remove_padding=True`): logits from `[total_nnz, V]`
    rmpad format, memory-efficient for variable-length sequences
  - **Standard** (`use_remove_padding=False`): logits from `[B, T, V]`
- **Gradient accumulation**: `loss_scale_factor` for dynamic batch size
- **Mixed precision**: `PrecisionType` toggles fp32/bf16/fp16
- **Ulysses sequence parallel**: `gather_outputs_and_unpad()` for long contexts
- **Fused kernels**: `use_fused_kernels` for memory-bound forward+backward
- **KL loss**: `use_kl_loss` for KL penalty against reference policy

Our patch adds **one auxiliary loss term** to `update_policy()`:

```python
# After the existing policy_loss is computed:
fc_opd_loss = fc_opd_loss_per_token.sum() / fc_opd_active_weight.sum().clamp_min(1.0)
policy_loss = policy_loss + fc_opd_loss * fc_opd_coef
```

That's it. The FSDP backward, gradient sync, and optimizer step are completely
unchanged. The patch also preserves `fc_*` keys through `select_keys` so they
survive `data.select()`.

**We do not write an actor worker.** We add one loss term to the existing one.

### 2.3 Rollout Workers — vLLM Async Generation

```
third_party/verl/verl/workers/rollout/vllm_rollout/
```

The rollout infrastructure uses vLLM with tensor parallel, continuous batching,
and async dispatch via Ray. We don't touch this at all — the trainer patch
runs **after** rollout generation completes and `gen_batch_output` is unioned.

**We do not write rollout workers.** The current policy's tokens arrive via the
existing `rollout_wg.async_generate_sequences()`.

### 2.4 DataProto — Batch Abstraction

```python
from verl import DataProto

batch = DataProto.from_single_dict(...)
batch = batch.union(gen_batch_output)   # merge rollout results
batch.batch["fc_teacher_topk_indices"] = ...  # attach FC-OPD tensors
batch = batch.select(batch_keys=[...])  # filter for actor
```

`DataProto` is verl's core batch type — it wraps `TensorDict` (`.batch`) plus
`dict` (`.non_tensor_batch`). It provides `union()`, `pop()`, `repeat()`,
`select()`, `split()`, all of which we reuse without modification.

**We do not write a batch abstraction.** We attach tensors to `DataProto.batch`
and the existing machinery transports them through the pipeline.

### 2.5 Optimizer, Scheduler, Checkpointing

All inherited from `RayPPOTrainer` and `BasePPOActor`:

- **Optimizer**: `self.actor_optimizer.step()` / `zero_grad()`
- **LR scheduler**: config-driven, applied per-step
- **Checkpointing**: `actor_wg.save_checkpoint()` with FSDP state dict
- **Metrics**: `metrics.update()` aggregated across workers

**We do not write any of this.** Our hook only adds metric keys like
`actor/fc_opd_loss` and `actor/fc_opd_coef`.

### 2.6 Summary: What's Ours vs. What's Reused

```
┌─────────────────────────────────────────────────────────┐
│  OUR CODE (src/dual_track_opd/fc_opd/)                  │
│                                                         │
│  online_batch.py        ← FC-OPD batch logic            │
│  verl_integration.py    ← DataProto tensor adapter      │
│  verl_sparse_kd.py      ← Sparse top-K KD math          │
│  loss.py / router.py    ← Condition routing + KL loss   │
│  signal_decomposer.py   ← Capability delta computation  │
│  teacher_client.py      ← HTTP teacher calls            │
│  chunk_parser.py        ← Response structure parsing    │
│  verifier.py            ← Answer verification           │
│                                                         │
│  verl_post_rollout_hook.py ← Trainer hook impl          │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  REUSED FROM verl (third_party/verl/)                  │
│                                                         │
│  RayPPOTrainer.fit()     ← Training loop (1627 lines)   │
│  DataParallelPPOActor    ← FSDP actor (676 lines)       │
│  vLLM rollout workers    ← Async generation             │
│  DataProto               ← Batch abstraction            │
│  Optimizer / Scheduler   ← PyTorch + verl wrappers      │
│  Checkpointing           ← FSDP state dict save/load    │
│  Metrics / Logging       ← WandB / TensorBoard          │
│  Ray orchestration       ← Distributed scheduling       │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  PATCHES (patches/verl/)                                │
│                                                         │
│  fc_opd_ray_trainer_post_rollout_hook.patch  (26 lines) │
│  fc_opd_fsdp_actor_aux_kd.patch             (177 lines) │
│                                                         │
│  Total patched: 203 lines out of 2303+ lines reused     │
└─────────────────────────────────────────────────────────┘
```

## 3. Post-Rollout Hook Implementation

### 3.1 The Interface Contract

The trainer patch expects a function at `algorithm.fc_opd.post_rollout_hook`
(a fully-qualified dotted name, loaded via `load_class_from_fqn`). The function
signature:

```python
def post_rollout_hook(
    *,
    batch: DataProto,          # current batch with responses, input_ids, attention_mask, response_mask
    tokenizer: Any,            # HF tokenizer or verl tokenizer wrapper
    processor: Any | None,     # HF processor (for vision models)
    config: dict,              # full verl config dict
    global_steps: int,         # current training step
) -> tuple[DataProto, dict]:   # (modified batch, metrics dict)
```

The trainer calls it immediately after:

```python
batch = batch.union(gen_batch_output)                    # line ~1359
batch.batch["response_mask"] = compute_response_mask(batch)  # line ~1362
# ─── HOOK CALLED HERE ───
batch, fc_opd_metrics = post_rollout_hook(batch=batch, ...)
```

At this point `batch.batch` contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `prompts` | `[B, prompt_len]` | Input token IDs (prompt only) |
| `responses` | `[B, response_len]` | Current on-policy rollout token IDs |
| `input_ids` | `[B, prompt_len + response_len]` | Full sequence |
| `attention_mask` | `[B, prompt_len + response_len]` | Attention mask |
| `response_mask` | `[B, response_len]` | Valid response positions (1/0) |
| `position_ids` | `[B, prompt_len + response_len]` | Position IDs |

And `batch.non_tensor_batch` contains:

| Key | Description |
|-----|-------------|
| `messages` | Original prompt messages (for decoding prompts/images) |
| `multi_modal_inputs` | Vision inputs (images, image_grid_thw, etc.) |

### 3.2 What the Hook Must Do

```
Step 1: Extract per-sample data from the batch
  ├── For each sample in the batch (index i):
  │   ├── Decode prompt tokens → question text
  │   ├── Extract vision inputs → ImageInput objects
  │   ├── Decode response tokens → rollout_text
  │   ├── Build ConditionInputs (either from evidence cache or from prompt)
  │   └── Create OnlineFCOPDSample
  │
Step 2: Verify each rollout
  └── For each sample:
      └── verify_geometry3k_response() → verifier_result

Step 3: Teacher forced scoring (6 conditions × B samples)
  └── For each sample:
      └── TeacherClient.score_teacher_conditions(
            question, condition_inputs, rollout_text, rollout_token_ids, images
          ) → {Condition: TeacherTopK}

Step 4: Student forced scoring (6 conditions × B samples)
  └── For each sample, render teacher prompt per condition, run
      student forward pass → OnlineStudentScores

Step 5: Compute FC-OPD batch
  └── compute_online_fc_opd_batch(samples, ...) → OnlineFCOPDBatchOutput

Step 6: Convert to verl tensors
  └── online_batch_output_to_verl_tensors(output) → VerlFCOPDTensors

Step 7: Attach to DataProto.batch
  └── batch.batch.update(tensors.as_batch_dict())

Step 8: Return
  └── return batch, {"actor/fc_opd_pre_loss": total_loss, ...}
```

### 3.3 Key Challenges for the Hook

**Challenge 1: Batch teacher scoring is slow.**
6 conditions × B samples = 6B sequential HTTP calls to the teacher service.
For B=8, that's 48 HTTP round-trips per training step.

Mitigation options:
- **Now (smoke)**: Accept the latency for single-GPU smoke. Batch size 1-2.
- **Near-term**: Batch multi-condition requests into one HTTP call (modify `teacher_service.py`).
- **Production**: Port teacher to ZMQ + vLLM following GKD's `teacher/` pattern.

**Challenge 2: Student forced scoring requires rendering teacher prompts.**
The teacher prompt per condition includes `<image>` tags and condition-specific
text. The student model needs to see these prompts to produce condition-aware
log-probs.

Mitigation: The hook can reuse `render_teacher_prompt()` from
`teacher_prompts.py`. The student forced scorer already handles this in the
smoke script (`_HFStudentScorer` in `run_fc_opd_online_train_step_smoke.py`).

**Challenge 3: Vision inputs must survive the batch pipeline.**
`batch.non_tensor_batch["multi_modal_inputs"]` contains images as PIL objects
or tensors. The hook must extract per-sample images and pass them to the
teacher and student scorers.

**Challenge 4: The hook runs on the trainer process (CPU), but student scoring
needs GPU.**
In verl's architecture, the trainer is a CPU-side Ray actor. GPU compute
happens on rollout workers and actor workers.

Options:
- Run student forced scoring on a dedicated GPU worker (add to `ResourcePoolManager`)
- Accept CPU offloading for the student (slow but correct for smoke)
- Run the student forward pass inside the actor patch instead (but then teacher
  scores aren't available yet)

### 3.4 Hook File

```
src/dual_track_opd/fc_opd/verl_post_rollout_hook.py
```

This file exports:

```python
def fc_opd_post_rollout_hook(
    *,
    batch,        # DataProto
    tokenizer,    # verl tokenizer wrapper (has .encode, .decode)
    processor,    # HF processor or None
    config,       # dict
    global_steps, # int
) -> tuple:
    """Post-rollout FC-OPD hook for verl RayPPOTrainer.fit()."""
    ...
```

The FQN in the verl config would be:

```yaml
algorithm:
  fc_opd:
    post_rollout_hook: dual_track_opd.fc_opd.verl_post_rollout_hook.fc_opd_post_rollout_hook
    loss_coef: 0.1
```

## 4. What Codex Should Implement Next

### Priority order:

1. **[P0 done]** Write `src/dual_track_opd/fc_opd/verl_post_rollout_hook.py`
   - `fc_opd_post_rollout_hook()` implements Steps 1-8 from §3.2
   - Handles padded verl response tensors by scoring valid `response_mask`
     tokens and padding `fc_*` tensors back to actor `response_len`
   - Reuses `TeacherClient` for teacher calls when `teacher_url` is configured
   - Requires an explicit `student_scorer`/`student_scorer_fqn` so the hook does
     not silently load a model or consume stale offline scores

2. **[P0]** Write a smoke test that exercises the hook end-to-end
   - Load a model, generate one rollout, run the hook, verify `fc_*` tensors
     appear in `batch.batch` with correct shapes
   - Use the existing online smoke path as reference

3. **[P1]** Extend teacher service to accept batched multi-condition requests
   - One HTTP call with `[sample1_conditions..., sample2_conditions...]`
   - Reduces latency from O(B×C) to O(1) per step

4. **[P1]** Write a Hydra config for FC-OPD training
   - `recipe/fc_opd/config/fc_opd_trainer.yaml`
   - References `algorithm.fc_opd.post_rollout_hook` and `algorithm.fc_opd.loss_coef`

5. **[P2]** Apply patches and run a full FC-OPD training smoke on GPU
   - Apply both patches to `third_party/verl`
   - Run one training step with a real batch
   - Verify `actor/fc_opd_loss` appears in metrics

## 5. Verification Checklist

Before merging, verify:

- [ ] `git apply --check --directory=third_party/verl` both patches clean
- [ ] `pytest tests/fc_opd/ -q` passes (currently 195 passed, 2 skipped)
- [x] Hook function loads via `load_class_from_fqn` without import errors
- [x] Hook returns `(batch, metrics)` tuple with `fc_*` keys in `batch.batch`
- [x] `fc_teacher_topk_indices` has shape `[B, C, T, K]`
- [x] `fc_condition_weights` has shape `[B, C, T]`, all values non-negative
- [x] `fc_condition_ids` matches `DEFAULT_VERL_CONDITION_ORDER`
- [ ] Smoke step: one optimizer step with `fc_opd_coef > 0` changes actor params
- [ ] `actor/fc_opd_loss` > 0 (not collapsed to zero)
- [x] No offline JSONL is read by the hook

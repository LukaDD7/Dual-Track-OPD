# FC-OPD verl Integration Survey

Conducted 2026-06-28. Target: integrate FC-OPD 6-condition on-policy scoring
into the verl training loop.

## 1. verl Repository Status

- **Type**: **Direct clone** (NOT a fork) of `https://github.com/verl-project/verl.git`
- **Version**: v0.7.1, detached HEAD at `bec9ef74`
- **Location**: `third_party/verl/`
- **Branch**: main track (no custom branch)
- **Implication**: Cannot push upstream to verl-project. All FC-OPD changes must
  live in `third_party/verl/` as local patches OR as a new recipe under
  `recipe/fc_opd/` following the established GKD pattern.

## 2. verl On-Policy Distillation Architecture (Current)

### 2.1 The GKD Recipe (`recipe/gkd/megatron/`) — Only OPD Path

This version of verl has NO core distillation module. All on-policy distillation
lives exclusively in `recipe/gkd/megatron/`. The upcoming v0.8+ core
`distillation` module does not exist here.

**File map:**

```
recipe/gkd/megatron/
├── main_gkd.py                    # Hydra entry point
├── ray_trainer.py                 # OnPolicyDistillTrainer(RayPPOTrainer)
├── megatron_workers.py            # Custom actor/rollout workers
├── megatron_distill_losses.py     # KL, RKL, KL+RKL, JSD (vocab-parallel)
├── megatron_kl_loss.py            # Legacy KL loss
├── teacher_utils.py               # get_teacher_knowledge() orchestrator
├── teacher/
│   ├── client.py                  # TeacherClient (ZMQ REQ, threaded)
│   ├── worker.py                  # ZMQ REP worker loop
│   ├── vllm_engine.py             # VLLMEngine wrapper for top-k logprobs
│   ├── proxy.py                   # ZMQ ROUTER/DEALER proxy for scaling
│   ├── utils.py                   # torch.save/load serialization
│   ├── start_server.sh            # Launch proxy + 1 worker
│   └── join_server.sh             # Join additional workers
└── config/
    └── on_policy_distill_trainer.yaml
```

### 2.2 Data Flow (Complete Training Step)

```
Step 1: Rollout Generation
  batch = DataProto.from_single_dict(batch_dict)
  gen_batch = batch.pop(prompt keys)
  gen_batch_output = rollout_wg.async_generate_sequences(gen_batch)

Step 2: Teacher Scoring (Async)
  _async_get_teacher_knowledge():
    future = gen_batch_output.get()
    teacher_output = get_teacher_knowledge(gen_batch_output, teacher_client)
    → teacher_client.submit(per_sequence_token_ids)
    → ZMQ REQ → Teacher Worker (vLLM) → ZMQ REP
    → Returns: (responses, teacher_topk_logps: [seq_len, K], teacher_topk_indices: [seq_len, K])
    → Padded to [batch_size, max_seq_len, K]
    → Packed into DataProto.non_tensor_batch

Step 3: Batch Merge
  batch = batch.union(gen_batch_output)       # responses, input_ids, attention_mask
  batch = batch.union(teacher_batch_output)   # teacher_topk_logps, teacher_topk_indices

Step 4: Actor Update → Loss
  actor_wg.update_actor(batch)
    → forward_backward_batch(data):
      extract teacher_topk_logps/indices from non_tensor_batch
      split into micro-batches
      for each micro-batch (last pipeline stage):
        calc_kl_mask = response_positions_only
        logits_processor(logits, teacher_topk_logps, teacher_topk_indices, calc_kl_mask):
          distill_loss_op(masked_logits, masked_teacher_topk_logps, masked_teacher_topk_indices)
          → per_token_kl_loss
      loss = per_token_kl_loss.mean()  # no per-condition weighting, no chunk gating
```

### 2.3 Integration Points (Exact Line References)

| # | Point | File | Lines | What |
|---|-------|------|-------|------|
| 1 | Teacher client init | `ray_trainer.py` | 170-172 | `TeacherClient(ip, port, n_workers)` |
| 2 | Rollout dispatch | `ray_trainer.py` | 340-366 | `rollout_wg.async_generate_sequences()` |
| 3 | Teacher scoring dispatch | `ray_trainer.py` | 368-390 | `_async_get_teacher_knowledge()` |
| 4 | Teacher result assembly | `teacher_utils.py` | 32-143 | `get_teacher_knowledge()` |
| 5 | Batch merge (rollout) | `ray_trainer.py` | 636 | `batch.union(gen_batch_output)` |
| 6 | Batch merge (teacher) | `ray_trainer.py` | 647 | `batch.union(teacher_batch_output)` |
| 7 | Actor update call | `ray_trainer.py` | 654-656 | `actor_wg.update_actor(batch)` |
| 8 | Teacher data extraction | `megatron_workers.py` | 243-284 | From `non_tensor_batch` |
| 9 | KL loss application | `megatron_workers.py` | 317-329 | `logits_processor` closure |
| 10 | Loss reduction | `megatron_workers.py` | 291-309 | `masked_kl_loss.mean()` |

### 2.4 Available Distillation Losses

| Name | Formula | Config Key |
|------|---------|------------|
| `kl` | KL(P_teacher_topk || Q_student_full) | `distill_loss.name=kl` |
| `rkl` | KL(Q_student_topk_hat || P_teacher_topk_hat) | `distill_loss.name=rkl` |
| `kl_rkl` | (1-r)*KL + r*RKL | `distill_loss.name=kl_rkl` |
| `jsd` | JSD_beta(P_teacher_topk, Q_student_full) | `distill_loss.name=jsd` |

All operate in vocab-parallel mode: `student_logits [total_nnz, V_part]` ×
`teacher_topk [total_nnz, K]`.

### 2.5 Recipe Registration Pattern

verl has **no recipe registry**. Recipes are self-contained packages discovered
by filesystem convention. Each recipe:
1. Has its own `main_*.py` with `@hydra.main`
2. Has its own `ray_trainer.py` subclassing `RayPPOTrainer`
3. Has its own `config/` directory
4. Provides custom `WorkerType` classes mapped via `Role` enum
5. Is invoked as `python -m recipe.<name>.main_<name>`

To add FC-OPD, create `recipe/fc_opd/` following this exact pattern.

## 3. Key Gap: Single-Condition → Multi-Condition

### What GKD Provides
- ONE teacher model
- ONE set of top-k logprobs per token
- ONE kl_loss per token
- Mean reduction → scalar loss

### What FC-OPD Needs
- ONE teacher model queried under **6 different condition contexts**
- Per-condition top-k logprobs: `teacher_scores[condition][t]`
- Per-condition student logprobs: `student_scores[condition][t]`
- Capability attribution: `visual_detail = full - degraded` etc.
- Student deficit gate, chunk compatibility, verifier gate
- Router: `condition_weights[condition][t]`
- Loss: `Σ_c weight[c][t] × KL(teacher_c[t] || student[t])`

### Where the Gap Must Be Bridged

```
Currently:
  get_teacher_knowledge() → ONE set of topk_logps/indices
  forward_backward_batch() → ONE distill_loss_op call
  loss = mean(per_token_kl * calc_kl_mask)

Needed:
  get_teacher_knowledge() → SIX sets of topk_logps/indices (per condition)
  forward_backward_batch() → SIX teacher scores + capability routing
  loss = Σ condition_weight × KL(teacher_condition || student)
```

## 4. Third-Party References

### 4.1 OPSD (On-Policy Student Distillation) — HJSang/OPSD_OnPolicyDistillation
- Builds directly on verl's GKD recipe
- Supports forward KL, reverse KL, JSD
- **Reward-weighted distillation**: combines RL reward signal with distillation
- Multi-turn agent-loop support
- Chunked loss computation for memory efficiency
- **Most relevant**: proves custom loss + multi-signal routing is feasible within verl's GKD architecture

### 4.2 Distill-R1 — ByungKwanLee/Distill-R1
- Teacher generates own rollouts too (two-stream)
- JSD-based losses on both student and teacher data
- Supports GRPO, DAPO, REINFORCE++, RLOO
- FSDP + CPU offloading for memory

### 4.3 verl v0.8+ Core Distillation (Future, Not In Our Version)
- `verl/trainer/distillation/` — core distillation module
- `MultiTeacherModelManager` — route samples to different teachers by `data_source`
- `AsyncTeacherLLMServerManager` — async logprob dispatch
- Chunked top-K for long contexts (≥64K)
- Policy-gradient OPD: negative KL as dense reward
- **Not available in v0.7.1** — would need to upgrade verl to use this

## 5. Recommended Integration Strategy

### Option A: New Recipe `recipe/fc_opd/` (RECOMMENDED)

Create `recipe/fc_opd/megatron/` mirroring the GKD pattern:

```
recipe/fc_opd/
├── main_fc_opd.py                 # Hydra entry point
├── ray_trainer.py                 # FCOPDOnPolicyDistillTrainer
├── megatron_workers.py            # FCOPDActor, FCOPDActorWorker
├── fc_opd_teacher_utils.py        # get_fc_opd_teacher_knowledge()
├── fc_opd_distill_losses.py       # condition-weighted sparse KL
└── config/
    └── fc_opd_trainer.yaml
```

Key changes vs. GKD:
1. **`fc_opd_teacher_utils.py`**: Instead of ONE `get_teacher_knowledge()` call,
   makes 6 calls per batch (one per condition), each with different
   `ConditionInputs` rendered through `render_teacher_prompt()`.

2. **`ray_trainer.py`**: `_async_get_teacher_knowledge()` dispatches 6 parallel
   teacher requests per condition. `batch.union()` merges 6 sets of
   `teacher_topk_logps_{condition}` into `non_tensor_batch`.

3. **`megatron_workers.py`**: `forward_backward_batch()` reads 6 teacher score
   tensors, calls `compute_online_fc_opd_batch()` (from `online_batch.py`) to
   compute condition weights, then feeds per-condition weighted KL into loss.

4. **`fc_opd_distill_losses.py`**: Thin wrapper — reuses `VocabParallelDistillLoss`
   from GKD for each condition's KL term, then applies condition weights.

### Option B: Patch GKD Recipe In-Place

Modify `recipe/gkd/megatron/` directly to add 6-condition support. Faster but
dirties the upstream GKD code. Not recommended for clean research tracking.

### Option C: Upgrade verl to v0.8+ and Use Core Distillation

verl v0.8+ has `MultiTeacherModelManager` designed for multi-teacher routing.
FC-OPD could register each condition as a "teacher" with the same model but
different prompt contexts. This is architecturally cleaner but requires
upgrading verl, which may break GKD compatibility.

## 6. Teacher Service: HTTP vs ZMQ

| Aspect | GKD's ZMQ Teacher | FC-OPD's HTTP Teacher |
|--------|-------------------|----------------------|
| Protocol | ZMQ REQ/REP + proxy | HTTP REST (port 18080) |
| Backend | vLLM | Transformers (HF) |
| Serialization | torch.save/load | JSON + base64 tensors |
| Multi-condition | Single prompt → one output | 6 prompts → needs 6 forward passes |
| Batch efficiency | Queue-based micro-batching | Per-request, no batching |

**Gap**: Our HTTP teacher service (`teacher_service.py`) processes one condition
at a time. For online training with batch_size > 1 and 6 conditions, this
creates 6×B sequential HTTP calls. The verl GKD pattern of ZMQ + micro-batching
is more efficient.

**Recommendation**: Either:
1. Extend HTTP teacher to accept batched multi-condition requests
2. Port FC-OPD teacher to ZMQ pattern (following `recipe/gkd/megatron/teacher/`)
3. Use vLLM backend for teacher (already supported in GKD) for faster inference

## 7. Online Smoke Result (2026-06-28)

First end-to-end on-policy FC-OPD step executed successfully:

```
loss: 0.221
selected_parameter_delta_norm: 4.1e-08  (parameter updated)
rollout_num_tokens: 223
teacher_conditions: [full, degraded, free, task_visible, task_infer, task_solve]  (6/6)
verifier_outcome: wrong_but_format_valid

Per-condition loss:
  task_solve:   0.511  ← largest deficit
  task_infer:   0.365
  task_visible: 0.339
  full:         0.078
  degraded:     0.000  (student not yet sensitive to degradation)
  free:         0.000  (student not yet sensitive to free-text)

Per-chunk loss:
  visible_evidence:  0.246  ← most signal
  reasoning:         0.147
  answer:            0.002  ← verifier gate suppressed (correct=0)
  diagram_inference: 0.000

Verifier gate: wrong_valid answer_chunk_weight=0, non-answer weights higher ✓
```

## 8. Action Items for Codex

1. **[P0]** Design `recipe/fc_opd/` following the GKD recipe pattern
2. **[P0]** Extend `get_teacher_knowledge()` to support 6-condition scoring
   (or write a new `fc_opd_teacher_utils.py` that calls our HTTP teacher 6×)
3. **[P0]** Modify `forward_backward_batch()` to consume 6-condition scores
   and call `compute_online_fc_opd_batch()`
4. **[P1]** Implement `fc_opd_distill_losses.py` with condition-weighted sparse KL
5. **[P1]** Write Hydra config for FC-OPD training
6. **[P2]** Port teacher service to ZMQ + vLLM for production throughput
7. **[P2]** Investigate verl v0.8+ core distillation for multi-teacher routing

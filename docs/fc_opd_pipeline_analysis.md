# FC-OPD Pipeline Analysis & Design Rationale

Written 2026-06-29 after end-to-end 8-GPU smoke validation.

## 1. End-to-End Pipeline (5 Layers)

### Layer 1 — Data Loading

```
train.parquet (2 rows, Geometry3K)
  → FCOPDDataset.__getitem__()
    → 读到 {prompt, images, question, choices, answer, condition_inputs}
    → 所有 FC-OPD 字段 copy 进 extra_info（verl AgentLoop 只保留这个）
    → tokenizer + processor 编码 prompt
    → 返回 {input_ids, attention_mask, image_grid_thw, extra_info}
  → DataLoader 组装 batch [B=2]
  → AgentLoopWorker 分发到 vLLM
```

**Why custom dataset (FCOPDDataset):** verl's AgentLoop drops non-standard
columns. Only `extra_info` survives. FCOPDDataset copies `question`, `choices`,
`answer`, `condition_inputs` into `extra_info` at load time.

### Layer 2 — Rollout (vLLM Generation)

```
vLLMHttpServer (Student 4B model, GPU)
  → 接收 prompt tokens + images
  → 自回归生成 response tokens [B, T=512]
  → 返回:
    - batch.batch["responses"]      [B, T]    rollout token IDs
    - batch.batch["response_mask"]  [B, T]    bool: which tokens are model-generated
    - batch.batch["old_log_probs"]  [B, T]    log-probs at generation time (for PPO ratio)
```

**Key:** vLLM generates from the STUDENT model (4B), NOT the teacher.
The 6 conditions are only used during scoring (Layer 3), not generation.

### Layer 3 — FC-OPD Hook (Core Innovation) ⭐

**Insertion point:** `RayPPOTrainer.fit()`, after rollout + response_mask,
before advantage estimation. Trainer patch: 26 lines.

```
fc_opd_post_rollout_hook(batch, tokenizer, processor, config)
  │
  ├─ 1. Build samples from batch rows → OnlineFCOPDSample
  │     sample = {
  │       question,              ← extra_info (NOT prompt decode!)
  │       rollout_token_ids,     ← response_mask 选出的有效 tokens
  │       condition_inputs,      ← 6 个条件的差异证据
  │       choices, answer        ← 给 verifier 用
  │     }
  │
  ├─ 2. Teacher Scoring (32B model, separate GPU, HTTP)
  │     _build_teacher_scorer()
  │     → score_teacher_conditions_multi_sample(samples, conditions)
  │     → 一次 HTTP POST 给 Teacher Service
  │     → Teacher 对每个 condition 构建 prompt + 图像
  │     → Forced forward: teacher prompt + student rollout tokens
  │     → 返回 per-condition top-K teacher log-probs [B, C, T, K]
  │
  ├─ 3. Student Scoring (4B model, Ray GPU actor or local CUDA)
  │     _build_student_scorer()
  │     → Direct CUDA instantiation (local smoke)
  │     → OR Ray GPU actor proxy (CPU-only TaskRunner)
  │     → Student 对每个 condition 构建 prompt
  │     → Forced forward: student prompt + student rollout tokens
  │     → 返回 per-condition student log-probs [B, C, T]
  │
  ├─ 4. compute_online_fc_opd_batch()
  │     │
  │     ├─ parse_response_chunks() — XML tag parsing
  │     │   <visible_evidence>, <diagram_inference>, <reasoning>, <answer>
  │     │   → 4 bool masks, [1, T] each, mutually exclusive
  │     │
  │     ├─ compute_student_deficit_capability_scores()
  │     │   For each capability (visual_detail, evidence_selection,
  │     │   visual_text_inference, solving):
  │     │
  │     │   teacher_delta = t_pos_logprob - t_neg_logprob
  │     │   student_delta = s_pos_logprob - s_neg_logprob
  │     │   teacher_attribution = max(0, teacher_delta)
  │     │   student_deficit = max(0, teacher_delta - student_delta)
  │     │   final_weight = attribution × deficit × chunk_compat × verifier_gate
  │     │
  │     ├─ route_condition_weights()
  │     │   Maps capability scores → condition weights per token
  │     │   Router mode: student_deficit_chunk_gated
  │     │   Each token gets ≤2 conditions, weights normalised to 1.0
  │     │
  │     └─ compute_fc_opd_loss()
  │       sparse_forward_kl(student_logits, teacher_topk, condition_weights)
  │
  ├─ 5. Write to batch
  │     batch.batch["fc_teacher_topk_indices"]     [B, C, T, K=32]
  │     batch.batch["fc_teacher_topk_log_probs"]   [B, C, T, K=32]
  │     batch.batch["fc_condition_weights"]          [B, C, T]
  │     batch.batch["fc_opd_coef"]                   [B]
  │     batch.non_tensor_batch["fc_condition_ids"]   [B, C]  (non-tensor: shape≠[B])
  │     batch.non_tensor_batch["fc_opd_loss_mode"]   [B]     (non-tensor: string)
  │
  └─ 6. Return metrics
       fc_opd/hook_loss, fc_opd/hook_active_weight, ...
```

### Layer 4 — PPO Training Loop

```
RayPPOTrainer.fit() — one step:
  │
  ├─ 1. Rollout (vLLM)                ← generate responses [38.8s total]
  ├─ 2. Compute old_log_probs          ← actor forward on responses
  ├─ 3. Compute response_mask
  ├─ 4. ★ FC-OPD Hook                  ← inject teacher/student tensors
  ├─ 5. Compute rewards                ← Geometry3K answer matching
  ├─ 6. Compute advantages (GRPO)      ← group-relative advantage
  └─ 7. PPO update (dp_actor)          ← FC-OPD tensors consumed here
```

### Layer 5 — Actor Parameter Update (dp_actor.py patch)

```python
# dp_actor.update_policy(micro_batch):
model.forward() → student_logits [B, T, vocab]

# 1. PPO policy loss (GRPO)
policy_loss = -min(ratio * advantages, clip(ratio, 0.8, 1.2) * advantages)

# 2. FC-OPD sparse KD loss
fc_coef = micro_batch.batch["fc_opd_coef"]          # 0.1
topk_idx  = micro_batch.batch["fc_teacher_topk_indices"]
topk_prob = micro_batch.batch["fc_teacher_topk_log_probs"]
weights   = micro_batch.batch["fc_condition_weights"]

if loss_mode == "reverse":
    fc_opd = compute_verl_sparse_reverse_kl(logits, topk_idx, topk_prob, weights)
else:
    fc_opd = compute_verl_sparse_topk_kd(logits, topk_idx, topk_prob, weights)

# 3. Total loss
total_loss = policy_loss + fc_coef * fc_opd_loss
total_loss.backward()
optimizer.step()
```

---

## 2. Key Design Decisions & Rationale

### 2.1 Forward KL vs Reverse KL

**Current: Forward KL** = KL(P_teacher || P_student)

```
per_token = Σ_k teacher_p[k] * log(teacher_p[k] / student_p[k]) + tail_term
```

- **Behavior:** Mode-covering. Student must assign probability wherever
  teacher does. Penalizes heavily when student ignores a teacher mode.
- **Problem for 32B→4B:** 4B model lacks capacity to cover all 32B modes.
  Forward KL can introduce noise by forcing the small model to spread its
  probability too thin.
- **Implemented:** `compute_verl_sparse_topk_kd` (verl_sparse_kd.py)

**Proposed: Reverse KL** = KL(P_student || P_teacher)

```
student_probs = softmax(student_logits)
P_s_at_topk = gather(student_probs, teacher_topk_indices)
reverse_kl = Σ_k P_s(k) * (log P_s(k) - log P_t(k)) + tail_term
```

- **Behavior:** Mode-seeking. Student is penalized where IT places mass
  but teacher disagrees. Ignores teacher modes student doesn't cover.
- **Better for RLVR:** Only corrects the student at its actual choices.
- **Implemented:** `compute_verl_sparse_reverse_kl` (verl_sparse_kd.py)
- **Toggle:** `--loss-mode reverse` in the smoke script.

### 2.2 Why Top-K=32 + Tail (Not Full Vocabulary)

The teacher returns top-32 token log-probs per position + a `tail_log_prob`
capturing all remaining probability mass:

```
P_teacher = top32_prob + tail_prob  (tail = 1 - Σ top32_prob)
KL = Σ_{top32}(P_t×log(P_t/P_s)) + P_tail×(log P_tail - log(1-Σ_top32_P_s))
```

- K=32 captures 30–50% of probability mass for a 32B model
- Tail bucket ensures KL is well-defined (covers full distribution)
- K=100 would capture 60–80% for better precision at 3.1× memory cost
- **Recommendation:** Validate with K=32, compare with K=100 for production

### 2.3 Capability Deficit: Why Δlog-prob, Not JSD

The router computes **student capability deficits** using sampled-token
log-probability deltas, NOT Jensen-Shannon divergence:

```python
# For visual_detail: contrast FULL vs DEGRADED
t_pos = teacher_logprob[FULL][rollout_token]     # "what teacher thinks of this token under FULL"
t_neg = teacher_logprob[DEGRADED][rollout_token] # "what teacher thinks under DEGRADED"
s_pos = student_logprob[FULL][rollout_token]
s_neg = student_logprob[DEGRADED][rollout_token]

teacher_delta = t_pos - t_neg   # teacher: how much does FULL help?
student_delta = s_pos - s_neg   # student: how much does FULL help?

teacher_attribution = max(0, teacher_delta)     # this token needs visual detail
student_deficit     = max(0, teacher_delta - student_delta)  # student gap
```

**Why this is better than JSD:**
- JSD measures overall distribution difference — doesn't care what token
  the student actually chose
- Δlog-prob measures the gap specifically at the student's rollout token
- Aligns with RLVR philosophy: only care about what the student actually did

### 2.4 Chunk Compatibility Gate (chunk_compat)

Not all capabilities are relevant to all response chunks:

```
CHUNK_CAPABILITY_COMPATIBILITY:
                        visual_detail  evidence_sel  infer  solving
visible_evidence:          1.0            1.0        0.25    0.0
diagram_inference:         0.25           0.25       1.0     0.25
reasoning:                 0.0            0.25       0.75    1.0
answer:                    0.0            0.0        0.25    1.0
```

- `visible_evidence × evidence_selection = 1.0`: Extracting visual evidence
  IS the purpose of this chunk
- `answer × visual_detail = 0.0`: The answer choice doesn't depend on image quality
- Suppresses noise: deficit signals in irrelevant chunks are zeroed out

### 2.5 Verifier Learning Value Gate (verifier_gate)

Different learning values for correct vs incorrect responses:

```
correct:            answer=0.00  (don't unlearn what already works)
wrong_but_valid:    answer=0.50, reasoning=0.75, visible_evidence=1.00
malformed:          answer=0.00, reasoning=0.00
```

This implements the core RLVR idea: **focus FC-OPD correction on incorrect
samples, and within those, on the chunks most responsible for the error.**

### 2.6 Loss Formula — Full Derivation

```
per_condition[c][t] = Σ_k teacher_p[k] × log(teacher_p[k] / student_p[k]) + tail_term
                       └──────┬──────┘   └──────────────┬──────────────┘
                         teacher weight        log-ratio (information gain)
                         (how confident is     (how much worse is student
                          teacher on this      at this specific token?)
                          token?)

per_token[t] = Σ_c per_condition[c][t] × condition_weights[c][t]
               └──────────────────────┘   └────────────────────┘
               per-condition KL loss       router-assigned weight
                                           (which condition "owns" this token)

fc_opd_loss = (per_token × response_mask).sum() / active_weight.sum()
              └─────────────────────────┘    └─────────────────────┘
              total weighted KL over         normalisation factor
              valid response tokens          (condition_weights sum)
```

**Why `teacher_p × log(teacher_p/student_p)`?**
This is the definition of KL divergence: expectation under the teacher
distribution of the log-ratio. The teacher's probability mass acts as the
weight — tokens the teacher is confident about matter more.

**Example** (token at position t):
| token k | teacher_p | student_p | teacher_p × log(ratio) |
|---------|-----------|-----------|----------------------|
| angle   | 0.6       | 0.05      | 0.6×log(12)=**1.49** |
| triangle| 0.3       | 0.25      | 0.3×log(1.2)=**0.05**|
| tail    | 0.1       | 0.7       | 0.1×log(0.14)=**−0.19**|

→ per_condition[t] = 1.49 + 0.05 − 0.19 = **1.35** (nats of KL at this position)

---

## 3. GPU Resource Allocation (8×H200)

```
GPU 0: Teacher Service (Qwen3-VL-32B, 66GB/141GB)
GPU 1: verl WorkerDict[0] (10.8GB) + vLLM Worker (0.9GB)
GPU 2: verl WorkerDict[1] (10.8GB) + vLLM Worker (0.9GB)
GPU 3: StudentScorer Ray Actor (Qwen3-VL-4B, ~8GB)
GPU 4–7: idle (keep-alive / future scale-out)
```

**Why StudentScorer needs a separate GPU:**
The hook runs in TaskRunner (`ray.remote(num_cpus=1)`, CPU-only). Direct CUDA
instantiation fails → falls back to Ray GPU actor (`ray.remote(num_gpus=1)`).
This actor loads a separate 4B model copy (~8GB). Future optimization: score
inside the actor worker (reuse actor model, eliminate duplicate load).

---

## 4. Smoke Test Results (2026-06-29)

### 4.1 Metrics

| Metric | Value | Status |
|--------|-------|--------|
| actor/fc_opd_loss | 0.327 | ✅ Non-zero — KD signal active |
| fc_opd/hook_active_weight | 208.4 | ✅ Router assigned weights |
| actor/fc_opd_coef | 0.1 | ✅ Coefficient correctly wired |
| actor/grad_norm | 8.67 | ✅ Gradients flowing |
| actor/pg_loss | −0.999 | Expected (reward=1.0, no discrimination) |

### 4.2 Per-Condition Breakdown

| Condition | Per-Token Loss | Selection % |
|-----------|---------------|-------------|
| full | 0.114 | 18.5% |
| task_solve | 0.298 | 9.1% |
| task_infer | 0.469 | 7.7% |
| task_visible | 0.418 | 3.7% |
| degraded | 0.0 | 0% |
| free | 0.0 | 0% |

degraded/free got zero selection because the student_deficit_chunk_gated
router found no capability deficit signals for these conditions in the
2-sample Geometry3K batch.

### 4.3 Timing

| Phase | Time (s) |
|-------|----------|
| Rollout generation | 15.0 |
| FC-OPD Hook (Teacher + Student) | 19.7 |
| Old log-probs | 0.7 |
| Actor update (forward + backward) | 1.1 |
| **Total per step** | **38.8** |

---

## 5. Future Directions

### P0 — Real Reward Function
Switch from dummy `smoke_reward.py` (always returns 1.0) to Geometry3K
answer matching. This enables GRPO to actually discriminate correct vs
incorrect responses, and the verifier_learning_value_gate to focus FC-OPD
on wrong samples only.

### P1 — Reverse KL Comparison
Run the same smoke with `--loss-mode reverse` to compare:
- Loss magnitudes
- Gradient norms
- Per-condition selection patterns
- Training stability over multiple steps

### P2 — Multi-Step Training & Scaling
- Run 10+ steps to verify loss doesn't collapse
- Scale to batch_size=32 with real Geometry3K data
- Test with 4+ GPUs for FSDP
- Switch to K=100 top-K for higher precision

### P3 — Student Scoring in Actor Worker
Move StudentScorer from standalone Ray actor to inside the actor worker:
- Eliminates duplicate 4B model load (saves ~8GB GPU memory)
- Eliminates CUDA↔CPU tensor serialization overhead
- Reference: GKD's `_async_get_teacher_knowledge` pattern

### P4 — Teacher Service: ZMQ + vLLM Backend
- Replace HTTP with ZMQ for lower latency
- Use vLLM async engine for batched teacher forward passes
- Support dynamic batching across workers

---

## 6. Running Instructions

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD

# Forward KL (current default)
CC=/usr/bin/gcc bash scripts/hpc/run_verl_fc_opd_smoke.sh --gpus 4 --top-k 32 --steps 1

# Reverse KL
CC=/usr/bin/gcc bash scripts/hpc/run_verl_fc_opd_smoke.sh --gpus 4 --top-k 32 --steps 1 --loss-mode reverse

# Dry-run to check config
bash scripts/hpc/run_verl_fc_opd_smoke.sh --gpus 4 --dry-run
```

# Visual Grounding Gap OPD: Executable Design Plan

## 0. Goal

We want an OPD method for VLMs that keeps the core strength of VA-OPD:

- student generates on-policy rollouts;
- teacher forced-scores the exact student rollouts;
- distillation loss is applied on student-visited states;
- visual supervision is not diluted by language template tokens.

The extension is not "more teacher conditions". The extension is:

> Distill more strongly where the teacher counterfactually relies on visual evidence and the student shows weaker intrinsic visual grounding on the same prefix/token/chunk.

This avoids the earlier subject/object confusion from teacher-generated captions. Caption/evidence generation can remain a diagnostic or data-audit tool, but it should not be the primary training condition in the first clean OPD implementation.

Relevant references:

- VA-OPD: Visual-Advantage On-Policy Distillation for Vision-Language Models, arXiv 2605.21924.
- VGPO: Visually-Guided Policy Optimization for Multimodal Reasoning, ACL 2026.
- KAWHI: Bridging Visual Representation and Reinforcement Learning from Verifiable Rewards in Large Vision-Language Models, arXiv 2603.27375.

## 1. Core Conceptual Alignment

### 1.1 Prediction-state alignment

For a generated response:

```text
y = (y_1, ..., y_T)
```

the probability of token `y_t` is produced from the prefix state:

```text
x, y_<t
```

In a causal LM implementation, logits are shifted:

```text
hidden state at prefix position -> logits for next token y_t
```

Therefore, token attribution for `y_t` must use the hidden state that predicts `y_t`, not a hidden state after consuming `y_t` unless the implementation explicitly shifts it back.

Implementation rule:

```text
token t logprob, KL, VA, and visual-focus attribution must all align to the same prediction event:
P(y_t | image, question, y_<t)
```

This matters because the hidden state after consuming `y_t` represents the model state after the sampled token is already known. It is useful for next-token prediction but is not the cleanest attribution for why `y_t` was chosen.

### 1.2 Hidden state vs activation vs logits

Use the following vocabulary:

- `hidden state`: the transformer residual vector at a sequence position/layer.
- `activation`: broad term; hidden states, MLP activations, attention scores, etc. are all activations.
- `logits`: vocabulary-sized vector computed from a hidden state through the LM head.
- `logprobs`: softmax/log-softmax over logits.

For our method:

- KL uses logits/logprobs.
- Visual grounding attribution uses hidden states.
- Teacher visual advantage uses teacher logprobs under full vs degraded image.

## 2. On-Policy Data Flow

For each prompt `x = (I, q)`:

1. Student rollout:

```text
y^(k) ~ pi_S(. | I, q), k = 1..K
```

2. Teacher forced-forward on each student rollout under full image:

```text
T_full: P_T(. | I, q, y_<t)
```

3. Teacher forced-forward under degraded image:

```text
T_deg: P_T(. | degraded(I), q, y_<t)
```

4. Student forced-forward under full image:

```text
S_full: P_S(. | I, q, y_<t)
```

5. Optional verifier:

```text
verify(y^(k)) -> correct | wrong_but_format_valid | malformed
```

The training target is still the teacher full-image distribution. Degraded teacher and hidden-state visual focus are used only to decide where OPD should be stronger.

## 3. Signals

### 3.1 Teacher Visual Advantage

For token `y_t` in rollout `k`:

```text
VA_raw_{k,t} = log P_T_full(y_t | I, q, y_<t)
             - log P_T_deg(y_t | degraded(I), q, y_<t)
```

Do not overwrite this signed value.

Why:

- positive: teacher supports this token more when fine visual detail is available;
- near zero: token is not visually sensitive under this contrast;
- negative: degraded condition supports the token more, often noise or spurious reliance.

For training weights, convert to a nonnegative quantity only after diagnostics:

```text
VA_pos_{k,t} = max(VA_raw_{k,t}, 0)
```

or use rank/top-quantile membership. The raw signed value must stay in the JSONL for audit.

### 3.2 Teacher and Student Visual Focus Score

For model `M in {T, S}`:

1. Run forced-forward with hidden states enabled.
2. Identify visual token positions in the multimodal prompt.
3. Identify prediction states for response tokens.
4. Choose hidden layer:

```text
default: last layer
optional: mean of last 4 layers for stability
```

5. Build visual prototype:

```text
mu_v^M = mean_pool(hidden_states_of_visual_tokens)
```

6. Compute token visual focus:

```text
vfs^M_{k,t} = 0.5 * (cos(h^M_{k,t,predict}, mu_v^M) + 1)
```

This is VGPO-like. VGPO uses hidden-state similarity between generated-token states and a visual prototype to obtain visual focus scores, then uses intra-trajectory and inter-trajectory reweighting to reinforce visual faithfulness.

Important:

Teacher and student hidden spaces are not directly comparable. Do not compare raw cosine scores across models.

Use within-model normalization:

```text
vfs_rank^M_{k,t} = percentile_rank(vfs^M_{k,t} within rollout k)
```

or robust z-score:

```text
vfs_z^M_{k,t} = (vfs^M_{k,t} - median(vfs^M_k)) / MAD(vfs^M_k)
```

Recommended first version:

```text
vfs_rank in [0, 1]
```

Then define student visual grounding gap:

```text
gap_raw_{k,t} = vfs_rank^T_{k,t} - vfs_rank^S_{k,t}
gap_pos_{k,t} = max(gap_raw_{k,t}, 0)
```

Meaning:

```text
The teacher treats this prediction event as more visually grounded than the student does, relative to each model's own rollout-level visual focus distribution.
```

### 3.3 Verifier Learning-Value Gate

Use verifier outcome as learning-value, not correct-only filtering.

Default gates:

| Outcome | visible_evidence | diagram_inference | reasoning | answer |
|---|---:|---:|---:|---:|
| correct | 0.25 | 0.25 | 0.10 | 0.00 |
| wrong_but_format_valid | 1.00 | 1.00 | 0.75 | 0.50 |
| malformed | 0.25 | 0.10 | 0.00 | 0.00 |

Rationale:

- OPD is most valuable on student-visited failure states.
- Correct rollouts have lower learning value and higher teacher-style-noise risk.
- Malformed rollouts have limited value except early visual/format chunks.

## 4. Chunk Mapping

Use the structured response format:

```xml
<visible_evidence>...</visible_evidence>
<diagram_inference>...</diagram_inference>
<reasoning>...</reasoning>
<answer>...</answer>
```

For each response token, assign:

```text
chunk(t) in {visible_evidence, diagram_inference, reasoning, answer, outside}
```

The chunk parser must return token spans, not just character spans. If char spans are easier, map char spans to token spans through tokenizer offset mapping when available; otherwise tokenize each chunk boundary deterministically and verify round-trip.

## 5. Stable Signal Propagation

The upper-level problem:

> Which student-visited prediction events should receive stronger teacher KL because they are visually important, under-grounded in the student, and valuable for learning?

We separate this into four factors:

1. Teacher visual reliance: `VA`.
2. Student visual grounding gap: `gap`.
3. Structural location: chunk.
4. Learning value: verifier outcome.

Do not directly multiply arbitrary raw scores. Use grouped normalization.

### 5.1 Rollout-level VA weight

For each rollout `k`, compute a robust visual reliance summary:

```text
rollout_va_k = mean(top 20% of VA_pos_{k,t} over response tokens)
```

Then normalize among sibling rollouts for the same prompt:

```text
w_rollout_k = softmax(rollout_va_k / tau_rollout)
```

Rescale to mean 1 across the K rollouts:

```text
w_rollout_k = K * softmax(rollout_va_k / tau_rollout)
```

This follows the VA-OPD intuition: more visually dependent rollouts receive larger OPD weight.

Default:

```text
top_q = 0.20
tau_rollout = 1.0
```

But log summary must report sensitivity for `top_q in {0.10, 0.20, 0.30}` if cheap.

### 5.2 Chunk-level visual grounding gap

For each chunk `c`:

```text
chunk_gap_{k,c} = mean(top 20% of gap_pos_{k,t}, t in chunk c)
```

Then normalize within the rollout:

```text
chunk_gap_rank_{k,c} = percentile_rank(chunk_gap_{k,c} among chunks in rollout k)
```

Convert to a smooth gate:

```text
g_gap_{k,c} = sigmoid((chunk_gap_rank_{k,c} - 0.5) / tau_chunk)
```

Default:

```text
tau_chunk = 0.15
```

Meaning:

- chunks where teacher-student visual focus gap is relatively high get stronger OPD;
- chunks where both models are similarly grounded do not receive extra pressure;
- no cross-model raw cosine comparison is used.

### 5.3 Token-level VA grouping inside each chunk

Within each rollout/chunk, split tokens by teacher VA:

```text
HighVA_{k,c} = top 20% tokens by VA_pos within chunk c
LowVA_{k,c} = remaining valid tokens in chunk c
```

If a chunk has too few tokens:

```text
min_high_tokens = 1
```

This follows VA-OPD's key anti-dilution idea:

```text
high-VA tokens and low-VA tokens are averaged separately, so a small number of critical visual tokens is not washed out by abundant language scaffolding.
```

### 5.4 Grouped KL loss

Let:

```text
KL_{k,t} = KL(P_T_full(. | I, q, y_<t) || P_S(. | I, q, y_<t))
```

In practice, use sparse top-k KL over teacher top-k tokens plus optional tail mass.

For each rollout/chunk:

```text
L_high_{k,c} = mean(KL_{k,t}, t in HighVA_{k,c})
L_low_{k,c}  = mean(KL_{k,t}, t in LowVA_{k,c})
```

Chunk loss:

```text
L_{k,c} =
  verifier_gate(outcome_k, c)
  * w_rollout_k
  * [
      alpha_high_{k,c} * L_high_{k,c}
      + alpha_low * L_low_{k,c}
    ]
```

where:

```text
alpha_high_{k,c} = 1.0 + lambda_gap * g_gap_{k,c}
alpha_low = 0.25
lambda_gap = 1.0
```

This is the stable version of:

```text
w_t = VA_group_weight_t * sigmoid(visual_gap_t / tau)
```

but implemented as grouped means rather than noisy per-token multiplication.

Why grouped means:

- VA-OPD shows VA is concentrated in a small minority of tokens.
- Mean over all tokens dilutes the signal.
- Per-token raw multiplication is noisy.
- Chunk/group averaging preserves signal while controlling variance.

### 5.5 Total loss

```text
L_OPD = sum_k sum_c L_{k,c} / normalizer
```

Recommended normalizer:

```text
sum of active rollout/chunk weights
```

not total token count. Log both:

- token-mean loss;
- active-weight-normalized loss.

## 6. Top-k KL Implementation

Teacher forced-forward saves:

```text
teacher_topk_token_ids_{k,t}
teacher_topk_logprobs_{k,t}
teacher_tail_logprob_{k,t}
teacher_logprob_sampled_token_{k,t}
```

Student training forward computes logprobs at the teacher top-k token ids:

```text
student_logprob_on_teacher_topk
student_tail_logprob
```

Then compute sparse KL:

```text
KL_topk_tail(P_T_full || P_S)
```

VA uses sampled-token logprob:

```text
VA_raw = logp_T_full(sampled y_t) - logp_T_deg(sampled y_t)
```

The KL target is teacher full-image distribution, not degraded distribution.

## 7. Diagnostics Before Training

Before any real training, run diagnostics on 64 or 256 prompts with K=4 or K=8.

Required reports:

### 7.1 Distribution diagnostics

- VA_raw mean/std/quantiles.
- VA_pos sparsity.
- rollout_va distribution.
- teacher vfs distribution.
- student vfs distribution.
- gap_pos distribution.
- correlation between VA_pos and gap_pos.

### 7.2 Chunk diagnostics

For each chunk:

- token count;
- mean VA_pos;
- top20 VA_pos;
- mean gap_pos;
- top20 gap_pos;
- mean KL;
- active OPD weight.

### 7.3 Human-readable examples

For top-N weighted tokens/chunks, decode:

```text
sample_uid
question
response chunk
top weighted token text
VA_raw
VA_pos
teacher_vfs_rank
student_vfs_rank
gap_pos
verifier outcome
final weight
```

The method should be abandoned or simplified if high-weight tokens are mostly formatting, boilerplate, or answer-style artifacts.

## 8. Ablation Plan

Run in this order:

### A0: VA-OPD reproduction baseline

Use only:

```text
rollout_va softmax
highVA/lowVA grouped KL
```

No hidden-state gap.

Purpose:

```text
Confirm our implementation can reproduce the VA-OPD mechanism.
```

### A1: Hidden-state diagnostics only

Compute teacher/student visual focus and gap, but do not use it in loss.

Purpose:

```text
Check whether gap_pos is meaningful and whether it aligns with human-readable visual reasoning chunks.
```

### A2: Chunk gap gated OPD

Enable:

```text
alpha_high = 1 + lambda_gap * g_gap_chunk
```

Purpose:

```text
Test whether student-specific grounding gap improves over teacher-only VA.
```

### A3: Verifier learning-value gate

Enable failure-focused verifier gates.

Purpose:

```text
Use wrong-but-format-valid rollouts as high-value OPD states while reducing style-copying from correct rollouts.
```

### A4: KAWHI-style paragraph/chunk aggregation

Compare XML chunking against automatic paragraph segmentation.

Purpose:

```text
Check whether explicit chunk schema is necessary or paragraph-level credit is sufficient.
```

## 9. Implementation Tasks for Codex

### 9.1 Do not modify third_party/verl initially

Build a standalone prototype first:

```text
scripts/hpc/build_visual_grounding_gap_opd_batch.py
```

Inputs:

- dataset path;
- student model;
- teacher model/service;
- K rollouts;
- degraded mode;
- output JSONL.

Outputs:

- rollout JSONL with token ids;
- teacher full top-k;
- teacher degraded sampled-token logprobs;
- student full sampled-token logprobs/top-k optional;
- teacher/student visual focus scores;
- VA;
- gap;
- chunk spans;
- verifier outcome;
- final weights.

### 9.2 Add forced-forward hidden-state scorer

Create:

```text
dual_track_opd/fc_opd/hidden_state_visual_focus.py
```

Functions:

```text
extract_visual_token_positions(...)
extract_prediction_state_positions(...)
compute_visual_prototype(...)
compute_visual_focus_scores(...)
rank_normalize_scores(...)
compute_teacher_student_visual_gap(...)
```

Tests:

- response token count equals score count;
- prediction-state shift is correct;
- no future token leakage;
- teacher/student raw dimensions are never directly compared;
- rank-normalized scores are in [0, 1].

### 9.3 Add VA/grouped KL loss module

Create:

```text
dual_track_opd/fc_opd/visual_grounding_gap_loss.py
```

Functions:

```text
compute_va_raw(...)
compute_rollout_va_weights(...)
split_high_low_va_groups(...)
compute_chunk_gap_gate(...)
compute_grouped_sparse_kl_loss(...)
```

Tests:

- highVA group is averaged separately from lowVA group;
- highVA minority is not diluted by long lowVA chunk;
- rollout weights sum to K;
- signed VA is preserved;
- final training weights are nonnegative;
- correct answer chunk gate is zero by default.

### 9.4 Add diagnostics renderer

Create:

```text
scripts/hpc/inspect_visual_grounding_gap_batch.py
```

It prints:

- top weighted examples;
- per-chunk score table;
- VA/gap histograms as JSON summaries;
- warnings if top weighted tokens are mostly XML tags or punctuation.

### 9.5 Training integration

Only after standalone diagnostics pass:

- integrate as pure OPD auxiliary-free training;
- no GRPO in first training run;
- no teacher-generated caption conditions in the main path;
- no offline replay from stale policies.

## 10. Key Design Decisions

### 10.1 Why not direct caption conditions?

Because teacher-generated captions/inferences change the input modality and can introduce subject/object confusion:

```text
Teacher writes evidence -> scorer evaluates student rollout under text-evidence condition.
```

This no longer cleanly answers whether the student's own visual grounding is weak on the raw image prompt.

Use captions only for:

- diagnostics;
- interpretability;
- possible future hint-injection OPD variant.

### 10.2 Why use teacher VA?

Teacher VA is a counterfactual visual reliance signal:

```text
Would the teacher support this student token less if fine visual information were removed?
```

This is stronger than raw hidden-state focus alone because it tests sensitivity to image degradation.

### 10.3 Why add student hidden-state gap?

Teacher VA alone says:

```text
The teacher relies on vision here.
```

It does not say:

```text
The student lacks visual grounding here.
```

The teacher/student visual focus gap adds a student-specific deficiency signal.

### 10.4 Why chunk aggregation?

Token-level visual attribution is noisy. KAWHI's paragraph-level credit reallocation suggests that reasoning units are often larger than individual tokens. Our XML chunks provide a controlled version of that idea.

Use chunks to stabilize:

- visual grounding gap;
- verifier gate;
- grouped KL budgets.

### 10.5 Why not raw multiplication?

Raw multiplication:

```text
VA_t * gap_t
```

is a brittle AND gate. It can erase useful signal if either proxy is noisy.

The stable version is:

```text
VA controls rollout and high/low token grouping.
gap controls chunk-level high-group amplification.
verifier controls learning value by outcome/chunk.
```

This preserves the semantics of each signal.

## 11. Minimal First Experiment

Dataset:

```text
Geometry3K train subset
```

Models:

```text
student: Qwen3-VL-4B-Instruct
teacher: Qwen3-VL-32B-Instruct
```

Rollouts:

```text
K = 4 first, then K = 8
temperature = 0.7
max_new_tokens = 768
```

Stages:

1. Build diagnostic batch, no training.
2. Train A0 VA-OPD reproduction.
3. Train A2 chunk gap gated OPD.
4. Train A3 verifier-gated chunk gap OPD.

Primary success criterion:

```text
A2/A3 improves visual reasoning benchmarks over A0, not merely over broken baselines.
```

Secondary success criterion:

```text
High-weight tokens/chunks are human-interpretable visual reasoning locations.
```

## 12. Stop Conditions

Stop and revise if:

- high weights concentrate on XML tags, punctuation, option letters, or generic phrases;
- teacher/student visual focus gap is uncorrelated with human visual chunks;
- degraded images destroy global structure instead of fine detail;
- training improves formatting but hurts answer accuracy;
- wrong-but-format-valid rollouts dominate with nonsense reasoning.

## 13. Short Name

Working name:

```text
Visual Grounding Gap OPD (VGG-OPD)
```

or:

```text
Student-Grounded VA-OPD
```

The second name is safer for a paper because it clearly communicates the delta from VA-OPD.

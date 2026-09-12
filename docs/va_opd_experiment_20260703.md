# VA-OPD Pure Distillation Experiment — 2026-07-03

> Historical note: this run predates the 2026-07-05 VA-OPD correctness fixes
> for teacher batching, degraded-image protocol, prompt equivalence, and the
> actor-side `va_opd` patch. Treat the results below as a pre-fix postmortem,
> not as evidence about a faithful VA-OPD reproduction.

## Setup

- **Algorithm**: VA-OPD (arXiv 2605.21924 §3.2-3.3) — pure reverse KL distillation, no GRPO
- **Student**: Qwen2.5-VL-4B-Instruct
- **Teacher**: Qwen2.5-VL-32B-Instruct
- **Dataset**: Geometry3K (multiple-choice visual geometry)
- **Conditions**: FULL (raw image) + DEGRADED (10% bilinear downsample, nearest upsample)
- **GPU**: 4× H200, 200 steps
- **Loss**: `L = Σ_k w^(k) · L_group^(k)` (formula 7), no external `loss_coef`

## Pipeline

```
Student rollout → Teacher forced-forward (full + degraded, exact log P_T)
                → VA = log P_T(full) - log P_T(degraded), rectified ≥ 0
                → HighVA/LowVA split (top 20%)
                → L_group = 0.5·mean(KL_rev, HighVA) + 0.5·mean(KL_rev, LowVA)
                → w^(k) = softmax(z_score(ā^(k)) / τ)
                → Loss = Σ w^(k) · L_group^(k)
                → backward → optimizer.step()
```

## Key Implementation Details

| Layer | Detail |
|-------|--------|
| VA compute | Full-vocab `log_softmax` gather at sampled token position — **no top-32 tail approximation** |
| KL | Reverse KL: `KL(P_S \|\| P_T)` (mode-seeking, standard for distillation) |
| Student prompt | `<image>\n{canonical_question}` — choices allowed, no XML |
| Teacher prompt | same canonical question + image — no format bias |
| Rollout weights | z-score normalized softmax with τ=1.0, sums to 1 per prompt sibling group |
| Grouped KL | λ=0.5, HighVA=top 20%, LowVA=bottom 80% |
| Grad check | All `requires_grad=True` verified: `per_token_kl.grad_fn=<MulBackward0>` → `L_group.grad_fn=<AddBackward0>` → `numerator.grad_fn=<AddBackward0>` |

## Results Summary

```text
Steps: 200
Entropy: min=0.05, max=7.38, mean=2.11
fc_opd_loss: ~0.20 (stable)
Score: 0.0 (no reward model in pure distillation)
Grad norm: 12–34 (healthy, no vanishing/exploding)
Response length: oscillating 1–768 tokens
```

### Collapse Pattern (entropy sampled every 20 steps)

```
step   1: ent=0.22  len=372   ← initial low entropy
step  21: ent=5.07  len=555   ← exploring
step  41: ent=5.51  len=428   ← exploring
step  61: ent=1.30  len=89    ← converging
step  81: ent=1.69  len=106   ← converging
step 101: ent=0.10  len=768   ← COLLAPSED (all max-length generic text)
step 121: ent=2.15  len=693   ← bouncing back
step 141: ent=6.15  len=54    ← re-exploring
step 161: ent=1.26  len=129   ← re-converging
step 181: ent=0.20  len=487   ← re-collapsing
```

## Diagnosis

**Pure reverse KL distillation oscillates — entropy cycles between collapse and re-exploration.**

Root cause: `KL(P_S || P_T)` is **mode-seeking** without an anchor. The student finds *any* low-KL mode of the teacher distribution (generic geometric writing style, filler tokens) rather than the *correct reasoning* mode. The loss decreases but answer correctness doesn't improve. When the model collapses to a degenerate mode, gradient diversity eventually kicks it out, but it just finds another degenerate mode.

### What's Missing (standard in MiniLLM / RLHF / Distillation literature)

```
L = KL(P_S || P_T) / len          ← we have this
  + β · KL(P_S || P_init)          ← anchor to initial model — NOT implemented
```

`KL(P_S || P_init)` prevents the student from drifting too far from its initialization, which would otherwise produce reasonable (if imperfect) answers. This is a **necessity** for pure distillation, not an optional trick.

## Logs

Full training log: `artifacts/fc_opd/nohup_vgpo_20260703_080310.log`
Checkpoint: `checkpoints/verl_vgpo/va_opd_20260703_080310/global_step_200/`

## Code Changes

All VA-OPD implementation files changed from `8f79753`:
- `va_opd_loss.py` — VA-OPD loss core (VA compute, rollout weights, grouped KL)
- `teacher_transformers.py` — exact full-vocab log-prob gather
- `teacher_protocol.py` — `sampled_token_log_probs` field
- `signal_decomposer.py` — `TeacherTopK.sampled_log_probs` field
- `teacher_client.py` — HTTP → TeacherTopK conversion
- `teacher_prompts.py` — cleaned (no XML; same canonical question as student)
- `verl_dataset.py` — clean student prompts
- `verl_integration.py` — `teacher_sampled_log_probs` tensor
- `online_batch.py` — `skip_routing` fast path for VA-OPD
- `verl_post_rollout_hook.py` — VA-OPD detection, pipeline verification
- `dp_actor.py` — `va_opd` loss mode, pure distillation path
- `run_verl_vgpo.sh` — launch script (loss_mode=va_opd)

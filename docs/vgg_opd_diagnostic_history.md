# VGG-OPD / VA-OPD / VGPO: Diagnostic History of Attempts

## Motivation

We want to improve visual grounding in VLM on-policy distillation. The core question:

> Can we make the student rely more on visual evidence by identifying _where_ the teacher uses vision and _where_ the student lacks it?

Three progressively refined approaches were attempted. This document records what worked, what failed, and why — serving as a diagnostic contribution for future work.

---

## Attempt 1: VGG-OPD (Visual Grounding Gap OPD)

**Branch**: `codex/vgg-opd-frozen`  
**Based on**: Design doc `plan/fc_opd_visual_grounding_gap_plan.md`

### Design

Multi-signal distillation weighting:

1. **Teacher Visual Advantage (VA)**: `VA = log P_T(full) - log P_T(degraded)` — counterfactual visual reliance
2. **Student Visual Grounding Gap**: `gap = vfs_rank_T - vfs_rank_S` — student-specific visual deficit
3. **Chunk-level gating**: XML-parsed chunks (visible_evidence, diagram_inference, reasoning, answer)
4. **Verifier learning-value gate**: failure-focused OPD weighting
5. **Grouped KL**: high-VA and low-VA tokens averaged separately (anti-dilution)

### Training Mode

- **Pure distillation**: No GRPO. `policy_loss = L_VGG-OPD` directly.
- vLLM rollout → teacher forced-score (FULL + DEGRADED with hidden states) → student forced-score → VGG-OPD grouped KL → backward

### Result

**Failed — entropy collapse**. Within 50-70 steps, responses collapsed to 3-25 tokens. All answers wrong (`score=0.0`). Root cause: pure distillation without task reward signal allows the model to minimize KL by outputting trivially short answers.

### Key Finding

Pure distillation (no RL reward) is insufficient to maintain response quality. The student discovers that "A" or "B" minimizes KL divergence with the teacher.

---

## Attempt 2: VA-OPD Pure Distillation

**Branch**: `codex/va-opd`  
**Based on**: VA-OPD paper (arXiv 2605.21924), strict formula alignment

### Design

Simplified from VGG-OPD — stripped to paper's exact formulas:

1. `VA_t = max(log P_T(full) - log P_T(degraded), 0)` — rectified visual advantage
2. `ā^(k) = (1/T) Σ_t VA_t` — trajectory mean over ALL tokens
3. `ẑ^(k) = (ā^(k) - μ) / σ` — z-score within sibling group
4. `w^(k) = softmax(ẑ^(k) / 1.0)` — rollout weight (sums to 1)
5. HighVA = top 20% by VA; LowVA = rest
6. `L_group = 0.5·mean(KL_rev_High) + 0.5·mean(KL_rev_Low)` — reverse KL, λ=0.5
7. `L_VA-OPD = Σ_k w^(k)·L_group^(k)` — pure distillation

### Training Mode

Same as VGG-OPD: pure distillation, no GRPO. `policy_loss = L_VA-OPD`.

### Result

**Failed — same entropy collapse**. Despite reverse KL (mode-seeking) and correct formulas, response collapsed within 30 steps even faster than VGG-OPD. Reverse KL actually accelerates collapse because mode-seeking pushes the student to cover fewer modes.

### Key Finding

Whether forward KL or reverse KL, pure distillation without task reward inevitably collapses. The model finds the optimal strategy: output minimal tokens → minimize KL. This is not a bug in VA-OPD or VGG-OPD — it's fundamental to pure distillation on small models in low-data regimes.

---

## Attempt 3: VA-OPD as GRPO Auxiliary Loss

**Branch**: `codex/va-opd` (modified in-place)  
**Based on**: VA-OPD paper's claim of "plug-and-play with any RLVR method"

### Design

Same VA-OPD formulas as Attempt 2, but training mode changed:

```
L_total = L_GRPO + 0.01 × L_VA-OPD
```

- GRPO provides task reward signal (prevents collapse)
- VA-OPD provides visual-grounded distillation
- `fc_opd_coef=0.01` — GRPO dominates early, OPD guides visual reliance

### Result

**Partially successful — delayed but eventual collapse**.

| Epoch | fc_opd_loss | entropy | response_len | score |
|-------|------------|---------|-------------|-------|
| Step 1 | 11.7 | 0.17 | 363 | 1.00 |
| Step 20 | 7.5 | 0.57 | 538 | 0.75 |
| Step 50 | 2.6 | 5.44 | 438 | 0.25 |
| Step 80 | 1.2 | 6.14 | 658 | 0.38 |
| Step 198 | 2.3 | 4.15 | 14 | 0.00 |

- Entropy rose to 6.14 (good exploration), response lengths stayed healthy through step 80
- But by step 198: `response_length=14`, `score=0.0` — collapse returned
- GRPO advantage dropped to zero as all answers became wrong → OPD dominated again

### Key Finding

`0.01` coefficient delayed collapse from ~30 steps to ~80 steps, proving GRPO's task signal is essential. But once GRPO advantage vanishes (all answers wrong), the auxiliary OPD loss dominates and collapse recurs. Two loss terms competing is inherently unstable.

---

## Attempt 4: VGPO — Advantage Modulation (Current)

**Branch**: `codex/va-opd` (continuing)  
**Based on**: VGPO paper (ACL 2026, arXiv 2604.09349)

### Design

**Instead of adding a separate distillation loss**, directly modulate the GRPO advantage with visual signals:

```
Â_{i,t}^V = Â_i · (1 + ψ_{i,t}) · (1 + φ_i)
```

Where:
- `Â_i` = original GRPO advantage (task-reward-driven)
- `ψ_{i,t}` = intra-trajectory visual focus (token-level, zero-centered)
- `φ_i` = inter-trajectory visual grounding (rollout-level, zero-centered)

Then feed `Â^V` into the standard PPO clipped loss. **No extra loss term, no coefficient tuning.** Visual signal tells the policy **which tokens to reinforce more**, not which distribution to mimic.

### Why This Should Work

1. **Single objective**: Only one loss (PPO), no competing terms
2. **Task signal preserved**: GRPO advantage `Â_i` still drives toward correct answers
3. **Visual modulation**: `(1+ψ)(1+φ)` amplifies advantage on visually-grounded tokens and rollouts
4. **Zero-centered**: ψ and φ are zero-mean, so average advantage is unchanged — only _relative_ weighting changes
5. **No collapse pathway**: Task reward still required for positive advantage → model cannot cheat with short answers

### Differences from Attempts 1-3

| | VGG-OPD / VA-OPD | VGPO |
|---|---|---|
| Visual signal | Separate KL loss | Advantage modulation |
| Loss terms | 2 (GRPO + OPD) | 1 (PPO with Â^V) |
| Coefficient tuning | Required (α) | None |
| Collapse risk | High (competing losses) | Low (same loss) |
| Teacher needed | For KL computation | Only for reward (standard GRPO) |

---

## Summary

| Attempt | Approach | Collapse? | Key Insight |
|---------|----------|-----------|-------------|
| VGG-OPD pure | Multi-signal distillation, no GRPO | Yes, ~50 steps | Pure distillation collapses |
| VA-OPD pure | Paper-exact formulas, reverse KL | Yes, ~30 steps | Reverse KL collapses faster |
| VA-OPD aux | GRPO + 0.01×VA-OPD | Delayed, ~80 steps | Competing losses unstable |
| **VGPO** | Advantage modulation | **TBD** | Single objective, no coefficient |

### Diagnostic Contribution

The progression from VGG-OPD → VA-OPD → VGPO demonstrates a clear pattern:

1. **Pure distillation is insufficient** for maintaining generation quality in small models
2. **RL reward signal is necessary** to anchor the policy toward correct outputs
3. **Auxiliary loss coefficients are fragile** — the balance between RL and distillation shifts as training progresses
4. **Advantage modulation is theoretically cleaner** — it guides _where_ to learn within the existing RL framework rather than adding a competing objective

These findings are relevant beyond visual grounding — they apply to any attempt to inject inductive bias into RL-trained language models.

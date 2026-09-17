# NLL-TailOPD v1 — Implementation & Validation Plan

**Status:** implementation-ready  
**Primary goal:** test whether student on-policy rollouts that are relatively low-likelihood under the student itself should receive more OPD supervision.  
**Design principle:** first change only **rollout-level weighting**. Do not add correctness routing, teacher–student compatibility, or token-level tail weighting in v1.

---

## 0. Fixed technical stack

Active environment:

`/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/va-opd-qwen35-v090-cu132-r595-v1`

Recorded stack:

- Python 3.12
- torch 2.13.0+cu132
- vLLM 0.27.1
- transformers 5.12.0
- flashinfer 0.6.16.post3
- flash-attn 2.8.3
- Ray 2.55.1
- verl 0.9.0, backend baseline commit `483b8a00`
- NCCL 2.29.7
- target GPU: H200 / SM90
- GPU smoke / NCCL / 3-step training already passed

Training entry chain:

1. Repo wrapper  
   `scripts/run_qwen35_cu132_formal.sh`

2. External verl backend launcher  
   `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/backends/verl-qwen35-v090-cu132/examples/on_policy_distillation_trainer/run_qwen3_5_4b_fsdp.sh`

3. Python entry  
   `python3 -m verl.trainer.main_ppo`

**Important:** modify the external backend above, not `third_party/verl`.

Backend root:

`/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/backends/verl-qwen35-v090-cu132`

---

## 1. Research hypothesis

For prompt \(x_b\), let the student produce \(K\) on-policy rollouts:

\[
y_{b1},\ldots,y_{bK}\sim\pi_{\mathrm{old}}(\cdot|x_b).
\]

For rollout \(k\), define sequence-normalized sampled NLL:

\[
s_{bk}
=
-\frac{1}{T_{bk}}
\sum_{t=1}^{T_{bk}}
\log \pi_{\mathrm{old}}
(y_{bkt}\mid h_{bkt}).
\]

Interpretation:

- larger \(s_{bk}\): the sampled trajectory is relatively low-likelihood / surprising under the student;
- smaller \(s_{bk}\): the trajectory lies in a more strongly modeled region.

v1 hypothesis:

> **Within sibling on-policy rollouts of the same prompt, higher-NLL trajectories should receive more teacher supervision.**

No correctness/reward term is used in v1.

---

## 2. TailOPD v1 objective

### 2.1 Rollout score

\[
s_{bk}
=
-\frac{
\sum_t m_{bkt}\log\pi_{\mathrm{old}}(y_{bkt}|h_{bkt})
}{
\sum_t m_{bkt}
}.
\]

Requirements:

- mask padding/non-response tokens;
- normalize by response length;
- `detach()` / no gradient through the score;
- use rollout-policy / old-policy log probabilities.

### 2.2 Within-prompt normalization

\[
\mu_b=\frac1K\sum_k s_{bk},
\qquad
\sigma_b=
\sqrt{\frac1K\sum_k(s_{bk}-\mu_b)^2}.
\]

\[
z_{bk}
=
\frac{s_{bk}-\mu_b}{\sigma_b+\epsilon}.
\]

Default: `eps = 1e-6`.

If group std is numerically zero, use uniform weights.

### 2.3 Convert relative NLL to rollout weights

\[
w_{bk}
=
\frac{\exp(z_{bk}/\tau)}
{\sum_{j=1}^{K}\exp(z_{bj}/\tau)}.
\]

Default:

\[
\tau=1.0.
\]

Properties:

\[
w_{bk}>0,\qquad \sum_k w_{bk}=1.
\]

This mirrors VA-OPD's rollout pipeline:

`token score -> rollout mean -> within-group z-score -> softmax -> rollout weight`.

### 2.4 OPD signal

Do **not** change the existing OPD estimator in v1.

Let the current backend already produce the per-token OPD advantage \(d_{bkt}\) (or an equivalent distillation loss converted to advantage).

TailOPD modifies only rollout scale:

\[
A^{\mathrm{TailOPD}}_{bkt}
=
\alpha_{bk}d_{bkt},
\qquad
\alpha_{bk}=K w_{bk}.
\]

Why \(K w_{bk}\)?

Under uniform weighting:

\[
w_{bk}=1/K
\Rightarrow
\alpha_{bk}=1,
\]

so the existing OPD gradient scale is recovered exactly.

Also:

\[
\frac1K\sum_k\alpha_{bk}=1.
\]

Thus v1 redistributes distillation budget rather than globally increasing it.

---

## 3. Loss reduction must be explicit

Intended objective:

\[
L_b
=
\sum_{k=1}^{K}
w_{bk}
\left[
\frac1{T_{bk}}
\sum_t L^{OPD}_{bkt}
\right],
\]

\[
L
=
\frac1B\sum_bL_b.
\]

For clean interpretation, TailOPD and Vanilla OPD must use the same reduction.

Preferred experiment setting:

`loss_agg_mode = seq-mean-token-mean`

or the exact equivalent in this patched backend.

Do not silently rely on global `token-mean`, because longer rollouts then receive more gradient mass.

Implementation rule:

1. inspect the active backend;
2. identify the active OPD aggregation;
3. create an explicit experiment config where **both Vanilla OPD and TailOPD use the same sequence-normalized aggregation**;
4. do not change the global default for unrelated experiments.

---

## 4. Code archaeology before editing

```bash
BACKEND=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/backends/verl-qwen35-v090-cu132

cd "$BACKEND"

git rev-parse HEAD
git status
git log --oneline -20
git diff 483b8a00 -- \
  verl/trainer/distillation \
  verl/trainer/ppo \
  examples/on_policy_distillation_trainer
```

Locate relevant functions:

```bash
grep -Rni "compute_distillation_loss_reverse_kl_estimator" "$BACKEND/verl"
grep -Rni "advantages.*distillation" "$BACKEND/verl"
grep -Rni "loss_agg_mode" "$BACKEND/verl/trainer"
grep -Rni "old_log_prob" "$BACKEND/verl/trainer"
grep -Rni "rollout.n" "$BACKEND/examples/on_policy_distillation_trainer"
```

Do not assume upstream-main line numbers; this backend is a patched v0.9-era branch.

---

## 5. Recommended implementation architecture

### 5.1 Keep rollout generation untouched

Do not modify:

- vLLM sampling;
- prompt repeat logic;
- teacher forward;
- reward model/verifier;
- PPO importance ratio;
- clipping.

TailOPD v1 operates after rollout logprobs are available and before / while the OPD advantage is passed to policy loss.

### 5.2 Add an isolated weighting utility

Recommended conceptual API:

```python
def compute_nll_tail_rollout_weights(
    old_log_probs,
    response_mask,
    group_ids,
    temperature=1.0,
    eps=1e-6,
):
    # returns weights [BK], scales [BK], scores [BK], zscores [BK]
    ...
```

Use the prompt `uid` / group identifier already carried by verl.

**Do not assume sibling rollouts are contiguous.**

### 5.3 Apply scale to existing OPD advantage

```python
tail_scale = K * rollout_weight
tail_scale = tail_scale.detach()

opd_advantage = existing_opd_advantage * tail_scale[:, None]
```

Everything after that stays in the existing OPD/PPO pipeline.

---

## 6. Config flags

Suggested backward-compatible block:

```yaml
algorithm:
  tail_opd:
    enabled: false
    score_type: nll
    group_normalization: zscore_softmax
    temperature: 1.0
    eps: 1.0e-6
    score_source: old_log_probs
    detach_score: true
    zero_std_behavior: uniform
    rollout_scale_mode: mean_one
```

Requirements:

- `enabled: false` exactly reproduces current OPD;
- no environment changes;
- no dependency changes;
- no model-architecture changes.

---

## 7. Required diagnostics

Log:

- `tail_opd/nll_mean`
- `tail_opd/nll_std`
- `tail_opd/nll_p10`
- `tail_opd/nll_p50`
- `tail_opd/nll_p90`
- `tail_opd/weight_min`
- `tail_opd/weight_max`
- `tail_opd/weight_entropy`
- `tail_opd/effective_rollouts`
- fraction of groups with near-zero NLL std
- correlation between normalized NLL and response length
- mean and max of \(K w\)
- OPD loss / KL
- response length

Effective rollout count:

\[
N_{\mathrm{eff}}
=
\frac{1}{\sum_k w_k^2}.
\]

Uniform weighting gives \(N_{\mathrm{eff}}=K\).

---

## 8. Unit / invariant tests

1. **Uniform recovery:** identical scores -> `w=1/K`, `K*w=1`, TailOPD equals Vanilla OPD.
2. **Ordering:** scores `[1,2,3,4]` -> strictly increasing weights.
3. **Normalization:** group weights sum to 1; group mean of `K*w` equals 1.
4. **Padding invariance:** padding does not alter NLL.
5. **Group isolation:** changing prompt B scores does not affect prompt A weights.
6. **No-gradient:** weights have `requires_grad=False`.

---

## 9. Training validation schedule

### 9.1 Smoke stage

- 3 optimizer steps;
- then 20–30 steps if stable.

Check:

- no NaN/Inf;
- group normalization correct;
- `mean(K*w) ~ 1`;
- loss scale comparable to vanilla;
- checkpoint save/load works.

### 9.2 Fast research validation: Geometry3K

Recommended first method-validation dataset: **Geometry3K**.

Why:

- small (~2.1K problems);
- direct VA-OPD precedent;
- VA-OPD trained 5 epochs with \(K=4\), batch size 16;
- useful for separating algorithm problems from pipeline problems.

Default first A/B:

- maximum OPD training: **5 epochs**
- rollout K: **4**
- `tau=1.0`
- evaluate/checkpoint every epoch, and optionally mid-epoch
- same optimizer/LR/teacher/student for Vanilla OPD and TailOPD

**Do not automatically use epoch 5; use the best validation checkpoint.**

A 1–2 epoch negative result is insufficient because VA-OPD itself reports that its gain appears after an early alignment phase.

### Required first comparison

1. Base student
2. Vanilla OPD
3. NLL-TailOPD
4. optional strong control: shuffled-NLL weighting

---

## 10. OPD stopping / switch criterion

Do not switch by a fixed epoch alone.

`Sequential Beats Joint` found that the **OPD validation score at the switching point** largely determines final post-RL performance. Their ablation used switch points 20 / 60 / 100; when OPD was still improving, later switching helped, while once validation saturated, later switching gave little extra benefit.

Practical v1 rule:

### Geometry3K
- hard cap: 5 epochs;
- evaluate at least once per epoch;
- choose the best validation checkpoint;
- allow early switch after clear plateau for 2 consecutive evaluations.

Configurable engineering heuristic:

- `patience = 2`
- optional `min_delta = 0.2 percentage points`

This heuristic is not a paper claim.

### Larger datasets such as MMFineReason-123K

Do **not** copy “5 epochs”.

Use:

- initial OPD budget: 100–200 optimizer steps;
- evaluate every 20–25 steps;
- extend only while validation improves.

---

## 11. Stage-2 RL choice

### v1 recommendation: GRPO first

Run pure GRPO from the selected OPD checkpoint.

Why:

1. `Sequential Beats Joint` directly validates OPD -> GRPO.
2. GRPO is the cleaner causal test of whether TailOPD supplies a better initialization.
3. DAPO simultaneously adds clip-higher, dynamic sampling, token-level PG aggregation, and overlong reward shaping.
4. Those extra mechanisms can obscure whether the gain comes from TailOPD.

First scientific comparison:

\[
\text{Vanilla OPD}\rightarrow\text{GRPO}
\]

vs.

\[
\text{NLL-TailOPD}\rightarrow\text{GRPO}.
\]

Keep identical:

- RL data;
- verifier/reward;
- rollout K;
- RL step budget;
- stage-2 optimizer settings.

No teacher/TailOPD term during RL.

### Later: DAPO robustness / final system

After TailOPD works under GRPO:

\[
\text{Vanilla OPD}\rightarrow\text{DAPO}
\]

vs.

\[
\text{NLL-TailOPD}\rightarrow\text{DAPO}.
\]

DAPO is more appropriate for the final strong project system, but not the cleanest first mechanism test.

---

## 12. Suggested experiment matrix

### Phase A — OPD-only mechanism test

| Run | Stage 1 | K | Max OPD | Purpose |
|---|---|---:|---:|---|
| A0 | Base | — | — | starting point |
| A1 | Vanilla OPD | 4 | 5 epochs | control |
| A2 | NLL-TailOPD, tau=1 | 4 | 5 epochs | main test |
| A3 | Shuffled-NLL OPD | 4 | 5 epochs | mechanism control |

If A2 does not beat A1/A3 consistently, diagnose before adding RL.

### Phase B — sensitivity after positive signal

Optional:

- `tau = 0.5`
- `tau = 2.0`
- K=8 rerun for more stable within-prompt statistics

### Phase C — sequential RL

Take the selected Vanilla and TailOPD checkpoints and continue both with identical GRPO.

Compare:

- post-OPD performance;
- post-GRPO performance;
- pass@1 / avg@K;
- pass@K where meaningful.

### Phase D — final stronger system

Only after v1 validation:

- larger/project-aligned multimodal data;
- final 8B student;
- stronger teacher;
- DAPO stage;
- full project benchmark suite.

---

## 13. Go / no-go criterion

Proceed from OPD-only to RL if:

1. TailOPD beats Vanilla OPD on the primary validation metric at its selected checkpoint;
2. shuffled-NLL does not reproduce the gain;
3. there is no obvious collapse in output length, KL, or weight concentration.

If the result is noisy:

1. inspect `N_eff`;
2. inspect NLL spread;
3. inspect NLL–length correlation;
4. then test K=8;
5. only then tune temperature.

Do not immediately add correctness routing, teacher compatibility, or token-level weighting.

---

## 14. Git discipline

Recommended branch:

`exp/nll-tailopd-v1`

Agent requirements:

1. inspect backend before editing;
2. do not modify environment packages;
3. do not modify `third_party/verl`;
4. TailOPD must be behind a default-off switch;
5. disabled mode must preserve Vanilla OPD;
6. add invariant tests;
7. run 3-step smoke;
8. run 20–30 step check;
9. only then launch formal run;
10. document changed files and mathematical mapping.

Before push, report:

- commit SHA;
- changed files;
- launch command;
- config diff;
- smoke logs;
- normalization-test results.

---

## 15. Explicit non-goals for v1

Do not implement yet:

- correctness-aware routing;
- success-only weighting;
- MaxRL-style \(1/q\) prompt weighting;
- teacher–student compatibility gating;
- token-level NLL weighting;
- entropy-based weighting;
- Visual Advantage;
- joint OPD + RL;
- PTD-PO privileged hints.

---

## 16. References

1. **Visual-Advantage On-Policy Distillation for Vision-Language Models**, arXiv:2605.21924.  
   Relevant: trajectory-mean score -> group z-score -> softmax; Geometry3K ~2.1K; 5 epochs; K=4; batch size 16.

2. **Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR**, arXiv:2609.04108.  
   Relevant: OPD -> GRPO; switch by validation quality; switch ablation at steps 20/60/100; step 60 used as default at end of fast OPD improvement.

---

## 17. One-sentence target

> Implement a default-off NLL-based rollout reweighting layer that computes length-normalized old-policy NLL for each student rollout, converts sibling scores into z-score + softmax weights, rescales the existing OPD advantage by \(K w_k\), and leaves the rest of the verl OPD/PPO pipeline unchanged.

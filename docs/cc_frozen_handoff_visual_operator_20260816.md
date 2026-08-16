# Frozen Research Direction: Visual Handoff + Operator Need (authority doc)

Date: 2026-08-16

This is the single research authority for the next implementation phase,
frozen after the user's ChatGPT review.  Previous STP/RL/M0-M4/h* work remains
in the repo as historical assets and baselines; it no longer decides the
route.

Branch `codex/va-opd`.  Repo baseline `e03480e` (Phase D gate re-run landed
in `758da50`).

## 0. Constraints (do not launch without a new decision)

- No STP training, no GRPO/RL, no full FKL/RKL training, no router.
- No further tuning of h* AUROC / M0-M4 proxy families.
- Only two diagnostics may run first; both must produce a report before any
  training method is synthesized.

## Question A - Visual Handoff Diagnostic

Core question: for a small VLM that can be rescued by a teacher prefix, does
the teacher mainly complete the visual-evidence -> explicit-representation
conversion, after which the student's own reasoning becomes executable?

This is a hypothesis, not a premise.  If the visual story fails, weaken/abandon
it.

A1. Data: rare-success prompts + rescue-positive prompts + matched
rescue-negative controls.  One to a few verified correct natural teacher
traces per prompt; no XML / handcrafted structure labels.

A2. Segmentation: split the teacher response into reasoning blocks
`B_1..B_m` (natural-language sentence, newline-separated derivation, complete
LaTeX/equation block, coherent therefore-step, final answer block).  Every
block must map to an exact token span.  Fixed token horizons (64/128/256/512)
remain only as legacy baseline.

A3. Multimodal counterfactual scoring: generate the teacher trajectory once;
forced-forward the *same* tokens under full image `I` and degraded image
`I~`:

```
V_t^T = log p_T(y_t^T | I, c_t) - log p_T(y_t^T | I~, c_t)
V_t^S = log p_S(y_t^T | I, c_t) - log p_S(y_t^T | I~, c_t)
DeltaV_t = V_t^T - V_t^S
```

Block aggregate `DeltaV_i = mean_{t in B_i} DeltaV_t`.

A4. Candidate handoff: find the change point from a high-visual-gap regime to
a sustained low-visual-gap regime on the `DeltaV_i` block sequence.  This
`h_V` is a cheap localization candidate only, not a claim `h_V == h*`.

A5. Causal validation: student continuation only at `h_V - 1`, `h_V`,
`h_V + 1` blocks, `q(h) = P_S(R=1 | I, q, y<=h^T)`, K=4 (K=8 if ambiguous).
No full-trajectory brute force.

Question-A success criterion: is the visual-gap change point significantly
enriched near the actual rescue transition?  NO -> visual attribution is
mechanism analysis only; YES -> visual-to-symbolic handoff is a method
candidate.

## Question B - FKL vs RKL Operator Need

Scope: only inside teacher prefixes already confirmed usable by the student.

B1. Local support quantities on teacher-prefix context `c_t`:

```
S_t^S(K) = TopK(p_S(.))
S_t^T(K) = TopK(p_T(.))
U_t = S_t^S(K) union S_t^T(K)
D_t = KL( pbar_T^U || pbar_S^U )
C_t = sum_{v in S_t^S(K)} p_T(v)
```

Use exact cross-gather (present in the symmetric Top-100 cache); never fall
back to the intersection-only lower bound.  K=16 primary; derive K=8/32/64
from the cache on CPU.

B2. RKL-need (TA-OPD): `R_t = D_t^L = Dtilde_t * Ctilde_t`.

B3. Incompatible disagreement: `D_t^I = Dtilde_t * (1 - Ctilde_t)`.  Low C
alone does not imply FKL.

B4. Teacher off-support clarity:

```
A_t^off = S_t^T(K) \ S_t^S(K)
Q_t = 1 - H(p_T^off) / log|A_t^off|
```

High = clear missing teacher target; low = diffuse off-support disagreement.

B5. FKL-need hypothesis (diagnostic only, not a training mask):
`F_t = D_t^I * Q_t`.

B6. Aggregate to blocks: `R_i`, `F_i`, `V_i`; do not multiply the three axes
prematurely.

B7. Micro-operator causal experiment (the key new experiment): from the same
frozen checkpoint `theta_0`, for a few candidate blocks (high-F, high-R,
low-F/low-R controls), run a very short Top-100+tail FKL update (1-4 steps)
or the symmetric RKL update on that block only:

```
theta_i^FKL = theta_0 - eta * grad L_FKL(B_i)
q_i^FKL = P_theta_i^FKL(R=1 | I, q)   # fresh native rollout, no prefix
G_i^FKL = q_i^FKL - q_0
G_i^RKL = q_i^RKL - q_0
```

B8. Hypotheses to verify:

```
F_i up  => G_i^FKL - G_i^RKL up
R_i up  => G_i^RKL - G_i^FKL up
```

If either relation fails, stop that operator-routing story.

B9. Multimodal analysis: `V_i` is an explanation axis only.  If high-F +
high-V blocks show the highest micro-FKL gain, that supports the VLM-specific
claim (missing frontier concentrated in visual-supported but student-
off-support visual-to-symbolic transitions); otherwise drop visual-specific
routing and keep the generic support-acquisition story.

## Outputs (only two allowed)

1. Visual Handoff Report: does the multimodal visual-gap regime change predict
   where a rare-success student becomes able to take over?
2. Operator-Need Report: does support-aligned disagreement predict RKL
   utility, and does off-support + teacher clarity predict FKL utility?

Until both reports exist: no full FKL training, no full RKL training, no RL,
no STP four-arm, no router.

## One-line research logic

A. Where does the multimodal bottleneck end?  B. What is missing before that
point (reorder vs missing mode)?  C. Which operator actually fixes each type
(verified by micro-intervention, not by KL form).

## Implementation status (2026-08-16, branch `codex/va-opd`)

Code landed (no training, no RL):

- `src/dual_track_opd/support_aware/reasoning_blocks.py` - deterministic
  reasoning-block segmenter with exact token spans (A2).
- `src/dual_track_opd/support_aware/operator_need.py` - CPU-only B-side
  derivation of C_t / D_t / Q_t / R_t / F_t from the symmetric Top-100 cache
  (exact cross-gather), block aggregation, Operator-Need Report.
- `src/dual_track_opd/support_aware/visual_handoff.py` - GPU A-side: full /
  degraded counterfactual scoring, block DeltaV curve, BIC change point,
  continuation validation at h_V +/- 1 block; `merge` subcommand produces the
  Visual Handoff Report.
- `scripts/hpc/launch_visual_handoff.sh` - one-shot sharding across
  STUDENT:TEACHER GPU pairs.
- Unit tests in `tests/support_aware/test_reasoning_blocks.py` and
  `tests/support_aware/test_operator_need.py` (all support_aware tests pass).

First B-side report (data, not in git):
`support_aware_opd/operator_need_20260816/` - 88 prompts / 38 rescue-positive /
12,979 blocks.  Key descriptive finding: C_i median ~0.99998 on the teacher
trajectory (teacher's next token almost always inside the student's Top-16),
i.e. local compatibility is high; the rare-success barrier is path-level
reachability, not per-token incompatibility.  The B7 micro-operator causal
experiment is the next step and depends on both reports.

Reproduce B-side:

```bash
OUT=$DTOPD_OUTPUT_ROOT/support_aware_opd
$DTOPD_PYTHON -m dual_track_opd.support_aware.operator_need \
  --token-dirs "$OUT/reachability_proxy_20260814" "$OUT/reachability_proxy_20260815" "$OUT/reachability_proxy_20260816" \
  --rescue-dirs "$OUT/prefix_intervention_20260806_merged" "$OUT/prefix_intervention_20260815_merged" "$OUT/prefix_intervention_20260815b_merged" "$OUT/prefix_intervention_20260816_merged" \
  --proposal-dirs "$OUT/proposal_feasibility_20260805_merged" "$OUT/proposal_feasibility_20260815_merged" "$OUT/proposal_feasibility_20260816_merged" \
  --output-dir "$OUT/operator_need_20260816"
```

Run A-side on a GPU instance (8 GPUs):

```bash
bash scripts/hpc/launch_visual_handoff.sh 0:1,2:3,4:5,6:7
```

with `VISUAL_HANDOFF_PREFIX=visual_handoff_20260816` (default) and
`VISUAL_HANDOFF_MAX_PROMPTS=2` for a smoke check first.

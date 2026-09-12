# Micro-Operator Experiment - v1 report (low dose, null)

Date: 2026-08-17.  Branch `codex/va-opd`.

Question-B causal experiment (frozen plan B7/B8), first full run.

## 1. Run facts

- 30 candidate blocks = 10 rescue-positive prompts x (high-F / high-R /
  control); every block receives **both** a micro-FKL and a micro-RKL update
  (same-block paired difference).
- Dose: 2 optimizer steps, lr 2e-5, block-only; native rollout K=4, no prefix.
- Student Qwen3-VL-4B frozen checkpoint; teacher 32B scores the block.
- Outputs: `support_aware_opd/micro_operator_20260817/` (not in git).

## 2. Results

| metric | value |
|---|---:|
| candidate blocks | 30 |
| results (FKL+RKL) | 60 |
| baseline q0 | 0.00 for all 60 |
| rows with G_i > 0 | 4 (3 x +0.25, 1 x +0.75) |
| mean G_FKL | 0.017 |
| mean G_RKL | 0.033 |
| mean G_FKL by role | FKL-role 0.00, RKL-role 0.025, control 0.025 |
| corr(F_i, G_FKL - G_RKL) | 0.011 |
| corr(R_i, G_RKL - G_FKL) | 0.059 |

Non-zero rows:

| prompt | block | role | F | R | op | G |
|---|---|---|---|---|---|---:|
| geo3k:1208 | 20 | RKL | 0.04 | 0.02 | FKL | +0.25 |
| geo3k:1208 | 19 | control | 0.00 | 0.00 | FKL | +0.25 |
| geo3k:1208 | 19 | control | 0.00 | 0.00 | RKL | +0.25 |
| geo3k:1199 | 1 | RKL | 0.96 | 0.04 | RKL | +0.75 |

## 3. Interpretation

**B8 hypotheses are not supported at this dose**, but the measurement is
low-power and cannot yet distinguish "hypothesis false" from "dose too
small":

1. q0 = 0 everywhere (rescue-positive prompts are hard unaided), so G is
   one-sided and capped at 0.25 granularity with K=4.
2. 2 steps x lr 2e-5 on a single ~20-40 token block is a tiny perturbation;
   most updates never change native rollout behavior.
3. The single large effect (geo3k:1199 block 1, F=0.96) is an RKL gain of
   +0.75 - the direction opposite to the FKL hypothesis - so even the
   positive evidence does not support F-routing.

## 4. Next step: dose-response before any stop decision

Per the frozen plan ("if either relation fails, stop the operator-routing
story"), a fair stop decision requires a dose check.  v1.1 adds a **dose
sanity metric** (`block_logp_delta`: mean log P(block tokens) before vs after
the update) so we can confirm the update actually moved the model.  Rerun
with: 8 steps, lr 1e-4, rollout K=8, 12 candidates (~1.5-2h).  If
`block_logp_delta` is non-trivial (> ~0.1) and the F/R-conditional gains are
still null, then stop the operator-routing story.

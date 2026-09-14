# Region/Path-level Acquisition Diagnostic - v1 report (heavy dose, all-negative)

Date: 2026-08-17.  Branch `codex/va-opd`.

First run of the corrected region-level diagnostic (supersedes the
single-block micro-operator experiment as the Question-B mechanism test).

## 1. Run facts

- 8 rescue-positive prompts x 4 arms = 32 results; full-prefix regions
  `[0, h*)` (58-509 supervised tokens).
- Arms: soft Top-100+tail FKL, full-vocab RKL, hard-token CE, matched control
  (post-`h*` same-length span, soft FKL).
- Update: 8 steps, lr 1e-4, grad clip 1.0, weight-decay 0, fp32 loss math.
- Primary outcome: teacher-path suffix `log P(suffix | prefix)` before/after.
- Outputs: `support_aware_opd/region_acquisition_20260817/` (incremental
  append + resume, 32/32 rows saved).

## 2. Results

| arm | mean Delta path-logp | native G |
|---|---:|---:|
| FKL | -1.52 | -0.03 |
| RKL | -5.45 | -0.03 |
| HARD_CE | -2.58 | -0.03 |
| control | -0.77 | 0.00 |

Every arm decreases teacher-path suffix log-likelihood.  RKL is consistently
the most destructive (per-region -3.4 to -7.1).  Control is near-zero on two
regions (+0.04/+0.09) but -0.8 to -1.9 on six, so even the post-`h*` control
degrades the path at this dose.  Native q stays at the 0 floor (one control
row reached 0.25).

## 3. Mechanism confound

Grad norms per arm (all clipped to 1.0 every step):

| arm | median | max |
|---|---:|---:|
| FKL | 44 | 76 |
| HARD_CE | 77 | 692 |
| RKL | 282 | 1048 |
| control | 46 | 968 |

Every step is a fixed-norm jump in the normalized-gradient direction; the
updates are clip-dominated, not clean small steps.  Because even the control
arm degrades path support, the all-negative result is dominated by update
mechanics at this dose and **cannot be read as an operator-level null**.

## 4. Conclusion and next step

Heavy clip-dominated region supervision does not acquire teacher-path support
under any operator.  Before any operator conclusion, rerun with a gentle dose:
lr 1e-5, 16 steps, clip 10.0 (new `--clip-max-norm` knob), same 8 regions.
Decision rule: if gentle FKL/RKL then show Delta path-logp >= 0 with
FKL/CE > control, operator-specific acquisition is supported; if still <= 0
for all arms, region-local acquisition does not transfer and the route moves
to full-prefix STP-style training (the original Dual-Track design).

# Region/Path-level Acquisition Diagnostic - v2 final report (gentle dose, no operator signal)

Date: 2026-08-17.  Branch `codex/va-opd`.

Gentle-dose rerun of the region-level acquisition diagnostic (v1 was
clip-dominated and uninterpretable).  Same 8 rescue-positive regions, 4 arms.

## 1. Run facts

- Update: 16 steps, lr 1e-5, grad clip 10 (v1 was lr 1e-4, clip 1.0),
  weight-decay 0, fp32 loss math.
- 8 regions x 4 arms = 32 rows, all saved incrementally.

## 2. Results

| arm | mean Delta path-logp | positive regions | native G |
|---|---:|---:|---:|
| control (post-h*, soft FKL) | +0.085 | 7/8 | 0.00 |
| FKL (pre-h*, soft Top-K+tail) | +0.014 | 4/8 | -0.03 |
| RKL (pre-h*, full-vocab) | -0.001 | 2/8 | -0.03 |
| HARD_CE (pre-h*, SFT) | -0.070 | 1/8 | -0.03 |

Per-region deltas are all small (+-0.2 nats max); native rollout conversion
is absent everywhere (q stays at the 0 floor, one control row reaches 0.25).

## 3. Interpretation

- The control arm (same-length span after `h*`, where the student already
  succeeds) improves teacher-path suffix log-likelihood *more* than any
  pre-`h*` learning arm.  FKL is mixed (4/8 positive), CE is consistently
  negative (7/8), RKL is flat-to-negative.
- No operator beats the matched control; no operator produces native
  conversion.  Region-local supervision on the hard pre-`h*` prefix does not
  acquire the missing teacher path.
- Grad norms were still above clip 10 (medians 21-35, max 214), so the update
  is not perfectly unclipped, but the 10x smaller lr makes the deltas ~10x
  smaller than v1 and the qualitative ordering is stable.

## 4. Final verdict (preregistered decision rule)

**Operator-specific acquisition is NOT supported.**  Per the rule agreed in
`cc_region_acquisition_report_v1_20260817.md`, this closes the region-local
operator question:

- stop single-block AND region-local FKL/RKL routing as a mechanism to move
  teacher-path support into the student;
- the mechanism evidence (C_i ~ 0.99998, no local-transfer under any
  operator) supports the path-reachability view: the student's barrier is
  entering the correct state sequence, not local token probabilities;
- the remaining route is the original Dual-Track design: full-prefix
  FKL/CE + suffix OPD/RL supervision on the usable prefix, with these four
  diagnostics (visual v1-v3, operator v1-v2, region v1-v2) archived as
  published negative evidence.

Optional rigor: a no-clip round (`--clip-max-norm 1e9`) would remove the last
mechanism confound, but the 10x-smaller deltas and stable arm ordering make a
different qualitative outcome unlikely.

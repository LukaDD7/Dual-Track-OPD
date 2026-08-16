# CC Handback: Phase C Sensitivity Re-Check (K, W, endpoint alignment)

Date: 2026-08-16

Execution contract: `docs/cc_handoff_viability_proxy_adjustment_20260815.md`
§3.2/§3.3 and the Phase C handback §5 item 2.  This is the registered
sensitivity pass on the existing v2 cache, CPU-only; it is not a new feature
search and it does not re-run the gate models.

## 1. Repository status

- Branch `codex/va-opd`.  The Phase A/B/C head (`6d4e454`, NO-GO handback) was
  pushed to origin.  This handback adds one local commit on top.
- Changed surface: `src/dual_track_opd/support_aware/reachability_proxy.py`
  (feature derivation only) and `tests/support_aware/test_reachability_proxy.py`
  (+2 regression tests).  `MODEL_SETS` (M0–M4) and the evaluator are untouched.
- STP canary / four-arm / SFT / FKL / RL: still paused.  No loss, no training,
  no checkpoint update, no Phase-7 launch.

## 2. What was added

`derive_salvaged_features` now derives, from the cached Top-100 arrays only
(no extra model forwards):

- `C_h^K` (teacher mass on student Top-K) and `O_h^K` (Top-K overlap) for
  `K in {4,64}`, K-sliced from the same cache.  K=16 keeps the cached endpoint
  fields so Phase C numbers are bit-identical; `C_h_16_derived` /
  `O_h_16_derived` are written as a QA cross-check.
- Endpoint-alignment sensitivity: the same C/O family read at position `h`
  (the first takeoff token) instead of position `h-1` (the last produced
  token), named `*_next`.  Out-of-range (h == trace length) is `None`.
- `delta_handoff_32` alongside the registered `delta_handoff_64`.

Tests: K-slicing matches hand computation and the cached K=16 fields; overlap
is `|TopK_S ∩ TopK_T| / K`; endpoint +1 variant and out-of-range handling;
`delta_handoff_32` hand computation.  23/23 tests pass.

## 3. Evidence (same cohort: 52 prompts / 194 rows / 27 positive prompts)

Single-feature AUROC (tie-correct, prompt-cluster bootstrap CI95):

| feature | AUROC (CI95) | h* r@1 |
|---|---:|---:|
| position (baseline) | 0.767 (0.69–0.84) | 0.26 |
| C_h^4 | 0.519 (0.41–0.63) | 0.33 |
| C_h^16 | 0.561 (0.46–0.65) | 0.37 |
| C_h^64 | 0.536 (0.44–0.63) | 0.41 |
| C_h^4_next | 0.472 (0.39–0.56) | 0.30 |
| C_h^16_next | 0.475 (0.40–0.56) | 0.26 |
| C_h^64_next | 0.436 (0.37–0.51) | 0.30 |
| O_h^4 | 0.502 (0.43–0.59) | 0.30 |
| O_h^16 | 0.507 (0.43–0.59) | 0.22 |
| O_h^64 | 0.508 (0.41–0.61) | 0.30 |
| O_h^4_next | 0.571 (0.50–0.64) | 0.44 |
| O_h^16_next | 0.540 (0.44–0.64) | 0.19 |
| O_h^64_next | 0.488 (0.39–0.59) | 0.30 |
| delta_handoff_32 | 0.545 (0.45–0.64) | 0.30 |
| delta_handoff_64 | 0.561 (0.49–0.63) | 0.44 |

Registered models are bit-identical to Phase C: M0 0.776, M1 0.479, M2 0.681,
M3 0.749; paired fold comparisons unchanged (0%/24%/40% ≥ M0).

## 4. Reading

- K sensitivity does not help: C_h^16 remains the best C variant; the K=4/64
  slices stay ~0.52–0.56, all well below position.
- Endpoint alignment matters directionally for C (the `h-1` state is better),
  and is mixed for O.  The best O variant (`O_h^4_next`, 0.571) has a
  bootstrap lower bound at 0.50, no robust edge over chance, and does not
  localize `h*` (r@1 0.44, same ceiling as cumulative features).
- W sensitivity: `delta_handoff_32` ≈ `delta_handoff_64`; both remain weak.
- The compatibility family C/M/O carries no handoff-viability signal at any
  registered K or alignment on this gold set.  Feature construction is not the
  bottleneck; sample size/label quality is.

## 5. Decision and next step

**NO-GO for Question 1 is confirmed and robust to the registered sensitivity
variants.**  No compatibility/executability model or feature variant beats
position under the registered protocol.

Next (Phase D, **not launched**): top up handoff gold via the frozen-pool
protocol to ≥40 confirmed positives without changing rescue thresholds; keep
`teacher_trace_unavailable` / `wrong_control_unavailable` / no-wrong as
separate skips; then re-run Phase C only after the top-up.

Raw feature tables and analysis JSON live outside Git under
`reachability_proxy_sensitivity_20260816/` (feature table sha256
`92798bafc1b2dec1…`).  No loss, training, checkpoint update, or Phase-7 launch
occurred.

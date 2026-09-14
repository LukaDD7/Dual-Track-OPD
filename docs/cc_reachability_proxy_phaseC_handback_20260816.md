# CC Handback: Phase C Registered Proxy Evaluation → NO-GO

Date: 2026-08-16

Execution contract: `docs/cc_handoff_viability_proxy_adjustment_20260815.md`
(Codex refocus brief).  This handback reports the corrected prefix-level
evaluation on the single-pass symmetric Top-100 cache and the registered
model gate.

## 1. Repository status

- Branch `codex/va-opd`, HEAD `2dbfc5e` (all Phase A/B/C code committed).
- STP canary / four-arm / SFT / FKL / RL: paused.  No loss, no training, no
  checkpoint update, and no Phase-7 launch occurred.

## 2. What changed since the 2026-08-15 handback

- **Evaluation repair (Phase A, commit `1f53d22`)**: removed label-conditioned
  row selection.  Evaluation unit is now every `(prompt_id, horizon)` row
  (`X_x,h -> Y_x,h`); splits by prompt; tie-correct AUROC; prompt-cluster
  bootstrap; grouped repeated CV; null-feature audit; regression tests.
- **Symmetric cache (Phase B, commit `5a929d3` + `02e0349` rescore script)**:
  single-pass per-trace forwards now persist student Top-100 IDs/log-probs,
  teacher cross-gather on student support, both directional tails, exact
  entropies, Top1-Top2 margins, and per-token `C_h^16` / `O_h^16`.  All 64
  traces (52 expansion + 12 Experiment-B) rescored to the v2 schema; v1 rows
  are invalidated by the schema check, so stale caches are never reused.
- **Registered evaluation (Phase C, commits `838aaff` / `2dbfc5e`)**:
  OOF grouped-CV logistic models M0-M3, paired delta vs M0, within-prompt h*
  metrics, strata reporting, and a versioned feature table
  (`proxy_feature_table.jsonl`, 194 rows, sha256 `e9e8b1b8…`).

## 3. Evidence (52 prompts / 194 rows / 27 positive prompts)

Single features (prefix-level, tie-correct AUROC, cluster-bootstrap CI95):

| feature | AUROC (CI95) |
|---|---:|
| position | 0.767 (0.69–0.84) |
| C_h_16 | 0.561 (0.46–0.65) |
| O_h_16 | 0.507 (0.43–0.59) |
| M_h_16 | 0.508 (0.42–0.59) |
| cum NLL / FKL | 0.555 / 0.557 |
| fkl_takeoff_64 | 0.312 (inverted direction) |
| delta_handoff_64 | 0.561 |

Registered models (OOF grouped 5-fold × 5 repeats, fit on train only):

| model | held-out AUROC | AUPRC | delta vs M0 | h* r@1 |
|---|---:|---:|---:|---:|
| M0 position | 0.776 | 0.643 | — | 0.26 |
| M1 compat (C/M/O) | 0.479 | 0.347 | −0.297 (0% folds ≥ M0) | 0.19 |
| M2 takeoff | 0.681 | 0.546 | −0.096 (24%) | 0.07 |
| M3 mechanistic-lite | 0.749 | 0.629 | −0.027 (40%) | 0.26 |

Within-prompt h* (27 positive prompts): no feature/model reaches useful
localization (best r@1 ≈ 0.44 for cumulative/contrast features; position 0.26;
all mean abs bin distance ≥ 0.85).

Strata AUROC: position 0.71–0.80 across all three strata; fkl_takeoff_64
0.62–0.74; C_h_16 0.55–0.63; O_h_16 0.46–0.55.

## 4. Decision (registered §8 gate)

**NO-GO for Question 1 (training).**  On prompt-disjoint evidence:

- no compatibility/executability model has AUROC confidence above chance with
  AUPRC above prevalence while improving on position: M1 compat is at/below
  chance (0.479), M2/M3 stay below M0 position (0.681/0.749 vs 0.776) and lose
  the paired fold comparison (0%/24%/40% folds ≥ M0);
- within-prompt h* ranking is not useful or stable (r@1 ≤ 0.44, unstable
  across features and strata);
- position (progress) remains the only robust prompt-level signal, which the
  brief explicitly does not accept as a substitute for the handoff-location
  proxy.

The compatibility family C/M/O — the core of the registered Question-1 design —
does not carry the handoff-viability signal at K=16 on this gold set.

## 5. Recommended next steps (not launched)

1. Collect/refine handoff gold (Phase D): top up the frozen-pool protocol to
   ≥40 confirmed positives without changing rescue thresholds; keep
   `teacher_trace_unavailable` / `wrong_control_unavailable` / no-wrong as
   separate skips.
2. Before re-running the gate, re-examine feature construction on the v2
   cache — endpoint alignment and windowed compatibility variants (K∈{4,64}
   sensitivity, takeoff window W∈{32,64}) — as registered sensitivity, not a
   new feature search.
3. Only if a compatibility/executability model then beats position under
   paired prompt-cluster bootstrap with stable h* ranking, propose Question-2
   (barrier `[a*, h*]`) and the selective Top-100 FKL conversion, still behind
   explicit user approval.

No loss, training, checkpoint update, or Phase-7 launch occurred.

# CC Handback: Phase D gold top-up -> Phase C gate re-run (v2 combined, 76 prompts)

Date: 2026-08-16

Branch `codex/va-opd`.  No loss, training, checkpoint update, or Phase-7
launch occurred.  This handback re-runs the registered Phase C gate on the
combined gold after the Phase D top-up (20260816 manifest).

## 1. What ran

- **Phase D top-up** (`launch_reachability_expansion.sh`, 8 GPUs, 20260816
  manifest): 60 prompts frozen from the pool (mixed_support 20 +
  no_correct_observed 40).  Adaptive rescue completed 24 prompts with wrong
  controls, 87 rescue comparisons, 11 minimal-rescue prompts.
- **Phase 5**: frozen proxy scored the 24 new prompts into
  `reachability_proxy_20260816`.
- **Combined strict study**: `analyze-combined` over 3 token dirs / 4 rescue
  dirs / 3 proposal dirs -> `reachability_proxy_combined_v2_20260816`.

## 2. Coverage (combined)

| cohort | prompts | rows | rescue-positive prompts |
|---|---:|---:|---:|
| previous (A/B/C) | 52 | 194 | 27 |
| Phase D top-up | 24 | 87 | 11 |
| combined | 76 | 281 | 38 (51 positive rows) |

New and old cohorts are disjoint (no prompt overlap).  All expected minimal
horizons are reproduced (`minimal_horizons_reproduced: true`).

## 3. Prompt-level evaluation (`analyze-combined`)

One label per prompt (scaffoldable vs not); 40% test split by prompt
(seed 20260815).

| feature | full AUROC (CI95) | full AUPRC | held-out AUROC | held-out AUPRC |
|---|---:|---:|---:|---:|
| position | 0.641 (0.513-0.762) | 0.702 | 0.764 | 0.799 |
| cum_student_nll | 0.824 (0.723-0.911) | 0.832 | 0.782 | 0.741 |
| cum_top100_fkl_tail | 0.813 (0.713-0.900) | 0.825 | 0.773 | 0.738 |

## 4. Registered gate (`salvage`, row-level, 281 rows)

OOF grouped 5-fold x 5 repeats; models fit on train folds only.

| model | held-out AUROC | AUPRC | delta vs M0 | h* r@1 | +-1 bin |
|---|---:|---:|---:|---:|---:|
| M0 position | 0.751 | 0.597 | - | 0.237 | 0.553 |
| M1 compat (C/M/O) | 0.450 | 0.299 | -0.301 (0% folds >= M0) | 0.237 | 0.658 |
| M1 compat partial (M) | 0.501 | 0.323 | -0.250 (0% folds) | 0.263 | 0.632 |
| M2 takeoff | 0.682 | 0.527 | -0.068 | 0.184 | 0.632 |
| M3 mechanistic-lite | 0.741 | 0.586 | -0.010 | 0.263 | 0.579 |

Single features (row-level, tie-correct): position 0.750; cum NLL / FKL tail
0.586 / 0.587; C_h_16 0.557; M_h and O_h families 0.50-0.52 (chance);
fkl/nll takeoff inverted (0.32-0.38, i.e. anti-correlated as before).

Note: `analyze-combined` and `salvage` report different position AUROC
(0.641 vs 0.750) because the former aggregates to one label per prompt while
the latter is row-level; both use tie correction.

## 5. Verdict (registered gate, unchanged)

**NO-GO for Question 1 (training).**  On the larger, disjoint gold set:

- M1 compat is at/below chance (0.450) and loses 100% of paired folds to M0;
- M2/M3 stay below M0 position (0.682 / 0.741 vs 0.751);
- within-prompt h* localization remains poor (r@1 <= 0.26, +-1 bin ~0.55-0.66).

The top-up moved positive prompts 27 -> 38 (target >= 40; 51 positive rows),
but the gap is not the blocker: the compatibility family is far below the
gate, so 2 more positives would not change the decision.

This is consistent with the rebuttal
(`docs/cc_reachability_proxy_rebuttal_20260816.md`): the gate metric (h*
min-horizon) and the C/M/O feature family are the problem, not gold size.
Next step is the section-10 closed-loop verification (S0 schedule ->
S1 self-judgment -> S2 online audit -> S3 trigger), pending Codex/ChatGPT
review.

## 6. Artifacts (outputs, not in git)

- `support_aware_opd/prefix_intervention_20260816_merged/` (new rescue gold)
- `support_aware_opd/reachability_proxy_20260816/` (new proxy rows)
- `support_aware_opd/reachability_proxy_combined_v2_20260816/`
  (`proxy_analysis.json`, `proxy_study.csv`,
  `proxy_feature_table.jsonl`, `proxy_prefix_salvage_analysis.json`)

Reproduce:

```bash
OUT=$DTOPD_OUTPUT_ROOT/support_aware_opd
$DTOPD_PYTHON -m dual_track_opd.support_aware.reachability_proxy analyze-combined \
  --token-dirs "$OUT/reachability_proxy_20260814" "$OUT/reachability_proxy_20260815" "$OUT/reachability_proxy_20260816" \
  --rescue-dirs "$OUT/prefix_intervention_20260806_merged" "$OUT/prefix_intervention_20260815_merged" "$OUT/prefix_intervention_20260815b_merged" "$OUT/prefix_intervention_20260816_merged" \
  --proposal-dirs "$OUT/proposal_feasibility_20260805_merged" "$OUT/proposal_feasibility_20260815_merged" "$OUT/proposal_feasibility_20260816_merged" \
  --output-dir "$OUT/reachability_proxy_combined_v2_20260816"
```

and the same token/rescue/proposal sources with `salvage` for the registered
gate.

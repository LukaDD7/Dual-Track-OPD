# Visual Handoff - v3 final report (full-vocab JS, terminal NO)

Date: 2026-08-17.  Branch `codex/va-opd`.

Final Question-A round per the preregistered gate
(`docs/chatgpt_research_dialogue_handoff_20260817.md`).  Supersedes the v1/v2
reports: v1 had an invalid continuation verdict, v2 used the reference-token
score.  v3 uses the corrected full-vocabulary Jensen-Shannon attribution.

## 1. Run facts (validated)

- 99 prompts / 0 shard errors / 4 shards; schema
  `support-aware-visual-handoff-v3` on all records.
- `V_t^M = JS(p_M(.|full), p_M(.|degraded))`, blank null control; BIC gate
  requires significance; `h_V = end(B_(tau-1))`; continuation around that
  center (K=4, K=8 if ambiguous).
- Degraded mode `lowres_20_bilinear_nearest`; output
  `visual_handoff_js_v3_20260817` (kept separate from v2).

## 2. Results

| metric | value |
|---|---:|
| records | 99 |
| with significant change point `h_V` | 19 |
| with `h_V` and rescue gold | 7 |
| enrichment (`h_V` within +-1 block of `h*`) | 0/7 = 0.0 |
| null-control enrichment | 0/7 = 0.0 |
| median `h_V` block fraction | 0.07 |
| `h_V` before `h*` | 6/7 |
| per-token JS `V_T` median / p90 | 0.0003 / 0.007 |
| per-token JS `V_S` median / p90 | 0.0005 / 0.010 |
| per-token JS null `V_T` median / p90 | 0.0013 / 0.038 |
| continuation pass rate at `h_V` (full) | 0.037-0.039 |
| continuation pass rate at `h_V` (degraded) | 0.026 |

Reference: rescue protocol teacher-prefix pass rate at `h*` ~0.52.

## 3. Interpretation

1. Even the distribution-level JS attribution is per-token sparse on this
   dataset: median JS ~0.0003-0.0013 nats.  Most teacher-trace tokens are
   text-template tokens whose full distributions barely change under image
   counterfactuals (the earlier causal-probe 0.69 value was a selected-window
   maximum, not the per-token median).
2. Wherever a significant change point exists, it does not land near `h*`
   (enrichment 0.0 on both degraded and null curves) and continuation there
   is ~4% vs ~52% at `h*`.

## 4. Decision (preregistered gate)

**Stop using the visual score as a prefix localizer.**  Keep visual
attribution only as a mechanism-analysis axis (e.g., "which tokens are
image-sensitive at all" - the sparse p90 tail).  Do not retry further
degradations, smoothers, or change-point variants on Question A.

Current focus: the region/path-level acquisition diagnostic
(`region_acquisition.py`), which tests whether continuous pre-`h*` teacher
regions transfer path support under soft FKL / full-vocab RKL / hard CE /
control - independent of the visual question.

## 5. Artifacts

- `support_aware_opd/visual_handoff_js_v3_20260817/visual_handoff_records.jsonl`
- `support_aware_opd/visual_handoff_js_v3_20260817/visual_handoff_report.json`

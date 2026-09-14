# Visual Handoff Diagnostic - v2 report (clean measurement, NO for localization)

Date: 2026-08-17.  Branch `codex/va-opd`, HEAD `c0eb63a`.

Supersedes `cc_visual_handoff_report_v1_20260817.md`.  v1's continuation
column was invalid (empty-text verdict) and its degradation was too weak; both
are fixed here.  The v2 run is the first *valid* measurement of Question A.

## 1. Run facts (validated)

- 99 prompts / 0 shard errors; 4 shards x 1 GPU pair (8xH200, cu132 env).
- `schema_version = support-aware-visual-handoff-v2` on all 99 records.
- Degraded condition `lowres_20_bilinear_nearest` + blank null control
  (`DeltaV_null`), BIC change point on both curves.
- Continuation verdict now reads `generation.response_text_display` (same
  path as the rescue protocol).

## 2. Results

| metric | value |
|---|---:|
| records | 99 |
| with change point `h_V` (lowres_20) | 58 |
| with `h_V` and rescue gold | 24 |
| enrichment (`h_V` within +-1 block of `h*`) | 2/24 = 0.083 |
| `h_V` significant change points | 34 |
| `h_V_null` (blank) change points | 40 |
| null enrichment | 2/17 = 0.118 |
| median `h_V` block fraction | 0.09 |
| median `h*` block fraction | 0.22 |
| `h_V` before `h*` | 19/24 |
| median abs `V_T` (lowres_20) | 0.0054 |
| median abs `V_S` (lowres_20) | 0.0090 |
| median abs `V_T` / `V_S` (blank) | 0.0125 / 0.0194 |
| continuation pass rate at `h_V - 1 / h_V / h_V + 1` | 0.084 / 0.097 / 0.121 |
| continuation pass rate at `h_V` (degraded) | 0.116 |

Reference: the rescue protocol's teacher-prefix arm at the real `h*` has
~0.52 pass rate with the same continuation machinery, so `h_V` sits far
outside the usable handoff region.

## 3. Interpretation

**Question-A localization is a NO.**  Two independent facts:

1. Per-token reference log-prob visual attribution is noise-level: even the
   blank-image condition moves `log P(y_t^T | condition)` by only ~0.01-0.02
   nats on the teacher-trace tokens (geometry responses are template-heavy,
   the chosen token is predictable from text alone).  The `DeltaV` block
   curves are flat (mean by decile ~0.003 -> 0.004, no sustained drop), so the
   BIC fires on early arbitrary transitions.
2. Wherever `h_V` lands, it does not enrich near `h*` (0.08-0.12 vs the
   ~3/93-block chance window ~0.03), and continuation there is ~10% vs ~52%
   at `h*`.

The earlier causal probe's strong *distributional* JS signal (blank vs full,
~0.69) does not transfer to *reference-token* support: the models' full
distributions shift, but the specific teacher token stays almost as probable.
`V_t` as defined in the frozen plan therefore cannot localize the handoff on
this dataset.

## 4. Decision (frozen plan)

- Visual attribution downgraded to **mechanism-analysis axis only**: no
  visual routing, no `h_V`-based truncation, no visual FKL region selection.
- Continue with **Question B (operator need)** on the rescue-valid prefixes:
  the micro-operator experiment (`micro_operator.py`) is independent of A's
  outcome and is the current focus.
- The visual story is not refuted as a *mechanism* (teacher may still
  externalize visual evidence), but it is not measurable as a *localization
  proxy* with this counterfactual estimator; stronger interventions
  (e.g., full-distribution attribution on figure-critical tokens) would be a
  separate research question, not a blocking prerequisite.

## 5. Artifacts

- `support_aware_opd/visual_handoff_20260816/visual_handoff_records.jsonl`
- `support_aware_opd/visual_handoff_20260816/visual_handoff_report.json`

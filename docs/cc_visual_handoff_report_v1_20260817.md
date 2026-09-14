# Visual Handoff Diagnostic - v1 full run report

Date: 2026-08-17.  Branch `codex/va-opd`.

Question-A (frozen plan) first full run on the shared cu132 instance
(8xH200, `va-opd-qwen35-v090-cu132-r595-v1`, driver 595/CUDA 13.2).

## 1. Run facts

- 99 prompts (all retained teacher traces with cohort rows), 38 with rescue
  gold `h*`; 4 shards x 1 GPU pair; 0 shard errors.
- Degraded condition: `blur_sigma_2` (this run).
- Outputs (not in git):
  `support_aware_opd/visual_handoff_20260816/visual_handoff_records.jsonl`
  + `visual_handoff_report.json`.

## 2. Results

| metric | value |
|---|---:|
| records | 99 |
| with BIC change point (`h_V`) | 78 |
| significant change points | 44 |
| with `h_V` and rescue gold | 25 |
| `h_V` within +-1 block of `h*` | 3 |
| enrichment | 0.12 |
| median `h_V` block fraction | 0.10 |
| median `h*` block fraction | 0.28 |
| median abs `V_T` (per block) | 0.0034 |
| median abs `V_S` (per block) | 0.0054 |
| continuation pass rate at `h_V` | invalid (see 3) |

`h_V` is systematically early (median block fraction 0.10 vs 0.28 for `h*`);
the DeltaV curve is essentially noise under `blur_sigma_2` (per-token
log-prob differences of order 1e-3), so the BIC picks early arbitrary
transitions that do not localize the rescue handoff.

## 3. Measurement bug found (fixed in this commit)

`generate_continuation` returns `{"generation": <record>, ...}` with **no
`"text"` key**.  The v1 judge read `generation.get("text")`, so every
continuation was verified against an empty string -> all 234 continuation
evaluations (and the degraded arm) reported 0% pass rate.  That column is
invalid in v1.  Fixed to verify
`generation["generation"].response_text_display` (the same path the rescue
protocol uses).

## 4. Interpretation

- At `blur_sigma_2` the counterfactual is too weak for either model: the
  teacher trace tokens barely change log-prob under blur, so
  `DeltaV = V_T - V_S` carries no structure.  This is consistent with the
  earlier causal probe (blur ~ no signal, blank ~ strong signal).
- The visual-handoff claim is therefore **not supported by v1**, but this is
  not yet a fair NO: the intervention strength was too weak and the
  continuation measurement was broken.
- Per the frozen plan ("if the data is not like this, abandon the
  visual-centric story"), v2 must first fix the measurement before any
  conclusion.

## 5. v2 plan (implemented, needs re-run)

1. Default degraded mode -> `lowres_20_bilinear_nearest` (keeps layout,
   removes fine detail: angles/labels/coordinates).
2. Add blank-image null control: per block also `DeltaV_null`; BIC change
   point on both curves (`h_V`, `h_V_null`).
3. Continuation verification fixed; pass rate at `h_V` (and tau) now valid.
4. Re-run the same 4-shard launch; enrichment is computed for both change
   points.

If v2 also shows no enrichment, the visual-centric story is dropped and
Question B (operator need) proceeds on the rescue-valid prefixes regardless.

## 6. Reproduce

```bash
git pull origin codex/va-opd
export DTOPD_PYTHON=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/va-opd-qwen35-v090-cu132-r595-v1/bin/python
export DTOPD_OUTPUT_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs
export DTOPD_MODEL_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models
export PYTHONPATH="$PWD/src"
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/hpc/launch_visual_handoff.sh 0:1,2:3,4:5,6:7
```

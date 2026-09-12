# MMF 14-Bench Eval — Status & openpyxl Scoring Bug (2026-09-08)

Snapshot of the 7-A dual-arm VLMEvalKit run (`T20260907-152150`, judge
Qwen3-VL-32B vLLM :8801 on GPU 3) plus the root-cause and fix for the
scoring-stage crash observed on the base arm.

## Live status (01:25 UTC refresh)

| Arm | GPU | Bench | Progress |
|---|---|---|---|
| base | 0 | DynaMath (5th bench) | infer started 01:17 |
| tailsft | 1 | MMMU (1st bench) | 747/1050 (71%), ~71-83 s/it |

**GSPO run #3 (GPUs 4-7)**: relaunched by user ~00:42, filter fix
(num_proc=16) verified working, engine init + Triton JIT at 00:53,
step:0 validation ran, **step:3 at 01:20** — training is live. Reward
trace: `critic/score/mean` 0.2578 (step 1) → 0.4844 (step 2);
response_length/mean ~7.1-7.4K (max 16384, i.e. some rollouts hit
the response cap — expected with 16K budget on MFR long-CoT).

Both run.py processes alive; logs still growing; judge only logs when a
judge-required bench is scoring, so its idle mtime is normal between
Math-bench scoring phases.

## Per-bench state (status.json)

**base** (4 of 14 attempted, 2 fully scored):

| Bench | State | Score |
|---|---|---|
| MMMU_DEV_VAL | infer done, **score missing (openpyxl bug)** | re-derive later |
| MathVista_MINI | infer done, **score missing (openpyxl bug)** | re-derive later |
| MathVision | done ✅ | Overall acc **25.13** (764/3040) |
| MathVerse_MINI | done ✅ (01:17) | Text Dominant Overall **37.56** (primary); Vision Only 26.65 / Text Lite 36.17 / Vision Intensive 34.14 |
| DynaMath | infer in progress | — |

MathVerse scoring went through the judge (Qwen3-VL-32B) and openpyxl —
both healthy post-fix.

**tailsft** — MMMU_DEV_VAL: infer in progress (71%), untouched by the bug.

## The openpyxl scoring bug

**Symptom** (base arm, 23:52 / 23:57):

```
Model qwen3vl_8b_base x Dataset MMMU_DEV_VAL combination failed:
`Import openpyxl` failed.  Use pip or conda to install the openpyxl package.
```

Inference completed for both benches; the crash fires at the *scoring*
entry point — `evaluate_heuristic()` → `load(eval_file)` → `pd.read_excel`
(`vlmeval/dataset/image_vqa.py:603`, `vlmeval/smp/file.py:269`).

**Root cause**: the eval env
(`envs/va-opd-qwen35-v090-cu132-r595-v1`) had openpyxl 3.1.5 installed but
its runtime dependency `et_xmlfile` was missing, so
`import openpyxl` chain-dies at `openpyxl/xml/functions.py:36`
(`from et_xmlfile import xmlfile`) and pandas re-raises as the misleading
"install openpyxl" ImportError.

**Fix applied** (2026-09-08 00:2x, CPU instance): copied the pure-Python
`et_xmlfile` 2.0.0 package (+ dist-info) from `envs/qwen3vl-eval` into the
eval env's site-packages; verified `import openpyxl` now succeeds and
`pd.read_excel` reads the exported prediction files (1050/1000 rows,
predictions non-null).

No network needed — this is why `pip install` was not used; the instance's
pip could not resolve any index.

## What still needs manual follow-up

1. The two running run.py processes import openpyxl lazily at each bench's
   scoring stage, so they pick up the fix automatically from the next bench
   onward (MathVision's scoring should succeed).
2. The already-skipped base MMMU_DEV_VAL / MathVista_MINI **scores** must
   be re-derived after the run finishes: predictions are intact
   (`qwen3vl_8b_base_{MMMU_DEV_VAL,MathVista_MINI}.xlsx`, valid zips);
   rerun with `--reuse` (script `scripts/eval/launch_mmf_eval.sh base 0
   --reuse`) or hand-score from the xlsx. Details in manuscript-tailsft-gpu
   第 6 步注意点 block (2026-09-08 00:20 note).
3. tailsft arm is unaffected so far (its MMMU scoring comes later and will
   import the fixed openpyxl).

## Related

- GSPO second-death postmortem (same night): commit `210fae8` wired
  `data.filter_overlong_prompts_workers=16` into
  `run_gspo_mmf_tailsft.sh` — restart command manuscript line 509.

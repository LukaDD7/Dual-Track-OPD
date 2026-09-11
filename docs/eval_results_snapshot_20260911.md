# Evaluation results snapshot — 2026-09-11

This snapshot records the final persisted state after the 2026-09-10 GPU
instances were reclaimed. Raw outputs, caches, and model weights remain
outside Git.

## MMF-14 paper protocol

Protocol: VLMEvalKit, `max_new_tokens=32768`, `max_model_len=262144`,
temperature 0, repetition penalty 1.05, judge `Qwen3-VL-32B-Instruct`.

| Arm | Status |
|---|---|
| Base | 14/14 complete |
| PTD-PO r4 step390 | 14/14 inference complete; CharXiv reasoning needs one rescore |
| TailSFT | 3/14 complete; MathVerse interrupted at 2,463/3,940 |

### Base

| Bench | Score |
|---|---:|
| MMMU_DEV_VAL | 65.89 |
| MathVista_MINI | 66.30 |
| MathVision | 25.13 |
| MathVerse_MINI Text Dominant | 37.56 |
| DynaMath | 41.46 |
| LogicVista | 40.04 |
| VisuLogic | 10.00 |
| ScienceQA_VAL | 93.37 |
| RealWorldQA | 71.50 |
| MMBench_DEV_EN | 85.57 |
| MMStar | 65.20 |
| AI2D_TEST | 85.78 |
| CharXiv descriptive | 81.18 |
| CharXiv reasoning | 45.70 |

### PTD-PO r4 step390

| Bench | Score |
|---|---:|
| MMMU_DEV_VAL | 65.33 |
| MathVista_MINI | 65.50 |
| MathVision | 26.55 |
| MathVerse_MINI Text Dominant | 37.94 |
| DynaMath | 39.17 |
| LogicVista | 41.16 |
| VisuLogic | 5.20 |
| ScienceQA_VAL | 93.42 |
| RealWorldQA | 72.29 |
| MMBench_DEV_EN | 85.65 |
| MMStar | 65.20 |
| AI2D_TEST | 85.70 |
| CharXiv descriptive | 80.45 |
| CharXiv reasoning | pending rescore |

### TailSFT

| Bench | Score / state |
|---|---:|
| MMMU_DEV_VAL | 63.56 |
| MathVista_MINI | 67.50 |
| MathVision | complete; score in status.json |
| MathVerse_MINI | interrupted at 2,463/3,940 |

MMF status files:

- Base: `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/eval/vlmeval_runs/qwen3vl_8b_base/status.json`
- TailSFT: `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/eval/vlmeval_runs/qwen3vl_8b_tailsft/status.json`
- PTD-PO: `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/eval/vlmeval_runs/qwen3vl_8b_ptdpo_r4/status.json`

## Project15 v1 offline recovery

The networkless-GPU failures for MindCube, ScienceQA, MMSI, BLINK, MathVista,
MMVet, and VQAv2 were fixed with local parquet tasks under
`eval_tasks/opd_v1_offline/`.

### Offline core

| Arm | MindCube | ScienceQA | MMSI | BLINK |
|---|---:|---:|---:|---:|
| Base | 0.2975 | 0.9465 | 0.3070 | 0.6512 |
| PTD-PO | 0.2996 | 0.6946 | 0.2980 | 0.1394 |
| TailSFT | 0.0534 | 0.0000 | 0.1870 | 0.0000 |

### Offline judged

| Arm | MathVista | MMVet |
|---|---:|---:|
| Base | 72.70 | 74.27 |
| PTD-PO | 73.30 | 72.48 |
| TailSFT | interrupted at 2,816/3,000 cached MathVista responses | pending |

### Offline VQAv2

All three VQAv2 runs generated many successful server requests, but their
response-cache files were not persisted before the GPU instances were
reclaimed. Treat the three VQAv2 runs as not persisted and restart them from
the beginning.

Project15 offline run root:

`/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline`

## Project15 v2 long-output correction protocol

All three arms are 4/4 complete.

| Arm | GQA | DynaMath | ViewSpatial | MMMU-Pro |
|---|---:|---:|---:|---:|
| Base | 0.6163 | 0.6291 | 0.4090 | 0.4821 |
| PTD-PO | 0.6152 | 0.6218 | 0.4091 | 0.4763 |
| TailSFT | 0.3209* | 0.5762 | 0.1947 | 0.4040 |

*TailSFT GQA is not capability-comparable because 41.6% of responses hit the
4096-token ceiling. Ceiling-hit accuracy was 0.000 while non-ceiling accuracy
was 0.5492. The full audit is retained below.

### GQA v2 TailSFT truncation audit

| Audit | Value |
|---|---:|
| GQA samples | 12,578 |
| Responses at the 4096-token ceiling | 5,229 |
| Ceiling-hit share | 41.6% |
| Accuracy on ceiling-hit samples | 0.000 |
| Accuracy on non-ceiling samples | 0.5492 |
| Accuracy with an explicit `<answer>` tag | 0.5887 |
| Accuracy without an explicit `<answer>` tag | 0.2020 |

Base and PTD-PO are unaffected: their median output length is 4 tokens.
The earlier v1 TailSFT GQA score of 13.85 under a 128-token cap was almost
certainly the same truncation mechanism.

## CharXiv reasoning scoring fix

`charxiv.py` built the judge prompt with `line["prediction"]` directly. Numeric
and NaN predictions caused:

`TypeError: replace() argument 2 must be str, not float`

The prediction is now cast to a string before replacement. The patch is
`patches/vlmeval/0001-charxiv-cast-prediction-to-string.patch`, applied to the
shared VLMEvalKit runtime. Base CharXiv reasoning was successfully rescored to
45.70. PTD-PO has 881/1000 judge results already cached and needs only a
rescore continuation.

## Remaining work

1. PTD-PO CharXiv reasoning rescore: 881/1000 judge results cached.
2. Resume TailSFT offline judged MathVista: 2,816/3,000 responses cached.
3. Resume MMF TailSFT MathVerse: 2,463/3,940 samples.
4. Restart Project15 offline VQAv2 for all three arms.
5. Optionally add a short-answer GQA protocol for TailSFT capability analysis.

## HF and source assets

- PTD-PO r4 step390 model: `lzy-vlm-lab-opd/qwen3vl-8b-ptdpo-r4-step390`
- Shared eval framework: `lzy-vlm-lab-opd/dual-track-opd-eval-framework`
- GitHub shared branch: `eval-shared`

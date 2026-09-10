# Evaluation results snapshot — 2026-09-09

This snapshot records the current state of the three active evaluation tracks:

1. MMF-14 paper protocol
2. Project-15 contract protocol
3. Project-15 v2 long-output correction protocol

Raw outputs, caches, and model weights remain outside Git. This document records
scores, completion status, and the paths needed to audit the raw files.

## MMF-14 paper protocol

Protocol: VLMEvalKit, `max_new_tokens=32768`, `max_model_len=262144`,
temperature 0, repetition penalty 1.05, judge `Qwen3-VL-32B-Instruct`.

| Arm | Bench | Status | Score |
|---|---|---|---:|
| base | MMMU_DEV_VAL | done | 65.89 |
| base | MathVista_MINI | done | 66.30 |
| base | MathVision | done | 25.13 |
| base | MathVerse_MINI (Text Dominant) | done | 37.56 |
| base | DynaMath | done | 41.46 |
| base | LogicVista | done | 40.04 |
| base | VisuLogic | running (90%) | — |
| TailSFT | MMMU_DEV_VAL | done | 63.56 |
| TailSFT | MathVista_MINI | done | 67.50 |
| TailSFT | MathVision | running (27%) | — |
| PTD-PO r4 step390 | MMMU_DEV_VAL | done | 65.33 |
| PTD-PO r4 step390 | MathVista_MINI | done | 65.50 |
| PTD-PO r4 step390 | MathVision | done | 26.55 |
| PTD-PO r4 step390 | MathVerse_MINI (Text Dominant) | done | 37.94 |
| PTD-PO r4 step390 | DynaMath | done | 39.17 |
| PTD-PO r4 step390 | LogicVista | done | 41.16 |
| PTD-PO r4 step390 | VisuLogic | running (30%) | — |

Early read:

- PTD-PO is slightly ahead of base on MathVision (+1.41), LogicVista
  (+1.12), and MathVerse (+0.38), slightly behind on MathVista (-0.80), and
  behind on DynaMath.
- TailSFT currently has the strongest MMMU and MathVista scores among the
  three arms, but the run is still early in the remaining MMF-14 benches.

MMF status files:

- Base: `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/eval/vlmeval_runs/qwen3vl_8b_base/status.json`
- TailSFT: `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/eval/vlmeval_runs/qwen3vl_8b_tailsft/status.json`
- PTD-PO: `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/eval/vlmeval_runs/qwen3vl_8b_ptdpo_r4/status.json`

## Project-15 contract protocol (v1)

Protocol: `configs/eval/project_vision_opd.yaml`; judge
`Qwen3-VL-32B-Instruct`.

### Base

| Bench | Status | Score |
|---|---|---:|
| ViewSpatial | complete | 42.31 |
| MindCube | complete | result not summarized |
| GQA | complete | 61.64 |
| VQAv2 | complete | result not summarized |
| ScienceQA | complete | result not summarized |
| MV-MATH | complete | replay metric not finalized |
| ReMI | complete | 96.73 |
| DynaMath | complete | 62.79 |
| MMSI | complete | result not summarized |
| BLINK | complete | result not summarized |
| MMMU-Pro | complete | 39.83 |
| MathVerse | complete | 52.36 |
| MathVista | complete | result not summarized |
| MMBench | complete | 85.14 |
| MMVet | complete | result not summarized |

### TailSFT

| Bench | Status | Score |
|---|---|---:|
| ViewSpatial | complete | 6.79 |
| MindCube | complete | result not summarized |
| GQA | complete | 13.85 |
| VQAv2 | complete | result not summarized |
| ScienceQA | complete | result not summarized |
| MV-MATH | complete | replay metric not finalized |
| ReMI | complete | 45.08 |
| DynaMath | complete | 57.88 |
| MMSI | complete | result not summarized |
| BLINK | complete | result not summarized |
| MMMU-Pro | complete | 23.53 |
| MathVerse | complete | 61.02 |
| MathVista | complete | result not summarized |
| MMBench | complete | 47.94 |
| MMVet | complete | result not summarized |

### PTD-PO r4 step390

The first project15 PTD-PO run had two invalid benches (`DynaMath` and
`MMMU-Pro`) caused by requests hitting the wrong vLLM server/model and
returning 404. The clean redo completed both:

| Bench | Status | Score |
|---|---|---:|
| DynaMath | clean redo complete | 62.12 |
| MMMU-Pro | clean redo complete | 38.96 |

A full 15-bench PTD-PO rerun (`project15_ptdpo_r4_step390_full`) is currently
running. Its purpose is to put all PTD-PO v1 results into one clean run
directory; the redo scores above are already valid.

Important interpretation note: v1's short token caps (GQA 128, ViewSpatial 256,
MMMU-Pro 2048, etc.) systematically truncate long-CoT checkpoints. TailSFT's
v1 ViewSpatial/GQA/MMMU-Pro scores should not be interpreted as capability
without the v2 protocol.

Project-15 run directories:

- Base: `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/project15_base_qwen3vl8b_{nojudge,judged}`
- TailSFT: `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/project15_tailsft_mmf122k_1ep_{nojudge,judged}`
- PTD-PO redo: `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/project15_ptdpo_r4_step390_redo_nojudge`
- PTD-PO full rerun: `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/project15_ptdpo_r4_step390_full_nojudge`

## Project-15 v2 protocol

Protocol: `configs/eval/project_vision_opd_v2.yaml`, uniform 4096-token output
budget and answer-tag-aware deterministic scoring for GQA, DynaMath,
ViewSpatial, and MMMU-Pro.

Current completed scores:

| Arm | Bench | Status | Score |
|---|---|---|---:|
| base | GQA | done | 0.6163 |
| base | DynaMath | done | 0.6291 |
| base | ViewSpatial | done | 0.4090 |
| base | MMMU-Pro | running | — |
| TailSFT | GQA | done, not capability-comparable | 0.3209 |
| TailSFT | DynaMath | done | 0.5762 |
| TailSFT | ViewSpatial | pending | — |
| TailSFT | MMMU-Pro | pending | — |
| PTD-PO | GQA | done | 0.6152 |
| PTD-PO | DynaMath | done | 0.6218 |
| PTD-PO | ViewSpatial | running | — |
| PTD-PO | MMMU-Pro | pending | — |

v2 result root:

`/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_v2}`

The literal trailing `}` is a historical environment-variable issue in the v2
runner's default output-root expansion. The files under it are valid; a clean
run should set `DTOPD_EVAL_ROOT` explicitly.

### GQA v2 TailSFT truncation audit — 2026-09-10

The TailSFT GQA v2 score of 0.3209 is not a valid capability comparison.
Although the v2 protocol increased GQA's output budget from v1's 128 tokens to
4096 tokens, TailSFT still hit the generation ceiling on many long-CoT samples:

| Audit | Value |
|---|---:|
| GQA samples | 12,578 |
| Responses at the 4096-token ceiling | 5,229 |
| Ceiling-hit share | 41.6% |
| Accuracy on ceiling-hit samples | 0.000 |
| Accuracy on non-ceiling samples | 0.5492 |
| Accuracy with an explicit `<answer>` tag | 0.5887 |
| Accuracy without an explicit `<answer>` tag | 0.2020 |

The prompt matches lmms-eval's official GQA prompt
(`Answer the question using a single word or phrase.`), but TailSFT frequently
does not follow it and continues reasoning until the ceiling. Manual samples
end mid-sentence without a final answer. The scorer then compares the full
unfinished response against the one-word gold answer, which explains the large
drop.

Base and PTD-PO are not affected: their median output length is 4 tokens and
their GQA scores are 0.6163 and 0.6152. The earlier v1 TailSFT GQA score of
13.85 under a 128-token cap was almost certainly caused by the same mechanism.

Do not use TailSFT GQA v1/v2 as a capability conclusion. If needed, rerun GQA
with a new short-answer protocol (for example, a new `gqa_v3_short` task,
stronger one-word-only instruction, and a 16–32 token cap) or a separate
answer-tag-forced long-CoT protocol. Use a new run name so the completed GQA
manifest entry and response cache do not skip the rerun.

## HF and source assets

- PTD-PO r4 step390 model: `lzy-vlm-lab-opd/qwen3vl-8b-ptdpo-r4-step390`
- Shared eval framework: `lzy-vlm-lab-opd/dual-track-opd-eval-framework`
- GitHub shared branch: `eval-shared`

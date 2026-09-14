# Project15 offline dataset recovery — 2026-09-10

## Problem

The GPU instances have no outbound network. The original Project15 v1 runner set
`HF_HUB_OFFLINE=1`, but seven benchmarks had no prepared `datasets` cache:

- MindCube
- VQAv2
- ScienceQA
- MathVista
- MMSI-Bench
- BLINK
- MMVet

The old `run_manifest.json` marks those failed benchmarks as `completed` because
the subprocess return code was still zero. Do not resume the old full run for
these benchmarks; the resume logic will skip them.

## Fix

The local NFS already contained parquet snapshots for six benchmarks. MindCube
required the current parquet release (`full/combined-*`), which is downloaded by:

```bash
python scripts/sft_rl/download_mindcube_offline.py --workers 12
```

The new task set under `eval_tasks/opd_v1_offline/` loads those files directly
through the `parquet` builder, so lmms-eval does not contact Hugging Face.
`configs/eval/project_vision_opd_v1_offline.yaml` maps the same Project15
benchmark IDs to the offline task names.

Offline validation completed on 2026-09-10:

| Benchmark | Rows |
|---|---:|
| MindCube full/combined | 21,154 |
| VQAv2 validation | 214,354 |
| ScienceQA-IMG test | 2,017 |
| MathVista testmini | 1,000 |
| MMSI-Bench test | 1,000 |
| BLINK val | 1,901 |
| MMVet test | 218 |

All nine lmms-eval task YAMLs (including the three MathVista prompt variants)
were also loaded with `TaskManager` and successfully built two requests each.

## Run commands

Use a new run name; do not reuse `project15_ptdpo_r4_step390_full_*` for resume.
For the no-judge missing benchmarks:

```bash
EVAL_CKPT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/hf/qwen3vl_ptdpo_r4_step390 \
EVAL_RUN_NAME=project15_ptdpo_r4_step390_offline \
EVAL_SERVED_MODEL=Qwen3-VL-8B-PTDPO-R4 \
EVAL_GPU=2 \
EVAL_JUDGE_GPU=3 \
EVAL_PORT=8006 \
EVAL_JUDGE_PORT=8007 \
SFT_RL_EVAL_CONFIG=configs/eval/project_vision_opd_v1_offline.yaml \
SFT_RL_BENCHMARKS=mindcube,vqav2,scienceqa,mmsi_bench,blink \
SFT_RL_JUDGE_BENCHMARKS= \
SFT_RL_KEEP_GOING=1 \
bash scripts/eval/run_target_benchmarks.sh
```

For the judged missing benchmarks, use a separate run name:

```bash
EVAL_CKPT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/hf/qwen3vl_ptdpo_r4_step390 \
EVAL_RUN_NAME=project15_ptdpo_r4_step390_offline \
EVAL_SERVED_MODEL=Qwen3-VL-8B-PTDPO-R4 \
EVAL_GPU=2 \
EVAL_JUDGE_GPU=3 \
EVAL_PORT=8006 \
EVAL_JUDGE_PORT=8007 \
SFT_RL_EVAL_CONFIG=configs/eval/project_vision_opd_v1_offline.yaml \
SFT_RL_BENCHMARKS= \
SFT_RL_JUDGE_BENCHMARKS=mathvista,mmvet \
SFT_RL_KEEP_GOING=1 \
bash scripts/eval/run_target_benchmarks.sh
```

Raw data locations:

- `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/MindCube_lmmseval`
- `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/VQAv2`
- `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/ScienceQA-IMG`
- `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/MathVista`
- `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/MMSI-Bench`
- `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/BLINK`
- `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/MMVet`

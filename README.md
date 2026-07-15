# Dual-Track-OPD

Dual-Track-OPD is a long-term research engineering repository for on-policy distillation (OPD) in vision-language models, with Qwen3-VL-8B-Instruct as the first target model.

The central hypothesis is that standard token-level OPD can dilute sparse visual-dependence signals in VLM outputs. This repo is designed to separate, combine, and analyze text-side and vision-side distillation signals through token-level anchors, chunk-level anchors, and dual-track weighting.

## Why This Is Not A `verl` Fork

`verl` is planned as a pinned third-party backend, not the research repo itself. This repository owns:

- experiment configs and registries
- distillation methods and loss code
- rollout/evaluation adapters
- analysis scripts and paper assets
- reproducibility records

Use `third_party/verl/` for a future submodule or checkout and `patches/verl/` for minimal backend patches.

## Local Install

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Useful environment variables:

```bash
export DTOPD_ROOT="$PWD"
export DTOPD_DATA_ROOT="$PWD/data"
export DTOPD_MODEL_ROOT="$HOME/models"
export DTOPD_OUTPUT_ROOT="$PWD/outputs"
```

## HPC Workflow

The intended workflow is:

1. develop locally on Mac
2. push to GitHub
3. pull on an HPC CPU node
4. prepare configs, manifests, and environments
5. launch GPU jobs on H200 nodes
6. write large outputs to NFS paths outside Git
7. sync summaries, metrics, tables, and notes back into the repo

See [docs/hpc_workflow.md](docs/hpc_workflow.md).

For the contract benchmark baseline using the reproduced Vision-OPD checkpoint,
see [docs/project_benchmark_vision_opd.md](docs/project_benchmark_vision_opd.md).

## First Milestones

1. Toy OPD loss: keep CPU tests green and validate weighted KL behavior.
2. Baseline evaluation integration: add raw JSONL summaries and benchmark manifests.
3. `verl` smoke training: connect configs to a minimal Qwen3-VL rollout/training path.

## Safety

Do not commit secrets, `.env`, model weights, checkpoints, datasets, raw benchmark outputs, `outputs/`, `eval_runs/`, `wandb/`, `mlruns/`, or cache directories.

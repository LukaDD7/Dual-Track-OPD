# HPC Workflow

## Local Mac

Use the Mac for code editing, tests, configs, documentation, and lightweight smoke checks.

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## GitHub

Push code and small reproducibility metadata to GitHub. Do not push datasets, checkpoints, raw outputs, or secrets.

## HPC CPU Node

Pull the repository, create the Python environment, resolve configs, build dataset manifests, and verify paths.

```bash
bash scripts/hpc/setup_cpu_env.sh
```

## HPC GPU Node

Run training and evaluation jobs on GPU nodes. Write large outputs to `DTOPD_OUTPUT_ROOT`, preferably on NFS or scratch storage outside Git.

```bash
bash scripts/hpc/launch_gpu_job.sh configs/experiment/000_smoke_toy_opd.yaml
```

## Sync Back

Sync only summaries, registry updates, plots, paper notes, and small tables back into Git.


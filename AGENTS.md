# Agent Instructions

This repository is a research operating system for dual-track on-policy distillation in VLMs. Treat reproducibility and separation of concerns as first-class requirements.

## Working Rules

- Keep research logic in `src/dual_track_opd/`, not inside third-party backend code.
- Keep experiments configuration-driven under `configs/`.
- Prefer small, tested utilities over large framework rewrites.
- Do not commit secrets, `.env`, model weights, datasets, checkpoints, raw eval JSONL outputs, or large artifacts.
- Keep `third_party/verl/` as a placeholder/submodule area unless explicitly asked to vendor or fork.
- Store minimal patches under `patches/verl/` with clear explanations.

## Reproducibility

Every experiment should record:

- repo git commit and dirty status
- backend commit or package version
- full resolved config
- dataset manifest hash
- model checkpoint path
- raw eval output path outside Git
- summary metrics and notes

## Commands

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m dual_track_opd.eval.summarize path/to/raw.jsonl
```

## FC-OPD Training Notes

- Do not remove or bypass the online student scorer in the main FC-OPD training path. The core research claim depends on comparing teacher vs student sensitivity under different visual/text conditions to localize student capability deficits. A teacher-only router is only acceptable as an explicitly named ablation.
- For long HPC runs, a post-run GPU keepalive may be launched after training has completed and after Ray/teacher cleanup, preferably only when all requested steps finished successfully. The current requested command is:

```bash
CUDA_VISIBLE_DEVICES=3,4 KEEPALIVE_TARGET_UTIL=0.45 KEEPALIVE_WORK_ITERS=32 python -u /inspire/hdd/global_user/mengweicheng-240108120092/lzy/scripts/busy_keepalive.py
```

- A suitable integration point is near the end of `scripts/hpc/run_verl_fc_opd_overnight.sh`, after `ray stop -f` and teacher shutdown, before the final `exit ${VERL_EXIT}`. Keep it opt-in, for example behind `--keepalive-after-success` or an environment variable, and do not start it on failed or interrupted runs unless explicitly requested.

## Git Hygiene

Before committing, inspect:

```bash
git status --short
git diff --stat
```

Do not add large files. If a file looks like data, weights, raw output, or a secret, leave it out and update `.gitignore` instead.

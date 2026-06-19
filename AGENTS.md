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

## Git Hygiene

Before committing, inspect:

```bash
git status --short
git diff --stat
```

Do not add large files. If a file looks like data, weights, raw output, or a secret, leave it out and update `.gitignore` instead.


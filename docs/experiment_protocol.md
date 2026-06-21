# Experiment Protocol

Every experiment must record:

- repo git commit, branch, and dirty status
- backend name and commit or version
- full resolved config
- dataset manifest path and hash
- model checkpoint path
- raw evaluation path outside Git
- summary metrics
- notes about failures, anomalies, and hardware
- metric tier: official, community-standard, or internal diagnostic
- evaluator source, commit/version, and judge model details when applicable

The experiment registry lives at `experiments/registry.csv`. Large raw outputs should stay in `DTOPD_OUTPUT_ROOT`, not in the repository.

Paper-facing benchmark metrics must use official or community-standard evaluators. Conservative deterministic scripts in this repository are internal diagnostics unless explicitly validated against the benchmark's official evaluation path. See `docs/evaluation_plan.md`.

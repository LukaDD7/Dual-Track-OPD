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

The experiment registry lives at `experiments/registry.csv`. Large raw outputs should stay in `DTOPD_OUTPUT_ROOT`, not in the repository.


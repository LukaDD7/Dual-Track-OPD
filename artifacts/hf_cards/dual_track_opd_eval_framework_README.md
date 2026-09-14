---
license: apache-2.0
tags:
- evaluation
- vision-language-model
- lmms-eval
- dual-track-opd
---

# Dual-Track-OPD shared evaluation framework

This dataset repository packages the shared evaluation code, protocol
configuration, truncation audit, and launchers used by the
Dual-Track-OPD / FC-OPD group.

## Contents

- `configs/eval`: v1, v1-aligned, and v2 benchmark suites
- `scripts/eval`: MMF, project15, v2, judge, and truncation-audit launchers
- `eval_tasks/opd_v2`: deterministic long-CoT scoring tasks
- `docs/eval_truncation_audit_20260909.md`: measured v1 truncation artifact

## Source

Canonical source: https://github.com/LukaDD7/Dual-Track-OPD

Shared branch: `eval-shared`

## Notes

Raw benchmark outputs, caches, checkpoints, and private model weights are not
stored in this dataset repository.


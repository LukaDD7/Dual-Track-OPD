# Claude Code Notes

Read `AGENTS.md` first. Keep changes small, importable, tested, and reproducible.

Default verification:

```bash
pytest -q
```

Do not commit `.env`, datasets, raw JSONL outputs, checkpoints, model weights, caches, or experiment output directories.

## FC-OPD Mainline Guardrail

Do not speed up the main FC-OPD path by removing the online student scorer. The main algorithm relies on teacher-vs-student condition sensitivity to identify student capability deficits. Teacher-only routing is an ablation only.

## HPC Post-Run Keepalive

If asked to keep an H200 instance alive after a successful full FC-OPD run, add the keepalive as an opt-in post-success step near the end of `scripts/hpc/run_verl_fc_opd_overnight.sh`, after Ray and teacher cleanup and before `exit ${VERL_EXIT}`:

```bash
CUDA_VISIBLE_DEVICES=3,4 KEEPALIVE_TARGET_UTIL=0.45 KEEPALIVE_WORK_ITERS=32 python -u /inspire/hdd/global_user/mengweicheng-240108120092/lzy/scripts/busy_keepalive.py
```

Prefer a flag or env var such as `--keepalive-after-success`; do not start keepalive after failed or interrupted runs unless explicitly requested.

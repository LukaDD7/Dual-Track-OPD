# Pending target-benchmark evaluation handoff

Date: 2026-08-25
Checkpoint: `Vision-OPD-Qwen3.5-4B/global_step_65`
Canonical entry point: `scripts/eval/run_target_benchmarks.sh`

## Current state

The GPU instance was stopped before the six-benchmark run finished. The
authoritative state is on disk under:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/vision_opd_gs65_target_r3_nojudge
```

Completed and valid:

- `gqa`
- `dynamath`

Interrupted:

- `viewspatial` was approximately 94% complete when the instance was stopped.
  The response cache should resume most completed requests automatically.

Not yet started:

- `mmmu_pro`
- `remi`
- `mmbench`

Do **not** use:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline/vision_opd_gs65_remi_direct/replay/remi.jsonl
```

That file contains 2,600 proxy-related connection errors and is not a valid
ReMI result.

## Resume command

Use the same run name so the existing output directory and cache are reused:

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD

EVAL_RUN_NAME=vision_opd_gs65_target_r3 \
EVAL_CKPT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/third_party/Vision-OPD/checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65 \
EVAL_GPU=1 \
EVAL_JUDGE_GPU=3 \
EVAL_PORT=8020 \
EVAL_JUDGE_PORT=8021 \
bash scripts/eval/run_target_benchmarks.sh
```

The wrapper detects the existing `vision_opd_gs65_target_r3_nojudge` directory
and resumes from its cache and manifest. `gqa` and `dynamath` are already marked
completed and will be skipped.

## Notes for the next agent

- GPU 0 is not a safe default; it is often occupied by the long-running hint
  generation service.
- GPU 1 and GPU 3 were used for the interrupted run.
- If the ports are occupied again, change `EVAL_PORT` and `EVAL_JUDGE_PORT`
  together.
- A fresh console log may be written to a new `.out` file; do not infer result
  loss from an overwritten console log. Check `run_manifest.json`, result JSON,
  sample JSONL, and the response cache first.

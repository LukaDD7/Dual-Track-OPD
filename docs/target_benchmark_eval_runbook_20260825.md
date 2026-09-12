# Target benchmark evaluation runbook

This is the canonical handoff for evaluating a trained HF-format VLM checkpoint on
the six project target benchmarks:

- GQA
- DynaMath
- ViewSpatial-Bench
- MMMU-Pro
- ReMI
- MMBench

## Canonical entry point

Use [scripts/eval/run_target_benchmarks.sh](../scripts/eval/run_target_benchmarks.sh),
not an ad-hoc combination of upstream eval scripts.

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD

EVAL_CKPT=/absolute/path/to/exported/hf_checkpoint \
EVAL_RUN_NAME=my_checkpoint_target \
EVAL_GPU=1 \
EVAL_JUDGE_GPU=3 \
EVAL_PORT=8020 \
EVAL_JUDGE_PORT=8021 \
EVAL_SERVED_MODEL=My-Checkpoint \
bash scripts/eval/run_target_benchmarks.sh
```

The default checkpoint is the recovered Vision-OPD checkpoint:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/third_party/Vision-OPD/checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65
```

For a quick 8-sample-per-task smoke run:

```bash
EVAL_SMOKE=1 \
EVAL_CKPT=/absolute/path/to/exported/hf_checkpoint \
EVAL_RUN_NAME=my_checkpoint_smoke \
bash scripts/eval/run_target_benchmarks.sh
```

Always set a fresh `EVAL_RUN_NAME` for a new checkpoint. The runner creates two
directories under `$DTOPD_EVAL_ROOT`:

```text
${EVAL_RUN_NAME}_nojudge/   # GQA, DynaMath, ViewSpatial, MMMU-Pro, ReMI
${EVAL_RUN_NAME}_judged/    # MMBench with the Qwen3-VL-32B judge
```

## What the entry point runs

The wrapper delegates to
[scripts/sft_rl/run_sftrl_benchmarks.sh](../scripts/sft_rl/run_sftrl_benchmarks.sh),
which:

1. Uses the isolated `vision-opd-cu128` evaluation environment.
2. Starts the evaluated checkpoint with vLLM on `EVAL_GPU:EVAL_PORT`.
3. Starts Qwen3-VL-32B-Instruct-FP8 as the MMBench judge on
   `EVAL_JUDGE_GPU:EVAL_JUDGE_PORT`.
4. Exports the shared CUDA 12.8 toolchain for FlashInfer GDN JIT.
5. Runs the five no-judge benchmarks through
   `dual_track_opd.eval.benchmark_suite`.
6. Runs MMBench with judge scoring.
7. Writes `summary.json` and `summary.csv` in each run directory.

The benchmark registry and the `target_benchmarks` profile are in
[configs/eval/project_vision_opd.yaml](../configs/eval/project_vision_opd.yaml).

## Outputs and summaries

Default output root:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/vision_opd_project_baseline
```

After completion, read:

```text
${EVAL_RUN_NAME}_nojudge/summary.json
${EVAL_RUN_NAME}_nojudge/summary.csv
${EVAL_RUN_NAME}_judged/summary.json
${EVAL_RUN_NAME}_judged/summary.csv
```

To regenerate a summary:

```bash
PYTHONPATH=src \
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python \
  -m dual_track_opd.eval.project_summary \
  /absolute/path/to/run_directory
```

Raw outputs are authoritative even if a console log was overwritten. Do not infer
loss of results from a overwritten `.out` log; check `run_manifest.json`,
`lmms/**/results.json`, sample JSONL files, and replay JSONL files first.

## ReMI behavior

ReMI is not registered in the pinned lmms-eval release. It is replayed with
[src/dual_track_opd/eval/run_vlm_eval.py](../src/dual_track_opd/eval/run_vlm_eval.py)
using the historical prompt rows from:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/qwen3vl8b_baseline/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl
```

Images are loaded from:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/ReMI
```

ReMI is a Tier-3 internal diagnostic. Scoring uses explicit final-answer
extraction (`\boxed{}`, `<answer>`, or a final-answer line) followed by
normalized exact match in
[src/dual_track_opd/eval/score_open.py](../src/dual_track_opd/eval/score_open.py).

The replay client now:

- ignores HTTP proxies, including for `127.0.0.1`;
- writes JSONL atomically;
- supports `--resume`, preserving successful rows and retrying only failed or
  missing rows.

To run or resume ReMI alone while an existing Vision-OPD endpoint is available:

```bash
PYTHONPATH=src \
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python \
  -m dual_track_opd.eval.run_vlm_eval \
  --input-jsonl /inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/qwen3vl8b_baseline/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl \
  --output-jsonl /absolute/path/to/run/replay/remi.jsonl \
  --dataset ReMI \
  --dataset-root /inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset \
  --api-base http://127.0.0.1:8020/v1 \
  --api-key EMPTY \
  --model Vision-OPD-4B \
  --max-tokens 2048 \
  --workers 8 \
  --resume
```

## Operational cautions

- Do not use GPU 0 by default; it is often occupied by the long-running hint
  generation service.
- Pick two effectively idle GPUs. The 32B judge needs nearly a full H200 when
  using the default `gpu_memory_utilization=0.85`.
- If ports are occupied, change both `EVAL_PORT` and `EVAL_JUDGE_PORT`.
- Set a new console log variable before backgrounding another run. Reusing a
  stale `LOG` variable can overwrite an older `.out` file.
- The first Vision-OPD multimodal request can spend several minutes compiling
  FlashInfer GDN kernels. Do not kill the server during this phase.
- The benchmark suite supports `--resume-from <run_dir>` and skips benchmarks
  already marked completed in `run_manifest.json`.

## Relevant code

- `scripts/eval/run_target_benchmarks.sh`: checkpoint-independent entry point.
- `scripts/sft_rl/run_sftrl_benchmarks.sh`: server launch and benchmark sequence.
- `configs/eval/project_vision_opd.yaml`: benchmark registry and profiles.
- `src/dual_track_opd/eval/benchmark_suite.py`: command construction, resume,
  and execution.
- `src/dual_track_opd/eval/run_vlm_eval.py`: ReMI/MV-MATH OpenAI-compatible replay.
- `src/dual_track_opd/eval/score_open.py`: final-answer extraction and normalization.
- `src/dual_track_opd/eval/project_summary.py`: primary metric collection.

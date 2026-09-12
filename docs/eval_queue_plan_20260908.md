# Eval queue plan for MMF and project benchmarks (2026-09-08)

This runbook covers the three evaluation requests:

1. Validate PTD-PO r4 step 390 on the MMF paper benchmark suite.
2. Continue the interrupted base/tailSFT MMF 14-bench evaluation.
3. Run base/tailSFT/PTD-PO on the project 15-benchmark suite.

## Recommended GPU plans

### If a 4-GPU instance arrives first

Run the MMF pack first, because it has the most reusable partial results:

```bash
bash scripts/eval/launch_mmf_eval_pack.sh 0 base,tailsft,ptdpo
```

After the MMF pack finishes, run the project 15-bench pack on the same 4 GPUs:

```bash
bash scripts/eval/launch_project15_eval_pack.sh 0 base,tailsft,ptdpo 2
```

### If an 8-GPU instance arrives first

Split the node into two 4-GPU blocks:

```bash
# Shell 1: MMF 14-bench on GPUs 0-3
bash scripts/eval/launch_mmf_eval_pack.sh 0 base,tailsft,ptdpo

# Shell 2: project 15-bench on GPUs 4-7
bash scripts/eval/launch_project15_eval_pack.sh 4 base,tailsft,ptdpo 2
```

This gives the fastest overall turnaround while keeping the two suites isolated.

## What the scripts do

- `launch_mmf_eval_pack.sh`
  - Exports PTD-PO step 390 to HF if needed.
  - Starts the shared Qwen3-VL-32B judge.
  - Resumes base/tailSFT MMF runs with `--reuse`.
  - Launches the PTD-PO MMF arm.

- `launch_project15_eval_pack.sh`
  - Exports PTD-PO step 390 to HF if needed.
  - Runs all 15 project benchmarks.
  - Uses 2 GPUs per arm (model + judge).
  - Supports 2 or 3 arms in parallel.

## Benchmark coverage

### MMF 14-bench suite

MMMU_DEV_VAL, MathVista_MINI, MathVision, MathVerse_MINI, DynaMath, LogicVista,
VisuLogic, ScienceQA_VAL, RealWorldQA, MMBench_DEV_EN, MMStar, AI2D_TEST,
CharXiv_descriptive_val, CharXiv_reasoning_val.

### Project 15-bench suite

ViewSpatial, MindCube, GQA, VQAv2, ScienceQA, MV-MATH, ReMI, DynaMath,
MathVerse, MathVista, MMSI-Bench, BLINK, MMBench, MMMU-Pro, MMVet.

## Notes

- The project 15-bench suite uses the v1 contract config.
- The v2 protocol covers 4 rule-based tasks with fixed decoding/scoring; use it when you want the
  de-biased long-CoT diagnostic, but it is not the 15-bench contract suite.
- `mv_math` and `remi` are replay diagnostics in the 15-bench suite.

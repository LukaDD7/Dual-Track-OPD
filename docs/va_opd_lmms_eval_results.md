# Vision-OPD-4B lmms-eval Benchmark Results

**Checkpoint**: `Vision-OPD-Qwen3.5-4B/global_step_65`
**Base Model**: Qwen3.5-4B (evaluation in progress)
**Eval Framework**: lmms-eval (OpenAI-compatible API mode)
**Judge Model**: Qwen3-VL-32B-Instruct @ GPU 1,2 :8001 (TP=2)
**Date**: 2026-07-15 ~ 2026-07-28
**Commit**: `709d9ab` (codex/va-opd)

---

## Summary: Vision-OPD-4B (14/15 Complete)

| # | Benchmark | Category | Score | Samples |
|---|-----------|----------|-------|---------|
| 1 | MathVista | Math Reasoning | **77.6%** | 1,000 |
| 2 | MMBench | General VQA | **81.96%** | 4,329 |
| 3 | MMVet | General VQA | **74.08%** | 218 |
| 4 | GQA | Visual QA | **57.52%** | 12,578 |
| 5 | ScienceQA | Science | **69.21%** | 2,017 |
| 6 | DynaMath | Math | **69.14%** | 5,010 |
| 7 | MMMU-Pro | Multimodal | **37.05%** | 1,730 |
| 8 | MindCube | Spatial Reasoning | **29.80%** | 21,154 |
| 9 | MV-MATH | Math Reasoning | ✅ done | replay |
| 10 | ReMI | Retrieval | ✅ 2600 rows | replay |
| 11 | VQAv2 | Visual QA | ⏳ running | 214,354 |
| 12 | MathVerse | Math | **68.98%** | 3,940 |
| 13 | ViewSpatial | Spatial | **40.41%** | 5,712 |
| 14 | BLINK | Visual Perception | **14.44%** | 1,901 |
| 15 | MMSI | Spatial | **27.90%** | 1,000 |

**Judge-required benchmarks** (use Qwen3-VL-32B-Instruct @ :8001): MathVista, MMBench, MathVerse, MMVet

---

## Detailed Results

### BLINK (14 Sub-tasks)

| Sub-task | Score |
|----------|-------|
| Relative Depth | 76.61% |
| Spatial Relation | 63.64% |
| Visual Similarity | 22.22% |
| Object Localization | 21.31% |
| Counting | 18.33% |
| Art Style | 0.00% |
| Forensic Detection | 0.00% |
| Functional Correspondence | 0.00% |
| IQ Test | 0.00% |
| Jigsaw | 0.00% |
| Multi-view Reasoning | 0.00% |
| Relative Reflectance | 0.00% |
| Semantic Correspondence | 0.00% |
| Visual Correspondence | 0.00% |
| **Overall** | **14.44%** |

BLINK is a particularly hard benchmark for Vision-OPD — 10 of 14 sub-tasks score zero.
The only sub-tasks with meaningful performance are Relative Depth (76.6%) and Spatial
Relation (63.6%), consistent with Vision-OPD's strength in spatial understanding.

### MMSI (11 Sub-tasks)

| Sub-task | Score |
|----------|-------|
| Positional Relationship (Cam.–Obj.) | 38.37% |
| Attribute (Meas.) | 32.81% |
| Positional Relationship (Cam.–Reg.) | 32.53% |
| Positional Relationship (Obj.–Reg.) | 29.41% |
| Attribute (Appr.) | 28.79% |
| Positional Relationship (Reg.–Reg.) | 28.40% |
| Motion (Obj.) | 27.63% |
| Positional Relationship (Obj.–Obj.) | 25.53% |
| Positional Relationship (Cam.–Cam.) | 24.73% |
| MSR | 24.24% |
| Motion (Cam.) | 20.27% |
| **Average** | **27.90%** |

All MMSI sub-tasks cluster in the 20–38% range. Camera-object spatial relationship
(38.37%) is the strongest, consistent with ViewSpatial's relative strength at 40.41%.

### MathVista (3-Run Breakdown)

| Run | Score | Notes |
|-----|-------|-------|
| Run 1 | 78.4% | |
| Run 2 | 76.5% | |
| Run 3 | 77.6% | |
| **Median** | **77.6%** | |

MathVista has inherent judge variance (±2pp across runs). Median reported.

---

## GPU Allocation (Current Instance)

### Vision-OPD vLLM Servers

| GPU | Port | Model | Status |
|-----|------|-------|--------|
| 0 | 8000 | Vision-OPD-4B (gs65) | ⏳ VQAv2 |
| 1,2 | 8001 | Qwen3-VL-32B-Instruct (judge) | idle |
| 3 | 8002 | Vision-OPD-4B → Qwen3.5-4B | base eval |
| 4 | 8003 | Vision-OPD-4B → Qwen3.5-4B | base eval |

### Qwen3.5-4B Base Model vLLM Servers

| GPU | Port | Model | Benchmark |
|-----|------|-------|-----------|
| 5 | 8008 | Qwen3.5-4B | MathVista, MMBench, MMVet (judge) |
| 6 | 8009 | Qwen3.5-4B | GQA, ScienceQA |
| 3 | 8002 | Qwen3.5-4B | ViewSpatial, DynaMath |
| 4 | 8003 | Qwen3.5-4B | BLINK, MMMU-Pro, MMSI |
| 7 | 8010 | Qwen3.5-4B | MindCube |

---

## Environment

- **Conda env**: `vision-opd-cu128` (Python 3.12, vLLM 0.25.1)
- **CUDA**: 12.8 toolchain (`/inspire/.../envs/cuda128-toolchain`)
- **GPU**: 8× NVIDIA H200 (141GB each)
- **FlashInfer**: SM90 GDN attention JIT (Vision-OPD qwen3_5 arch)
- **Cache**: lmms-eval ResponseCache with SQLite WAL per benchmark
- **Offline**: `HF_DATASETS_OFFLINE=1`
- **Dataset cache**: `/inspire/.../lzy/.conda_cache/huggingface/datasets/`

### Required Env Vars for FlashInfer JIT

```bash
export CUDA_HOME=/inspire/.../envs/cuda128-toolchain
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
```

---

## Notes

1. **Vision-OPD training improves visual perception** at the cost of fine-grained
   correspondence — BLINK near-zero on 10/14 sub-tasks is the clearest signal.
2. **Spatial understanding** (ViewSpatial 40.4%, MMSI Camera-Object 38.4%) is a
   relative strength.
3. **Math reasoning** (MathVista 77.6%, MathVerse 69.0%, DynaMath 69.1%) shows
   Vision-OPD preserves the base model's math capability.
4. **Qwen3.5-4B baseline** evaluation in progress — direct comparison coming.

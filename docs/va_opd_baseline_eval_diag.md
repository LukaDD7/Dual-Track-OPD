# Vision-OPD Baseline Evaluation — Diagnostic Report

## Date: 2026-07-15

## Overview

Attempting to run the Vision-OPD full benchmark suite (15 benchmarks) on an 8×H200 GPU node.
Two models evaluated: Vision-OPD-Qwen3.5-4B (checkpoint global_step_65) and Qwen3.5-4B (untrained base).

The eval pipeline (`third_party/Vision-OPD/eval/run_eval.sh`):
1. `prepare_data.py` — download/prepare benchmark data from HuggingFace
2. `infer.py` — OpenAI-compatible API inference against vLLM server
3. `judge_qwenlm.py` — LLM judge (Qwen3-VL-32B-Instruct, loaded in-process via vLLM LLM class)
4. `cal_acc.py` — accuracy calculation

## Infrastructure

### Instance Isolation
- **CPU instance**: Has internet, no GPU, used for code editing and data preparation
- **GPU instance** (8×H200, 141GB each): NO internet, used for model inference
- **NFS shared**: `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/` — shared between both instances
- **NOT shared**: `/tmp`, `/root/.cache/` (local to each instance)

### Environment
- **Conda env**: `vision-opd-cu128` (Python 3.12, vLLM 0.18.0)
- **CUDA toolchain**: `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/cuda128-toolchain/bin` (CUDA 12.8, provides nvcc)
- **gcc**: Installed via conda (`gcc_linux-64 gxx_linux-64` in vision-opd-cu128), provides `$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc`
- **FlashInfer JIT cache**: `/root/.cache/flashinfer/0.6.6/90a/` (GPU-local, persists across runs)

### GPU Allocation
| GPU | Usage |
|-----|-------|
| 0 | 4B eval model vLLM serve (port 8000) |
| 7 | 32B judge model, in-process via judge_qwenlm.py |
| 1-6 | Free |

## Issues Encountered

### Issue 1: nvcc not found
- **Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'nvcc'`
- **Root cause**: FlashInfer JIT compilation needs nvcc, but `vision-opd-cu128` doesn't bundle it
- **Fix**: Add `cuda128-toolchain/bin` to PATH
- **Script snippet**:
```bash
TOOLCHAIN_BIN="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/cuda128-toolchain/bin"
if [ -d "${TOOLCHAIN_BIN}" ]; then
    export PATH="${TOOLCHAIN_BIN}:${PATH}"
fi
```

### Issue 2: Wrong conda environment
- **Mistake**: Initially used `fc-opd-verl071-cu128` (FC-OPD training env)
- **Correct env**: `vision-opd-cu128` (used by Vision-OPD upstream smoke tests)
- **Reference**: `baselines/vision_opd/scripts/run_smoke_4xh200.sh`

### Issue 3: gcc not found
- **Error**: `FileNotFoundError: [Errno 2] No such file or directory: '/usr/bin/gcc'`
- **Root cause**: GPU node's `/usr/bin/gcc` doesn't exist; initially hardcoded `/usr/bin/gcc`
- **Fix**: Install gcc via conda, use conda-specific binary name
```bash
conda install -n vision-opd-cu128 -c conda-forge gcc_linux-64 gxx_linux-64 -y
# Binary: $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc (NOT plain "gcc")
```
- **gcc version warning**: conda gcc is newer than 14, nvcc warns but accepts it (with `-allow-unsupported-compiler` implicitly). Compilation succeeds.

### Issue 4: vLLM FlashInfer gcc version mismatch
- **Warning**: `#error -- unsupported GNU version! gcc versions later than 14 are not supported!`
- **Status**: Non-fatal warning. FlashInfer kernels compiled successfully despite the warning.
- **Duration**: First run ~287s for init engine (JIT compilation). Subsequent runs faster (cache hit).

### Issue 5: Health check URL bug
- **Bug**: `wait_for_health "http://localhost:${EVAL_PORT}/v1"` → checks `/v1/health`
- **Correct URL**: vLLM health endpoint is `/health`, not `/v1/health`
- **Fix**: Changed to `wait_for_health "http://localhost:${EVAL_PORT}"`
- **Impact**: Without fix, script waits 300s then dies; vLLM server left running

### Issue 6: Old results reuse (infer.py checkpoint mechanism)
- **Bug**: `infer.py` has resume logic: if `model_answer/<bench>/<model>_answer.jsonl` already has complete records, it skips the benchmark
- **Impact**: Old results from June 25 were reused instead of running fresh inference
- **Fix**: Delete old `model_answer/` and `judge/` directories before re-running
- **Note**: This is a design issue — model_name alone identifies results, not the checkpoint step

### Issue 7: vLLM 500 on multimodal requests (UNRESOLVED)
- **Symptom**: All `/v1/chat/completions` requests return HTTP 500, then vLLM shuts down
- **Observed behavior**:
  1. vLLM starts, multimodal warmup completes successfully
  2. `infer.py` sends requests with `image_url` (base64 data URI) + text content
  3. vLLM returns 500 for the first few requests
  4. vLLM process exits ("Application shutdown complete")
  5. Remaining requests get "Connection error" (server gone)
  6. All 191 output records contain `[API_ERROR] Connection error.`
- **Model info**:
  - Architecture: `Qwen3_5ForConditionalGeneration` (extends `Qwen3VLForConditionalGeneration`)
  - `model_type`: `qwen3_5`
  - Has both `text_config` and `vision_config` — native multimodal
  - Image tokens: `<|vision_start|><|image_pad|><|vision_end|>`
  - vLLM 0.18.0 registers `qwen3_5` in multimodal registry
- **Potential causes (to investigate)**:
  1. Checkpoint missing `preprocessor_config.json` — only had `processor_config.json` (different format: nested `image_processor`/`video_processor` keys vs flat). **Fixed**: copied `preprocessor_config.json` from base model
  2. Checkpoint missing tokenizer files (`merges.txt`, `vocab.json`) — **Fixed**: copied from base model
  3. `processor_config.json` uses `Qwen2VLImageProcessor` (non-Fast) vs base's `Qwen2VLImageProcessorFast`
  4. Possible vLLM 0.18.0 bug with `qwen3_5` multimodal API serving
  5. The eval script's base64 data URI format might not be properly handled by vLLM's Qwen3.5-VL processor
- **Diagnostic script**: `scripts/eval/diag_vllm.sh` — tests health, text-only, and multimodal requests

## Model Paths

```
Vision-OPD checkpoint: third_party/Vision-OPD/checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65/
Base model:           /inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3.5-4B
Judge model:          /inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct
```

### Checkpoint file state (after fixes applied)
```
actor/
chat_template.jinja
config.json
generation_config.json
merges.txt              ← copied from base model
model.safetensors
preprocessor_config.json ← copied from base model
processor_config.json    ← original (different format)
tokenizer.json
tokenizer_config.json
video_preprocessor_config.json ← copied from base model
vocab.json              ← copied from base model
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/eval/run_vision_opd_baselines.sh` | Main all-in-one eval script |
| `scripts/eval/summarize_baselines.sh` | Post-eval results comparison table |
| `scripts/eval/diag_vllm.sh` | Quick vLLM multimodal diagnostic |
| `third_party/Vision-OPD/eval/run_eval.sh` | Upstream eval orchestrator |
| `third_party/Vision-OPD/eval/infer.py` | Inference via OpenAI API |
| `third_party/Vision-OPD/eval/judge_qwenlm.py` | LLM judge (supports api_base or local vLLM) |

## vLLM Serve Command

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "${model_path}" \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --served-model-name "${model_id}" \
    --trust-remote-code \
    --enforce-eager \
    --disable-custom-all-reduce \
    --port 8000
```

Environment: `VLLM_USE_V1=1`, `unset VLLM_ATTENTION_BACKEND`

## Next Steps for codex

1. Investigate why vLLM 0.18.0 returns 500 for multimodal requests with Qwen3.5-4B (model_type=qwen3_5)
2. Test text-only request first (should work) to isolate image processing
3. Test multimodal request with both data URI and file:// URL formats
4. Check vLLM's Qwen3.5-VL image processing pipeline — specifically how `Qwen3_5ForConditionalGeneration` handles images via the OpenAI API
5. Consider alternative: use `vllm.LLM` offline inference API directly (same approach as judge_qwenlm.py uses for the 32B model)

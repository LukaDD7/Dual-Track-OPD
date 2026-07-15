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

## Codex lmms-eval Pipeline

The new evaluation approach (codex branch `codex/va-opd`) uses `lmms-eval` v0.7.1 instead of the upstream Vision-OPD `infer.py`/`judge_qwenlm.py`.

Key files:
- `configs/eval/project_vision_opd.yaml` — benchmark suite config (15 benchmarks, 3 profiles)
- `src/dual_track_opd/eval/benchmark_suite.py` — orchestrator (plans + runs)
- `scripts/eval/start_vision_opd_server.sh` — vLLM launch (clean, no enforce-eager)
- `scripts/eval/run_project_vision_opd.sh` — smoke/all eval runner

### Issue 8: Dataset downloading on GPU node (FIXED)

- **Symptom**: `ConnectTimeout` from `huggingface_hub.hf_api.repo_info` during task init
- **Root cause**: GPU node has no internet; lmms-eval detects NFS cache as REMOTE, falls back to node-local `/tmp` which has nothing
- **Fix 1**: Pre-download datasets via CPU instance
  ```bash
  LMMS_EVAL_DATASETS_CACHE=".../lzy/.conda_cache/huggingface/datasets" \
  python -c "from datasets import load_dataset; load_dataset('...')"
  ```
- **Fix 2**: Set env vars before running eval
  ```bash
  export LMMS_EVAL_DATASETS_CACHE="/inspire/.../lzy/.conda_cache/huggingface/datasets"
  export HF_DATASETS_OFFLINE=1
  ```
  - `LMMS_EVAL_DATASETS_CACHE` bypasses lmms-eval's remote FS detection
  - `HF_DATASETS_OFFLINE=1` tells `datasets` library to skip HuggingFace Hub API calls

### Issue 9: batch_size duplicate argument (FIXED)

- **Error**: `OpenAICompatible() got multiple values for keyword argument 'batch_size'`
- **Root cause**: `benchmark_suite.py::_model_args()` included `batch_size=1` in `model_args` string AND CLI had `--batch_size 1` 
- **Fix**: Removed `batch_size` from `_model_args()` for `openai` backend. `lmms-eval` passes `--batch_size` as a separate CLI arg.

### Issue 10: vLLM 500 — gcc version mismatch (FIXED)

- **Symptom**: vLLM returns 500 on first multimodal request, then EngineCore dies
- **EngineCore error**:
  ```
  #error -- unsupported GNU version! gcc versions later than 14 are not supported!
  ninja: build stopped: subcommand failed.
  ```
- **Root cause**: FlashInfer JIT compilation of `gdn_prefill_sm90` kernels fails because conda gcc > 14; nvcc rejects it. The pre-existing `/root/.cache/flashinfer/0.6.6/90a/` cache didn't contain `gdn_prefill` kernels (these are only needed at first actual multimodal inference, not at model-load warmup time).
- **Fix**: Install gcc 14 in the conda env
  ```bash
  conda install -c conda-forge gcc=14 -y -p .../vision-opd-cu128
  ```

### Issue 11: vLLM 500 — FlashInfer linking fails: cannot find -lcuda (CURRENT)

- **Symptom**: After fixing gcc, FlashInfer JIT compiles `.cu → .cuda.o` successfully, but **linking** the `.so` fails
- **Error**:
  ```
  x86_64-conda-linux-gnu-c++ ... -L.../cuda128-toolchain/lib64 \
    -L.../cuda128-toolchain/lib64/stubs -lcudart -lcuda \
    -o .../gdn_prefill_sm90.so
  
  .../bin/ld: cannot find -lcuda: No such file or directory
  collect2: error: ld returned 1 exit status
  ```
- **Root cause**: Conda cuda128-toolchain has a **non-standard directory layout** vs what FlashInfer expects:

  | FlashInfer expects | Conda cuda128-toolchain has |
  |---|---|
  | `$CUDA_HOME/bin/nvcc` ✓ | `bin/nvcc` ✓ |
  | `$CUDA_HOME/lib64/stubs/libcuda.so` | `targets/x86_64-linux/lib/stubs/libcuda.so` |
  | `$CUDA_HOME/lib64/libcudart.so` | `targets/x86_64-linux/lib/libcudart.so` |

  - `get_cuda_path()` derives `CUDA_HOME = dirname(dirname(which nvcc))` = `cuda128-toolchain/`
  - FlashInfer hardcodes `-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib64/stubs` 
  - But `cuda128-toolchain/lib64/` **doesn't exist** — the real libs are under `targets/x86_64-linux/lib/`

- **FlashInfer's `get_cuda_path()` logic** (`flashinfer/jit/cpp_ext.py:48-64`):
  1. Read `CUDA_HOME` or `CUDA_PATH` env var → return if set
  2. Try `which nvcc` → derive CUDA_HOME from nvcc path
  3. Fall back to `/usr/local/cuda` → check if exists
- **Current state**: We set PATH to include nvcc (step 2 works), but step 2 derives the wrong layout prefix.
- **Options**:
  1. Set `CUDA_HOME` to `cuda128-toolchain/targets/x86_64-linux` — but then FlashInfer looks for `lib64/stubs/libcuda.so` not `lib/stubs/libcuda.so` (still wrong)
  2. Create symlinks in `cuda128-toolchain/`:
     ```bash
     mkdir -p cuda128-toolchain/lib64/stubs
     ln -sf ../../targets/x86_64-linux/lib/stubs/libcuda.so cuda128-toolchain/lib64/stubs/libcuda.so
     ln -sf ../../targets/x86_64-linux/lib/libcudart.so cuda128-toolchain/lib64/libcudart.so
     ```
  3. Use the **system CUDA** on the GPU node (NVIDIA driver provides `/usr/local/cuda` or similar with standard layout) — set `CUDA_HOME` to the system CUDA path
  4. Install FlashInfer from pre-built wheels (skip JIT entirely)

## Next Steps for codex

1. **Fix Issue 11** (FlashInfer `-lcuda` linker error) — create compat symlinks or use system CUDA path
2. Once vLLM multimodal inference works, run full smoke test with `--limit 8` to verify end-to-end
3. If smoke passes, download remaining datasets for `--profile all` and run full eval
4. Run baseline for Qwen3.5-4B base model (not just Vision-OPD checkpoint)
5. For judge-required benchmarks (MathVerse, MathVista, MMBench, MMVet): configure judge model endpoint

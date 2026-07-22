# VA-OPD cu132 GPU Verification Commands

GPU node: 8× H200, Driver 595.58.03, CUDA 13.2

## Phase 0: Environment Setup (run on internet-connected node first)

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export VA_OPD_ENV_PREFIX="$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu132"
export VERL_VA_OPD_DIR="$DTOPD_ROOT/fc-opd-storage/backends/verl-va-opd-e0031631"
export VA_OPD_CUDA_TOOLCHAIN="$DTOPD_ROOT/fc-opd-storage/toolchains/cuda-13.2"

cd "$DTOPD_ROOT/projects/Dual-Track-OPD"
MAX_JOBS=16 bash scripts/hpc/setup_va_opd_native_env_cu132.sh
```

If `cuda-nvcc=13.2` is not available in conda, try `cuda-nvcc` without version pin, or check:
```bash
micromamba search cuda-nvcc -c nvidia 2>/dev/null | tail -20
```

---

## Phase 1: GPU Readiness Check (run on GPU instance)

First, kill the keepalive on GPUs 4-7 (PID 150169) to free all 8 GPUs:
```bash
kill 150169
sleep 3
nvidia-smi
```

### 1a. Environment activation (REQUIRED for all subsequent phases)
```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export VA_OPD_ENV_PREFIX="$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu132"
export VERL_VA_OPD_DIR="$DTOPD_ROOT/fc-opd-storage/backends/verl-va-opd-e0031631"
export VA_OPD_CUDA_TOOLCHAIN="$DTOPD_ROOT/fc-opd-storage/toolchains/cuda-13.2"
export VA_OPD_PROJECT="$DTOPD_ROOT/projects/Dual-Track-OPD"
export PYTHONPATH="$VA_OPD_PROJECT/src:$VERL_VA_OPD_DIR:$PYTHONPATH"

# These override the cu128 defaults in run_va_opd_native.sh
export VA_OPD_STUDENT_MODEL="$DTOPD_ROOT/models/Qwen3-VL-4B-Instruct"
export VA_OPD_TEACHER_MODEL="$DTOPD_ROOT/models/Qwen3-VL-32B-Instruct"
export GEOMETRY3K_SOURCE="$DTOPD_ROOT/dataset/geometry3k/data/train-00000-of-00001.parquet"
export GEOMETRY3K_VA_OPD_DATA_DIR="$DTOPD_ROOT/fc-opd-storage/outputs/fc_opd/geometry3k_gkd"

# Verify env is readable
$VA_OPD_ENV_PREFIX/bin/python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
print(f'cuda version: {torch.version.cuda}')
print(f'nccl: {torch.cuda.nccl.version()}')
print(f'device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem/1024**3:.0f} GB)')
"
```

Expected output:
```
torch: 2.11.0+cu132
cuda available: True
cuda version: 13.2
nccl: (2, 28, ...) or (2, 29, ...)
device count: 8
GPU 0-7: NVIDIA H200 (141 GB)
```

### 1b. vLLM server smoke
```bash
# Test vllm can start and serve (uses 1 GPU)
CUDA_VISIBLE_DEVICES=0 $VA_OPD_ENV_PREFIX/bin/python -c "
from vllm import LLM
llm = LLM(
    model='$DTOPD_ROOT/models/Qwen3-VL-4B-Instruct',
    max_model_len=1024,
    gpu_memory_utilization=0.3,
    enforce_eager=True,
)
output = llm.generate(['Hello, world!'], sampling_params={'max_tokens': 10})
print(f'vLLM generation: {output[0].outputs[0].text}')
print('vLLM smoke: PASS')
"
```

---

## Phase 2: NCCL Collective Smoke (4-rank)

```bash
cd "$VA_OPD_PROJECT"

# Run the 4-rank NCCL allgather+allreduce stress test
# This exercises the exact collective sizes that caused deadlock on CUDA 12.8
CUDA_VISIBLE_DEVICES=0,1,2,3 $VA_OPD_ENV_PREFIX/bin/torchrun \
    --nproc_per_node=4 \
    scripts/hpc/smoke_va_opd_nccl.py \
    --elements 4198742,83678470 \
    --iterations 5
```

Expected: `PASS` for both element sizes (0.008 GiB and 0.156 GiB per rank).

If this hangs for >60s → NCCL deadlock STILL present on CUDA 13.2. Ctrl-C and report.

---

## Phase 3: OPD 3-step Smoke (reverse KL baseline, no VA)

```bash
cd "$VA_OPD_PROJECT"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/hpc/run_va_opd_native.sh \
    --objective opd \
    --profile smoke \
    --visible-gpus 0,1,2,3,4,5 \
    --actor-gpus 4 \
    --teacher-gpus 2 \
    --teacher-tp 2 \
    --student-model "$DTOPD_ROOT/models/Qwen3-VL-4B-Instruct" \
    --teacher-model "$DTOPD_ROOT/models/Qwen3-VL-32B-Instruct" \
    --source-data "$DTOPD_ROOT/dataset/geometry3k/data/train-00000-of-00001.parquet" \
    --name cu132_smoke
```

This runs 3 training steps (batch=4, K=4). Key checks in output:
- `distillation/abs_loss` is finite
- `actor/entropy` is > 0 and < 5
- `response_length/clip_ratio` < 1
- No NCCL timeout errors
- Step 3 completes with gradient update

---

## Phase 4: VA-OPD 3-step Smoke

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/hpc/run_va_opd_native.sh \
    --objective va_opd \
    --profile smoke \
    --visible-gpus 0,1,2,3,4,5 \
    --actor-gpus 4 \
    --teacher-gpus 2 \
    --teacher-tp 2 \
    --student-model "$DTOPD_ROOT/models/Qwen3-VL-4B-Instruct" \
    --teacher-model "$DTOPD_ROOT/models/Qwen3-VL-32B-Instruct" \
    --source-data "$DTOPD_ROOT/dataset/geometry3k/data/train-00000-of-00001.parquet" \
    --name cu132_smoke
```

Additional VA-specific metrics in output:
- `va_opd/mean` > 0 (not identically zero)
- `va_opd/positive_ratio` between 0.1 and 0.9 (not degenerate)
- `va_opd/group_weight_sum_max_error` ≤ 1e-5

---

## Phase 5: Full Run (only if Phase 3-4 pass)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/hpc/run_va_opd_native.sh \
    --objective va_opd \
    --profile train \
    --visible-gpus 0,1,2,3,4,5 \
    --actor-gpus 4 \
    --teacher-gpus 2 \
    --teacher-tp 2 \
    --student-model "$DTOPD_ROOT/models/Qwen3-VL-4B-Instruct" \
    --teacher-model "$DTOPD_ROOT/models/Qwen3-VL-32B-Instruct" \
    --source-data "$DTOPD_ROOT/dataset/geometry3k/data/train-00000-of-00001.parquet" \
    --name cu132_full
```

---

## Risk Summary

| # | Risk | Level | Verification |
|---|------|-------|-------------|
| 1 | cuda-nvcc=13.2 not in conda | 🟡 | Phase 0 — fallback: skip flash-attn or manual CUDA install |
| 2 | NCCL 2.27.x in torch 2.11 still deadlocks | 🟡 | Phase 2 — if hang, try torch 2.13.0+cu132 (see fallback below) |
| 3 | vllm 0.25.1 Python API breaks verl imports | 🟢 | Phase 1b — verified by vLLM import + generation test |
| 4 | flashinfer JIT compilation on CUDA 13.2 | 🟡 | Phase 3 — first run compiles kernels (~5 min), subsequent fast |
| 5 | flash-attn build fails on CUDA 13.2 | 🟡 | Phase 0 — skip with VA_OPD_SKIP_FLASH_ATTN_BUILD=1 if needed |
| 6 | vllm 0.25 + torch 2.11 combined instability | 🟢 | Phase 1b/3 — both vllm and verl basic ops verified |

### Fallback: torch 2.13.0+cu132 (if NCCL deadlock with torch 2.11)

```bash
# Install torch 2.13.0 cu132 OVER the existing env
$VA_OPD_ENV_PREFIX/bin/pip install \
    --index-url https://download.pytorch.org/whl/cu132 \
    torch==2.13.0 torchvision==0.28.0 \
    --force-reinstall --no-deps

# Reinstall vllm (it will see torch 2.13 already, skip its torch==2.11 pin)
$VA_OPD_ENV_PREFIX/bin/pip install --no-deps --force-reinstall vllm==0.25.1

# Verify
$VA_OPD_ENV_PREFIX/bin/python -c "
import torch
print(f'torch: {torch.__version__}, cuda: {torch.version.cuda}, nccl: {torch.cuda.nccl.version()}')
"
# Then re-run Phase 2 NCCL smoke
```

# VA-OPD GKD Environment Audit — 2026-07-09

## 1. Current Bug Judgment

The legacy PPO/FSDP line has been exhaustively tested:

| Test | Config | Result | Conclusion |
|------|--------|--------|------------|
| 4-rank power-of-2 | fsdp_size=4, Trees | Step 27 silent hang | Allreduce deadlock, not allgather |
| fsdp_size=1 (replicate) | No FSDP sharding | Step 25 silent hang | **Confirmed: gradient allreduce, not FSDP allgather** |
| NCCL_ALGO=Ring | 4-rank, Ring algo | Step 56+ alive | Ring avoids Trees deadlock path |

**Root cause**: NCCL 2.27.3 + CUDA 12.8 + 4×H200 has a probabilistic collective deadlock in the Trees algorithm path that affects both allgather and allreduce at gradient synchronization. The Ring algorithm works around it but is not a guaranteed permanent fix.

**Strategic decision**: Rather than further patching the legacy PPO/FSDP stack, migrate to the maintained GKD/Megatron OPD pipeline which uses a different collective pattern (Megatron TP/PP/DP instead of FSDP2) and has active upstream maintenance.

## 2. High-Risk Version Mismatches

### a. vLLM default wheel may be CUDA 12.9

vLLM publishes wheels compiled against specific CUDA versions. A `cu129` wheel loaded under driver 570.x (CUDA 12.8 max) will fail at `cudaLaunchKernel` with `cudaErrorInvalidDeviceFunction` or silently corrupt NCCL communicators.

**Mitigation**: Pin vLLM 0.11.0 with explicit `cu128` / `torch2.8` variant.

### b. Driver 570.x only supports CUDA ≤ 12.8

```text
Driver: 570.x  →  CUDA 12.8 max
```

Any wheel compiled for `cu129` or `cu130` will fail. The CUDA forward-compat guarantee is one minor version; 12.9 and 13.0 are beyond 12.8.

**Mitigation**: Audit all installed wheels for `cu129` / `cu130` strings before any training launch.

### c. conda PyTorch NCCL vs vLLM NCCL conflict

conda PyTorch statically links its own NCCL build. vLLM may ship a different NCCL version. If both are loaded into the same process (e.g., Ray workers that import both torch and vllm), symbol conflicts can corrupt NCCL communicators or cause silent hangs.

**Mitigation**: Isolated conda env with unified cu128 wheel chain; verify with `pip freeze | grep nccl`.

### d. verl 0.7.1 is NOT the GKD recipe pin

Current `third_party/verl` is at `bec9ef74` (verl 0.7.1). The GKD recipe is developed and tested against a specific commit (`bcb638649a50e58494a8ddd92085ad1174f674b8`). API surface changes between 0.7.1 and the GKD pin can cause import errors, config schema mismatches, or runtime crashes.

**Mitigation**: Isolated checkout at `external/verl_gkd/verl` — do NOT modify `third_party/verl`.

## 3. GKD Recommended Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.12 | Required by torch 2.8 |
| torch | 2.8.0+cu128 | From `--index-url https://download.pytorch.org/whl/cu128` |
| torchvision | 0.23.0+cu128 | Same index |
| torchaudio | 2.8.0+cu128 | Same index |
| verl | commit `bcb638649a50e58494a8ddd92085ad1174f674b8` | GKD recipe pin, NOT verl 0.7.1 |
| recipe submodule | commit `ba246418f4de12b845a09bba975f1a5242adc898` | `git submodule update --init --recursive recipe` |
| vLLM | 0.11.0 | cu128 / torch2.8 variant only |
| flash-attn | 2.8.1+cu12torch2.8 | Match CUDA 12.x + torch 2.8 |
| flashinfer-python | 0.3.1 | Required by vLLM 0.11.0 |
| TransformerEngine | v2.6 | Megatron backend requirement |
| Megatron-LM | core_v0.13.1 | GKD Megatron workers requirement |
| nvidia-cudnn-cu12 | 9.10.2.21 | Explicit pin to avoid conda/pip resolver conflicts |

## 4. Conda Environment Layout

```
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/miniconda3/envs/
├── fc-opd-verl071-cu128     ← LEGACY: do NOT touch
└── vaopd-gkd-cu128          ← NEW: GKD isolated env
```

## 5. Repository Layout

```
Dual-Track-OPD/
├── third_party/verl/          ← LEGACY (verl 0.7.1): do NOT touch
└── external/verl_gkd/         ← NEW: GKD pinned verl checkout
    └── verl/                  ← commit bcb638649a50e58494a8ddd92085ad1174f674b8
        └── recipe/gkd/        ← submodule ba246418f4de12b845a09bba975f1a5242adc898
```

## 6. Audit Checklist

Before any training:

- [ ] `nvidia-smi` shows driver 570.x
- [ ] `nvcc --version` (if present) shows CUDA ≤ 12.8
- [ ] `python -c "import torch; print(torch.version.cuda)"` prints `12.8`
- [ ] `pip freeze | grep -E 'torch|vllm|ray|flash|nccl|nvidia-cudnn|transformer-engine|megatron'` — no `cu129` or `cu130` anywhere
- [ ] `pip show torch vllm ray nvidia-nccl-cu12 nvidia-cudnn-cu12` — all cu128 path
- [ ] nccl-tests `all_reduce_perf` passes on 4 GPUs
- [ ] GKD text smoke 10 steps passes
- [ ] No `third_party/verl` modifications
- [ ] `external/verl_gkd/` is at correct verl + recipe commits

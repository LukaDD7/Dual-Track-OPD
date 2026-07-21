# VA-OPD Build Blocker — 2026-07-21

## Status: BLOCKED

vllm==0.18.0 + transformers==5.5.0 不可共存。

## Error

```
ERROR: Cannot install vllm==0.18.0 because these package versions have conflicting dependencies.
The conflict is caused by:
    vllm 0.18.0 depends on transformers<5 and >=4.56.0
    The user requested (constraint) transformers==5.5.0
```

来源：`configs/environment/verl_va_opd_e003_cu128.constraints.txt` (commit e315262)

## Reproduce

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
unset CC CXX CPP CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LDFLAGS_LD
unset CMAKE_ARGS CMAKE_PREFIX_PATH
unset CUDA_HOME CUDA_PATH NVCC
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export VA_OPD_ENV_PREFIX="$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu128"
export VERL_VA_OPD_DIR="$DTOPD_ROOT/fc-opd-storage/backends/verl-va-opd-e0031631"
export VA_OPD_CUDA_TOOLCHAIN="$DTOPD_ROOT/fc-opd-storage/toolchains/cuda-12.8"
MAX_JOBS=16 bash scripts/hpc/setup_va_opd_native_env.sh
```

## Completed before failure

- [x] Python 3.12 conda prefix created
- [x] verl backend cloned at exact commit e003163181731412595257a72ec173071efb125f
- [x] 3-file patch applied and reverse-checked
- [ ] vLLM install — FAILED

## GPU facts already collected

- 8× H200 (141 GB each), driver 570.124.06, CUDA 12.8
- NV18 full mesh topology
- GPUs 0-3 idle, GPUs 4-7 running keepalive (PID 974678, owned by user)
- Shared storage: 240 GB free
- `NVIDIA_VISIBLE_DEVICES` has 8 GPU UUIDs

## Fix needed

Resolve `vllm==0.18.0` ↔ `transformers==5.5.0` conflict in constraints file.

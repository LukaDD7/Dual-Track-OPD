# VA-OPD Build Blocker — 2026-07-21

> 2026-07-23 update: this file records the first cu128 recovery attempt. New
> environments must use the unified `$DTOPD_ROOT/conda-envs/` layout and the
> cu128-on-R595 candidate defined in `docs/environment_registry.md`; do not
> reuse or move the historical prefixes below.

## Status: FIX PREPARED — CPU source build and GPU runtime gates still required

原环境矩阵不可重建，且不止一个问题：

1. `vllm==0.18.0` 要求 `transformers>=4.56,<5`，与 `transformers==5.5.0` 直接冲突；
2. exact verl commit 的 `setup.py` 声明 `vllm>=0.8.5,<=0.12.0`，所以 0.18.0 超出 backend 支持范围；
3. 目标 H200 节点是 driver 570 / CUDA 12.8。vLLM 新版发布 wheel 可能含 CUDA 12.9/13 扩展，import 成功也不能证明 kernel 可执行；
4. 失败前创建的旧 prefix 不是干净、可审计的交付环境。

## Error

```
ERROR: Cannot install vllm==0.18.0 because these package versions have conflicting dependencies.
The conflict is caused by:
    vllm 0.18.0 depends on transformers<5 and >=4.56.0
    The user requested (constraint) transformers==5.5.0
```

来源：`configs/environment/verl_va_opd_e003_cu128.constraints.txt`（commit `e315262` 中的旧版本）。

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

## 采用的修复

不对旧 prefix 做手工修补，也不强行 `--no-deps` 隐藏冲突。新矩阵是：

| component | fixed version | reason |
|---|---:|---|
| verl | `e003163181731412595257a72ec173071efb125f` + repo patch | native distillation API target |
| vLLM | upstream `0.12.0`, source commit `4fd9d6a...`; installed version `0.12.0+cu128` | exact verl 声明范围的最新版本；local suffix 证明非默认 wheel |
| torch family | `2.9.0/0.24.0/2.9.0` | vLLM 0.12.0 的 exact metadata；从官方 cu128 index 安装 |
| Transformers | `4.57.3` | 满足 vLLM `<5,>=4.56`，包含 Qwen3-VL，并与项目 teacher extra 一致 |
| FlashInfer | `0.5.3` | vLLM 0.12.0 exact metadata |
| CUDA build | conda `cuda-toolkit=12.8`, GCC/G++ 12, H200 `SM90` | 不读取 `/usr` CUDA；匹配 driver 570 |

新 prefix 是：

```text
$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu128-v2
```

旧 prefix 保留作失败证据，但不再用于 VA-OPD。setup 会从 vLLM tag 的固定 commit 在 CPU 实例构建 cu128 wheel，将 wheel SHA-256 和 source commit 写入 environment manifest，最后运行统一 resolver、`pip check` 和 native VA loss import gate。

## 下一步验证

CPU CC 按 `docs/va_opd_hpc_runbook.md` 第 4–7 节执行。只有下面四项都通过，状态才能从“fix prepared”改成“environment verified”：

- vLLM 0.12.0 source wheel 构建成功；
- flash-attn 2.8.3 在 SM90/cu128 toolchain 下构建成功；
- `pip check` 无冲突，environment manifest 完整；
- GPU 节点完成真实 kernel/model smoke；仅 `import vllm` 不算通过。

# VA-OPD cu132 环境构建 — 状态与阻塞点

> **最终判定（2026-07-23）：QUARANTINED。** 本文记录失败事实，不是可执行安装指南。
> 不要继续修补或激活该 prefix，不要执行旧 cu132 setup。当前主线是在 R595
> 节点上使用受支持的 cu128 用户态矩阵；统一目录和完整迁移步骤见
> `docs/environment_registry.md`。

**日期**: 2026-07-22
**分支**: `codex/va-opd` (commit `affb6e1`)
**目标**: 在 8×H200 + Driver 595.58.03 (CUDA 13.2) 上复现 VA-OPD

---

## 1. 已完成：自包含 cu132 环境 ✅

### 环境路径

```
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/envs/va-opd-verl-e003-cu132
```

### 版本矩阵

| 组件 | 版本 | 来源 |
|------|------|------|
| Python | 3.12.13 | conda |
| torch | 2.13.0+cu132 | `https://download.pytorch.org/whl/cu132` |
| vllm | 0.25.1 | PyPI (`cp38-abi3-manylinux_2_28_x86_64`) |
| transformers | 5.14.1 | PyPI |
| flashinfer-python | 0.6.13 | PyPI |
| flash-attn | 2.8.3 | 源码编译 (conda CUDA 13.2) |
| NCCL | 2.29.7 | torch 内置 |
| tensordict | 0.10.0 | PyPI |
| ray | 2.53.0 | PyPI |

### CUDA Toolchain（全部在 conda env 内，无 /usr/ 依赖）

```bash
conda install -p $ENV_PREFIX -c nvidia -c conda-forge \
    cuda-nvcc=13.2 cuda-cudart-dev=13.2 cuda-cccl=13.2 \
    libcublas-dev cuda-nvrtc cuda-version=13.2
```

- nvcc: 13.2.86
- libcudart.so, libcublas.so 等全部在 `$ENV_PREFIX/lib/`

### 安装策略（关键）

vllm 0.25.1 元数据硬锁 `torch==2.11.0`，但 cu132 要求 torch >= 2.12。安装顺序：

1. `pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132`
2. `pip install vllm==0.25.1`（拉下 torch 2.11.0 和旧 nvidia-* 包）
3. `pip install --force-reinstall torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132`（覆盖回 torch 2.13 + 新版 nvidia-* 包）
4. `pip uninstall torchaudio`（cu132 无 torchaudio 2.13 wheel，VA-OPD 不需要音频）

---

## 2. NCCL Smoke Test — PASS ✅

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/hpc/smoke_va_opd_nccl.py --elements 4198742,83678470 --iterations 5
```

**结果**: 4-rank allgather + allreduce 压力测试通过，无死锁。
对比旧环境 (CUDA 12.8 + NCCL 2.27.3 + Driver 570.x) 的概率性 deadlock 已解决。

---

## 3. GPU 实例信息

```
nvidia-smi: Driver 595.58.03, CUDA 13.2
GPU: 8× NVIDIA H200 (141 GB), NV18 full mesh topology
GPUs 0-3: 空闲 (用于 4-rank FSDP2 actor)
GPUs 4-5: 空闲 (用于 2-GPU TP=2 teacher)
GPUs 6-7: 空闲
```

### libcuda 路径（非标准）

```bash
$ ldconfig -p | grep libcuda.so
libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
libcuda.so (libc6,x86-64)   => /lib/x86_64-linux-gnu/libcuda.so
```

**注意**: 不在标准 `/usr/lib64/`，在 `/lib/x86_64-linux-gnu/`。本次已在 launcher 中加入此路径。

### 网络状态

GPU 节点**无外网访问**。所有依赖必须在有网的 CPU 节点提前构建到共享存储 (NFS)。
环境构建脚本: `scripts/hpc/setup_va_opd_native_env_cu132.sh`

---

## 4. 核心阻塞：vllm 0.25.1 与 torch 2.13 C++ ABI 不兼容 🔴

### 诊断过程

**现象**: Ray worker 中 `from vllm.lora.lora_model import LoRAModel` 触发 vllm 导入链，最终 `vllm.vllm_flash_attn._vllm_fa2_C.abi3.so` 加载失败。

**LD_LIBRARY_PATH 修复历程**:
1. 首次报错 → 确认 launcher 需要设置 `LD_LIBRARY_PATH`（已加）
2. 第二次报错 → 确认 `libcuda.so.1` 在 `/lib/x86_64-linux-gnu/`（已加）
3. 第三次报错 → 确认 Ray `runtime_env.env_vars` 需要显式转发 `LD_LIBRARY_PATH`（修改 `verl/trainer/constants_ppo.py`）
4. 第四次仍报错 → ctypes 直接加载测试

**根因确认** (ctypes 直接加载):

```python
ctypes.CDLL('.../vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so')
# 错误: undefined symbol: _ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v
```

**分析**:
- `_vllm_fa2_C.abi3.so` 是 vllm 0.25.1 wheel 内的**预编译** C++ 扩展
- 编译时链接的是 torch **2.11.0** 的 libtorch
- torch 2.11 → 2.13 之间 C++ 模板符号发生了变化
- 该 `.so` 的 NEEDED 列表: `libtorch.so`, `libcudart.so.13`, `libcuda.so.1`, `libtorch_cpu.so`, `libtorch_cuda.so`, `libc10_cuda.so`, `libc10.so`
- **无 RPATH/RUNPATH**，完全依赖 `LD_LIBRARY_PATH`

**关键矛盾**:
- cu132 (CUDA 13.2) → torch 最低版本 2.12.0（PyTorch 2.11 不支持 CUDA 13.2）
- vllm 0.25.1 → torch 硬锁 2.11.0（且编译的 `.so` 文件需要 torch 2.11 ABI）
- torch 2.12 → 可能仍有相同 ABI 问题（模板符号变化从 2.12 开始）
- vllm 0.25.1 是 PyPI 最新版，没有更高版本

**这个版本的 vllm 没有源码级的 torch 版本约束，但有二进制级的 libtorch ABI 约束。**

### tested-but-failed 的 workarounds

| 尝试 | 结果 |
|------|------|
| torch 2.13 + LD_LIBRARY_PATH 完整 | ❌ ABI mismatch |
| torch 2.13 + Ray runtime_env LD_LIBRARY_PATH | ❌ 同上 |
| vllm.lora.models → lora_model 重定向 | ✅ 导入通过，但后续 FA2 仍崩 |
| torchaudio 移除 | ✅ 消除 CUDA 版本检查 |

---

## 5. 已修改的文件清单

### 本项目 (commit `affb6e1`, branch `codex/va-opd`)

| 文件 | 修改内容 |
|------|---------|
| `configs/environment/verl_va_opd_e003_cu132.constraints.txt` | **新增** — cu132 版本锁 |
| `scripts/hpc/setup_va_opd_native_env_cu132.sh` | **新增** — cu132 自包含环境构建 |
| `docs/va_opd_cu132_gpu_verify.md` | **新增** — 五阶段 GPU 验证手册 |
| `scripts/hpc/run_va_opd_native.sh` | LD_LIBRARY_PATH + `/lib/x86_64-linux-gnu` |
| `scripts/hpc/preflight_va_opd_native.py` | cu128/cu132 双运行时 + git porcelain 修复 + vllm_wheel 可选 + 环境 manifest 校验 |
| `scripts/setup/prepare_va_opd_native_verl.sh` | `VERL_BACKEND_NO_FETCH` 离线 GPU 节点支持 |
| `src/dual_track_opd/fc_opd/verl_dataset.py` | `maybe_filter_out_long_prompts` 重写 (多进程兼容) |

### Verl backend (`verl-va-opd-e0031631`, 独立 repo)

| 文件 | 修改内容 |
|------|---------|
| `verl/utils/vllm/utils.py` | LoRA 导入: `vllm.lora.models` → `vllm.lora.lora_model` |
| `verl/trainer/constants_ppo.py` | Ray runtime_env 转发 `LD_LIBRARY_PATH` |

### 原始 3-file VA-OPD patch（来自 `patches/verl/va_opd_native_e0031631.patch`）

| 文件 | 修改内容 |
|------|---------|
| `verl/experimental/agent_loop/agent_loop.py` | 第二次 degraded-image teacher pass |
| `verl/trainer/distillation/losses.py` | `register_native_verl_loss(va_opd_k1)` |
| `verl/trainer/ppo/ray_trainer.py` | `prepare_native_verl_batch` 调用点 |

---

## 6. 环境 manifest

```
$ENV_PREFIX/share/dual-track-opd/va_opd_environment_manifest.json
```

内容:
```json
{
  "build_kind": "cpu-source-build-cu132-h200-sm90",
  "vllm_source_commit": "4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e",
  "torch_cuda": "13.2",
  "torch_cuda_arch_list": "9.0",
  "verl_backend_commit": "e003163181731412595257a72ec173071efb125f",
  "constraints_file": "configs/environment/verl_va_opd_e003_cu132.constraints.txt"
}
```

上面是失败环境当时写出的 manifest，不是可信 provenance。其中
`4fd9d6a85c...` 实际对应 vLLM **0.12.0** tag；vLLM 0.25.1 tag 是
`752a3a5044...`。因此该 manifest 除了记录“当时声称的状态”之外，不得用于重建。

---

## 7. 历史运行命令（禁止执行）

以下命令仅保留用于解释事故。该 prefix 已 quarantined。

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export VA_OPD_ENV_PREFIX="$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu132"
export VA_OPD_STUDENT_MODEL="$DTOPD_ROOT/models/Qwen3-VL-4B-Instruct"
export VA_OPD_TEACHER_MODEL="$DTOPD_ROOT/models/Qwen3-VL-32B-Instruct"
export GEOMETRY3K_SOURCE="$DTOPD_ROOT/dataset/geometry3k/data/train-00000-of-00001.parquet"
export GEOMETRY3K_VA_OPD_DATA_DIR="$DTOPD_ROOT/fc-opd-storage/outputs/fc_opd/geometry3k_gkd"

cd "$DTOPD_ROOT/projects/Dual-Track-OPD"
VERL_BACKEND_NO_FETCH=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
    bash scripts/hpc/run_va_opd_native.sh \
    --objective opd \
    --profile smoke \
    --visible-gpus 0,1,2,3,4,5 \
    --actor-gpus 4 \
    --teacher-gpus 2 \
    --teacher-tp 2 \
    --name cu132_smoke
```

---

## 8. 最终推进路径

不采用此前列出的 cu132 workaround：

- attention backend 在 native extension 导入失败之后才有意义，不能修 ABI；
- 源码编 vLLM 0.25.1 即使解决 torch ABI，仍越过 verl `<=0.12.0` 的 API 边界；
- nightly 不满足固定版本和可复现要求；
- torch 2.11/cu130 + vLLM 0.25.1 仍不属于当前 verl backend 的支持矩阵。

当前主线是在 R595/H200 节点运行 verl e003 + vLLM 0.12 + torch
2.9/cu128，并使用全新的统一 prefix：

```text
$DTOPD_ROOT/conda-envs/va-opd-native-e003-cu128-r595-v1
```

先运行同样的 4-rank NCCL smoke。只有它在新驱动上仍失败时，才建立单独的
cu128 + newer-NCCL candidate，保持 torch/vLLM/Transformers 不变，以隔离变量。
完整命令、环境台账和升降级规则见 `docs/environment_registry.md`。

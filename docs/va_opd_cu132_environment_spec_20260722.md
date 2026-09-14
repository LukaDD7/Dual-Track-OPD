# VA-OPD cu132 环境规格说明书

> **历史事故交接文档，状态为 QUARANTINED。** 下面的版本和路径用于解释
> 2026-07-21 的失败环境，不代表当前推荐配置。不得执行本文旧安装步骤。
> 当前统一目录、环境台账和 cu128-on-R595 恢复步骤见
> `docs/environment_registry.md`。

**日期**: 2026-07-22
**GPU 节点**: 8× H200, Driver 595.58.03, CUDA 13.2
**CPU 节点**: 有外网，conda 可用
**共享存储**: NFS，路径 `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/`
**项目分支**: `codex/va-opd` @ `d9cceb8`

---

## 1. 目录布局

```
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/
│
├── projects/Dual-Track-OPD/          ← 主仓库 (git: LukaDD7/Dual-Track-OPD, branch codex/va-opd)
│   ├── src/dual_track_opd/
│   │   ├── fc_opd/                   ← FC-OPD 核心 (va_opd_loss.py, verl_dataset.py, ...)
│   │   └── va_opd/                   ← VA-OPD 纯研究逻辑 (native_verl.py, objective.py)
│   ├── configs/environment/
│   │   ├── verl_va_opd_e003_cu128.constraints.txt   ← cu128 版本锁 (旧)
│   │   └── verl_va_opd_e003_cu132.constraints.txt   ← cu132 版本锁 (新，本次创建)
│   ├── scripts/hpc/
│   │   ├── run_va_opd_native.sh      ← 主启动脚本
│   │   ├── setup_va_opd_native_env_cu132.sh ← cu132 环境构建
│   │   ├── preflight_va_opd_native.py ← 预检脚本
│   │   └── smoke_va_opd_nccl.py      ← NCCL 4-rank 压力测试
│   ├── scripts/setup/
│   │   └── prepare_va_opd_native_verl.sh ← verl backend 准备
│   ├── patches/verl/
│   │   └── va_opd_native_e0031631.patch ← 原始 3-file VA-OPD patch
│   └── docs/
│       ├── va_opd_cu132_status_20260722.md ← 状态总览
│       ├── va_opd_cu132_environment_spec_20260722.md ← 本文档
│       └── va_opd_cu132_gpu_verify.md ← GPU 验证命令
│
├── fc-opd-storage/
│   ├── envs/
│   │   ├── va-opd-verl-e003-cu128/   ← cu128 旧环境 (不要动)
│   │   └── va-opd-verl-e003-cu132/   ← ★ cu132 新环境 (本次构建)
│   ├── backends/
│   │   └── verl-va-opd-e0031631/     ← ★ verl backend (e0031631 + patches)
│   ├── outputs/fc_opd/geometry3k_gkd/
│   │   ├── train.parquet             ← 训练集 (1901 条)
│   │   └── val.parquet               ← 验证集 (200 条)
│   ├── runs/va_opd_native/           ← 运行输出
│   ├── checkpoints/va_opd_native/    ← checkpoint
│   └── toolchains/                   ← (空，cu128 用；cu132 不再用此目录)
│
├── models/
│   ├── Qwen3-VL-4B-Instruct/         ← Student
│   └── Qwen3-VL-32B-Instruct/        ← Teacher
│
├── dataset/geometry3k/data/
│   └── train-00000-of-00001.parquet  ← 原始数据
│
└── scripts/
    └── busy_keepalive.py             ← HPC keepalive 脚本
```

---

## 2. cu132 环境详细配置

### 2.1 环境路径

```bash
ENV_PREFIX=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/envs/va-opd-verl-e003-cu132
```

### 2.2 Python & 基础

| 项目 | 值 |
|------|-----|
| Python | 3.12.13 (conda) |
| pip | 25.1.1 |
| conda tool | `conda` (miniconda3, `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/miniconda3/condabin/conda`) |

### 2.3 CUDA Toolchain（全部在 conda env 内，无 /usr/ 依赖）

```bash
# 安装命令：
conda install -y -p "$ENV_PREFIX" -c nvidia -c conda-forge \
    cuda-nvcc=13.2 \
    cuda-cudart-dev=13.2 \
    cuda-cccl=13.2 \
    libcublas-dev \
    cuda-nvrtc \
    cuda-version=13.2
```

| 组件 | 版本 | 路径 |
|------|------|------|
| nvcc | 13.2.86 | `$ENV_PREFIX/bin/nvcc` |
| libcudart.so.13 | 13.2.86 | `$ENV_PREFIX/lib/libcudart.so.13` |
| libcublas.so.13 | 13.4.1.3 | `$ENV_PREFIX/lib/libcublas.so.13` |
| cuda_runtime.h | - | `$ENV_PREFIX/targets/x86_64-linux/include/cuda_runtime.h` |

### 2.4 运行时包（pip 安装）

| 包 | 版本 | 安装源 |
|---|------|--------|
| torch | 2.13.0+cu132 | `https://download.pytorch.org/whl/cu132` |
| torchvision | 0.28.0+cu132 | 同上 |
| vllm | 0.25.1 | PyPI (`cp38-abi3-manylinux_2_28_x86_64`) |
| transformers | 5.14.1 | PyPI |
| flashinfer-python | 0.6.13 | PyPI |
| flashinfer-cubin | 0.6.13 | PyPI |
| flash-attn | 2.8.3 | 源码编译 (SM90 only, TORCH_CUDA_ARCH_LIST=9.0) |
| tensordict | 0.10.0 | PyPI |
| ray | 2.53.0 | PyPI |
| datasets | 4.4.2 | PyPI |
| accelerate | 1.12.0 | PyPI |
| peft | 0.18.0 | PyPI |
| qwen-vl-utils | 0.0.14 | PyPI |

### 2.5 PyTorch 自带的 nvidia-* 包（由 torch 2.13.0+cu132 提供）

| 包 | 版本 |
|---|------|
| nvidia-cublas | 13.4.0.1 |
| nvidia-cuda-cupti | 13.2.75 |
| nvidia-cuda-nvrtc | 13.2.78 |
| nvidia-cuda-runtime | 13.2.75 |
| nvidia-cudnn-cu13 | 9.20.0.48 |
| nvidia-cufft | 12.2.0.46 |
| nvidia-curand | 10.4.2.55 |
| nvidia-cusolver | 12.2.0.1 |
| nvidia-cusparse | 12.7.10.1 |
| nvidia-cusparselt-cu13 | 0.8.1 |
| nvidia-nccl-cu13 | **2.29.7** |
| nvidia-nvjitlink | 13.2.78 |
| nvidia-nvtx | 13.2.75 |
| triton | 3.7.1 |

### 2.6 torchaudio

**已移除**。vllm 安装时拉入 `torchaudio==2.11.0+cu130`，与 torch 2.13.0+cu132 的 CUDA 版本检查冲突。
VA-OPD 不需要音频处理，故卸载。

### 2.7 环境 manifest

```
$ENV_PREFIX/share/dual-track-opd/va_opd_environment_manifest.json
```

```json
{
  "build_kind": "cpu-source-build-cu132-h200-sm90",
  "vllm_source_commit": "4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e",
  "torch_cuda": "13.2",
  "torch_cuda_arch_list": "9.0",
  "verl_backend_commit": "e003163181731412595257a72ec173071efb125f",
  "constraints_file": "configs/environment/verl_va_opd_e003_cu132.constraints.txt",
  "constraints_sha256": "2e4a06fe02732502093d2a51dcfb61d886616b80536f39d293bafeaa23c51da3",
  "build_date": "2026-07-21",
  "python": "3.12.13",
  "packages": {
    "torch": "2.13.0+cu132",
    "vllm": "0.25.1",
    "transformers": "5.14.1",
    "tensordict": "0.10.0",
    "ray": "2.53.0"
  }
}
```

### 2.8 版本约束文件

路径: `configs/environment/verl_va_opd_e003_cu132.constraints.txt`

```ini
# 策略: torch 2.13.0+cu132 先装, vllm 0.25.1 正常安装(会降级 torch 到 2.11),
# 然后 torch 2.13.0+cu132 强制重装覆盖。flash-attn 从源码编译。
torch==2.13.0
torchvision==0.28.0
vllm==0.25.1
transformers==5.14.1
tensordict==0.10.0
flashinfer-python==0.6.13
flashinfer-cubin==0.6.13
numpy>=2.0.0
pyarrow>=19.0.0
pandas
ray[default]==2.53.0
datasets==4.4.2
accelerate==1.12.0
peft==0.18.0
qwen-vl-utils==0.0.14
pillow
```

---

## 3. Verl Backend 配置

### 3.1 路径

```bash
VERL_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/backends/verl-va-opd-e0031631
```

### 3.2 基础 commit

```
e003163181731412595257a72ec173071efb125f
[rollout] fix: handle malformed Qwen3 XML tool calls (#7044)
Date: 2026-07-17
```

### 3.3 已应用的修改（git status 会显示 5 个 modified 文件）

| 文件 | 来源 | 说明 |
|------|------|------|
| `verl/experimental/agent_loop/agent_loop.py` | VA-OPD patch | 第二次 degraded-image teacher pass |
| `verl/trainer/distillation/losses.py` | VA-OPD patch | `register_native_verl_loss(va_opd_k1)` |
| `verl/trainer/ppo/ray_trainer.py` | VA-OPD patch | `prepare_native_verl_batch` 调用点 |
| `verl/utils/vllm/utils.py` | cu132 compat | LoRA 导入: `vllm.lora.models` → `vllm.lora.lora_model` |
| `verl/trainer/constants_ppo.py` | cu132 compat | Ray runtime_env 转发 `LD_LIBRARY_PATH` |

**重要**: 前 3 个文件的 SHA256 在校验清单中 (`preflight_va_opd_native.py` 的 `EXPECTED_PATCHED_FILE_SHA256`)。
后 2 个文件是 vllm 0.25.x 兼容性修改，不在 SHA256 校验范围内。

### 3.4 安装方式

```bash
pip install --no-deps -e "$VERL_DIR"
```

Editable install，修改 Python 源文件立即生效（无需重装）。

---

## 4. GPU 实例信息

### 4.1 硬件

```
GPU:      8× NVIDIA H200 (141 GB each)
Topology: NV18 full mesh
Driver:   595.58.03
CUDA:     13.2
```

### 4.2 关键路径

```bash
# libcuda.so（驱动库，不在 conda 管辖范围）
/lib/x86_64-linux-gnu/libcuda.so.1
/lib/x86_64-linux-gnu/libcuda.so

# 网络：GPU 节点无外网访问
# 所有依赖必须在 CPU 节点预装到共享存储
```

### 4.3 运行前必须设置的环境变量

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export VA_OPD_ENV_PREFIX="$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu132"
export VA_OPD_STUDENT_MODEL="$DTOPD_ROOT/models/Qwen3-VL-4B-Instruct"
export VA_OPD_TEACHER_MODEL="$DTOPD_ROOT/models/Qwen3-VL-32B-Instruct"
export GEOMETRY3K_SOURCE="$DTOPD_ROOT/dataset/geometry3k/data/train-00000-of-00001.parquet"
export GEOMETRY3K_VA_OPD_DATA_DIR="$DTOPD_ROOT/fc-opd-storage/outputs/fc_opd/geometry3k_gkd"
```

### 4.4 Launcher 设置的 LD_LIBRARY_PATH

```
${ENV_PREFIX}/lib                                        ← libcudart.so.13, libcublas.so.13, ...
${ENV_PREFIX}/lib/python3.12/site-packages/torch/lib     ← libtorch.so, libc10.so, ...
${ENV_PREFIX}/targets/x86_64-linux/lib                   ← conda CUDA 运行时
/lib/x86_64-linux-gnu                                    ← libcuda.so.1 (驱动)
```

---

## 5. 历史环境构建步骤（禁止重放）

以下命令只记录当时做过什么。`setup_va_opd_native_env_cu132.sh` 已改为
fail closed；不要通过复制旧命令绕过它。

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export VA_OPD_ENV_PREFIX="$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu132"
export VERL_VA_OPD_DIR="$DTOPD_ROOT/fc-opd-storage/backends/verl-va-opd-e0031631"
export VA_OPD_CUDA_TOOLCHAIN="$DTOPD_ROOT/fc-opd-storage/toolchains/cuda-13.2"

cd "$DTOPD_ROOT/projects/Dual-Track-OPD"

# 历史命令，禁止执行
MAX_JOBS=16 bash scripts/hpc/setup_va_opd_native_env_cu132.sh

# 跳过 flash-attn 编译（训练用 PyTorch SDPA fallback）
VA_OPD_SKIP_FLASH_ATTN_BUILD=1 MAX_JOBS=16 \
    bash scripts/hpc/setup_va_opd_native_env_cu132.sh
```

构建脚本: `scripts/hpc/setup_va_opd_native_env_cu132.sh`

### 关键安装顺序

1. `conda create python=3.12` + CUDA 13.2 toolchain
2. `pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132`
3. `pip install vllm==0.25.1`（会拉下 torch 2.11.0 + 旧 nvidia-* 包）
4. `pip install --force-reinstall torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132`（覆盖回 torch 2.13）
5. `pip uninstall torchaudio`
6. `MAX_JOBS=16 pip install --no-build-isolation flash-attn==2.8.3`（需设置 CUDA_HOME=$ENV_PREFIX 和 include 路径）
7. `pip install --no-deps -e "$VERL_DIR"` + `pip install --no-deps -e "$REPO_ROOT"`

---

## 6. 当前状态

### 已通过 ✅

| Gate | 状态 |
|------|------|
| 环境构建 (CPU) | ✅ 通过 |
| torch/vllm/transformers import | ✅ 通过 |
| flash-attn 编译安装 | ✅ 通过 (2.8.3, SM90) |
| va_opd_k1 loss 注册 | ✅ 通过 |
| 数据集加载 (1901 train, 200 val) | ✅ 通过 |
| NCCL 4-rank smoke (0.008 + 0.156 GiB) | ✅ 通过 |
| Preflight 校验 | ✅ 通过 |

### 阻塞中 🔴

| Gate | 状态 | 原因 |
|------|------|------|
| OPD 3-step smoke | ❌ 阻塞 | vllm `_vllm_fa2_C.abi3.so` 无法在 Ray worker 中加载 |
| VA-OPD 3-step smoke | ❌ 阻塞 | 同上 |

### 阻塞根因

vllm 0.25.1 的预编译 C++ 扩展 (`_vllm_fa2_C.abi3.so`, `_vllm_fa3_C.abi3.so`) 链接了 torch **2.11.0** 的 libtorch，与 torch **2.13.0** 的 C++ ABI 不兼容。

ctypes 直接加载报错:
```
undefined symbol: _ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v
```

这是 `at::TensorBase::const_data_ptr()` 模板实例化符号，torch 2.11 → 2.13 之间发生了变化。

**核心矛盾**:
- cu132 → torch >= 2.12（PyTorch 不支持 CUDA 13.2 低于此版本）
- vllm 0.25.1 → 编译时链接 torch 2.11（且 PyPI 无更高版本）
- 不是 `LD_LIBRARY_PATH` 问题（已排除）

---

## 7. 已否决路径与当前决定

### 已否决：绕过 vllm 内置 FA2

attention backend 选择不能修复发生在模块导入阶段的 libtorch ABI 错误。

### 已否决：在本 backend 上源码编译 vLLM 0.25.1

这仍越过 verl e003 声明的 vLLM `<=0.12.0` API 边界，而且旧文档引用的
`4fd9d6a...` 实际是 vLLM 0.12.0，不是 0.25.1。

### 已否决：torch 2.11 + CUDA 13.0 + vLLM 0.25.1

虽然该组合可满足 vLLM 0.25.1 的 torch pin，但仍不满足 verl e003 的
vLLM API 边界，并且会把多个变量同时带入实验。

### 已否决：vLLM nightly

nightly 不能提供本实验所需的固定 backend/ABI/API provenance。

### 当前决定

在 R595/H200 节点使用受支持的 CUDA 12.8 用户态栈，统一 prefix 为：

```text
$DTOPD_ROOT/envs/va-opd-native-e003-cu128-r595-v1
```

R595 驱动向后兼容 cu128 应用。先重建干净 backend（仅 3-file VA patch）和
干净环境，再执行 NCCL、OPD、VA-OPD、paired pilot gates。详见
`docs/environment_registry.md`。

---

## 8. 关键文件索引

| 文件 | 说明 |
|------|------|
| `docs/va_opd_cu132_environment_spec_20260722.md` | 本文档 — 环境规格 |
| `docs/va_opd_cu132_status_20260722.md` | 构建过程与诊断记录 |
| `docs/va_opd_cu132_gpu_verify.md` | GPU 五阶段验证命令 |
| `configs/environment/verl_va_opd_e003_cu132.constraints.txt` | quarantined 历史版本记录，不得传给 pip |
| `scripts/hpc/setup_va_opd_native_env_cu132.sh` | fail-closed 事故保护入口 |
| `scripts/hpc/run_va_opd_native.sh` | 训练主启动脚本 |
| `scripts/hpc/preflight_va_opd_native.py` | 预检脚本 (修改后) |
| `scripts/setup/prepare_va_opd_native_verl.sh` | verl backend 准备 (修改后) |
| `src/dual_track_opd/fc_opd/verl_dataset.py` | 数据集适配 (修改后) |
| `$ENV_PREFIX/share/dual-track-opd/va_opd_environment_manifest.json` | 环境 manifest |
| `$VERL_DIR/verl/trainer/constants_ppo.py` | Ray LD_LIBRARY_PATH 转发 (修改后) |
| `$VERL_DIR/verl/utils/vllm/utils.py` | LoRA import 兼容 (修改后) |

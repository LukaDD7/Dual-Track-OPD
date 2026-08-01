# Dual-Track-OPD 环境台账与目录规范

最后更新：2026-07-25

本文是本项目在共享集群上的 Conda/Python 环境单一事实源。新建、重建、废弃或验证环境后，必须同时更新本文和环境内的 manifest。历史事故文档可以保留当时路径，但新的脚本和操作说明不得再自行发明环境目录。

## 1. 统一目录

项目专用 Conda prefix 统一放在：

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
# DTOPD = Dual-Track OPD, 项目在所有环境变量和路径中的统一前缀
```

目录约定：

```text
$DTOPD_ROOT/
├── envs/                               # 所有 Conda/Python prefix 及编译工具链
│   ├── va-opd-native-e003-cu128-r595-v1/   ← cu128 主线目标（备选）
│   ├── va-opd-native-e003-cu132-r595-v1/   ← cu132 主线目标（当前推进）
│   ├── vaopd-gkd-cu128/                    ← legacy
│   ├── vision-opd-cu128/                   ← legacy
│   ├── cuda128-toolchain/                  ← CUDA 12.8 编译工具链（非 Python env）
│   ├── cuda132-toolchain/                  ← CUDA 13.2 编译工具链（非 Python env）
│   ├── *.bad_*                             ← 失败尝试（诊断保留）
│   └── _backups/
├── fc-opd-storage/
│   ├── backends/                       # verl/vLLM 独立源码 checkout
│   ├── wheelhouse/                     # 可重建 wheel 与 SHA-256
│   ├── outputs/                        # 数据产物和 raw run output
│   ├── checkpoints/
│   └── diagnostics/
```

规则：

1. `envs/` 存放 Python 环境和 CUDA toolchain；backend checkout、wheelhouse、模型和数据不得混入。
2. 新脚本默认 `VA_OPD_ENV_PREFIX="$DTOPD_ROOT/envs/va-opd-native-e003-cu128-r595-v1"`，允许显式覆盖，但不得再默认到 `fc-opd-storage/envs/`、`miniconda3/envs/` 或其他零散目录。
3. Conda prefix 含绝对路径和 shebang，禁止用 `mv`、`cp -a` 或软链接伪装迁移。旧环境必须根据 lock/manifest 在统一目录重新创建并重新验证。
4. 环境名必须包含用途、backend 代际和 CUDA 用户态版本；需要并存的重建用 `-v2` 或日期后缀，不得原地覆盖一个已用于实验的 prefix。
5. “GPU 节点显示 CUDA 13.2”表示 R595 驱动可支持的最高 CUDA 版本，不要求 Python wheel 必须是 cu132。R595 可运行本项目锁定的 cu128 用户态栈。
6. 任何 `pip install --force-reinstall` 越过另一个包的精确 torch pin 都视为环境失效；`pip check`、native extension load 和真实 GPU kernel smoke 缺一不可。

## 2. 环境状态定义

| 状态 | 含义 | 是否可启动实验 |
|---|---|---:|
| `active` | 版本、manifest、GPU smoke 和指定训练 gate 均通过 | 是 |
| `candidate` | 干净解析成功，等待目标 GPU gate | 仅 preflight/smoke |
| `legacy` | 用于历史复现，禁止作为新主线 | 仅复现实验 |
| `quarantined` | 依赖、ABI、provenance 或行为已证伪 | 否 |
| `planned` | 只有设计，尚未完成构建和验证 | 否 |

“能 import”不能把环境从 `candidate` 升为 `active`。对 vLLM/FlashInfer/NCCL，必须执行真实模型加载、kernel 和 collective smoke。

## 3. 已知环境总表

构建日期无法从现有记录精确恢复时，表中明确写“首次证据日期”，不虚构创建时间。

| 环境 | 历史/目标路径 | 构建或首次证据日期 | 用途 | 关键版本 | 状态 | 后续处理 |
|---|---|---|---|---|---|---|
| FC-OPD verl 0.7.1 | 历史：`$DTOPD_ROOT/fc-opd-storage/envs/fc-opd-verl071-cu128` | 环境审计 2026-06-25；最早保存训练日志 2026-06-29 | FC-OPD/FSDP、项目 HTTP teacher、早期 Geometry3K GKD | Python 3.12、torch 2.8 cu128、vLLM 0.11、verl 0.7.1 | `legacy` | 不修改旧 prefix；只有复现实验需要时才按 freeze 重建到统一目录 |
| VAOPD GKD/Megatron | `$DTOPD_ROOT/envs/vaopd-gkd-cu128` | 环境审计 2026-07-09；smoke 2026-07-10 | 旧 `recipe/gkd`、Megatron、forward-GKD ablation | Python 3.12、torch 2.8 cu128、vLLM 0.11、Megatron Core 0.13.1 | `legacy` | 保留作 ablation，不用于 native VA-OPD 主线 |
| Vision-OPD baseline eval | `$DTOPD_ROOT/envs/vision-opd-cu128` | 首次完整诊断记录 2026-07-15 | Vision-OPD/Qwen3.5-4B benchmark 推理和 judge | Python 3.12、vLLM 0.18、Transformers 5.x、cu128 | `legacy` | 与训练环境严格隔离；只用于已记录的 baseline eval |
| Native VA-OPD cu128 初版 | `$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu128` | 构建尝试 2026-07-21 | 第一次 native OPD 环境尝试 | 初始错误矩阵含 vLLM 0.18 / Transformers 5.5 | `quarantined` | 保留失败证据，不修补、不激活 |
| Native VA-OPD cu128 v2 | `$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu128-v2` | 方案提交 2026-07-21；实际完整构建日期未确认 | 锁定 verl e003 + vLLM 0.12 的恢复方案 | torch 2.9 cu128、vLLM 0.12、Transformers 4.57.3 | `candidate`/待服务器确认 | 不直接搬迁；在统一目录重建为 R595 v1，并重新跑全部 gate |
| Native VA-OPD cu132 事故环境 | `$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu132` | 2026-07-21 | 试图利用 R595/cu132/NCCL 2.29.7 | torch 2.13 cu132 + vLLM 0.25.1 **wheel** + Transformers 5.14.1 | `quarantined` | 预编译 wheel ABI 断裂 + manifest provenance 错误；保留诊断，不激活 |
| Native VA-OPD cu132 on R595 v1（源码编译） | `$DTOPD_ROOT/envs/va-opd-native-e003-cu132-r595-v1` | 2026-07-25（10 次修复迭代后成功） | 当前 cu132 主线：source-built vLLM 0.25.1 + torch 2.13 | verl e003 + 5-file patch、torch 2.13.0+cu132、vLLM 0.25.1 **source wheel** (460MB)、Transformers 5.14.1、NCCL 2.29.7、flash-attn 2.8.3、flashinfer-python 0.6.13 | `candidate` | GPU kernel smoke、NCCL smoke、training smoke 待跑；详见 §5.1 构建问题清单 |
| Native VA-OPD cu128 on R595 v1 | `$DTOPD_ROOT/envs/va-opd-native-e003-cu128-r595-v1` | 2026-08-01（CPU 实例源码编译完成；vLLM wheel 缓存于 wheelhouse/va-opd-cu128-r595-v1） | 当前 cu128 主线：source-built vLLM 0.12.0 + torch 2.9 cu128 | verl e003 + 3-file patch、torch 2.9.0+cu128、vLLM 0.12.0 **source wheel** (744MB)、Transformers 4.57.3、flash-attn 2.8.3（源码编译 cu128）、flashinfer-python 0.5.3、NCCL 2.27.5 | `candidate` | GPU kernel smoke、NCCL smoke、training smoke 待跑；构建细节见 manifest |

CUDA 12.8 编译工具链位于 `$DTOPD_ROOT/envs/cuda128-toolchain`（nvcc V12.8.61），CUDA 13.2 编译工具链位于 `$DTOPD_ROOT/envs/cuda132-toolchain`。两者均为独立 Conda prefix，不与 Python 环境混合。

## 4. 当前主线选择

VA-OPD 不是从旧 `recipe/gkd` 继续打补丁。当前主线由三部分组成：

1. 项目自有研究逻辑：`src/dual_track_opd/va_opd/`，实现 visual advantage、K=4 sibling grouping、rollout softmax 和 high/low token weighting。
2. verl native OPD：student rollout、Qwen3-VL teacher force-scoring、FSDP2 actor 和 native sampled-token reverse-KL。
3. 最小 backend overlay：`patches/verl/va_opd_native_e0031631.patch`，只修改 3 个 verl 文件，增加 degraded-image teacher pass、batch adapter 调用和 loss registration。

`recipe/gkd`、HTTP teacher 和 vLLM 0.25 兼容补丁都不是 native VA-OPD 主线的一部分。

## 5. cu132 事故回顾与源码编译修正

### 5.0 2026-07-21 首次尝试（已隔离）

2026-07-21 构建的 cu132 环境失败原因已明确——**预编译** vLLM 0.25.1 wheel 的 C++ 扩展链接 torch 2.11 的 libtorch，与 CUDA 13.2 必需的 torch ≥2.12 不兼容。此外：

- verl commit `e0031631` 声明 `vllm>=0.8.5,<=0.12.0`——但这是 pip 元数据约束，不是运行时 API 硬断；
- 旧 manifest 记录 `vllm_source_commit: 4fd9d6a85c...`（vLLM **0.12.0** tag），实际安装的是 vLLM 0.25.1（tag `752a3a5044...`），provenance 错误；
- `pip install --force-reinstall torch==2.13` 是破坏性修补，不可复现；
- Conda CUDA 库与 torch wheel 的 `nvidia-*` runtime 同时进入 `LD_LIBRARY_PATH`，无审计。

旧 prefix 保留为 `$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu132`，仅诊断价值。

### 5.1 2026-07-25 源码编译：10 次修复记录

脚本：`scripts/hpc/setup_va_opd_native_env_cu132.sh`，2026-07-23 重写为可执行源码编译脚本后，于 2026-07-25 经过 10 轮迭代修复成功。

| # | 失败现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | `requirements/build.txt` not found | vLLM 0.25.1 将 `requirements/build.txt`（单文件）重构为 `requirements/build/`（目录，含平台特定文件） | 路径改为 `requirements/build/cuda.txt` |
| 2 | `RuntimeError: Cannot find CMake executable` | pip 安装的 cmake 二进制在 `${ENV_PREFIX}/bin`，但 PATH 只含 `${CUDA_TOOLCHAIN}/bin` | `export PATH="${ENV_PREFIX}/bin:${CUDA_TOOLCHAIN}/bin:${PATH}"` |
| 3 | `find_package(CUDA)` 模块不存在 | pip cmake 4.4.0 默认启用 CMP0146=NEW，移除旧版 FindCUDA | constraints 文件加 `cmake<4.0` + 显式 `pip install cmake>=3.26.1,<4.0` |
| 4 | `Could NOT find CUDA (missing: CUDA_INCLUDE_DIRS)` | cmake 3.31 中 CMP0146 默认值已变 NEW；`CUDA_TOOLKIT_ROOT_DIR` 未设置 | `export CMAKE_POLICY_DEFAULT_CMP0146=OLD` + `export CUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLCHAIN}"` |
| 5 | `cuda_select_nvcc_arch_flags` 未定义 | Python patch 替换 `find_package(CUDA)` 块时提前 `return()`，跳过了 arch flag 生成和 `enable_language(CUDA)` | Python 脚本精确替换，用 `include(FindCUDA)` 加载函数，硬编码 CUDA 变量但保留后续流程 |
| 6 | qutlass FetchContent 下载失败 | GitHub 临时不可达（`Failed to connect to github.com port 443`） | 网络恢复后重跑 |
| 7 | DeepGEMM `CUDA_HOME not found` | `build_deepgemm_C.py` 通过 `torch.utils.cpp_extension.CUDA_HOME` 检测 CUDA，cmake→ninja→Python 环境变量链路断裂 | setup.py patch 增加 `-DCUDA_HOME=...` cmake 参数 |
| 8 | DeepGEMM `fatal error: cuda.h: No such file or directory` | conda CUDA 13.2 的 `cuda.h` 在 `targets/x86_64-linux/include/`，不在 `${CUDA_HOME}/include/` | 尝试 `add_custom_command(ENVIRONMENT ...)` 无效（cmake 3.31+Ninja 将此关键字误解析为依赖名） |
| 9 | DeepGEMM `fatal error: device_types.h: No such file or directory` | 初次只 symlink 了 `cuda.h`、`cccl`、`crt`，缺其他头文件 | **最终方案**：symlink `targets/x86_64-linux/include/*` 全部到 `${CUDA_TOOLCHAIN}/include/`（见下文 §5.2） |
| 10 | `AssertionError: torch 2.13.0+cu132 != 2.13.0` | `importlib.metadata.version("torch")` 包含 local suffix `+cu132`，脚本用精确等号比较 | 期望值改为 `"2.13.0+cu132"` |

### 5.2 最终构建方案

**构建环境**：CPU 实例（64 核，503GB RAM），`MAX_JOBS=32`

**关键修复汇总**（`scripts/hpc/setup_va_opd_native_env_cu132.sh`）：

1. **cmake 版本约束**：`cmake>=3.26.1,<4.0`——cmake 4.x 移除 FindCUDA 模块
2. **FindCUDA 恢复**：`CMAKE_POLICY_DEFAULT_CMP0146=OLD` 环境变量
3. **CUDA 检测绕过**：Python 内联脚本 patch torch 的 `cuda.cmake`，硬编码 conda CUDA 13.2 布局下的路径到 `targets/x86_64-linux`
4. **CUDA header 兼容**：构建前 symlink `targets/x86_64-linux/include/*` → `${CUDA_TOOLCHAIN}/include/`（conda 将头文件放在 `targets/` 子目录，但 vLLM/build_deepgemm_C.py 期望在 `${CUDA_HOME}/include/`）
5. **vLLM torch 版本检查**：`TORCH_SUPPORTED_VERSION_CUDA` 从 `2.11.0` 改为 `2.13.0`
6. **PATH 完整**：确保 `${ENV_PREFIX}/bin` 在 PATH 中（cmake、ninja 由 pip 安装）
7. **CUDA_HOME 转发**：setup.py patch 增加 `-DCUDA_HOME=...` cmake 参数

**最终产物**：

| 项目 | 值 |
|---|---|
| 环境 prefix | `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/va-opd-native-e003-cu132-r595-v1` |
| Wheelhouse | `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/wheelhouse/va-opd-cu132-r595-v1/` |
| vLLM wheel | `vllm-0.25.1-cp312-cp312-linux_x86_64.whl` (460 MB) |
| SHA-256 | `94781779a6cf50aec1df1e93ac75847fafeeddcb7af4fb0bd30955d1fd034ba8` |
| torch | 2.13.0+cu132 |
| torchvision | 0.28.0+cu132 |
| transformers | 5.14.1 |
| flash-attn | 2.8.3 |
| flashinfer-python | 0.6.13 |
| tensordict | 0.10.0 |
| ray | 2.53.0 |
| vLLM source | `752a3a504485790a2e8491cacbb35c137339ad34` (v0.25.1 tag) |
| verl backend | `e003163181731412595257a72ec173071efb125f` |
| CUDA toolchain | nvcc V13.2.86, GCC 12.4.0 (conda-forge) |
| NCCL | 2.29.7 |
| 目标 GPU | NVIDIA H200 (SM90) |
| pip check | pass ✅ |
| 状态 | `candidate` — GPU kernel/NCCL/training smoke 待跑 |

## 6. 推荐环境的构建与验证

CPU 实例上：

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export VA_OPD_ENV_PREFIX="$DTOPD_ROOT/envs/va-opd-native-e003-cu128-r595-v1"
export VERL_VA_OPD_DIR="$DTOPD_ROOT/fc-opd-storage/backends/verl-va-opd-e0031631-clean"
export VA_OPD_CUDA_TOOLCHAIN="$DTOPD_ROOT/envs/cuda128-toolchain"
export VA_OPD_WHEELHOUSE="$DTOPD_ROOT/fc-opd-storage/wheelhouse/va-opd-cu128-r595-v1"

cd "$DTOPD_ROOT/projects/Dual-Track-OPD"
MAX_JOBS=16 bash scripts/hpc/setup_va_opd_native_env.sh
```

注意：backend 使用新的 `-clean` checkout，不能复用已经加过 vLLM 0.25 LoRA/Ray 兼容修改的目录。

构建完成后先检查：

```bash
"$VA_OPD_ENV_PREFIX/bin/python" -m pip check
"$VA_OPD_ENV_PREFIX/bin/python" -m json.tool \
  "$VA_OPD_ENV_PREFIX/share/dual-track-opd/va_opd_environment_manifest.json"
git -C "$VERL_VA_OPD_DIR" status --short
```

backend 只应出现 native patch 的 3 个 modified files。随后在 R595/H200 节点依次执行：

1. `collect_va_opd_gpu_facts.sh`；
2. 4-rank NCCL allgather/allreduce smoke；
3. native OPD 3-step smoke；
4. native VA-OPD 3-step smoke；
5. OPD 与 VA-OPD 各 50-step pilot；
6. 只有两个 pilot 的 loss、grad、entropy、clip ratio、VA 指标和 validation trajectory 都正常，才跑 full 5 epochs。

如果支持矩阵下仍复现 NCCL hang，第二阶段只替换 NCCL 做隔离实验，例如建立单独的 `...-nccl2297-v1` candidate；不要同时升级 torch、vLLM 和 Transformers。NCCL 变体必须有独立 prefix、manifest 和 smoke 记录。

## 7. 每个环境必须保存的 manifest

路径固定为：

```text
$CONDA_PREFIX/share/dual-track-opd/environment_manifest.json
```

native VA-OPD 同时保留兼容路径：

```text
$CONDA_PREFIX/share/dual-track-opd/va_opd_environment_manifest.json
```

至少记录：

```json
{
  "schema_version": 1,
  "environment_name": "va-opd-native-e003-cu128-r595-v1",
  "environment_prefix": "/absolute/path",
  "purpose": "native verl OPD and VA-OPD on H200",
  "status_at_build": "candidate",
  "created_at_utc": "ISO-8601 timestamp",
  "repo_commit": "git SHA",
  "repo_dirty": false,
  "backend_commit": "git SHA or package version",
  "constraints_sha256": "sha256",
  "python": "version",
  "packages": {},
  "torch_cuda": "12.8",
  "nccl": "version",
  "target_gpu": "H200 SM90",
  "build_node_kind": "CPU with internet",
  "verification": {
    "pip_check": "pass",
    "gpu_kernel_smoke": "pending",
    "nccl_smoke": "pending",
    "training_smoke": "pending"
  }
}
```

manifest 中的“日期”是实际构建完成时间，不是文档编写日期。每次重建创建新 prefix；不得覆盖旧 manifest 后继续沿用旧环境名。

## 8. CPU CC 交接要求

CPU 实例执行者完成一次环境工作后，必须回传并提交以下 Git-safe 信息：

- 环境名、绝对 prefix、用途、实际构建日期和状态；
- repo SHA/dirty 状态、backend SHA/patch SHA；
- constraints hash、wheel hash、关键包版本；
- `pip check` 结果；
- GPU facts、NCCL/kernel/training gate 的 pass/fail 和 raw log 路径；
- 若失败，保留 prefix 但移动逻辑状态到 `quarantined`，不得通过 force reinstall 继续污染；
- 更新本文总表。raw logs、hostname、GPU UUID、checkpoint、数据和权重仍留在 Git 外。

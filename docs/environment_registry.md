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
| Native VA-OPD cu128 on R595 v1 | `$DTOPD_ROOT/envs/va-opd-native-e003-cu128-r595-v1` | 2026-08-01 构建完成；2026-08-06 native real-image canary PASS | 当前 cu128 主线：source-built vLLM 0.12.0 + torch 2.9 cu128 | verl e003 + 3-file patch、torch 2.9.0+cu128、vLLM 0.12.0 **source wheel** (744MB)、Transformers 4.57.3、flash-attn 2.8.3（源码编译 cu128）、flashinfer-python 0.5.3、NCCL 2.27.5 | `active` | 2026-08-06 `run_va_opd_native.sh --objective opd --profile smoke --steps 4` 通过（真实 Geometry3K image、student rollout、teacher force-score、反向更新 4 步；exit 0、finite loss/grad、clip 0.25、mean len 1015）；run=`fc-opd-storage/runs/va_opd_native/qwen3vl_geometry3k_native_opd_smoke_canary_20260806_20260806_095422`，ckpt=`fc-opd-storage/checkpoints/va_opd_native/…_095422`；STP-OPD 四臂 mechanics pilot 的前置训练 gate 已清除 |
| qwen35 蒸馏 cu132 v0.9.0（新主线） | `$DTOPD_ROOT/envs/va-opd-qwen35-v090-cu132-r595-v1` | 2026-08-16 构建完成；GPU 三关全部通过 | qwen3.5/3.6 on-policy 蒸馏栈（27B→4B 等） | torch 2.13.0+cu132、vLLM 0.27.1 **cu130 wheel**（免源码编译）、transformers 5.12.0、flashinfer-python 0.6.16.post3、flash-attn 2.8.3（SM90 源码）、verl v0.9.0（`483b8a00` + 2 处本地补丁）、NCCL 2.29.7 | `active` | GPU 三关：preflight（eager+CUDA-graphs）✅（#50445 未复现）；NCCL 4-rank ✅（需 `NCCL_NVLS_ENABLE=0`，本实例 NVLS multicast 不可用）；训练 smoke ✅ 3/3 步（V0 trainer，checkpoint `global_step_3`，~36-41s/step）。**注意**：verl 0.9.0 V1 强制 transfer_queue 在本共享节点卡死（见 proposal §5.3），smoke/AB 用 V0；V1+delta_sharded 留作独立 candidate。对照组 `va-opd-qwen35-cu128` 保留 |

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

## 9. Codex 交接：cu128 R595 v1 GPU smoke 状态 (2026-08-01)

### 9.1 环境状态

| 项目 | 状态 |
|---|---|
| CPU 编译 (vLLM + flash-attn) | ✅ 完成 |
| import gate (所有版本检查) | ✅ 通过 |
| GPU 4×H200 – 模型加载 (Qwen3-4B-Instruct-2507) | ✅ 通过 |
| GPU – vLLM teacher server 启动 | ✅ 通过 |
| GPU – FSDP2 student init | ✅ 通过 |
| GPU – DataLoader | ✅ 通过 |
| GPU – 训练 step | ❌ reward function crash → 已修复 data_source |

### 9.2 环境激活命令

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export CUDA_HOME="${DTOPD_ROOT}/envs/cuda128-toolchain"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
conda activate "${DTOPD_ROOT}/envs/va-opd-native-e003-cu128-r595-v1"
```

### 9.3 Phase 1 Smoke 命令（forward_kl_topk，纯文本 Qwen3）

```bash
STUDENT_MODEL="${DTOPD_ROOT}/models/Qwen3-4B-Instruct-2507" \
TEACHER_MODEL="${DTOPD_ROOT}/models/Qwen3-4B-Instruct-2507" \
TRAIN_FILE="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/geometry3k_gkd/train_text_only.parquet" \
VAL_FILE="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/geometry3k_gkd/val_text_only.parquet" \
NGPUS_PER_NODE=4 \
TEACHER_WORLD_SIZE=1 \
TEACHER_TP=1 \
TEACHER_EP=1 \
rollout_gpu_mem_util=0.3 \
teacher_gpu_mem_util=0.4 \
max_prompt_length=2048 \
max_response_length=512 \
ppo_max_token_len_per_gpu=16384 \
train_batch_size=8 \
ppo_mini_batch_size=8 \
distillation_loss_mode=forward_kl_topk \
use_policy_gradient=False \
distillation_topk=32 \
total_epochs=1 \
save_freq=9999 \
test_freq=9999 \
bash "${DTOPD_ROOT}/fc-opd-storage/backends/verl-va-opd-e0031631-clean/examples/on_policy_distillation_trainer/run_qwen3_5_4b_fsdp.sh"
```

Phase 1 通过后，改 `distillation_loss_mode=va_opd_k1 use_policy_gradient=True` 跑 Phase 2。

### 9.4 已修复的问题

| 问题 | 修复 |
|---|---|
| `AttributeError: ThrMma` (cutlass 兼容) | `${ENV_PREFIX}/lib/.../nvidia_cutlass_dsl/.../core.py` 添加了 `ThrMma = object` stub |
| `ModuleNotFoundError: uvloop` | `pip install uvloop` |
| `RuntimeError: python-multipart` | `pip install python-multipart` |
| `AssertionError: processor needed` (数据有 images 列) | 创建 text_only 版 parquet（去掉了 images/condition_inputs 列） |
| `NotImplementedError: geometry3k reward` | 将 data_source 改为 `hiyouga/geometry3k` |
| `ModuleNotFoundError: mathruler` (geo3k reward 打分) | 从 `${DTOPD_ROOT}/envs/vision-opd-cu128` 复制 `mathruler` + `mathruler-0.1.0.dist-info` 到 `${ENV_PREFIX}/lib/python3.12/site-packages/`（纯 Python 包，已用目标 env python 验证 `verl.utils.reward_score.geo3k` import 与 compute_score） |
| `AssertionError: number of items:[0] < k_partitions:[4]` | mathruler 缺失的次生错误：reward 全部失败 → batch 为空 → `_balance_batch` 崩溃；补装 mathruler 后消除 |

### 9.5 文本数据文件

- Train: `${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/geometry3k_gkd/train_text_only.parquet` (1901 rows, `data_source=hiyouga/geometry3k`)
- Val: `${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/geometry3k_gkd/val_text_only.parquet` (200 rows, `data_source=hiyouga/geometry3k`)

### 9.6 已知待解决问题

1. **Qwen3-VL (视觉模型) 使用场景 → cutlass 冲突**：`nvidia-cutlass-dsl 4.6.1` 与 `flash-attn 2.8.3` 的 `cute` 模块不兼容。`ThrMma` patch 只修复第一个 AttributeError，后续 `ModuleNotFoundError: cutlass.utils.ampere_helpers` 会被 verl 的 `except ImportError` 捕获（仅影响 deepseek MoE 模型加载，Qwen3-VL 为 dense 模型不受影响）。**该路径在此 GPU node 上尚未验证。**

2. **Qwen3.5 模型**：`model_type=qwen3_5` 需要 `transformers>=5.x`，但 vLLM 0.12.0 约束 `transformers<5`。当前不可用。

### 9.7 关键路径

- 环境 prefix: `${DTOPD_ROOT}/envs/va-opd-native-e003-cu128-r595-v1`
- CUDA 工具链: `${DTOPD_ROOT}/envs/cuda128-toolchain` (nvcc 12.8.61)
- verl backend: `${DTOPD_ROOT}/fc-opd-storage/backends/verl-va-opd-e0031631-clean` (e0031631 + VA-OPD 3-file patch)
- vLLM wheel: `${DTOPD_ROOT}/fc-opd-storage/wheelhouse/va-opd-cu128-r595-v1/vllm-0.12.0-cp312-cp312-linux_x86_64.whl` (744MB)
- 约束文件: `configs/environment/verl_va_opd_e003_cu128.constraints.txt`
- 构建脚本: `scripts/hpc/build_fresh_cu128_opd_env.sh`

### 9.8 2026-08-01 磁盘满 + checkpoint 保存失败处置（codex 接手记录）

**现象**：默认配置（k1 / batch 128 / 15 epochs / save_freq=200）训练 210/210 步全部完成，最终保存
`global_step_210` 时 `PytorchStreamWriter failed writing file ... file write failed` / `unexpected pos ...`。

**根因**：共享 FS `${DTOPD_ROOT}` 所在 gpfs 卷 100% 满（0 可用），torch.save 写 zip 被截断。
占用大头：`projects/Dual-Track-OPD` 下 284G core dump + `third_party/Vision-OPD` 下 145G core dump
（6 月/7 月崩溃残留，已全部删除）+ 1.7T 历史实验 checkpoint（`projects/Dual-Track-OPD/checkpoints/gkd_geometry3k/`，7 月 Qwen3-VL GKD 跑批，未动）。

**当前状态**：
- 磁盘已恢复 446G 可用。
- 完整 checkpoint 仅 `global_step_200`（model+optim+extra_state+hf config 齐全，41G）。
- `global_step_210` 为损坏 partial（optimizer 截断、缺 extra_state），已删除。

**续跑命令**（从 step_200 继续跑完剩余 10 步，约 6 分钟）：
```bash
ray stop --force   # 先清掉上次崩溃残留的 ray
conda activate "${DTOPD_ROOT}/envs/va-opd-native-e003-cu128-r595-v1"
cd "${DTOPD_ROOT}/fc-opd-storage/backends/verl-va-opd-e0031631-clean/examples/on_policy_distillation_trainer"
bash run_qwen3_5_4b_fsdp.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=${DTOPD_ROOT}/checkpoints/verl_distill_geo3k/qwen3_5_4b_from_qwen3_5_35b_vllm_fsdp/global_step_200
```
注意 resume 配置须与原跑一致（即脚本默认值，不要再覆盖 batch/loss_mode 等）。

**防止复发建议**：
- 训练脚本加 `ulimit -c 0`，避免崩溃时再落 20-40G 的 core dump。
- 不再需要的旧实验 checkpoint（`checkpoints/gkd_geometry3k/` 1.7T）建议清理或归档到冷存储。

## 10. qwen3.5/3.6 家族 cu128 训练环境（2026-08-01 codex 构建）

### 10.1 背景与驱动约束

GPU 节点实测 `nvidia-smi`：**Driver 570.124.06 / CUDA Version 12.8**——即本节点
最高支持 CUDA 12.8。**cu130/cu132 用户态栈均不可用**（cu132 主线环境在本节点
无法启动）；qwen3.5/3.6（`model_type=qwen3_5/qwen3_5_moe`）需要
`transformers>=5` + 带 `qwen3_5` 实现的 vLLM（≥0.18，0.23 起有独立 `qwen3_5.py`），
因此必须构建 **cu128 + transformers 5.x + vLLM ≥0.23** 的组合。

### 10.2 环境

| 项目 | 值 |
|---|---|
| prefix | `${DTOPD_ROOT}/envs/va-opd-qwen35-cu128`（新建，非克隆） |
| python | 3.12.13 |
| torch | 2.11.0+cu129（torchvision 0.26.0+cu129 / torchaudio 2.11.0+cu129，CUDA 12.9，2026-08-02 由 cu128 切换） |
| vllm | 0.23.0+cu129（wheels.vllm.ai 官方 cu129 变体 wheel，477MB；依赖 flashinfer-python/cubin 0.6.12） |
| transformers | 5.12.0 |
| flash-attn | 2.8.3（cu128 wheel，torch 2.10 构建；verl unpad 必需，已在本栈验证 import + CUDA 扩展加载 OK） |
| verl | 0.9.0.dev0 @ `334d9f8b`（editable，`${DTOPD_ROOT}/repos/verl-cu130-vllm`，git 干净无 patch） |
| ray / tensordict | 2.55.1 / 0.10.0 |
| 蒸馏 loss modes | `k1/k2/k3/kl/abs/mse/low_var_kl/forward_kl_topk` 全部注册 ✓ |
| 其他 | mathruler 0.1.0、qwen_vl_utils 0.0.14（geo extras）、numpy 1.26.4（保持不动，torch cu129 不强制 numpy 2） |
| 离线 wheelhouse | `${DTOPD_ROOT}/fc-opd-storage/wheelhouse/va-opd-qwen35-cu129/`（36 个 wheel + cutlass v4.4.2 源码，供 GPU 节点离线重装） |

### 10.3 激活与运行前必设

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export CUDA_HOME="${DTOPD_ROOT}/envs/cuda128-toolchain"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
# flashinfer 需要可写 workspace（默认 $HOME/.cache 在部分节点只读）
export FLASHINFER_WORKSPACE_BASE="${DTOPD_ROOT}/.cache/flashinfer"
conda activate "${DTOPD_ROOT}/envs/va-opd-qwen35-cu128"
```

### 10.4 已验证（CPU import gate）

- torch 2.11.0+cu128 / vllm 0.23.0（registry 含 `Qwen3_5ForConditionalGeneration`）/
  transformers 5.12.0 加载 `qwen3.6-27B`（model_type=qwen3_5）✓
- verl 蒸馏 registry + geo3k reward（mathruler）✓

### 10.5 待做（GPU 节点）

- **✅ 已完成**：cu129 栈 GPU 预检 + qwen3.5-4B ← qwen3.5-4B 自蒸馏 smoke（4/4 步，
  2026-08-02），详见 10.6。
- **正式实验**：`scripts/run_qwen35_formal.sh`（默认 qwen3.6-27B → Qwen3.5-4B，
  k1，use_task_rewards=False，batch 56，response 2048，7+1 卡，含 teacher 预检）。
  变量开关：`LOSS_MODE`、`USE_TASK_REWARDS`、`STUDENT`（Qwen3.5-9B）、`TEACHER`
  （qwen3.6-35B-A3B 需 6+2 卡 + TP=2）、`TRAIN_BATCH_SIZE`（56/112/168…）。
- qwen3.6-27B → Qwen3.5-9B 容量差实验（STUDENT 切换后复跑）。

### 10.6 cu13 排除结论与最优组合（2026-08-02 调研定稿）

**执行状态（2026-08-02 07:01 UTC）**：cu129 栈已在共享存储上完成切换并通过 CPU import
gate —— torch 2.11.0+cu129 / cuda 12.9 / vllm 0.23.0+cu129 / transformers 5.12.0 /
verl `run_ppo` import 全部 OK；`vllm/_C.abi3.so` NEEDED `libcudart.so.12`；
cu13 组件（nvidia-cuda-runtime 13.3.29、cutlass-dsl-libs-cu13、nvcc/nvrtc/nvjitlink 13.x）已卸载；
numpy 保持 1.26.4。全部 wheel 已离线缓存于
`fc-opd-storage/wheelhouse/va-opd-qwen35-cu129/`，GPU 节点可用
`scripts/setup_qwen35_cu129.sh` 随时重放。

**GPU 预检（2026-08-02 07:17 UTC）**：✅ MVC 结论实测成立 —— H200 + driver 570.124.06 +
cu129 栈，vLLM 0.23.0 引擎完整初始化成功（Qwen3_5ForConditionalGeneration、
FlashAttention v3、FlashInfer top-k/top-p、FlashInfer GDN prefill JIT、模型加载 8.61 GiB、
KV cache 29.28 GiB）。唯一报错为 preflight 脚本 API 兼容：vLLM 0.23 的
`LLM.generate()` 不再接受 `max_tokens` 关键字，已改为
`LLM.generate(..., SamplingParams(max_tokens=8))`（2026-08-02 07:26 修复）。
首启引擎 init 耗时 219s（flashinfer JIT 一次性），JIT 缓存已写入
`${DTOPD_ROOT}/.cache/flashinfer`，重跑会更快。

**全链路 smoke（2026-08-02 08:0x UTC）**：✅ qwen3.5-4B ← qwen3.5-4B 自蒸馏 4/4 步完成，
~35s/step（H200×8，7 actor + 1 teacher），checkpoint 落盘于
`${DTOPD_ROOT}/repos/verl-cu130-vllm/examples/on_policy_distillation_trainer/checkpoints/
verl_distill_qwen35/qwen3_5_4b_self_gpu_smoke/global_step_4/`。
关键指标：actor/distillation/loss 0.0001→0.0005、rollout_corr/kl ~0.0005、
rollout_probs_diff_mean ~0.0036（self-distillation 同模型，差小属预期）、
reward=0.0（smoke 配置 use_task_rewards=False，蒸馏项为学习信号）。
过程中修复：agent.num_workers=7（batch 21 切分约束）、flash_attn 2.8.3 安装
（verl unpad 必需）。日志尾部 torchdata dataloader worker "Killed" 为训练完成后
teardown 阶段的无害噪音（非训练失败）。

**正式实验 #1（2026-08-02 09:2x UTC）**：✅ qwen3.6-27B → Qwen3.5-4B，k1，
use_task_rewards=False，4 卡（3 actor + 1 teacher，另一实验占 GPU 4-7，FORMAL_GPUS=0,1,2,3），
batch 24，79/79 步（1901 行），~63.5s/it，共 1h24m。
关键指标（step 79）：actor/entropy 0.44、distillation/abs_loss 0.175（≈ 自蒸馏 smoke 的 10 倍，
容量差信号正常）、distillation/loss 0.0805、grad_norm 1.65、rollout_probs_diff_mean 0.0028
（学生紧跟 27B teacher）、KL 0.00032、reward 0（use_task_rewards=False 属预期）。
checkpoint：`checkpoints/verl_distill_qwen35/qwen3_6_27b_to_qwen3_5_4b_k1_false/`
（global_step_10..70,79；**每个约 51GB，共 407GB**，注意磁盘规划）。
**踩坑记录**：vLLM 日志默认写 stdout（Ray 分流到 worker-*.out），诊断需
`VLLM_LOGGING_STREAM=ext://sys.stderr` 或同时 tail .out；teacher GPU 残留/被占用时
`request_memory` 报 free<desired；GPU 实例同时跑多个实验时必须用 FORMAL_GPUS 显式限卡。

**驱动硬约束**：GPU 节点 driver 570.124.06，nvidia-smi 显示最大 CUDA 12.8。
已实测 vllm 0.23.0 PyPI wheel（CUDA 13.0 构建，`_C.abi3.so` NEEDED `libcudart.so.13`，
deps 含 `nvidia-cutlass-dsl[cu13]`）引擎启动报
`CUDA driver version is insufficient for CUDA runtime version`。
NVIDIA 官方 MVC 兼容性表确认：CUDA 12.x 应用最小驱动 >= 525（Linux，上限 <580），
CUDA 13.x 应用最小驱动 >= 580 → **cu130/cu132 用户态栈在本节点全部不可用**。

**最优组合（首选，免编译）**：

| 组件 | 版本 | 来源 |
|---|---|---|
| torch | 2.11.0+cu129 | `https://download.pytorch.org/whl/cu129` |
| torchvision / torchaudio | 0.26.0+cu129 / 2.11.0+cu129 | 同上 |
| vllm | 0.23.0+cu129 | `https://wheels.vllm.ai/0.23.0/cu129/vllm/vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl` |
| flashinfer-python / -cubin | 0.6.12（cu12/cu13 无关） | PyPI（已装） |
| nvidia-cutlass-dsl | 4.5.2 `[cu12]` 风味 | PyPI |
| humming-kernels | 0.1.4 `[cu12]` 风味 | PyPI |
| transformers / verl | 5.12.0 / `334d9f8b` | 不变 |

要点：
- vllm 0.23.0 默认 PyPI wheel 是 cu130；cu129 变体只发布在
  `wheels.vllm.ai/0.23.0/cu129`，必须显式指定 URL（uv/pip 的
  `--torch-backend=cu129` 可能仍解析回 cu130 wheel，见 vllm #44335/#42338）。
- cu129 运行在 driver 570 依赖 MVC；若实测仍报 driver 不足（部分 kernel 特性可能
  要求新驱动），回退方案 B 为源码构建 vllm 0.23.0（tag `91df0fad4`）于
  cuda128-toolchain（qutlass pin `830d2c45` 与本地缓存一致；cutlass v4.4.2、
  triton_kernels v3.5.1 需网络或降级容忍）。
- 脚本：`scripts/setup_qwen35_cu129.sh`（切换 12.9 栈）→
  `scripts/run_qwen35_gpu_smoke.sh`（8 卡预检 + 自蒸馏 smoke）。
- 需要先探测 GPU 节点到 `download.pytorch.org` / `wheels.vllm.ai` 的网络。

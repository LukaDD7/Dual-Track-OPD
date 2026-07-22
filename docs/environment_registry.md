# Dual-Track-OPD 环境台账与目录规范

最后更新：2026-07-23

本文是本项目在共享集群上的 Conda/Python 环境单一事实源。新建、重建、废弃或验证环境后，必须同时更新本文和环境内的 manifest。历史事故文档可以保留当时路径，但新的脚本和操作说明不得再自行发明环境目录。

## 1. 统一目录

项目专用 Conda prefix 统一放在：

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export DTOPD_CONDA_ENVS_ROOT="${DTOPD_ROOT}/conda-envs"
```

目录约定：

```text
$DTOPD_ROOT/
├── conda-envs/                         # 只放可激活的 Conda/Python prefix
│   ├── fc-opd-verl071-cu128/
│   ├── vaopd-gkd-cu128/
│   ├── vision-opd-cu128/
│   ├── va-opd-native-e003-cu128-r595-v1/
│   └── quarantine/                     # 失败环境只保留诊断，不再运行
├── fc-opd-storage/
│   ├── backends/                       # verl/vLLM 独立源码 checkout
│   ├── wheelhouse/                     # 可重建 wheel 与 SHA-256
│   ├── outputs/                        # 数据产物和 raw run output
│   ├── checkpoints/
│   └── diagnostics/
└── toolchains/                         # nvcc/GCC 等编译工具链，不是 Python env
    ├── cuda-12.8/
    └── cuda-13.2/                      # 仅保留 cu132 事故复现所需时使用
```

规则：

1. `conda-envs/` 只存放 Python 环境；CUDA toolchain、backend checkout、wheelhouse、模型和数据不得混入。
2. 新脚本使用 `DTOPD_CONDA_ENVS_ROOT`，允许显式 `VA_OPD_ENV_PREFIX` 等变量覆盖，但不得再默认到 `fc-opd-storage/envs/`、`miniconda3/envs/` 或零散的 `$DTOPD_ROOT/envs/`。
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
| FC-OPD verl 0.7.1 | 历史：`$DTOPD_ROOT/fc-opd-storage/envs/fc-opd-verl071-cu128`；统一目标：`$DTOPD_CONDA_ENVS_ROOT/fc-opd-verl071-cu128` | 环境审计 2026-06-25；最早保存训练日志 2026-06-29 | FC-OPD/FSDP、项目 HTTP teacher、早期 Geometry3K GKD | Python 3.12、torch 2.8 cu128、vLLM 0.11、verl 0.7.1 | `legacy` | 不修改旧 prefix；只有复现实验需要时才按 freeze 重建到统一目录 |
| VAOPD GKD/Megatron | 历史：`$DTOPD_ROOT/envs/vaopd-gkd-cu128`；统一目标：`$DTOPD_CONDA_ENVS_ROOT/vaopd-gkd-cu128` | 环境审计 2026-07-09；smoke 2026-07-10 | 旧 `recipe/gkd`、Megatron、forward-GKD ablation | Python 3.12、torch 2.8 cu128、vLLM 0.11、Megatron Core 0.13.1 | `legacy` | 保留作 ablation，不用于 native VA-OPD 主线 |
| Vision-OPD baseline eval | 当前 Conda name/prefix：`vision-opd-cu128`；统一目标：`$DTOPD_CONDA_ENVS_ROOT/vision-opd-cu128` | 首次完整诊断记录 2026-07-15 | Vision-OPD/Qwen3.5-4B benchmark 推理和 judge | Python 3.12、vLLM 0.18、Transformers 5.x、cu128 | `legacy` | 与训练环境严格隔离；只用于已记录的 baseline eval |
| Native VA-OPD cu128 初版 | `$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu128` | 构建尝试 2026-07-21 | 第一次 native OPD 环境尝试 | 初始错误矩阵含 vLLM 0.18 / Transformers 5.5 | `quarantined` | 保留失败证据，不修补、不激活 |
| Native VA-OPD cu128 v2 | `$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu128-v2` | 方案提交 2026-07-21；实际完整构建日期未确认 | 锁定 verl e003 + vLLM 0.12 的恢复方案 | torch 2.9 cu128、vLLM 0.12、Transformers 4.57.3 | `candidate`/待服务器确认 | 不直接搬迁；在统一目录重建为 R595 v1，并重新跑全部 gate |
| Native VA-OPD cu132 事故环境 | `$DTOPD_ROOT/fc-opd-storage/envs/va-opd-verl-e003-cu132` | 2026-07-21 | 试图利用 R595/cu132/NCCL 2.29.7 | torch 2.13 cu132 + vLLM 0.25.1 wheel + Transformers 5.14.1 | `quarantined` | ABI 与 verl API 均不受支持；不得运行、不得作为 base env |
| Native VA-OPD cu128 on R595 v1 | `$DTOPD_CONDA_ENVS_ROOT/va-opd-native-e003-cu128-r595-v1` | `planned`，由 CPU 实例完成后写入 manifest | 当前推荐的 OPD/VA-OPD 主线 | verl e003 + 3-file patch、torch 2.9 cu128、vLLM 0.12 cu128 source wheel、Transformers 4.57.3 | `planned` | 下一步构建；先 NCCL smoke，再 OPD/VA smoke 和成对 pilot |

另有 `$DTOPD_ROOT/envs/cuda128-toolchain`。它是 CUDA 12.8 编译工具链，不是 Conda Python 环境；应在下一次重建时改为 `$DTOPD_ROOT/toolchains/cuda-12.8`，旧路径在历史复现完成前不删除。

## 4. 当前主线选择

VA-OPD 不是从旧 `recipe/gkd` 继续打补丁。当前主线由三部分组成：

1. 项目自有研究逻辑：`src/dual_track_opd/va_opd/`，实现 visual advantage、K=4 sibling grouping、rollout softmax 和 high/low token weighting。
2. verl native OPD：student rollout、Qwen3-VL teacher force-scoring、FSDP2 actor 和 native sampled-token reverse-KL。
3. 最小 backend overlay：`patches/verl/va_opd_native_e0031631.patch`，只修改 3 个 verl 文件，增加 degraded-image teacher pass、batch adapter 调用和 loss registration。

`recipe/gkd`、HTTP teacher 和 vLLM 0.25 兼容补丁都不是 native VA-OPD 主线的一部分。

## 5. cu132 环境为什么被隔离

2026-07-21 构建的 cu132 环境不是少一个环境变量，而是越过了多重兼容边界：

- verl commit `e0031631` 声明 `vllm>=0.8.5,<=0.12.0`，而事故环境安装了 vLLM 0.25.1；
- vLLM 0.25.1 wheel 精确绑定 torch 2.11，随后强制覆盖成 torch 2.13，导致 `_vllm_fa2_C.abi3.so` 缺少 libtorch 符号；
- `VLLM_ATTENTION_BACKEND` 不能修复模块导入阶段的 C++ ABI 错误；
- 记录中的 `4fd9d6a85c...` 是 vLLM 0.12.0 tag，不是 0.25.1；vLLM 0.25.1 tag 是 `752a3a5044...`，因此旧 manifest 的 source provenance 不真实；
- setup 脚本、constraints 和 preflight 分别要求不同的 torch/Transformers 版本，不能从空 prefix 可重复构建；
- Conda CUDA libraries 与 torch wheel 的 `nvidia-*` runtime 同时优先进入 `LD_LIBRARY_PATH`，形成第二套未审计的动态库选择。

该 prefix 仅保留诊断价值。`scripts/hpc/setup_va_opd_native_env_cu132.sh` 现在会 fail closed，避免再次生成已知无效环境。

## 6. 推荐环境的构建与验证

CPU 实例上：

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export DTOPD_CONDA_ENVS_ROOT="$DTOPD_ROOT/conda-envs"
export VA_OPD_ENV_PREFIX="$DTOPD_CONDA_ENVS_ROOT/va-opd-native-e003-cu128-r595-v1"
export VERL_VA_OPD_DIR="$DTOPD_ROOT/fc-opd-storage/backends/verl-va-opd-e0031631-clean"
export VA_OPD_CUDA_TOOLCHAIN="$DTOPD_ROOT/toolchains/cuda-12.8"
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

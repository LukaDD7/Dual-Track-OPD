# VA-OPD qwen3.5/3.6 蒸馏栈 cu132 环境 Proposal（2026-08-16）

> 状态：`planned → candidate → active`（2026-08-16 构建完成；import gate 全绿；
> GPU 引擎/NCCL/训练三关全部通过，manifest 已升 active）。
> 本文件是版本矩阵、构建规则和 GPU gate 的单一事实源。

## 1. 背景

2026-07-21 第一次 cu132 尝试（torch 2.13+cu132 × vLLM 0.25.1 PyPI wheel）因
**libtorch C++ ABI 断裂**而 quarantine（`docs/va_opd_cu132_status_20260722.md`）；
2026-07-25 源码编译 vLLM 0.25.1 成功（`va-opd-native-e003-cu132-r595-v1`，
`candidate`），但 GPU 三关一直未跑，且 0.25.1 时代需要源码编译的根本原因——
vLLM wheel 对齐 torch 2.11——已经被 vLLM 0.27.1（官方对齐 torch 2.13）消除。

新 GPU 实例：Driver **595.58.03 / CUDA 13.2**（高于 cu132 需要的 ≥580），
8×H200（SM90，143771 MiB），NV18 全互联 mesh，双 NUMA（GPU0-3 / GPU4-7），
每卡一根本地 mlx5 NIC。系统无 nvcc（已实测 `command not found`），
`libcuda.so.1` 在非标准路径 `/lib/x86_64-linux-gnu/`。

## 2. 推荐版本矩阵（已逐项核实）

| 组件 | 版本 | 来源/依据 |
|---|---|---|
| torch | 2.13.0+cu132 | `download.pytorch.org/whl/cu132`（cp312 wheel 已确认存在） |
| torchvision | 0.28.0+cu132 | 同上 |
| torchaudio | 2.11.0（--no-deps） | vLLM 0.27.1 硬性依赖 `torchaudio==2.11.0`；该 wheel **无任何依赖**，且 2.13.0+cu132 的 torchaudio 不存在；本栈不用音频 |
| vLLM | **0.27.1（cu130 wheel）** | `wheels.vllm.ai/0.27.1/cu130/vllm/vllm-0.27.1-cp38-abi3-manylinux_2_28_x86_64.whl`（已确认存在；PyPI 默认也是 cu130 风味，显式 URL 仅为 provenance） |
| transformers | 5.12.0 | qwen3.5 栈已验证版本；vLLM 0.27.1 要求 `>=5.5.3` 满足 |
| flashinfer-python | 0.6.16.post3 | vLLM 0.27.1 精确 pin；**不上 0.6.17**（Hopper fa2→fa3 decode 回归，SGLang #18364） |
| flash-attn | 2.8.3（源码编译 SM90） | verl unpad 必需；PyPI 预编译 wheel 是 cu12x 风味，必须 `--no-binary :all:` 源码构建 |
| verl | **0.9.0**（tag `483b8a00`，08-13 发布） | delta_sharded 权重同步（7B/32B/72B 官方 2.4x/1.9x/3.1x）、Qwen3 MoE FSDP 同步修复、V1 unified trainer |
| ray | 2.55.1 | 与 qwen35 栈一致 |
| tensordict | 0.10.0 | verl pin 上限 |
| numpy | >=2.0.0 | verl 0.9.0 要求（qwen35 栈保留 1.26.4 是旧栈约束，不迁移） |

关键兼容性结论：

1. **ABI 断裂点消失**：vLLM 0.27.1 wheel 编译时对齐 torch 2.13，torch 2.13.0+cu132
   满足其 `torch==2.13.0` pin（PEP 440 本地版本号 `+cu132` 不影响等值匹配）。
2. **CUDA 运行时统一为 13 系**：driver 595 对 13.0/13.1/13.2 运行时全部兼容；
   vLLM 0.27.1 的 `_C.abi3.so` NEEDED `libcudart.so.13`，由 torch 2.13+cu132
   自带的 `nvidia-cuda-runtime-cu13`（13.2.x）提供。
3. **verl 0.9.0 元数据 `transformers<5.11` 是保守 pin**，与本栈（qwen3.5 需要
   transformers 5.12+）冲突；verl 以 `--no-deps` editable 安装绕过，由 import gate
   和真实模型加载 gate 兜底验证。

## 3. 环境构建规则（从 git 历史提炼的"潜规则"）

来源：`scripts/setup_qwen35_cu129.sh`、`scripts/hpc/setup_va_opd_native_env_cu132.sh`、
`docs/environment_registry.md`（§5.1/§10）、`scripts/hpc/run_va_opd_native.sh`。

1. **GPU 节点无外网**：所有依赖在 CPU 节点装到共享存储 prefix，GPU 节点直接消费；
   wheelhouse 仅用于重建/离线恢复，`--no-index --no-deps` 安装。
2. **运行时自包含 NVIDIA 库**：libcudart/cublas/cudnn/nvrtc/nvjitlink/nccl 全部是
   env 内 pip `nvidia-*` 包；系统只允许 driver 的 `libcuda.so.1`
   （`/lib/x86_64-linux-gnu/`）。系统 nvcc 不存在。
3. **nvcc/GCC 独立工具链 prefix**（`cuda132-toolchain`，nvcc 13.2.86/GCC 12.4），
   不混入 Python env；仅 flashinfer JIT 和源码编译时使用。
4. **版本风味显式 + readelf 审计**：vLLM wheel 显式 URL；装完
   `readelf -d vllm/_C.abi3.so | grep NEEDED` 验证 `libcudart.so.13`；
   pip list 审计 cu12/cu13 残留。
5. **launcher 运行时环境**：`LD_LIBRARY_PATH` 加入 `${CUDA_HOME}/lib`、
   `${CUDA_HOME}/targets/x86_64-linux/lib`、`site-packages/torch/lib`、
   `/lib/x86_64-linux-gnu`；Ray 转发 LD_LIBRARY_PATH；`FLASHINFER_WORKSPACE_BASE=
   ${DTOPD_ROOT}/.cache/flashinfer`；`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。
6. **可复现三关**：禁止 `--force-reinstall` 越过 torch pin；`pip check` +
   native extension 加载 + 真实 GPU kernel smoke；manifest 记录真实 commit/wheel
   SHA-256；状态机 `candidate → active`。
7. **conda CUDA 布局坑**：headers 在 `targets/x86_64-linux/include`（构建前 symlink
   到 `${CUDA_HOME}/include/`）；cmake `<4.0` + `CMP0146=OLD`（FindCUDA）；
   `TORCH_CUDA_ARCH_LIST=9.0`。

## 4. 构建资产

| 项 | 路径 |
|---|---|
| 环境 prefix | `${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1` |
| wheelhouse | `${DTOPD_ROOT}/fc-opd-storage/wheelhouse/va-opd-qwen35-cu132-r595-v1/` |
| verl backend | `${DTOPD_ROOT}/fc-opd-storage/backends/verl-qwen35-v090-cu132` @ `483b8a00`（+2 本地改动：vLLM engine diag hook、run 脚本 env 开关） |
| 构建脚本 | `scripts/hpc/setup_va_opd_qwen35_cu132_env.sh` |
| 约束文件 | `configs/environment/verl_va_opd_qwen35_cu132.constraints.txt` |
| manifest | `${ENV_PREFIX}/share/dual-track-opd/va_opd_qwen35_environment_manifest.json` |

对照组保留：`va-opd-qwen35-cu128`（torch 2.11+cu129 / vLLM 0.23+cu129，
`active`）——新栈 AB 对比的 baseline。

## 5. GPU gate（GPU 节点执行，三关全过后升 `active`）

前置（每条命令前都要 export）：

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export CUDA_HOME="${DTOPD_ROOT}/envs/cuda132-toolchain"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="${DTOPD_ROOT}/.cache/flashinfer"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1"
```

1. **内核/引擎 gate**：`python ${DTOPD_ROOT}/projects/Dual-Track-OPD/scripts/qwen35_vllm_preflight.py`
   ——真实 Qwen3.5-4B 引擎初始化 + 8 token 生成（注意 vLLM 0.27 的 CUDA-graph
   warmup 问题见 §6，preflight 用 eager 兜底）。
2. **NCCL gate**（4 rank，用空闲 GPU 2-5）：
   `CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 ${DTOPD_ROOT}/projects/Dual-Track-OPD/scripts/hpc/smoke_va_opd_nccl.py --elements 4198742,83678470 --iterations 5`
3. **训练 gate**：qwen3.5-4B ← qwen3.5-4B 自蒸馏 4-step smoke
   （`scripts/run_qwen35_gpu_smoke.sh` 的 cu132 变体，6 卡时
   `FORMAL_GPUS=2,3,4,5,6,7` / `NGPUS_PER_NODE=5`）；通过后正式 27B→4B AB 对比
   （baseline：`va-opd-qwen35-cu128` ~63.5s/it @4卡）。

## 5.1 构建结果（2026-08-16）

| 项 | 值 |
|---|---|
| 环境 prefix | `${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1`（`candidate`） |
| manifest | `${ENV_PREFIX}/share/dual-track-opd/va_opd_qwen35_environment_manifest.json` |
| import gate | ✅ torch 2.13.0+cu132 / vllm 0.27.1（qwen3_5 arch 在 registry，367 archs）/ transformers 5.12.0（`Qwen3_5ForConditionalGeneration` OK）/ flashinfer 0.6.16.post3 / flash-attn 2.8.3 / verl 0.9.0 蒸馏 loss 全注册 |
| readelf 审计 | `vllm/_C_stable_libtorch.abi3.so` NEEDED `libcudart.so.13` + `libcuda.so.1` ✅（vllm 0.27 为 stable-libtorch 构建，无旧 `_C.abi3.so`） |
| pip check | 唯一警告：verl 0.9.0 元数据 `transformers<5.11` vs 实际 5.12.0（预期覆盖，见 §2.3） |
| NCCL | 2.29.7（torch 自带 `nvidia-nccl-cu13`） |
| 构建脚本 | `scripts/hpc/setup_va_opd_qwen35_cu132_env.sh`（幂等可重跑） |
| GPU smoke 脚本 | `scripts/run_qwen35_cu132_gpu_smoke.sh`（默认 6 卡：GPU2-7 = 5 actor + 1 teacher） |

构建中处理的两个环境级问题（记录在案，重建脚本已内置修复）：

1. **wheels.vllm.ai 真实 URL 带 build-commit 前缀**：列表页 `/0.27.1/cu130/vllm/`
   下的直链是 404，文件实际在 `/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/`。
2. **torchaudio 2.11.0（cu130 风味）与 torch 2.13.0+cu132 的 CUDA 版本检查冲突**：
   vLLM 0.27.1 硬性 pin `torchaudio==2.11.0`，该 wheel 是 CUDA 13.0 构建，而
   torchaudio `_check_cuda_version()` 对任意小版本差异抛错，连带 transformers /
   verl / vLLM registry 导入全挂。已将该检查放宽为**只比大版本**（driver 595
   兼容全部 13.x 运行时，本栈不用音频），patch 打在 env 内
   `torchaudio/_extension/utils.py`（带 `DT_OPD` 标记，幂等）。

### 5.2 GPU 节点执行顺序（完整命令）

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export CUDA_HOME="${DTOPD_ROOT}/envs/cuda132-toolchain"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1/lib/python3.12/site-packages/torch/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="${DTOPD_ROOT}/.cache/flashinfer"
export VERL_DIAG_DIR="${DTOPD_ROOT}/fc-opd-storage/logs"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "${DTOPD_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1"
cd "${DTOPD_ROOT}/projects/Dual-Track-OPD"

# 1) 环境审计（nvcc/驱动/torch/vllm 版本 + 残留检查）
bash scripts/hpc/audit_nccl_stack.sh --env-path "${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1"

# 2) 引擎 gate（eager，验证栈本身；PREFLIGHT_EAGER=0 再跑一次验证 CUDA-graph warmup）
python scripts/qwen35_vllm_preflight.py
PREFLIGHT_EAGER=0 python scripts/qwen35_vllm_preflight.py

# 3) NCCL gate（4 rank，GPU2-5）
CUDA_VISIBLE_DEVICES=2,3,4,5 NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=4 \
  scripts/hpc/smoke_va_opd_nccl.py --elements 4198742,83678470 --iterations 5

# 4) 训练 gate（默认 GPU2-7 = 5 actor + 1 teacher，~4 步）
bash scripts/run_qwen35_cu132_gpu_smoke.sh   # 默认 USE_V1=0（V0，绕开 transfer_queue；AB 对照与旧 baseline 同为 V0）

# 5) （可选）8 卡自蒸馏 smoke（需先释放 GPU0-1）
SMOKE_GPUS=0,1,2,3,4,5,6,7 bash scripts/run_qwen35_cu132_gpu_smoke.sh
```

## 5.3 GPU 实测记录（2026-08-16）

| Gate | 结果 | 备注 |
|---|---|---|
| 环境审计 | ✅ | cu129/cu130 残留扫描 OK；`nvidia-cudnn-cu12/cublas-cu12` 未安装属预期（纯 cu13 栈） |
| preflight（eager） | ✅（重跑后） | 首次失败为 vLLM 内存 profiling 与 GPU0-1 上 vaopd-gkd-qwen35 进程竞争（`Initial free 74.79 GiB, current free 130.17 GiB`），非栈问题；释放 GPU 后通过 |
| preflight（CUDA graphs，`PREFLIGHT_EAGER=0`） | ✅ | torch.compile 53.7s + CUDA graph capture 完成；**vLLM #50445 的 Qwen3.5 GDN warmup 崩溃未复现** |
| NCCL 4-rank smoke | ❌ → workaround | NCCL 2.29.7 默认启用 NVLink SHARP（NVLS），本实例 multicast 绑定失败（`CUDA error 401`，疑似 Fabric Manager/NVSwitch 配置，host `opd-lzy-lowp-*`）；**`NCCL_NVLS_ENABLE=0` 后应可跑**（待复测） |
| 训练 smoke 配置干跑（hydra `--cfg job`） | ✅ EXIT=0 | 首次发现构建脚本 sed 补丁 bug：`use_task_rewards` 变量定义被跳过（grep 误匹配 EXTRA 行）+ `agent.num_workers` 重复插入 3 次 → `unbound variable`；已改为幂等 python 补丁并修复 backend 文件；`--cfg job` 全量 override 解析通过 |
| 训练 smoke（首次真实运行） | ❌ → 已修复 | 09:16 日志（`qwen3_5-35b-fsdp2-20260816_091611.log`）完整 traceback：`RayTaskError(IndentationError)`，发生在 `reward loop manager initialized` 之后、teacher/rollout vLLM server 初始化时。根因：移植的 `verl/.../vllm_async_server.py` diagnostics wrap 缩进错误（`try:` 与 body 同缩进），CPU import gate 不加载该模块所以未暴露。已修复并 `py_compile` 通过；构建脚本已改为"修复/保证"幂等模式（每次重建自动重排 wrap + 编译校验），旧 cu128 栈文件编译正常（确认是新移植引入） |
| 训练 smoke（第二次真实运行） | ❌ transfer_queue 卡死 | 14:11 运行（`qwen3_5-35b-fsdp2-20260816_141102.log`，32min）：IndentationError 已过、actor/ref/reward/teacher PG 全部初始化成功，但 verl 0.9.0 **V1 trainer 强制 transfer_queue**（`SimpleStorage` 8 个存储单元）在共享 GPU 节点上 actor 创建卡 ~27min，最后 `TransferQueueStorageUnit#7` worker 非 OOM 死亡（`Worker connection closed unexpectedly`）。CPU 节点同版本同配置复现：8/8 存储单元 2 秒创建成功 → 平台/负载相关而非包 bug。`tq.init` 无 enable 开关，V1 内无法旁路 |
| 训练 smoke（V0 旁路） | ✅ 配置验证 EXIT=0 | 方案：`trainer.use_v1=False` 走 V0 trainer（与已验证的 qwen35 cu129 旧栈一致，不碰 transfer_queue）；hydra 干跑通过。AB 对照用 V0 才公平（旧 baseline 就是 V0）；V1 + delta_sharded 留作后续独立 candidate |
| 训练 smoke（V0，最终） | ✅ PASS | 16:10 运行（`qwen3_5-35b-fsdp2-20260816_161010.log`）：**3/3 步完成**，checkpoint 落盘 `global_step_3`（含 actor 权重），尾部验证跑完、driver 干净退出。指标：step 耗时 77.3/36.3/40.7s（首步含 JIT warmup），吞吐 46.7/99.6/90.4 tok/s；actor/entropy 0.63→0.67，distillation/abs_loss 0.011→0.015，k1 loss≈0（自蒸馏符合预期），grad_norm 0.14→0.55，GPU 峰值 33.5GB/卡。尾部 `DataLoader worker Killed` = 已知 teardown 无害噪音（与 cu129 栈一致） |
| 训练 smoke | ⏳ 待重跑 | launcher 默认 `NCCL_NVLS_ENABLE=0`；`ATTENTION_IMPL` 已改数组传参；ray/vLLM 日志统一进 tee 日志 |

> 备注：2026-08-16 06:52 的 `logs/qwen3_5-35b-fsdp2-20260816_065226.log` 其实是
> 构建侧 hydra `--cfg job` 干跑经 run 脚本 tee 出来的（只有 config dump，非真实训练）。
> 真实训练前还发现一个**必炸的雷**：verl 0.9.0 的 V1 trainer 无条件
> `import transfer_queue`，该包不在 PyPI、无 dist-info；已验证的 qwen35 cu128 栈
> 以裸目录形式带 0.1.8。已从 `va-opd-qwen35-cu128` 复制同版本到新 env
> （`site-packages/transfer_queue`，`__version__=0.1.8`），构建脚本已加幂等步骤。

后续：NCCL 复测 + 训练 4 步 smoke 通过后升 `active`；NVLS 是否可恢复（Fabric
Manager 服务）留作平台侧排查项，不影响主线（NVLink P2P 18-link 全互联仍可用）。

## 6. 风险清单

- **vLLM #50445**：Qwen3.5 GDN 模型在 torch 2.13 CUDA-graph warmup 崩溃（open
  issue）——第一道 gate 必须先验证 CUDA-graph 路径；preflight 用
  `disable_cuda_graphs=True` 先确认栈本身可用，再测 graphs。
- torch 2.13 下 vLLM 启动 OOM warning 刷屏（#193195，无害）。
- verl 0.9.0 breaking：V1 trainer 默认化、checkpoint 参数 `trainer→actor_wg` 改名、
  vLLM <0.18 兼容移除——启动脚本参数需按新 schema 核对。
- cu130 wheel 与 driver CUDA 13.2 实机兼容需 smoke 验证（理论 MVC 兼容）。
- transformers 5.12.0 × vLLM 0.27.1 的 qwen3_5 模型注册需 import gate 验证；
  若 5.12.0 不满足 vLLM 0.27 的 GDN 代码路径，切 5.14.1 重验。
- NCCL 2.29.7（torch 自带）为初版；NCCL 2.30 单独 candidate 隔离验证。
- flashinfer 0.6.16.post3 首次 JIT 需要 nvcc，GPU 节点无工具链时 JIT 缓存
  必须在 CPU 侧预热或首次在工具链可及的环境跑。

## 7. 性能提升预期

- vLLM 0.27 针对 Qwen3.5：RMSNorm+all-reduce 融合（#46998）、MoE reduce-scatter
  （#47006）、per-KV-group attention backend、KV cache 布局重构。
- verl 0.9.0 delta_sharded：打中 27B→4B 每步 teacher 权重同步墙钟
  （当前 ~63.5s/it 的显著组成部分）；NV18 全互联拓扑利好。
- 配置级：CUDA graphs、更大 batch、fa2 vs fa3 decode 实测对比。

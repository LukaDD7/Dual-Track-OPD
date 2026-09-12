# VA-OPD HPC 操作说明书

适用目标：在共享存储、CPU 构建实例和 H200 GPU 实例上，运行 Qwen3-VL-4B student / Qwen3-VL-32B teacher 的 native verl OPD baseline 与 VA-OPD。

本手册按“CPU 实例准备一次，GPU 实例只消费产物”的方式编写。CPU 实例上的 Claude Code 应逐条执行并保存输出，不应自行替换 backend、升级包、把 K 改成 1、改 prompt，或临时使用 `/usr/bin/nvcc`。

## 角色边界与反馈闭环

当前本机 Codex 看不到 CPU/GPU 实例，CPU 实例 Claude Code 也可能看不到 GPU。按以下闭环协作，不把“无法直接观察”误写成“已经验证”：

1. 本机 Codex 维护公式、代码、tests、canonical commands 和判定标准；
2. CPU CC 同步代码，构建共享 environment/backend/data，运行 CPU-safe preflight；
3. CPU CC 从本手册选择下一条 GPU 命令，连同预期输出和停止条件交给用户；
4. 用户在真实 GPU allocation 原样执行并把完整输出路径/错误反馈给 CPU CC；
5. CPU CC 将事实整理到 `docs/va_opd_server_readiness_template.md` 的副本中，但 raw logs 留在 Git 外；
6. 只有上一 gate 的证据满足标准，CPU CC 才下发下一条命令；遇到偏差先记录，不自行改版本、GPU 拓扑或目标函数；
7. 用户把 readiness summary、关键日志片段和 run IDs 回传本机 Codex，必要时再调整代码。

GPU 第一次分配后，用户应先运行只读事实收集，不直接训练：

```bash
cd "$VA_OPD_PROJECT"
bash scripts/hpc/collect_va_opd_gpu_facts.sh
```

脚本将报告写到：

```text
$DTOPD_ROOT/fc-opd-storage/diagnostics/va_opd_gpu_facts/<timestamp>/gpu_facts.txt
```

将该路径和完整文件交给 CPU CC。报告包含 GPU/driver/MIG/topology、空闲显存和 PID、共享路径、runtime packages、torch CUDA/NCCL、backend identity、当前 nvcc 来源、Ray 状态等。它不执行训练，也不修改环境。不要把包含 hostname、GPU UUID、用户名或进程信息的 raw report 提交到 Git。

CPU CC 需要维护的 Git-safe readiness summary 模板位于 `docs/va_opd_server_readiness_template.md`。目前尚缺、必须由真实服务器反馈确认的信息包括：

- 实际 scheduler 分配的 GPU 数量、物理/逻辑 ID 和 `CUDA_VISIBLE_DEVICES` 映射；
- GPU 型号、每卡显存、driver、MIG、NVLink/PCIe topology；
- 现有 compute PID 的归属及卡是否真正空闲；
- GPU 节点能否读取 CPU 生成的 environment、backend、模型、parquet 和图像资产；
- torch 所见 CUDA/NCCL 与锁定版本是否一致；
- 独立 conda CUDA 12.8 toolchain 是否可读，JIT 是否错误落到系统 nvcc；
- 历史 all-gather size 在本次 4-rank allocation 上是否通过；
- native OPD/VA smoke 的 loss、gradient、entropy、length、VA 和 group metrics；
- CPU RSS、GPU memory、NCCL/Xid/worker error trajectory；
- 50-step pilot 的最佳验证 step，作为 full run 的 early-checkpoint 规则。

## 0. 不可变规则

执行前先读完本节：

1. 项目代码分支必须包含 `docs/va_opd_root_cause_and_recovery_20260719.md` 和 `scripts/hpc/run_va_opd_native.sh`。
2. native verl backend 必须是 commit `e003163181731412595257a72ec173071efb125f`。
3. 不修改 `third_party/verl` gitlink；backend 放在共享存储的独立目录。
4. 不在 backend 里手写研究公式；只应用仓库里的最小 patch。
5. Python 环境必须是独立 prefix，不复用旧 `vaopd-gkd-cu128` 环境。
6. CUDA 编译工具链必须是独立 conda prefix；不要用 `/usr/bin/nvcc`、`/usr/local/cuda/bin/nvcc`，也不要为了过编译去改系统 symlink。
7. 不把 `/usr/local/cuda/lib64` 注入新的 environment。
8. actor 必须是 4-rank FSDP2。默认 6 张可见 GPU 中，其余 2 张是独立 teacher pool；绝不把 actor 改成 6 ranks。
9. `rollout.n` 必须是 4。baseline 也用 4，才能公平比较。
10. parameter offload 和 optimizer offload 都必须关闭。
11. response length 是 2048；不要退回 1024。
12. 先 OPD smoke，再 VA smoke，再成对 pilot。任一 gate 失败都不跑 full。
13. 不把权重、parquet、原始 rollout、checkpoint 或 `.env` 放进 Git。
14. keepalive 只允许在训练成功、Ray/teacher 已清理后显式启动。

## 1. 路径约定

以下路径是当前集群的默认值。若挂载点变化，只覆盖列出的环境变量，不编辑脚本：

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export VA_OPD_PROJECT="$DTOPD_ROOT/Dual-track OPD"
export VA_OPD_ENV_PREFIX="$DTOPD_ROOT/envs/va-opd-native-e003-cu128-r595-v1"
export VERL_VA_OPD_DIR="$DTOPD_ROOT/fc-opd-storage/backends/verl-va-opd-e0031631-clean"
export VA_OPD_CUDA_TOOLCHAIN="$DTOPD_ROOT/envs/cuda128-toolchain"
export VA_OPD_VLLM_SOURCE="$DTOPD_ROOT/fc-opd-storage/backends/vllm-va-opd-v0120"
export VA_OPD_WHEELHOUSE="$DTOPD_ROOT/fc-opd-storage/wheelhouse/va-opd-cu128-r595-v1"
export VA_OPD_STUDENT_MODEL="$DTOPD_ROOT/models/Qwen3-VL-4B-Instruct"
export VA_OPD_TEACHER_MODEL="$DTOPD_ROOT/models/Qwen3-VL-32B-Instruct"
export GEOMETRY3K_SOURCE="$DTOPD_ROOT/dataset/geometry3k/data/train-00000-of-00001.parquet"
export GEOMETRY3K_VA_OPD_DATA_DIR="$DTOPD_ROOT/fc-opd-storage/outputs/fc_opd/geometry3k_gkd"
```

以上是 2026-07-23 起的统一目录。历史环境路径和构建/证据日期见
`docs/environment_registry.md`。不要直接移动旧 Conda prefix；需要复用时按锁文件
在统一目录重建。

每个新 shell 都重新 export。不要假设上一次 shell 的变量仍存在。

快速检查变量没有空值或拼错：

```bash
for value in \
  "$DTOPD_ROOT" \
  "$VA_OPD_PROJECT" \
  "$VA_OPD_ENV_PREFIX" \
  "$VERL_VA_OPD_DIR" \
  "$VA_OPD_CUDA_TOOLCHAIN" \
  "$VA_OPD_VLLM_SOURCE" \
  "$VA_OPD_WHEELHOUSE" \
  "$VA_OPD_STUDENT_MODEL" \
  "$VA_OPD_TEACHER_MODEL" \
  "$GEOMETRY3K_SOURCE"; do
  test -n "$value" || exit 1
  printf '%s\n' "$value"
done
```

## 2. CPU 实例：代码同步

### 2.1 已有 checkout

进入项目后先检查，不要在 dirty tree 上盲目 pull：

```bash
cd "$VA_OPD_PROJECT"
git branch --show-current
git status --short --ignore-submodules=all
git log -1 --oneline
```

预期分支是交付 VA-OPD 改动的分支。若有未知修改，停止并报告文件列表；不要 `git reset --hard`，不要覆盖他人的工作。

确认干净后再同步指定远程分支：

```bash
git fetch origin
git pull --ff-only
```

确认关键文件存在：

```bash
test -f scripts/hpc/run_va_opd_native.sh
test -f scripts/hpc/setup_va_opd_native_env.sh
test -f scripts/hpc/preflight_va_opd_native.py
test -f scripts/hpc/finalize_va_opd_native_run.py
test -f scripts/hpc/smoke_va_opd_nccl.py
test -f patches/verl/va_opd_native_e0031631.patch
test -f configs/environment/verl_va_opd_e003_cu128.constraints.txt
```

### 2.2 新 checkout

只有目标目录不存在时才 clone。仓库 URL 用项目实际 remote，不要猜：

```bash
mkdir -p "$DTOPD_ROOT"
git clone <PROJECT_GIT_URL> "$VA_OPD_PROJECT"
cd "$VA_OPD_PROJECT"
git switch <VA_OPD_DELIVERY_BRANCH>
```

`<...>` 是人工必须替换的占位符，不能原样执行。

## 3. CPU 实例：基础资源检查

CPU 节点无需 GPU，但建议至少满足：

- 64 GB RAM；
- 150 GB 共享存储余量；
- `conda` 或 `micromamba`；
- 能访问 GitHub/PyPI/conda channels；
- 模型和数据路径通过共享挂载可读。

执行：

```bash
df -h "$DTOPD_ROOT"
command -v conda || command -v micromamba
test -r "$GEOMETRY3K_SOURCE"
test -r "$VA_OPD_STUDENT_MODEL/config.json"
test -r "$VA_OPD_TEACHER_MODEL/config.json"
```

检查系统 nvcc 只用于识别风险，不用于构建：

```bash
command -v nvcc || true
```

如果输出 `/usr/...` 或 `/usr/local/...`，不要调用它。setup 会先 unset `CUDA_HOME CUDA_PATH NVCC`，然后建立独立 `VA_OPD_CUDA_TOOLCHAIN`。

## 4. CPU 实例：构建 backend 与 environment

### 4.1 正常构建

```bash
cd "$VA_OPD_PROJECT"
MAX_JOBS=16 bash scripts/hpc/setup_va_opd_native_env.sh \
  2>&1 | tee "$DTOPD_ROOT/fc-opd-storage/va_opd_env_build_$(date +%Y%m%d_%H%M%S).log"
```

脚本会：

1. 创建 Python 3.12 prefix；
2. clone/fetch exact verl commit；
3. 应用三文件 patch，并用 reverse check 验证；
4. 从官方 PyTorch cu128 index 安装 torch 2.9.0 family，并立即检查 `torch.version.cuda == 12.8`；
5. 创建/补全独立 conda CUDA 12.8 + GCC/G++ 12 compiler prefix；
6. clone 固定 vLLM 0.12.0 source commit，在临时 checkout 中构建 H200 SM90/cu128 wheel；
7. 把本地 vLLM wheel、exact verl requirements、Transformers 4.57.3 等作为一次 pip solve，先 `--dry-run` 再安装；
8. 强制用 source-built wheel 覆盖任何同版本发布 wheel，并构建 flash-attn 2.8.3；
9. editable-install backend 和项目，执行 `pip check`；
10. 写 environment build manifest（含 vLLM commit/wheel SHA-256/toolchain），import 并检查 `va_opd_k1` 已注册。

构建期间不要另开 pip/conda 修改同一 prefix。

### 4.2 CPU 内存或编译时间不足

先降低并发，继续使用同一个脚本：

```bash
MAX_JOBS=4 bash scripts/hpc/setup_va_opd_native_env.sh
```

不要设置旧的 `VA_OPD_SKIP_FLASH_ATTN_BUILD`；v2 pipeline 把 flash-attn 和 build manifest 作为正式 preflight gate。若 `MAX_JOBS=4` 仍 OOM，保存日志并申请更大 CPU RAM，不要把失败的半环境交给 GPU。

### 4.3 conda CUDA 包解析失败

不要切换到系统 nvcc。先记录完整 solver 输出，然后执行：

```bash
conda search -c nvidia cuda-toolkit=12.8
conda search -c conda-forge gcc_linux-64=12
conda search -c conda-forge gxx_linux-64=12
```

如果集群 mirror 暂时没有 exact build，优先让管理员恢复 `nvidia` channel 或把已验证的 `VA_OPD_CUDA_TOOLCHAIN` prefix 从另一 CPU 节点同步到共享存储。不要自行改为 12.9/13.0，因为 torch runtime 被锁到 12.8。

### 4.4 CPU CC 必须回传的构建信息

无论成功或失败，都不要只回复“装好了”或一张终端截图。把 build log 留在 Git 外，并向本机 Codex 回传以下脱敏文本：

```bash
git -C "$VA_OPD_PROJECT" rev-parse HEAD
git -C "$VERL_VA_OPD_DIR" rev-parse HEAD
git -C "$VA_OPD_VLLM_SOURCE" rev-parse HEAD
"$VA_OPD_ENV_PREFIX/bin/python" -m pip check
"$VA_OPD_ENV_PREFIX/bin/python" -m pip freeze | grep -E '^(torch|torchvision|torchaudio|vllm|transformers|flashinfer-python|flash-attn|ray|tensordict|numpy|nvidia-)'
"$VA_OPD_CUDA_TOOLCHAIN/bin/nvcc" --version
realpath "$VA_OPD_CUDA_TOOLCHAIN/bin/nvcc"
"$VA_OPD_ENV_PREFIX/bin/python" -m json.tool \
  "$VA_OPD_ENV_PREFIX/share/dual-track-opd/va_opd_environment_manifest.json"
```

失败时另外回传：失败命令上方至少 80 行和下方全部 traceback、build log 的完整路径、`df -h "$DTOPD_ROOT"`、`free -h`，以及是否为第一次运行或从中断后重跑。不要自行尝试以下“修复”：

- 不要对 vLLM/verl 用 `--no-deps` 绕过 resolver（setup 最后的受控 editable install 除外）；
- 不要把 Transformers 升到 5.x；
- 不要把 vLLM 升到 0.18.x；
- 不要换成 CUDA 12.9/13 wheel 或系统 `/usr` nvcc；
- 不要复用旧 `va-opd-verl-e003-cu128` prefix；
- 不要删除旧 prefix、wheelhouse 或构建日志；先报告，便于比较证据。

## 5. CPU 实例：构建后审计

```bash
"$VA_OPD_ENV_PREFIX/bin/python" - <<'PY'
import importlib.metadata as md
import torch

expected = {
    "torch": "2.9.0",
    "vllm": "0.12.0+cu128",
    "transformers": "4.57.3",
    "ray": "2.53.0",
    "tensordict": "0.10.0",
    "flashinfer-python": "0.5.3",
    "flash-attn": "2.8.3",
}
for package, wanted in expected.items():
    actual = md.version(package)
    print(f"{package}=={actual}")
    assert actual == wanted, (package, actual, wanted)
print("torch.version.cuda=", torch.version.cuda)
assert torch.version.cuda == "12.8"
PY
```

再检查 source-build provenance；缺少此文件即视为环境未完成：

```bash
ENV_MANIFEST="$VA_OPD_ENV_PREFIX/share/dual-track-opd/va_opd_environment_manifest.json"
test -s "$ENV_MANIFEST"
"$VA_OPD_ENV_PREFIX/bin/python" -m json.tool "$ENV_MANIFEST"
grep -F '4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e' "$ENV_MANIFEST"
grep -F 'cpu-source-build-cu128-h200-sm90' "$ENV_MANIFEST"
```

backend 审计：

```bash
test "$(git -C "$VERL_VA_OPD_DIR" rev-parse HEAD)" = e003163181731412595257a72ec173071efb125f
git -C "$VERL_VA_OPD_DIR" status --short --untracked-files=no
git -C "$VERL_VA_OPD_DIR" apply --reverse --check "$VA_OPD_PROJECT/patches/verl/va_opd_native_e0031631.patch"
```

status 只应显示：

```text
 M verl/experimental/agent_loop/agent_loop.py
 M verl/trainer/distillation/losses.py
 M verl/trainer/ppo/ray_trainer.py
```

任何第四个文件或 unrecognized edit 都停止。不要在现有目录继续“修一下”。安全做法是换一个新 backend 目录变量重新运行 setup。

确认 compiler 不是系统路径：

```bash
test -x "$VA_OPD_CUDA_TOOLCHAIN/bin/nvcc"
case "$(realpath "$VA_OPD_CUDA_TOOLCHAIN/bin/nvcc")" in
  /usr/*|/usr/local/*) exit 1 ;;
esac
"$VA_OPD_CUDA_TOOLCHAIN/bin/nvcc" --version
```

## 6. CPU 实例：数据准备与全量图像审计

launcher 发现 train/val 不存在时会自动生成。CPU 实例建议显式生成一次，以便保留 manifest 和提前发现坏图：

```bash
mkdir -p "$GEOMETRY3K_VA_OPD_DATA_DIR"
"$VA_OPD_ENV_PREFIX/bin/python" scripts/gen_geometry3k_parquet.py \
  --source "$GEOMETRY3K_SOURCE" \
  --output "$GEOMETRY3K_VA_OPD_DATA_DIR/train.parquet" \
  --val-output "$GEOMETRY3K_VA_OPD_DATA_DIR/val.parquet" \
  --asset-dir "$GEOMETRY3K_VA_OPD_DATA_DIR/assets" \
  --val-size 200 \
  --holdout-val
```

不要把 `assets/` 移走。parquet 内记录的是实际图像路径；移动后会在 preflight 失败。

执行 CPU-safe 全量 preflight：

```bash
bash scripts/hpc/run_va_opd_native.sh \
  --objective va_opd \
  --profile smoke \
  --preflight-only \
  --audit-all-images
```

预期末行：

```text
NATIVE VA-OPD PREFLIGHT PASSED
```

preflight 会在 Git 外的 run directory 写 `run_manifest.json`，其中包含：

- 项目 commit/dirty status；
- backend commit、patch hash、patched files；
- resolved environment versions；
- dataset parquet SHA-256；
- model config/tokenizer hashes；
- visible/actor/teacher GPU layout；
- output/checkpoint paths。

CPU 实例交付给 GPU 实例前保存以下输出：

```bash
git -C "$VA_OPD_PROJECT" rev-parse HEAD
git -C "$VERL_VA_OPD_DIR" rev-parse HEAD
sha256sum "$GEOMETRY3K_VA_OPD_DATA_DIR/train.parquet" "$GEOMETRY3K_VA_OPD_DATA_DIR/val.parquet"
"$VA_OPD_ENV_PREFIX/bin/python" -m pip freeze > "$DTOPD_ROOT/fc-opd-storage/va_opd_native_pip_freeze.txt"
```

## 7. CPU 到 GPU 的交接清单

如果是共享存储，只需核实 GPU 节点能读取同一路径；不要重新 pip install。

交接必须包括：

- project path 与 project commit；
- environment prefix；
- backend path 与 commit；
- CUDA toolchain prefix；
- student/teacher model paths；
- train/val paths与 hashes；
- CPU preflight run directory；
- environment build log；
- vLLM source commit、wheel SHA-256 与 environment manifest 路径；
- environment build log 中最后一次 `pip check` 和 `native VA-OPD environment: PASS`；
- flash-attn 已安装且未跳过。

GPU 操作者收到后先执行只读检查，不立即跑训练。

## 8. GPU 实例：节点与挂载检查

重新 export 第 1 节全部变量，然后：

```bash
cd "$VA_OPD_PROJECT"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h "$DTOPD_ROOT"
test -x "$VA_OPD_ENV_PREFIX/bin/python"
test -x "$VA_OPD_ENV_PREFIX/bin/torchrun"
test -r "$VA_OPD_STUDENT_MODEL/config.json"
test -r "$VA_OPD_TEACHER_MODEL/config.json"
test -r "$GEOMETRY3K_VA_OPD_DATA_DIR/train.parquet"
```

默认需要至少 6 张空闲 H200。检查 compute PID：

```bash
for gpu in 0 1 2 3 4 5; do
  printf 'GPU %s: ' "$gpu"
  nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true
done
```

有 PID 时不要 kill 不认识的进程，也不要使用 `--allow-busy-gpus` 逃过检查。先确认作业归属。

驱动必须支持 CUDA 12.8 runtime。运行 import smoke：

```bash
"$VA_OPD_ENV_PREFIX/bin/python" - <<'PY'
import torch
import vllm
import transformers
print("torch", torch.__version__, "runtime", torch.version.cuda)
print("visible", torch.cuda.device_count())
print("gpu0", torch.cuda.get_device_name(0))
assert torch.__version__.startswith("2.9.0")
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()
PY
```

检查当前 shell 的 nvcc：

```bash
command -v nvcc || true
```

训练本身通常不需要 nvcc。若 vLLM/FlashInfer JIT 需要编译，显式启用独立 toolchain：

```bash
export CUDA_HOME="$VA_OPD_CUDA_TOOLCHAIN"
export CUDA_PATH="$VA_OPD_CUDA_TOOLCHAIN"
export PATH="$VA_OPD_CUDA_TOOLCHAIN/bin:$PATH"
export LD_LIBRARY_PATH="$VA_OPD_CUDA_TOOLCHAIN/lib:$VA_OPD_CUDA_TOOLCHAIN/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
```

再次确认：

```bash
case "$(realpath "$(command -v nvcc)")" in
  /usr/*|/usr/local/*) echo 'FATAL system nvcc' >&2; exit 1 ;;
esac
```

不要使用 launcher 的 `--allow-system-nvcc`，除非负责人明确批准一个记录原因的诊断实验。

## 9. GPU Gate A：4-rank collective smoke

该 smoke 不编译 nccl-tests，也不依赖系统 CUDA toolkit。它用环境内 PyTorch 在 actor 的 4 张卡上执行历史事故大小的 BF16 all-gather 和 all-reduce：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,COLL \
"$VA_OPD_ENV_PREFIX/bin/torchrun" \
  --standalone \
  --nproc-per-node=4 \
  scripts/hpc/smoke_va_opd_nccl.py \
  --iterations 3 \
  2>&1 | tee "$DTOPD_ROOT/fc-opd-storage/va_opd_nccl_smoke_$(date +%Y%m%d_%H%M%S).log"
```

预期：两个 element size 都打印 PASS，最后为 `VA-OPD four-rank NCCL smoke: PASS`。

若 hang：

1. 等到 5 分钟 process-group timeout 收敛；
2. 保存日志和 `nvidia-smi topo -m`；
3. 停止，不进入训练；
4. 不先用 `NCCL_ALGO=Ring` 把问题掩盖掉；
5. 比对 driver、torch/NCCL 与 CPU manifest。

## 10. GPU Gate B：native OPD 3-step smoke

先证明最小的 full-image native RKL pipeline：

```bash
bash scripts/hpc/run_va_opd_native.sh \
  --objective opd \
  --profile smoke \
  --visible-gpus 0,1,2,3,4,5 \
  --actor-gpus 4 \
  --teacher-gpus 2 \
  --teacher-tp 2 \
  --name gate_b_opd
```

默认 smoke：prompt batch 4、K=4、3 updates。尽管 baseline 不使用 sibling weights，仍保持 K=4 以保证 rollout compute 和 VA experiment 可比。

成功标准：

- shell exit 0；
- `result.json` 的 `passed=true`；
- completed steps ≥ 3；
- finite distillation loss count > 0；
- finite gradient count > 0；
- finite entropy count > 0；
- `last_response_length_clip_ratio` 被记录；
- Ray 已由 launcher cleanup。

定位最新结果：

```bash
find "$DTOPD_ROOT/fc-opd-storage/runs/va_opd_native" -name result.json -type f -print | sort | tail
```

不要只看 trainer exit code；必须查看 `result.json` 和 `train.log`。

## 11. GPU Gate C：VA-OPD 3-step smoke

Gate B 完成后再运行：

```bash
bash scripts/hpc/run_va_opd_native.sh \
  --objective va_opd \
  --profile smoke \
  --visible-gpus 0,1,2,3,4,5 \
  --actor-gpus 4 \
  --teacher-gpus 2 \
  --teacher-tp 2 \
  --name gate_c_va
```

除 Gate B 标准外还必须满足：

- 有 `va_opd/mean`；
- 有 `va_opd/positive_ratio`；
- `va_opd/group_weight_sum_max_error <= 1e-5`；
- `va_opd/prompt_groups` 与 batch prompt 数一致；
- 没有 `teacher IDs ... not aligned`；
- 没有 sibling group-size error；
- degraded pass 没有图像尺寸错误。

若 `VA mean == 0` 但程序没有报错，也不要继续。抽查 dataset 的 full/degraded 图像是否真不同：

```bash
"$VA_OPD_ENV_PREFIX/bin/python" - <<'PY'
from pathlib import Path
import pandas as pd
from PIL import Image, ImageChops

data = Path("/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/geometry3k_gkd/train.parquet")
frame = pd.read_parquet(data)
for index in range(min(8, len(frame))):
    ci = frame.iloc[index]["condition_inputs"]
    if hasattr(ci, "tolist"):
        ci = ci.tolist()
    if not isinstance(ci, dict):
        ci = dict(ci)
    full = Image.open(ci["full_image"]["path"]).convert("RGB")
    degraded = Image.open(ci["degraded_image"]["path"]).convert("RGB")
    assert full.size == degraded.size
    print(index, full.size, ImageChops.difference(full, degraded).getbbox() is not None)
PY
```

所有自然含细节的图通常应打印 `True`。

## 12. 资源不足时的唯一推荐 fallback

若只有 5 张 H200 可用，保持 actor 4 ranks，把 teacher 改为单卡：

```bash
bash scripts/hpc/run_va_opd_native.sh \
  --objective va_opd \
  --profile smoke \
  --visible-gpus 0,1,2,3,4 \
  --actor-gpus 4 \
  --teacher-gpus 1 \
  --teacher-tp 1 \
  --name teacher_tp1_fallback
```

32B teacher 在 141 GB H200 上预计能放下，但更慢且 vLLM KV cache 余量更小。发生 teacher OOM 时，先在脚本的独立实验分支降低 teacher `gpu_memory_utilization` 或 `max_num_seqs`，不要减少 actor 到 3 ranks，也不要打开 actor CPU offload。

只有 4 张卡时，当前 pipeline 没有安全的同节点 full setup，因为 actor 与 32B teacher 需要隔离 resource pool。不要把 teacher 和 actor 强行 colocate；需要申请更多 GPU 或实现经过评审的跨节点 teacher service。

## 13. GPU Gate D：成对 50-step pilot

先 baseline：

```bash
bash scripts/hpc/run_va_opd_native.sh \
  --objective opd \
  --profile train \
  --steps 50 \
  --visible-gpus 0,1,2,3,4,5 \
  --actor-gpus 4 \
  --teacher-gpus 2 \
  --teacher-tp 2 \
  --name pilot50_opd
```

再 VA：

```bash
bash scripts/hpc/run_va_opd_native.sh \
  --objective va_opd \
  --profile train \
  --steps 50 \
  --visible-gpus 0,1,2,3,4,5 \
  --actor-gpus 4 \
  --teacher-gpus 2 \
  --teacher-tp 2 \
  --name pilot50_va
```

不要并行运行两个 pilot；它们会竞争同一组 GPU/Ray。

每 5–10 step 人工查看：

```bash
tail -n 200 <RUN_DIR>/train.log
```

重点字段：

- `training/global_step`；
- `distillation/loss`；
- `distillation/abs_loss`；
- `actor/grad_norm`；
- `actor/entropy`；
- `critic/score/mean`；
- `response_length/mean`；
- `response_length/clip_ratio`；
- `va_opd/mean`、`positive_ratio`、rollout min/max 和 group error。

立即停止条件：

- loss/grad/entropy 出现 NaN 或 Inf；
- entropy 快速降到接近 0，且输出变成高度重复；
- entropy 异常升高，同时 reward 下降、response 全部顶到 2048；
- clip ratio 持续接近 1；
- VA 长期全 0；
- group weight error > 1e-5；
- CPU RSS 持续大幅增长；
- 任一 NCCL watchdog、Xid、worker lost；
- 连续多个 validation point reward 明显退化。

中止用 scheduler 的正常 cancel 或当前 shell `Ctrl-C`，让 trap 执行 `ray stop -f`。不要直接 kill -9 controller，除非普通终止无响应且已保存诊断。

### pilot 通过标准

两个 run 都需完成 50 steps。然后人工形成一张对比表：

| 项目 | OPD | VA-OPD |
|---|---:|---:|
| best validation score |  |  |
| best step |  |  |
| step-50 score |  |  |
| entropy range |  |  |
| response clip-ratio range |  |  |
| loss range |  |  |
| grad range |  |  |
| VA mean/positive ratio | N/A |  |
| collapse/hang |  |  |

VA full run 的放行条件不是“VA loss 能跑”，而是：没有通信/数值/长度 collapse，且 validation trajectory 至少不比公平 baseline 显著差。若最佳点很早，full run 必须保持频繁 checkpoint，并按 validation 选择模型。

## 14. GPU Gate E：5-epoch 正式对比

确认 scheduler wall-time 和共享存储容量后，依次运行：

```bash
bash scripts/hpc/run_va_opd_native.sh \
  --objective opd \
  --profile train \
  --visible-gpus 0,1,2,3,4,5 \
  --actor-gpus 4 \
  --teacher-gpus 2 \
  --teacher-tp 2 \
  --name full5e_opd
```

```bash
bash scripts/hpc/run_va_opd_native.sh \
  --objective va_opd \
  --profile train \
  --visible-gpus 0,1,2,3,4,5 \
  --actor-gpus 4 \
  --teacher-gpus 2 \
  --teacher-tp 2 \
  --name full5e_va
```

`--profile train` 默认 prompt batch 16、5 epochs、K=4、test every 25、save every 50。不要加 `--steps`，否则会变成 step-limited pilot。

如果需要成功后 keepalive，只在负责人明确要求时给最后一个命令加：

```text
--keepalive-after-success
```

launcher 只会在 trainer 成功、finalizer 成功、Ray cleanup 后启动：

```bash
CUDA_VISIBLE_DEVICES=3,4 KEEPALIVE_TARGET_UTIL=0.45 KEEPALIVE_WORK_ITERS=32 \
python -u /inspire/hdd/global_user/mengweicheng-240108120092/lzy/scripts/busy_keepalive.py
```

失败或中断时不会启动。

## 15. 结果与复现材料

每个 run directory 结构：

```text
<run>/
  preflight.log
  run_manifest.json
  train.log
  result.json
  rollouts/
  validation/
  keepalive.log       # 仅显式启用且成功
  keepalive.pid       # 同上
```

checkpoint 在独立目录：

```text
$DTOPD_ROOT/fc-opd-storage/checkpoints/va_opd_native/<RUN_ID>/
```

最终实验记录必须包含：

- `run_manifest.json`；
- `result.json`；
- 选中的 checkpoint path；
- 为什么选择该 checkpoint（validation step/metric）；
- 原始 rollout 和 validation output 路径；
- baseline/VA 对比表；
- 任何 override；
- node name、GPU 型号、driver；
- 是否发生 retry；
- repo/backend dirty status。

这些 artifacts 留在 Git 外。Git 只收小型 summary 和必要说明。

## 16. 常见失败决策树

### A. backend patch check 失败

症状：`patch does not apply`、patched SHA mismatch。

处理：

1. `git -C "$VERL_VA_OPD_DIR" rev-parse HEAD`；
2. `git -C "$VERL_VA_OPD_DIR" status --short`；
3. 若非 exact commit 或多余修改，停止；
4. 指向一个新的空 backend path，重新运行 setup；
5. 不在旧目录 `git reset --hard`，除非负责人确认它没有别人的工作。

### B. `va_opd_k1` unsupported

说明 project editable install 或 patch registration 没加载。

```bash
export PYTHONPATH="$VA_OPD_PROJECT/src:$VERL_VA_OPD_DIR:${PYTHONPATH:-}"
"$VA_OPD_ENV_PREFIX/bin/python" - <<'PY'
from verl.trainer.distillation.losses import get_distillation_loss_settings
print(get_distillation_loss_settings("va_opd_k1"))
PY
```

仍失败则重新执行 environment setup，不要改 loss mode 为 `k1` 来假装 VA 已运行。

### C. teacher/student response ID mismatch

这是 hard failure，不应 mask：

- 确认两个 checkpoint vocab size 一致；
- 确认 student/teacher 都是 Qwen3-VL Instruct family；
- 确认同一 sequence IDs 被送入两次 teacher call；
- 确认 full/degraded image 尺寸一致；
- 检查 transformers/vLLM 版本是否 drift。

不要 trim IDs 或按字符串重新 tokenize；那会破坏 sampled-token objective。

### D. degraded image missing/size mismatch

重新执行数据生成并保持 asset 路径。不要运行时临时 resize，因为 transform 必须进入 manifest。

### E. GPU OOM

先区分 actor 还是 teacher：

- teacher OOM：降低 teacher `max_num_seqs`，必要时在 2 GPUs TP=2 下减 memory utilization；
- actor rollout OOM：先把 rollout memory utilization 从 0.55 小幅降低；
- actor train OOM：保持 4 ranks，降低 microbatch/token budget 或增加 checkpointing；
- 不优先打开 CPU offload；旧事故证明它会把问题转为 CPU RAM/swap/NCCL timeout；
- 不把 response length 直接退回 1024；先看 clip ratio 和实际 prompt length。

所有配置变化都要形成新 run name，并写入说明。

### F. NCCL hang

收集：

```bash
nvidia-smi topo -m
nvidia-smi -q -d ECC,ERROR
ps -ef | grep -E 'ray|torch|vllm' | grep -v grep
```

保存 train.log/NCCL log，确认 actor world size 是 4。不要因为总共可见 6 张卡就误认为 6-rank FSDP 是预期行为。

### G. CPU RAM 增长

确认 resolved config 中：

```text
actor_rollout_ref.actor.fsdp_config.param_offload=false
actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
```

若仍增长，记录每 step RSS、Ray object store、teacher server RSS，再定位，不要先加 swap。

### H. 训练结束但 finalizer 失败

finalizer 失败代表 run 不满足实验 gate。查看 `result.json.failures`。常见原因：

- requested steps 未完成；
- 没有 finite grad；
- entropy 没记录；
- VA metric 缺失；
- rollout group sum error 超限。

不要手改 `passed=true`。

## 17. Claude Code 交接模板

CPU 实例 Claude Code 完成后，应使用以下格式回复 GPU 操作者：

```text
VA-OPD CPU preparation status: PASS/FAIL
project path:
project commit:
project dirty status:
backend path:
backend commit:
backend changed files:
environment prefix:
CUDA toolchain prefix and nvcc path:
torch/vLLM/transformers/Ray/TensorDict versions:
student path and config hash:
teacher path and config hash:
train parquet path and sha256:
val parquet path and sha256:
CPU preflight run directory:
environment build log:
flash-attn built: yes/no
overrides used: none/list
remaining blocker: none/details
```

没有填完这些字段，不算完成交接。

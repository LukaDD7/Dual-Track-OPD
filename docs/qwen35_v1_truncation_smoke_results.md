# Qwen3.5 v1 对齐 + 截断消融结果报告（Step A + Step B）

> **2026-08-04 归因更正（必须先读）**：本文记录的 token 数、clip rate、boxed
> rate 和 accuracy 数值仍然有效，但“模型在训练采样下也不会自然收尾”“继续加长无效”
> 这两个因果结论已经撤销。Step A/B/C 的这些验证生成走的是 pinned verl
> `334d9f8b` 的 validation sampler，实际为 greedy decoding（`temperature=0`,
> `top_p=1`, `top_k=-1`）；训练 rollout 走的是非 greedy sampling。Qwen3.5 官方明确
> 警告 greedy decoding 可能导致 endless repetition。因此 2048/4096 均 96% 顶满首先
> 证明的是 **greedy validation contract 有问题**，不能外推为训练 rollout 的截断率。
> 权威纠正与下一步见 `docs/qwen35_training_environment_status_20260804.md`。

> 日期：2026-08-03 ｜ 分支：`codex/va-opd` ｜ 运行时代码：`8c353db`
> 依据：`docs/qwen35_v1_truncation_next_steps_for_claude.md`（`9fac2eb`）
> 约束遵守：未用 `CLEAN_START=1`；未跑 `ROLLOUT_N=4`；未跑超过 20 步的实验；
> 未触碰并行诊断实验的未提交改动（GPU 4-7 全程由对方占用）。

## 1. 一句话结论

**Step A（v1 task-reward integration smoke）通过**：v1 trainer 下任务分进入优化项
（3/20 步出现非 0 pg_loss），蒸馏/梯度全程有限，manifest `completed rc=0`。
**Step B（4096 val-only 截断消融）完成**：clip rate 在 4096 下仍为 **96%**（与 2048 相同），
模型按预算长度续写，单纯加长不能解决截断——按文档决策规则，下一步应做
**2048 下的精简收尾指令消融**（独立版本），而不是继续加长。
**Step C（boxed_only 精简收尾指令，2048 val-only）完成**：clip rate **96%→85.5%**，
boxed 率 0.10→0.25、acc 0.02→0.035；未截断的 29 个样本 boxed=100%、acc=17.2%，
但 85.5% 仍顶满预算——指令被遵守时有效，模型仍不主动收尾。clip 仍 >70%，
下一步（更严格变体或长度策略）按决策规则交 codex 定夺。

## 2. 可复现信息

| 项 | 值 |
|---|---|
| 项目 repo | `Dual-Track-OPD` @ `8c353db`（codex/va-opd；工作树含既有诊断实验未提交改动，本线未触碰） |
| verl 后端 | `repos/verl-cu130-vllm` @ `334d9f8b`（tracked dirty：`run_qwen3_5_4b_fsdp.sh`、`vllm_async_server.py`；diff sha256 `953c8a07…`） |
| 训练器 | v1（`TRAINER_USE_V1=True`，启动日志 `TaskRunnerV1`） |
| 环境 | conda `va-opd-qwen35-cu128`：torch 2.11.0+cu129 / vllm 0.23.0+cu129 / transformers 5.12.0 / flash_attn 2.8.3 / ray 2.55.1 / verl 0.9.0.dev0 |
| GPU | NVIDIA H200 ×8（driver 570.124.06）；两轮均 `FORMAL_GPUS=0,1,2,3`（3 actor + 1 teacher） |
| 数据 | train sha256 `63d56af2c3e5bedafc783cab6bdcc1c2f30c0e2b06aea859238df8ed89499a61`；val sha256 `87d7f0411767696fd51c29f6dfcf01d6e28a811f642dc0063479513b8f052943` |
| 模型 | 学生 `models/Qwen3.5-4B`（config sha256 `ddc63e1c…`）；老师 `models/qwen3.6-27B`（config sha256 `69db4eb7…`） |

## 3. Step A：v1 task-reward integration smoke（20 步，`USE_TASK_REWARDS=True`）

### 3.1 命令与产物

- 启动：`bash scripts/hpc/run_qwen35_v1_stepa.sh`（内含 Step A 全部 env）
- metadata：`fc-opd-storage/logs/qwen35_runs/k1_reward_v1_smoke_r2/`
  - `run_manifest.json`（`status: completed`，`returncode: 0`）
  - `resolved_launch_config.json`（`trainer_use_v1: True`；launch_config sha256 `…`）
  - `hydra/.hydra/config.yaml`（`trainer.use_v1: true`）、`overrides.yaml`、`hydra.yaml`
  - `train.log`
- val dump：`fc-opd-storage/logs/val_dump_k1_reward_v1_smoke/{0,5,10,15,20}.jsonl`
- 日志：`artifacts/fc_opd/nohup_v1_reward_smoke_r2_20260803_144724.log`
- 墙钟：31m56s（14:47:34 → 15:19:30）；actor 峰值显存 ~28.3 GB/卡

### 3.2 证据

- 启动选择 `TaskRunnerV1`；`Using dataset class: FCOPDDataset`；`Total training steps: 20`
- **3/20 步训练 batch 含正确样本并产生任务梯度**：

| global_step | critic/score/mean | rewards/max | advantages/max | actor/pg_loss |
|---:|---:|---:|---:|---:|
| 5 | 0.075 | 0.9 | 0.9 | **-0.0612** |
| 14 | 0.075 | 0.9 | 0.9 | **-0.0395** |
| 17 | 0.038 | 0.9 | 0.9 | **-0.0182** |

- 其余 17 步 `score=0.0`（24 样本 batch 内无正确样本，`(1-0.02)^24≈0.6` 的稀疏抽样），pg_loss 相应为 0；不构成失败
- 全程有限：distillation loss 0.068-0.187、entropy 0.32-0.70、grad_norm 1.5-5.8；`pg_clipfrac=0`（按文档不要求）
- step 0 验证：`val reward/mean@1=0.018`，`val acc/mean@1=0.018`（路由 `hiyouga/geometry3k`）

判定：**通过**（文档 §3 的 7 项证据全部满足）。

## 4. Step B：4096-token validation-only 截断消融

### 4.1 命令与产物

- 启动：`bash scripts/hpc/run_qwen35_v1_stepb.sh`（`MAX_RESPONSE_LENGTH=4096`、
  `trainer.val_only=True`、`TOTAL_TRAINING_STEPS=1`，其余与实验 #2 step 0 配置一致）
- 有效 max_model_len = `1024 + 4096 + 1 = 5121`（preflight/启动打印确认）
- metadata：`fc-opd-storage/logs/qwen35_runs/k1_promptfix_r4096_valonly/`（`completed rc=0`）
- val dump：`fc-opd-storage/logs/val_dump_k1_promptfix_r4096_valonly/0.jsonl`
- 日志：`artifacts/fc_opd/nohup_v1_r4096_valonly_20260803_153201.log`
- 墙钟：11m29s（15:32:12 → 15:43:42，含引擎启动 + 200 样本 4096 生成）；无 OOM/vLLM/teacher 错误

### 4.2 200 样本对比（与实验 #2 step 0 @ 2048）

token 长度用学生 tokenizer（Qwen3.5-4B）对生成部分计数：

| 指标 | @2048（实验 #2 step0） | @4096（Step B） |
|---|---:|---:|
| token 长度 mean / median / p90 / max | 2025 / 2048 / 2048 / 2048 | 3991 / 4096 / 4096 / 4096 |
| **clip rate（顶满上限比例）** | **0.960** | **0.960** |
| boxed_rate | 0.100 | 0.125 |
| format_rate | 0.000 | 0.000 |
| accuracy_rate | 0.020（4/200） | 0.025（5/200） |
| reward mean / std | 0.018 / 0.126 | 0.0225 / 0.140 |
| 分数计数 | 196×0.0 + 4×0.9 | 195×0.0 + 5×0.9 |
| 答对样本中被截断的比例 | 1/4 | 0/5 |

### 4.3 解读

1. **clip rate 在两个上限下都是 96%**：模型按可用预算续写推理，不会自然收尾；4096 只是把
   "截断点"后移，没有降低截断比例。
2. 加长确实让已完成的答案不被截断（4096 下 5/5 答对样本都写完了，2048 下 1/4 被截断），
   boxed/acc 小幅改善（0.10→0.125、0.02→0.025），但成本是生成时长和显存预算翻倍。
3. format_rate 仍为 0：模型输出无字面 `<think>...</think>`，符合既有结论。
4. 按文档 §4 决策规则：**clip rate > 70% → 不盲目继续加长**，下一步做
   **2048 下、独立版本的"精简收尾指令"消融**（如要求最后一行只给 `\boxed{答案}`），
   不动共享 scorer，一次只改一个变量。

## 5. 对下一步的推荐（供 codex 决策）

1. **响应长度**：保持 2048 作为训练口径，先跑精简指令消融；只有当精简指令能显著压低
   clip rate（<30%）且 boxed/acc 不退化时才考虑 4096 作为训练口径。
2. **ROLLOUT_N=4 的 24 序列问题**：应把 prompt batch 从 24 降到 **6**（6 prompts × 4 = 24
   有效序列），`PPO_MINI_BATCH_SIZE=24`（序列口径）不变。注意当前 wrapper 的
   `TRAIN_BATCH_SIZE % ROLLOUT_NUM_WORKERS == 0` 校验是按 n=1 写的（6 不被 8 整除），
   n>1 时需要重写该校验/对齐 rollout worker 数（如 `ROLLOUT_NUM_WORKERS=6`），并重算显存。
3. **manifest 已自动落盘**（两轮均在 `fc-opd-storage/logs/qwen35_runs/` 下），包含完整
   SHA256、后端 dirty/diff、Hydra config、launch config、train.log；raw dump/日志未进 Git。

## 6. 过程备注

- **v1 首次启动失败**：`ModuleNotFoundError: No module named 'transfer_queue'`（v1 trainer
  依赖 TransferQueue，`va-opd-qwen35-cu128` env 缺失）。已把纯 Python 的
  `transfer_queue 0.1.8` 从 `verl-cu130-vllm` env 复制进该 env（只新增包，不影响其它依赖），
  复跑后 v1 正常起步。
- 复跑时因终端折行导致 `RUN_METADATA_DIR` 值被截断两次（`command not found`），已新增
  `scripts/hpc/run_qwen35_v1_stepa.sh` / `run_qwen35_v1_stepb.sh` launcher（`8c353db`）
  避免贴长命令。
- 两次运行均未 `CLEAN_START`；GPU 4-7 的并行诊断实验全程未受影响。

## 7. Step C：boxed_only 精简收尾指令消融（2048 val-only）

### 7.1 动机与命令

Step B 结论是"加长只把截断点后移"，因此按 §4 决策规则做 **2048 下独立版本的精简
收尾指令消融**：prompt 从 `Put the final answer in \boxed{}.` 改为
`Briefly reason. Then output your final answer as exactly one line: \boxed{<answer>}.
Stop immediately after that line; write nothing else.`。一次只改 prompt 一个变量，
共享 scorer 未动，长度保持 2048。

```bash
bash scripts/hpc/run_qwen35_v1_promptboxed_valonly.sh
```

- metadata：`fc-opd-storage/logs/qwen35_runs/k1_promptfix_boxedonly_r4/`
  （`run_manifest.json`：`completed rc=0`；`hydra/.hydra/config.yaml` 含
  `data.prompt_version: boxed_only`）
- val dump：`fc-opd-storage/logs/val_dump_k1_promptfix_boxedonly_r4/0.jsonl`
- 日志：`artifacts/fc_opd/nohup_v1_pvboxed_only_valonly_20260804_022233.log`
- 墙钟：**8m36s**（02:22:34→02:31:10 UTC）；无 OOM/vLLM/teacher 错误；GPU 0-3
  （4-7 诊断实验未受影响）
- 运行时代码：repo `316d559`（工作树 dirty，diff_sha256 `4856283d…`；r4 修复
  随本报告提交）；后端 verl-cu130-vllm @ `334d9f8b`（tracked dirty，同前）

### 7.2 200 样本对比（口径与 §4.2 一致：学生 tokenizer 数生成部分，verl geo3k scorer）

| 指标 | exp#2 @2048（v1） | Step B @4096（v1） | **Step C @2048（boxed_only）** |
|---|---:|---:|---:|
| token 长度 mean / median / p90 / max | 2025 / 2048 / 2048 / 2048 | 3991 / 4096 / 4096 / 4096 | **1920 / 2048 / 2048 / 2048** |
| **clip rate（顶满预算比例）** | **0.960** | **0.960** | **0.855（171/200）** |
| boxed_rate | 0.100 | 0.125 | **0.250（50/200）** |
| format_rate | 0.000 | 0.000 | 0.000 |
| accuracy_rate | 0.020（4/200） | 0.025（5/200） | **0.035（7/200）** |
| reward mean / std | 0.018 / 0.126 | 0.0225 / 0.140 | 0.0315 / 0.166 |
| 分数计数 | 196×0.0 + 4×0.9 | 195×0.0 + 5×0.9 | 193×0.0 + 7×0.9 |

按截断/未截断分组：

| 组 | n | boxed_rate | acc |
|---|---:|---:|---:|
| 截断（≥2048 tokens） | 171 | 0.123 | 0.012（2/171） |
| 未截断（<2048 tokens） | 29 | **1.000** | **0.172（5/29）** |

### 7.3 解读

1. **指令有效但收尾仍不可靠**：29 个未截断样本 100% 带 `\boxed{}`、acc 17.2%；
   而 171 个截断样本 boxed 只有 12.3%、acc 1.2%——截断是 acc 的主要杀手。
   与 Step B 对比，boxed_only 让"写完了"的样本几乎全部遵守格式（Step B 未截断
   样本中 boxed 非 100%）。
2. **clip rate 仍 85.5%**（>70%）：模型多数时候写完 `\boxed{}` 后继续推理（截断组
   内 boxed 样本即属此类），"stop immediately" 指令压不住续写。按 §4 决策规则
   **不继续加长**；下一步选项交 codex：
   - 更严格的"只答不推理"变体（prompt 明令不输出推理，直接 `\boxed{}`），仍是
     一次只改 prompt 一个变量；
   - 或接受"生成预算内不保证收尾"的事实，调整训练/评估口径（如对截断样本用
     更长预算重生成，或把收尾行为作为训练目标）。
3. format_rate 仍为 0（无字面 `<think>`），共享 scorer 未动。

### 7.4 过程备注（r1-r4）

- r1：Hydra struct 拒绝新增键 `data.prompt_version` → 改 `+data.prompt_version=...`
  追加并补测试（`e04a702`）。
- r2：`_clean_prompts` 静态方法内引用 `self` → 改实例方法（`316d559`）。
- r3：实例属性 `prompt_version` 未赋值 → 改为显式传参的静态方法
  `_clean_prompts(dataframe, prompt_version)`，不依赖实例状态；未知版本在
  `__init__` fail fast。
- 各轮 failed manifest 分别保留在 `k1_promptfix_boxedonly{, _r2, _r3}/`；
  r4 为 completed。

## 8. Step D1：采样 validation（boxed_only @2048）

> 依据：`docs/qwen35_stepc_codex_decisions_and_next_run.md`（`3960527`）。
> 归因纠正：Step A/B/C 走 verl validation 路径，pinned 后端 `val_kwargs` 默认
> greedy（`temperature=0, top_p=1.0, top_k=-1, do_sample=false`），训练 rollout
> 是采样；因此 A/B/C 的 clip rate 不构成训练采样口径。D1 用训练采样参数复测。

### 8.1 命令与产物

```bash
bash scripts/hpc/run_qwen35_v1_boxedonly_sampled_valonly.sh
```

即 Step C r4 配置 + `val_kwargs.do_sample=True, temperature=1.0, top_p=0.95,
top_k=-1`（与训练 rollout 采样器一致）；prompt 仍为 boxed_only、cap 2048。

- metadata：`fc-opd-storage/logs/qwen35_runs/k1_boxedonly_sampled_d1/`
  （`completed rc=0`；`resolved_launch_config.json` 含 val_kwargs 覆盖）
- val dump：`fc-opd-storage/logs/val_dump_k1_boxedonly_sampled_d1/0.jsonl`
- 日志：`artifacts/fc_opd/nohup_v1_boxedonly_sampled_d1_20260804_062414.log`
- 墙钟：**11m06s**（06:24:29→06:35:35 UTC）；无 OOM/报错；GPU 0-3，4-7 未受影响
- 运行时代码：repo `3960527`（tracked_dirty=True，diff 为并行诊断线既有改动）；
  后端 verl-cu130-vllm @ `334d9f8b`

### 8.2 结果（200 样本，口径同前）

| 指标 | Step C r4（greedy @2048） | **D1（sampled @2048）** |
|---|---:|---:|
| token 长度 mean / p50 / p90 / p95 / p99 / max | 1920 / 2048 / 2048 / — / — / 2048 | 2020 / 2048 / 2048 / 2048 / 2048 / 2048 |
| **clip rate（==2048）** | **0.855** | **0.945（189/200）** |
| EOS 率（<2048 自然结束） | 0.145 | **0.055（11/200）** |
| boxed_rate | 0.250 | 0.125（25/200） |
| format_rate | 0.000 | 0.000 |
| accuracy_rate | 0.035 | 0.015（3/200） |
| reward mean / std | 0.0315 / 0.166 | 0.0135 / 0.110 |

重复度诊断（token 级 4-gram 重复比 / 最长重复跨度）：

| 指标 | Step C r4（greedy） | D1（sampled） |
|---|---:|---:|
| rep4 ratio mean / median / p90 | 0.584 / 0.565 / 0.846 | **0.340 / 0.346 / 0.400** |
| 最长重复跨度 mean / p90 / max（tokens） | 388 / 1289 / 1618 | **22 / 33 / 67** |
| rep4 ratio >0.05 的样本占比 | 1.000 | 1.000 |

分组：

| 组 | n | boxed_rate | acc |
|---|---:|---:|---:|
| EOS（<2048） | 11 | 0.727 | 0.182（2/11） |
| clip（==2048） | 189 | 0.090 | 0.005（1/189） |

### 8.3 解读与判定

1. **codex 的 greedy 归因方向被证实**：greedy 下最长重复跨度均值 388 tokens、
   最大 1618（响应在自我重复）；采样把重复跨度压到均值 22、最大 67，
   rep4 重复比也从 0.584 降到 0.340。A/B/C"顶满预算"里确实有大量 greedy 重复伪影。
2. **但采样不解决截断**：clip rate 94.5%（比 greedy 的 85.5% 还高），EOS 只有
   5.5%——模型在采样下写的是**长而低重复的推理**，仍然不主动收尾；boxed/acc
   反而低于 greedy（0.125/0.015 vs 0.250/0.035）。
3. **"写完即高质量"规律依旧成立**：EOS 组 boxed 72.7%、acc 18.2%，clip 组
   boxed 9.0%、acc 0.5%。
4. 按 codex go/no-go：D1 clip 94.5% >10% → **下一步跑 D1-L**（同采样器，
   cap 8192，同一 200 有序 prompts），测自然完成长度分布与长推理质量，
   不做 8192 训练口径。

## 9. Step D1-L：采样 validation @8192（自然完成长度分布）

### 9.1 命令与产物

```bash
bash scripts/hpc/run_qwen35_v1_boxedonly_sampled_r8192_valonly.sh
```

同 D1（boxed_only、`temp=1.0 top_p=0.95 top_k=-1`、同一 200 有序 prompts），
只改 cap：2048→8192。不构成 8192 训练口径。

- metadata：`fc-opd-storage/logs/qwen35_runs/k1_boxedonly_sampled_r8192_d1l/`
  （`completed rc=0`）
- val dump：`fc-opd-storage/logs/val_dump_k1_boxedonly_sampled_r8192_d1l/0.jsonl`
- 日志：`artifacts/fc_opd/nohup_v1_boxedonly_sampled_r8192_d1l_20260804_072104.log`
- 墙钟：**11m16s**（676s）；无 OOM/报错；GPU 0-3，4-7 未受影响
- 运行时代码：repo `1f4558c`（tracked_dirty=True）；后端 `334d9f8b`

### 9.2 结果（200 样本，口径同前）

| 指标 | Step C greedy @2048 | D1 sampled @2048 | **D1-L sampled @8192** |
|---|---:|---:|---:|
| token 长度 mean / p50 / p90 / p95 / p99 / max | 1920 / 2048 / 2048 / — / — / 2048 | 2020 / 2048 / 2048 / 2048 / 2048 / 2048 | 7709 / 8192 / 8192 / 8192 / 8192 / 8197 |
| **clip rate（顶满 cap）** | 0.855 | 0.945 | **0.830（166/200）** |
| EOS 率（<cap 自然结束） | 0.145 | 0.055 | **0.170（34/200）** |
| boxed_rate | 0.250 | 0.125 | 0.245（49/200） |
| format_rate | 0.000 | 0.000 | 0.000 |
| accuracy_rate | 0.035 | 0.015 | 0.025（5/200） |
| reward mean / std | 0.0315 / 0.166 | 0.0135 / 0.110 | 0.0225 / 0.141 |
| rep4 ratio mean / median | 0.584 / 0.565 | 0.340 / 0.346 | 0.483 / 0.491 |
| 最长重复跨度 mean / max（tokens） | 388 / 1618 | 22 / 67 | 35 / 70 |

自然完成（EOS）样本分布：

| 组 | n | 长度 min / p50 / p90 / max | 其中 ≤2048 | boxed_rate | acc |
|---|---:|---:|---:|---:|---:|
| EOS（<8192） | 34 | 646 / 5838 / 8191 / 8191 | 6/34 | 0.794 | 0.147（5/34） |
| clip（==8192） | 166 | — | — | 0.133 | 0.000（0/166） |

### 9.3 解读与判定

1. **自然推理本身就极长**：34 个自然结束的样本中位数 5838 tokens、p90 8191，
   只有 6 个能在 2048 内写完。2048 对自然思考远不够，加长只是把截断点后移。
2. **加长换来的质量增益极小**：acc 1.5%→2.5%（3→5 个正确，其中 2 个超过 2048），
   boxed 12.5%→24.5%；token 成本却翻 4 倍。EOS 组质量依旧远高于 clip 组
   （acc 0.147 vs 0.000）。
3. 采样下重复受控但随长度上升：rep4 0.34→0.48，最长重复跨度 22→35，
   仍远低于 greedy（388），codex 的 greedy 归因在重复维度成立。
4. **按 codex go/no-go：D1-L clip 83% >30%、增益不足以 justify 成本 →
   reject "just make it longer" → 下一步跑 D2**（`enable_thinking=False`，
   采样 @2048，prompt 仍为 boxed_only 保留简短可见推理；不是 answer_only）。

## 10. Step D2：硬非思考采样 validation（boxed_only @2048，enable_thinking=False）

### 10.1 命令与产物

```bash
bash scripts/hpc/run_qwen35_v1_boxedonly_nonthinking_sampled_valonly.sh
```

同 D1（boxed_only、`temp=1.0 top_p=0.95 top_k=-1`、cap 2048），唯一新增变量
`+data.apply_chat_template_kwargs.enable_thinking=False`；prompt 仍请求简短可见
推理，不是 answer_only。

- metadata：`fc-opd-storage/logs/qwen35_runs/k1_boxedonly_nonthinking_sampled_d2/`
  （`completed rc=0`；hydra config 含 `enable_thinking: false`）
- val dump：`fc-opd-storage/logs/val_dump_k1_boxedonly_nonthinking_sampled_d2/0.jsonl`
- 日志：`artifacts/fc_opd/nohup_v1_boxedonly_nonthinking_sampled_d2_20260804_075445.log`
- 墙钟：**7m24s**（444s）；无 OOM/报错；GPU 0-3，4-7 未受影响
- 运行时代码：repo `7cfe870`（tracked_dirty=True）；后端 `334d9f8b`

### 10.2 结果（200 样本，口径同前）

| 指标 | Step C greedy @2048 | D1 sampled @2048 | D1-L sampled @8192 | **D2 sampled @2048 non-thinking** |
|---|---:|---:|---:|---:|
| token 长度 mean / p50 / p90 / max | 1920 / 2048 / 2048 / 2048 | 2020 / 2048 / 2048 / 2048 | 7709 / 8192 / 8192 / 8197 | **922 / 381 / 2048 / 2048** |
| **clip rate** | 0.855 | 0.945 | 0.830 | **0.285（57/200）** |
| EOS 率 | 0.145 | 0.055 | 0.170 | **0.715（143/200）** |
| boxed_rate | 0.250 | 0.125 | 0.245 | **0.715（143/200）** |
| format_rate | 0.000 | 0.000 | 0.000 | 0.000 |
| accuracy_rate | 0.035 | 0.015 | 0.025 | **0.090（18/200）** |
| reward mean / std | 0.0315 / 0.166 | 0.0135 / 0.110 | 0.0225 / 0.141 | **0.081 / 0.258** |
| rep4 ratio mean / median | 0.584 / 0.565 | 0.340 / 0.346 | 0.483 / 0.491 | **0.187 / 0.180** |
| 最长重复跨度 mean / max（tokens） | 388 / 1618 | 22 / 67 | 35 / 70 | **15.7 / 76** |

自然完成（EOS）样本分布：n=143，长度 min/p50/p90/max = 35 / 247 / 1218 / 2047；
其中 boxed=98.6%、acc=12.6%（18/143）。clip 组（57）boxed=3.5%、acc=0。

### 10.3 解读与判定

1. **D2 通过 codex gate**：clip 94.5%→28.5%（≤30%），acc 1.5%→9.0%（不降反升），
   boxed 12.5%→71.5%，EOS 组 boxed 98.6%。`boxed_only + enable_thinking=False`
   @2048 成为首个训练 smoke 的候选口径。
2. 非思考下自然完成长度中位数仅 247 tokens（思考模式为 5838），重复度也是
   四条线里最低（rep4 0.187、最长重复跨度 15.7）。
3. 仍未注入 `<think>`，format_rate 保持 0（Q4 已定：接受为已知限制，任务奖励
   开关前另立模型原生格式奖励）。
4. **训练前仍需做 D3 cap gate**：D2 的方向已通过，但 28.5% sequence clip 对
   `n=4` 仍偏高；下一步固定 D2 其余变量，只把 cap 改为 4096。详见
   `docs/qwen35_d1_d2_codex_decision_20260804.md`。

## 11. 学生/teacher 渲染前缀对照（enable_thinking 决策证据）

用 val 第 1 条 question + boxed_only prompt，分别用学生
（Qwen3.5-4B）与 teacher（qwen3.6-27B）tokenizer 渲染
`apply_chat_template(messages, add_generation_prompt=True)`：

| 侧 | 默认（thinking）尾部 | `enable_thinking=False` 尾部 |
|---|---|---|
| student Qwen3.5-4B | `…<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n` | `…<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
| teacher qwen3.6-27B | `…<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n` | `…<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |

结论（模板观察有效；正式训练链路解释已被后续 source audit 纠正）：

1. **两个模型的 chat template 都支持 `enable_thinking` 且行为完全一致**：
   默认渲染 `<think>\n` 头；`enable_thinking=False` 渲染空的
   `<think>\n\n</think>\n\n` 块，随后生成可见输出（D2 dump 的生成里无 think
   标签，验证一致）。
2. **正式 pinned verl OPD 不存在上述 mismatch**：`AgentLoopWorker` 把学生的
   `prompt_ids + response_ids` 原样作为 `sequence_ids` 交给 teacher；teacher manager
   再调用 `client.generate(prompt_ids=sequence_ids, ...)` 计算 prompt logprobs，不会
   从 raw messages 独立重渲染。因此学生的空 think block 已在 teacher 输入序列中。
3. **不要给正式训练链路新增 teacher-side template propagation**。项目内 standalone
   FC-OPD teacher service 会自己渲染 raw messages，那条路径若启用仍需单独贯通
   `chat_template_kwargs`；它不是当前 `scripts/run_qwen35_formal.sh` 的阻塞项。
4. 直接传学生 token IDs 的真实前置条件是两边 canonical token-ID mapping 完全一致；
   文本前缀相同不能证明 token 对齐。wrapper 已新增 fail-fast tokenizer alignment
   preflight，D3/训练必须先通过。

# Qwen3.5 v1 对齐 + 截断消融结果报告（Step A + Step B）

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

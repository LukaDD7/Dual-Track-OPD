# Qwen3.5 Geometry3K prompt-fix smoke 结果报告（实验 #2 + #3）

> 日期：2026-08-03 ｜ 分支：`codex/va-opd` ｜ 提交：`c40e469`（本次报告另立新 commit）
> 上游依据：`docs/qwen35_promptfix_review_for_claude.md`（`87225df`）

## 1. 一句话结论

prompt-fix 接线验证完成：`FCOPDDataset` 注入的 boxed 指令真实进入 rollout，学生开始输出
`\boxed{}` 并产生可判分的正确样本（acc 2-3%，reward 非零且有方差）；开启
`USE_TASK_REWARDS=True` 后任务分进入 critic/advantage、`actor/pg_loss` 变为非 0——
**两条 smoke 均通过，任务奖励链路打通**。当前主要限制是响应顶满 2048 token 截断
（训练 batch `response_length/clip_ratio = 1.0`），下一步应先做「只改一个变量」的截断修复。

## 2. 可复现信息（对应 review §9 清单）

| 项 | 值 |
|---|---|
| 项目 repo | `Dual-Track-OPD` @ `c40e469`（codex/va-opd；工作树含既有诊断实验未提交改动，本线未触碰） |
| verl 后端 | `repos/verl-cu130-vllm` @ `334d9f8b`（本地有 `run_qwen3_5_4b_fsdp.sh` + `vllm_async_server.py` 两处研究 patch，dirty） |
| 训练器 | v0 trainer（`trainer.use_v1` 默认 false，未显式设置） |
| 环境 | conda `va-opd-qwen35-cu128`：torch 2.11.0+cu129 / cuda 12.9 / transformers 5.12.0 / flash_attn 2.8.3 / vLLM 0.23.0+cu129 |
| GPU | NVIDIA H200；`FORMAL_GPUS=0,1,2,3`（3 actor + 1 teacher），GPU 4-7 为并行诊断实验占用 |
| 学生 / 老师 | `models/Qwen3.5-4B` / `models/qwen3.6-27B`（TP=1, mem 0.55） |
| 数据 | `geometry3k_gkd/train_text_only.parquet` sha256 `63d56af2…99a61`；`val_text_only.parquet` sha256 `87d7f041…f2943` |
| 数据类 | `FCOPDDataset`（`data.custom_cls.path=pkg://dual_track_opd.fc_opd.verl_dataset`），prompt 注入 `Put the final answer in \boxed{}.` |
| 参数 | batch 24 / mini-batch 24 / `rollout.n=1` / prompt 1024 / response 2048 / k1 / `use_policy_gradient=True` / `loss_max_clamp=10` / lr 1e-6 / `resume_mode=disable` / `val_before_train=True` / `total_training_steps=20` / `save_freq=-1` / `test_freq=5` |
| checkpoint | `SAVE_FREQ=-1`，未落盘（无 51GB checkpoint） |
| 日志 | `artifacts/fc_opd/nohup_promptfix_smoke_20260803_105554.log`；`artifacts/fc_opd/nohup_promptfix_reward_smoke_20260803_*.log` |
| val dump（不进 Git） | `fc-opd-storage/logs/val_dump_k1_promptfix_smoke/{0,5,10,15,20}.jsonl`；`fc-opd-storage/logs/val_dump_k1_reward_smoke/{0,5,10,15,20}.jsonl` |

dump 记录字段：`input`（含角色前缀的完整渲染）、`output`、`gts`、`score`、`step`，以及
`acc`/`reward` 额外字段。分析口径：用后端 `verl.utils.reward_score.geo3k.compute_score`
复算（acc = `compute_score(..., format_score=0.0)`；format = `format_reward`），
与训练日志中的 `val-core/hiyouga/geometry3k/acc/mean@1` 一致。

## 3. 实验 #2：promptfix 接线 smoke（`USE_TASK_REWARDS=False`）

目的：验证 boxed 指令进 prompt、学生能输出 boxed+正确、奖励管线打对分、是否被截断。

### 3.1 运行命令

```bash
FORMAL_GPUS=0,1,2,3 NGPUS_PER_NODE=3 TRAIN_BATCH_SIZE=24 PPO_MINI_BATCH_SIZE=24 \
USE_FCOP_DATASET=1 USE_TASK_REWARDS=False ROLLOUT_N=1 RESUME_MODE=disable \
VAL_BEFORE_TRAIN=True TOTAL_TRAINING_STEPS=20 SAVE_FREQ=-1 TEST_FREQ=5 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_taskfalse_fcop_n1_promptfix_smoke \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_promptfix_smoke \
bash scripts/run_qwen35_formal.sh
```

### 3.2 验证点

- 启动日志：`Using dataset class: FCOPDDataset`（train/val 两处）、`Total training steps: 20`。
- dump `input` 确含注入指令，`gts` 有值（例：`gts: 109`）。
- 奖励路由不变：`val-core/hiyouga/geometry3k/acc/mean@1`。

### 3.3 验证集指标（每点 200 样本）

| step | boxed_rate | acc_rate | reward mean | reward std | 0.0 / 0.9 计数 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.100 | 0.020 | 0.018 | 0.126 | 196 / 4 |
| 5 | 0.310 | 0.030 | 0.027 | 0.154 | 194 / 6 |
| 10 | 0.190 | 0.030 | 0.027 | 0.154 | 194 / 6 |
| 15 | 0.150 | 0.025 | 0.022 | 0.141 | 195 / 5 |
| 20 | 0.170 | 0.030 | 0.027 | 0.154 | 194 / 6 |

boxed_rate 从实验 #1 的 0 起步（step0 0.10，step5 峰值 0.31，后回落至 0.15-0.19——
任务奖励关闭时模型无保持 boxed 格式的压力，属预期）。分数分布**只有 0.0/0.9**：
正确 boxed 无字面 think 标签 = 0.9，与 review §4 奖励表完全一致；所有输出均无字面
`<think>...</think>`，format 分量结构性为 0。

## 4. 实验 #3：任务奖励集成 smoke（`USE_TASK_REWARDS=True`）

目的：验证开启任务奖励后优化项真正变化且无数值失败（review §7.3）。

### 4.1 运行命令

```bash
FORMAL_GPUS=0,1,2,3 NGPUS_PER_NODE=3 TRAIN_BATCH_SIZE=24 PPO_MINI_BATCH_SIZE=24 \
USE_FCOP_DATASET=1 USE_TASK_REWARDS=True ROLLOUT_N=1 RESUME_MODE=disable \
VAL_BEFORE_TRAIN=True TOTAL_TRAINING_STEPS=20 SAVE_FREQ=-1 TEST_FREQ=5 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_tasktrue_fcop_n1_reward_smoke \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_reward_smoke \
bash scripts/run_qwen35_formal.sh
```

### 4.2 训练批次指标（20/20 步完成）

20 步中 4 步的训练 batch（24 样本 × n=1）含正确样本，任务分进入优化项：

| global_step | critic/score/mean | rewards/max | advantages/max | actor/pg_loss |
|---:|---:|---:|---:|---:|
| 5 | 0.075 | 0.9 | 0.9 | **-0.0662** |
| 14 | 0.075 | 0.9 | 0.9 | **-0.0288** |
| 16 | 0.038 | 0.9 | 0.9 | **-0.0237** |
| 17 | 0.075 | 0.9 | 0.9 | **-0.0548** |

其余 16 步 `critic/score/mean=0.0`（batch 内无正确样本，`(1-0.02)^24 ≈ 0.6` 的抽样概率），
对应 `pg_loss=0`。全程 `actor/distillation/loss` 0.08-0.19 有限、`actor/grad_norm` 1.5-7
有限、`actor/pg_clipfrac=0.0`（按 review §7.3 不要求非 0）、reward 不单调（不要求）。

### 4.3 验证集 step0 dump（200 样本）

`boxed=0.100, acc=0.015, mean=0.0135, std=0.109, 分布 197×0.0 + 3×0.9`——与实验 #2
同分布，奖励路由在 `USE_TASK_REWARDS=True` 下保持不变。

## 5. 判定

- 实验 #2：strong go 条件满足（`accuracy_reward_rate>0` 且 reward 有非零方差）→ GO。
- 实验 #3：集成证据满足（任务分进 critic/advantage、pg_loss 非 0、蒸馏/梯度有限）→ 通过。
- 两条 smoke 均为 20 步接线验证，**不构成学习效果结论**；任务奖励的语义是
  `ROLLOUT_N=1` 单样本组（GRPO 特判 mean=0/std=1）≈ 无基线 REINFORCE 式更新。

## 6. 遗留问题与下一步候选（供 codex 决策）

1. **截断（建议优先，review §6）**：训练 batch `response_length/clip_ratio=1.0` 全部顶满
   2048 token；val 输出 3-8k 字符，正确 boxed 常出现在超限前，boxed/acc 率被显著压低。
   下一实验**只改一个变量**：
   - 方案 A：`MAX_RESPONSE_LENGTH` 上调（需先重算 vLLM/token 显存与 `PPO_MAX_TOKEN_LEN_PER_GPU`）；
   - 方案 B：更精简的收尾指令（如"最后一行只给 \boxed{答案}"），不改 scorer。
2. **研究级任务-RL 对比（review §7.4）**：`ROLLOUT_N=4/8`，重算有效序列 batch 与显存，
   明确 prompt batch vs generated batch。
3. **trainer 对齐**：本轮为 v0 trainer（`use_v1` 默认 false），实验 #1 日志为 v1
   （TaskRunnerV1）；接线结论不受影响，若要与 #1 严格对齐可显式加 `trainer.use_v1=True`。
4. **更长 seeded run**：20 步 smoke 不支撑学习结论；任务奖励的收益/蒸馏退化权衡需更长
   seeded run（>100 步）观测。
5. **8 卡恢复**：GPU 4-7 释放后可去掉 `FORMAL_GPUS`、`NGPUS_PER_NODE=7`、`TRAIN_BATCH_SIZE=56`。

## 7. 过程性备注

- 脚本加固（`716741a`/`83e584d`/`c40e469`）：run 隔离 + resume guard + DRY_RUN +
  数值 env 校验；首次运行曾因 env 变量间缺空格（`TRAIN_BATCH_SIZE=24PPO_MINI_BATCH_SIZE=24`）
  带病启动，已由校验 FATAL 兜底并重跑。
- 两次运行均未 `CLEAN_START`、未触碰 GPU 4-7 的并行诊断实验；raw dump/日志未进 Git。

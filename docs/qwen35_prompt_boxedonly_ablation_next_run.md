# Qwen3.5 Step C：2048 精简收尾指令消融（boxed_only prompt）

> 日期：2026-08-04 ｜ 分支：`codex/va-opd`
> 依据：`docs/qwen35_v1_truncation_next_steps_for_claude.md` §4 决策规则
> （clip rate >70% → 不盲目加长，准备 2048 下独立版本的精简收尾指令消融）

## 1. 背景与动机

Step B（4096 val-only）结果：clip rate 在两个上限下都是 **96%**——模型按可用
token 预算续写推理，不会自然收尾。因此把 4096 当作训练口径没有意义；下一步的
可辩护长度契约应先在 2048 下尝试**让模型早点收尾**，而不是继续加长。

本次消融只改一个变量：**prompt 收尾指令文本**（`PROMPT_VERSION=boxed_only`），
其余全部与实验 #2 step 0（2048、FCOPDDataset、task reward off）保持一致；
共享 scorer 不动，`<think>` 标记不注入。

## 2. 新 prompt（v2 / boxed_only）

```text
<image>
{question}

Briefly reason. Then output your final answer as exactly one line: \boxed{<answer>}.
Stop immediately after that line; write nothing else.
```

对照 v1（实验 #2 / Step B 使用）：

```text
<image>
{question}

Put the final answer in \boxed{}.
```

实现位置：`src/dual_track_opd/fc_opd/prompt_contracts.py` 新增
`geometry3k_training_prompt_boxed_only` + 版本注册表；
`FCOPDDataset` 通过 `data.prompt_version`（wrapper env `PROMPT_VERSION`）选择版本。

## 3. 运行方式（GPU 实例，一条命令）

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
bash scripts/hpc/run_qwen35_v1_promptboxed_valonly.sh
```

等价于实验 #2 step 0 配置 + `PROMPT_VERSION=boxed_only` +
`trainer.val_only=True` + `TOTAL_TRAINING_STEPS=1`（只做 200 个 val 样本生成与打分，
不更新优化器）。产物：

- metadata：`fc-opd-storage/logs/qwen35_runs/k1_promptfix_boxedonly_r2/`
  （`run_manifest.json`、`resolved_launch_config.json`、`hydra/.hydra/config.yaml`、
  `train.log`）
- val dump：`fc-opd-storage/logs/val_dump_k1_promptfix_boxedonly_r2/0.jsonl`
- 外层日志：`artifacts/fc_opd/nohup_v1_pvboxed_only_valonly_*.log`

预计耗时：约 5-10 分钟（2048 生成，比 Step B 的 4096 短一半）。

> 注：首次启动（r1）因 Hydra 结构化配置拒绝新增键 `data.prompt_version` 而失败；
> 已改为 `+data.prompt_version=...` 追加并补测试（74c684b 后续修复提交）。
> r1 的 failed manifest 保留在 `k1_promptfix_boxedonly/`，r2 用全新目录。

## 4. 判读口径（与实验 #2 step 0 @2048 对比）

| 指标 | 实验 #2 step 0（v1 prompt） | Step C（boxed_only） |
|---|---:|---:|
| response token mean / median / p90 / max | 2025 / 2048 / 2048 / 2048 | 待填 |
| **clip rate（顶满 2048 比例）** | **0.960** | 待填 |
| boxed_rate | 0.100 | 待填 |
| format_rate | 0.000 | 待填 |
| accuracy_rate | 0.020 | 待填 |
| reward mean / std | 0.018 / 0.126 | 待填 |

分析口径与之前一致：Qwen3.5-4B tokenizer 数生成部分 token，
`verl.utils.reward_score.geo3k` 的 compute_score/format_reward 打分（去 dump 的
`assistant\n` 前缀）。

判定规则（沿用 handoff §4）：

- **clip rate <30%** 且 boxed/acc 不退化 → 精简收尾指令有效，2048 可作为训练口径，
  再考虑 `ROLLOUT_N=4` 的 batch 设计；
- **clip rate 仍 >70%** → prompt 指令本身压不住续写，报告长度分位数与按
  截断/未截断分组的 acc，再决定是否试更严格的"只答不推理"变体或换训练策略，
  **不要继续加长**；
- 30-70% → 报告分位数与分组 acc，再定。

## 5. 约束（与 handoff 一致）

- 不跑 `ROLLOUT_N=4`；不跑超过 20 步的实验；本轮只有 1 个 val 步。
- 不用 `CLEAN_START=1`（GPU 4-7 并行诊断实验不受影响）。
- 不动共享 scorer；一次只改一个变量（本次只有 prompt）。
- 只 push 代码/文档，raw dump 与日志留在 Git 外。

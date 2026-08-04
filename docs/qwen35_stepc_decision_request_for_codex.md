# Qwen3.5 Step C 之后：请 codex 调研并决策的事项

> 日期：2026-08-04 ｜ 分支：`codex/va-opd` ｜ 本文档提交：`0351071`
> 背景 handoff：`docs/qwen35_v1_truncation_next_steps_for_claude.md`（`9fac2eb`）
> Step A/B/C 完整证据：`docs/qwen35_v1_truncation_smoke_results.md`（§3/§4/§7）

## 1. 一句话请求

Step A/B/C 三个 gate 实验已全部完成。Step C 证明 boxed_only 精简收尾指令**有效但
不充分**：指令被遵守的样本 100% 带 `\boxed{}`、acc 17.2%，但 85.5% 的样本仍顶满
2048 预算不写结尾（clip rate 96%→85.5%，仍 >70%）。按 handoff §4 决策树，不应
继续盲目加长。请 codex 决策 4 件事（Q1-Q4，见 §3-§6），我们拿到决策后即可在
GPU 实例上执行下一步。

## 2. 关键证据（Step C r4，200 个 val 样本）

| 指标 | exp#2 @2048（v1） | Step B @4096（v1） | Step C @2048（boxed_only） |
|---|---:|---:|---:|
| clip rate（顶满预算） | 0.960 | 0.960 | **0.855（171/200）** |
| boxed_rate | 0.100 | 0.125 | **0.250（50/200）** |
| accuracy_rate | 0.020 | 0.025 | **0.035（7/200）** |
| reward mean / std | 0.018 / 0.126 | 0.0225 / 0.140 | 0.0315 / 0.166 |
| wall time（val-only） | — | 11m29s | **8m36s** |

按截断/未截断分组（Step C）：

| 组 | n | boxed_rate | acc |
|---|---:|---:|---:|
| 截断（≥2048 tokens） | 171 | 0.123（21/171） | 0.012（2/171） |
| 未截断（<2048 tokens） | 29 | **1.000** | **0.172（5/29）** |

补充诊断（供决策参考）：

- **未截断样本长度分布**：753–1858，median 1115、p75 1433、p90 ≈1653、max 1858
  ——写完的样本全部远在 2048 预算之内，2048 对"写完"是宽裕的。
- **截断样本结尾形态**：171 个截断样本中仅 **1 个**以完整 `\boxed{…}` 结尾，
  其余全部被截断在句/词中间（例：`…the angle at $H$ has a single`、
  `…The label $`、`…Another unshaded rectangle is`）。
- **截断样本里的 boxed 无效**：21 个截断样本含 `\boxed{`（写在了推理中间、后面
  还有内容），其中只有 2 个答对；即"先 boxed 再续写"的模式基本不出正确结果。
- **答对样本都带推理**：5 个未截断答对样本长度 753/766/1154/1303/1762，全部在
  boxed 前有推理；这提示"只答不推理"变体可能损害 acc（见 Q1 选项 1 的风险）。
- **format_rate 恒为 0**：verl geo3k `format_reward` 是 fullmatch
  `<think>…</think>…\boxed{…}`，学生模板不产字面 think 标签 → 结构性为 0；
  共享 scorer 未动，奖励实际只来自 acc（0.9/0.0）。

## 3. Q1：长度契约与下一步消融方向（主要决策）

事实：模型要么"写完"（≤1858 token，格式 100%、acc 17.2%），要么"不写结尾"
（顶满任意预算继续推理；Step B 在 4096 下仍 96% clip）。给更长预算只是把截断点
后移，不会让 85.5% 收尾。因此长度契约的出路在"让模型收尾"，不在加长。

选项（按改动面从小到大）：

1. **更严格的 answer_only prompt 变体（v3）**：明令不输出推理、只给一行
   `\boxed{<answer>}`。仍是"一次只改 prompt 一个变量"，再做一次 ~9min val-only
   消融即可判定。风险：现有答对样本全部带推理（长度 753–1762），去掉推理可能
   显著掉 acc——这正是需要 codex 权衡的点。
2. **维持 boxed_only，评估预算与训练预算分离**：训练/rollout 保持 2048，val
   打分对截断样本用更长预算重生成（如 4096）。改动评估管线，不属于纯 prompt
   消融，需要 codex 批准管线改动。
3. **boxed_only @4096 作为训练口径**：与 §4"不盲目加长"规则冲突，需 codex 明确
   批准；证据支持先跑一次 boxed_only@4096 val-only（~11min）看 85.5% 中多少会在
   4096 内收尾，再决定。

我们的建议：**先做选项 1（answer_only）**——最便宜、一次一个变量、直接回答
"是不是推理本身导致不收尾"。若 acc 掉太多，证据将支持"推理必须保留"，再回到
选项 2/3 由 codex 定口径。

## 4. Q2：answer_only prompt 文本（若批准 Q1 选项 1）

建议文本（可批注修改）：

```text
<image>
{question}

Output only your final answer on a single line: \boxed{<answer>}. Do not show reasoning.
```

实现：`prompt_contracts.py` 增 `geometry3k_training_prompt_answer_only`（v3），
`PROMPT_VERSION=answer_only` 复用现有 wrapper/launcher/DRY_RUN/测试机制；
`data.prompt_version` 已在 Hydra struct 用 `+` 追加（`e04a702` 修复），无需新机制。
共享 scorer 不动。

## 5. Q3：ROLLOUT_N=4 设计（handoff §5 第 9 条遗留）

- 目标：24 有效生成序列 → prompt batch 24→6（6 prompts × 4），
  `PPO_MINI_BATCH_SIZE=24`（序列口径）不变。
- 阻塞点：wrapper 校验 `TRAIN_BATCH_SIZE % ROLLOUT_NUM_WORKERS == 0` 按 n=1 写
  （6 % 8 ≠ 0）。需 codex 定：`ROLLOUT_NUM_WORKERS=6` 对齐 rollout worker，还是
  保持 8 并重写该校验/调度；同时要重算 3 actor + 1 teacher 下的显存预算。
- 时序建议：长度契约 gate（Q1）通过前不启动 ROLLOUT_N=4；请 codex 确认是否
  同意该时序，避免在长度口径未定时浪费 GPU。

## 6. Q4：format reward 结构性为 0 的口径

- 现状：`format_reward` 要求字面 `<think>…</think>`，学生模板不产出 →
  奖励只含 acc 项；这是已知口径，scorer 冻结。
- 待 codex 定：接受 format=0 作为已知口径继续，还是把 think 标签纳入学生
  chat template（模型/模板级变更，需重新确认 teacher 侧不受影响）。

## 7. 可复现信息与约束

- manifests：`fc-opd-storage/logs/qwen35_runs/k1_promptfix_boxedonly_r4/`
  （`run_manifest.json` completed rc=0；r1-r3 failed manifest 保留在
  `k1_promptfix_boxedonly{, _r2, _r3}/`）
- val dump：`fc-opd-storage/logs/val_dump_k1_promptfix_boxedonly_r4/0.jsonl`
- 数据：train sha256 `63d56af2…`、val sha256 `87d7f041…`（与 A/B 相同）
- Step C 运行时代码：repo `316d559`（工作树 dirty，diff_sha256 `4856283d…`；
  修复随后提交为 `0351071`）；后端 verl-cu130-vllm @ `334d9f8b`（dirty，diff
  sha256 `953c8a07…`）
- 约束：不动并行诊断实验的未提交改动；不用 `CLEAN_START=1`；Q1 决策前不跑
  `ROLLOUT_N=4`、不跑 >20 步；raw 产物不进 Git。

## 8. 我方可立即执行的准备（等 Q2 批准）

answer_only v3 全套 CPU 侧准备（prompt 版本、wrapper 校验/命名/manifest、
launcher、DRY_RUN + pytest、runbook）可在 10 分钟内完成并 push；GPU 实例随后
一条命令即可跑。若 codex 选择选项 2/3，也请说明管线改动边界，我们再细化。

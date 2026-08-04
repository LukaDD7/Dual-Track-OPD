# Dual-Track OPD 当前状态：Qwen3.5 可训练环境与 support diagnostic

> 状态日期：2026-08-04 ｜ 当前权威状态
> D1/D1-L/D2 详细决策见 `qwen35_d1_d2_codex_decision_20260804.md`。

## 1. 总结

可训练环境的基础设施已经打通，D1/D2 也已经回答了 decoding contract 的主要问题：

- greedy 确实制造了严重重复，但不是全部截断的唯一原因；
- sampled natural thinking 在 8192 仍有 83% 截断，质量收益不足，第一轮训练不采用；
- `boxed_only + enable_thinking=False + sampled` 在 2048 将 clip 降到 28.5%，
  accuracy 提到 9%，是当前训练候选；
- D3 @4096 已完成：clip 19%、EOS 81%、accuracy 10%，重复度未恶化；4096 冻结为
  首个 `n=4` 短 smoke 的 cap，但还不是长期训练定稿。

当前准确表述：**train-capable，但尚未 long-run-ready。** tokenizer-ID alignment
已在真实模型上通过（248,070 IDs、0 mismatch）；当前唯一主 gate 是跑通 `n=4` 的
20-step 训练证据并观察实际 group-level clip、任务信号与显存稳定性。

## 2. Track A：可训练 Qwen3.6 → Qwen3.5 OPD/GKD

### 已完成

1. cu129/Qwen3.5 环境已加载 student、teacher、vLLM rollout、Ray、FSDP 和在线
   teacher scorer。
2. 79-step K1 run 完成，loss/entropy/gradient 有限，checkpoint 可写。
3. FCOP Geometry3K prompt 与 scorer 接线完成；20-step v1 smoke 中 3 个 batch 出现
   正确样本和非零 task PG loss。
4. manifest、resolved Hydra、repo/backend/dataset/model hashes、日志和 val dump 路径
   已纳入运行记录。
5. `n=4` 语义已确认：6 prompt rows × 4 rollouts = 24 sequences；
   `PPO_MINI_BATCH_SIZE=6` 是 prompt 口径；8 workers 可整除 24。
6. D1/D1-L/D2 已完成：

| arm | thinking | cap | clip | EOS | boxed | acc |
|---|---|---:|---:|---:|---:|---:|
| D1 | default, sampled | 2048 | 94.5% | 5.5% | 12.5% | 1.5% |
| D1-L | default, sampled | 8192 | 83.0% | 17.0% | 24.5% | 2.5% |
| D2 | disabled, sampled | 2048 | **28.5%** | **71.5%** | **71.5%** | **9.0%** |
| D3 | disabled, sampled | 4096 | **19.0%** | **81.0%** | **82.5%** | **10.0%** |

### teacher prefix 的最终判断

正式训练使用 pinned verl `334d9f8b` 的 agent/teacher loop。student rollout 得到的
`prompt_ids + response_ids` 被原样传入 teacher 的 `generate(prompt_ids=...)` 计算
prompt logprobs；teacher 不重新渲染 raw messages。因此 D2 的
`enable_thinking=False` 前缀天然对两侧一致，`6a560b4` 提出的 teacher-side template
传播不是正式 OPD 的 blocker。

但直接传 token IDs 要求 student/teacher 对每个 ID 的语义一致。文本渲染看起来一致
并不能证明这一点；wrapper 现已增加 canonical ID-to-token mapping 的 fail-fast 审计，
结果进入 `tokenizer_alignment.json` 和 run manifest。远端若不通过，必须先换成共享
tokenizer 的模型组合，不能继续 D3/训练。

项目内 standalone FC-OPD teacher service 的确会自行渲染 raw messages；它是另一条
实现，后续如使用仍需单独贯通 `chat_template_kwargs`，但不要把它与正式 verl 路径混淆。

### 下一步顺序

1. 运行 `scripts/hpc/run_qwen35_v1_n4_nonthinking_sampled_r4096_smoke.sh`：6 prompt
   batch、6 prompt PPO mini-batch、4 rollouts、8 workers、3 actor GPUs，最多 20 步。
2. 训练和 validation 都显式固定 `temperature=1.0, top_p=.95, top_k=-1`。pinned
   backend 的训练默认其实是 `top_p=1.0`；此前“D1 完全复现默认训练 sampler”的表述
   不准确。现在选择的是 D1-D3 已验证的候选 sampler，不能依赖 backend 默认值。
3. smoke 使用 `USE_TASK_REWARDS=True`，验证完整目标；legacy format 分量仍结构性为 0，
   因此必须分开报告 accuracy reward 和 format reward，不能把总 reward 当成格式信号。
4. 必报 sequence/group clip、EOS、至少一条正确 rollout 的 group 比例、OPD/task loss、
   entropy/KL、grad、吞吐和峰值显存。若 group-clip 仍高于约 60% 或出现稳定性问题，
   回退 2048 并另立 overlong 稳定化实验，不试 8192。
5. smoke 通过后才冻结较长 seeded run；训练中仍须监测长度膨胀，D2/D3 不能替代
   training-time stability evidence。

## 3. Track B：support-aware frozen-policy diagnostic

K=32 四片已完成，exact-token strict merge 有效，64 prompts / 2112 rows 无协议错误。
但主统计 gate 未通过：

- teacher-gap within-prompt AUC 0.564（目标 >=0.60）；
- correct-tail 11（目标 >=20）；
- malformed 20.2%（目标 <=10%）；
- truncation 23.6%（目标 <=15%）；
- correct-tail rank@1 lift 为负。

正信号只在短响应层：<=512 tokens 的 AUC 0.792，95% CI 0.625–0.932，但只有 7 个
eligible prompts。K=8→K=32 stratum agreement 71.9%，RL-ready 从 18 调整到 27。

决策：**不启动五臂 bridge。** 新增的 verified teacher-proposal feasibility 实验已经
实现并通过 CPU 测试，但尚无 GPU 结果；它只生成 verified proposal cache，不更新策略。
调度上优先完成 Track A 的 4-GPU `n=4` smoke；Track B 可在其余一对 GPU 并行做
2-prompt proposal smoke，8-GPU full proposal run 等 Track A smoke 结束后再启动。

## 4. 当前不做的事

- 不把 natural thinking 8192 作为训练口径；
- 不跑 answer-only 主线；
- 不对正式 verl teacher 增加一次多余 chat-template 渲染；
- 不以 stop-on-first-box 截断生成；
- 不在 `n=4` smoke 之前开长跑；
- 不因 K=32 的短响应子组正信号直接启动 bridge training。

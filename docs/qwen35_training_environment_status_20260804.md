# Dual-Track OPD 当前状态：Qwen3.5 可训练环境与 support diagnostic

> 状态日期：2026-08-04 ｜ 当前权威状态
> D1/D1-L/D2 详细决策见 `qwen35_d1_d2_codex_decision_20260804.md`。

## 1. 总结

可训练环境的基础设施已经打通，D1/D2 也已经回答了 decoding contract 的主要问题：

- greedy 确实制造了严重重复，但不是全部截断的唯一原因；
- sampled natural thinking 在 8192 仍有 83% 截断，质量收益不足，第一轮训练不采用；
- `boxed_only + enable_thinking=False + sampled` 在 2048 将 clip 降到 28.5%，
  accuracy 提到 9%，是当前训练候选；
- 2048 尚未冻结，因为 `n=4` 下 28.5% 的 sequence clip 会大量污染 prompt group；
  下一步 D3 只把 cap 提到 4096，之后立即进入不超过 20 步的 `n=4` smoke。

当前准确表述：**train-capable，但尚未 long-run-ready。** 剩余 blocker 是先让新增的
tokenizer-ID alignment preflight 在远端模型上通过，再选择最终 cap 并跑通 `n=4` 的
短训练证据；不是 CUDA、Ray、FSDP、reward wiring 或 teacher template propagation。

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

1. 在远端确认 tokenizer alignment preflight PASS；这是 D3/训练的共同前置条件。
2. 跑 D3：
   `scripts/hpc/run_qwen35_v1_boxedonly_nonthinking_sampled_r4096_valonly.sh`。
3. D3 clip <=10% 且质量不降则选 4096；clip >20% 或无质量收益则选 2048，不再试
   8192。10%–20% 按质量/token 效率决定。
4. 用冻结 cap 跑 `n=4`、最多 20 步：6 prompt batch、6 prompt PPO mini-batch、
   8 workers、3 actor GPUs，记录 group-level clip/EOS 与 exact teacher-token identity。
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

决策：**不启动五臂 bridge。** 冻结 K=32 frontier 的 27 个候选，先做独立、预注册的
短响应确认/扩充短响应层；不得把整体主 gate 描述成通过。Track B 与 Track A 分开推进，
不阻塞 D3 和 `n=4` trainability smoke。

## 4. 当前不做的事

- 不把 natural thinking 8192 作为训练口径；
- 不跑 answer-only 主线；
- 不对正式 verl teacher 增加一次多余 chat-template 渲染；
- 不以 stop-on-first-box 截断生成；
- 不在 D3 和 `n=4` smoke 之前开长跑；
- 不因 K=32 的短响应子组正信号直接启动 bridge training。

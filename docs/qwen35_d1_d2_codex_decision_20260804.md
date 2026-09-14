# Qwen3.5 D1/D1-L/D2 后的 Codex 研判与下一步

> 日期：2026-08-04 ｜ 输入提交：`6a560b4` ｜ verl：`334d9f8b`
> 本文回答 D1/D2 之后的长度、thinking、teacher 前缀和训练时序问题。

## 1. 结论先行

1. **自然 thinking 不进入第一轮训练配置。** D1 证明 sampling 大幅消除了 greedy
   重复伪影，却仍有 94.5% 截断；D1-L 在 8192 仍截断 83%，只多得到 2 个正确答案。
   所以核心事实是“greedy 放大重复 + Qwen3.5 自然 thinking 本身过长”，不能再归因成
   单一的 greedy 问题，也不能靠继续加长解决。
2. **D2 是正确方向，但 2048 尚未冻结为训练 cap。** native
   `enable_thinking=False` 把 clip 降到 28.5%、accuracy 提到 9.0%，已通过方向 gate；
   但 28.5% 对 `n=4` 仍偏高。若暂按 rollout 独立估算，一组至少一个截断样本的概率
   为 `1-(1-0.285)^4 = 73.8%`。实际存在 prompt 内相关性，因此训练时必须直接记录
   group-level clip，而不能只依赖这个估算。
3. **下一步只跑 D3：D2 + 4096 cap。** 除 2048→4096 外不改 prompt、thinking 或
   sampler。D2 已自然结束的样本 p50=247、p90=1218，因此大部分样本不会因为 cap
   增大而被迫多生成；D3 专门测剩余 28.5% 长尾是否能自然结束，以及额外 token 是否
   换来答案质量。
4. **正式 verl OPD 没有 teacher 独立重渲染导致的 prefix mismatch。** pinned v1
   agent loop 把学生的 `prompt_ids + response_ids` 直接作为 `sequence_ids` 交给 teacher
   vLLM 的 `prompt_ids` 接口。D2 的空 think block 已经包含在学生 token 序列中；无需
   给正式训练链路新增 teacher-side `apply_chat_template_kwargs`。这也暴露出真正的
   fail-fast 条件：student/teacher 的 token-ID 映射必须完全一致。

## 2. D1/D1-L/D2 证据如何改写判断

| arm | thinking | cap | clip | EOS | boxed | acc | 解释 |
|---|---|---:|---:|---:|---:|---:|---|
| Step C | default, greedy | 2048 | 85.5% | 14.5% | 25.0% | 3.5% | 重复伪影严重，不代表训练 sampler |
| D1 | default, sampled | 2048 | 94.5% | 5.5% | 12.5% | 1.5% | 重复显著降低，截断未降低 |
| D1-L | default, sampled | 8192 | 83.0% | 17.0% | 24.5% | 2.5% | 自然 thinking 极长，4 倍预算收益很低 |
| D2 | disabled, sampled | 2048 | **28.5%** | **71.5%** | **71.5%** | **9.0%** | 首个可训练候选，但尾部仍需定 cap |

D1 相对 greedy 的 repeated-4gram 均值从 0.584 降到 0.340、最长重复跨度均值从
388 降到 22，说明早期对 greedy 的纠正是必要的。与此同时，D1-L 的自然结束样本
长度中位数 5838、p90 8191，说明“非 greedy 就会自然变短”也是错误推断。两个原因
同时成立：greedy 会制造重复性顶满；sampled natural thinking 仍会产生非重复的超长推理。

## 3. 对 `6a560b4` teacher 前缀问题的纠正

`6a560b4` 的模板对照本身有效：student 和 teacher 模板都能把
`enable_thinking=False` 渲染成空 think block。但它对**正式 Qwen3.5 OPD 路径**的
阻塞判断不成立。

固定后端 `334d9f8b` 的真实调用链为：

```text
student dataset/chat template
  -> student prompt_ids
  -> rollout response_ids
  -> AgentLoopWorker._compute_teacher_logprobs(
       sequence_ids = prompt_ids + response_ids)
  -> AsyncTeacherLLMServerManager.compute_teacher_logprobs_single
  -> teacher client.generate(prompt_ids=sequence_ids, prompt_logprobs=...)
```

证据位置：

- `verl/experimental/agent_loop/agent_loop.py`：`_compute_teacher_logprobs` 明确拼接
  `prompt_ids + response_ids`；
- `verl/experimental/teacher_loop/teacher_manager.py`：直接把 `sequence_ids` 传给
  `client.generate(prompt_ids=...)`，并断言 teacher 返回长度等于该序列长度。

所以 teacher 在正式 OPD 中不是生成答案，也不重新调用 chat template；它是在学生
实际采样序列上做 teacher forcing/logprob scoring。**不得为了“修 prefix”而改这条
正式链路。** 但只有 student 与 teacher 对每个 token ID 的语义一致，这种 forced
scoring 才有效；“两个模型渲染出的文本尾部相同”不能替代 tokenizer identity 检查。

此前 Qwen35 wrapper 没有强制做这项检查。本次新增
`scripts/qwen35_tokenizer_alignment.py`：在启动 manifest、teacher vLLM 和 Ray 前比较
两边 canonical ID-to-token mapping，不一致立即退出，并把审计结果写入
`tokenizer_alignment.json` 和 run manifest。**D3 和训练都以该 preflight PASS 为前置；
如果远端实际模型不通过，停止 D3，先更换为共享 tokenizer 的 teacher/student 组合，
不能用“训练以前跑通过”放行。**

项目内 standalone FC-OPD teacher service 是另一条实现：它接收 raw messages 并用
自己的 processor 渲染。在那条路径中，`chat_template_kwargs` 是否贯通仍是一个真实
工程问题，但它不阻塞当前 `scripts/run_qwen35_formal.sh` 训练线。两条实现以后必须在
文档和 manifest 中明确标注，避免再次混淆。

## 4. D3 的唯一变量和 gate

执行：

```bash
bash scripts/hpc/run_qwen35_v1_boxedonly_nonthinking_sampled_r4096_valonly.sh
```

固定不变：同一 200 个有序 Geometry3K val prompts、`boxed_only`、
`enable_thinking=False`、`temperature=1.0`、`top_p=.95`、`top_k=-1`。唯一变量为
`MAX_RESPONSE_LENGTH=4096`。

按以下规则冻结训练 cap：

- **clip <=10% 且 acc/boxed 不下降：** 选 4096，进入 `n=4` 训练 smoke；
- **clip 10%–20%：** 若正确数或 boxed 明显增加，仍可选 4096 做不超过 20 步 smoke，
  但把 overlong/group-clip 设成必报指标；
- **clip >20% 或质量无增益：** 不再试 8192。第一轮 `n=4` smoke 使用 2048，把
  overlong 作为已知训练样本类型；另立 sampler/termination 稳定化实验，不混入主线；
- 不增加 stop-on-first-box：此前 boxed 后续写的样本大多不是可靠最终答案。

D3 必须报告 overall 与 EOS/clip 分组的长度、boxed、accuracy、reward、重复度，另报
正确答案/千生成 token。只比较 clip 而不比较质量与 token 效率，不能据此选 4096。

## 5. Geometry3K/Qwen 外部实践如何解读

公开实践支持“长度问题常见、训练时要显式治理”，但没有找到与本实验同模型、同 prompt、
同 sampler 的公开 clip rate，不能拿别人的配置冒充可比结果：

- Qwen3.5 官方建议 sampled decoding，并警告 greedy 可能 endless repetition；D1 已复现
  重复维度，但也证明 sampling 不保证自然停止；
- EasyR1 的 Geometry3K launcher 复用其 sampled rollout 配置；当前默认配置为
  2048 response cap、`n=5`、`temperature=1.0`。它没有发布与本实验可直接比较的
  Geometry3K clip rate；
- RLinf 的 Qwen3-VL Geometry3K recipe 用 4096 cap、每 prompt 8 个 rollout，并引入
  truncated importance sampling 处理训练中的过长序列增长/潜在 reward collapse；
- ModelScope 的 Qwen3.5 GRPO/GKD 实践直接关闭 thinking 以避免过长 CoT，同时仍给
  8192 completion budget。这与 D2 的方向一致，但不证明本项目也应使用 8192。

因此别人常用的办法不是“换成 non-greedy 就不会截断”，而是 sampler、thinking mode、
长度 cap、clip metric 和 training-time stabilization 一起使用。当前先用 D3 做单变量
cap 选择；若 D3 仍失败，再单列 Qwen 官方 `top_k=20` sampler 或长度稳定化实验，不能
把它们混入 D3。

参考：

- https://huggingface.co/Qwen/Qwen3.5-4B
- https://github.com/hiyouga/EasyR1/blob/main/examples/qwen2_5_vl_7b_geo3k_grpo.sh
- https://github.com/hiyouga/EasyR1/blob/main/examples/config.yaml
- https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/agentic/qwen3_vl_geo3k.html
- https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md

## 6. D3 之后的首个训练 smoke

训练 smoke 不是学习效果实验，只验证 `n=4`、online teacher、loss、reward、显存和
记录链路。固定不超过 20 步：

```text
TRAIN_BATCH_SIZE=6
PPO_MINI_BATCH_SIZE=6
ROLLOUT_N=4
ROLLOUT_NUM_WORKERS=8
NGPUS_PER_NODE=3
PROMPT_VERSION=boxed_only
enable_thinking=False
MAX_RESPONSE_LENGTH=<D3 冻结值>
```

有效 batch 是 6 prompts × 4 = 24 sequences；8 workers 可整除 24。训练启动前再
决定是否打开 task reward：若打开，必须把 legacy `<think>...</think>` format 分量
结构性为 0 写入 manifest，并分别报告 accuracy reward 与 format reward；不能把总奖励
当成格式信号。首轮通过条件：

- tokenizer alignment preflight PASS，且 teacher_ids 与学生实际
  `prompt_ids + response_ids` 长度/内容一致；
- OPD loss、gradient、entropy/KL 全部有限，无 OOM/Ray/teacher 错误；
- 每步报告 sequence-level 和 prompt-group-level clip/EOS；
- 正确/错误、EOS/clip 都保留原始计数，不能只报均值；
- metadata、resolved Hydra config、git/backend/dataset hash 与 raw dump 路径完整。

通过这个 smoke 才能冻结第一轮较长 seeded 训练；D2/D3 本身只解决初始 rollout
contract，不证明训练中长度不会再次膨胀。

## 7. 两条任务线的当前状态

- **Track A（可训练环境）：** CUDA/Ray/FSDP/vLLM/online teacher、20-step v1、task
  reward 接线、manifest 和 `n=4` batch 语义均已通过；D2 已选择 non-thinking 方向。
  剩余顺序是 D3 选 cap → `n=4` ≤20-step smoke → 冻结长跑配置。当前是
  “train-capable，尚未 long-run-ready”。
- **Track B（support-aware）：** K=32 exact-token strict merge 已通过，但总体主 gate
  未通过（teacher-gap AUC 0.564、correct-tail 11、malformed 20.2%、truncation
  23.6%）。≤512 token 子组信号为正（AUC 0.792，CI 0.625–0.932），样本仅 7 个。
  暂停五臂 bridge；冻结 K=32 frontier 27 个 RL-ready 候选，先做独立、预注册的短响应
  确认。Track B 不阻塞 Track A。

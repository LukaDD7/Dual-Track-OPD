# PTD-PO (arXiv:2606.07000) 复现调研 — SFT-then-RL 的低限制增强方案 (2026-08-21)

> 背景: PRISM (arXiv:2604.28123) 调研后认为其限制太大 (外部教师蒸馏数据、
> MoE 判别器 + 120K warmup、强制 `<caption>/<think>/<answer>` 格式、verl 0.6.1
> 深度魔改)。本文件调研 PTD-PO (Privileged Tutoring Distillation Policy
> Optimization) 作为替代: 一个直接加在现有 GRPO 上的稠密监督模块,
> 不需要训练任何新模型。

## 0. TL;DR

- PTD-PO = "教思路, 不教答案": RLVR 的最终答案奖励稀疏, 失败 rollout 没有
  token 级信号; PTD-PO 离线用强模型 (Qwen3-VL-235B-Thinking / Gemini-3.0-Pro)
  给每题构造 **answer-free 结构化 hint** (空间定位 + 高维推理方向, 禁止泄题),
  训练时把 hint 拼进 **frozen reference model** 的上下文 (x^h=[x;h]),
  对失败轨迹用 **Top-K JSD (K=100, 带 tail 补偿)** 对齐 hint 增强教师分布;
  学生策略仍在 question-only 上下文下被 GRPO 优化。
- 8B 上 overall 71.86 vs GRPO 68.78 (+3.08); 4B +3.17; 2B +10.6;
  兼容 GRPO/DAPO/GSPO; 优于 HDPO (答案泄题自蒸馏) 和 PAPO (感知感知 RL)。
- 与 PRISM 相比的最大优势: **零新模型、零格式重构、奖励/数据/KL 全不动**,
  是现有 GRPO 的 drop-in 附加损失 (λ_ptd·L_ptd 只作用于失败轨迹)。
- 复现门槛: ① hint 数据需自建 (官方数据集未发布; 公开版 ViRL39K 无 hint 列);
  ② PTD 代码基于 EasyR1 (verl 0.3.3.dev0 fork), 需移植到我们的 verl 0.9.0 V0,
  但改动面很小 (4 个文件, 且 compute_ptd_mask / compute_topk_kl 是引擎无关的)。
- 最大的掉点风险 = **hint 质量**: 论文消融显示去掉结构化 hint 设计后
  8B 增益从 +3.55% 变成 -0.90%。hint 必须 zero-spoiler + 空间锚定 + 陷阱过滤,
  并做自动质量检查 (答案/关键中间数值泄漏检测)。

## 1. 方法摘要 (论文 v1 + 官方 repo 交叉核实)

### 1.1 动机

- 论文实测 (图 2): GRPO 训练中 all-fail 组 (m=0) 早期主导, informative 组
  (0<m<G) 始终只占 ~1/3 → 大量轨迹的组内优势几乎无信息。
- 完全 CoT+GT 条件化教师会诱发 shortcut (分布偏移大、输出变短、熵塌缩);
  GT answer 单独给又几乎不改变教师分布 (信息不足); hint 条件化是中间态:
  "足够纠正、不过度确定"。

### 1.2 机制

1. 离线构造结构化 hint (I, q, a*) → 强模型; 三条硬约束:
   - 与正确推理路径一致 (不生成完整 CoT);
   - zero-spoiler: 禁止最终答案、精确中间数值、可识别答案的对象名;
   - 显式抑制视觉干扰物与常见推理陷阱。
   生成后质量检查: 含答案 / 暴露决定性中间数值 / 漏关键视觉证据 /
   退化为完整 CoT → 丢弃或重生成。
2. 训练时: 学生按 question-only 上下文 rollout, r_i = 规则奖励 ∈ {0,1};
   组准确率 r̄ < τ_ptd (默认 1.0, 即所有失败轨迹) 且 r_i=0 的位置进入 PTD 集。
3. 教师 = frozen ref, 条件化为 x^h=[x;h] + 学生已生成前缀, 得到 q^h;
   学生 p 仍在 x 下; 损失 L_ptd = mean_{(i,t)∈F} D_ptd(q^h, p)。
4. D_ptd = Top-K JSD (K=100) + tail 补偿: 支撑集 = student top-K ∪ teacher
   top-K, 外部质量合并为 tail bucket, 避免硬截断丢概率质量; 显存从 O(BTV)
   降到 O(BTK)。
5. 总目标: L = -J_GRPO + λ_ptd · L_ptd; KL 惩罚保持标准 GRPO 设置
   (kl_coef 1e-2, 与 PRISM 的 KL=0 相反)。

### 1.3 关键超参 (Table 3 + 官方脚本)

| 项 | 值 |
|---|---|
| 数据 | ViRL39K (38,870 题), 2 epochs |
| 优化 | AdamW, lr 1e-6, wd 1e-2, constant, warmup 0, bf16, 不冻结 vision |
| RL | global batch 128, rollout batch 384, n=8, top-p 0.99, max_response 4096 |
| 奖励 | 0.1·format + 0.9·accuracy (规则二分) |
| KL | 1e-2 (low_var_kl) |
| PTD | λ_ptd: 4B=5e-1, **8B=5e-2**; τ=1.0; top-K=100; jsd_kl; frozen ref teacher |
| 硬件 | 8/16×H100; 评测 temp=1.0, top-p=1.0, max 4096 |

### 1.4 结果与消融

- 主表 (overall): 2B 61.21 vs GRPO 50.63; 4B 71.23 vs 68.06;
  8B 71.86 vs 68.78; 均优于 HDPO、PAPO、OPSD。
- 兼容性 (4B): GRPO +5.13%、DAPO +3.00%、GSPO +1.92%。
- 消融:
  - w/o 结构化 hint 设计: 8B 增益 +3.55% → -0.90% (hint 质量是命门);
  - τ=1.0 (所有失败轨迹) 最优; all-trajectories 过正则 (8B 只 +1.27%);
  - 小阈值 (0.2/0.4) 信号不足。
- 附带收益: 缓解熵塌缩、失败轨迹可恢复性提升、蒸馏显存开销小。

## 2. 与 PRISM 的对比 (为什么这个更契合我们)

| 维度 | PRISM (2604.28123) | PTD-PO (2606.07000) |
|---|---|---|
| 要修的问题 | SFT 引入的分布漂移 (SFT→RL 之间插阶段) | RLVR 稀疏奖励对失败轨迹无 token 级信号 |
| 新增训练模型 | MoE 判别器 (4×Qwen3-VL-2B) + 120K warmup, 需 transformers 4.57 补丁 | 无; frozen ref 即自教师 |
| 外部教师依赖 | 训练期需要 Gemini-3-Flash 蒸馏监督池 | 只在离线 hint 构造用 (本地 235B-FP8 即可) |
| 格式约束 | 全链路强制 `<caption>/<think>/<answer>`, 格式错奖励=0 | 保持现有 `<think>`/`<answer>` 格式, 奖励规则不变 |
| KL | 对齐阶段 KL=0 (反对锚定 SFT) | 保持标准 KL 1e-2 |
| 数据规模 | SFT 1.37M + 113K 蒸馏; 复用官方资产才保真 | 只需给现有 RL 池加一列 hint (可自建) |
| verl 改动 | mm_gad critic +875 行 + grpo_multi_reward + teacher 字段透传 | 4 个文件, 核心两函数引擎无关 |
| 8B 增益 | +6.0 avg (7 项 benchmark) | +3.08 overall (PAPO 套件) |
| 与现有三臂融合 | 难 (数据/格式/判别器都要变) | 易 (SFT-then-RL 的 RL 段直接加 PTD 损失) |

结论: PTD-PO 是我们现有 GRPO 的**第四臂候选** (SFT-then-RL+PTD),
不是替代 SFT-then-RL 的新范式; PRISM 则要求重做数据与训练栈。

## 3. 复现路径 (映射到本地栈)

### 3.0 现状盘点

- 本地有 `models/Qwen3-VL-235B-A22B-Instruct-FP8` (104GB, 24 shards 完整)
  → 可离线生成 hint (vLLM TP=2~4, 8×H200 141GB 可行)。
- 本地栈: verl 0.9.0 (V0, tag 483b8a0) + transformers 5.12 + 8×H200;
  MMF RL 3K 池 (mmf_rl_3k, train 3000/val 150) + mmf_reward + warmup ckpt。
- 官方 PTD-PO 代码: github.com/XszNeverSleep/PTD-PO (EasyR1, verl 0.3.3.dev0
  fork; 数据未发布, README TODO)。
- 公开数据: TIGER-Lab/ViRL39K; PAPOGalaxy/PAPO_ViRL39K_train (~2.9GB,
  列只有 images/problem/answer, **无 hint 列**); PAPOGalaxy/PAPO_MMK12_test
  可做评测。

### 3.1 Step 0 — 数据: 构造 hint 列 (offline, 一次性)

1. 从 RL 池选训练集 (先 3K mmf_rl_3k 或 geometry3k 全量; 全量 MMF 117K 可选)。
2. 用本地 235B-FP8 按三约束协议生成 hint (输入 I, q, a*; 输出 answer-free
   结构化 hint)。**注意**: 论文 Figure 7 的完整 system prompt 在 PDF 截图里
   (源码/代码库未含文本), 需从论文图或作者处取原文; 可按论文文字描述重建
   (三约束 + 简洁祈使句 + 仅对视觉歧义/难点给更细指引)。
3. 自动质量检查 (必须有):
   - hint 文本包含 GT 答案串 / 关键中间数值 (与 answer/original_answer 比对)
     → 丢弃或重生成;
   - 长度/格式 sanity: 明显退化为完整 CoT (>N tokens) → 丢弃;
   - 随机抽查 50-100 条人工确认空间锚定与陷阱过滤质量。
4. 产物列: `prompt` (原 question-only chat, 现有)、`images`、`answer`/
   `ground_truth`、`prompt_with_hint` (字符串: `<image>` 占位 + 原问题 + hint,
   PTD-PO 的 dataset.py 会对 hint 套同一个 format_prompt jinja)。

### 3.2 Step 1 — 移植 PTD 到 verl 0.9.0 V0 (参考官方源码, 4 处)

| 位置 | 改动 | 官方参考 (PTD-PO) |
|---|---|---|
| `verl/utils/dataset/rl_dataset.py` (V0) | 读 `prompt_with_hint` 列, 用已处理 images 二次 tokenize 出 `hint_input_ids/attention_mask/position_ids` 进 batch | `verl/utils/dataset.py` `_build_hint_messages` + tokenize 段 |
| `verl/trainer/ppo/ray_trainer.py` (V0) | 奖励后按组准确率+失败标志算 `ptd_mask/ptd_group_mask`; enable_ptd 时对失败轨迹用 hint 上下文跑 ref forward 得到 ref top-K log-probs; 度量 ptd/mask_ratio 等 | `ray_trainer.py` 660-800 行 |
| `verl/workers/actor/dp_actor.py` (V0) | update 前向时额外输出 student top-K (logits/probs/ids), K=100 | `dp_actor.py` compute_log_prob ptd_top_k 分支 |
| `verl/trainer/ppo/core_algos.py` (V0) | 原样搬 `compute_ptd_mask` + `compute_topk_kl` (jsd_kl + tail); policy loss 里加 `λ_ptd·L_ptd` (只在 ptd_mask 位置) | `core_algos.py` 640-830 行 |

配置新增: `enable_ptd / ptd_threshold=1.0 / ptd_top_k=100 / ptd_coef=5e-2 (8B) /
ptd_kl_direction=jsd_kl / ptd_use_ref_teacher=true`。ref 模型即现有 GRPO 的 ref
(warmup ckpt, frozen), 无需新模型。注意 V0 的 no_padding/dynamic bsz 与
hint_input_ids 的批内拼接, 以及 warmup ckpt lm_head=F32 → 继续
`use_fused_kernels=False`。

### 3.3 Step 2 — 冒烟 (隔离变量)

- 8B warmup ckpt + mmf_rl_3k (含 hint) + 现有 GRPO 配置 (batch 16, n=8,
  response 8192→先修 clip_ratio, 见风险 2), 25-50 步。
- 监控: ptd/mask_ratio (早期应接近 1, 后期下降)、ptd loss 量级、
  ref vs student top-K JSD、reward 曲线不塌; 与同预算纯 GRPO 对照。

### 3.4 Step 3 — 正式

- 与三臂同数据/评测/reward; 加第四臂 SFT-then-RL+PTD (λ_ptd 从 5e-2 扫
  {2e-2, 5e-2, 1e-1})。
- 评测: 现有 6 项 + geometry3k test + 补 MathVista/We-Math (对齐论文口径);
  可选 PAPO 套件 (PAPOGalaxy/PAPO_MMK12_test 等) 直接对标论文数字。

## 4. 关键差异 → 掉点风险表

| # | 差异 | 影响机制 | 预期影响 | 缓解 |
|---|---|---|---|---|
| 1 | **hint 质量/协议** | 结构化 hint 是全部增益来源; 非结构化 hint 在 8B 上反而 -0.90 | **最高** | 严格三约束 + 自动泄漏检测 + 人工抽查; 用 235B 而非弱模型 |
| 2 | 响应长度: 论文 4096 vs 我们 8192 但 clip_ratio 0.42 | 截断 → 最终答案被裁 → reward=0 且 PTD 对齐到不完整前缀 | 高 | 先升 max_response 12288 或按教师 CoT 长度过滤 (runbook 7.13 已有方案) |
| 3 | RL 集: 38.9K ViRL39K vs 我们 3K 且全 pass_rate=0 | 集小 + 全失败组多 → PTD 激活率高, 若 hint 质量不稳会把错误方向放大; 且统计噪声大 | 中-高 | 先 3K 验证机制, 再扩 (sample_mmf_rl.py 可加难度分层) |
| 4 | 引擎: EasyR1 verl 0.3.3 vs 我们 0.9.0 V0 | V0 batch/no_padding/dynamic bsz 结构不同, hint_input_ids 拼接易踩坑 | 中 (工程) | 按官方源码逐段移植 + 单测; 参考仓库已有 STP loss patch 经验 |
| 5 | λ_ptd: 4B 5e-1 vs 8B 5e-2 | 系数过大 → over-regularize 正确轨迹; 过小 → 信号不足 | 中 | 8B 从 5e-2 起步扫 3 档 |
| 6 | 教师模型: 论文 235B-Thinking (+Gemini 兜底) | 本地 235B 是 Instruct-FP8 (非 Thinking), hint 生成质量可能略降 | 中 | 先小批量抽查; 必要时混入 32B judge 校验; 或等官方数据集发布 |
| 7 | 评测口径: PAPO 套件 (rule-based, 4096) vs 我们 6 项 + 32B judge | 数字不可直接对比; 部分任务 judge 依赖 | 中 | 补 We-Math/MathVista; 或直接上 PAPO 套件 |
| 8 | 训练预算: 2 epochs 38.9K vs 我们 1 epoch 3K | 小集单 epoch 下 PTD 的长期收益未必显现 | 中 | 保持三臂同预算对比; 只看相对增益 |
| 9 | 计算开销: hint 上下文 ref forward (仅失败轨迹) | 早期 all-fail 比例高 → 前期 +30-60% ref 前向; 后期下降 | 低-中 | 冒烟测吞吐; 可对 hint 前缀做长度上限 |
| 10 | 基座: 论文 Qwen3-VL-Thinking vs 我们 Instruct+warmup | 绝对数字不同, 但 PTD 是模块化附加, 机制应迁移 | 低 | 不跨论文比绝对值, 只比组内 Δ |

## 5. 建议行动

1. **先做数据**: 用本地 235B-FP8 在 mmf_rl_3k 上生成 hint (先 200 条验证协议 +
   质量检查, 再全量), 同时向作者/论文图要 Figure 7 的原始 system prompt。
2. **并行做移植**: core_algos 两函数可先落地 + 单测 (引擎无关),
   再接 dp_actor top-K 与 ray_trainer mask/ref-hint forward。
3. **冒烟对照**: 同预算 GRPO vs GRPO+PTD, 25-50 步, 看 ptd mask 占比与
   reward 曲线; 达标后扩到正式第四臂。
4. **决策点**: 若 PTD 增益确认, 再评估是否与 PRISM 式对齐叠加
   (两者修的问题不同: SFT 漂移 vs RL 稀疏奖励, 理论可互补, 但工程分步做)。

## 5.5 Hint 构造流水线 (自家数据落地规格, 2026-08-21 补充)

> 原则: **只给训练集加 hint 列, val/test 与三臂其余各臂完全不动**;
> 同一数据池、同一 reward、同一评测集、同一预算, PTD 臂唯一差异 = hint 列 +
> PTD 损失。这不影响公平性, 反而把问题精确变成
> "对同一份 RL 数据, PTD 的稠密监督是否提升 GRPO"。

### 5.5.1 输入 (mmf_rl_3k 现状, 实测 schema)

每行已有: `images` (bytes), `question` (含 `<image>` 占位), `answer`
(ground_truth, 字母/是或否/数值/LaTeX), `original_answer` (详细推导, 仅供 QC
与一致性参照, **不得进入 hint**), `source`, `sample_uid`。

### 5.5.2 生成协议 (离线, 一次性)

- 服务: vLLM serve `models/Qwen3-VL-235B-A22B-Instruct-FP8` (TP=2~4,
  gpu_memory_utilization 0.9, 8×H200 富余), 或 32B 做 easy 集兜底;
  批量客户端并发请求, 输出 max_tokens=768, temp=0.7, 每题 1 次 + 失败重试
  (temp=0.9) 1 次。
- 输入: (image, question, ground_truth answer) — answer 仅离线可见, 属
  privileged 信息; 可另传 original_answer 作为"验证参考" (显式标注为私有,
  只用于内部一致性, 禁止写入 hint)。
- 输出: `hint` 字符串 → 新列 `prompt_with_hint`:
  `[{"role": "user", "content": "<question 原文>\\n\\n[hint 文本]"}]`,
  与现有 `prompt` 列同构 (V0 数据加载器按同一 chat 模板二次 tokenize,
  复用同一批 images)。

### 5.5.3 草拟 system prompt (按论文三约束重建, Figure 7 原文待 OCR/索取)

```text
You are an expert multimodal reasoning tutor. Given an image, a question, and the
verified correct answer (PRIVATE — never reveal it), produce a CONCISE, structured
hint that teaches the reasoning path WITHOUT exposing the answer.

Hard rules:
1. Solution-consistent: point to the visual evidence and the reasoning steps that
   lead to the correct solution; align with the verified reasoning direction.
2. Zero-spoiler: NEVER output the final answer, exact intermediate numerical
   results, or object names that uniquely identify the answer. Do NOT write the
   solution trace or a chain of thought.
3. Distractor suppression: explicitly list which visual elements or textual cues
   are irrelevant or traps and should be ignored.

Format: imperative bullets, 2-6 bullets, 40-200 words. Give slightly more detail
only for visually ambiguous or logically difficult steps.
```

### 5.5.4 质量检查 (必须, 决定成败)

| 检查 | 规则 | 处置 |
|---|---|---|
| 答案泄漏 | `answer` 归一化 (去空格/大小写/LaTeX) 后是 hint 子串; 或 hint 中任何数值经 mathruler 判等 GT | 丢弃 → 重生成 (temp=0.9) → 仍失败则 hint=空 (PTD 对该样本不激活) |
| MCQ 泄漏 | 选项字母以独立 token 出现在 hint 中 | 同上 |
| 决定性中间数 | hint 与 original_answer 的数字交集 ≥2 (排除 0/1/2/3 等平凡值) | 软拒绝: 记录 + 抽样人工复核 |
| CoT 退化 | hint 长度 > 512 tokens 或含"最终答案/因此答案是"句式 | 丢弃重生成 |
| 遗漏视觉证据 | 无任何空间指代 (区域/对象/相对位置/图表元素) | 软拒绝 + 抽样复核 |
| 人工抽查 | 每个 source 抽 5-10 条 (共 50-100) 核对空间锚定 + 零泄题 | 不达标则调 prompt 重跑 |

QC 脚本与生成脚本同库落盘, 附生成配置 (模型/温度/seed/prompt 版本/QC 阈值),
保证可复现 (AGENTS.md 可复现性要求)。

### 5.5.5 成本

3K 题 × ~0.5-1.5K 输出 tokens + 图片, 235B-FP8 (TP=2) 约 1-2 小时;
扩到全量 117K 约 1-2 天 (可并行分片)。相比 8B GRPO 单轮 8.4h, 完全可接受。

### 5.5.6 移植锚点 (官方实现机制, 已核实)

官方 PTD 实现 (EasyR1/verl 0.3.3) 的机制:
1. ref 侧 `compute_teacher_log_prob`: 用 hint 上下文 (hint_input_ids) 对
   **全部**样本做 no_grad 前向 (防 FSDP deadlock), 输出 teacher top-K
   [B,T,K] (detached), 非 PTD 位置清零;
2. actor 更新时 `_compute_ptd_loss`: student top-K (with grad) + teacher
   top-K → `compute_topk_kl` (jsd_kl + tail), 只取 ptd_mask 位置,
   加 `λ_ptd·L_ptd` 进总 loss;
3. ray_trainer: 奖励后按组准确率算 `ptd_mask/ptd_group_mask` (τ=1.0:
   失败轨迹), PTD 组零化 reward 侧的 KL penalty (use_kl_loss=False 路径),
   度量 ptd/mask_ratio。

对应咱们 verl 0.9.0 V0 的 4 个落点:
| 文件 (本后端) | 改动 |
|---|---|
| `verl/utils/dataset/rl_dataset.py` (RLHFDataset) | 加 `prompt_with_hint_key`; 复用已处理 images 二次 tokenize hint messages → `hint_input_ids/attention_mask/position_ids` 进 batch |
| `verl/workers/actor/dp_actor.py` + `fsdp_workers.py` | ref 侧 hint 前向出 teacher top-K; actor 更新侧 student top-K + `_compute_ptd_loss` (compute_topk_kl 原样搬) |
| `verl/trainer/ppo/ray_trainer.py` (V0) | 奖励后算 ptd_mask; 组装 hint batch 调 ref; 度量 + PTD 组 KL 处理 |
| `configs/experiment/*.yaml` | `enable_ptd=true, ptd_threshold=1.0, ptd_top_k=100, ptd_coef=5e-2 (8B), ptd_kl_direction=jsd_kl, ptd_use_ref_teacher=true, data.prompt_with_hint_key=prompt_with_hint` |

`compute_ptd_mask` 与 `compute_topk_kl` 为引擎无关函数, 可从官方源码直接搬并
先写单测, 不依赖 verl 版本。

## 6. 参考

- 论文: https://arxiv.org/abs/2606.07000 (Teaching the Way, Not the Answer:
  Privileged Tutoring Distillation for Multimodal Policy Optimization)
- 官方 repo: https://github.com/XszNeverSleep/PTD-PO (EasyR1/verl 0.3.3.dev0 fork;
  数据集未发布)
- 公开数据: TIGER-Lab/ViRL39K; PAPOGalaxy/PAPO_ViRL39K_train (无 hint 列);
  PAPOGalaxy/PAPO_MMK12_test
- 本文档事实均为 2026-08-21 实测 (论文全文/源码/数据 schema 交叉核对)。

## 8. 实现状态 (2026-08-21, 代码已落地)

### 8.1 已实现

**数据侧 (本 repo, 可直接在 GPU 实例跑)**
- `scripts/sft_rl/hint_gen_prompt.md`: hint 生成 system prompt 草稿 v1
  (论文 Figure 7 重建版)。
- `scripts/sft_rl/build_mmf_hints.py`: 批量 hint 生成 + QC + 落
  `prompt_with_hint` 列 (幂等续跑, schema 与 mmf_rl_3k 一致, 9/9 单测通过,
  合成数据端到端冒烟通过)。
- `scripts/sft_rl/serve_hint_gen.sh`: vLLM serve Qwen3-VL-235B-A22B-Instruct-FP8
  (TP=2) + 可选自动跑生成。
- `tests/sft_rl/test_build_mmf_hints_qc.py`: 泄漏/退化/空间锚定 QC 单测。

**训练侧 (verl 0.9.0 V0 backend, 补丁已导出)**
- `patches/verl/ptd_opd_20260821.patch` (751 行): 对
  `fc-opd-storage/backends/verl-qwen35-v090-cu132` 的完整改动。
- 机制: 复用内置 top-K 蒸馏管线 (`distillation_ppo_loss` + logits processor),
  新增 `loss_mode=jsd_topk` (fsdp `compute_jsd_topk` + registered final variant),
  教师 top-K 由冻结 ref 模型 + hint 上下文前向产生
  (`TrainingWorker.compute_teacher_log_prob`, padded forward, mRoPE 位置扩展),
  trainer 侧按组准确率算 `ptd_mask` (失败轨迹, τ=1.0) 并 mask 损失,
  总 loss = GRPO policy loss + `coef · JSD`。
- 配置: `algorithm.ptd.*` (`PtdConfig`, hydra `ppo_trainer.yaml` 已加字段,
  `--cfg job` 干跑 EXIT=0)。
- 启动: `scripts/sft_rl/run_grpo_mmf_ptd.sh` (默认 8B: coef=5e-2, top_k=100,
  jsd_kl, 数据 = mmf_rl_3k_hint, val 用原 mmf_rl_3k)。

### 8.2 ⚠️ 重要发现: 官方发布版 jsd_kl 的梯度恒为零

核对官方源码 (PTD-PO `core_algos.py` compute_topk_kl) 时发现其 jsd_kl 分支对
student 概率做了 `.detach()` (注释称防 p_s→0 梯度爆炸)。用有限差分验证:
**该 detach 模式使整个 JSD 的 autograd 梯度精确为零** (loss 数值非零但无梯度),
即发布版 `jsd_kl` 实际不提供训练信号 (官方 8B 脚本正是 `ptd_kl_direction=jsd_kl`)。
我们的移植去掉 student 权重上的 detach, 实现数学正确的 JSD 梯度, 并用单测
(`test_jsd_gradient_nonzero_guard`) 锁定——若未来换回官方写法测试会失败。
稳定性由 `log_prob_min_clamp`/`loss_max_clamp` 兜底。

### 8.3 待做 (GPU 冒烟)

1. GPU 冒烟 `run_grpo_mmf_ptd.sh` 25-50 步 (同预算 vs 纯 GRPO 对照), 监控
   `ptd/mask_ratio` (早期≈1)、`ptd/jsd_mean`、`ptd/teacher_mass`、reward 曲线;
2. λ_ptd 扫 {2e-2, 5e-2, 1e-1}; 评估同一套 benchmark;
3. 若 PTD 确认有效, 再评估与 PRISM 式对齐叠加 (两者修的问题不同, 可互补)。

## 9. 2026-08-24 修复与数据复检

### 9.1 step-0 崩溃根因: mRoPE position_ids 微批切片切错轴（已修）

8/22 凌晨 12 次 `qwen3vl_grpo_mmf_ptd_*` 冒烟全部死在 step 0, 错误链一路收敛
到最后一处: `engine_workers.py compute_teacher_log_prob` 内 Qwen3VL RoPE
`size of tensor a (3) vs b (2)`。根因:

- agent loop 的 `_compute_position_ids` 返回 `[1, 4, seq]`, 按 `dim=0` 拼 batch
  后是 `[B, 4, seq]`;
- `engine_workers.py` 先 `transpose(0,1)` 正确得到 `[4, B, seq]`
  (text/time/height/width), 但微批循环里 `teacher_pos[i:end]` 切的是 **dim=0
  (4 个 mRoPE 分量轴)**, 把前 2 个分量当成 batch 切走 → RoPE 内 3 vs mb 维度
  不匹配。

修复: 3-D 时改切 `dim=1` (`teacher_pos[:, i:end]`), 并在
`tests/sft_rl/test_ptd_jsd_loss.py` 加回归测试
`test_hint_teacher_position_ids_microbatch_slicing` 锁住布局链。backend 工作区
已就地修改 (与既有 PTD patch 同一 dirty tree)。

### 9.2 响应截断风险: PTD 臂默认 12288

`run_grpo_mmf_ptd.sh` 的 `MAX_RESPONSE_LENGTH` 默认从 8192 升到 **12288**
(`run_grpo_mmf.sh` 基线此前已升)。理由: 8192 下 clip_ratio=0.42, 被截断的失败
轨迹会把 PTD 的对齐目标变成"不完整前缀"——学生学的是如何被截断, 对 PTD 的
伤害比纯 GRPO 更大。两臂必须同响应长度, 否则对照被混淆。

### 9.3 hint QC v2: 多段答案分量泄漏（已修 + 全量复检）

3K hint 人工抽查发现两类旧 QC 漏网:

1. **分量泄漏**: 答案 `50, 57.6, 292` 的 hint 里写 `total = 32 ÷ 0.64 = 50 days`
   ——旧 QC 只查完整答案串, 单个分量漏过;
2. **断言句式**: `the correct response is null` (且与 GT 相悖) 不在旧
   CoT 退化正则里。

QC v2 (`build_mmf_hints.py`):

- 多段答案按 `[,;、；]` 拆分量, 数字分量逐个查泄漏; "other" 答案内嵌数字
  (如 `80 decimeters` 的 80) 同样查; 问题文本中出现过的数字豁免, 但出现在
  算式结果位置 (`= 50`) 的不豁免;
- 断言正则加 `correct response / the response (is|should be) / 正确答案 / 应选`;
- 新增 `--recheck` 模式: 对已生成分片离线复检, 新命中硬拒的 hint 置空并备份
  原分片, 无需重新调生成模型;
- 生成 prompt (内联默认 + `hint_gen_prompt_v2.md`) 增加多段答案规则。

**3K 全量复检结果** (`hint_recheck_stats.json`): 3000 行中 1886 保留、
769 置空 (新命中: component_numeric 1116 / too_long 96 / component_word 32 /
cot_degenerate 2)、345 原本就为空。抽样人工核验: 绝大多数为真泄漏或论文
明确禁止的"精确中间数值" (公式原样复述、参数表全量罗列、算式结果); 少量
坐标分解边缘误杀可接受 (置空 = PTD 不激活, 安全方向)。当前 hint 覆盖率
62.9%, 冒烟够用; 全量池生成时配合 v2 prompt + 235B 教师可降低泄漏率,
生成后再 `--retry-rejected` 补齐。

### 9.4 遗留事实修正

- 3K hint 实际由 **Qwen3-VL-32B-Instruct** 生成 (`hint_gen_config.json`),
  不是本 review §3.0 建议的 235B-FP8。全量池生成应换 235B-FP8
  (`models/Qwen3-VL-235B-A22B-Instruct-FP8`, TP=2~4)。
- 数学层单测当前 7/7 通过 (含 position_ids 回归); hint QC 单测 13/13 通过。

### 9.5 全量 hint 池构建（2026-08-24/25，已落盘）

`scripts/sft_rl/prep_hint_pools.py` 构建与 SFT-then-RL 主线池 B 对齐的
PTD hint 生成输入池; `build_mmf_hints.py` 已扩展支持 Cauldron 的
`images=[{"image_url": <绝对路径>}]` 外置图片 (含 mime 嗅探) 与
`--shard-prefix`。

| 池 | 路径 (`fc-opd-storage/outputs/fc_opd/sft_rl/hint_pools/`) | 行数 | 说明 |
|---|---|---:|---|
| MMF 可验证 | `mmf_pool/` (64 shards) | **95,128** | 117,688 剔除 other 22,560; SFT/RL 重叠按论文口径保留 |
| Cauldron 全量 | `cauldron_pool_full/` (613 shards) | **918,254** | 14 闭式子集, 多轮会话按 QA 对展开, 与池 B 全量口径一致 |
| Cauldron 推荐 | `cauldron_pool_100k/` (59 shards) | **87,365** | 每子集 cap 6500 (seed 42); 与 MMF 95K 合计 ~183K, hint 生成量可控 |

Cauldron 展开规则: sft_full 每行是共享一张图的多轮会话, 只有第一轮带
`<image>` 占位符; 展开后每个 (user, assistant) 轮独立成样本, 占位符数
重整为 `len(images)`, GT 清洗 `Answer: B -> B`, uid =
`{source}:{image_hash}:{turn_idx}`。图片保持外置 `image_url` 引用 (零拷贝)。

**注意**: Cauldron 全量 918K hint 用 235B 生成约需 2+ 周, 不现实;
推荐 MMF 95K 全量 + Cauldron 100K 分层版 (合计 ~183K, 约为论文 ViRL39K
的 4.7 倍), 或先用 SFT ckpt 对 Cauldron 预筛 pass-rate 后只给失败层生成。
Cauldron RL 侧还需要一个按 source 的闭式 reward 扩展 (mmf_reward 目前
覆盖 letter/yesno/numeric/LaTeX, VQA 短答需 normalized exact match)。

## 10. 2026-08-25 全量 MMF hint 与复现运行准备

### 10.1 教师模型更替与全量结果

原计划使用的 `Qwen3-VL-235B-A22B-Instruct-FP8` 权重不完整 (24 个 shard
只存在 7 个且无 index), vLLM 静默加载残片后输出 `!×768` 退化文本。该目录已
删除; 全量生成改用本地完整 `qwen3.6-35B-A3B` (`qwen3_5_moe` VLM, cu132
vLLM 0.27.1 原生支持), 单卡 H200 部署。生成时显式传
`chat_template_kwargs={"enable_thinking": false}`, 避免默认 thinking 模式
输出完整解题过程。

MMF 95,128 行生成 + hard-reject retry 后最终落盘:

| 指标 | 数量 | 比例 |
|---|---:|---:|
| 总样本 | 95,128 | 100% |
| 有效 hint | **87,767** | **92.26%** |
| 空 hint / hard reject | 7,361 | 7.74% |
| 完全干净 (`hint_reason=""`) | **77,697** | **81.68%** |
| 有效但带 soft flag | 10,070 | 10.58% |

输出: `fc-opd-storage/outputs/fc_opd/sft_rl/hint_pools/mmf_hint/`
(64 shards, 4.2 GB)。空 hint 保留原始 RL 行, 但 `prompt_with_hint=[]`;
PTD mask 会排除这些行, GRPO 数据分布不变。

### 10.2 复现运行前的代码修复

- `run_grpo_mmf_ptd.sh` 默认训练集从旧 3K hint 切到 **95K MMF hint 池**;
- `TOTAL_TRAINING_STEPS / TRAINER_SAVE_FREQ / TRAINER_TEST_FREQ` 改为环境变量,
  便于 2-step GPU 冒烟;
- `engine_workers.compute_teacher_log_prob` 的 response attention 从
  `ones_like(responses)` 改为读取完整 student `attention_mask` 的 response 段,
  不再把右侧 padding 当有效 teacher context;
- 保持 2026-08-24 的 mRoPE 微批切片修复: 3-D teacher positions 按 dim=1
  切 batch, 而不是 dim=0 切 mRoPE 分量。

验证结果:

- `tests/sft_rl/test_ptd_jsd_loss.py`: 7/7 通过;
- `tests/sft_rl/test_build_mmf_hints_qc.py`: 14/14 通过;
- `tests/sft_rl/test_prep_hint_pools.py`: 3/3 通过;
- Hydra `ppo_trainer.yaml + algorithm.ptd.*` 配置解析通过;
- 真实 warmup processor 抽样 token 化 hint prompt: 11 条长度
  285-1,021 tokens, 无超过 `max_prompt_length=2048` 的样本。

### 10.3 GPU 冒烟入口

2-step 机制冒烟建议使用:

```bash
SFT_RL_MODEL=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_warmup_20260820_0640/global_step_1086/huggingface \
SFT_RL_GPUS=1,2,3,4,5,6,7 \
SFT_RL_MAX_SAMPLES=32 \
TOTAL_TRAINING_STEPS=2 \
TRAINER_TEST_FREQ=1 \
TRAINER_SAVE_FREQ=2 \
SFT_RL_NAME=qwen3vl_grpo_mmf95k_ptd_smoke \
bash scripts/sft_rl/run_grpo_mmf_ptd.sh
```

冒烟通过标准:

1. step 0/1 均完成 actor update, 无 mRoPE shape 错误;
2. 日志出现 `ptd/mask_ratio` 与 `ptd/active`, 且 0 < active < rollout 数
   (既有失败也有成功时) 或早期接近全失败;
3. `ptd/jsd_mean > 0`, `ptd/student_mass` 与 `ptd/teacher_mass` 接近 1;
4. GRPO policy loss / reward / entropy 无 NaN, 无显存 OOM;
5. checkpoint 目录可写, `global_step_*` 正常生成。

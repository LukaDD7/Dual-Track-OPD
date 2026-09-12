# PRISM (arXiv:2604.28123) 复现调研 — SFT-then-RL 增强方案评估 (2026-08-21)

> 背景: SFT-then-RL 轨道 (Qwen3-VL-8B, verl 0.9.0 V0, 8×H200) 正在评测中
> (grpo184 ckpt)。本文调研 PRISM: "Beyond SFT-to-RL: Pre-alignment via
> Black-Box On-Policy Distillation for Multimodal RL",评估如何在当前栈上复现,
> 以及哪些关键差异会导致效果下降。

## 0. TL;DR

- PRISM = 在 SFT 与 RLVR 之间插入一个"分布对齐"阶段: 策略与一个 MoE
  (perception + reasoning 双专家) 判别器做对抗 OPD。判别器把教师/学生
  (caption / think) 拆开打 BT 分,策略用 GRPO + 组内归一化优势去骗过判别器。
  论文在 Qwen3-VL 4B/8B 上 +4.4 / +6.0 avg (GRPO, 7 项 benchmark)。
- 官方资产齐全: 113K Gemini-3-Flash 蒸馏数据、1.26M 公共演示、5.9K 对齐集、
  2K/10K RL 集、120K 判别器 warmup 配对、预热的 MoE 判别器 ckpt (10.6GB),
  以及 4B/8B SFT/PRISM 后 ckpt。
- 复现最大障碍不是数据,是**工程**: PRISM 的 verl 是 0.6.1 深度魔改
  (mm_gad critic / grpo_multi_reward / teacher 字段透传), 我们的栈是
  verl 0.9.0 V0 + transformers 5.12, 不能直接合 patch, 需要按 0.9.0 架构重写
  核心 3-5 个文件。
- 效果下降的关键风险 (按影响排序):
  1. **数据/格式不同**: PRISM 全链路强制 `<caption>/<think>/<answer>` XML,
     教师是 Gemini-3-Flash 高保真蒸馏 (113K 全是 pass_rate=0 难样本);
     我们的 warmup 无 caption 目标、RL 奖励无 format 项。
  2. **SFT 规模**: PRISM SFT 1.37M 1 epoch; 我们 warmup 34.8K。
     SFT-107K vs 1.37M 消融差 -3.7 avg; 初始 gap 越大, 判别器越容易饱和,
     对齐增益越小。
  3. **KL=0**: alignment 阶段 PRISM 显式关 KL (反对锚定 SFT 初始化);
     直接复用我们 GRPO 的 kl=0.01 会抵消漂移校正。
  4. **判别器 warmup 缺失**: 判别器必须预热 (120K 配对), 否则 trivially
     separable → 信号无信息 (论文 w/o SFT 变体 -16.8)。
  5. **RL 集难度**: PRISM RLVR 用 pass@16 ∈ [0.2,0.8] 过滤的 2K (8B: 10K);
     我们 MMF RL 池全 pass_rate=0, 稀疏奖励会稀释对齐带来的收益。

## 1. 方法摘要 (从论文 v3 + 官方 repo 交叉核实)

### 1.1 三步流水线

| 阶段 | 内容 | 目标 |
|---|---|---|
| 1. SFT 冷启动 | LLaMA-Factory, 全参 LLM + 冻结 vision/projector, 1 epoch, 1.37M (113K Gemini 蒸馏 + 1.26M Gemini 家族公共演示) | 缩小与监督分布的初始 gap |
| 2. PRISM 对齐 | verl, 策略与 MoE 判别器联合训练 500 步 | 把后 SFT 策略拉回监督分布 |
| 3. RLVR | verl, GRPO/DAPO/GSPO 1500 步, 2K 难度过滤集 | 结果正确性 + 格式 |

### 1.2 对齐阶段机制

- 响应解析为 `<caption>` c 与 `<think>` t; 两个专家: 感知 D_v(x,c)、
  推理 D_r(x,t), 组合分 r = α·D_v + (1-α)·D_r, α=0.5。
- 判别器损失 (每个专家独立 BT):
  L_Dk = -log σ(D_k(x, y+_k) - D_k(x, y-_k)), y- 来自当前策略 rollout,
  y+ 为教师响应。**判别器与策略全程联合更新** (避免 reward staleness)。
- 策略: 每 prompt 采 N=16, temp=1.0, 判别器分做组内归一化优势, GRPO 更新,
  **KL 系数=0** (显式允许漂移校正)。
- 判别器初始化: 4×Qwen3-VL-2B dense upcycle 成 MoE (4 专家 top-2,
  decoder_sparse_step=1), 先在 120K 教师/学生 caption+cot 配对上 warmup
  (BT + load-balancing loss)。

### 1.3 训练超参 (论文 Table 3)

| 项 | SFT | 对齐 | RLVR |
|---|---|---|---|
| lr / schedule | 1e-5 / cosine | 1e-6 / constant | 1e-6 / constant |
| 步数 | 1 epoch | 500 (repo 脚本写 1500, 需核实) | 1500 |
| 全局 batch | 2 | 4 | 32 |
| max prompt / response | - / 8192 | 2048 / 6144 | 2048 / 8192 |
| rollout N / temp | - | 16 / 1.0 | 16 / 1.0 |
| KL | - | 0.0 | 0.001 (GRPO 脚本) |
| 奖励 | - | α=0.5 MoE 分 | 0.8·acc + 0.2·format |
| 硬件 | 8×H100-80G | 8×H100-80G | 8×H100-80G |

### 1.4 关键结论 (论文 + 消融)

- MoE 判别器 vs dense-4B: **-3.4 avg**; vs text-only: **-3.9 avg**;
  w/o 对齐: -4.4; w/o SFT: -16.8; SFT-107K (vs 1.37M): -3.7。
- 对齐后 checkpoint 本身精度与 SFT 相当 (分布校正而非答案优化), 增益体现在
  RLVR 之后; 且 token 更少。
- 8B 上 SFT 漂移更严重 (Instruct→SFT 平均 -5.2), 对齐的绝对增益也更大 (+6.0)。

## 2. 官方资产与成本

| 资产 | 位置 | 大小 | 用途 |
|---|---|---|---|
| gemini_distill (113K, 含图) | prism-vlm/gemini_distill | ~6.8GB (18 shards) | SFT 教师 |
| gemini_public_mmr1 (1.26M) | prism-vlm/gemini_public_mmr1 | ~9.9GB (49 files) | SFT 池 |
| 对齐集 5.9K | prism-vlm/rl_dataset: rl_training_data_5.9k.parquet | 463MB | Stage 2 |
| RLVR 集 2K / 10K | prism-vlm/rl_dataset: *_filtered*.parquet / *_10k.parquet | 120MB / 958MB | Stage 3 |
| MoE warmup 配对 120K | prism-vlm/rl_dataset: pairwise jsonl + images tar.gz | 763MB + 2.19GB | 判别器 warmup |
| MoE 判别器 ckpt | prism-vlm/Qwen3-VL-2B-4X-Moe-warmup-120k | 10.6GB (18 files) | Stage 2 critic |
| 官方 SFT ckpt | prism-vlm/Qwen3-VL-{4B,8B}-Instruct-SFT | ~17-35GB 量级 | 跳过 Stage 1 |
| 官方 PRISM ckpt | prism-vlm/Qwen3-VL-{4B,8B}-Instruct-SFT-PRISM | 同上 | 跳过 Stage 2 |

对齐集 schema (实测, 5978 行):
`images (bytes 内嵌), data_source=mm_gad, prompt (chat, 含强制
<caption>/<think>/<answer> system prompt), teacher_caption, teacher_cot,
teacher_response, answer, reward_model{ground_truth, style:rule}`。

warmup 配对 schema (实测): `id, image, question, teacher_caption,
teacher_cot, student_caption, student_cot`。

## 3. 复现路径

### 路径 A — 官方资产端到端复现 (保真, 隔离变量)

1. **数据**: 在有网节点下载 gemini_distill + rl_dataset + MoE ckpt + 官方
   8B SFT ckpt (跳过 Stage 1 最省; 总量 ~55-75GB, GPFS 剩余 650G 可行,
   注意共享盘压力)。GPU 节点用现有 offline 缓存流程
   (snapshot_download + HF_HUB_OFFLINE) 同步。
2. **验证 MoE ckpt 可加载**: 本地 transformers 5.12 已含
   `Qwen3VLMoeForConditionalGeneration` (qwen3_vl_moe), config 字段
   (decoder_sparse_step / moe_intermediate_size / 4 experts top-2) 兼容,
   先做 `from_pretrained` CPU 冒烟; 缺 `Qwen3VLMoeForTokenClassification`
   与 value head 类, 需补小补丁 (PRISM transformers-4.57 的
   `modeling_qwen3_vl_moe.py` +79 行可参考移植)。
3. **移植 verl 0.9.0 V0** (见 §4 清单), 冒烟: 4B 官方 SFT ckpt + 5.9K 对齐集
   + MoE ckpt, 2-4 GPU, ~50 步, 看 critic d_acc 上升 / 策略 reward 分离 /
   无 NaN。
4. **对齐 500 步 → RLVR**: 8B 官方 SFT ckpt → 对齐 → 官方 10K RLVR 集;
   同预算跑 SFT→RLVR 对照组 (对齐 500 步换算成 RLVR 步数), 在 PRISM 的
   7 项 benchmark (至少 MathVerse/MMMU-Pro 与我们重叠) 上对比。

### 路径 B — PRISM-lite on 自家数据 (与三臂可比)

1. 数据改造: 用 MMF 的 `caption` 字段 + `<think>` 长 CoT 构造
   teacher_caption / teacher_cot; 统一 SFT/RL/eval 输出为
   `<caption>/<think>/<answer>`; RL 奖励加 format 项 (0.8 acc + 0.2 fmt)。
2. 对齐阶段: 复用官方 MoE ckpt (不自训), KL=0, N=16, temp=1.0, 500 步;
   RL 阶段保持现有 mmf_reward + geometry3k。
3. 结论口径: 与三臂 (SFT-only / RL-only / SFT-then-RL) 直接可比; 但预期增益
   低于论文 (数据/格式差异, 见 §5)。

> 建议先 A 后 B: A 验证"机制是否有效", B 回答"对自家数据是否值得投入"。

## 4. verl 0.9.0 V0 移植清单 (PRISM 基于 0.6.1, 不能直接合)

PRISM 对 upstream verl 0.6.1 的改动 (官方 `difference/verl_diff_EN.md`):

| 文件 | 改动 | 0.9.0 移植要点 |
|---|---|---|
| `workers/critic/dp_critic.py` | +875/-64: mm_gad 双通道 (caption/cot) 打分, 格式解析 + token 边界定位, 4 次前向/样本, BT 损失 + 跨 rank valid_count sync, score_clip/长度校正/score_reg | 0.9.0 V0 的 dp_critic 结构不同, 需按新 batch 结构重写 update_critic 分支 |
| `trainer/ppo/core_algos.py` | +`grpo_multi_reward` 优势: caption/cot 分别组内归一化, 按 format/acc gating 组合 (多模式) | 0.9.0 已有 GRPO_VECTORIZED; 加 multi-reward 分支 |
| `trainer/ppo/ray_trainer.py` | teacher_response/caption/cot 从 batch 透传 critic, 不进 actor | 0.9.0 V0 的 batch keys / pop 逻辑位置不同 |
| `utils/dataset/rl_dataset.py` | 解析 teacher_* 列并 tokenize | 0.9.0 RL 数据集类重写 (V0), 需加同逻辑 |
| `utils/reward_score/` | mm_gad_no_llm / math_verify_with_format 等 (format 0.2 + acc 0.8) | 可直接移植, 纯 Python, 与版本无关 |
| transformers 4.57 | `Qwen3VLMoeForTokenClassification` + value head (moe/value_head.py) | 5.12 已含 qwen3_vl_moe 主干, 只需补 token-classification/value-head 类 |

依赖: `critic.use_mm_gad=True`、`algorithm.adv_estimator=grpo_multi_reward`、
`critic.model.path=<MoE ckpt>`、`critic.model.tokenizer_path=<策略 tokenizer>`。
注意官方脚本里 alignment `trainer.total_training_steps=1500` 与论文 500 步
不一致, 复现时以论文为准并记录。

## 5. 关键差异 → 效果下降风险表

| # | 差异 | 影响机制 | 预期影响 | 缓解 |
|---|---|---|---|---|
| 1 | 教师分布: Gemini-3-Flash 蒸馏 (113K 难样本, caption+think+answer) vs 我们 Qwen3-VL-235B CoT + 人类短答, 且无 caption 目标 | 判别器对齐目标 = 教师分布; 教师风格/格式不同, 学到的"对齐"不同 | **高**: 直接决定对齐收益 | 用官方 5.9K 集 (路径 A) 或先做格式统一 (路径 B) |
| 2 | SFT 规模: 1.37M vs 34.8K warmup | 初始 gap 大 → 判别器饱和 → 信号退化; 论文 SFT 107K vs 1.37M 差 -3.7 | **高** | 用官方 8B SFT ckpt; 或先扩 warmup |
| 3 | 输出格式: PRISM 全链路强制 XML, 格式错判 reward=0; 我们无 format 项 | mm_gad 判别器依赖 caption/cot 可解析; 格式失效则对齐信号全零 | **高** | 统一 system prompt + RL 奖励加 format 权重 |
| 4 | KL: 对齐阶段 0 vs 我们 GRPO 0.01 | KL 锚定 SFT 初始化, 直接抵消漂移校正 | **中-高** | 对齐阶段显式 use_kl_loss=False |
| 5 | 判别器: MoE 2B-4x warmup 120K vs 无 | 未 warmup → trivially separable; dense/text-only 判别器分别 -3.4/-3.9 | **中-高** | 用官方 MoE ckpt, 不自训 |
| 6 | RL 集难度: pass@16∈[0.2,0.8] 2K/10K vs 全 pass_rate=0 117K | 稀疏奖励 + 最难样本 → RL 阶段信号差, 混淆对齐增益归因 | **中** | RL 集按 pass_rate 过滤/采样 (runbook 已有思路) |
| 7 | 训练动态: 判别器与策略联合更新, N=16, batch=4 vs 我们固定判别器经验 | reward staleness → 策略过拟合 stale 判别器 | 中 | 按官方交替更新; 判别器 lr=1e-6 同步 |
| 8 | 引擎: verl 0.6.1 魔改 + transformers 4.57 vs 0.9.0 V0 + 5.12 | 移植 bug / 兼容性风险 (MoE 加载, mrope, flash-attn) | 中 (工程) | §4 清单 + 先 smoke |
| 9 | 评测口径: 7 项 lmms-eval + Qwen3-30B judge vs 我们 6 项 + 32B judge | 数字不可直接对比; MathVision/WeMath 是 PRISM 增益大头, 我们没测 | 中 | 至少补 MathVision/WeMath; judge 对齐 |
| 10 | 显存/耗时: 对齐阶段 4 次 MoE 前向/样本, 8B + MoE 同卡 offload | 每步耗时显著高于现有 GRPO 160s; 500 步对齐需数天 | 中 (成本) | 先 4B smoke 定吞吐, 再排 8B 预算 |

## 6. 建议行动

1. **立即 (低投入)**: 给 RL 奖励加 format 项 + RL 集按 pass_rate 过滤
   (PRISM 结论里免费的午餐), 这本身就大概率提升 SFT-then-RL。
2. **本周**: 下载官方资产 + MoE ckpt 加载冒烟 + verl 0.9.0 移植核心两文件
   (rl_dataset teacher 字段 + dp_critic mm_gad), 4B smoke 50 步验证机制。
3. **随后**: 路径 A 8B 官方端到端复现 (对齐 500 步 → RLVR), 与现有
   SFT→RLVR 同预算对照, 用重叠 benchmark (MathVerse/MMMU-Pro/MMBench) +
   补 MathVision/WeMath 对齐论文口径。
4. **决策点**: 若 A 增益显著, 再上路径 B (自家数据 PRISM-lite) 与三臂融合。

## 7. 参考

- 论文: https://arxiv.org/abs/2604.28123 (v3, 2026-06-29)
- 官方 repo: https://github.com/XIAO4579/PRISM
- HF 资产: prism-vlm/{gemini_distill, gemini_public_mmr1, rl_dataset,
  Qwen3-VL-2B-4X-Moe-warmup-120k, Qwen3-VL-{4B,8B}-Instruct-SFT,
  Qwen3-VL-{4B,8B}-Instruct-SFT-PRISM}
- 本文档数据均为 2026-08-21 实测 (下载 + schema 解析 + 官方代码核对)。

# RL 混合难度采样改造：MMF 难题 + Cauldron 广覆盖（2026-09-01）

> 用户结论：**GRPO/PTD-PO 不能只用 MMF 的难题**——上一轮 50-step PTD-PO 全部
> 训练在 MMF hint 池（95,128 行，100% pass_rate=0 最难样本）上，RL 收益只
> 体现在长 CoT benchmark，MCQ 短答案 benchmark 反而大幅下滑，其他分布看不到
> 收益。本文给出：诊断 → 混合配方（具体行数）→ 奖励修复 → 参数配置 →
> 4 卡启动命令。配套脚本与测试已落地并通过。

---

## 1. 诊断：为什么「只用 MMF 难题」导致 benchmark 下滑

### 1.1 评测证据（`sft_rl_full_results_ptdpo_20260831.md`）

ptd_grpo（MMF-only 训练）vs base 的 B 段对比：

| benchmark | base | ptd_grpo | Δ | 类型 |
|---|---|---|---|---|
| ReMI exact（诚实口径） | 0.2785 | **0.3623** | **+0.084** | 长 CoT 密集推理 ✅ |
| DynaMath | 0.6297 | 0.6275 | −0.002 | 持平 |
| ViewSpatial | 0.4231 | 0.2272 | **−0.196** | MCQ 短答案 ❌ |
| MMMU-Pro | 0.3960 | 0.2445 | **−0.152** | MCQ 短答案 ❌ |
| MMBench (judge) | 85.137 | 73.024 | **−12.1** | MCQ 短答案 ❌ |
| GQA | 0.6163 | 0.5542 | **−0.062** | MCQ 风格 VQA ❌ |

规律：**RL 的增益完全集中在训练分布（MMF = 最难的长 CoT 推理题）上；
凡是与训练分布不重叠的 MCQ/短答案综合题全部退化**。GRPO 是 on-policy
的，策略只在采样到的分布上收到奖励信号——MMF 池 100% 是
pass_rate=0 的最难样本，模型在这些题上 rollout 的成功率极低，
advantage 信号大量来自格式/抽取修正而非答题能力；同时训练分布里
**没有任何一行**接近 ViewSpatial/MMMU-Pro/MMBench/GQA 那种
「看图选字母/短答案」的题型，能力被推向长 CoT 专用。

### 1.2 数据证据

Cauldron（the_cauldron 14 个 closed-form 子集，与 SFT warmup 同源）恰好
是退化的那类分布：MCQ 字母 32,945 行 + 是非题 9,667 行 + 简单数值
18,833 行，共 **rule_ok 61,445 行**（按修复后的奖励口径）。SFT warmup
里 Cauldron 长行占 85% token、16 子集各 4K——模型的基本盘就在这里，
RL 阶段完全不采样它，等于把 SFT 学到的 MCQ 答题先验交给负梯度去漂移。

---

## 2. 混合配方（默认参数，`prep_mixed_rl_pools.py` 直接产出）

**核心机制**：verl `create_rl_sampler` 用 shuffle=True 的池级随机采样，
**池的构成比例 = 每个 batch 的期望构成**。所以改采样策略 = 改池子。

| 成分 | 行数 | 说明 |
|---|---|---|
| MMF hard | **12,000** | 从 95,128 行 hint 池按 source 分层（覆盖全部 source），带已有 hint |
| Cauldron broad | **19,356** | 14 子集各 cap 1,500（aokvqa 仅 44 行 rule-ok 全取；docvqa 1,427、textvqa 1,385 全取），图片字节内联 |
| **合计** | **31,356** | MMF 38.3% / Cauldron 61.7% |

50 步 × batch 16 = **800 prompts**，池是它的 ~39 倍，每 epoch 采样不重复。
每个 16-prompt batch 的期望构成：**MMF 难题 ~6.1 + Cauldron ~9.9**，
其中 Cauldron 每个子集期望 ~0.77 行——14 个域全部持续见到正奖励。

Cauldron 子集构成（rule_ok 口径，cap 1500）：

| 子集 | rule_ok | 入池 | | 子集 | rule_ok | 入池 |
|---|---|---|---|---|---|---|
| ai2d | 6500 | 1500 | | raven | 6500 | 1500 |
| scienceqa | 6069 | 1500 | | tallyqa | 6500 | 1500 |
| visual7w | 6500 | 1500 | | chartqa | 5393 | 1500 |
| tqa | 6451 | 1500 | | iconqa | 6332 | 1500 |
| vqav2 | 3273 | 1500 | | docvqa | 1427 | 1427 |
| vsr | 3345 | 1500 | | textvqa | 1385 | 1385 |
| infographic_vqa | 1726 | 1500 | | aokvqa | 44 | 44 |

比例依据：MMF 难题仍是 PTD 的主战场（ReMI +0.084 是真实增益，PTD
hint 在失败轨迹上的 dense supervision 也主要作用于难题），保留 ~38%
hard 保证推理继续涨；Cauldron ~62% 让每 batch 过半行落在退化的分布上，
GRPO 的正 advantage 能稳定注入。这个 6:10 的构成与 SFT warmup 数据
（长 CoT 61% 行）量级对齐，也和「junior_617 GRPO 全面退化而
sft_6939 GRPO 全面正向」的结论一致——起点相同、分布混合才是变量。

### 2.1 为什么 ratio 定在 MMF 38% / Cauldron 62%

- PTD 的 `threshold=1.0`：组内 accuracy < 1 才激活蒸馏， Cauldron
  简单行若模型已全对，会退化为纯 GRPO 的零 advantage 行（hint 不起作用，
  无害）；错的行则同样获得 hint 监督——Cauldron 行在 PTD 下不浪费。
- MMF 低于 50% 的下限来自 800-prompt 预算：若 MMF 只占 20%（160 行），
  50 步内每个 MMF source 平均只见 ~7 行，ReMI 增益会被稀释。
- Cauldron 不超过 70% 是避免回到「简单题 RL 无信号」的另一个极端
  （简单行全对 → zero-advantage 噪声行过多）。

---

## 3. 奖励修复（前置必要条件）

`scripts/sft_rl/mmf_reward.py`（已修，16/16 自测通过，17 个新单测通过）：

1. **GT 尾部句号剥离**：Cauldron GT 形如 `Yes.` / `No.` / `4.` / `B.`，
   原来的 yesno/letter `re.fullmatch` 全部 miss → 这些行在原奖励下
   **永远 0 分**，等于往 GRPO 里注入全负样本。现在 `compute_score`
   顶部统一 `gt = re.sub(r"\.+$", "", gt).strip()`。
2. **字母范围 A–E → A–H**：raven GT 均匀分布 A..H（F/G/H 合计 ~1,200 行），
   原匹配器够不到。
3. **向后兼容已验证**：对全 95,128 行 MMF hint 池做路由翻转扫描，
   只有 2 行（GT `400.` other→number），均为修复；MMF 干净 yesno
   与 A–E 字母行为不变。

> 注意：没有这个修复，混入 Cauldron 行的奖励是系统性 0 分，
> GRPO 会把「答对」也判成失败——比不用 Cauldron 更糟。**必须先修再训**。

---

## 4. 参数配置（沿用已验证的 4 卡配置，只换数据）

`scripts/sft_rl/run_grpo_mmf_ptd.sh` 的既有 env 全部可用；改变的只有
训练数据与池子。不变量（与上一轮 50-step PTD 完全一致，是已验证跑通
的组合）：

| 项 | 值 | 说明 |
|---|---|---|
| 模型 | `qwen3vl_sft_warmup_20260826_1324/global_step_6939/huggingface` | 同上一轮起点 |
| lr | 1e-6 | PTD 恒定 |
| PTD | coef 5e-2 / top_k 100 / threshold 1.0 / jsd_kl / use_ref_teacher | 同上 |
| KL | use_kl_loss=True, kl_loss_coef=0.01, low_var_kl | PTD 稳定项 |
| 采样 | rollout_n=8, batch 16, mini 8, max_prompt 2048, max_response 12288 | 同上 |
| 优化 | token-mean, clip 0.2/0.28, fused_kernels=False | PTD 强制 eager |
| 步数 | 50, save_freq 25, test_freq 10 | 短程方案 |

`SFT_RL_REWARD` 不需要覆盖——修复后的 `mmf_reward.py` 同一文件同时
正确处理 MMF（A–E/yesno/数值/LaTeX）与 Cauldron（A–H/带句号/纯数字）
GT；这是同一个奖励函数的两个子集。

## 5. 数据准备命令（GPU 实例上执行）

### 5.1 Cauldron hint 生成（8010 教师管线，幂等可续跑）

Cauldron 行目前没有 hint（`cauldron_pool_100k` 是裸池）。PTD 需要
`prompt_with_hint` 列；缺 hint 的行会退化成纯 GRPO，但既然教师服务器
和管线都在，建议给入池的 ~19,356 行都补上。流程与 MMF 完全一致
（`build_mmf_hints.py` 已支持 Cauldron 的 image_url 外链读盘）：

```bash
# 1) 起 8010 教师服务器（若未在跑）
HINT_SERVE_ONLY=1 bash scripts/sft_rl/serve_hint_gen.sh

# 2) 对 cauldron_pool_100k 生成 hint（读全部 87,365 行、QC 拒绝的行 hint 为空）
HINT_INPUT_DIR=$R/fc-opd-storage/outputs/fc_opd/sft_rl/hint_pools/cauldron_pool_100k \
HINT_OUT_DIR=$R/fc-opd-storage/outputs/fc_opd/sft_rl/hint_pools/cauldron_hint \
HINT_SHARD_PREFIX=cauldron_hint \
  bash scripts/sft_rl/run_hint_gen.sh
# 监控: $R/fc-opd-storage/logs/hint_gen_build_full.log
# 进度: cauldron_hint/hint_gen_progress.json （resume 按 sample_uid 幂等）
```

> 成本参考：MMF 95K 行 8 workers 跑完过夜量级；Cauldron 只需对
> rule_ok 的 61K 行有效生成（脚本会对全部行尝试，QC 硬拒的行留空，
> 无需干预）。等不及可以先 `--limit 20000`：混合脚本对空 hint 行
> 自动退化纯 GRPO，不阻塞启动。

### 5.2 混合池产出

```bash
# 3) 混合采样（CPU 即可，分钟级；--mmf-rows/--cauldron-per-subset 可调）
$R/envs/va-opd-qwen35-v090-cu132-r595-v1/bin/python \
  scripts/sft_rl/prep_mixed_rl_pools.py \
  --out-dir $R/fc-opd-storage/outputs/fc_opd/sft_rl/rl_mixed_31k
# 产出 rl_mixed_train__part_*.parquet（统一 {bytes,path} schema + hint 列）
# + rl_mixed_stats.json（构成快照，对表用）
```

### 5.3 val 集

沿用 `mmf_rl_3k/mmf_rl_val__part_0000.parquet`（150 行，与训练池无
泄漏：mmf_rl_3k 的 val 是从池外抽样、且 verl 只把 val 当 test_freq 探针
不进训练梯度）。若想同时监控 MCQ 侧，可加 Cauldron held-out
（`--out-dir` 里留 500 行不给训练）——本期先不做，靠 A/B 段评测判定。

## 6. 4 卡启动命令（GPU 0,1,2,3）

```bash
SFT_RL_GPUS=0,1,2,3 \
SFT_RL_MODEL=$R/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_warmup_20260826_1324/global_step_6939/huggingface \
SFT_RL_TRAIN="$R/fc-opd-storage/outputs/fc_opd/sft_rl/rl_mixed_31k/rl_mixed_train__part_*.parquet" \
SFT_RL_NAME=qwen3vl_grpo_mixed31k_ptd_sft6939_coef5e2_steps50 \
TOTAL_TRAINING_STEPS=50 \
TRAINER_SAVE_FREQ=25 TRAINER_TEST_FREQ=10 \
  bash scripts/sft_rl/run_grpo_mmf_ptd.sh
# 日志: $R/fc-opd-storage/logs/qwen3vl_grpo_mixed31k_ptd_sft6939_coef5e2_steps50.log
# ckpt: .../sft_rl/ckpt/qwen3vl_grpo_mixed31k_ptd_sft6939_coef5e2_steps50/global_step_{25,50}/huggingface
```

（PTD_COEF 等 env 留默认即 5e-2/100/1.0/jsd_kl。）

## 7. 验证与收表

- 单测：`pytest -q tests/sft_rl/`（新增 test_mmf_reward_cauldron.py
  9 例 + test_prep_mixed_rl_pools.py 8 例，基线 24 例全绿）。
- 训练后跑 `run_sft_eval_ladder_4gpu.sh`（A 段）+ B 段六项，与
  `sft_rl_full_results_ptdpo_20260831.md` §1/§2 同表对比：
  - 主判据：**ViewSpatial / MMMU-Pro / MMBench / GQA 相对 ptd_grpo
    （MMF-only）的回升幅度**，以及 ReMI exact 是否守住 ≥0.35。
  - 若 MCQ 回升但 ReMI 跌回 sft_6939 水平（≤0.347），说明 Cauldron
    占比过高，下轮 `--cauldron-per-subset 1000` 重配。
- 决策树：四臂对照后，若 mixed 臂在两类 benchmark 同时不劣于
  ptd_grpo 且 MCQ 显著回升，则定稿为 RL 数据配方。

## 8. 文件清单

| 文件 | 状态 |
|---|---|
| `scripts/sft_rl/mmf_reward.py` | 已修（句号剥离 + A–H），16/16 自测 |
| `tests/sft_rl/test_mmf_reward_cauldron.py` | 新增 9 例 |
| `scripts/sft_rl/prep_mixed_rl_pools.py` | 新增（真实池 smoke 通过：schema 统一、HF datasets 可加载、图片字节非空、seed 确定性） |
| `tests/sft_rl/test_prep_mixed_rl_pools.py` | 新增 8 例 |
| `scripts/sft_rl/run_hint_gen.sh` | 加 `HINT_SHARD_PREFIX` env（Cauldron 输出命名） |

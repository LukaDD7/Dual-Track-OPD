# SFT-then-RL 四臂全面评测结果 + 单源 SFT 消融 + PTD-PO 训练配置详情（2026-08-31）

> 本文记录 FC-OPD「SFT-then-RL」track（`codex/va-opd` 分支）最新一轮的完整评测结果、
> ReMI 评测 Bug 的定位与诚实重算、以及 GRPO / PTD-PO 各训练配置与数据使用详情。
> 与 `sft_rl_track_runbook_20260819.md` 互补：那篇是运行手册，本篇是**结果与配置事实**。
> 评测口径见 `target_benchmark_eval_runbook_20260825.md`（B 段六项）与
> `scripts/sft_rl/run_sft_eval_ladder_4gpu.sh`（A 段 4 卡并行）。

---

## 0. 四臂（+ 参考）定义

| 臂 | 模型 ckpt | 说明 |
|---|---|---|
| **base** | `$R/models/Qwen3-VL-8B-Instruct` | 未训练的基座（下界参考） |
| **sft_6939** | `.../qwen3vl_sft_warmup_20260826_1324/global_step_6939/huggingface` | SFT warmup 主臂 |
| **junior_617** | `$R/models/qwen3vl8b_mixed_sft_groupbalanced_617` | 师弟的 mixed-SFT 模型（raw SFT 下界参考） |
| **junior617_grpo50** | `.../qwen3vl_grpo_mmf_junior617_lr5e6_steps50/global_step_50/huggingface` | vanilla GRPO，从 junior_617 出发，50 step |
| **grpo50** | `.../qwen3vl_grpo_mmf_sft6939_lr5e6_steps50/global_step_50/huggingface` | vanilla GRPO，从 sft_6939 出发，50 step |
| **ptd_grpo** | `.../qwen3vl_grpo_mmf95k_ptd_sft6939_coef5e4_steps50/global_step_50/huggingface` | PTD-PO，从 sft_6939 出发，coef 5e-4→5e-2，50 step |
| **mmf_only_1ep** | `.../qwen3vl_sft_mmf122k_1ep/global_step_1774/huggingface` | 单源消融：MMF-only 113,537 train，1 epoch |
| **cauldron_only_1ep** | `.../qwen3vl_sft_cauldron16full_1ep/global_step_5934/huggingface` | 单源消融：Cauldron-16-only 379,787 train，1 epoch |

补充参考（仅 A 段，绘制 SFT 训练曲线用）：**sft_1000**（`.../20260826_1324/global_step_1000`）、
**sft_1086**（`.../20260820_0640/global_step_1086`，早期 SFT run）。

---

## 1. A 段：保留集 pass@1 / pass@8 / 格式合格率

- 数据：geo3k 官方测试集 601 行（held-out）+ MMF RL val 500 行。
- pass@1 = temp 0 单次；pass@8 = temp 0.7 采样 8 次取 ≥1 命中；格式合格率 = 从 pass@1
  输出中 `\boxed{` / `<answer>` / `answer:` 的可抽取比例（`run_sft_eval_ladder_4gpu.sh`
  内联统计，非训练指标）。

| 臂 | geo3k p@1 | geo3k p@8 | geo3k fmt | mmf p@1 | mmf p@8 | mmf fmt |
|---|---|---|---|---|---|---|
| base | 0.5391 | 0.7754 | 0.7055 | 0.1780 | 0.4400 | 0.4740 |
| sft_1000 | 0.3161 | 0.5940 | 0.3910 | 0.1820 | 0.4100 | 0.3440 |
| sft_1086 | 0.3195 | 0.6323 | 0.3827 | 0.2000 | 0.4560 | 0.3640 |
| sft_6939 | 0.3794 | 0.6889 | 0.4393 | 0.2980 | 0.5240 | 0.4940 |
| junior_617 | 0.3894 | 0.7238 | 0.5358 | 0.1680 | 0.4100 | 0.3020 |
| junior617_grpo50 | 0.2047 | 0.5458 | 0.2562 | 0.1700 | 0.3780 | 0.1980 |
| grpo50 | **0.4210** | **0.6955** | 0.4359 | **0.3400** | **0.5480** | **0.5120** |
| ptd_grpo | 0.3661 | 0.6689 | 0.4210 | 0.3080 | 0.5200 | 0.4580 |

要点：
- MMF val 上 **grpo50 全面领先**（p@1 0.3400 > ptd 0.3080 > sft_6939 0.2980），vanilla GRPO
  在保留集上仍是最强；ptd_grpo 也高于 sft_6939 起点但与 vanilla GRPO 有差距。
- geo3k 上 base 的 p@1(0.5391) 反而是最高——基座在几何上有强先验，SFT/RL 后波动，
  grpo50(0.4210) 在训练臂里最高。
- 格式合格率：grpo50 在 mmf 上最高(0.5120)；junior_617 在 geo3k 上最高(0.5358)。
- **junior617_grpo50（负结果）**：从师弟 mixed-SFT 出发的 vanilla GRPO 全面退化——geo3k p@1
  0.3894→0.2047（六臂最低）、格式合格率塌方（geo3k 0.5358→0.2562、mmf 0.3020→0.1980）。
  对照 grpo50（从 sft_6939 出发为正向）说明：GRPO 会放大起点的 answer-only 格式发散，
  junior_617 抽取率最低(0.8988，见 §4) 正是被 RL 反向放大的根因。

---

## 2. B 段：六项 benchmark（规则判分 + judge）

- GQA / DynaMath / ViewSpatial / MMMU-Pro / ReMI 为规则判分（`*_nojudge/summary.json`），
  MMBench 用 Qwen3-VL-32B-Instruct-FP8 judge（`*_judged/summary.json`）。
- 六项已全部完成（2026-08-31 收束；GPU 实例中途曾自动回收，经 `manuscript-sft-rl-gpu` 续跑补齐）。

| 臂 | GQA | DynaMath | ViewSpatial | MMMU-Pro | ReMI(见§4) | MMBench |
|---|---|---|---|---|---|---|
| base | 0.6163 | 0.6297 | 0.4231 | 0.3960 | 0.2785 | 85.137 |
| sft_6939 | 0.5541 | 0.6343 | 0.2435 | 0.2526 | 0.3469 | 72.938 |
| junior_617 | 0.6107 | 0.5333 | 0.3953 | 0.4006 | 0.1954 | 85.052 |
| junior617_grpo50 | 0.6101 | 0.5120 | 0.3937 | 0.4058 | 0.2458 | 86.082 |
| grpo50 | 0.5483 | 0.6345 | 0.3046 | 0.2526 | 0.3415 | 71.907 |
| ptd_grpo | 0.5542 | 0.6275 | 0.2272 | 0.2445 | 0.3623 | 73.024 |
| mmf_only_1ep | 0.3759 | 0.6253 | 0.0940 | 0.2312 | 0.3662 | 53.608 |
| cauldron_only_1ep | 0.4008 | 0.0565 | 0.2598 | 0.1960 | 0.1296 | 42.440 |

单源 1ep 消融要点（注意：这是「混合 3ep(sft_6939) vs 单源 1ep 全量」的数据层面归因，
非严格同 epoch 数对照）：
- **MMF 是长推理/ReMI 的主要贡献源**：mmf_only_1ep 的 DynaMath 0.6253 与
  sft_6939/grpo50/ptd_grpo(≈0.63) 同档；ReMI exact 0.3662 为全部八臂最高
  （> ptd_grpo 0.3623、sft_6939 0.3469）。
- **Cauldron 相对保 GQA/ViewSpatial，但严重伤数学推理**：GQA 0.4008、ViewSpatial 0.2598
  均高于 mmf_only，但 DynaMath 0.0565、ReMI exact 0.1296 大幅塌方。
- 两个单源臂的 MMBench 都显著低于混合臂（53.6 / 42.4 vs sft 系 71.9–73.0），
  说明 Cauldron 单独也不能解释混合 SFT 在 MMBench 上的下滑。
- 与 base 相比，两个单源 SFT 臂在 GQA、ViewSpatial、MMMU-Pro、MMBench 全部明显退化；
  其中 MMF 臂靠 DynaMath/ReMI 补回，Cauldron 臂则只剩 GQA/ViewSpatial 相对优势。

要点：
- 原四臂与参考的 DynaMath 都落在 0.51–0.63，sft_6939/grpo50/ptd_grpo(≈0.63) 略高于
  base(0.6297)、junior(0.5333)、junior617_grpo50(0.5120)；单源消融暴露出强分布效应：
  mmf_only 0.6253 保持同档，cauldron_only 0.0565 大幅塌方。
- **junior617_grpo50**：B 段六项与 junior_617 起点几乎持平（GQA 0.6101 vs 0.6107、ViewSpatial
  0.3937 vs 0.3953、MMMU-Pro 0.4058 vs 0.4006），50 步 GRPO 未带来 B 段增益；ReMI exact 0.2458
  （vs raw-SFT 0.1954）略有回升但仍六臂第二低。塌方集中在 A 段（geo3k/格式，见 §1）而非 B 段。
- ViewSpatial 与 MMMU-Pro 上 **base 与 junior_617 明显领先**训练 arm，sft_6939/grpo50/ptd_grpo 掉得较多
  （如 viewspatial ptd_grpo=0.2272、sft_6939=0.2435 vs base 0.4231）——长 CoT 的 SFT 在这些 MCQ 短答案
  综合题上出现格式/判卷口径倒退，ptd_grpo 在 ViewSpatial 上四臂最低。
- MMBench（judge）：base 85.1 ≈ junior_617 85.1 > sft_6939 72.9 ≈ grpo50 71.9 ≈ ptd 73.0。
- **ptd_grpo 画像**：ReMI 诚实口径 **0.3623 四臂第一**（> sft_6939 0.3469、grpo50 0.3415）；
  DynaMath 0.6275 与 sft/grpo50 同档；但 ViewSpatial/MMMU-Pro 与 MMBench 均与 sft/grpo50 同档下滑——
  PTD-PO 在长 CoT 密集推理上最强，MCQ 短答案综合题上未跑赢 vanilla GRPO。

---

## 3. RL / PTD-PO 训练配置详情

环境统一：verl 0.9.0 V0（`verl-qwen35-v090-cu132`），env
`va-opd-qwen35-v090-cu132-r595-v1`，CUDA toolchain `cuda132-toolchain`，4 卡 FSDP2。
三份启动脚本在同一 repo：`scripts/sft_rl/{run_grpo_mmf.sh, run_grpo_mmf_ptd.sh}`。

### 3.1 SFT warmup（`run_sft_warmup.sh`）—— 所有 RL arm 的起点

| 项 | 值 |
|---|---|
| 数据 | `warmup_t2/sft_warmup_train__part_*.parquet`：**103 分片 / 148,054 行**；val **3,032 行**。来源 = MMFineReason 长 CoT 分层 100K + the_cauldron 16 子集每子集 4K，长 CoT ~61% 行 / ~85% token |
| 模型 | `Qwen3-VL-8B-Instruct` |
| lr | 5e-5，cosine，warmup 10%，min_lr_ratio 0.1，wd 0.01，betas [0.9, 0.999]，clip_grad 1.0 |
| 轮数 | 3（对齐 arXiv:2604.23747 §A） |
| 长度/动态 bsz | max_length 12288（长 CoT 不截尾），truncation right，use_dynamic_bsz，max_token_len_per_gpu 98304，train_batch_size 64（软上界） |
| 引擎 | engine=fsdp2，strategy fsdp2，use_remove_padding |
| 保存 | save_contents=[model,hf_model]，max_ckpt_to_keep 3，resume_mode auto，save_freq 200 |

产出 checkpoint：`qwen3vl_sft_warmup_20260826_1324`（global_step 1000/6600/6800/**6939**）、
`_20260820_0148`、`_20260820_0640`（global_step_1086）。

### 3.2 vanilla GRPO（`run_grpo_mmf.sh`）—— 臂 grpo50 / junior_617 GRPO

| 项 | 值 |
|---|---|
| 模型 | sft warmup ckpt 的 huggingface 目录（grpo50 从 sft_6939 出发） |
| 数据 train | `mmf_rl_20k/mmf_rl_train__part_*.parquet`（**20,000 行，14 分片**） |
| 数据 val | `mmf_rl_20k/mmf_rl_val__part_0000.parquet`（**500 行**） |
| 算法 | adv_estimator=grpo，use_kl_in_reward=False，norm_adv_by_std_in_grpo=False |
| lr | **5e-6** |
| clip | clip_ratio_low=0.2 / clip_ratio_high=0.28（**非对称**），loss_agg_mode=token-mean |
| KL | use_kl_loss=**False**，entropy_coeff=0 |
| 采样 | rollout_n=8，train_batch_size 16，mini_batch 8，max_prompt 2048，max_response 12288 |
| token 预算 | actor log_prob / ref max_token_len_per_gpu = 24576 |
| 引擎 | fsdp2，use_fused_kernels=**False**（warmup ckpt lm_head=F32 强制） |
| 步数 | total_training_steps 50，total_epochs 1，save_freq 25，test_freq 10 |
| 奖励 | `mmf_reward.py::compute_score` |

产出：`qwen3vl_grpo_mmf_sft6939_lr5e6_steps50`（= grpo50 臂）、
`qwen3vl_grpo_mmf_junior617_lr5e6_steps50`（junior 起点对照）。

### 3.3 PTD-PO（`run_grpo_mmf_ptd.sh`）—— 臂 ptd_grpo（PTD-PO 第四臂）

在 GRPO 基础上叠加 PTD（privileged tutoring distillation，dense supervision on failed
trajectories），复用 verl 的 top-K 蒸馏管线（`algorithm.ptd.*` + `loss_mode=jsd_topk`）。

| 项 | 值 |
|---|---|
| 模型 | sft_6939（`qwen3vl_sft_warmup_20260826_1324/global_step_6939/huggingface`） |
| 数据 train | `hint_pools/mmf_hint/mmf_hint__part_*.parquet`（**64 分片 / 95,128 行**，含 hint 字段，见 §5） |
| 数据 val | `mmf_rl_3k/mmf_rl_val__part_0000.parquet`（**150 行**） |
| 算法 | adv_estimator=grpo，use_kl_in_reward=False + **`algorithm.ptd.*`** |
| lr | **1e-6** |
| KL | **use_kl_loss=True，kl_loss_coef=0.01，kl_loss_type=low_var_kl**（PTD 需 KL 项稳定 distill） |
| clip / loss_agg | 沿用 GRPO 默认（token-mean） |
| PTD 超参 | coef=**5e-2**，top_k=**100**，threshold=**1.0**，kl_direction=**jsd_kl**，all_trajectories=**false**，use_ref_teacher=**true** |
| 采样 | rollout_n=8，batch 16，mini 8，max_prompt 2048，max_response 12288 |
| 引擎 | fsdp2，**use_fused_kernels=False（PTD 强制）**：fused kernels 不产 top-K 辅助输出，需要 eager logits 路径 |
| 步数 | total_training_steps 50，save_freq 25，test_freq 10 |

产出：`qwen3vl_grpo_mmf95k_ptd_sft6939_coef5e4_steps50`（global_step 25/50）。
末步（step 50）观测：`critic/score/mean ≈ 0.4609`，global_seqlen max ≈ 27.2 万 token、
perf/throughput ≈ 615 tok/s。

> 命名中 `coef5e4` 为启动名沿用（早期 coef=5e-4 摸底后，正式 run 定为 5e-2）。

---

## 4. ReMI 评测 Bug + 诚实重算

### 4.1 Bug

`*_nojudge/summary.json` 里 ReMI 行用的是 `scoring=normalized_exact_diagnostic`（
`project_summary._summarize_replay`）：**分母只统计「可抽取/已匹配」的行**，把
「模型答案无法被规则抽取」的行排除在外，导致准确率虚高、并掩盖模型发出的
answer-only 格式发散（例如模型明明要求只答答案却吐 CoT）。

证据：base（未训练基座）ReMI 显示 **0.9688**，明显高于 sft_6939(0.4974)/grpo50(0.4809)/junior_617(0.5023)，
与直觉矛盾——并非 base 更强，而是 base 的抽取率低（漏掉的难行更多），分母被砍小。

### 4.2 诚实口径：`remi_reeval.py --mode exact`

`scripts/sft_rl/remi_reeval.py --mode exact`（只读、无 GPU、无网络）做 task-aware 精确匹配，
**分母 = 全文件 2,600 行**（13 任务 ×200，含无法抽取的行）：

| 臂 | correct | extractable | 抽取率 | **exact（全分母 2600）** | summary.json 里被虚高的值 |
|---|---|---|---|---|---|
| base | 724 | 2492 | 0.9585 | **0.2785** | 0.9688 |
| sft_6939 | 902 | 2596 | 0.9985 | **0.3469** | 0.4974 |
| junior_617 | 508 | 2337 | 0.8988 | **0.1954** | 0.5023 |
| junior617_grpo50 | 639 | 2525 | 0.9712 | **0.2458** | 0.5224 |
| grpo50 | 888 | 2598 | 0.9992 | **0.3415** | 0.4809 |
| ptd_grpo | 942 | 2596 | 0.9985 | **0.3623** | 0.5028 |
| mmf_only_1ep | 952 | 2594 | 0.9977 | **0.3662** | 0.5381 |
| cauldron_only_1ep | 337 | 2492 | 0.9585 | **0.1296** | 1.0000 |

结论：
- 诚实口径下顺序完全反转：**mmf_only_1ep(0.3662) > ptd_grpo(0.3623) > sft_6939(0.3469) ≈
  grpo50(0.3415) > base(0.2785) > junior_617(0.1954) > cauldron_only_1ep(0.1296)**。
- ptd_grpo 抽取率 0.9985 与 sft_6939 持平、correct=942 为 RL 四臂最高，说明 PTD-PO 在 ReMI 上**未出现
  answer-only 格式发散**。此前 0.5242「低抽取率」是漏传 `--label-jsonl` 的口径错误（task 判定退化、
  只抓 boxed/answer 标记，且 per_task 塌缩为单一 unknown），非模型真实行为；补传 sidecar 后与其它臂同口径。
- grpo50 与 sft_6939 抽取率都 ≈0.999（几乎全部可抽取），说明 RL 后 answer-only 格式收敛；
  junior_617 抽取率最低(0.8988)——其 answer-only 格式发散的样本最多（ReMI 上短板）。
- 后续所有对表必须用 `remi_reeval.py --mode exact` 的全分母数值，禁止引用 summary.json 的
  `normalized_exact_diagnostic`。`--mode judge`（Qwen3-VL-32B-Instruct llm-as-judge）仅作
  对「可抽取但错」样本的格式对齐诊断，不替代 exact。

---

## 5. 数据使用详情

### 5.1 MMFineReason RL 采样（`sample_mmf_rl.py`，seed 42）

- 源：`MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking`，共 122,603 行，其中 answer 空 4,915 行。
- 规则可验证 GT 形式过滤后：**95,128 行**（letter 21,115 / yesno 922 / pure_number 33,261 /
  has_number 39,830），丢弃「other」无数字自由文本 **22,560** 行（否则 GRPO 首轮注入 ~0 奖励噪声）。
- warmup 去重默认 OFF（同域 RL 是有益的，非泄漏）。

产物两档：
| 集 | pool | train | val | 用途 |
|---|---|---|---|---|
| `mmf_rl_20k` | 95,128 | 20,000（14 分片） | 500 | vanilla GRPO / RL-only arm 训练 |
| `mmf_rl_3k` | 91,954（额外丢 warmup_overlap 3,174） | 3,000 | 150 | PTD-PO 的 val |

`mmf_rl_3k` 训练源分布（top）：MMR1 1,651 / GameQA-140K 453 /
FineVision-visualwebinstruct 272 / BMMR 216 …

### 5.2 PTD hint 池（`prep_hint_pools.py --pool mmf` → `build_mmf_hints.py`）

- 池：MMFineReason 规则可验证子集 → **95,128 行（64 分片）**，`question/images/reward_model.ground_truth/extra_info` schema。
- hint 生成（`run_hint_gen.sh`）：教师 **Qwen3.6-35B-A3B**（api_base `127.0.0.1:8010/v1`），
  system_prompt 版本 `53996bfc0313`（zero-spoiler 规则：不泄露答案/选项字母/自算数字），
  max_new_tokens 768，temperature 0.6，n_rows 95,128，hint_ratio 0.8168。
- 生成结果：ok **77,697**；硬拒因 answer_leak_letter 3,355 / answer_leak_component_numeric 5,413 /
  answer_leak_numeric 232 / answer_leak_substr 593 / answer_leak_component_word 198 /
  answer_leak_word 34 / too_long 229 / cot_degenerate 80。
- recheck（2026-08-25）：kept 83,248，already_empty 11,880 → **最终 64 分片合计 95,128 行，
  其中 87,767 行有有效 hint、7,361 行 hint 为空**（PTD 在空 hint 行退化为普通 GRPO）。

### 5.3 保留集评测数据

- geo3k 官方测试集（held-out）601 行：`geo3k_grpo_val_test.parquet`。
- MMF RL val 500 行：`mmf_rl_20k/mmf_rl_val__part_0000.parquet`。

---

## 6. 复现与续跑状态（2026-08-31）

- A 段 + B 段四臂（+ 参考，含 **junior617_grpo50**）全部跑完，2026-08-31 收表：§2 B 段表与 §4 ReMI
  exact 已补齐。junior617_grpo50 用同一 4 卡阶梯（`run_sft_eval_ladder_4gpu.sh`）跑完并已并入 §1/§2/§4
  （ReMI 用 `--mode exact` 全分母 **0.2458**，非 summary.json 虚高 0.5224）。
  ptd_grpo B 段历经 GPU 实例自动回收、由 `manuscript-sft-rl-gpu` 续跑完成（监控
  `fc-opd-storage/logs/bonly_monitor.sh` 现报 `DONE=1`）。
- 2026-09-02 增补 **mmf_only_1ep** 与 **cauldron_only_1ep** 单源 SFT B 段消融：MMF final step
  1774，Cauldron final step 5934；两者均按同一六项 benchmark + ReMI `--mode exact` 全分母
  口径并入 §2/§4。Cauldron SFT 曾因 GPU 实例回收从 step 4000 续跑完成。
- 已知后续项：ReMI summary.json 的虚高分母尚未在 `project_summary.py` 内修复（用
  `remi_reeval.py` 旁路），可在对表前决定是否回修主链路。

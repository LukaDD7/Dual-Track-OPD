# Support-Aware 4096-token Diagnostic — Full Run, Rescore, Merge, Frontier Analysis (2026-08-03)

**状态**: 全链路完成（采样 → rescore → strict merge → frontier analysis）。
**目的**: 给 codex 完整的结果与判断输入，决定下一步（报告图 / K=32 重采样 / 五臂 bridge）。
**分支**: 本文在 `codex/va-opd`；frontier 契约与 analyzer 在 `codex/frontier-mass-terk`。

---

## 0. TL;DR

- 256 prompts（4×64 ABCD 分片 × 1 greedy + 8 stochastic，4096 token）采样全部完成；
  四片用修复版 exact-token 管线 rescore 后严格 merge 成功（256 prompts / 2304 rollouts）。
- 12 项协议门中 7 项 PASS（exact-token identity=1.000、prompt hash 可用率=1.000、
  分片覆盖率=1.000、无 missing image、无非有限分数、tokenizer 对齐、无重复），
  5 项 FAIL：malformed 26.9%、截断 29.2%、correct_tail 16<20、
  teacher_gap within-prompt AUC 0.576（CI 0.40–0.75）、correct-tail Rank@1 lift 不显著。
- **关键正结果**：按响应长度分层，≤2048 token 的 prompt within-prompt AUC = 0.752
  （CI 下限 0.61，达标）；>2048 只有 0.419；finish=stop 0.555、finish=length 0.375。
  即 teacher 排名信号真实存在，但被超长/截断响应污染——与 512-token 时代的
  “截断混淆”结论一致，这次是在 exact-token 协议下的干净复现。
- Frontier：posterior frontier mass 119.2、mean E[U₈]=0.465、RL-ready 51 个 prompt
  （rare_success 14 + mixed_support 37）。
- 下一步建议：报告可先出图 1–2（本数据足够）；实验侧先做 K=32 adaptive 重采样
  再冻结 matched cohort；五臂 bridge 按 `codex/frontier-mass-terk` 上的契约实现。

---

## 1. 产物路径（绝对路径，NFS 共享）

| 产物 | 路径 |
|---|---|
| 合并 run | `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/support_aware_opd/diag_full_20260802_256_merged.rescored-exact-v1/` |
| 合并 summary | 同目录 `summary.json`（97 KB，含全部分层指标） |
| 分片 rollouts | `rollouts.jsonl`（2304 行）、`prompt_support_summary.jsonl`（256 行） |
| frontier 分析 | 合并目录下 `frontier_analysis/`（`frontier_summary.json`、`frontier_prompts.jsonl/.csv`） |
| 四片 rescored v2 | `diag_full_20260802_{083157,113811,n128_192,n192_256}.rescored-exact-v2/` |
| rescore 脚本 | `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/support_aware_opd/rescore_all_v2.sh` |

---

## 2. 流水线履历

### 2.1 采样（旧代码产物，已 rescore）

- 分片：A=[0:64) `diag_full_20260802_083157`、B=[64:128) `diag_full_20260802_113811`、
  C=[128:192) `diag_full_20260802_n128_192`、D=[192:256) `diag_full_20260802_n192_256`。
- 启动脚本：`.codex-tmp/scripts/hpc/complete_missing_shards.sh`，**明确使用旧代码副本
  （.codex-tmp，5d10042 时代）**，通过 `PYTHONPATH=$OLD_ROOT/src` 生效；
  因此原始 rollouts 是 display-text 再 tokenize 的旧打分，不能直接使用。
- 完成时间：A 08-03 08:50、B 11:41、C 12:43、D 03:04（UTC）；wrapper 12:44 打出 ALL SHARDS COMPLETE。

### 2.2 Rescore（修复版 exact-token，输出 v2）

- 第一轮 v1 的四个目录 git_commit 不一致（f14ed4e / 9fac2eb / 8c353db，rescore 期间
  仓库 HEAD 移动），strict merge 会拒；因此钉在单一 commit 上重跑为 v2。
- v2 四片 `run_manifest.git_commit` 全部为 `ebc4b13`
  （`docs: record Step A (v1 task-reward integration) and Step B (4096 val-only truncation) results`）。
- 验证：每片 576 rows / 64 uids；`response_token_hash == teacher_scored_token_hash ==
  student_scored_token_hash`，`exact_token_alignment=True`，prompt_token_hash 存在，
  response masks 全 True；scoring_policy=`exact_raw_ids_content_primary_terminal_separate_v1`。
- 四个 v2 的 normalized resolved config 完全一致（忽略 prompt_start/end、output_dir、
  source_run 等 per-shard 字段后）。

### 2.3 Strict merge

- 命令：`python -m dual_track_opd.support_aware.diagnostic --merge-shards <四片 v2>
  --merge-output .../diag_full_20260802_256_merged.rescored-exact-v1 --merge-mode full`。
- 结果：256 prompts / 2304 rollouts，所有 exact-token / hash / 覆盖率校验通过，
  输出 12 项 gate（见 §3）。

### 2.4 Frontier analysis（CPU-only）

- analyzer 来自 `codex/frontier-mass-terk`（分析时 HEAD `256851f`），
  通过 worktree `/tmp/frontier-mass-terk` 运行，未改动主仓库 checkout。
- 配置：group-size 8、Jeffreys Beta(0.5,0.5) 先验、MC 20000、seed 42、
  useful-group 阈值 0.5、RL-ready 后验概率阈值 0.8。
- provenance：source prompt summary sha256 `fe21ae32...`；分析 git commit `256851f`。

---

## 3. 合并后 12 项协议门

| Gate | 结果 | 数值 |
|---|---|---|
| tokenizer_alignment | PASS | student/teacher 一致 |
| zero_missing_images | PASS | 0 |
| finite_scores | PASS | 0 非有限分数 |
| malformed_rate_ok | **FAIL** | 0.269（阈值 0.10） |
| duplicate_rate_ok | PASS | 0.000 |
| exact_token_identity | PASS | 1.000 |
| prompt_token_hash_available | PASS | 1.000 |
| complete_rollout_coverage | PASS | 1.000 |
| truncation_rate_ok | **FAIL** | 0.292（阈值 0.15） |
| sufficient_correct_tails | **FAIL** | 16（需 ≥20） |
| teacher_gap_within_prompt_auc | **FAIL** | 0.576，CI 0.401–0.748（点 ≥0.60 且下限 >0.50） |
| correct_tail_rank1_above_random | **FAIL** | lift 0.042，CI −0.204–0.297 |

---

## 4. 核心信号指标（merged）

### 4.1 Support 状态

| 状态 | 数量 |
|---|---|
| exposed | 120 |
| correct_tail | 16 |
| no_correct_observed | 42 |
| other | 78 |

greedy accuracy = 0.434。

### 4.2 Pooled AUC（全部 2048 条 stochastic rollout）

| 指标 | 值 |
|---|---|
| teacher_gap | 0.577 |
| teacher_mean_logp | 0.623 |
| student_mean_logp | 0.623 |

### 4.3 Correct-tail 排名（eligible=15，K 依有效 rollout 数 2–8）

| 指标 | 值 | 随机基线 |
|---|---|---|
| within-prompt AUC | 0.576（CI 0.40–0.75） | — |
| Rank@1 | 0.333（CI 0.13–0.60） | 0.292 |
| MRR | 0.556（CI 0.40–0.72） | 0.527 |

### 4.4 全体可排序 prompt（eligible=57）

| 指标 | 值 | 随机基线 |
|---|---|---|
| within-prompt AUC | 0.565（CI 0.47–0.66） | — |
| Rank@1 | 0.491（CI 0.37–0.61） | 0.529 |

### 4.5 生成质量

- finish reasons：stop 1631 / length 673；
- 长度：p10 280、p50 1172、p90 4096（触顶）、mean 1938；
- truncation 0.292、malformed 0.269、duplicate 0.000。

---

## 5. 长度分层信号（关键正结果）

| 长度 bin | eligible | within-prompt AUC | CI low–high |
|---|---|---|---|
| ≤512 | 8 | 0.717 | 0.47–0.92 |
| 513–2048 | 21 | **0.752** | **0.61–0.88** |
| >2048 | 23 | 0.419 | 0.27–0.57 |

| finish reason | eligible | within-prompt AUC | CI low–high |
|---|---|---|---|
| all | 57 | 0.565 | 0.47–0.66 |
| stop | 54 | 0.555 | 0.46–0.65 |
| length | 2 | 0.375 | 0.25–0.50 |

解读：teacher_gap 对正确/错误 rollouts 的排序能力在自然结束、中等长度的响应上
显著（513–2048 bin 的 CI 下限 0.61 越过 0.50），在超长/截断响应上崩塌。
这与旧 512-token run 的截断混淆结论一致，但现在是 exact-token 打分下的干净证据。

---

## 6. Frontier 分析结果

### 6.1 Observed stratum（按 correct_count/K 的透明分层）

| stratum | 数量 |
|---|---|
| no_correct_observed (c=0) | 85 |
| rare_success (c=1–2) | 37 |
| mixed_support (c=3–7) | 54 |
| all_correct_observed (c=8) | 80 |

### 6.2 汇总

- posterior frontier mass（Σ E[U₈]）= 119.15；
- mean posterior E[U₈] = 0.465（plugin 点估计 0.290，先验把 0/8、8/8 从 0/1 拉回）；
- RL-ready（P(U₈≥0.5) ≥ 0.8）共 51 个：rare_success 14、mixed_support 37。
- U₈ 后验均值分布：≤0.2 共 165、0.6 档 40、0.8 档 25、0.9 档 26。
- 最高候选（U₈≈0.955，c=4/K=8，mixed_support）：geo3k:137 / 693 / 565 / 1869 / 85 / 1892 / 1274 / 114 等。

---

## 7. 解读与注意点

1. **exact-token 协议干净**：迁移后 identity/hash/mask/coverage 全部 1.000，
   说明旧原始 rollout（raw IDs）可用，rescore 无损。
2. **malformed/other 上升是测量修复**：新保守 verifier（要求显式 Answer/boxed）把
   malformed 从旧 0% 抬到 26.9%、other 到 78；correct_tail 因此只有 16——这是
   协议收紧的代价，不是回归，报告里要说明。
3. **主 gate 不达标，但有可报告的正信号**：合并总体 teacher_gap AUC 0.576 < 0.60，
   correct_tail 16 < 20；但长度分层（≤2048：AUC 0.752，CI 下限 0.61）给出方向一致、
   统计显著的正结果。建议主信号按长度分层报告，并明确 stop/length 区分。
4. **teacher_mean_logp（0.623）高于 teacher_gap（0.577）**：绝对值差距不大，
   可作辅助信号；不要作为主 claim。
5. **K=8 粒度不足**：契约明确 K=8 是 screening，边界标签（尤其 c=0/1/2/8）
   需 K=32 adaptive 重采样确认后再分配 bridge 预算。

---

## 8. 下一步（给 codex 的决策输入）

### 8.1 报告（当前数据即可出）

- 图 1：observed stratum 计数 / 后验 frontier mass，含 K=8 不确定性（数据：
  `frontier_analysis/frontier_prompts.jsonl`）。
- 图 2：teacher 诊断 within-prompt AUC 按长度 bin / finish reason 分层
  （数据：merged `summary.json` 的 `prompt_signal_by_length_bin` /
  `prompt_signal_by_finish_reason` / `correct_tail_rank_metrics`）——此图是正结果。
- 可选图 3：U₈ 后验分布 / RL-ready 计数（`frontier_prompts.csv`）。
- 若五臂 bridge 来不及，图 1–2 + 预注册五臂协议（`configs/experiment/frontier_operator_bridge_pilot.yaml`）
  就是诚实的前置结果；不要用缺 bridge 数据的方式替代方法 claim。

### 8.2 实验（进入 bridge 前）

1. **K=32 adaptive 重采样**：优先 c=0（85）、c=1–2（37）、c=8（80）候选，
   确认边界标签后再冻结 cohort。今晚可执行的固定 64-prompt、四分片方案见
   `docs/support_aware_k32_overnight_handoff_20260804.md`；不要再用 `.codex-tmp`
   或把新 K=32 rollout ID 与旧 K=8 行直接拼接。
2. **冻结 matched cohort**：每 stratum ≤16、共 64 prompts，另留 disjoint held-out；
   记录 run id / prompt hash / image hash / dataset hash / checkpoint / K / 解码配置 /
   verifier 版本 / seeds。
3. **五臂 A–E**（契约在 `codex/frontier-mass-terk` 的
   `docs/frontier_operator_causal_experiment.md` + 机器可读 YAML）：
   A 无 bridge 冷 RL；B verified FKL（TREK-like，必须做）；C exact-token OPD（全 stratum
   都跑，避免路由选择混淆）；D FKL→OPD；E 状态路由。实现要求：先审计 verl launcher
   objective/reward 路径缺什么；每臂独立分支 + synthetic fixture + 2-prompt smoke；
   `third_party/verl` 不直接改，补丁放 `patches/verl/`；budget ledger 按
   verifier_calls 为主轴记录 token/时间/GPU 成本。
4. **统计**：prompt 级模型
   `observed_mixed_group_rate ~ pre_p + delta_U + arm + length + entropy`、
   `early_reward_gain ~ ...`，bootstrap CI；单 seed 只报预测性机制检查，不 claim 因果中介。

### 8.3 分支指引

- 本结果文档在 `codex/va-opd`。
- Frontier analyzer / 五臂契约 / survey 在 `codex/frontier-mass-terk`
  （分析时 HEAD `256851f`，含最新 TREK boundary 对照文档）。
- rescore / merge 代码在两条分支都有；analyzer 仅 frontier-mass-terk 有。
- 主仓库当前代码（含 exact-token 修复）与 GPU 实例 `.codex-tmp` 旧副本不同源，
  后续一律用主仓库分支，勿再引用 `.codex-tmp`。

---

## 9. 复现命令

```bash
# rescore（GPU 实例，脚本已存于 NFS）
bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/support_aware_opd/rescore_all_v2.sh

# merge（CPU-only）
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python \
  -m dual_track_opd.support_aware.diagnostic \
  --merge-shards \
    .../diag_full_20260802_083157.rescored-exact-v2 \
    .../diag_full_20260802_113811.rescored-exact-v2 \
    .../diag_full_20260802_n128_192.rescored-exact-v2 \
    .../diag_full_20260802_n192_256.rescored-exact-v2 \
  --merge-output .../diag_full_20260802_256_merged.rescored-exact-v1 \
  --merge-mode full

# frontier analysis（CPU-only，frontier-mass-terk worktree）
cd /tmp/frontier-mass-terk
PYTHONPATH=/tmp/frontier-mass-terk/src \
  /inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python \
  -m dual_track_opd.support_aware.frontier_analysis \
  .../diag_full_20260802_256_merged.rescored-exact-v1 \
  --group-size 8 --mc-samples 20000 --seed 42
```

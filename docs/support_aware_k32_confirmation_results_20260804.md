# Support-Aware K=32 Confirmation — 2026-08-04 结果

## 0. 一句话结论

K=32 四片全部完成、strict merge 通过（`k32_validation.json` → `valid: true`，无协议错误）。
统计门仍未达标（整体 teacher_gap AUC 0.564、correct_tail 11 < 20、malformed 20.2%、truncation 23.6%），
但 K=8→K=32 重采样给出明确校准信息：RL-ready 候选 18 → 27（净 +9），
长度分层正信号复现（≤512 tokens AUC 0.792，CI 下限 0.625 达标）。
**按计划停在 K=32 confirmation，未启动五臂 bridge。**

---

## 1. 产物路径（NFS 共享，绝对路径）

| 产物 | 路径 |
|---|---|
| K=32 合并 run | `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/support_aware_opd/diag_full_k32_20260804_merged/` |
| 合并 summary | 同目录 `summary.json` |
| 合并 rollouts | `rollouts.jsonl`（2112 行） |
| 合并校验 | 同目录 `k32_validation.json`（`valid: true`） |
| git 分歧审计 | 同目录 `git_divergence_audit.json` |
| frontier（K=32 口径） | 同目录 `frontier_analysis/`（`frontier_summary.json`、`frontier_prompts.jsonl/.csv`） |
| K8→K32 对比 | 同目录 `frontier_comparison/k8_to_k32_transitions.csv/.jsonl` |
| smoke run | `diag_full_k32_20260804_smoke2/`（2 prompts × 33，验证用，不入 merge） |
| 四片 | `diag_full_k32_20260804_s{0,1,2,3}_{0_16,16_32,32_48,48_64}/` |
| 重启脚本 | `$OUT/support_aware_opd/restart_k32_gpu_20260804.sh` |

---

## 2. 流水线履历

### 2.1 采样

- cohort：`k32_cohort_20260804/`（64 prompts，16/stratum，按 K=8 分层，seed 20260804，
  构建时 git `15de721`）。
- smoke（01:29–03:06 UTC）：2 prompts × 33 rollouts；exact identity / prompt hash /
  coverage 全部 1.0，missing=0，non-finite=0，`SMOKE OK`。
  geo3k:1124 的 33 条与昨晚部分 run **逐字节一致**（0 处差异），生成+exact-token 打分全链路可复现。
- 四片（03:08 起）：s0/s1/s3 从昨晚部分进度 `--resume` 续跑；s2 已完成直接跳过。
  完成时间：s3 04:53、s0 05:21、s1 07:15（UTC）。

### 2.2 中断与重启（必须记录）

- 昨晚（08-03）实例被平台自动回收，四片被杀时 s0=11/16、s1=9/16、s2=16/16（已完成）、s3=12/16；
  日志无报错（errors 全空、无 traceback）。
- 08-04 新实例用修复后的 `restart_k32_gpu_20260804.sh` 重启（修复了 teacher 启动时
  `PATH=$PY:$PATH` 把二进制路径当目录的 bug），smoke 通过后自动续跑四片。

### 2.3 git 分歧（merge 前必须说明）

- s2 记录 `git_commit=15de721`；s0/s1/s3 记录 `46dfb93`。原因：四片运行期间
  qwen35 工作线在同一仓库推进 HEAD（08-04 01:39–02:52 多个 commit，之后又有 merge/commit）。
- diff 审计（`15de721..46dfb93`）：只涉及 qwen35 文档/启动脚本、
  `fc_opd/prompt_contracts.py`（新增 boxed_only 函数，原 `geometry3k_training_prompt` 逻辑未变）、
  `fc_opd/verl_dataset.py`（qwen35 训练路径）。**support_aware 诊断与 teacher_service
  均不 import 这两个模块**，K=32 管线代码逐字节相同。
- 处理：merge 时显式启用 `DTOPD_ALLOW_GIT_DIVERGENCE=1`，在合并目录写入
  `git_divergence_audit.json`（记录各分片 commit 与审计说明）；除 git 元数据一致性外，
  其余 strict-merge 检查（tokenizer/model/数据集 hash、UID 切片、prompt_version、
  scoring policy、完整性）全部照常执行。

---

## 3. 合并后 12 项协议门

| Gate | 结果 | 数值 |
|---|---|---|
| tokenizer_alignment | PASS | student/teacher 一致 |
| zero_missing_images | PASS | 0 |
| finite_scores | PASS | 0 非有限分数 |
| malformed_rate_ok | **FAIL** | 0.202（阈值 0.10） |
| duplicate_rate_ok | PASS | 0.000 |
| exact_token_identity | PASS | 1.000 |
| prompt_token_hash_available | PASS | 1.000 |
| complete_rollout_coverage | PASS | 1.000 |
| truncation_rate_ok | **FAIL** | 0.236（阈值 0.15） |
| sufficient_correct_tails | **FAIL** | 11（需 ≥20） |
| teacher_gap_within_prompt_auc | **FAIL** | 0.548，CI 0.391–0.713（点 ≥0.60 且下限 >0.50） |
| correct_tail_rank1_above_random | **FAIL** | lift −0.009，CI −0.182–0.201 |

`k32_validation.json`：`valid: true`，64 prompts、K=32、2112 行，`protocol_errors: []`。
统计 gate 失败按设计“记录但不抑制”（`gate_failures_are_reported_not_suppressed`）。

---

## 4. 核心信号指标（merged）

- support_state：exposed 31、correct_tail 11、no_correct_observed 6、other 16；greedy 正确率 0.422。
- 总体 within-prompt：teacher_gap AUC 0.564（CI 0.479–0.650，n=37 合格 prompt）；
  teacher_mean_logp AUC 0.552；student_mean_logp AUC 0.551。
- rank@1 lift −0.016（CI −0.124–0.097）；MRR 0.602 vs 随机 0.625（lift −0.023）。
- correct_tail（n=11）：within-prompt AUC 0.548（CI 0.391–0.713）；rank@1 lift −0.009
  （CI −0.182–0.201）；MRR 0.372 vs 随机 0.378。
- 长度分层（teacher_gap within-prompt AUC）：

| bin | n | AUC（CI95） | rank@1 lift（CI95） |
|---|---|---|---|
| ≤512 | 7 | **0.792（0.625–0.932）** | **+0.295（0.104–0.495）** |
| 513–2048 | 18 | 0.547（0.424–0.667） | ≈0.000（−0.147–0.150） |
| >2048 | 23 | 0.492（0.387–0.601） | −0.111（−0.244–0.028） |

- finish-reason 分层：stop n=34 AUC 0.575（CI 0.483–0.669）；length n=3 AUC 0.454。
- 响应长度：p10 286 / p50 1219 / p90 4096 / mean 1875.6；finish stop 1613、length 499（23.6%）。
- 协议质量：exact identity 1.0、prompt hash 1.0、coverage 1.0、missing 0、non-finite 0、
  duplicate 0.0、malformed 20.2%（426/2112）。

---

## 5. K=8 → K=32 校准（frontier 口径一致：Jeffreys Beta(0.5,0.5)、group-size 8、MC 20000、seed 42、useful≥0.5、RL-ready P≥0.8）

### 5.1 stratum 重分类矩阵（行=K8，列=K32）

| K8 \ K32 | no_correct | rare_success | mixed_support | all_correct | total |
|---|---:|---:|---:|---:|---:|
| no_correct_observed | 10 | 5 | 1 | 0 | 16 |
| rare_success | 1 | 11 | 4 | 0 | 16 |
| mixed_support | 0 | 1 | 15 | 0 | 16 |
| all_correct_observed | 0 | 0 | 6 | 10 | 16 |

对角线一致 46/64 = **71.9%**。K=8 的 0/8 与 8/8 边界大量被 K=32 修正：
0/8 中 6/16 上移到 rare/mixed；8/8 中 6/16 掉到 mixed。

### 5.2 E[U8] 与 RL-ready（同一 64 个 prompt 对比）

| 指标 | K=8（筛选时） | K=32（确认后） |
|---|---:|---:|
| Σ E[U8]（posterior frontier mass） | 34.21 | 31.33 |
| mean E[U8] | 0.535 | 0.490 |
| plugin mean U8 | — | 0.449 |
| RL-ready（P(U8≥0.5)≥0.8） | 18 | **27** |

- RL-ready 转移：stay 16、drop 2、gain 11、never 35。
- E[U8](K8) 与 E[U8](K32) 相关性 0.80。
- K=32 口径下 RL-ready 按 stratum：mixed_support 20/26、rare_success 7/17、
  no_correct_observed 0/11、all_correct_observed 0/10（全正确组 U8→0，符合“混合组才有梯度”定义）。
- 最高候选（K32 E[U8]≥0.91）：geo3k:137（16/32,0.984）、1208（16/32,0.984）、
  1569（17/32,0.983）、191（13/32,0.974）、1113（19/32,0.974）、299（19/32,0.974）、
  1087（12/32,0.965）、2033（11/32,0.952）、2096（21/32,0.952）、1907（11/32,0.952）、
  1025（10/32,0.934）、1260（23/32,0.911）。

---

## 6. 解读与注意点

1. **数据管线干净**：exact identity / hash / mask / coverage 全部 1.0，无缺失图、无非有限分数；
   smoke 与正式四片全链路可复现（逐字节一致）。
2. **主 gate 不达标，但正信号可报告**：整体 AUC 0.564 < 0.60；≤512 token 分层 AUC 0.792
   （CI 下限 0.625 达标，rank@1 lift 显著为正），与 K=8 轮长度分层结论方向一致。
3. **malformed 20.2% / truncation 23.6% 是测量收紧的代价**：保守 verifier（要求显式
   Answer/boxed）与 4096 上限共同导致；与 K=8 轮同源，不是回归。truncation 略好于 K=8（0.292→0.236）。
4. **K=32 重采样价值明确**：71.9% 对角线一致、E[U8] 相关 0.80，K=8 边界标签（0/8、8/8）
   大量被修正；RL-ready 净增 9 个（27 个）。这正好完成 handoff 里“K=8 粒度不足需 K=32 确认”的闭环。
5. **“RL-ready”定义提醒**：all_correct_observed 组 U8=0 是定义使然（全对组没有正负混合），
   不代表该 prompt 无价值，只代表它不是“混合组”型 RL 候选。
6. **git 分歧已审计**：四片代码等价（diff 只涉及 qwen35 训练路径），合并产物中留有审计文件，
   不掩盖来源 commit。

---

## 7. 下一步建议（供决策）

- **今晚停在 K=32 confirmation，不启动五臂 bridge**（保持本轮边界）。
- bridge 候选直接采用 K=32 口径的 RL-ready 27 个（`frontier_analysis/frontier_prompts.csv`，
  `rl_ready=true`），按 E[U8] 降序分配预算；drop 的 2 个与 gain 的 11 个是 K=8 决策需要修正的对象。
- 报告主图建议：图 1 = stratum 重分类矩阵/RL-ready 转移；图 2 = within-prompt AUC 按长度 bin
  （≤512 是正结果）；图 3 = E[U8] K8 vs K32 散点（相关 0.80，含对角线）。
- 若后续要出五臂协议，先预注册 `configs/experiment/frontier_operator_bridge_pilot.yaml` 的
  候选名单（基于 K=32 frontier_prompts），不要用 K=8 边界标签直接分配。

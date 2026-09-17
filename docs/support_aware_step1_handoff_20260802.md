# Support-Aware SFT-RL-OPD — Step 1 交接文档（2026-08-02）

**作者**: codex（GPU 侧）
**交接给**: 本地电脑的 codex 继续调研/细化方案
**分支**: `codex/va-opd`（已 push，见文末 commit 列表）
**当前日期**: 2026-08-02（北京时间 ~21:30）
**硬性时间线**: 后天（2026-08-03）要汇报，Step 1 必须在明天给出结论

---

## 1. TL;DR

Support-Aware OPD 的 Step 1（frozen-policy diagnostic）已经修复了两个致命 bug，
当前正用 **4 条并行流水线在 2 台 GPU 实例上跑 256 个 prompt**（4096 token、
每 prompt 1 greedy + 8 stochastic）。所有增量数据安全落盘，实例回收可随时 resume。

**请本地 codex 重点看：**
1. §3 的 bug 历史（尤其 `8a5d3cc` teacher prompt 丢失——这是 0.665→0.309 AUC 反转的根因）
2. §4 的代码架构 + §5 的分片/合并/resume 操作手册
3. §6 的当前运行清单 + §7 的明早合并分析要点
4. §8 的 Step 2 go/no-go 判据

---

## 2. Step 1 背景与指标定义

Step 1 目标：冻结 4B student（Qwen3-VL-4B-Instruct）和 32B teacher
（Qwen3-VL-32B-Instruct），验证 **teacher_gap 能否在学生自己没把握时
（correct_tail）把正确 rollout 排到前面**——这是 Step 2 support-gated
RKL/RL 微调的信号基础。

### 数据流

```
Geometry3K parquet (2101 valid rows)
  → SHA256(sample_uid) 排序，取前 N 个（确定性选择）
  → 每个 prompt: 生成 1 greedy (T=0) + 8 stochastic (T=0.7, top_p=0.95)
  → teacher 32B forced-logp 打分（HTTP batch，精确 prompt）
  → student 4B forced-logp 打分（本地 batch forward）
  → teacher_gap = teacher_mean_logp − student_mean_logp
  → support state 分类 → 汇总指标 → 7 个 acceptance gates
```

### 核心量

| 量 | 定义 | 含义 |
|---|---|---|
| `student_mean_logp` | 学生逐 token forced logp 均值 | 学生自己认为这段话多"自然" |
| `teacher_mean_logp` | 老师对同一段话的 forced logp 均值 | 老师认为这段话多"自然" |
| `teacher_gap` | teacher − student | 正 = 老师比学生更确信（对应正确答案的候选） |

### Support state（每 prompt，按 greedy + 8 条随机的结果分类）

| 状态 | 条件 | 含义 |
|---|---|---|
| `exposed` | greedy 对，或 correct ≥ K/2 | 正确答案模式已被充分采样 |
| `correct_tail` | greedy 错，且 0 < correct < K/2 | **正确答案只在尾部出现——OPD 目标区** |
| `no_correct_observed` | greedy 错，且 correct = 0 | 采样预算内没见正确答案 |
| `other` | greedy 不可判定等边界 | — |

### 关键指标

| 指标 | 算法 | 含义 |
|---|---|---|
| `greedy_accuracy` | greedy 答对比例 | 学生确定性能力 |
| `auc_teacher_gap_correct_vs_wrong` 等 3 个 AUC | 全部随机 rollout 的分数 vs correct 标签，`sklearn.roc_auc_score` | 分数能否区分对错；0.5=随机 |
| `correct_tail_rank_metrics.rank1` | **仅 correct_tail 的 prompt**，按 teacher_gap 排序后 top1 正确的比例 | 随机基线 1/K=0.125 |
| `correct_tail_rank_metrics.mrr` | 1/rank 平均 | 1.0 = 正确永远排第一 |
| `response_length_percentiles` | 响应 token 数分位数 | 判断截断（`==max_new_tokens` 比例） |

### 7 个 acceptance gates

| Gate | 阈值 | 说明 |
|---|---|---|
| `tokenizer_alignment` | hash 一致 | forced scoring 要求 token 对齐 |
| `zero_missing_images` | = 0 | 图片缺失直接污染分数 |
| `finite_scores` | 0 NaN/Inf | 分数必须可算 |
| `malformed_rate_ok` | ≤ 10% | 回答解析率 |
| `duplicate_rate_ok` | ≤ 25% | 采样多样性 |
| `sufficient_correct_tails` | ≥ 20（full mode） | correct_tail 样本量 |
| `teacher_gap_auc` | ≥ 0.60（full mode） | teacher_gap 区分度 |

**512-token 基线 run（`diag_full_20260801_013648`，旧代码，仅供参考）**：
greedy acc 22.7%；states 35 exposed / 22 correct_tail / 69 no_correct / 2 other；
AUC teacher_gap 0.665 / teacher_mean 0.763 / student_mean 0.823；
correct_tail Rank@1 0.409、MRR 0.592。**注意这些数字被 71.4% 截断污染，不可直接用于判断**。

---

## 3. Bug 历史与修复（按 commit，从新到旧）

所有代码在 `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD`。

### 5d10042 — fix: add `--run-id` override（2026-08-02）
**现象**：两条流水线同一秒启动，`_make_run_id()` 用秒级时间戳
（`diag_%Y%m%d_%H%M%S`）生成相同 run id，两个进程写同一个输出目录。
**修复**：`DiagnosticConfig.run_id_override` + CLI `--run-id` + run 脚本透传；
`--resume` 优先级高于 `--run-id`。
**文件**：`src/dual_track_opd/support_aware/diagnostic.py`、
`scripts/hpc/run_support_aware_diagnostic.sh`。

### 3f4c7bb — chore: pass `--num-prompts` through run script
**文件**：`scripts/hpc/run_support_aware_diagnostic.sh`（256 样本需要）。

### 27a83fd — feat: sharded parallel diagnostic runs + merge tool
**动机**：单进程串行 128 prompts × ~13 min ≈ 28h，实例生命周期只有 ~11h；
多 GPU 实例无法直接加速单进程。方案 = 确定性切片 + 多实例并行 + 合并。
**实现**：
- `slice_prompts(prompts, start, end)`：在 SHA256 排序后切片，写回 manifest
  （`prompt_start` / `prompt_end` / `selection_rule` 标注 shard）。
- CLI `--prompt-start N --prompt-end M`（run 脚本已透传）。
- 把 run_diagnostic 里内联的汇总计算抽成 `_build_summary(...)`，单 run 与
  合并路径共用同一套指标/gates 代码。
- `merge_shard_runs(shard_dirs, output_dir, mode)`：加载各分片
  `rollouts.jsonl` + `prompt_support_summary.jsonl`，按
  `(sample_uid, is_greedy, rollout_id)` 去重，重算完整 summary + gates，
  写出合并后的三个文件。CLI：`--merge-shards DIR... --merge-output DIR
  --merge-mode full`（不需要 `--config`）。
**文件**：`src/dual_track_opd/support_aware/diagnostic.py`、
`tests/test_support_diagnostic_shards.py`、run 脚本、状态文档。

### 8a5d3cc — fix: teacher batch scoring must pass exact student prompt（关键修复）
**现象**：2048-token full run（`diag_full_20260801_163119`）gate 失败，
`teacher_gap AUC = 0.309`（阈值 0.60），比随机还差；而 512-token 旧 run 是 0.665。
**根因**：`TeacherScorer.score_batch`（`src/dual_track_opd/support_aware/scorer.py`）
构造 3 元组 sample `(token_ids, question, condition_inputs)` → 发到服务端时
`prompt=None`。teacher 服务端 `_prepare_prompt()`
（`src/dual_track_opd/fc_opd/teacher_transformers.py`）在 `prompt=None` 时回退到
`render_teacher_prompt()`（`src/dual_track_opd/fc_opd/teacher_prompts.py`），
生成的文本是 `"Question:\n{question}"`——**而学生实际看到的是带
"Think step by step..." 指令的 `_PROMPT_TEMPLATE`**
（`build_prompt()`，`diagnostic.py` 第 65 行附近）。两边条件历史不同 → 分数失真。
注意 `score_batch` 签名里的 `prompt_texts` 参数**存在但从未被使用**——这就是回归点。
**证据链**（全部来自磁盘数据交叉验证）：
- 331 条两 run 完全相同的 rollout：teacher 分数平均差 **0.18**（corr 0.86），
  student 只差 0.0012（corr 0.999）→ student 没问题，teacher 链路变了。
- batch 模式下 teacher_mean 与长度几乎无关（corr 0.01），而 student 强负相关
  （-0.77）→ teacher_gap 被学生长度惩罚主导，长而错的回答 gap 最大 → AUC < 0.5。
- `diag_full_20260801_123045`（同为 batch 代码、另一实例）与 163119 分数
  **逐位一致**（corr 1.0）→ batch 路径确定，纯粹是 prompt 用错。
- 修复后 smoke（`diag_smoke_20260802_073956`）与 per-item 精确 prompt 路径
  （013648）的 teacher 分数 **diff = 0.000000、corr = 1.000** → 修复完全复刻
  per-item 行为。
**修复**：`score_batch` 改为 4 元组，显式携带学生 prompt：
`(token_ids, question, condition_inputs, ({"role":"user","content":[{"type":"image","image":image_paths[i]},{"type":"text","text":prompt_texts[i]}]},))`。
**测试**：`tests/test_support_scorer_batch.py` 新增
`test_teacher_score_batch_preserves_exact_student_prompt`（mock 网络层）。
**附带变更**：`configs/experiment/support_aware_geometry3k_pilot.yaml`
`max_new_tokens: 2048 → 4096`（2048 下截断仍 44%，runbook 阈值 15%）。

### 5fd8f1a — fix: run script resolves a torch-enabled python
**现象**：shell 未激活 conda 环境时，`python` 解析到 `/usr/bin/python`，
`ModuleNotFoundError: No module named 'dual_track_opd'`。
**修复**：`resolve_python()` 依次尝试 `$DTOPD_PYTHON` → PATH 里带 torch 的
`python` → `vision-opd-cu128` 环境绝对路径。
**文件**：`scripts/hpc/run_support_aware_diagnostic.sh`。
**注意**：`scripts/hpc/start_fc_teacher.sh` **仍然用裸 `python`**，需要
`PATH=/inspire/.../envs/vision-opd-cu128/bin:$PATH` 前缀或先 activate。

### 6ea8da5 — feat: resume partial diagnostic runs via `--resume RUN_ID`
增量保存：每条 rollout 打分后立即 append 到 `rollouts.jsonl`；每 prompt 完成后
append `prompt_support_summary.jsonl` 并重写 `_partial_summary.json`。
`_load_resume_state()`（`diagnostic.py`）只保留 **有完整 prompt summary 的 uid**
对应的 rollouts（半途被杀 prompt 的残片被丢弃、整体重做）；完成时
`write_rollouts_jsonl()` 以 `"w"` 模式重写干净文件。
**警告**：resume 前必须确认旧进程已死，否则双写同一目录。

### c32a1e9 — fix: batch student scoring passes per-row multimodal tensors
**现象**：多行 `input_ids` 打包但复用单张图的 `pixel_values/image_grid_thw/
mm_token_type_ids` → Qwen3-VL placeholder check 崩溃：
`ValueError: Image features and image tokens do not match, tokens: 630, features: 70`。
**修复**：一次 `processor(text=full_texts, images=[image]*n, padding=True)`，
每行自带图像张量 + padded `mm_token_type_ids`。
**文件**：`src/dual_track_opd/support_aware/scorer.py`、
`tests/test_support_scorer_batch.py`。

### fe2af95 — feat: batch teacher+student scoring, incremental save, 2048 tokens
Claude 的初始提交（batch 打分 + 增量保存 + 2048），是上述 bug 的载体。

---

## 4. 代码架构与文件路径

### 关键文件

| 文件 | 作用 |
|---|---|
| `src/dual_track_opd/support_aware/diagnostic.py` | 主流程：选择→生成→打分→summary/gates；resume；分片；merge |
| `src/dual_track_opd/support_aware/scorer.py` | `TeacherScorer`（HTTP）+ `StudentScorer`（本地），含 batch 版 |
| `src/dual_track_opd/support_aware/support_state.py` | `SupportState` 枚举 + `classify_support_state()` |
| `src/dual_track_opd/support_aware/verifier.py` | Geometry3K 答案抽取 + 校验 |
| `src/dual_track_opd/support_aware/reporter.py` | JSONL/summary/manifest 写盘 |
| `src/dual_track_opd/fc_opd/teacher_client.py` | `score_teacher_conditions_multi_sample`（batch 请求构造） |
| `src/dual_track_opd/fc_opd/teacher_transformers.py` | teacher 服务端 `_prepare_prompt()`（prompt=None 回退逻辑在这里） |
| `src/dual_track_opd/fc_opd/teacher_service.py` | 独立 HTTP 服务（单线程 `HTTPServer`，一次处理一个请求） |
| `src/dual_track_opd/fc_opd/teacher_prompts.py` | `render_teacher_prompt()`（"Question:\n" 渲染，非学生 prompt） |
| `configs/experiment/support_aware_geometry3k_pilot.yaml` | 实验配置（`max_new_tokens: 4096`、`num_prompts: 128` 等） |
| `scripts/hpc/run_support_aware_diagnostic.sh` | HPC 启动器（python 解析、GPU 检查、参数透传） |
| `scripts/hpc/start_fc_teacher.sh` | teacher 启动器（裸 python，需 PATH 前缀） |
| `docs/support_aware_opd_diagnostic_status.md` | 状态文档（含 2026-08-02 调查结论） |
| `docs/support_aware_sft_rl_opd_experiment_runbook.md` | **Step 2 runbook**（50-step micro-training pilot 定义） |

### 测试

```bash
# 在 vision-opd-cu128 环境（本地无 GPU 也能跑，除 model 相关用例会 skip）
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python -m pytest \
  tests/test_support_diagnostic.py \
  tests/test_support_diagnostic_resume.py \
  tests/test_support_scorer_batch.py \
  tests/test_support_diagnostic_shards.py -q
# 51 passed（含新增的 teacher-prompt 回归 + shard/merge 测试）
```

### 关键函数位置（diagnostic.py 行号随版本漂移，用 rg 定位）

- `build_prompt(question)`：学生真实 prompt 模板（含 step-by-step 指令）
- `load_and_select_prompts()`：SHA256 排序选择
- `slice_prompts()`：分片切片
- `_load_resume_state()` / `_ResumeState`：resume 语义
- `_build_summary()`：汇总 + gates（单 run 与 merge 共用）
- `merge_shard_runs()`：分片合并
- `_compute_auc()`：ROC AUC（sklearn 优先，手算兜底）
- `_compute_ranking_metrics()` / `_compute_correct_tail_ranks()`：rank/MRR
- `_check_gates()`：7 gates

### 环境变量

```bash
export DTOPD_MODEL_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models
export DTOPD_OUTPUT_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs
# 运行输出根：$DTOPD_OUTPUT_ROOT/support_aware_opd/
# conda env：vision-opd-cu128（Python 3.12, torch 2.10.0+cu128）
```

---

## 5. 分片 / 合并 / resume 操作手册

### teacher 启动（每台实例都要起，GPU 独占）

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
PATH=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin:$PATH \
FC_OPD_TEACHER_MODEL=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct \
CUDA_VISIBLE_DEVICES=7 nohup bash scripts/hpc/start_fc_teacher.sh \
    > $DTOPD_OUTPUT_ROOT/support_aware_opd/teacher.log 2>&1 &
# 第二个 teacher（并行流水线用，端口 18081）：
FC_OPD_TEACHER_MODEL=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct \
FC_OPD_TEACHER_PORT=18081 \
CUDA_VISIBLE_DEVICES=6 nohup bash scripts/hpc/start_fc_teacher.sh \
    > $DTOPD_OUTPUT_ROOT/support_aware_opd/teacher2.log 2>&1 &
until curl -s --max-time 5 http://127.0.0.1:18080/health | grep -q '"ok"'; do sleep 10; done
until curl -s --max-time 5 http://127.0.0.1:18081/health | grep -q '"ok"'; do sleep 10; done
```

### 分片启动模板

```bash
DTOPD_MODEL_ROOT=... DTOPD_OUTPUT_ROOT=... CUDA_VISIBLE_DEVICES=4,5,6 nohup bash scripts/hpc/run_support_aware_diagnostic.sh \
    --config configs/experiment/support_aware_geometry3k_pilot.yaml \
    --mode full \
    --num-prompts 256 \          # 256 样本才需要；128 用默认即可
    --run-id diag_full_20260802_xxx \   # 并行启动必须显式指定，防时间戳碰撞
    --teacher-url http://127.0.0.1:18080 \
    --prompt-start 0 --prompt-end 32 \
    > $DTOPD_OUTPUT_ROOT/support_aware_opd/shard_0_32.log 2>&1 &
```

注意：**同一实例上两条流水线必须各配一个 teacher**（teacher 是单线程 HTTP，
共享会排队）。标准布局：teacher1 GPU 7（18080）+ teacher2 GPU 6（18081），
两条 student 分别用 4,5 和 0,1,2,3。

### resume

```bash
... --mode full --resume diag_full_20260802_083157 --prompt-start 0 --prompt-end 64 ...
# resume 时不要加 --run-id（--resume 优先）；确认旧进程已死
```

### 合并（所有分片完成后）

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python \
    -m dual_track_opd.support_aware.diagnostic \
    --merge-shards \
      $DTOPD_OUTPUT_ROOT/support_aware_opd/diag_full_20260802_083157 \
      $DTOPD_OUTPUT_ROOT/support_aware_opd/diag_full_20260802_113811 \
      $DTOPD_OUTPUT_ROOT/support_aware_opd/diag_full_20260802_n128_192 \
      $DTOPD_OUTPUT_ROOT/support_aware_opd/diag_full_20260802_n192_256 \
    --merge-output $DTOPD_OUTPUT_ROOT/support_aware_opd/merged_full_20260802 \
    --merge-mode full
```

合并输出：`rollouts.jsonl`、`prompt_support_summary.jsonl`、`summary.json`
（含 7 gates）。分片只取其已完成 prompt（增量语义），半截的分片自动少算，
不会报错。

### 常见坑

- **PID 是机器本地的**：不要在实例 A 上 kill 实例 B 的 PID；`pkill` 匹配进程名
  而非 PID。先 `ps aux | grep "dual_track_opd.support_aware.diagnostic"` 确认。
- **同秒启动撞 run id**：并行启动必须用 `--run-id`。
- **resume 前确认旧进程死透**，否则双写。
- 跑之前确认代码在 `5d10042`（`git log --oneline -1`），共享存储上改动即时生效。

---

## 6. 当前运行清单（2026-08-02 21:30 北京 / 13:30 UTC 快照）

**实例 1**（旧实例，~20:40 UTC 回收；teacher GPU7:18080）

| 流水线 | run id | 区间 | student GPU | 日志 | 进度(13:30) |
|---|---|---|---|---|---|
| 分片 A | `diag_full_20260802_083157` | 0-64 | 4,5,6 | `diag_shard_0_64.log` | 25/64, 225 rollouts |
| 分片 B | `diag_full_20260802_113811` | 64-128 | 0,1,2,3 | `diag_shard_64_128.log` | 10/64, 90 rollouts |

**实例 2**（新实例，~23:15 UTC 回收；teacher1 GPU7:18080 + teacher2 GPU6:18081）

| 流水线 | run id | 区间 | student GPU | 日志 | 进度(13:30) |
|---|---|---|---|---|---|
| 新 1 | `diag_full_20260802_n128_192` | 128-192 | 4,5 | `diag_new_128_192.log` | 3/64, 27 rollouts |
| 新 2 | `diag_full_20260802_n192_256` | 192-256 | 0,1,2,3 | `diag_new_192_256.log` | 2/64, 18 rollouts |

全部 `errors=0 missing=0 malformed=0 nonfinite=0`。

**运行输出根**：`/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/support_aware_opd/`

**预期**：实例 1 回收时 A ≈ 50/64、B ≈ 37/64；实例 2 回收时各 ≈ 47-48/64。
凌晨总计完成 ~180-190/256。早上开新实例按 §5 resume 补尾（共 ~70 prompt，
可继续 2 teacher × 2 流水线并行，~7-8 小时）。

### 监控

```bash
cat $DTOPD_OUTPUT_ROOT/support_aware_opd/diag_full_20260802_*/_partial_summary.json
# 日志每 10 prompt 才打印一次；_partial_summary.json 每个 prompt 都更新
```

---

## 7. 明早合并后的分析要点

1. **7 gates**：重点看 `sufficient_correct_tails`（预计 256 样本下 45+，稳过）
   和 `teacher_gap_auc`（阈值 0.60）。
2. **截断率**：4096 下实测 ~25%（256 样本会稳定在 20-25%）。同时报
   **全量** 和 **仅非截断** 两套 AUC/Rank 指标，量化截断影响。
3. **信号准确性**：pooled `teacher_gap` AUC；correct_tail 内 Rank@1（>0.125）
   与 MRR；以及 teacher_mean vs student_mean AUC 对比。
4. **对照**：512-token 旧 run 数字见 §2 表格；修复后的早期信号
   （smoke/前 16 prompt）teacher_gap top1 ≈ 0.625，样本小仅参考。
5. teacher 日志里每条的 `tokenizer round-trip mismatch at last token 151645`
   是**良性警告**（代码保留原始 token id），所有 run 都有。

---

## 8. Step 2 go/no-go 判据与下一步

**go 的条件**（全部满足才进 Step 2）：
1. 7 gates 全绿（signal gates: correct_tails ≥ 20、teacher_gap AUC ≥ 0.60）
2. 截断率相比 2048（44%）显著下降；报告里给全量 + 非截断两组数字
3. correct_tail 内 teacher_gap Rank@1 明显高于 0.125，MRR ≥ 0.5
4. 注意预期：student 自己 logp 的 AUC 通常最高（自洽性），**OPD 核心验证点是
   correct_tail 内 teacher_gap 的 Rank@1/MRR**，别被 student AUC 带偏。

**下一步**：
- 合并 → 分析（§7）→ go/no-go 结论
- 若 go：按 `docs/support_aware_sft_rl_opd_experiment_runbook.md` 准备
  50-step micro-training pilot（RL-only vs uniform RKL vs support-gated RKL），
  Step 2 配置/数据提取脚本可先行准备（CPU 活）。
- 用户提到有一份与 ChatGPT 的对话包含 SFT-RL-OPD 相关思考，尚未提供；
  提供后只看 SFT-RL-OPD 部分，忽略 VA-OPD。

---

## 9. Commit 清单（codex/va-opd，全部已 push）

```
5d10042 fix: add --run-id override to avoid parallel-pipeline run dir collisions
3f4c7bb chore: pass --num-prompts through run_support_aware_diagnostic.sh
27a83fd feat: sharded parallel diagnostic runs + merge tool
8a5d3cc fix: teacher batch scoring must pass exact student prompt; bump diagnostic to 4096 tokens
5fd8f1a fix: run_support_aware_diagnostic.sh resolves a torch-enabled python
6ea8da5 feat: resume partial diagnostic runs via --resume RUN_ID
c32a1e9 fix: batch student scoring passes per-row multimodal tensors
fe2af95 feat: batch teacher+student scoring, incremental save, 2048 tokens
```

**给本地 codex 的待办**：
- 细化 Step 2 实验设计（对照 runbook 文档）
- 准备合并后分析脚本（pooled + 非截断两组指标）
- 如果用户贴出 ChatGPT 对话，提炼 SFT-RL-OPD 思路并和本方案对齐
- 复核 §8 的 go/no-go 判据是否完备

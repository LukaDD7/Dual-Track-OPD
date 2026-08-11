# Causal State Probe — 2026-08-08 Run Quality Report

> **2026-08-11 review:** The pre-check coverage and malformed semantics require
> correction before state-class or strongest-candidate claims are treated as
> final.  See `docs/causal_state_probe_precheck_review_20260811.md` for the
> prioritized findings and acceptance criteria.

## 1. Provenance

- Repo: `Dual-Track-OPD` @ `ab7d054` + local fixes（未提交）：
  - `causal_runtime.py`: teacher-path cross-model JS 设备对齐（cuda:1 → cuda:0）
  - 运行环境必须为 `va-opd-qwen35-cu128`（transformers 5.12.0），见 tracker #20
- 模型：student `Qwen3-VL-4B-Instruct`（cuda:0），teacher `Qwen3-VL-32B-Instruct`（cuda:1）
- 输入：`diag_full_k32_20260804_merged` + `k32_cohort_20260804` + `proposal_feasibility_20260805_merged`
- 证据 tokenizer hash：`b5005fea...`（与产物一致，preflight/运行时均通过）
- 已执行：
  - smoke（`causal_state_probe_20260808_smoke`）：2/2 units，`complete=true`
  - 串行正式版 unit 1（rollout-3, correct, 515 tokens, K=8/2048）：`causal_state_probe_20260808`
  - 并行分片已启动：`causal_state_probe_20260808_s{0..3}`（4×2-GPU pair，101 units 切分为 27/28/23/23）

## 2. 质量结论

管线端到端可用：tokenizer 契约、JS 计算、候选窗口、relay/transport/leakage 探针、verifier
verdict、报告与 resume 均按设计工作，无 NaN/Inf。

### 信号行为符合预期

- 视觉依赖：`js_full_null` 明显高于 `js_full_degraded`（unit1 median 6.5e-05 vs 7.5e-07；
  p95 0.159 vs 0.026；max 0.689 vs 0.246），方向正确；JS 峰值集中（>0.05 的 token 占 12%，
  >0.2 占 4%），符合"少数位置强视觉依赖"假设。
- 候选来源：smoke 9 个候选里 adaptive（high_visual_dependence 2 / visual_dependence_drop 3 /
  post_visual_low 2）与 fixed control（6）并存，NMS/吸附生效。
- relay：unit1 anchor=257（rel 0.50）三个长度 gain 均正且稳定
  （L32 +0.224 p=0.938 / L64 +0.113 p=0.734 / L128 +0.222 p=0.937），无 leakage 标记 →
  正确落入 `on_policy_repairable`，是 Phase I 想找的 Case 1。
- answer leakage：所有候选 pass_rate=0（L 探测中 relay 标注 malformed/leakage 的样本被保守排除）。
- teacher-path NLL barrier：3 个（rel 0.22/0.36/0.91），peak NLL 3.3-5.5；S/T JS 中位极低
  （7e-05）、overlap 中位 0.69，说明 student 在 teacher 路径上大部分位置分布接近，barrier 位置
  与整体低 NLL 形成对比，具备后续 bridge 上游区间候选意义。

## 3. 需要 codex 关注的问题

1. **continuation malformed 率偏高**：anchor=0 附近 student 续写 malformed 达 7-8/8
   （legacy_answer 契约 + 前缀已含答案时格式漂移）。按设计保守计入非成功，但会系统性压低
   anchor=0 附近的 relay/transport gain，建议后续对 start-of-response 候选的续写格式做
   normalization 或缩短 cap 再评估。
2. **relay gain 跨 L 不一致**：anchor=103 L32/L128 正（p≈0.71）而 L64 负（p=0.06），
   分类器正确判为 `insufficient_evidence`；建议 summary 增加跨 L 一致性指标，避免单 L 误判。
3. **潜在的 Case 2 苗头**：anchor=411 relay L128 显著为负（-0.556, p=0.005）但 transport
   +0.111（p=0.83）、vs wrong +0.111（p=0.83），低于 `min_mean_gain=0.20` 阈值未达标；
   全量 101 units 跑完后值得专门检查 transport>relay 的窗口是否出现。
4. **barrier 记录缺 `peaks` 字段**：`reachability_barriers` 只有 anchor/start/end/mean/peak_nll，
   没有 per-peak token 列表（文档 §5.5 提到 barrier 是后续 FKL 上游区间，建议明确是否要输出
   peak 明细）。
5. **resume 的 `started_at_unix` 不刷新**：manifest 复用时保留首次启动时间（provenance 小瑕疵，
   不影响正确性）。

## 4. 运行节奏与续跑

- 串行实测 ~3h/unit，101 units 需 10-12 天 → 已改为 4 对 GPU 并行（~3 天量级）。
- 实例 18h 自动回收：按分片目录 + 原子 JSON 续跑即可；重启后重跑同一条分片命令
  （保持 `SHARD_INDEX/NUM_SHARDS/CAUSAL_PROBE_OUTPUT_DIR` 不变），已完成轨迹自动跳过。
- 合并：`causal_state_probe summarize --input-dir s0..s3 --output-dir merged`。

## 5. 2026-08-09 优化：固定轨迹统计单次整段前向（tracker #22）

- 改动：`causal_runtime.py` 两个统计函数不再逐 64-token chunk 重放整个前缀，
  改为每模型/每图像条件一次整段前向（`logits_to_keep=T+1`），分块只做词表数学。
  因果注意力保证逐位置数学等价；峰值内存不变。
- 备份：`artifacts/fc_opd/backup_20260809_causal_probe_opt/`（含 `causal_runtime.py.orig`
  等三份原始文件 + git 状态）。
- 验证：
  - 新增 `tests/support_aware/test_causal_runtime_logits_equivalence.py`（toy causal LM，
    `logits_to_keep` 路径与 fallback 路径均逐位相等）；
  - 真实 Qwen3-VL-4B CPU 对拍（trajectory `geo3k:657:rollout-10`，110 tokens，含真实图片）：
    同长度前向逐位 bit-identical（fallback 下 chunk2 max diff=0）；其余位置 max diff 0.53 logit
    为 bf16 不同序列长度下的形状噪声；preflight 通过。
- 影响：长轨迹 unit 预计 3–5h → 0.5–1h；已完成的 26 个 unit 数值保持旧实现，
  后续 unit 用新实现（逐 unit 独立，schema 与合并流程不变）。
- 实测（2026-08-09 05:19–06:33 UTC）：新代码在 GPU 实例上端到端产出 2 个合法 unit
  （s0：407/483 token，schema 正确、无 NaN，分别耗时 ~44/30 min）；实例于 06:33 被
  回收，s1/s2/s3 首条长 unit（4096/3566 token）在途丢失，未及实测长 unit 提速。
  回收后累计 30/101 完成（28 旧 + 2 新），续跑命令不变。

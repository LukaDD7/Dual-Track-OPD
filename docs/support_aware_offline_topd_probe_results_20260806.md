# Offline TOPD Probe(实验 C)结果 — 2026-08-06

## 0. 一句话结论

**NO-GO:离线 OT 定位救不回 Gate C;且该离线代理没有通过自身 sanity(G2),
不能据此否定 TOPD 原方法的定位能力。** 不启动 minimal-bridge 训练臂,
保持 rare_success verified-FKL 方向。

- G1 数据可行性:通过(12/12 题,answer-free 前缀检查无失败)。
- G2 pooled loss-OT sanity:**失败** —— pooled Spearman(gap, OT) = −0.037,
  prompt-bootstrap CI [−0.084, +0.002]。论文的同款探测是弱正相关
  (Pearson r≈0.126 / Spearman ρ≈0.143),离线近似没有复现这个前提。
- G3 OT 定位预测 rescue:**失败** —— primary `div_window_512_1024_gap` 与
  `best_lift` 的 Spearman = −0.25(CI 含 0,n=8);horizon 匹配 2/7 精确、
  3/7 在 ±1 档内。
- 意外信号:`ot_window_512_1024_gap` 与 `best_lift` Spearman = **−0.71**
  (CI [−0.97, −0.04],n=8),方向与定位假设相反;按预注册原则只记录不追。

## 1. 运行信息

| 项 | 值 |
|---|---|
| 代码 | `src/dual_track_opd/support_aware/offline_topd_probe.py`(schema v1) |
| 输入 | `proposal_feasibility_20260805_merged` + `diag_full_k32_20260804_merged` + `prefix_intervention_20260806_merged` |
| 探测配置 | K=50、stride 1、warped 对齐、high-loss/high-OT 各 top 20% |
| 输出目录 | `$OUT/support_aware_opd/offline_topd_probe_20260806/` |
| 完成度 | complete=true,12/12 prompts,missing_uids=[] |
| 决策 | NO-GO(见 `summary.json`) |

## 2. 每题明细(probe_rows.jsonl)

| uid | stratum | t_len | s_len | pred_h | obs_h | best_lift | div_cnt | ot_512_1024 | div_512_1024 | ot_p256 | sp(gap,ot) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geo3k:1124 | rare | 4096 | 4096 | 512 | – | 0.001 | 156 | 0.569 | 0.002 | 0.558 | +0.003 |
| geo3k:1141 | rare | 531 | 882 | 256 | 256 | 0.889 | 51 | 0.409 | 0.000 | 0.481 | −0.057 |
| geo3k:1242 | no_correct | 552 | 406 | 512 | – | 0.001 | 10 | – | – | 0.292 | +0.092 |
| geo3k:17 | rare | 1084 | 756 | 512 | 64 | 0.890 | 31 | 0.517 | 0.067 | 0.467 | +0.080 |
| geo3k:1713 | rare | 4096 | 3895 | 512 | – | 0.111 | 127 | 0.511 | 0.047 | 0.449 | −0.016 |
| geo3k:2061 | rare | 2011 | 314 | 128 | 128 | 0.666 | 8 | – | – | 0.561 | −0.094 |
| geo3k:264 | rare | 451 | 198 | 128 | 256 | 0.666 | 5 | – | – | 0.533 | +0.100 |
| geo3k:374 | rare | 399 | 975 | 512 | 128 | 0.889 | 43 | 0.462 | 0.000 | 0.547 | −0.158 |
| geo3k:388 | rare | 3610 | 3141 | 512 | – | 0.001 | 128 | 0.554 | 0.096 | 0.441 | −0.020 |
| geo3k:428 | rare | 1468 | 4096 | 512 | – | 0.111 | 169 | 0.558 | 0.029 | 0.497 | +0.006 |
| geo3k:845 | rare | 2098 | 1023 | 512 | 64 | 0.446 | 39 | 0.543 | 0.063 | 0.466 | −0.039 |
| geo3k:974 | rare | 733 | 379 | 128 | 512 | 0.557 | 14 | – | – | 0.591 | +0.071 |

## 3. Gate 判定

| Gate | 要求 | 结果 |
|---|---|---|
| G1 | 12/12 题数据齐全 + minimal horizon 前缀 answer-free | ✓ |
| G2 | pooled Spearman(gap, OT) > 0 且 CI 下界 > 0 | −0.037,CI [−0.084, +0.002] ✗ |
| G3-a | `div_window_512_1024_gap` 与 best_lift Spearman > 0,CI 下界 > 0 | −0.25,CI 含 0,n=8 ✗ |
| G3-b | horizon 精确匹配 ≥3/7 且 ±1 档 ≥5/7 | 2/7、3/7 ✗ |

## 4. 解读

1. **G2 失败说明离线代理本身不可靠。** TOPD 的探测是从**同一学生前缀**重生成
   teacher/student 短窗;离线版用"缓存 teacher 轨迹的对应位置窗口"近似,前缀不同,
   warped 对齐只是把两条不同轨迹按"推理进度"粗略对应。结果是 token loss 与
   OT 距离没有正相关(甚至略负),与论文的弱正相关相反。因此 G3 的负结果应表述为
   "**这种离线近似**的 OT 定位不成立",而非"TOPD 的 OT 定位不成立"。
2. **G3 的两种解读都指向同一个行动。** (a) 若代理有效:OT 定位同样不能预测 rescue,
   Gate C 失败被复现;(b) 若代理无效(G2):G3 无判定意义。无论哪种,都不满足
   启动 minimal-bridge 训练臂的条件,与 Gate A–C 的整体 NO-GO 一致。
3. **horizon 预测系统性偏后。** "div 累计占比 ≥50%"规则对早 rescue 题
   (geo3k:845、17,observed 64)全部预测 512——div 随位置单调累积,规则天然偏向
   晚期;这是规则设计缺陷,不是 OT 统计量的独立证据。
4. **意外负相关(ot_window_512_1024 vs lift, −0.71)** 是 post-hoc、n=8,
   方向与假设相反,按预注册不追逐;如未来做敏感性分析,可把它列入检验清单。

## 5. 结论与后续

- **不启动**四臂 minimal-bridge 训练(预注册决策树触发 NO-GO)。
- 保持 **rare_success verified-FKL** 方向:proposal feasibility 的 GO +
  实验 B 证明 teacher answer-free 前缀能因果救援 frozen 学生,这两条证据不受本结果影响。
- 若要真正验证 TOPD 定位,唯一干净路径是同前缀重生成(需要 GPU,成本约 1.41×),
  不在当前决策树内;alignment/embedding 敏感性仅作研究性选项,不做自动推进。
- 产物保留:`offline_topd_probe_20260806/{summary,probe_rows,locator_analysis,run_manifest}`,
  输入 sha256 见 `run_manifest.json`。

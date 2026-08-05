# Support-Aware Prefix Intervention (Experiment B) — 2026-08-06 结果

## 0. 一句话结论

**因果救援成立,但定位一致性(Gate C)不成立。**

- 12 个 retained 题全部完成(8 片、848 条续写、strict merge 通过)。
- teacher answer-free 前缀把续写正确率从 unaided 6.3% / wrong 前缀 3.1% 提到
  **26.6%**(368 条),且正确率随前缀长度单调上升(h64 11.5% → h128 24.0% →
  h256 39.6% → h512 32.5%);wrong 前缀始终 ≈0–7%。
- **7/12 个 prompt 满足预注册救援规则**(E[p_T−p_W]≥0.20 且 P≥0.90,
  E[p_T−p_U]≥0.20 且 P≥0.90),最小救援前缀分布在 h=64/128/256/512(2/2/2/1),
  全部 answer-free(无 Answer:/boxed,唯一 skip 是 source_shorter_than_horizon ×2)。
- Gate B 通过(12 题三 arm 齐全、7 题过救援、聚合 lift 双对照为正、非 answer 邻接
  前缀、malformed/截断无法解释 lift——teacher 的 malformed/截断反而更高)。
- **Gate C 失败**:实验 A 的 gap 统计量与救援 lift 的 Spearman 均为负
  (window_512_1024: −0.13,prefix_256: −0.30,prefix_128: −0.22),局部化统计量
  不能预测哪些 prompt 被救援。

按计划 §7,Gate A–C 未全过,**不自动启动四臂训练**;rare_success 上的
verified-FKL 方向仍成立(来自 proposal feasibility 的 GO),但"最小局部化
bridge"的前提(用 gap 定位救援区间)无证据支持。

---

## 1. 产物路径与运行环境

| 项 | 值 |
|---|---|
| 运行 commit | `03fd8ea`(pinned worktree,`git_dirty=false`) |
| GPU | 8 片 × 单 GPU(0–7),student-only |
| 合并目录 | `$OUT/support_aware_opd/prefix_intervention_20260806_merged/` |
| 输入 | proposal 合并 12 个 retained 题(rare_success 11 + no_correct 1) |
| rollouts SHA256 | `continuation_rollouts.jsonl` = `9f7960a6...` |
| units SHA256 | `intervention_units.jsonl` = `65849b6d...` |
| rescue SHA256 | `rescue_comparisons.jsonl` = `5ba885a2...` |
| minimal SHA256 | `minimal_rescue_prefixes.jsonl` = `4b1c4f8b...` |
| 分片 | `prefix_intervention_20260806_s{0..7}/`(complete=true,12/12 prompts) |
| smoke | `prefix_intervention_20260806_smoke/`(1 prompt,SMOKE OK) |

协议:horizons 64/128/256/512、K=8 续写/arm、max 2048、temp 0.7、top-p 0.95、
seed 20260806、保守 verifier、Jeffreys 后验、救援阈值 mean lift 0.20 / prob 0.90。

---

## 2. 聚合结果(848 rollouts)

| arm | valid units | rollouts | correct | pass rate |
|---|---:|---:|---:|---:|
| fresh unaided | 12 | 96 | 6 | 6.25% |
| teacher prefix | 46 | 368 | 98 | **26.6%** |
| wrong student prefix | 48 | 384 | 12 | 3.12% |

teacher 前缀 pass rate 是 unaided 的 4.3×、wrong 前缀的 8.5×。

### 2.1 按 horizon(teacher vs wrong)

| horizon | teacher | wrong |
|---|---:|---:|
| 64 | 11/96 = 11.5% | 2/96 = 2.1% |
| 128 | 23/96 = 24.0% | 0/96 = 0.0% |
| 256 | 38/96 = 39.6% | 3/96 = 3.1% |
| 512 | 26/80 = 32.5% | 7/96 = 7.3% |

### 2.2 malformed / truncation(排除“干净度”解释)

| arm | malformed | truncation |
|---|---:|---:|
| unaided | 61.5% | 63.5% |
| teacher prefix | 51.9% | 54.3% |
| wrong prefix | 46.1% | 47.4% |

teacher 前缀的 malformed/截断率反而高于 wrong 前缀,正确率却高 8 倍——
lift 不是由“更干净/更短”造成的。

---

## 3. 预注册救援规则结果

7 个 prompt 存在至少一个 horizon 满足全部四条件:

| prompt | 最小救援 horizon | teacher 正确 | E[pT−pW] | P(pT>pW) | E[pT−pU] | P(pT>pU) |
|---|---:|---:|---:|---:|---:|---:|
| geo3k:2061 | 128 | 6/8 | +0.665 | 0.9997 | +0.668 | 0.9997 |
| geo3k:974 | 512 | 7/8 | +0.557 | 0.995 | +0.667 | 0.999 |
| geo3k:1141 | 256 | 8/8 | +0.888 | 1.000 | +0.557 | 0.999 |
| geo3k:264 | 256 | 6/8 | +0.666 | 0.9995 | +0.665 | 0.9995 |
| geo3k:374 | 128 | 4/8 | +0.442 | 0.993 | +0.444 | 0.993 |
| geo3k:845 | 64 | 4/8 | +0.446 | 0.994 | +0.333 | 0.950 |
| geo3k:17 | 64 | 6/8 | +0.554 | 0.994 | +0.556 | 0.995 |

最小救援 horizon 分布:64 ×2、128 ×2、256 ×2、512 ×1,不集中在答案邻接区。
7 个被救援 prompt 全部是 rare_success(唯一 no_correct 的 geo3k:1124 未过救援),
与 proposal feasibility 的 no-correct NO-GO 结论一致。

---

## 4. Gate 判定

| Gate | 要求 | 结果 |
|---|---|---|
| B-1 | ≥8 题三 arm 至少一个 horizon 齐全 | 12/12 ✓ |
| B-2 | ≥4 题过预注册救援规则 | 7/12 ✓ |
| B-3 | teacher 聚合 lift 对两个对照都为正 | +26.6% vs +3.1%/+6.3% ✓ |
| B-4 | 救援不集中在答案邻接前缀 | horizons 64–512 均出现 ✓ |
| B-5 | malformed/截断不能解释 lift | teacher 更脏反而更高 ✓ |
| C | 预注册 gap 统计量与救援 lift 正相关(Spearman) | window_512_1024 −0.13、prefix_256 −0.30、prefix_128 −0.22 ✗ |

**Gate B 通过、Gate C 失败。**

---

## 5. 下一步建议

1. 不自动启动四臂训练(计划 §7 要求 Gates A–C 全过)。
2. rare_success 上的 verified-FKL bridge(TREK 基线)仍可推进:proposal feasibility
   GO + 本实验证明 teacher answer-free 前缀能因果救援 frozen 学生的续写。
3. “最小局部化 bridge”的定位假设不成立:gap 统计量无法预测救援,若做 minimal-bridge
   臂,应把 span 选择改为“h=128–256 短前缀”这一经验区间,而不是基于实验 A 的窗口。
4. 若继续,下一实现 commit 需四臂匹配(rollout tokens/optimizer steps/prompt
   manifest/validation/GRPO group/seed),并单独记录 teacher 算力;先预注册再跑。

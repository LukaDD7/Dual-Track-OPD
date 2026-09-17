# Support-Aware Teacher-Proposal Feasibility — 2026-08-05 结果

## 0. 一句话结论

- **rare_success(17 题):可用性 64.7%、retained 11 题 / 20 条 → 有条件 GO**。
  32B teacher 能为罕见支持题稳定创造可验证正确提案,4B 学生对其 exact-token
  似然有限且不离谱(trimmed NLL 中位 0.358)。按 handoff 决策规则 4,可在
  rare_success 子集推进 verified-FKL / OPD-BRTS。
- **no_correct(11 题):可用性 9.1%、retained 仅 1 题 / 2 条 → NO-GO**。
  规则 3 触发(<25%):当前 teacher 不是"零支持题"的充分支持创造者,零支持
  创造视为未解决,不花一整夜训 FKL;需换 proposal 来源(答案提示 / 更强 teacher /
  策略引导)。
- 实验 A(gap robustness,CPU)已完成:唯一显著定位窗口 512–1024(paired
  delta AUC +0.157,CI 0.080–0.238),但在 stopped-only 分层内消失,暂无稳定
  局部化统计量。
- 实验 B(prefix intervention)已封装,输入为 12 个 retained 题。

---

## 1. 产物路径与运行环境

| 项 | 值 |
|---|---|
| 运行 commit | `03fd8ea`(pinned worktree,`git_dirty=false`) |
| GPU 映射 | 四对 `0:1, 2:3, 4:5, 6:7`(每片自装 32B teacher + 4B student) |
| 合并目录 | `$OUT/support_aware_opd/proposal_feasibility_20260805_merged/` |
| retained 缓存 SHA256 | `7f2745099b4882d0b0254b323ad98324ec9f4cad64128a5aa01792caca42f7c2`(`retained_proposals.jsonl`) |
| proposals SHA256 | `d19c7f5d10806f3e685487d09f53a1b7ad69c28db6c61bce46d85e50e192d24d` |
| 四片 | `proposal_feasibility_20260805_s{0..3}/`(各 7/7 prompts,complete=true) |
| 日志 | `$OUT/../logs/support_aware_proposals/proposal_feasibility_20260805/shard_{0..3}.log` |

协议:28 prompts(no_correct 11 + rare_success 17)、4 proposals/prompt、
max 4096、seed 20260805、保守 Geometry3K verifier、双侧 trim(0.10/0.02)
长度归一 NLL、每题保留 2 条。smoke(2 prompts)验收通过;全量 13:04–13:48
完成;strict merge 通过(`missing_uids=[]`、四片 commit/dirty 一致)。

---

## 2. 早间决策表(state-separated)

| 指标 | no_correct(11 题) | rare_success(17 题) | 总体(28 题) |
|---|---:|---:|---:|
| ≥1 条正确 proposal 的题 | 1/11 = 9.1% | 11/17 = 64.7% | 12/28 = 42.9% |
| 正确 proposal / 总数 | 4/44 = 9.1% | 26/68 = 38.2% | 30/112 = 26.8% |
| retained proposals(题数) | 2(1 题) | 20(11 题) | 22(12 题) |
| retained trimmed NLL 中位(p10–p90) | 0.052 | 0.358(0.282–0.470) | 0.326(0.163–0.467) |
| 截断率 | 5/44 = 11.4% | 14/68 = 20.6% | 19/112 = 17.0% |
| malformed 率 | 5/44 = 11.4% | 11/68 = 16.2% | 16/112 = 14.3% |
| 生成 token | 54,350 | 112,159 | 166,509 |

retained 覆盖的 12 题:geo3k:1124 / 1141 / 1242 / 17 / 1713 / 2061 / 264 /
374 / 388 / 428 / 845 / 974。

---

## 3. 决策规则对照(handoff `e74d5b8` + plan `03fd8ea`)

| 规则 | 判定 |
|---|---|
| 规则 1:可用性 ≥50% 且 retained ≥8 题 → 立即做 FKL bridge | no_correct 不满足(9.1%)；rare_success 满足(64.7%,11 题) |
| 规则 2:可用性高但 NLL 极端/大量截断 → 先测 BRTS/SGPO 兼容 | rare NLL 不离谱(中位 0.358),截断 20.6% 需报告,不触发极端分支 |
| 规则 3:可用性 <25% → 换 proposal 来源 | **no_correct 9.1% 触发**:不做小缓存 FKL,换源 |
| 规则 4:rare 成功而 no-correct 失败 → rare 上 OPD/BRTS,零支持创造未解决 | **当前状态**,按此执行 |
| Gate A(03fd8ea):retained ≥8 题 | 12 题通过,但质量偏斜 rare_success,报告需写明 |

---

## 4. 实验 A:robust teacher-gap(CPU,`gap_robustness_20260806/`)

12 个替代统计量 vs mean_gap(同一 rollout 集、同一 eligible prompts,paired
bootstrap 10k,seed 42):

| metric | n | delta AUC | CI95 |
|---|---:|---:|---|
| length_residual_gap | 37 | +0.018 | −0.006 ~ +0.047 |
| prefix_128_gap | 37 | +0.013 | −0.084 ~ +0.117 |
| prefix_256_gap | 37 | +0.037 | −0.056 ~ +0.133 |
| prefix_512_gap | 35 | +0.005 | −0.082 ~ +0.100 |
| suffix_64_gap | 37 | −0.033 | −0.143 ~ +0.076 |
| trek_trimmed_gap | 37 | +0.005 | −0.014 ~ +0.027 |
| window_0_128 / 128_256 | 37 | +0.013 / +0.008 | 均含 0 |
| **window_512_1024_gap** | 32 | **+0.157** | **+0.080 ~ +0.238** |
| window_1024_2048_gap | 23 | +0.071 | −0.017 ~ +0.160 |
| window_2048_end_gap | 23 | −0.123 | −0.271 ~ +0.017 |

读数:唯一 CI 排除 0 的是 512–1024 中段窗口,但相邻窗口无连贯趋势;
prefix 类统计量在 stopped-only 分层全部归零/转负(表观增益只存在于 length
截断层,n=3);`trek_trimmed` 与 `length_residual` ≈ 0。按 plan 要求,
实验 A 不构成 GO,给实验 B 一个待验证的位置假设(中段窗口)。

---

## 5. 下一步:实验 B(prefix intervention)已封装

- 输入:12 个 retained 题(需同时有 K32 wrong rollout,均已具备)。
- 封装脚本:`$OUT/support_aware_opd/run_prefix_intervention_20260806.sh`
  (钉死 worktree `03fd8ea` → 校验 proposal merge → 1-prompt smoke + 自动
  校验 → 8 卡全量)。
- smoke:1 题、horizons 64/128、2 续写/arm、max 256;通过标准见
  `docs/causal_frontier_bridge_executable_plan_20260806.md` §5.4。
- 全量:8 片 × 单 GPU(每片 1–2 题),horizons 64/128/256/512、K=8/arm、
  max 2048、seed 20260806;预注册 rescue 规则:`E[p_T−p_W]≥0.20 且
  P(p_T>p_W)≥0.90`,`E[p_T−p_U]≥0.20 且 P(p_T>p_U)≥0.90`。
- 之后:Gate B/C + 早间决策,再决定是否进入训练设计。

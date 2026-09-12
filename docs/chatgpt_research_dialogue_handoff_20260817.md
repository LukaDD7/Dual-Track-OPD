# ChatGPT 对话转录与 CC 纠错交接（2026-08-17）

## 文档性质与来源

- 原始共享对话：<https://chatgpt.com/share/6a831640-04e4-83ec-be34-36094013ae51>
- 对话标题：`后训练研究框架`
- 对话规模：19 轮用户问题，主题从 prefix rescue 的研究定位逐步收敛到 Visual Handoff（Question A）和 operator/path reachability（Question B）。
- 本文件是供 Codex / Claude Code（下称 CC）继续实施的**结构化转录**：保留问题演化、关键定义、否证条件和最终实施决议；不是逐字复制网页 UI、工具日志或中间检索结果。需要核对原话时以上述共享链接为准。
- 代码基线：`codex/va-opd@5bd4d60`。本地在 2026-08-17 从 `106e4eb` fast-forward 了 CC 的 32 个提交；原有未提交工作区改动未被覆盖。

## 19 轮对话的研究演化

1. 从 64 prompts、每题 K=32 的 student rollout 出发，区分 observed-zero support（0/32）与 rare success（1–8/32），并观察 teacher-correct prefix 对 rare-success 更容易救援。核心问题从“prefix 有没有用”升级为：学生 native support 与 rescueability 是否共同决定一个 prompt 能否被搬回可学习 frontier。
2. 放弃过早套用固定“六轴”框架，改为逐篇拆论文的对象、状态来源、privilege、干预、控制、学习信号、评价和排除区域。
3. 强调研究价值不在于给视觉/推理/答案段落命名，而在于构造无需人工标注、可计算且可验证的边界 proxy。
4. Cheap visual probe 应以 student forward 为主体；teacher 是 reference。单 token 峰值噪声太大，必须聚合到自然 reasoning block，并用有图/退化图的分布变化而非结构化模板标签。
5. 诊断必须最终服务于 post-training operator，而不能停在 prefix utility：候选包括 OPD/RKL、FKL、bridge 或 skip。
6. 对“学生会推理但正确路径密度低”的区域，短程 FKL 有合理性；但必须区分 sampled teacher token 的 hard CE 与真正 soft teacher-distribution FKL。
7. 三个早期 causal 变量（visual dependence、relay/transport gain 等）更适合作为机制解释，不宜直接成为昂贵的在线四路 router。
8. 主问题收敛为：RL 之前如何低成本增加可产生正负 reward contrast 的 frontier，而不是为每个样本跑 K=8/32 的完整反事实网格。
9. “teacher prefix → student rollout → suffix RL”本身与 PrefixRL 等工作高度重叠；创新不能只来自 prefix 长度超参，而要来自最小充分 scaffold、reachability barrier 和 scaffold withdrawal。
10. PW-OPSD 式路线的关键是用昂贵 intervention 定义可信 gold，再检验便宜 proxy；SFT、RL-only、SFT→RL 是训练基线，不等于机制定位。
11. sampled-token FKL 的期望是 cross entropy，且 `H(p_T,p_S)=H(p_T)+KL(p_T||p_S)`；单个 teacher sample 的 CE 是该期望的 Monte Carlo 估计，但一次 hard-token CE 不能冒充 Top-K/full-distribution FKL 实验。
12. 在 `codex/va-opd` 的 causal-probe 资产基础上，先做 gold diagnostic → cheap proxy，而不是直接扩大为四臂 STP-OPD 训练。
13. 必须拆开 scaffoldability 与 native reachability：给定 teacher state 后能否继续，不等于 student 能否自己到达该 state。
14. PW-OPSD 的 causal label 是 intervention 后 branch 是否仍可行，不是静态 token 标签；本项目的 gold 应是 teacher prefix 的 continuation outcome，并用 prompt/trajectory cluster bootstrap 控制重复观测相关性。
15. Phase C/sensitivity 的 NO-GO 说明不能要求一个统一 proxy 同时回答 handoff 与 FKL 截断。研究拆成 Question A（哪里可以交棒）和 Question B（为什么 native 到不了、应改变哪段概率质量）。
16. P-ALIGN/Prefix Teach–Suffix Fade 提供语义 block 与 change-point 思路；VLM 特点通过 image counterfactual 注入，但不能预设所有失败都来自视觉理解。
17. 最终收成两个动作：A 用 counterfactual distribution signal + reasoning segmentation 定位候选 handoff，再用 continuation 因果验证；B 只在 confirmed usable prefix 内做 support decomposition 与最小 operator intervention。
18. TA-OPD 的 disagreement 必须在 teacher/student Top-K 并集上比较，并使用 cross-gather；compatibility `C` 与 disagreement `D` 是两个量，不能把不相交 token 丢掉。
19. 对 CC 的 v1/v2 实验审计后发现：工程执行和 dose sanity 是有效的，但 A3、BIC 边界、TA-OPD normalization、micro-FKL 定义和 candidate 范围存在研究对象错位。因此 A/B 的旧 “NO”不能作为最终路线否证。

## 冻结定义

### Question A：Visual Handoff

目标不是证明视觉信号本身等于 gold handoff，而是生成一个便宜候选，再由 continuation 验证：

1. 使用同一条 teacher-correct response token 轨迹。
2. 分别在 full image 与 degraded/null image 下对 teacher 和 student forced-forward。
3. 每个 token 的视觉依赖改为**全词表 Jensen–Shannon divergence**：

   `V_t^M = JS(p_M(. | full, prefix_t), p_M(. | degraded, prefix_t))`

4. 聚合到自然 reasoning blocks，使用 `DeltaV_i = V_i^T - V_i^S`。
5. 用 one-change BIC 只产生候选边界；BIC 不显著时不得生成 `h_V`。
6. 若 split 为 `pre=z[:tau]`、`post=z[tau:]`，边界在 `B_(tau-1) | B_tau` 之间，因此中心 handoff 是 `end(B_(tau-1))`，再验证前后各一个 block。
7. 必须保留 degraded/null control 和与 gold `h*` 的 enrichment；若 v3 仍无 enrichment 且 continuation 无提升，停止把 visual score 当 prefix localizer，仅保留为 mechanism analysis。

### Question B：Local compatibility 与 path reachability

TA-OPD 风格量定义为：

- `C_t = sum_{v in TopK_S} p_T(v)`：teacher 想要的概率质量有多少已在 student 当前候选支持中。
- `D_t = KL(pbar_T^U || pbar_S^U)`，`U=TopK_T union TopK_S`：双方在重要候选并集上的分歧。
- `Q_t`：teacher 的 off-support mass 是否集中在少数清晰 target 上。
- `R_t = Dtilde_t * Ctilde_t`：learnable/compatible disagreement。
- `F_t = Dtilde_t * (1-Ctilde_t) * Q_t`：incompatible acquisition 假设。

`Ctilde`、`Dtilde` 必须按 TA-OPD 实现使用 5%/95% 分位的 robust min-max，并裁剪到 `[0,1]`。无界 z-score 会让 `1-Ctilde` 失去“不兼容程度”的语义，并可能产生负 `F/R`。

CC 现有真实结果（median `C_i` 约 0.99998）提示更重要的现象可能是：

> Teacher-forced state 上的 local compatibility 很高，但从原始 prompt 连续走完整条正确 path 的 native reachability 极低。

因此旧的 single-block → native pass-rate 测试只说明“一次局部小更新不足以跨过全路径 floor”，不能说明 soft FKL/RKL 完全无效。

## 已实施的纠错

### A 侧

- `visual_handoff.py` schema 升为 `support-aware-visual-handoff-v3`。
- reference-token `Delta log p(y_t^T)` 已替换为全词表 JS；模型每个 image condition 只做一次整轨迹 forward，JS 数学按 `chunk_size` 分块以控制临时显存。
- BIC 非显著时 `tau_block=None`；最佳但不显著的 split 只保存在 `candidate_tau_block` 供审计，不进入 continuation。
- `h_V_block` 改为最后一个 pre-change block（`tau-1`），`±1` continuation 也围绕该中心。
- 默认输出目录改为 `visual_handoff_js_v3_20260817`，避免与 v2 shard 混合。

### B 侧

- `operator_need.py` schema 升为 `support-aware-operator-need-v2`。
- `C/D` 改为 token-bank 范围的 5%/95% robust quantile normalization，并裁剪到 `[0,1]`；报告记录 normalization contract。
- `micro_operator.py` schema 升为 `support-aware-micro-operator-v2`。
- candidate 强制 `pre_hstar=True`；post-h* block 不再参加 high-F/high-R/control 选择。
- micro-FKL 改为 teacher Top-100 + complement-tail 的 soft forward KL。旧 `prefix_fkl_ce` 仍是合法的 hard-token/SFT estimator，但不再被此实验标为 soft FKL。
- 历史 `cc_micro_operator_report_v1/v2` 保留作可复现实验记录，但其中 `F_i/R_i` 机制解释和 “FKL vs RKL” 对比已被本文件 supersede。

## CC 的下一步执行顺序

1. **先跑 CPU/静态检查**：确认 v3/v2 schema、bounded `F/R`、`pre_hstar` coverage 和 soft-FKL 单测。
2. **Question A 只再跑一轮**：新目录跑 full-vocab JS + reasoning blocks + significant-BIC gate + `tau-1`/±1 continuation + degraded/null control。不要与 v2 结果拼表。
3. **按预注册 gate 决策 A**：若与 `h*` 无 enrichment 且 continuation 仍接近噪声，停止 visual localization；不要继续换 degradation、平滑器或 change-point 直到出现正结果。
4. **暂停复跑旧 single-block native-q**：它有强 floor effect，且旧 operator 不对称。
5. **先设计 region/path-level acquisition diagnostic**：只在 confirmed `h*` 之前选连续语义 region；比较 soft FKL、full-vocab RKL、hard-token CE 和 matched random/control region。首要 outcome 是 teacher-path log-likelihood/support 的 dose response，其次才是 no-prefix frontier conversion。
6. **在 region diagnostic 通过前不上 RL**：只有当连续 region 的参数更新能稳定提高 path support，并在 no-prefix rollout 中产生可检测 frontier，才进入 RL 或完整训练。

建议 region 诊断至少记录：region 起止 block/token、总监督 token 数、operator、Top-K、step/lr、更新前后 teacher-path NLL/KL、native K、随机种子、checkpoint、resolved config、数据 manifest hash、repo/backend commit 与 dirty status。

## 验证命令

```bash
python -m pytest -q \
  tests/support_aware/test_visual_handoff.py \
  tests/support_aware/test_operator_need.py \
  tests/support_aware/test_micro_operator.py \
  tests/support_aware/test_support_transition_loss.py
```

GPU 侧必须使用新的输出前缀：

```bash
VISUAL_HANDOFF_PREFIX=visual_handoff_js_v3_20260817 \
  bash scripts/hpc/launch_visual_handoff.sh 0:1,2:3,4:5,6:7
```

## 当前结论

- CC 的 segmentation、robust BIC 骨架、sharding/merge、dose sanity 和按 gate 停止的工程纪律值得保留。
- A 的 v2 “NO”针对的是 reference-token score，不是否证 distribution-level counterfactual score；v3 是最后一轮有明确停止条件的修正。
- B 的 v2 “NO”受错误 normalization、hard-token CE 冒充 soft FKL、post-h* candidate 和 native-q floor 共同影响，不能作为 operator/path 机制结论。
- 目前最值得继续的研究对象不是 token-level FKL/RKL router，而是 `local compatibility != sequential path reachability`：需要改变多长的连续 reasoning region，才能把低概率正确 mode 真正搬进 student 的 native exploration support。

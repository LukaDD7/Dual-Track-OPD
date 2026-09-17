# Rebuttal: 让 reachability proxy 变得有意义的路线（给 Codex 的辩论稿）

Date: 2026-08-16
Status: 供 Codex 审阅与回应。本文不启动任何训练、loss、checkpoint、Phase-7。

## 0. 这份文档的目的

Phase A/B/C 已修复评估泄漏并给出 NO-GO，结论是"compatibility/executability
模型在 h* 定位上不达标"。我们认同泄漏修复的必要性（label-conditioned row
selection 确实是 target leakage），也认同 symmetric cache 和 sensitivity 复检
的严谨性。但我们对 **gate 的指标设计**、**position 的处置**、**label 的噪声
上限**、**M1 特征族构造的锚点**有四个结构性异议。加上四篇 2026 年的相关工作
（PW-OPD / P-ALIGN / ADWIN / Relay-OPD），我们认为"proxy 没意义"这个结论
可能来自评估目标与部署目标错位，而不是信号不存在。下面给出证据和具体改法，
请 Codex 逐条回应。

## 1. 事实基础（已核对，全部来自本地数据与代码）

- 评估单元：(prompt_id, horizon)，194 行 / 52 prompts / 27 正例；
  horizons = {64, 128, 256, 512}，h* 分布 64→12, 128→3, 256→7, 512→5。
- position（行级，未残差化）：AUROC 0.767（CI95 0.693–0.837），AUPRC 0.602，
  heldout（grouped CV）AUROC 0.776；within-prompt h* r@1 = 0.26。
- 注册模型（OOF grouped CV）：M0 position 0.776 / M1 compat 0.479 /
  M2 takeoff 0.681 / M3 0.749；paired fold ≥ M0 的比例 0% / 24% / 40%。
- 单特征 h* r@1 天花板 ≈ 0.44（cum_student_nll、cum_top100_fkl_tail、
  delta_handoff_64、O_h^4_next），position 0.26（4 个 bin 的随机基线是 0.25）。
- 方向性证据：C_h^16 0.561 优于 C_h^16_next 0.475；O_h^4_next 0.571 优于
  O_h^4 0.502；M2（FKL_takeoff，走廊特征）0.681 明显高于 M1（端点特征）0.479。
- Phase D gold top-up 在跑：20260816 manifest（60 prompts），
  s2/s3 完成，s0/s1 未完成（14/15、11/15）。目标 ≥40 正例。

## 2. 辩论点 1：gate 指标把问题定义错了 —— h* r@1 是"极端顺序统计量"

注册 gate 要求"useful, stable within-prompt h* ranking"（r@1）。但

```text
h* = min { h : Y_{x,h} = 1 }
```

是**集合最小值**，不是"代表性可行位置"。任何单调刻画 viability 的特征
（position、cum NLL/FKL 天然单调）都会把**最后一个**可行行排高、把 h* 排低。
一个完美的单调 viability 模型在这个指标上也会失败。

数据佐证：position r@1 = 0.26 ≈ 随机；r@1 最高的恰恰是单调累积特征
（0.44）。这说明 r@1 测的是"单调方向碰巧和 h* 排序一致"，不是定位能力。

部署真正需要的不是"精确命中 h*"，而是：

1. **安全交接区**：选出的 ĥ 必须有高 P(Y=1)，最好是"第一个高概率可行点"（
   frontier 穿越），不是 argmax；
2. **单边风险**：截太早（学生接不住 → 尾部无学习信号/学坏）的代价远大于
   晚一两个 bin（只损失一点窗口）；
3. **校准**：P̂(Y=1 | ĥ) 要和真实成功率对得上，这样才敢上线。

建议把 gate 指标改为：safe-window precision@chosen、first-viable-h in top-k、
非对称窗口损失（early 惩罚 > late 惩罚）、校准曲线；h* r@1 降级为诊断指标，
不再作为 NO-GO 的唯一依据。

## 3. 辩论点 2：position 不是"不可部署"，而是"可计算的 schedule 基线"

brief 说 position 不被接受为 handoff-location proxy。这个判断只对一半：

- 如果任务是"预测 h*"，position 确实做不到（r@1 0.26），同意；
- 但部署蒸馏场景里，rollout 已经生成，trace length T 已知，τ·T 可直接计算
  （我们的 `position = h / trace_length` 就是线上可算的量），所以 position 作为
  **固定 schedule** 是零成本、可部署、可复现的基线，不是泄漏也不是空想。

PW-OPD（2605.21606）正是这么用的：残差化 position 在"正确 spine 内高熵候选"
子集上 AUROC 0.83，但**不用于截断**，而是变成递增 sigmoid 的 token 训练权重
（moderate schedule: w_min=0.25, τ=0.30, s=0.10，跨模型固定），AIME2024
+1.0、AIME2025 +1.1。也就是说，文献里 position 的合法用途是**权重/调度**，
不是 per-prompt h* 选择器。我们建议把"固定 position schedule"注册为 M0'，
任何自适应代理都必须先击败它——这才是公平的对照。

## 4. 辩论点 3：G=1 二元 label 的噪声上限

rescue_gold 是单次 rollout 的伯努利抽样。当潜在 P(success) ≈ 0.5 时，label
就是抛硬币；194 行 27 正例下，AUROC 的统计上限被 label 噪声压低。0.767 很
可能是**真实潜在 viability AUROC 的下界**，而不是上界。

建议：在歧义区域（h 接近预测 frontier 的行）补 G=4–8 次 rescue 抽样，
label 用成功率（分数目标）或多数投票，报告 (a) 对分数目标的秩相关/校准，
(b) 对多数投票二元 label 的 AUROC。这不需要全量重做，只需一个子集。

## 5. 辩论点 4：M1 compat 的构造锚点可能错了（端点 vs 走廊）

M1（C/M/O，K=16，端点 h-1）0.479 低于机会，被读作"compatibility 族没有
handoff 信号"。但 handoff viability 的定义是"交接后学生能继续走 W 步并最终
做对"——这是**走廊（corridor）性质**，不是端点状态性质。数据支持这个方向：

- M2 FKL_takeoff（未来窗口走廊）0.681 >> M1 端点 0.479；
- 交接点本身（h 处第一 token，`_next` 变体）比交接前最后 token（h-1）更有
  信号：O_h^4_next 0.571 > O_h^4 0.502；C 则相反；
- Relay-OPD（2607.26057）的交接触发器就是"教师 redirect、学生继续走"的
  **连续性不对称**——本质上是我们 cache 里已经有的 teacher-forced 走廊
  与 student continuation 的对比，只是没建成特征。

结论：不是"特征构造是瓶颈"这个结论完全错误，而是**当前特征族测错了对象**。
下一步应把走廊/不对称特征注册为新代理族（见 §7），而不是就此关闭 Question 1。

## 6. 文献证据（四篇，均为 2026，可直接打开）

### 6.1 PW-OPD — When Are Teacher Tokens Reliable?
https://arxiv.org/pdf/2605.21606

- 诊断：正确 spine 内、高 top-valid 熵候选位置，强制候选后 student-template
  续写能否恢复正确答案（real-uncertain vs benign diversity）。
- 结果：**within-problem 残差化后** oriented position AUROC 0.83（cluster
  bootstrap [0.66, 0.95]），局部不确定性 ≤0.57。
- 用途：递增 sigmoid position 权重（floor + per-seq mean），不是截断。
- 对我们：①残差化是必须补的注册视图（brief 已注册但未实现）；
  ②position 作为**权重**是有文献背书的下线路径。

### 6.2 P-ALIGN — Long-Chain Reasoning Distillation via Adaptive Prefix Alignment
https://arxiv.org/abs/2601.10064 （ACL 2026）

- 做法：**学生自己判断**当前 teacher 前缀是否"足够"（ENOUGH / NOT_ENOUGH），
  对句子做二分搜索 → 最小充分前缀；然后 prefix-aligned SFT：学生从保留前缀
  续写，**学生的续写**（不是教师尾部）作为 SFT 目标。
- 结果：超基线 >3%；固定比例截断（λ·|R|）明显次优。
- 对我们：这就是我们 rescue 协议反过来变成代理——"学生自评能否从这接下去"
  是免标签、逐 prompt、可部署的 handoff 判据，完全没在我们注册的特征族里。
  它应该作为新代理族 M5（self-sufficiency judgment），而不是 feature search。

### 6.3 ADWIN — Adaptive Windows for Horizon-Aware OPD
https://arxiv.org/abs/2605.28396

- 做法：**horizon 是线上可接受性决策**：短 teacher-anchored 前缀窗口同步
  更新，延迟的全 rollout probe 审计 prefix–full 梯度对齐，再自适应下一窗口。
- 结果：端到端 FLOP 最高省 4.1×，精度不降；教师 PPL 随位置上升、top-k
  存活率崩坍 → 后期 teacher 监督边际价值低。
- 对我们：训练期根本不需要"预测 h*"——horizon 可以由梯度对齐在线审计。
  这为 Question-1 提供了一个**决策规则族**（非模型），绕开 h* 预测难题；
  也支持"短窗口不是妥协，而是更优监督"的判断。

### 6.4 Relay-OPD — Pass the Baton: Trajectory-Relayed OPD
https://arxiv.org/abs/2607.26057

- 做法：失败前缀上教师会 redirect、学生继续原方向 → 免标签 handoff 触发器；
  教师短暂接管生成 relay leg，学生再续；轨迹长度 -50%。
- 结果：1.7B 学生 +5.73% vs OPD，八项基准最佳或次佳。
- 对我们：触发器可线上从"教师强制续写 vs 学生自由续写"算出，正是我们
  cache 里 O/C 走廊特征的机制化版本；和 §5 的观察互相印证。

## 7. 提议：让 proxy 变得"有意义"的路线

目标不变：提升 RL frontier —— 通过可靠的 prefix 截断/权重，让蒸馏后的
学生 native（无前缀）rollout 从 0/G 进入 0<k<G（mixed-support），RL 才有
可选择的模式。proxy 的价值最终要用 **downstream conversion** 衡量，而不是
AUROC 本身。

### D2（CPU-only，先做，不占 GPU）

1. 补注册过的 within-prompt residualized 视图（position 残差化后和论文 0.83
   对齐比较）；
2. 把 corridor/不对称特征（first-takeoff-token、教师 redirect 与学生继续的
   分歧点、前 W=32/64 的 top-k 存活率）从现有 cache 或一次追加 forward 派生，
   注册为 M1b；
3. 新指标：safe-window precision@chosen、first-viable-h in top-k、非对称
   窗口损失、校准曲线，全部在 prompt 级分组 CV 下重算。

### E1（小 GPU，需要注册）

1. 多采样 label：歧义子集 G=4–8 次 rescue，分数目标；
2. M5 self-sufficiency judgment：冻结学生（或小 verifier）对 (prompt, prefix)
   做 ENOUGH/NOT_ENOUGH + 置信度，评测其对 rescue gold 的 AUROC/校准；
3. M0' 固定 position schedule（PW-OPD moderate 参数）作为必须击败的基线。

### F（仍在用户批准后）

用胜出的 horizon 策略（自适应或 schedule）跑一次性蒸馏，测 native no-prefix
conversion 0/G → 0<k<G；只在该转换稳定后讨论 RL。所有 gold 阈值、rescue
协议、数据外置规则保持不变。

## 8. 给 Codex 的三个具体问题

1. **指标**：是否接受把 gate 从"h* r@1"扩展为"安全交接区 + 单边风险 +
   校准"，h* r@1 降为诊断？如果坚持 r@1 作 gate，请给出 h*（集合最小值）在
   单调 viability 模型下 r@1 可达上限的论证。
2. **position**：是否接受 position 的合法用途是"固定 schedule / 训练权重"
   （PW-OPD 路线），而不是 h* 预测器？是否接受 M0' = 固定 schedule 作为
   任何自适应代理的对照基线？
3. **新代理族**：是否同意把 P-ALIGN 式 self-sufficiency judgment（M5）和
   ADWIN 式梯度对齐审计注册为新的 Question-1 代理族，而不是当作 feature
   search 拒绝？以及是否同意在歧义子集补 G=4–8 多采样 label？

## 9. 边界声明

本稿不改变：gold 阈值、rescue 协议、数据外置、无训练/无 loss/无 checkpoint/
无 Phase-7。改动仅限评估指标、代理族注册和 label 采集协议，全部需要 Codex
回应后再冻结。

## 10. 正向构造路线：怎么做、怎么行、怎么落地（补充于 2026-08-16）

前八节回答"为什么现在的评估说明不了问题"，本节正面回答"proxy 应该被构造成
什么、如何验证它有用"。研究目标不变：让蒸馏后的学生 native（无前缀）rollout
进入 0<k<G（mixed-support），给 RL 提供可选择的模式。

### 10.1 重新定义任务

proxy 不是"h* 预测器"，而是

```text
P̂(viable | x, prefix_h, student)
```

的**校准估计器**。部署输出是每个 prompt 的交接点 ĥ（frontier 穿越点，取
"第一个校准后 P̂≥0.9 的位置"），或者一个全局 schedule。h* 只用于事后诊断
（ĥ 与 h* 的窗口差），不作为 gate。

### 10.2 四种可部署形态（从零成本到自适应，全部可上线）

| 形态 | 机制 | 新增成本 | 出处 |
|---|---|---|---|
| S0 schedule | 固定 position 权重/截断（PW-OPD moderate: w_min=0.25, τ=0.30, s=0.10） | 0 | PW-OPD |
| S1 self-judgment | 冻结学生/小 verifier 判 (prompt, prefix) 是否 ENOUGH + 置信度 | 1 次短 forward / prompt | P-ALIGN |
| S2 online audit | 短窗口同步训练 + 延迟全 rollout 探针审 prefix–full 梯度对齐 | 探针子集（训练期） | ADWIN |
| S3 trigger | 教师 redirect vs 学生继续的不对称检测 | 从现有 cache 派生 | Relay-OPD |

### 10.3 闭环验证（小规模，先于任何大训练）

1. 用本轮 35 个可救 prompts + 已有 27 正例构造评测集；歧义 band（P̂≈0.5 的
   (prompt, h)）补 G=4–8 多采样 label，做校准目标；
2. 同一批 prompts 跑三臂蒸馏：no-prefix 对照 / S0 schedule / S1 ĥ
   （窗口 [a*, ĥ] 只做 Top-100 FKL，遵守现有约束）；
3. 释放后测 native 0/G→0<k<G conversion + DynaMath heldout；
4. **proxy 的价值 = 下游 conversion/accuracy 差**；AUROC/校准只是诊断。

### 10.4 落地要求

- 单边安全：ĥ 取校准后 P̂≥0.9 的最早点，宁晚勿早；proxy 判"不可行"时
  fallback 到 full-rollout OPSD 或跳过该 prompt；
- 成本：S0=0、S1=1 次短 forward、S2=训练期探针；不需要新模型/verifier 上线；
- 确定性：同输入同输出；prompt 级分组评估；数据外置；
- 门槛：任何自适应形态必须在 paired bootstrap 下击败 S0，且校准误差可控。

### 10.5 与 RL frontier 的关系

proxy 实际测量"每个 prompt 需要的教师脚手架量"（distance-to-frontier）。
蒸馏后重测 native conversion：能预测"哪些 prompt 会从 0/G 转换到 0<k<G"
的 proxy 才是有效的。RL 只在转换稳定后讨论。

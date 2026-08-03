# TREK 竞争格局：Forward KL、OPD 与 RL-Ready Frontier

更新日期：2026-08-03。本文是
[`sft_rl_opd_narrow_survey_2026.md`](./sft_rl_opd_narrow_survey_2026.md)
的高分辨率核心矩阵。它不把“方法名”当作主要分类，而是拆开四个经常被混淆的变量：

1. **rollout occupancy**：训练前缀来自 teacher、student，还是二者的混合；
2. **divergence / update direction**：forward KL、reverse KL、RL advantage，还是组合；
3. **support state**：学生目前是 zero/rare/mixed/correct-tail/mastered；
4. **transfer unit**：完整轨迹、短窗口、策略、token、logit direction，还是 reward credit。

这四个变量共同决定方法是在**创造成功 support**、**修复学生已访问状态**、
还是**放大已存在的 reward-bearing 行为**。只比较“FKL 对 RKL”会把这些机制混在一起。

## 1. 结论先行

### 1.1 TREK 已经占据的创新空间

[TREK](https://arxiv.org/abs/2607.05339) 已经明确提出：在低通过率 prompt
上调用 teacher 或带特权上下文的同模型，筛选 verifier-passing 且学生可达的 proposal，
通过短暂 teacher-forced forward-KL/NLL 把新成功轨迹拉进学生 support，再回到普通
GRPO。它也报告了相对 direct GRPO 的 early-RL/sample-efficiency 改善。

因此，项目不能再声称：

- 首次在 RL 前用 distillation 扩展 support；
- 首次按低 pass-rate 路由 teacher proposal；
- 首次使用“distill to explore, RL to refine”的阶段式流程；
- 纯 OPD 可以直接修复从未被学生访问的完整 reasoning mode。

### 1.2 仍然有辨识度的主张

当前更强、也更可证伪的主张是：

> 不存在一个在所有 prompt 状态上都占优的 distillation operator。teacher-trajectory
> FKL 适合创造缺失 support；student-occupancy OPD 适合修复学生可达但不稳定的
> reasoning fork；RL 适合在已有 outcome variation 时做任务目标驱动的 refinement。
> operator 引起的 prompt-level support 状态迁移，决定了随后产生多少真正有梯度的
> mixed RL groups，并中介 early-RL sample efficiency。

这比“OPD 比 TREK 的 FKL 更好”更合理，也比单纯学习一个 OPD loss coefficient 更有
研究辨识度。它要求我们测量完整因果链：

```text
support state + teacher compatibility
    -> operator choice
    -> post-bridge p(correct), U_G(p), diversity, trajectory reachability
    -> realized mixed-group rate / advantage variance / useful RL gradient
    -> early-RL reward AUC and samples-to-target
```

### 1.3 对“TREK 证明 OPD 不如 FKL”的准确判断

TREK 的结果支持一个**局部命题**：对它选出的低通过率 prompt、它的 proposal source、
它的 training budget 和它实现的 OPD baseline，verified-proposal FKL 比 OPD-style
supervision 更适合 support expansion。它没有证明：

- FKL 在 rare/mixed-support prompt 上仍然优于 OPD；
- FKL 在 student-reachable trajectory forks 上优于 trajectory-aware OPD；
- FKL 的增益来自 KL direction，而不是 teacher occupancy、verified proposal、
  student-proximity selection 或额外 proposal compute；
- FKL 产生的 frontier 增量比 OPD 产生的 frontier 增量更能转化为后续 RL 增益；
- 继续 FKL 比 `FKL -> OPD -> RL` 更好。

所以当前不能“反驳 TREK 的实验结果”，但可以直接检验它的结论边界，并把贡献写成
**operator-by-state crossover** 与 **frontier-to-RL mediation**。

## 2. TREK 的完整技术路线

用户的概括“筛出错误样本，再从 teacher rollout 中选择学生 log-prob 更高的轨迹做
forward KL”需要两处修正。

第一，TREK 路由的是**低通过率 prompt**，不是训练学生的错误 rollout。第二，它在
teacher/self-context proposals 中先保留**正确轨迹**，学生 likelihood 只在这些正确
轨迹之间衡量可达性。

### 2.1 每轮算法

对 prompt `x`：

1. 从当前、无辅助学生采样 `K=16` 条 rollout，估计 `p_S(x)`；
2. 若 `p_S(x) > tau_low=1/8`，该 prompt 留在普通 GRPO stream；
3. 若 `p_S(x) <= 1/8`，调用 proposal source 最多采样 `M=4` 条轨迹；
4. proposal source 可以是 DeepSeek-V4、white-box teacher，或加了 failure-lesson
   memory 的同一个学生 checkpoint；
5. verifier 删除错误 proposal；
6. 对每条正确 proposal `y_T`，在**无辅助学生**下 teacher-force 计算 token NLL；
7. 去掉最低 10% token loss，避免 boilerplate/格式 token 让轨迹显得虚假地接近；
8. 去掉最高 2% token loss，避免孤立 rare-token outlier 让轨迹显得虚假地遥远；
9. 对剩余 token loss 做长度归一化，得到 reachability score `d_S(y_T|x)`；
10. 保留 `r=2` 条 `d_S` 最低的 verifier-passing proposal；
11. 对累计 proposal dataset 做一轮 teacher-forced NLL，即 sample-based
    `KL(q_prop || pi_student)`；
12. 更新后的 prompt 回到普通 GRPO stream，周期性重算 pass rate 与 proposal ranking。

若 hard prompt 没有任何保留下来的正确 proposal，它不会被错误 proposal 训练，而是
延迟到下次 routing refresh。TREK 还描述了异步双队列版本：student GRPO 与 teacher
proposal generation 并行；proposal 到达时用**当前**学生重新评分，以避免 stale
reachability ranking。

### 2.2 TREK 真正做对的四件事

| 设计 | 作用 | 不能被混淆成什么 |
|---|---|---|
| prompt-level low-pass routing | 把 proposal budget 用在学生探索最弱处 | 不是 sample-level 失败 token routing |
| verifier-first filtering | 防止把 teacher 错误直接注入 support | 不是 teacher likelihood ranking |
| student-proximity ranking | 在正确 proposal 中选择当前最可吸收的轨迹 | 不是选择“最优”或“最有新知识”的轨迹 |
| short FKL then GRPO | 先覆盖缺失 mode，再由 outcome reward refine | 不是永久把 teacher 当目标策略 |

### 2.3 TREK 的关键弱点与未回答问题

1. **可达不等于有新知识。**最低学生 NLL 会偏好已经接近学生的解法，可能主要提高
   surface-form likelihood，而没有增加新的 reasoning strategy。
2. **通过率不是 group utility。**它没有逐 prompt 测量
   `U_G(p)=1-p^G-(1-p)^G`、实际 mixed-group rate 或 advantage variance。
3. **缺少 mediator analysis。**论文显示 bridge 后的最终/早期性能，但没有检验
   `delta frontier -> future mixed groups -> delta RL` 的中介链。
4. **operator 与 occupancy 没完全解耦。**表格中的 OPD 是 self-context source 下的
   on-policy distillation；正文还声称 same-teacher-trajectory off-policy OPD 也较弱，
   但没有单列其 objective、完整配置和结果。公开表格不足以支持一个跨 occupancy 的
   全局“FKL > OPD”结论。
5. **等 optimizer step 不等 compute matching。**teacher proposal tokens、verifier
   calls、student rescoring 与 FKL tokens 都需要单列。
6. **student-proximity proxy 仍是 surface-sensitive。**trimmed NLL 会受长度、措辞、
   tokenizer 与推理风格影响；它不是 semantic learnability 或 causal teachability。
7. **主要优势可能是早期效率而非最终 ceiling。**TREK 自己报告 direct GRPO 训练足够久
   可在 ALFWorld 接近约 85%，因此需要把 samples-to-target 与最终上限分开。

## 3. 为什么 RL 前置状态下 mode-covering 往往优于 mode-seeking

令 `q` 是 proposal/teacher distribution，`p_theta` 是学生。

### 3.1 单个已访问 prefix 上的差别

Forward KL：

```math
D_KL(q || p_theta) = sum_j q_j log(q_j / p_j)
```

对学生 logit `z_j` 的梯度为 `p_j - q_j`。若 teacher 对 token `j` 有较大质量，而学生
`p_j` 很小，梯度仍会直接提高该 token。

Reverse KL：

```math
D_KL(p_theta || q) = sum_j p_j log(p_j / q_j)
```

其目标由学生质量加权；学生几乎不给质量的 mode 对目标贡献很小，因此更倾向保留/选择
学生已经能表达的高密度 mode。

但必须加一个重要限定：**full-vocabulary RKL 在一个已访问 prefix 上并非完全看不到未采样
token**，softmax normalization 仍会耦合所有 logits。真正严重的 missing-support 问题通常
发生在**轨迹 prefix/state occupancy**：如果进入正确 reasoning branch 的前缀概率是多个
小概率决策的乘积，学生几乎不会访问那些后续状态，OPD 就没有机会在那里获得 teacher
signal。

因此，在 true zero/near-zero pass prompt 上，teacher-trajectory FKL 的优势主要来自：

- teacher occupancy 把训练直接带到学生从未到达的正确 prefixes；
- FKL/NLL 对这些 prefixes 上 teacher token 的低学生概率施加直接惩罚；
- verifier 确保被覆盖的是 reward-bearing mode；
- student-proximity selection 控制 off-policy distance。

不能把所有功劳只归因于“FKL 数学上 mode-covering”。

### 3.2 为什么 covering 不是始终更好

当学生已经进入 rare/mixed support，问题从“没有正确 branch”变成“在自己会访问的
prefix 上哪一个局部决策不稳定”。此时 student-occupancy RKL/OPD 有四个潜在优势：

1. 训练状态与部署 occupancy 一致，减少 teacher forcing exposure bias；
2. 不必覆盖 teacher 的所有表达/策略 mode，避免 capacity dilution；
3. 可以在学生真实错误 fork 上获得 dense local correction；
4. mode-seeking 可保持学生已有的简洁、高概率成功策略，而不是复制 teacher 的多样风格。

TrOPD 给出了很强的反例：在其 student-generated prefixes 上使用 full FKL，数学平均准确率
几乎崩溃到 `1.40`；RKL OPD 为 `46.79`，只在 RKL outlier 区域切换到 FKL 后为
`49.00`，再加 off-policy guidance 的完整 TrOPD 为 `49.85`。这说明“FKL 是 covering，
所以一定更好”是错误命题；**方向必须与 occupancy 和 reliability region 匹配**。

### 3.3 四象限，而不是二选一

| Prefix / trajectory 来源 | Forward KL | Reverse KL / OPD |
|---|---|---|
| teacher / privileged proposal | TREK、BRTS teacher branch、TrOPD off-policy guidance：适合引入缺失 mode，但有 exposure bias | “off-policy OPD”定义依 estimator 而异；若仍只在 teacher tokens 上做 sampled RKL，方向与 occupancy 不匹配，不能代表标准 OPD |
| student rollout | TrOPD outlier FKL：可在已访问 prefix 上补 teacher-supported token；全量使用可能不稳定 | GKD/MiniLLM、TIP、TA-OPD、TOPD：适合修复 student-reachable states，但无法自动访问完整缺失 branch |

项目最值得检验的不是选某一列，而是**prompt/trajectory 何时从左上角切换到右下角**。

## 4. 高度相关核心文献矩阵（22 篇）

下表采用 Agentic OPD matrix 的研究问题粒度。外围背景文献仍保留在 46 篇总矩阵中。

### 4.1 Support creation 与 pre-RL bridge

| 工作 | 背景问题 | 核心假设 / RQ | 方法与关键实现 | 关键证据 | 局限 | 与本项目/TREK 的关系 |
|---|---|---|---|---|---|---|
| [TREK](https://arxiv.org/abs/2607.05339) | GRPO 在低 pass prompt 上没有 reward-bearing rollout | verified、student-proximal teacher proposal 能否通过 FKL 扩展 support，再提高 early RL？ | `K=16` 估 pass，`p<=1/8` 路由；`M=4` proposal；verify；trimmed student NLL；top `r=2`；1 epoch NLL；回 GRPO | Qwen3 1.7/8/14B 上 AIME24/25 一致提升；ALFWorld 75.8→82.8，ScienceWorld 12.5→26.7；self-context FKL 比对应 OPD 高 | 没有逐 prompt frontier/mixed-group mediation；budget 主要按 student steps；off-policy OPD 未单列 | 必须实现的最强低-support baseline；我们的贡献不能停留在“bridge then RL” |
| [ReGFT](https://arxiv.org/abs/2603.01223) | 人类 reference proof 往往不在模型 thought distribution，直接 SFT 无效 | 用 partial reference 引导模型自己生成正确轨迹，能否创造可训练 support？ | partial-reference conditioned generation → verified model-style traces → pre-RL FT → DAPO | 增加 solvable problems、正 reward 供给、DAPO 速度和最终 plateau | 依赖 reference；没有 OPD/FKL state phase diagram | 证明“模型风格的桥”比原始 teacher proof 重要；是 proposal representation baseline |
| [PACED](https://arxiv.org/abs/2603.11178) | 太易/太难样本的 distillation gradient SNR 低 | 学生能力 frontier 是否由 pass-rate uncertainty 界定？ | 用 `p(1-p)` 加权；先 FKL 覆盖、再 RKL/OPD consolidation | competence-frontier weighting 与两阶段 KL 方向优于静态 distillation | frontier 主要是优化权重，不测后续 RL mixed-group mediator | 已占据“pass-rate frontier + FKL→RKL”；我们必须强调 `U_G` 与 realized RL groups |
| [Tsallis continuum](https://arxiv.org/abs/2604.25907) | 纯 RL 在 cold-start 下难逃低 support；强 density matching 易受噪声 | 不同 density-estimation commitment 如何控制 cold-start 与稳健性？ | Tsallis 参数连续连接 RL-like exploitation 与 density estimation | 理论/实验支持：低 pass 时更强 coverage 更快，噪声大时更危险 | 没有 verifier-guided operator router 与因果 mediation | 给出 FKL 低-support 优势的理论背景，也要求测 teacher noise |
| [BRTS](https://arxiv.org/abs/2605.09725) | student prefixes 噪声大；单条 teacher rollout 高方差 | correctness-first、student-alignment-second 的 teacher branch 能否补足 OPD？ | student-context RKL + curated teacher-context FKL；`N` teacher samples；正确优先、top-K overlap 次之；全错时 gold-answer recovery；`lambda=10` | hard AIME 上 early gain 最大；Tier-2 对 AIME25 明显有益 | gold answer、同 tokenizer、teacher sampling 成本高；fallback 可使用错误近邻轨迹 | 已实现 hybrid occupancy/direction，是比“纯 OPD 反驳 TREK”更强的 baseline |
| [SGPO](https://arxiv.org/abs/2606.24064) | 完整轨迹 imitation 容易学表面步骤，不学可复用策略 | 是否可 distill strategy-conditioned distribution shift，而不是 teacher trajectory？ | `G1=8` autonomous + `G2=4` strategy-guided；只用 guided correct；选 unguided likelihood 最高轨迹；guided→unguided FKL；token KL clip；pass-gap adaptive weight；GRPO 联合 | Qwen2.5-7B 平均 42.2→52.1；去 target selection -4.0，去 KL clipping -3.4；FKL 优于 RKL | 需要强模型抽策略与 SFT warmup；策略文本质量是新混杂 | “transfer unit”突破 TREK 的直接证据：可扩展的是策略 support，不一定是完整轨迹 support |

### 4.2 OPD reliability、trajectory repair 与 KL 方向

| 工作 | 背景问题 | 核心假设 / RQ | 方法与关键实现 | 关键证据 | 局限 | 与本项目/TREK 的关系 |
|---|---|---|---|---|---|---|
| [GKD](https://arxiv.org/abs/2306.13649) | teacher-forced KD 有 exposure bias | 在 student-generated prefixes 上做 dense teacher supervision 是否更稳？ | student rollout + teacher distribution；支持多种 divergence | 建立现代 OPD 的 occupancy contract | 不处理 true missing trajectory support | 标准 OPD baseline；必须保留 exact student token/prefix |
| [Rethinking OPD](https://arxiv.org/abs/2604.13016) | 更强 teacher 不一定带来 OPD 增益 | OPD 是否同时需要 thought-mode overlap 与新增 capability？ | 诊断 overlap/capability increment，加入 offline warmup/recipe | teacher quality 单标量不足；相容性是关键 moderator | 没有后续 RL mediator | 提供 RQ2：为什么同一 support state 下 OPD 仍可能失败 |
| [TrOPD](https://arxiv.org/abs/2606.01249) | distribution mismatch 使 sampled RKL policy gradient 出现 outlier | 能否按 teacher/student agreement 分区使用 RKL 与 FKL，并用 teacher prefix 把 occupancy 拉回可靠区域？ | trust probability `min(p_T/p_S,1)`；trust region RKL；outlier top-k FKL；teacher prefix FKL + student continuation | OPD math avg 46.79；full student-prefix FKL 1.40；outlier FKL 49.00；完整 TrOPD 49.85 | trust proxy 是 token-local；缺少 prompt support transition | 最直接否定“FKL 全局优于 RKL”；支持 state/region-conditioned divergence |
| [TOPD](https://arxiv.org/abs/2606.00305) | 高 token loss 不等于真实 reasoning fork；单 token correction 无法修复 drift | near-future trajectory information 能否识别并桥接真实分叉？ | 以短窗口 OT distance 检测 divergence；向未来 50 tokens 分配 guidance | 约 30% high-loss token 属低-divergence；OPD 47.8→48.2（去 false alarms），TOPD 52.2；local OT 0.538→0.350(OPD)→0.204 | OT 成本与语义等价问题；仅短程 math | 说明 TREK 的“OPD 不会扩 support”部分可能是 baseline 太 token-local；要加入 trajectory-aware OPD |
| [TIP](https://arxiv.org/abs/2604.14084) | uniform KL 把有效 token signal 稀释 | 哪些 student entropy / teacher disagreement token 值得训练？ | token importance routing | 关键 token reweighting 优于 uniform OPD | token score 未必对应 trajectory causality | 提供 token-level operator，但不足以替代 prompt/trajectory support state |
| [Token Teachability](https://arxiv.org/abs/2605.26844) | 大 KL 可能只是不可学的 off-support disagreement | 哪些 teacher correction 位于 student local support？ | 区分 locally compatible 与 off-support teacher mass，teachability-aware OPD | raw disagreement 不足，local compatibility 预测 usefulness | local token support 不等于完整 outcome support | 为 OPD rare/mixed 状态 arm 提供必要 gate；也可替换 TREK 的 surface NLL reachability |
| [AOPD](https://arxiv.org/abs/2605.06387) | advantage-weighted OPD 有高方差、零优势梯度与 exploration bottleneck | 正优势做 exploitation，非正优势做局部 divergence minimization 是否更稳？ | asymmetric token objective：保留正 reinforcement，非正区域用 localized divergence | 弱初始化下平均提升更大，并保持 entropy | 仍依赖 student-reachable states | 是“RL 与 OPD 按局部 advantage routing”而非全局 loss mixing 的证据 |
| [DOPD](https://arxiv.org/abs/2606.30626) | privileged teacher/student signal 会混入不可部署的信息差 | 能否区分 transferable capability gap 与 privilege illusion？ | privileged teacher 与 privileged student 双路，按 advantage/probability token routing | LLM/VLM、稳定性、OOD 均优于 vanilla OPD | 双模型/特权分支成本高；复杂 gate | 提醒 proposal success 可能依赖不可迁移 privilege；需要 unaided recovery 检验 |

### 4.3 RL 与 distillation 的联合/路由

| 工作 | 背景问题 | 核心假设 / RQ | 方法与关键实现 | 关键证据 | 局限 | 与本项目/TREK 的关系 |
|---|---|---|---|---|---|---|
| [SRPO](https://arxiv.org/abs/2604.02288) | GRPO credit 粗；SDPO 早期快但后期崩 | correct sample 用 GRPO、failed sample 用 self-distillation 能否兼得速度与稳定？ | sample routing；correct→GRPO；有 correct sibling 的 failure→SDPO；entropy-aware teacher reliability | Qwen3-8B 五榜平均比 GRPO +3.4%、比 SDPO +6.3%，每步成本最多降 17.2% | all-zero group 仍无正确 sibling，不能创造 support | 是 mixed-support 的强 baseline；与 TREK 低-support 处理互补 |
| [HAPO](https://arxiv.org/abs/2603.11321) | all/low-success group 缺正样本 | 在 group 中注入一个 verified teacher sample 是否恢复 gradient？ | Bayesian confidence gate；替换失败 rollout；teacher use 随能力退火 | group-level teacher injection 提升低-support RL | teacher trajectory 进入 RL estimator 的 off-policy bias/成本 | 是“bridge 独立阶段”之外的在线 group intervention competitor |
| [RLAD](https://arxiv.org/abs/2602.22495) | 离线 KD 与后续 RL 目标错位 | distillation 能否感知 RL trust region 与 exploration？ | RL 内 selective imitation；teacher/old-policy mixture anchor；trust-region ratio | 优于 offline distill、GRPO、KL-OPD | 复杂 importance/trust calibration | 把 teacher 作为 RL update constraint，而非前置 bridge；需与 state operator 比较 |
| [Distilled RL](https://arxiv.org/abs/2607.17247) | RL reward 粗；OPD 无条件 imitation，跨 family 易失效 | teacher 能否只重分配正优势轨迹的 token credit？ | student rollout；teacher/old-student reverse importance ratio；clip；负优势权重 reset=1；每序列 geometric-mean normalization；嵌入 GRPO | DSQW-1.5B 十个数学榜平均：OPD 35.27、RL 36.86、OPD+RL 36.54、Distilled RL 40.00；pass@k/cross-family 也提升 | 仍需学生先采到正优势 trajectory；entropy case 只证明可传分布属性，不等同新 reasoning support | 强 mixed-support/RL arm；无法替代 zero-support bridge，但可能是 frontier 形成后的最佳 operator |
| [Sparse-to-Dense Reward Principle](https://arxiv.org/abs/2605.12483) | sparse reward 适合强模型探索，dense reward 适合小模型吸收 | 不同模型/阶段应如何分配 sparse 与 dense supervision？ | teacher RL → teacher-trajectory FKL → student-rollout OPD → optional student RL | 多阶段组件均有贡献 | 研究模型/阶段 allocation，不做 prompt-state causal routing | 已占据工业式四阶段 recipe；我们必须在 prompt-level mechanism 上区分 |

### 4.4 Frontier、跨模态与 teacher construction

| 工作 | 背景问题 | 核心假设 / RQ | 方法与关键实现 | 关键证据 | 局限 | 与本项目/TREK 的关系 |
|---|---|---|---|---|---|---|
| [VOLD](https://arxiv.org/abs/2510.23497) | 文本 LLM teacher 与 VLM student 初始 occupancy 不对齐 | SFT cold start 是否是 joint GRPO+OPD 生效前提？ | 先 SFT 对齐，再在纯文本数据上 GRPO+RKL OPD；失败 rollout 蒸馏 | 未 cold-start 时 distillation 几乎无效；对齐后跨模态 reasoning 提升 | 主要是文本 reasoning transfer，不定位视觉 fork | 说明 SFT 是 operator compatibility 的状态变换，不只是固定 pipeline stage |
| [W2S-OPD](https://arxiv.org/abs/2607.26246) | frontier student 可能没有更强完整 teacher | 能否从多个弱模型的 contrast direction 构造与强学生相邻的 proxy teacher？ | positive-negative weak-model logit difference + student base logits；student-rollout RKL | 弱 teacher 仍可改善更强 student；post-RL/scale/hint contrast 学到不同信号 | 依赖 logit access 与 contrast 质量；最新 preprint 待复现 | 把“teacher”从模型改成局部 capability direction，可能比 TREK 完整轨迹 proposal 更可扩展 |
| [ReOPD](https://arxiv.org/abs/2607.04763) | 多轮 agent 中 student prefixes 漂移会让 teacher reliability 下降 | replay 哪些 prefix、让谁 continuation，才能维持可靠 guidance？ | reliable teacher-prefix replay + student local action/continuation | 多轮 occupancy/reliability 优于静态 OPD | 环境与 replay engineering 重 | 将本项目的 support state 从 prompt 推广到 decision-prefix；VLM/agent 可作为第二阶段 |

## 5. 可以怎样“反驳”TREK：六个可证伪命题

这里的“反驳”不是预设 OPD 会赢，而是把 TREK 的广义解释拆成可以被数据推翻的命题。

| 命题 | 最小实验 | 支持我们时应看到 | 反对我们时应看到 |
|---|---|---|---|
| H1 support-state crossover | 在相同 prompt cohort、teacher、token budget 上，分别对 zero/rare/mixed 状态跑 FKL 与 OPD | FKL 在 zero/rare 领先；OPD 在 rare/mixed 交叉领先 | FKL 在所有 state 与预算上都领先 |
| H2 direction 不是唯一原因 | 做 teacher-prefix/student-prefix × FKL/RKL 2×2 | occupancy 与方向显著交互；teacher-prefix FKL 的优势不能归因于方向 alone | 只要换成 FKL，不论 prefix 来源都同样提升 |
| H3 trajectory-aware OPD 修复 TREK baseline 的局限 | 标准 OPD vs TOPD/teachability OPD | trajectory-aware OPD 在已有 correct tail 的 prompt 上提高 `delta U_G` 和 RL AUC | 改善 token/OT 指标但不产生更多 mixed groups 或 RL gain |
| H4 sequential transport 优于单 operator | FKL、OPD、FKL→OPD、continued FKL，桥接 token 数一致 | FKL 先创造 mode，OPD 再在 student states consolidate；顺序组合最好 | continued FKL 始终最好，OPD 阶段无增量 |
| H5 frontier 增量中介 RL efficiency | prompt-level `delta U_9` 预测下一阶段实际 mixed groups 与 reward AUC | mediation effect 显著，加入 `delta U_9` 后 arm 的 direct effect 缩小 | aggregate accuracy 提升但 `delta U_9` 不预测 future RL |
| H6 semantic/causal reachability 优于 surface NLL | TREK trimmed NLL vs teachability/strategy/fork score 选择 proposal | 新 selector 在相同 teacher budget 下产生更多 unaided recoveries 与 held-out gains | trimmed NLL 已足够，复杂 selector 无收益 |

### 5.1 当前 `k=9` 诊断可直接支持的实验

当前生成结构是 1 条 greedy + 8 条 `temperature=0.7` rollout。不要把 greedy 与 stochastic
样本简单合并成九次同分布 Bernoulli。建议同时报告：

- `p8 = correct_stochastic / 8`：用于估计 temperature-0.7 policy support；
- `greedy_correct`：单列 deterministic head；
- `mixed8 = 1[0 < correct_stochastic < 8]`：实际可观察 mixed group；
- `U8_hat = 1 - p8^8 - (1-p8)^8`：平滑 expected mixed-group utility；
- `state9`：greedy + stochastic 的操作性 cohort 标签，但不用于假装九条同分布；
- teacher local rankability：teacher score 对 8 条 stochastic rollout 的 outcome AUC；
- proposal reachability：trimmed NLL、teacher/student top-k overlap、短窗口 divergence；
- post-bridge fresh rollouts：必须重新采样，不能复用训练 proposal。

### 5.2 第一轮最小对照，不要一次铺满所有论文方法

在固定 prompt cohort 上先跑 5 个 bridge arms：

1. **No bridge**：直接进入 RL；
2. **TREK-FKL**：verified top-2 student-proximal proposal NLL；
3. **Exact-token OPD**：保持原始 response token IDs，在 student prefixes 上 RKL；
4. **FKL→OPD**：先把 zero/rare 移入 frontier，再用 fresh student rollout 做 OPD；
5. **State router**：zero→FKL，rare/mixed→OPD，mastered→skip。

第二轮只有在 standard OPD 显示局部潜力时再加入 TOPD/TA-OPD；否则会把工程复杂度和研究
问题同时放大。SRPO/Distilled RL 作为后续 RL optimizer arms，而不是第一轮 bridge arms。

公平性要求：同一初始 checkpoint、prompt IDs、proposal source、verifier、max response tokens、
student optimizer tokens；另记 teacher proposal tokens、teacher scoring tokens、verifier calls、
student rescoring tokens 与 wall-clock/GPU-hours。必须同时给 equal-student-update 与
equal-total-compute 两种视角。

## 6. 超越 TREK 的五个大维度创新点

### 6.1 从单一 hard gate 到 operator-by-state phase diagram

TREK 只有“低 pass→proposal FKL，否则 GRPO”。PACED 已经使单纯 support gate 很拥挤。
真正的新问题是：在 zero、rare、mixed、correct-tail、mastered 各状态下，哪个 operator
使 prompt 向哪个状态迁移？贡献是 phase diagram 与 crossover evidence，不是新的 scalar
weight。

### 6.2 从 aggregate gain 到可检验的 causal mediator

TREK 展示早期曲线，但不证明 support expansion 如何变成 RL gradient supply。本项目应同时
记录 prompt transition matrix、fresh-rollout mixed group、advantage variance、useful-gradient
比例和 early reward AUC，做 prompt-cluster bootstrap 与 mediation analysis。

### 6.3 从 frontier mass 到 frontier quality

相同 `p(correct)` 可能来自一个脆弱模板或多个可泛化策略。frontier 至少有四个维度：

- mass：`p(correct)` / `U_G`；
- diversity：独立成功 strategy/mode 数；
- reachability：fresh student 能否在无辅助条件下重现；
- teachability：teacher correction 是否落在 student-local token/trajectory support。

TREK 主要优化 mass + surface reachability；TOPD、TA-OPD、SGPO 提供了测 quality 的工具。

### 6.4 从完整轨迹迁移到 decision-level transport

完整 teacher solution 同时携带知识、策略、风格与冗余。更有潜力的 transfer unit 是：

- strategy-conditioned distribution shift（SGPO）；
- real trajectory fork 后的短窗口（TOPD）；
- locally teachable token set（TA-OPD）；
- privileged vs unprivileged advantage direction（DOPD）；
- weak-model contrast capability direction（W2S-OPD）。

这允许提出“最小充分 bridge”：注入足以让学生进入 RL-ready frontier 的最小 decision
information，而不是复制整条 teacher reasoning。

### 6.5 从算法有效到 compute-optimal routing

工业相关问题不是 teacher 是否有益，而是每一单位 teacher token/score/verifier call 能创造
多少新的 mixed RL group。最终 router 应优化：

```text
expected future useful-RL-groups / total teacher-and-student compute
```

这比 equal optimizer steps 更接近真实后训练系统的资源分配问题。

## 7. 汇报时建议使用的三句话

1. **TREK 已经证明 verified proposal FKL 可以把低-support prompt 拉回 RL 可探索区域，
   所以我们的贡献不能只是 SFT/Distillation before RL。**
2. **现有证据同时表明 FKL 不是全局最优：TrOPD 的 full FKL 会崩，BRTS/SGPO/TOPD
   则说明 occupancy、可达性和 transfer unit 与 KL 方向同样重要。**
3. **我们的核心实验检验 operator 引起的 support 状态迁移是否增加实际 mixed groups，
   以及这个 frontier 增量是否中介后续 RL sample efficiency。**

## 8. 本轮 llm-wiki Learning Brief

### Core Logic Chain

- TREK：缺失成功 occupancy → verified teacher proposal → student-proximal selection →
  teacher-trajectory FKL coverage → ordinary GRPO。
- BRTS/TrOPD：可靠区域与缺失区域需要不同 occupancy/direction；RKL 与 FKL 是互补的，
  不是全局二选一。
- TOPD：OPD 的真实失败可能发生在 trajectory repair，而不是简单的 mode-seeking；token
  loss 对 real fork 只有弱代理能力。
- SGPO：要 transfer 的可以是 strategy-induced distribution shift；完整 trajectory 不是唯一
  support bridge。
- Distilled RL：当学生已有正优势轨迹后，teacher 更适合作为 token credit redistributor，
  而非无条件 imitation target。

### Visual Insights

- Distilled RL 的训练图显示 OPD 早期快速上升后饱和/回落，RL 较慢但持续，简单
  `OPD+RL` 仍会后期下降；这支持“阶段/路由”而非固定 loss mixing。
- TOPD 的 OT 图显示 high-loss token 分布横跨低到高 trajectory divergence，且局部实验中
  standard OPD 只能部分降低 drift，TOPD 的短窗口修复更强。
- BRTS 的轨迹空间图把“correctness first, alignment second”画成两级 selector，并明确
  student-context RKL 与 teacher-context FKL 是两个互补分支。
- SGPO 的 KL-direction 图表明 reverse KL 会向 guided single strategy 收缩，而 FKL 在吸收
  guided strategy 的同时保留 autonomous modes；但其 ablation 又显示 reachability selection
  与 KL clipping 是稳定性的必要条件。

### Critical Friction Points

- TREK 表中只清楚列出 on-policy `OPD (self-context)`；正文关于 same-teacher-trajectory
  off-policy OPD 的主张缺少单列配置和结果，不能据此宣布跨 occupancy 的 FKL 全局胜利。
- TREK、BRTS、SGPO 都用 student likelihood/overlap 选“可达”target，但这些 proxy 可能
  偏向 surface similarity；是否真的增加新 strategy 仍未被充分测量。
- FKL 的 mode coverage 只有在 teacher proposal 正确、可吸收且与部署任务一致时才是优点；
  否则也会覆盖错误、冗余或不可迁移的 mode。
- TOPD 的 OT distance 与 semantic equivalence 并不相同；它适合作为诊断，不应未经验证
  就成为主实验的复杂默认实现。

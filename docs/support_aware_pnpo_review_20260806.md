# PNPO 论文研读(arXiv 2608.01418)+ 与 support-aware 实验线的对照 — 2026-08-06

## 0. 一句话

PNPO 研究"**复用 rollout 批次做多次 learner 更新**"时的 off-policy 修正:把
累积 importance ratio 换成**因果前缀上的似然比几何平均**,并配一个
位置相关的 hard-rejection 接受门。4-epoch 下三个基准都拿到最好
Avg@32(宏平均 50.24,比 GSPO 高 3.00 pp),且在匹配的 2400 次更新预算下用
1/4 的新生成 response 达到与 1-epoch 相当的最终性能(49.66 vs 49.56),
墙钟时间约 2.36× 加速。

它与我们实验线的直接关系:我们的 rare_success verified-FKL 方向正是
"在缓存的 teacher proposals(离线数据)上训练",论文的 limitation 明确说
离线数据 / replay / train-inference mismatch 是未测但自然的后续场景——
PNPO 的权重设计可以直接套进我们的缓存复用训练,也回应了 GPU 实例
"利用率低会被回收"的算力约束。

## 1. 论文摘要与方法

| 项 | 内容 |
|---|---|
| 问题 | 复用 rollout 批次做多轮更新 → 后期更新 off-policy;精确修正需要累积 importance ratio,乘积形式动态范围失控 |
| 方法 | PNPO:`w_PN(t) = C_{1:t}^{1/t} = exp((1/t) Σ_k log ρ_k)`,前缀似然比的几何平均;偏置但保留因果前缀依赖、压缩对数尺度、保序 |
| 目标 | `J = E[ (1/G) Σ_i (1/L_i) Σ_t sg[M_{i,t}·w_PN·Â_i,t] · log π(y_t\|x,y_<t) ]`,GRPO 式组内优势、detached surrogate、1/L_i 归一化 |
| 接受门 | `M_{i,t}`:位置相关尺度 h(t,L_i)=√(L_i/t) 放宽早位置区间,越界**硬拒绝**(不是 clip 到边界) |
| 设置 | DeepSeek-R1-Distill-Qwen-1.5B;DAPO-Math-17k;每步 256 prompts×8 responses=2048;minibatch 64 组(512 条);1 或 4 epoch;两种设置各 2400 次优化更新;32×H20;seed 42 单跑 |
| 评测 | AMC23 / AIME24 / AIME25;Avg@32(每题 32 条采样,temp 0.7,top-p 0.9);macro = 三基准无权重均值 |

结果(Table 1,best observed Avg@32):

| PPO epochs | GRPO | GSPO | PNPO |
|---|---:|---:|---:|
| 1 | 48.25 | 49.42 | 50.05 |
| 4 | 47.05 | 47.24 | **50.24** |

- 4-epoch PNPO 三个基准各自最高(AMC23 80.94、AIME24 38.85、AIME25 30.94),
  15 次评测中 14 次高于 GSPO,终局领先 2.66 pp。
- 匹配 2400 更新预算:4-epoch PNPO 150 批后 49.66 ≈ 1-epoch 600 批后 49.56,
  只用 1/4 的新 response。
- 墙钟:1-epoch 37.7–38.0h(1.00×);GSPO-4 22.4h(1.69×);**PNPO-4 16.0h(2.36×)**;
  到达平均奖励 0.25 的首达时间 15.4h → 6.4h。

作者自述限制:单一 1.5B 模型、三个数学基准、每配置单跑;接受门与 response 级聚合
未消融(不能单独归因于前缀归一化);mismatch 只由"fresh batch 上的重复更新"诱导,
异步收集 / replay / **离线数据** / **train-inference mismatch** 是不同且未测的缺口。

## 2. 与 TOPD 和我们的实验线的对照

三条线现在都围绕"前缀":

| 工作 | 前缀的角色 | 机制 | 我们的对应证据 |
|---|---|---|---|
| TOPD(2606.00305) | 固定学生前缀 + 近未来轨迹注入 | 高 loss 且高 OT 定位发散点,OT transport plan 构造 soft target | 实验 C(离线 OT 探测)NO-GO |
| 实验 B(prefix intervention) | teacher 前缀作为 context 注入 | answer-free 前缀因果救活 frozen 学生续写 | 26.6% vs 6.3%/3.1%,7/12 题救援 |
| PNPO(2608.01418) | 前缀长度的几何平均作为 off-policy 权重 | 偏置归一化 + 位置相关硬拒绝门 | 待用:缓存 proposal 复用的权重设计 |

PNPO 对我们最有用的三个点:

1. **缓存复用训练的正确姿势**:verified-FKL 若在 12 个 retained proposals 上
   多 epoch 训练,后期更新必然 off-policy(学生越来越不像采样时的 behavior policy)。
   原始累积 ratio 动态范围不可控,长度归一化 response ratio(GSPO)丢前缀结构;
   PNPO 的几何平均恰好是折中,且它自己的 limitation 点名离线数据是未测场景——
   我们的缓存训练就是那个场景。
2. **接受门文化一致**:PNPO 的 hard-rejection 门、我们 rescore 的 acceptance gates、
   实验 C 的 `div = high-loss AND high-ot`,都是"按 token 过滤再聚合",可以直接对齐。
3. **算力预算**:4-epoch 复用拿到 2.36× 墙钟加速、1/4 新 response,正是我们
   GPU 实例"利用率低会被回收"约束下的具体方案;若做训练臂,可把
   "PNPO 权重 + 4 epoch 缓存复用"作为一个显式臂,而不是默认 1 epoch。

## 3. 对我们决策树的更新

保持既有结论不变(不启动 minimal-bridge 训练臂;rare_success verified-FKL 为
当前唯一 GO 方向),新增两条"如果推进 verified-FKL 训练,应采用"的实现意见:

1. **权重**:多 epoch 复用缓存 teacher proposals 时,采用前缀几何平均权重
   (PNPO 式)或至少把 raw cumulative ratio / GSPO 式长度归一化作为对照臂;
2. **门**:沿用位置相关 hard-rejection 门与 verifier/answer-free 过滤,
   与已有 rescore/实验 B 的 gate 语义一致。

另外:PNPO 的单配置、单跑、纯文本数学设置,正好留给我们可补的空白——
Geometry3K-VL(视觉)、4B student + 32B teacher、支持状态分层(rare vs
no-correct)、预注册多臂匹配。这些都是后续训练设计里的显式对比维度。

## 4. 文档与产物索引

- 本轮实验结论:`docs/support_aware_offline_topd_probe_results_20260806.md`(NO-GO)
- 本轮实验方案:`docs/support_aware_offline_topd_probe_plan_20260806.md`
- 代码:`src/dual_track_opd/support_aware/offline_topd_probe.py`(schema v1)
- 运行产物:`$OUT/support_aware_opd/offline_topd_probe_20260806/`

# Offline TOPD Probe(实验 C):用缓存轨迹复现 TOPD 的 OT 发散定位,检验能否救回 Gate C

## 0. 一句话结论(方案摘要)

实验 A 的 gap 类统计量无法定位救援区间(Gate C 失败:Spearman 全为负);
TOPD 论文(arXiv 2606.00305)用"高 token loss **且** 高短窗 OT 距离"做发散定位,
是一个更强的定位器,但它需要对每个候选位置现场重生成 teacher/student 短窗(开销 1.41×)。
本方案把 TOPD 的**探测一半**改成纯离线:学生轨迹用缓存的 K32 wrong rollout,
teacher 轨迹用缓存的正确 teacher proposal,在 4B 学生的文本 embedding 空间算 K=50
短窗 OT,再检验 OT 定位统计量是否预测实验 B 的 rescue lift(Gate C')。
全部 CPU 执行,不重生成、不训练。

## 1. 背景与要回答的问题

```text
问题 A(已答):teacher gap 能定位发散点吗?      -> 不能(实验 A,唯一显著窗口 512-1024 但不连贯)
问题 B(已答):answer-free teacher 前缀能救学生吗? -> 能(实验 B:26.6% vs 6.3%/3.1%,7/12 题过救援)
问题 C(失败):gap 统计量能预测哪题被救援吗?       -> 不能(Spearman 全负)
问题 C'(本方案):OT 统计量(TOPD 式)能预测哪题被救援、预测哪个 horizon 吗?
```

TOPD 的探测方式:对高 loss 候选位置 t,退到 t−1 固定学生前缀,teacher/student 各生成
K=50 短窗,在 embedding 空间算 OT 距离;高 loss **且**高 OT = 真发散点。它比
per-token gap 多一个"未来轨迹是否真的分叉"的信号,这正是实验 A 缺失的维度。

## 2. 数据可用性(2026-08-06 实测)

| 项 | 值 |
|---|---|
| retained 题数 | 12(rare_success 11 + no_correct 1) |
| teacher proposal 内容长度 | 400–4096 token(全部 ≥ 64) |
| 每题 wrong K32 rollout | 1–32 条(用最 student-probable 的 1 条为主,其余做次级分析) |
| rescue 行 | 每题 3–4 个 horizon(teacher len < 512 的题缺 512 行) |
| 已观测 minimal rescue horizon | 7 题:64×2、128×2、256×2、512×1 |
| per-token 对齐 | teacher/student log-prob 与 response_token_ids 严格对齐(scored hash 相等) |

输入文件:

```text
proposal_feasibility_20260805_merged/retained_proposals.jsonl   # teacher 轨迹(token ids + content mask)
diag_full_k32_20260804_merged/rollouts.jsonl                    # 学生轨迹 + per-token teacher/student log-prob
prefix_intervention_20260806_merged/rescue_comparisons.jsonl    # 每 (题,horizon) 的 posterior lift
prefix_intervention_20260806_merged/minimal_rescue_prefixes.jsonl # 7 题 minimal horizon
```

## 3. 预注册设计(实现已按此写死)

### 3.1 轨迹与对齐

- 学生轨迹 S:每题取**最 student-probable 的 wrong rollout**(与实验 B 的 wrong 前缀选择一致),
  仅 content mask 内 token。
- teacher 轨迹 T:每题 top-1 retained 正确 proposal(与实验 B 的 teacher 前缀来源一致)。
- 对齐:`--alignment warped`(主分析)——学生窗口起点 t 映射到 teacher 位置
  `τ(t)=floor(t·L_T/L_S)`,即"推理进度"对齐;`--alignment absolute` 作为敏感性分析。
  这是与 TOPD 唯一的近似点:TOPD 从同一学生前缀重生成 teacher 短窗,离线版用缓存轨迹对。
- 窗口:默认 K=50(论文同款),`--stride 1`。

### 3.2 OT 距离

- embedding:学生模型(Qwen3-VL-4B)文本 tower 的 `embed_tokens`,float32、CPU、L2 归一化。
- ground cost:`c(i,j) = 1 − cosine(emb_T[i], emb_S[j])`。
- OT 距离:等质量 K=K 下的最小匹配(`scipy.optimize.linear_sum_assignment`,确定性、
  无超参),`D_OT = mean(匹配 cost)`。

### 3.3 发散点定义(复刻 TOPD 探测)

- per-token loss gap:`gap_t = log p_T(y_t^S) − log p_S(y_t^S)`(缓存数组直接算)。
- `high_loss` = gap 在该 rollout 的 top 20%(`--high-loss-quantile 0.80`)。
- `high_ot` = OT 距离在该题所有窗口的 top 20%(`--ot-quantile 0.80`)。
- `div_t = high_loss AND high_ot`(真发散点,即 TOPD 的"real divergent point")。

### 3.4 每题的 locator 统计量

| 统计量 | 含义 | 角色 |
|---|---|---|
| `div_window_512_1024_gap` | [512,1024) 内 div 占比(与实验 A 唯一显著窗口对齐) | **primary** |
| `ot_window_512_1024_gap` | 同窗口的 OT 均值 | secondary |
| `div_prefix_{64,128,256,512}_mass` | 前 h 个 token 内 div 占全题 div 的比例 | secondary |
| `predicted_minimal_horizon` | 最小 h 使 div 累计占比 ≥50%,否则 512 | 用于 horizon 匹配 |
| `spearman_gap_ot` / `pearson_gap_ot` | 复现论文的"loss 与 OT 弱正相关"检验 | G2 sanity |

### 3.5 与实验 B 的对照

- rescue 强度 outcome:`best_lift` = 该题所有 horizon 里
  `teacher_minus_wrong_posterior_mean` 的最大值(`best_gt_wrong_probability` 为次级)。
- horizon 匹配:对 7 个 rescued 题,比较 `predicted_minimal_horizon` 与
  `observed_minimal_horizon`(精确匹配数、±1 档匹配数,档位 64<128<256<512)。

### 3.6 Gates 与决策树

| Gate | 要求 | 失败含义 |
|---|---|---|
| G1 | 12/12 题有 teacher 轨迹、≥1 wrong rollout、rescue 行;minimal horizon 前缀 answer-free(复用 `prefix_leakage_reason`) | 数据不可用,停 |
| G2 | pooled Spearman(gap, OT) > 0 且 prompt-bootstrap CI 下界 > 0 | 离线 OT 代理无效,停 |
| G3 | primary locator 与 best_lift 的 Spearman > 0 且 CI 下界 > 0,**或** horizon 精确匹配 ≥3/7 且 ±1 档 ≥5/7 | 定位仍不成立,记录负面结果 |

```text
G3 PASS -> 设计 minimal-bridge 训练臂:每题 horizon 用 OT profile 的 h*,四臂匹配
G3 FAIL -> 不启动 bridge 训练;保持 rare_success verified-FKL 方向;把本结果写入报告
```

## 4. 命令

```bash
export DTOPD_PYTHON=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python
export DTOPD_MODEL_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models
export DTOPD_OUTPUT_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD

# smoke(2 题、K=16、stride 8,约 2-3 分钟含模型加载)
SMOKE=1 bash scripts/hpc/run_support_aware_offline_topd_probe.sh

# 全量(12 题、K=50、stride 1,CPU,预计 <10 分钟)
bash scripts/hpc/run_support_aware_offline_topd_probe.sh
```

直接等价命令(不依赖 runner):

```bash
$DTOPD_PYTHON -u -m dual_track_opd.support_aware.offline_topd_probe run \
  --config configs/experiment/support_aware_offline_topd_probe.yaml
```

## 5. 产物

```text
$DTOPD_OUTPUT_ROOT/support_aware_opd/offline_topd_probe_20260806/
  summary.json              # complete、gates、decision
  probe_rows.jsonl          # 每题 locator 统计量 + observed/predicted horizon + lift
  locator_analysis.json     # 全部 Spearman + bootstrap CI + horizon 匹配 + gates + provenance
  run_manifest.json         # config + 输入 sha256 + git state
```

## 6. 风险与限制(写进报告的已知项)

1. **离线近似的本质差异**:TOPD 从同一学生前缀重生成 teacher 短窗;离线版用缓存
   teacher proposal 的对应位置窗口,前缀不同。warped 对齐只近似"同一推理阶段"。
   G2( pooled loss-OT 相关性)就是这一近似的有效性检验:若 G2 都不通过,离线 OT 定位
   不可用,但**不**否定 TOPD 原方法。
2. **样本量小**:12 题、7 个 rescued 题;G3 的 CI 会很宽,只做"是否值得继续"的筛选,
   不做效应量估计。
3. **K=50 固定**:论文同款;若 G3 通过但 horizon 匹配差,下一步才做自适应 K 的变体。
4. **只用最 student-probable 的 wrong rollout**:次级分析可扩到全部 wrong rollouts
   取均值,本轮不做。
5. **embedding 选择**:用学生 4B 的 embed(与 TOPD 的"embedding 空间"定义不完全一致,
   论文未指定模型);本轮固定,不做 teacher embed 变体。

## 7. 实施状态

- [x] `src/dual_track_opd/support_aware/offline_topd_probe.py`(editable install 指向主 checkout)
- [x] `configs/experiment/support_aware_offline_topd_probe.yaml`
- [x] `scripts/hpc/run_support_aware_offline_topd_probe.sh`(SMOKE=1 支持)
- [x] smoke 已跑通:2 题、complete=true、G1 pass、decision 正常输出
- [ ] 全量 12 题(待运行)
- [ ] 结果入档 + 决定是否推进 minimal-bridge 训练臂

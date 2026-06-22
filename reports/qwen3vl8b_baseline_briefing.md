# Qwen3-VL-8B Baseline 内部诊断分析汇报

## 一句话结论

我们已经完成 Qwen3-VL-8B baseline raw responses 的第一版保守确定性诊断分析。当前结果适合用于内部汇报、错误分析和后续 OPD 实验设计，但还不能作为论文中的 official benchmark metric。整体来看，模型在部分高覆盖 MCQ / short-answer 任务上表现稳定，但在数学、复杂多选解析、隐藏标签和需要 judge 的开放式任务上，当前 deterministic scorer 覆盖率不足，需要接入官方或社区标准 evaluator。

## 数据与产物

Raw JSONL 不进入 Git，保存在 HPC 路径：

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/qwen3vl8b_baseline/raw_responses
```

Repo 内当前保留的是可审计的 summary / score / analysis 产物：

- `reports/tables/qwen3vl8b_baseline_raw_summary.csv`
- `reports/tables/qwen3vl8b_baseline_scores.csv`
- `reports/tables/qwen3vl8b_baseline_failure_audit.csv`
- `reports/tables/qwen3vl8b_baseline_score_diagnostics.csv`
- `reports/qwen3vl8b_baseline_diagnostic_analysis.md`

本次共纳入 16 个 preferred baseline raw files，总样本数 59,144。

## 方法说明

当前 scorer 是 `conservative_v1`，设计目标是保守、确定性、可审计：

- MCQ：只解析明确选项，例如 `Answer: C`、`(C)`、`C.`、`The answer is C`。
- Short answer：做 lowercase、去标点、去冠词、空格压缩后的 exact match。
- Numeric answer：只接受单一明确整数或小数。
- MMVet：标记为 `needs_judge`，不做确定性打分。

这个策略的优点是不会为了提高分数而过度猜测答案；缺点是覆盖率会偏低，特别是数学题、自由格式输出、隐藏答案或需要 judge 的任务。

需要特别说明：高覆盖率不等于 official metric 可靠。覆盖率高只说明当前 raw response 里有 ground truth，且我们的 parser 能把 prediction 和 ground truth 都转成可比较的字符串或选项。错误样本仍然可能来自非模型能力因素，例如答案别名没有归一、数字/单位格式不一致、VQA 官方 soft-agreement 没有实现、URL/标点被 normalization 改坏、或者官方 evaluator 本来需要 LLM/judge 抽取答案。因此高覆盖数据集可以作为内部诊断信号，但论文主表仍需要官方或社区标准 evaluator。

## VLMEvalKit 和当前 scorer 的区别

VLMEvalKit 是面向 VLM benchmark 的完整评测框架，而我们当前的 `conservative_v1` 是 repo 内部诊断脚本。二者目标不同：

| 维度 | VLMEvalKit / official evaluator | 当前 `conservative_v1` |
| --- | --- | --- |
| 目标 | 生成可复现、可比较、接近 leaderboard 的 benchmark 结果。 | 快速审计 raw responses，定位 parser / failure / coverage 问题。 |
| 数据准备 | 通常内置 benchmark 下载、格式转换、split 管理。 | 只读取已经生成好的 raw JSONL 和 manifest。 |
| 推理流程 | 可以统一调模型、分布式推理、保存 prediction。 | 不负责推理，只做 post-hoc scoring。 |
| 答案抽取 | 对 MMBench 等任务可使用 LLM-based answer extraction；部分 benchmark 有专用规则。 | 只用保守 regex / normalization；宁可 unparsed，也不猜。 |
| Metric | 实现或对齐 benchmark 官方 metric，例如 CircularEval、soft agreement、judge scoring。 | 只输出 parser-conditional accuracy、coverage、unparsed、length。 |
| 论文可用性 | 可作为 paper table 候选，前提是记录版本、prompt、judge model。 | 只能作为 internal diagnostic，不能写成 official accuracy。 |

以 MMBench 为例，MMBench 官方建议使用 VLMEvalKit；其评估中有 CircularEval，并可用 GPT-4 等 LLM 将 free-form prediction 映射到选项。我们的 scorer 只识别明确的 A/B/C/D 输出，因此更保守、更透明，但不等价于 MMBench official score。

参考：

- https://github.com/open-compass/VLMEvalKit
- https://github.com/open-compass/MMBench
- https://arxiv.org/html/2307.06281v5

## 总体结果

| 指标 | 数值 |
| --- | ---: |
| 数据集数 | 16 |
| 总样本数 | 59,144 |
| 被 scorer 成功计分的样本 | 50,200 |
| 计分覆盖率 | 84.9% |
| 被计分样本中的正确数 | 24,907 |
| Parser-conditional weighted accuracy | 49.6% |
| 非低覆盖 deterministic rows 的 macro accuracy | 56.9% |
| Unparsed rows | 8,233 |
| Unparsed rate | 13.9% |
| Length rows | 493 |
| Length rate | 0.8% |
| Error rows | 0 |

这里的 accuracy 是 parser-conditional accuracy，即只在 scorer 成功解析的样本上计算。覆盖率低的数据集不能把该 accuracy 解读为完整数据集准确率。

对应的可视化图表已生成：

```text
reports/figures/qwen3vl8b_baseline_diagnostic_scores.svg
```

图中柱子表示 parser-conditional accuracy，黑点表示 coverage。低覆盖数据集即使柱子较高，也不能直接解释为完整 benchmark 表现。

## 数据集分层

### 高覆盖 MCQ deterministic diagnostics

这些数据集覆盖率接近或达到 100%，当前数字适合作为内部 regression tracking 和错误分析依据：

| Dataset | Coverage | Parser-conditional accuracy | 备注 |
| --- | ---: | ---: | --- |
| MMBench | 100.0% | 89.7% | 仅 2 个 length rows；paper reporting 仍需 VLMEvalKit / CircularEval。 |
| MMSI-Bench | 100.0% | 32.4% | 覆盖率好，但准确率偏低，值得做错误类型分析。 |
| MindCube-Bench | 100.0% | 33.5% | 大样本集，低分可能指向空间/结构推理薄弱。 |
| ScienceQA-IMG | 100.0% | 89.8% | 表现强，可作为 smoke/regression anchor。 |
| ViewSpatial-Bench | 100.0% | 39.7% | 视觉空间能力较弱，是后续视觉侧 OPD 的重点候选。 |

### 高覆盖 rough exact-match diagnostics

这些不是官方指标，但覆盖率高，能提供内部趋势信号：

| Dataset | Coverage | Parser-conditional accuracy | 备注 |
| --- | ---: | ---: | --- |
| GQA | 100.0% | 70.6% | 官方 GQA 还包含 consistency / validity / plausibility 等指标。 |
| VQAv2 | 99.9% | 72.2% | 官方 VQA 使用 10-answer soft agreement，不是单答案 exact match。 |
| MathVista | 97.5% | 60.9% | mixed visual math，需官方 evaluator 才能论文汇报。 |
| ReMI | 94.5% | 23.8% | 覆盖率尚可但分数低，建议抽样看错误是否来自答案格式或真实能力。 |

### 低覆盖 diagnostics

这些数据集当前不能用 accuracy 代表完整数据集表现：

| Dataset | Coverage | Parser-conditional accuracy | 主要问题 |
| --- | ---: | ---: | --- |
| BLINK | 50.0% | 78.5% | 一半样本未解析，部分 ground truth 为 hidden。 |
| DynaMath_Sample | 56.9% | 26.0% | numeric exact 过保守，很多输出无法归一成单一数字。 |
| MMMU_Pro_10 | 17.9% | 30.1% | 多选解析覆盖很低，需更标准的 answer extraction。 |
| MMMU_Pro_4 | 38.0% | 52.2% | 同样受选项解析覆盖限制。 |
| MV-MATH | 1.1% | 78.3% | 只有 23 个样本被计分，另有 364 个 length rows；该数字几乎不能代表整体。 |
| MathVerse | 16.4% | 29.5% | MathVerse 不只是数字输出，numeric exact 不适合作为主评估。 |

### Needs judge

| Dataset | 当前处理 | 说明 |
| --- | --- | --- |
| MMVet | `needs_judge` | 开放式回答，需要 judge-based evaluation，不进入 deterministic aggregate。 |

## 数据集题型和样例解释

下面的解释用于帮助理解这些 benchmark 大致在测什么。当前 repo 内的 audit CSV 不保存完整题面和图片，只保存 row_id、prediction、ground truth、解析结果和 audit_reason；因此这里的“样例”主要展示真实 audit 中的预测-答案形态，而不是完整原题。完整题面需要回到 HPC raw JSONL 或原始 benchmark 数据中抽样查看。

| Dataset | 主要题型 | 当前 audit 中的真实样例形态 | 说明 |
| --- | --- | --- | --- |
| BLINK | 视觉感知/相对关系判断，常见二选一或多选一。 | row_id `val_Relative_Depth_16`: prediction `(A) A is closer`，ground truth `(B)`。 | 这类题很适合分析视觉依赖，但当前有 hidden label / unparsed 问题，覆盖率只有 50%。 |
| DynaMath_Sample | 动态/几何/数学类短答案，常见数值输出。 | prediction `3.141592653589793`，ground truth `2.0944`。 | numeric exact 很保守，很多带推导、单位、表达式的答案会 unparsed。 |
| GQA | 图像问答和组合式视觉推理，答案常为物体、属性、关系、yes/no。 | row_id `05515938`: prediction `cockatoo`，ground truth `parrot`。 | 覆盖率 100%，但官方 GQA 还有 consistency / validity / plausibility 等指标。 |
| MMBench | 多选视觉理解 benchmark，模型输出通常是 A/B/C/D。 | row_id `449`: prediction `B`，ground truth `A`。 | 当前 MCQ 解析覆盖很高，但 official reporting 应用 VLMEvalKit / CircularEval。 |
| MMMU_Pro_10 | 多学科、多选项、专业知识/图像理解任务。 | row_id `test_History_1`: prediction `Economic prosperity and population growth`，ground truth `B`，parsed_prediction 为空。 | 模型可能输出选项文本而不是字母，导致 conservative parser 无法计分。 |
| MMMU_Pro_4 | 四选项版多学科专业任务。 | row_id `test_Art_113`: prediction `A`，ground truth `C`。 | 覆盖率比 10 options 高，但仍有大量 unparsed。 |
| MMSI-Bench | 多模态/空间或结构相关选择题。 | row_id `0`: prediction `D`，ground truth `C`。 | 覆盖率高但准确率低，适合作为错误分析候选。 |
| MMVet | 开放式多模态问答，需要 judge 判断语义等价。 | row_id `v1_0`: prediction `-1 or -5`，ground truth `-1<AND>-5`。 | deterministic scorer 不适合，必须 needs_judge。 |
| MV-MATH | 视觉数学题，可能有选项、公式、推导或数值答案。 | row_id `1`: prediction `B`，ground truth `B`，但 numeric parser 没解析。 | 这个样例说明我们把它暂时设为 numeric_exact 不够合适，需要官方/专用 evaluator。 |
| MathVerse | 视觉数学推理，包含多选和自由回答。 | row_id `1`: prediction `C`，ground truth `D`，numeric parser 没解析。 | MathVerse 不是纯数字输出，用 numeric exact 会严重低覆盖。 |
| MathVista | mixed visual math，答案可能是数字、文本、选项或表达式。 | row_id `1`: prediction `0.023`，ground truth `1.2`。 | 当前 normalized exact 会把 `0.023` 归一成 `0 023`，这类数值格式应交给官方 evaluator 或更专门的 numeric parser。 |
| MindCube-Bench | 结构/空间/立方体或方位推理，多为选择题。 | row_id `among_group002_q0_1_1`: prediction `D`，ground truth `C`。 | 覆盖率 100%、准确率低，可能是视觉空间推理短板。 |
| ReMI | 多模态推理/匹配类短答案或选择式任务。 | row_id `0`: prediction `2`，ground truth `1`。 | 覆盖率较高但分数低，需要抽样判断是真错还是答案格式问题。 |
| ScienceQA-IMG | 带图科学问答，多选题。 | row_id `2`: prediction `A. weather`，ground truth `B. climate`。 | 当前 choice parser 能处理 `A. text` 格式，适合做 regression anchor。 |
| VQAv2 | 通用视觉问答，官方使用 10 个 annotator answer 的 soft agreement。 | row_id `393225000`: prediction `http://foodiebaker.com`，ground truth `foodiebakercom`。 | 这个样例显示 normalized exact 可能把 URL/标点处理成非官方形式，错误可能是 normalization mismatch。 |
| ViewSpatial-Bench | 视角、方位、空间关系判断，多为选择题。 | row_id `1`: prediction `A. left`，ground truth `D. back`。 | 覆盖率高、分数低，是视觉侧 OPD 的重点候选任务。 |

## 关键观察

1. **当前 baseline 对格式清晰的 MCQ/短答案任务比较稳定。**  
   MMBench、ScienceQA-IMG、GQA、VQAv2 的内部诊断分数较高，说明 Qwen3-VL-8B baseline 在常见 VQA/QA 任务上有可用基础。

2. **高覆盖数据集仍可能包含 evaluator mismatch。**  
   GQA、VQAv2、MathVista 这类数据集虽然覆盖率高，但当前并非官方 evaluator。比如 VQAv2 官方是 10-answer soft agreement，我们这里只做单答案 normalization；MathVista 的数字/表达式答案也可能被普通文本 normalization 处理坏。因此这些结果应理解为“内部趋势信号”，不是 official accuracy。

3. **空间/结构推理类任务暴露出明显弱点。**  
   ViewSpatial-Bench 为 39.7%，MindCube-Bench 为 33.5%，且二者覆盖率都是 100%。这类结果更像真实能力问题，而不是 parser 覆盖问题，适合成为视觉侧 OPD 的重点分析对象。

4. **数学类任务当前主要受 evaluator 不匹配影响。**  
   MV-MATH 和 MathVerse 的 coverage 很低，不能直接说模型数学准确率就是表中数字。MathVerse 包含多选和自由回答，官方/社区评估通常需要更复杂的答案抽取或 judge。

5. **低覆盖数据集不能用 accuracy 排名。**  
   例如 MV-MATH 的 parser-conditional accuracy 是 78.3%，但只覆盖 23/2009 个样本，这个数字只表示少数被清晰解析的数字答案中正确率较高，不代表 MV-MATH 整体。

6. **没有 error rows，说明 raw response 读取和基础 schema 可用。**  
   当前问题集中在答案解析和 metric 对齐，而不是 raw generation pipeline 失败。

## 对 Dual-Track OPD 的启发

这批 baseline 结果对后续研究设计有三点直接价值：

1. **视觉侧 supervision 应优先关注高覆盖但低分的视觉依赖任务。**  
   ViewSpatial-Bench、MindCube-Bench、MMSI-Bench 这类任务覆盖率高、诊断分低，适合做 visual-anchor token / chunk analysis，因为分数低更可能来自模型能力而不是 parser failure。

2. **弱视觉依赖或语言先验较强的数据集需要单独分层。**  
   GQA、VQAv2 等高覆盖 short-answer 任务可作为整体 QA 能力参考，但不能自动说明模型真正依赖视觉。后续应结合 image ablation、counterfactual image、visual-dependence score 来区分文本侧和视觉侧学习信号。

3. **数学和开放式任务应先补 evaluator，再纳入主实验结论。**  
   MathVerse、MV-MATH、MMVet 现在不适合作为 deterministic baseline 主结论。下一步应接入官方 evaluator / lmms-eval / judge-based scoring，再决定是否用于论文主表。

## 汇报时建议怎么讲

建议使用如下表述：

> 我们完成了 Qwen3-VL-8B baseline raw responses 的第一版内部诊断评估。当前 scorer 是 conservative deterministic scorer，目标是审计和定位问题，而不是替代官方 benchmark metric。整体 59,144 个样本中有 50,200 个被成功解析并计分，覆盖率 84.9%。在高覆盖 MCQ/短答案任务上，模型表现出稳定 baseline；但在空间推理、结构推理和部分数学任务上暴露出明显短板。数学类和开放式任务需要官方或 judge-based evaluator 后才能纳入论文主表。

如果需要一句更面向研究动机的总结：

> Baseline 结果显示，不同 VLM benchmark 的可解析性和视觉依赖程度差异很大。统一 token-level OPD 很可能把大量弱视觉或格式驱动样本与真正视觉依赖样本混在一起优化，因此后续 Dual-Track OPD 需要显式区分文本侧信号和视觉侧信号，并在空间/结构推理任务上重点验证 visual-anchor weighting 的收益。

## 下一步建议

1. 接入 official / community-standard evaluators：
   - GQA official evaluator
   - VQAv2 official soft-agreement accuracy
   - MMBench VLMEvalKit / CircularEval
   - MathVerse / MathVista official or lmms-eval path
   - MMVet judge-based evaluation

2. 对高覆盖低分数据集做 failure clustering：
   - ViewSpatial-Bench
   - MindCube-Bench
   - MMSI-Bench
   - ReMI

3. 为 Dual-Track OPD 构造 visual-dependence analysis：
   - 对同一问题做 image ablation / blank image / counterfactual image。
   - 标记答案 token 或 chunk 是否受视觉输入影响。
   - 比较 text-track OPD、visual-anchor OPD、dual-track OPD 的收益差异。

4. 论文汇报中把当前结果标为 internal diagnostic：
   - 不写成 official accuracy。
   - 保留 coverage、unparsed、length rows。
   - 对低覆盖数据集只做定性或 parser-conditional 讨论。

## CPU 实例抽样命令

如果要从 raw JSONL 中每个数据集抽 2 个样本，查看题面、选项、模型预测、推理/解释和 ground truth，可在 CPU 实例运行：

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD

git fetch origin
git switch analysis/qwen3vl8b-baseline-scoring
git pull --ff-only origin analysis/qwen3vl8b-baseline-scoring

python -m dual_track_opd.eval.sample_raw_examples \
  --raw-dir /inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/qwen3vl8b_baseline/raw_responses \
  --manifest data/manifests/qwen3vl8b_baseline_preferred_raw_files.jsonl \
  --samples-per-dataset 2 \
  --strategy first \
  --out-jsonl reports/qwen3vl8b_baseline_sample_examples.jsonl \
  --out-md reports/qwen3vl8b_baseline_sample_examples.md
```

如果希望每个 raw file 取“开头和结尾附近”的样本，而不是前两条，可把 `--strategy first` 改成：

```bash
--strategy even
```

抽样结果是小文件，可以用于人工分析；但仍建议先检查大小和内容再提交：

```bash
wc -l reports/qwen3vl8b_baseline_sample_examples.jsonl
sed -n '1,120p' reports/qwen3vl8b_baseline_sample_examples.md
du -h reports/qwen3vl8b_baseline_sample_examples.*
```

# MMF SFT 单源复现核对：数据 + 训练配置 + 评测口径（2026-09-03）

> 背景：MMFineReason 论文（arXiv:2601.21821）Figure 7 / §5.4 称 MMFineReason-123K SFT（基座
> Qwen3-VL-8B-Instruct，与我们相同）达到 **73.3 avg**；而我们的单源消融 `mmf_only_1ep`
> （`qwen3vl_sft_mmf122k_1ep/global_step_1774`）在 B 段六项上 Avg 仅 0.3714，且
> GQA 0.3759 / ViewSpatial 0.0940 / MMBench 53.6 相对 base 大幅下滑。本文记录逐项核对
> 结论：**数据没问题；训练配置与论文 recipe 有 8 处实质偏离；评测口径与论文不可比**
> （判分器、解码预算、停表规则均不同）。结论先行：0.3714 vs 73.3 的差距 ≈
> 训练配方偏离 × 评测口径不可比，两者叠加，并非数据错误。

---

## 1. 数据核对：✅ 无问题（ChatGPT 检查项 #3/#5）

### 1.1 来源与下载（ModelScope，已存档）

- 数据集：`OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking`（HF 与 ModelScope
  双源，文件按大小核对一致）。
- 下载：ModelScope（本集群实测单流 ~9.2 MB/s、16 并发 ~90 MB/s，HF 直连仅 ~0.65 MB/s）：
  ```bash
  modelscope download --repo-type dataset OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking \
    --local-dir $DTOPD_ROOT/dataset/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking \
    --include 'data/*.parquet' --max-workers 16
  ```
- 本地：18 个 parquet、**122,603 行**、7.19 GB，文件名与 HF 布局一致。
- 详见 `docs/dataset_mmfinereason_cauldron.md`（2026-08-19 获取记录）。

### 1.2 逐项核对

| ChatGPT 怀疑点 | 核对结果 | 证据 |
|---|---|---|
| 用了正确字段 `qwen3vl_235b_thinking_response` | ✅ | `convert_mmfinereason_sft.py` 取 `row.get("qwen3vl_235b_thinking_response", "")` 作 assistant target；全 122,603 行 pass_rate==0.0（= 论文 123K 的 pass_rate=0 难度定义，§4.2）；response 均长 CoT，均值 ~14.3K 字符，87.3% 含 `<answer>` 尾标签 |
| 123K → 113K 的 10K 差异 | ✅ 已解释 | `prep_sft_single_dataset.py` length guard（max_seq_len 12288）丢 **6,749** 行超长样本（top：MMR1 3,224 / GameQA-140K 1,660 / FineVision-raven 660 / BMMR 379 / Euclid30K 196），再切 **2,317** 行 val → train **113,537**。122,603 − 6,749 − 2,317 = 113,537，账目闭合。注意论文训练用 32,768 token packing，长样本不会被丢；我们的 12288 上限是 warmup 管线的历史约束，非数据本身问题 |
| 数据内容正确（question 含 `<image>`、answer 结构） | ✅ | 抽查训练样本：user = question（含 `<image>` 占位），assistant = 长 CoT 尾部 `Therefore, the final answer is <answer>C</answer>.` |
| CE 标签只覆盖 assistant | ✅ | verl `multiturn_sft_dataset.py`：`role=="assistant"` 时 `loss_mask=1`（剥 generation_prompt），否则 0；`sft_loss` 按左移 roll 消费 loss_mask。与 LLaMA-Factory 默认行为一致 |
| 是否误用了 pass_rate≠0 的其它子集 | ✅ 无 | 全行 pass_rate==0，正是论文 123K 定义 |

**结论：数据侧（下载、字段、过滤、账目）全部正确，113,537 训练行是 122,603 经
「12288 长度护栏丢弃 6,749 + 2,317 val」的确定性结果。**

---

## 2. 训练配置核对：❌ 8 处实质偏离论文 recipe（ChatGPT 检查项 #4）

论文 Table 8（SFT Params）：AdamW，**LR 1e-5**，cosine，**WD 0.0**，**epochs 3**，
**warmup 3%**，**seq len 32,768 + packing**，batch 32，liger kernel，**max pixels 768×768**，
min 32×32；框架 LLaMA-Factory。训练 2B/4B/8B-Instruct「under the same setup」。
**论文全文无一处提及 freeze vision tower / projector**——ChatGPT 的「vision tower and
projector frozen」是推测，论文原文不存在该设置（详见 §2.2 权重证据， ours 确实训了
vision tower，但论文没说它没训）。

我们 `run_sft_warmup.sh`（mmf122k_1ep 复用它启动）：

| 项 | 论文 Table 8 | 我们 mmf122k_1ep | 偏离 |
|---|---|---|---|
| LR | 1e-5 | **5e-5** | **5×** |
| WD | 0.0 | 0.01 | 小 |
| warmup ratio | 3% | 10% | 3.3× |
| epochs | 3 | **1** | **3×** |
| seq len | 32,768（packing） | 12,288（no packing，动态 bsz） | 长度护栏连带丢 6,749 行 |
| global batch | 32 | 64（token 预算主导，软上界） | 2× |
| max pixels（训练） | 768×768 = 589,824 px | **无上限**（默认上限 16,384·factor² ≈ 4.19M px，即 2048² 等效） | **~7× 面积** |
| 框架 | LLaMA-Factory（+liger） | verl sft_trainer | 等价性假设 |

（注：epoch 数 1 是消融设计「单源 1ep 全量」故意的，但与论文 3 epoch 不同。）

### 2.2 权重证据：vision tower 确实被训练（非冻结）

对 base `Qwen3-VL-8B-Instruct` vs `ckpt .../global_step_1774/huggingface` 的
safetensors 逐张量 diff：`model.visual.blocks.0.attn.proj.weight` max_abs_diff
**5.48e-3**、`qkv.weight` 5.12e-3、`mlp.linear_fc1.weight` 5.41e-3（全参 8.77B，
lora_rank=0，exclude_modules=None）。**vision tower + projector 均被更新**。
论文未声明冻结 → 两边都无法用「冻结与否」解释差距，但需注意：我们以 5× LR +
7× 图像分辨率上限全参训练 vision tower，是比论文（全参、768²、LR 1e-5）更激进的
视觉侧更新，可能放大视觉感知的漂移。

### 2.3 各偏离的影响定性

- **LR 5e-5（vs 1e-5）+ WD 0.01 + warmup 10%**：SFT 全参 8.77B 上 5e-5 偏激进；论文
  的 1e-5 更保守。MMF 长 CoT 目标长且分布尖锐，LR 过大易过冲，破坏 MCQ 短答案
  格式先验（A 段 fmt 合格率 geo3k 0.3910→0.4393、mmf 0.4940 的对照可见 sft_6939
  已在格式上受损，单源臂更极端）。
- **1 epoch vs 3**：训练量少 3 倍；但注意 sft_6939（混合 3ep）同样在 GQA/MMMU-Pro/
  MMBench/ViewSpatial 上相对 base 下滑（GQA 0.5541 vs 0.6163），说明「混合长 CoT SFT
  伤短答案 MCQ」不是 epoch 数单因，而是分布 + 配方共同作用。
- **训练图无 768² 上限**：高分辨率 → 超长视觉 token 序列（Qwen3-VL patch 16/merge 2），
  attention 冗余 + 显存/吞吐代价，论文 §5.5 明确说 2048² 对推理类任务「diminishing
  returns」且默认 768²。
- **12288 无 packing**：丢 6,749 行长 CoT（5.5%），恰是论文 32,768 能容纳的最长推理
  样本（MMR1/GameQA 恰是推理密集 source）。

---

## 3. 评测口径核对：❌ 与论文不可比（ChatGPT 检查项 #1）

论文 Table 10 + §B.2：**VLMEvalKit**、vLLM BF16、greedy T=0、top-p 1、top-k −1、
**max tokens 32,768**、**rep penalty 1.05**、max pixels 2048²、**无 system prompt**、
判分用 **compass-verifier（LLM-as-a-Judge）** 替代字符串精确匹配；benchmark 组成
（Table 3）：MMMU val、MathVista mini、MathVision test、MathVerse mini、DynaMath test、
LogicVista、VisuLogic、ScienceQA、RWQA、MMBench-EN、MMStar、AI2D、CharXiv（reas/desc）
——**13 项里没有 GQA / ViewSpatial / MMMU-Pro / ReMI**。

我们：lmms-eval v0.7.1 openai 后端 + vLLM serve、T=0、**max_new_tokens 按基准 128–4096**
（viewspatial 256 / gqa 128 / mmmu_pro 2048 / dynamath 4096）、无 rep penalty、
规则判分（MCQ 字母抽取 / exact match）、GQA/ViewSpatial/MMMU-Pro/ReMI/DynaMath/MMBench。

即：**两套数字根本不在同一评测协议下，73.3 不可作为我们 0.3714 的对照目标。** 更关键
的是，我们的口径对「长 CoT 模型」系统性不利，证据如下。

### 3.1 ViewSpatial 0.0940：主要是判分/截断伪影（量化）

复现 lmms-eval scorer（`viewspatial/utils.py::viewspatial_process_results`：取
`results[0].split("\n")[-1]` 再 `re.search(r"\b([A-D])\b")`）于
`samples_viewspatial.jsonl`（5,712 题）：

| 臂 | lmms 官方分 | 复算分 | none 预测 | 首字母（停表文本任意处） | 停表文本最后一个字母 |
|---|---|---|---|---|---|
| mmf_only_1ep | 0.0940 | 0.0940（复现成功） | **3,911/5,712 (68.5%)** | 0.2518 | 0.2838 |
| base | 0.4231 | 0.0.4231 | 0 | 0.4231 | 0.44231（同） |
| sft_6939 | 0.2435 | 0.2435 | 2,293 (40.2%) | 0.2454 | 0.2460 |

- mmf 臂 **99.7% 响应撞 256-token 上限**、且 prompt 要求「Reply only to the corresponding
  option」但 SFT 后模型只会长 CoT（`<think>` 风格开头），响应在 reasoning 中途被截断，
  last-line 无字母 → 68.5% 判 none → 计 0 分。base 直接答「A. right」类短答案。
- 即 **0.0940 里约六成是「截断+抽取失败」伪影**；换「停表文本任意字母」口径即 0.25 左右。
- 但要诚实：0.25~0.28 仍低于 base 0.4231——mmf SFT 确实伤了 ViewSpatial 短答案
  能力（与 sft_6939 同向），只是幅度被评测伪影放大了 ~3×。

### 3.2 GQA 0.3759（vs base 0.6163）：~24 分为能力+分布实损，非纯伪影

- 12,578 题中 68.1% 是短直答（<40 字符，直接给单词），31.9% 是被 `until=["\n",".",","]`
  停表 + 128 token 截断的长 CoT；answer-tag 重抽取仅 +0.3 分（0.3759→0.3786），
  CoT 文本中 gold 词出现率仅 0.253（CoT 行内）。
- 即：GQA 的下滑**大部分是真实能力/分布漂移**（MMF 数据全是难题长 CoT，没有
  「单词直答」监督；论文不报 GQA，其 13 项里也无此项），小部分是格式失配
  （base 直答命中 until 停表的高格式契合）。

### 3.3 DynaMath：重判分后 mmf 显著占优（0.5695 vs base 0.3739）

用 `<answer>` 标签 + 数值归一化重打 5,010 题：mmf_only_1ep **0.5695** vs base
0.3739。论文式 LLM judge 下 mmf 臂大概率还会更高（CoT 正确但答案格式不合规则
判分器的样本）。lmms 原始 0.6253 与我们的复算差 ~5.6 分，主要在 answer-tag
提取/数值 tolerance 的规则差异。**mmf SFT 的正向价值在数学推理上是真实的。**

### 3.4 MMMU-Pro：mmf 0.2312 低于 base 0.3960，但 55.1% 响应撞 2048 cap

- 44.9% 响应含 `<answer>` 尾标签（模型确实会输出它），但 lmms 的 mmmu_pro 判分
  取响应**首个** standalone 选项字母——长 CoT 中途字母先于 `<answer>` 出现，判错。
  按 `<answer>` 标签重打：mmf 0.2815（+5 分）、sft_6939 0.1416→0.2861。
- base 的 0.3960 也是同规则（直答首字母即答案），无此失配。

### 3.5 ReMI（B 段五项之一）：mmf_only_1ep exact 0.3662 为八臂最高

（见 `sft_rl_full_results_ptdpo_20260831.md` §4）——在 answer-only 短答案 + 严格
exact 口径下 mmf 臂仍是第一，再次说明其损害集中在「MCQ 短答案 + CoT 格式冲突」
的评测组合上，而非全面退化。

### 3.6 小结

| 基准 | 我们口径 | 伪影方向 | 论文协议下预期 |
|---|---|---|---|
| ViewSpatial | until/截断/首字母 | **严重低估 mmf**（~3×） | 不在论文 13 项内 |
| GQA | exact + 128 tok | 中度低估（+0.3 分） | 不在论文 13 项内 |
| DynaMath | exact | ~5.6 分规则差 | 论文有，judge 更高 |
| MMMU-Pro | 首字母 | 低估 ~5 分 | 论文是 MMMU val 非 Pro |
| MMBench | judge（Qwen3-VL-32B） | 中性 | 论文有 MMBench-EN |
| ReMI | 全分母 exact | 中性偏严 | 论文无 |

**论文 73.3 与我们 0.3714 无法直接对话：benchmark 组不同、judge 不同、解码预算
（32,768 vs 128–4096）不同、停表规则不同。唯一同名的 DynaMath 上我们重判分后
mmf 0.5695 vs base 0.3739（+19.6），方向与论文「MFR-8B DynaMath 83.4 vs
Qwen3-VL-8B 73.2」一致（幅度小，符合 1ep vs 3ep + recipe 偏离预期）。**

---

## 4. 综合结论与建议

1. **数据无错**（§1）。113,537 = 122,603 − 6,749（超 12,288 护栏）− 2,317（val）。
2. **训练配方 8 处偏离**（§2），其中影响最大的三处：LR 5×、epoch 1 vs 3、训练图
   无 768² 上限（连带 vision tower 以 5× LR 全参更新）。
3. **评测口径不可比且对长 CoT 模型系统性不利**（§3）：ViewSpatial 0.0940 中约
   六成是截断+抽取伪影；DynaMath 重判分后 mmf 反超 base ~20 分。
4. 因此「mmf SFT 掉 30 分」的叙事不成立；真实图景是「配方偏离 + 评测失配」的
   叠加，且 mmf SFT 在数学推理上的增益是真的。

### 建议（按优先级）

1. **先修评测，再谈重训**：
   - 对 viewspatial/mmmu_pro 类 MCQ：判分改为「优先 `<answer>` 标签，fallback 首行/
     末行字母」；max_new_tokens 提到 ≥4096（或按论文 32,768）；去掉对长 CoT 的
     until 截断。
   - 对 dynamath：判分用 `<answer>` + 数值归一化（本文 §3.3 复算脚本口径）。
   - 复跑 B 段 nojudge 四臂（base/mmf_only/sft_6939/junior_617）即可量化「伪影
     vs 真实损失」的边界。
2. **如需与论文 73.3 对齐，重跑 paper-aligned SFT**（单独实验，不覆盖现有臂）：
   冻结与否论文未说 → 默认全参（与论文一致），LR 1e-5、wd 0、warmup 3%、cosine、
   **3 epoch**、batch 32、**训练图 768² 上限**、seq 32,768 + packing（verl 无 packing
   可用 length 桶排序近似，或直接丢弃超长行但记录比例）；评测换 VLMEvalKit 口径
   （judge + 32,768 tokens + rep penalty 1.05 + 2048²）。
   预期：若重训后仍到不了 73.3 量级，则差异归于「LLaMA-Factory vs verl 细节 +
   评测协议」，届时再决定是否值得继续追。
3. **保护既有结论**：`sft_rl_full_results_ptdpo_20260831.md` 的四臂对比在同一口径内
   自洽（横向对比有效），仅「与论文 73.3 的纵向对比」不成立——文档中已用「单源
   1ep 消融」限定表述，无需回改表格，但建议在表头加一行「口径与 MMFineReason 论文
   不可比」的脚注。

---

## 附：复算脚本与数据路径

- ViewSpatial scorer 复现 + 变体：本文 §3.1（脚本逻辑内联，样本
  `eval_runs/vision_opd_project_baseline/{arm}_nojudge/lmms/viewspatial/Vision-OPD-4B/*samples_viewspatial.jsonl`）。
- GQA / DynaMath / MMMU-Pro 重判分：§3.2–3.4（样本 jsonl 同目录布局，按
  `filtered_resps` + `target` 字段重打）。
- 权重 diff：base `models/Qwen3-VL-8B-Instruct` vs
  `fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_mmf122k_1ep/global_step_1774/huggingface`。
- 论文原文：arXiv:2601.21821（PDF 抽取文本 `/tmp/mmf_paper.txt`，仅本机临时）；
  Table 3/8/10 见 §2/§3 引用处。

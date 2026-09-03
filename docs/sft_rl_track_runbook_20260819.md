# SFT-then-RL Track Runbook (2026-08-19)

Owner: SFT-then-RL 范式对比组。目标：SFT（Cauldron + MMFineReason-123K）→ RL
（可验证推理集）的两段式训练，与组内 SFT-only、RL-only 两支对齐评估。

## 1. 环境（cu132，已确认可用）

- Env: `$LZY_ROOT/envs/va-opd-qwen35-v090-cu132-r595-v1`
  torch 2.13.0+cu132 / vLLM 0.27.1 / transformers 5.12.0 / flashinfer 0.6.16.post3 /
  flash-attn 2.8.3 (SM90) / verl 0.9.0 (tag 483b8a00, V0 trainer)
- Backend: `$LZY_ROOT/fc-opd-storage/backends/verl-qwen35-v090-cu132`
- 平台硬约束（8×H200, driver 595.58.03, 共享 lowp 实例）：
  - **`NCCL_NVLS_ENABLE=0`**（实例无 Fabric Manager，NVLS multicast 报 CUDA error 401）
  - **`trainer.use_v1=False`**（verl 0.9.0 V1 在本实例 transfer_queue 卡死，V0 已验证）
  - `FLASHINFER_WORKSPACE_BASE=$LZY_ROOT/.cache/flashinfer`；
    `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
  - GPU 节点无外网；依赖已在共享 NFS prefix 上装好，节点直接消费
- CPU 侧 import 验证通过；SFT / GRPO 两个 hydra `--cfg job` 干跑均 EXIT=0

## 2. 数据（结构与下载状态）

### MMFineReason-SFT-123K（HF: OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking）

- 18 shards × ~6.8K rows ≈ 123K rows，7.2G；图片内嵌 parquet（`image: struct<bytes,path>`）
- 字段：`question`（含 `<image>` 占位）、`qwen3vl_235b_thinking_response`（`<think>` 长 CoT）、
  `answer`、`pass_rate`（本子集全 0）、`caption`、`source`、`is_consistent` 等
- 本地下载状态：3/18 shards 完整（2026-08-19 08:4x），转换器按 shard 幂等落盘

### the_cauldron（ModelScope: AI-ModelScope/the_cauldron；HF 原始: HuggingFaceM4/the_cauldron）

- 50+ configs（子集），每行 `images`（list<struct<bytes,path>>）+ `texts`
  （list<struct<user,assistant,source>>）；总计 ~169G
- 本地下载状态：127G 已完成，仅 16 个 `.incomplete` 文件（2026-08-19 08:5x），接近完成

### 转换 / 切分（scripts/sft_rl/）

```bash
# Cauldron -> verl SFT 格式（messages+images；可 --subsets 过滤，--combine 合并）
python3 scripts/sft_rl/convert_cauldron_sft.py \
  --input-dir dataset/the_cauldron \
  --output fc-opd-storage/outputs/fc_opd/sft_rl/cauldron/cauldron_sft.parquet --combine
# MMFineReason -> verl SFT 格式
python3 scripts/sft_rl/convert_mmfinereason_sft.py \
  --input-dir dataset/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking/data \
  --output fc-opd-storage/outputs/fc_opd/sft_rl/mmfinereason/mmfinereason_sft.parquet --combine
# SFT/RL 池不相交切分（按 image_hash+user 文本去重，输出 sft_train/val + rl_train/val）
python3 scripts/sft_rl/split_sft_rl.py \
  --sft <cauldron.parquet> <mmfinereason.parquet> --rl <verifiable_rl.parquet> \
  --out-dir fc-opd-storage/outputs/fc_opd/sft_rl/split
```

verl 默认 SFT 数据列：`messages`（list<role,content>）+ `images`；RL 默认列：
`prompt`（chat）+ `images` + `reward_model`（`{"style":"rule","ground_truth":...}`）。
geometry3k GRPO 文件已就绪：`fc-opd-storage/outputs/fc_opd/geometry3k_full/train.parquet`
（2101 行）；官方 test 集已转 GRPO val：
`fc-opd-storage/outputs/fc_opd/sft_rl/geo3k_grpo_val_test.parquet`（601 行，与 train 不相交）。

## 3. 官方 recipe（verl 0.9.0 backend）

- SFT VLM：`examples/sft/vlm/run_qwen3_vl_2b_fsdp.sh`（sft_trainer + fsdp2 + VLM processor）
- GRPO VLM（Geo3K canonical）：`examples/grpo_trainer/run_qwen3_vl_8b_fsdp.sh`
- SFT→RL 衔接：verl 无一键链式脚本；SFT 以 HF 格式存 `trainer.default_local_dir`，
  RL 配置 `actor_rollout_ref.model.path=<SFT-ckpt>` 直接续训（FSDP ckpt 同时含 HF 权重）。
- 本组 7 月已有 Qwen3-VL-4B GRPO 生产配置（`scripts/hpc/run_verl_fc_opd_overnight.sh`）可参考。

## 4. GPU 一键验证（geometry3K 先行）

```bash
# SFT smoke（84 行 geometry3k messages+images，4 卡）
bash scripts/sft_rl/run_sft_qwen3vl_geo3k_smoke.sh
# GRPO smoke（geometry3k train 2101 行，rollout n=2，4 卡，V0）
bash scripts/sft_rl/run_grpo_qwen3vl_geo3k_smoke.sh
```

日志落 `fc-opd-storage/logs/`。跑通后正式链路：
SFT（Cauldron+MMF）→ ckpt → GRPO（geometry3k / 其它可验证集，
`actor_rollout_ref.model.path` 指向 SFT ckpt）。

### 已知坑与修复（2026-08-19 GPU 实测）

- **GRPO 默认 naive reward 按 `data_source` 精确路由**（如 `hiyouga/geometry3k`）；
  本仓库 train parquet 的 `data_source=geometry3k`、自造 val 用
  `geometry3k_official_test`，都会 `NotImplementedError`。**修复**：用
  `reward.custom_reward_function.path=file://.../geo3k_reward.py` +
  `name=compute_score`（已写入 `run_grpo_qwen3vl_geo3k_smoke.sh`）。该 reward
  用 mathruler `grade_answer` + LaTeX 归一化（`\SQRT { 221 }`→`\sqrt{221}`），
  支持 `\boxed{}`/`<answer>`/数字兜底；正式 MMFineReason RL 的规则奖励可在此基础上扩展。
- **SFT checkpoint 需含 HF 权重才能直接喂 GRPO**：verl 只有
  `checkpoint.save_contents` 含 `hf_model` 才落 HF 权重（否则只有 FSDP shards）；
  smoke 脚本已改为 `[model,hf_model]`。GRPO 的
  `actor_rollout_ref.model.path` 指向 `<ckpt>/global_step_*/huggingface`。
- SFT 日志里的 `processor.__call__` kwargs 提示 = transformers 5.12 兼容性警告
  （verl 仍按旧签名传入，训练/val 正常）；尾部 TCPStore/NCCL 报错 =
  训练完成后的 Ray/GCS teardown 噪音（checkpoint 与验证均已落盘）。

### 状态记录

- 2026-08-19 09:13：GRPO smoke 通过（8/8 步，val reward 0.4375→0.5，自定义
  geo3k_reward 生效，运行段 0 报错）。
- 2026-08-19 11:48：SFT smoke 带 `hf_model` 重跑通过，
  `global_step_10/huggingface/model.safetensors`（17.7GB，713 tensors）完整落盘。
- SFT→RL 衔接验证：GRPO 的 `SFT_RL_MODEL` 指向
  `<sft-ckpt>/global_step_10/huggingface` 即可（脚本已支持覆盖 model.path）。

## 5. SFT-then-RL 范式要点（调研结论）

- **warmup 的角色**：不是“提前学一点”，而是格式/风格锚定（`<think>` 结构、可读性、
  语言一致性）+ 给 RL 一个不会塌缩的初始化。DeepSeek-R1 冷启动仅数千条长 CoT（相对
  DeepSeekMath 的 77.6 万 SFT 样本），Kimi k1.5 为 vanilla SFT(~1M) 后接 long-CoT SFT
  warmup 再 RL。
- **数据切分**：DeReason (arXiv 2603.11193) 表明“按推理难度解耦”优于随机拆分——
  广覆盖、非推理密集样本给 SFT（建立领域基础能力），难样本集中给 RL（培养复杂推理）。
  与本组规划天然吻合：Cauldron（广覆盖）→ SFT；MMFineReason-123K（最难 7%）→ RL 侧。
  RL 侧需规则可验证答案（geometry3k 有 verl 内置 `reward_score/geo3k.py`；
  MMFineReason 的 `answer` 为自由文本，需先筛出可规则判分子集或做答案归一化）。
- **warmup 停止时机**：不要用 SFT 训练 loss/分数当停止依据（ICLR 2026 Quagmires：
  高 SFT 分数与 RL 结果相关性差），用留出集泛化损失 + Pass@large-k 预测 RL 收益；
  同预算下“1 epoch 全量”不如“2 epoch 半量”（重复更稳）；只训短样本对 SFT 分数好但
  对 RL 有害。工程建议：1-2 epoch、固定步数（如 200-500 步）先做 warmup 消融，
  以 RL 学习曲线（而非 SFT loss）选停点。

## 6. 数据核实 + SFT warmup 切分建议（2026-08-19 本地实测）

### 6.1 Cauldron 16 个子集（组员选定的 SFT 池）——有 GT、无强模型 CoT

每行是**多轮** `texts`（每轮 `{user, assistant, source}`），assistant 即各源数据集
的**人类标注 GT 答案**（如 VQAv2 短答、ScienceQA 标签），**不是** 强模型生成的长 CoT。
行数 ≠ QA 对数（多轮行展开后是多个样本）。

| 子集 | 行数 | QA 对数 | 答案形式 | 规则可验证(RL) |
|---|---|---|---|---|
| vqav2 | 82,772 | 443,757 | 短答/是或否 | ✅ |
| visual7w | 14,366 | 69,817 | 选择题字母 | ✅ |
| aokvqa | 16,539 | 17,056 | 短答 | ✅ |
| tallyqa | 98,680 | 183,986 | 数字 | ✅ |
| vsr | 2,157 | 3,354 | 是/否 | ✅ |
| ai2d | 2,434 | 7,462 | 选择题字母 | ✅ |
| scienceqa | 4,976 | 6,149 | 选择题字母 | ✅ |
| tqa | 1,493 | 6,482 | 选择题字母 | ✅ |
| chartqa | 18,265 | 28,287 | 短答/是或否 | ✅ |
| textvqa | 21,953 | 34,602 | 短答(OCR) | ✅ |
| docvqa | 10,189 | 39,463 | 短答(OCR) | ✅ |
| infographic_vqa | 2,118 | 10,074 | 短答(OCR) | ✅ |
| iconqa | 27,307 | 29,841 | 选择题字母 | ✅ |
| raven | 42,000 | 42,000 | 字母 | ✅ |
| screen2words | 15,730 | 15,743 | 开放描述 | ❌ 需参考/LLM judge |
| textcaps | 21,953 | 21,953 | 开放描述 | ❌ 需参考/LLM judge |
| **合计** | **382,932** | **960,026** | | 14/16 闭式可验证 |

注意：组员给的 “VQA 110,410 / General 165,910” 与本地实测行数（214,514 / 168,418）和
QA 对数（717,970 / 242,056）都对不上，可能是计划采样数或另一口径，正式开工前需对齐。
转换器新增 `--max-per-subset N --seed S` 可做**按子集均衡采样**（分层、确定性、幂等），
避免 tallyqa/vqav2 这类大头子集淹没 raven/tqa 等小集。

### 6.2 MMFineReason-123K（122,603 行）——长 CoT + GT 双全，SFT-then-RL 的主池

- 强模型 CoT：`qwen3vl_235b_thinking_response`（`<think>` 包裹，单条可达 ~1 万字符）✅
- GT：`answer`（短答案，如 `65.12`）+ `original_answer`（详细推导）✅
- 全部 `pass_rate=0.0`（最难 7%，Qwen3-VL-4B 四次全失败的样本）、`is_consistent=True`
- `answer` 形态分布（全量统计）：单字母 21,102 (17%)、纯数字 33,261 (27%)、
  含数字 73,091 (60%)、长度 ≤12 的 87,011 (71%) → **规则判分覆盖大部分**；
  表达式/多部分答案（如 `(5,9)`、`4x+28.5=51.5`）需在 geo3k_reward 的 LaTeX 归一化 +
  mathruler 基础上扩展。
- source 分布：MMR1 59,775、FineVision-visualwebinstruct 23,509、GameQA-140K 15,231、
  BMMR 8,186、LLaVA-CoT 2,669、FineVision-raven 2,639、WaltonColdStart 2,433、
  Euclid30K 1,914 …（天然可按 source/难度分层切分）

### 6.3 结论：能不能“SFT 到底”，warmup 怎么切、停在哪

1. **SFT-then-RL 不建议把 40 万条全量 SFT 训满再进 RL**，但先区分两种设计：
   - **对比控制设计（推荐主臂）**：SFT-only 与 SFT-then-RL 的 SFT 阶段**同数据、同预算、
     同一 ckpt**（SFT-only = 该 ckpt 直接评估；SFT-then-RL = 该 ckpt 继续 RL），
     RL 增益才不被“SFT 数据量不同”混淆。此时 warmup 就是整个 SFT 阶段。
   - **warmup 消融（可选第 4 臂）**：复现 DeepSeek-R1/Kimi 式小 warmup——
     仅用 MMF 长 CoT 子集（如 20–50K，1 epoch）做格式锚定再 RL，验证“少 SFT + RL”
     是否足够；此臂与 SFT-only 的数据预算不同，属显式消融而非主对比。
2. **数据角色**：Cauldron（无 CoT、短答 GT）→ 只进 SFT warmup（任务广度/基础能力）；
   MMFineReason（长 CoT + GT）→ 切成两份：SFT warmup（长 CoT 锚定 `<think>` 格式）+
   RL（可验证答案）。切分用 `split_sft_rl.py` 按 image_hash+prompt 去重，SFT/RL、
   train/val 均不相交。
3. **规模参考**：DeepSeek-R1 冷启动仅数千条长 CoT；Kimi k1.5 = vanilla SFT(~1M) 后接
   long-CoT SFT warmup 再 RL；MMFineReason 论文证明 123K(7%) ≈ 全量 1.8M。
   对 4B/8B，建议 MMF warmup 20–50K（长 CoT）+ Cauldron 均衡采样
   （`--max-per-subset` 如 10–20K/子集，总量 150–250K），SFT 1 epoch。
4. **停止准则**（Quagmires, ICLR 2026, arXiv 2510.01624）：
   - 别用 SFT train loss/acc 选停点——高 SFT 分数与 RL 增益相关性差（R² 低至
     0.29–0.43），over-train 会压缩 RL 提升空间（Qwen3 在 2 epoch 后饱和/退化）。
   - 同预算“半量 × 2 epoch”通常优于“全量 × 1 epoch”；只训简单/短样本会让 SFT
     分数虚高但 RL 学不到东西。
   - 实用组合：① 固定预算（1 epoch / 300–600 步）为主；② 留出集 val 泛化损失
     flare-up 早停（同分布内有效）；③ 格式合规率（`<think>` 结构、语言一致）；
     ④ 候选 ckpt 用 Pass@large-k（k=16–64）预测 RL 收益；⑤ 最终以 RL 学习曲线
     （reward 上升、val reward 不塌）选点。
5. **RL 奖励**：geo3k_reward 的归一化 + mathruler 覆盖 MMF 表达式答案；按 source
   实现字母/数字/短答判分；Cauldron 闭式子集（14/16）可选混入 RL；
   screen2words/textcaps 不进 RL。

### 6.4 下一步

1. 对齐组员数据口径（110,410 / 165,910 的来源）。
2. `convert_cauldron_sft.py --max-per-subset 15000 --combine` 全量均衡转换；
   `convert_mmfinereason_sft.py --combine` 全量转换。
3. `split_sft_rl.py` 切 SFT warmup / RL（固定 seed）。
4. SFT→RL 衔接 smoke：`SFT_RL_MODEL=<sft-ckpt>/global_step_*/huggingface`
   `bash scripts/sft_rl/run_grpo_qwen3vl_geo3k_smoke.sh`。
5. 正式跑：Cauldron+MMF SFT（`save_contents=[model,hf_model]`）→ GRPO 从 SFT ckpt 续训。

### 6.5 转换状态（2026-08-19 实际执行）

- MMFineReason：18 个分片全部转换并 `--combine` 完成 →
  `$DTOPD_ROOT/fc-opd-storage/outputs/fc_opd/sft_rl/mmfinereason/mmfinereason_sft.parquet`
  （**122,603 行**，与源完全一致；列：messages/images/source/pass_rate/image_hash）。
- the_cauldron：**938/938 分片转换完成**（未合并，磁盘余量不足）→
  `.../sft_rl/cauldron/cauldron_sft__part_*.parquet`（159G，50 子集全覆盖；
  另有一批 7 子集验证残留 `cauldron_sft_partial__part_*`，可忽略）。
  - **verl 可直接吃 glob**（已实测，HF datasets 展开通配符）：
    `data.train_files="${...}/cauldron_sft__part_*.parquet"`（注意是双下划线，
    不会误吞 partial 残留）。避免 159G 全量合并撑爆共享 GPFS。
  - 等组内定下子集/采样上限后，用 `--subsets` + `--max-per-subset` 重新转换
    选定子集（幂等），或对 glob 结果做子集过滤再合出小文件。
- `--max-per-subset N --seed S` 已加入 `convert_cauldron_sft.py`（按子集分层、
  确定性采样，合成数据单测通过：30 行子集 cap=20 → 输出 20）。
- 磁盘：转换期间 GPFS 从 ~341G 可用降到 ~243G（共享盘，其他用户也在写）；
  正式 SFT/RL 训练时注意预留 ckpt（17.7G/份）与 HF 缓存空间。

## 7. answer null 处理 + 今晚 7 小时窗口执行方案（2026-08-19 深夜）

### 7.1 answer null 核实（本地全量统计）

- 122,603 行中 **answer=null 共 4,915 行**（0 空串）；`original_answer` **0 行缺失**；
  `qwen3vl_235b_thinking_response` 0 行缺失/空 → SFT 池不受影响。
- 分布：FineVision-visualwebinstruct(filtered) 4,483/23,509（19%）、LLaVA-CoT 192
  （7%）、MMR1 137（0.2%）、FineVision-ai2d_merged 98/175（56%）、BMMR 5。
- **处理策略**：
  - SFT warmup 池：全量 122,603（answer 与 SFT 无关，长 CoT 是全的）。
  - RL 池（规则判分）：`answer` 非空 117,688 行 → `mmfinereason_rl.parquet`。
  - RL 兜底变体：`--gt-policy answer_then_original` → `mmfinereason_rl_with_orig.parquet`
    （122,603 行，4,915 行 GT 用 original_answer 并在 extra_info.gt_source 标记；
    original_answer 多为长文，适合 judge 型奖励，规则判分建议只用 answer_only 池）。
  - 所有行 `<image>` 占位符与图片数一致（mismatch=0），可直接进 verl。

### 7.2 今晚产物（均已就绪，路径为绝对路径）

| 产物 | 路径 | 行数 |
|---|---|---:|
| MMF SFT 全量 | `fc-opd-storage/outputs/fc_opd/sft_rl/mmfinereason/mmfinereason_sft.parquet` | 122,603 |
| MMF RL（answer GT） | `.../mmfinereason/mmfinereason_rl.parquet` | 117,688 |
| MMF RL（含 original 兜底） | `.../mmfinereason/mmfinereason_rl_with_orig.parquet` | 122,603 |
| Cauldron 全量分片 | `.../cauldron/cauldron_sft__part_*.parquet` | 938 分片 |
| SFT warmup train/val | `.../warmup/sft_warmup_train__part_*.parquet`（24 分片，≤1500 行/片，verl SFT 需小分片避开 pyarrow 嵌套限制）/ `sft_warmup_val__part_0000.parquet` | 34,782 / 709 |
| MMF 规则奖励 | `scripts/sft_rl/mmf_reward.py`（10/10 单测通过：字母/数字/是或否/LaTeX） | - |

warmup 构成：MMF 长 CoT 分层 4,000（按 source）+ Cauldron 16 子集每子集 ~2,000（stratified，
seed 42）。train/val 按行 hash 切分不相交；`run_sft_warmup.sh` 自动把 24 个分片展开成
hydra list 传给 verl（已用 verl MultiTurnSFTDataset 在 CPU 实测加载 34,782 行通过；
24 分片 hydra `--cfg job` 干跑 EXIT=0）。行数 ÷ batch 32 ≈ 1.1K 步/epoch。

⚠️ 经验：verl SFT 用 pandas/pyarrow 读大 parquet（>~2K 行、images 大二进制）会踩
`ArrowNotImplementedError: nested chunked` 和 `offset overflow`；小分片 + hydra list
可绕开。**GRPO 大 RL 池同理**：不要用 `data.train_max_samples` 去 select 117K 行大文件
（会触发 take 的 offset overflow），先预采样一个小 RL parquet 再训练。

### 7.3 GPU 执行序列（用户在 GPU 节点粘贴；顺序即优先级）

```bash
cd "$DTOPD_ROOT/projects/Dual-Track-OPD"

# 0) 环境与路径 sanity（预期: 8×H200、NCCL_NVLS_ENABLE=0、文件存在）
nvidia-smi --query-gpu=name,memory.total --format=csv

# 1) SFT→RL 衔接 smoke（~30 分钟；用 11:48 已落盘的 HF ckpt 验证链路）
SFT_RL_MODEL="$DTOPD_ROOT/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_geo3k_smoke_20260819_1148/global_step_10/huggingface" \
  bash scripts/sft_rl/run_grpo_qwen3vl_geo3k_smoke.sh
#    预期: step:8 跑满、val reward >0.4、ckpt 落盘、运行段无 RayTaskError/OOM

# 2) SFT warmup（~1-3 小时；1 epoch，save_freq=150，HF 权重落盘）
bash scripts/sft_rl/run_sft_warmup.sh
#    预期: 首个 checkpoint 含 huggingface/model.safetensors；日志尾部 GCS/TCPStore
#          报错 = teardown 噪音；8 点回收前任何 step 数都是有效进展

# 3) GRPO 全量 geometry3k（~1.5-2.5 小时；时间允许才跑，从 warmup ckpt 续）
  SFT_RL_MODEL="$DTOPD_ROOT/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_warmup_*/global_step_*/huggingface" \
  bash scripts/sft_rl/run_grpo_overnight.sh
#    注意: SFT_RL_MODEL 的 glob 需展开成实际路径（先 ls 确认）
```

### 7.4 8 点实例回收后的续训

- 日志与 ckpt 全部在共享 NFS：`fc-opd-storage/logs/`、`fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/`。
- SFT 续训：`run_sft_warmup.sh` 已设 `resume_mode=auto`；重挂后固定
  `SFT_RL_NAME=<原实验名>` 重跑同一命令即从 `latest_checkpointed_iteration.txt` 续。
- 或直接用已落盘的 HF ckpt 重新初始化 GRPO（`SFT_RL_MODEL=<ckpt>/global_step_*/huggingface`）。
- 新实例记得：`NCCL_NVLS_ENABLE=0`、`trainer.use_v1=False`、HF offline 三件套。

### 7.5 18:20 事故：raven 双图占位符（已修）

- 现象：warmup 首跑在 DataLoader 崩 `AssertionError: image_offset 1 != len(images) 2`。
- 根因：the_cauldron `raven` 每行 2 张图（context + candidate），转换器只补了 1 个
  `<image>`；全 warmup 35,491 行中 1,006 行受影响（全来自 raven）。
- 修复：`scripts/sft_rl/fix_image_placeholders.py` 原地补齐占位符（1006 行，0 残留，
  已用 verl MultiTurnSFTDataset 在 CPU 实测 3 条 raven 双图样本处理通过）；
  `convert_cauldron_sft.py` 已改为按 images 数量补齐 `<image>`，后续全量转换不会再犯。
- 重启：warmup 直接重跑同一条命令即可（数据已修好）。

### 7.6 00:24 事故：docvqa 超大图超过 max_length（已修 + 数据重建）

- 现象：warmup 跑到 step 28 后前向崩
  `ValueError: Image features and image tokens do not match: tokens 13963, features 14355`。
- 根因：docvqa 高分辨率扫描图经动态分辨率展开后单图可达 16K+ 个 image token；
  `max_length=8192` 右截断了 input_ids 但没截 multi_modal_inputs → 前向断言失败
  （verl 的 no_padding 截断 bug）。
- 处置：
  1. `scan_oversized_samples.py --mismatch-only` 全量扫描（8 进程并行）定位
     `n_image_tokens != n_image_features` 的行 = 真实崩溃集；
  2. `filter_oversized_samples.py --mismatch-only` 只删这些行（本数据集 21 行，全 docvqa）；
  3. 为保险起见用修复后的转换器重建了 warmup（raven 占位符在源头修正），
     再过滤超大图，**复验 0 残留**；当前 train 34,764 / val 709。
- 经验：长 CoT 被截尾的行（668 条 MMF）图片 token 完好、**不会崩**，不要误删；
  判据必须是 `n_tok != n_feat`，而不是 `seqlen >= max_length`。
- 后续全量 Cauldron SFT（含 docvqa 等文档子集）会再遇到同样问题：届时对文档子集
  预处理降采样/设 processor max_pixels，或按上述判据过滤，二选一。

### 7.7 SFT warmup 详细训练配置（实录，实验 qwen3vl_sft_warmup_20260820_0148）

**运行环境**

- 实例：8×H200（本实验只用 0,1,2,3 四卡；4-7 留给其他进程）
- Env：`va-opd-qwen35-v090-cu132-r595-v1`（torch 2.13.0+cu132 / verl 0.9.0 / transformers 5.12）
- 关键 env：`NCCL_NVLS_ENABLE=0`、`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`、
  `FLASHINFER_WORKSPACE_BASE=$DTOPD_ROOT/.cache/flashinfer`、`CUDA_HOME=cuda132-toolchain`

**数据（prep_sft_warmup.py 产物）**

- train 34,764 行（24 个分片 ≤1500 行/片，hydra list 传入）+ val 709 行
- 构成：MMFineReason 长 CoT 分层 4,000（按 source）+ the_cauldron 16 子集每子集 ~2,000，
  seed 42；已修 raven 双图占位符、已滤 docvqa 超大图 21 行（mismatch 判据）
- 列：messages(list<role,content>) + images(list<struct<bytes,path>>) + source + image_hash

**SFT 训练参数（verl sft_trainer / FSDP2 / torchrun 4 rank）**

| 项 | 值 |
|---|---|
| 基座模型 | `$DTOPD_ROOT/models/Qwen3-VL-4B-Instruct` |
| data.train_batch_size | 32（动态 bsz 按 token 打包） |
| data.max_length | 8192 |
| data.pad_mode | no_padding |
| data.truncation | right（截尾仅文本；图片超限样本已预滤） |
| data.use_dynamic_bsz / max_token_len_per_gpu | True / 98304 |
| model.use_remove_padding | True |
| engine | fsdp，strategy=fsdp2，fsdp_size=-1，ulysses_sp=1 |
| optim.lr | 2e-5 |
| optim.lr_warmup_steps_ratio / warmup_style | 0.01 / cosine |
| optim.weight_decay / betas | 0.1 / [0.9,0.95] |
| optim.clip_grad / min_lr_ratio | 1.0 / 0.1 |
| trainer.total_epochs | 1（≈1,087 步） |
| trainer.test_freq / save_freq | 100 / 200 |
| trainer.max_ckpt_to_keep / resume_mode | 3 / auto |
| checkpoint.save_contents | [model, hf_model]（HF 权重可直喂 GRPO） |
| 启动方式 | `torchrun --standalone --nproc-per-node=4 -m verl.trainer.sft_trainer`（见 run_sft_warmup.sh） |

**实测指标（02:46，step 652）**

- 单步 ~5.4s（与重复任务 0154 抢卡时；独立后预计更快）
- 显存 40.6GB allocated / 81.1GB reserved 每卡（H200 141GB，余量充足）
- train/loss ~0.4-0.55 波动正常；val/loss 0.458→0.435（缓慢下降）
- checkpoint：global_step_200/400/600 均已落盘且含 huggingface/model.safetensors

**续跑 / 8B 切换**

- 续跑：固定 `SFT_RL_NAME=qwen3vl_sft_warmup_20260820_0148` 重跑同命令
  （resume_mode=auto 从 latest_checkpointed_iteration.txt 续）
- 8B：`SFT_RL_MODEL=$DTOPD_ROOT/models/Qwen3-VL-8B-Instruct`（同数据，从头训；
  显存/单步约翻倍，建议 batch 16、max_token_len_per_gpu 98304 不变）
- 完成后接 RL：`SFT_RL_MODEL=<final-ckpt>/global_step_*/huggingface bash scripts/sft_rl/run_grpo_overnight.sh`

**经验教训（本实验积累）**

1. verl SFT 大 parquet 必须小分片 + hydra list（pandas/pyarrow 嵌套 chunk 限制）；
2. `<image>` 占位符数必须 == images 数（raven 双图）；
3. `max_length` 截断 input_ids 但不动 multi_modal_inputs → 超大图必崩，预滤
   `n_image_tokens != n_image_features` 的行；
4. `setsid ... &` 终端显示 `[1]+ Done` 不代表失败（setsid 父进程立即退出），
   以日志/step 是否推进为准；不要重复前台再起一份。

### 7.8 SFT warmup 质量评估与 benchmark 方案（2026-08-20）

**原则**（Quagmires 结论）：SFT train/val loss 只说明拟合，不直接等于 RL 收益；
用 ① 留出集判分（规则可验证）+ ② Pass@k + ③ RL 学习曲线做 gate。

**Level 0 — geometry3k 官方 test（几分钟，首选）**

```bash
CUDA_VISIBLE_DEVICES=4 /inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/qwen3vl-cu128-vllm/bin/python \
  scripts/sft_rl/eval_geo3k.py \
  --model fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_warmup_20260820_0148/global_step_1086/huggingface \
  --data fc-opd-storage/outputs/fc_opd/sft_rl/geo3k_grpo_val_test.parquet \
  --out fc-opd-storage/outputs/fc_opd/sft_rl/eval/geo3k_sft_warmup.jsonl
```

- 输出：pass@1 准确率（601 题，geo3k_reward 判分）；`--rollouts 8` 可出 pass@8
  预测 RL 提升空间（Quagmires 指标）。
- 对照组：同命令 `--model models/Qwen3-VL-4B-Instruct` 跑 base；RL 后再跑一次，
  三臂对齐（SFT-only / RL-only / SFT-then-RL 同一 harness）。

**Level 1 — lmms-eval 子集（1-2 小时，与任务最相关 5 项）**

- MathVista（视觉数学，对应 MMFineReason）、GQA / VQAv2（对应 Cauldron VQA）、
  ScienceQA（科学推理）、MMMU-Pro（难多模态推理）。
- 组内已有 lmms-eval 栈：`envs/qwen3vl-eval`（lmms-eval）+ 32B judge；
  参照 `scripts/eval/run_vision_opd_baselines.sh` 的 vLLM serve + judge 模式，
  把模型路径换成 warmup ckpt 的 `huggingface/` 目录（该脚本当前写死
  Vision-OPD 路径，需要时改 `MODEL_ROOT`/ckpt 变量）。

**Level 2 — 完整 15 项套件（过夜）**

MathVista / MMBench / MMVet / GQA / ScienceQA / DynaMath / MMMU-Pro / MindCube /
MV-MATH / ReMI / VQAv2 / MathVerse / ViewSpatial / BLINK / MMSI（即
`docs/va_opd_lmms_eval_results.md` 的清单）。

**训练过程中持续判断（下一轮 8B / 正式跑）**

1. 每个 save_freq checkpoint（如每 200 步）跑 Level 0 的 100-200 题子集 →
   画 SFT→RL 前能力曲线；
2. 格式合规率（`<think>` 起止、answer 可提取率）作为 warmup 质量 sanity；
3. RL 首步 reward 对比 base：SFT ckpt 起步 reward 不应低于 base 起步
   （base smoke 记录 0.4375→0.5），否则说明 warmup 配方要调；
4. 最终以 RL 学习曲线（reward 上升 + val reward 不塌）选 SFT warmup 停点。

### 7.9 三臂对齐：数据切分 + 训练设置（2026-08-20，8B 为主模型）

**统一口径**：三臂（SFT-only / RL-only / SFT-then-RL）共用同一数据池、同一评估集、
同一 reward 定义；8B 全部从 `Qwen3-VL-8B-Instruct` 起步。不用 geometry3k 做 RL 数据。

**数据池**

| 池 | 内容 | 用途 |
|---|---|---|
| A = SFT 池 | MMFineReason 全量 122,603（长 CoT target）+ the_cauldron 16 子集（GT 短答 target） | SFT-only 全量 1 epoch；SFT-then-RL 的 warmup 从中采样 |
| B = RL 池（规则可验证） | MMF answer 非空 117,688 + Cauldron 14 闭式子集（vqav2/visual7w/aokvqa/tallyqa/vsr/ai2d/scienceqa/tqa/chartqa/textvqa/docvqa/infographic_vqa/iconqa/raven） | RL-only、SFT-then-RL 的 RL 段 |

**三臂用法**

- SFT-only：A 全量 1 epoch（≈队友口径 276K+122.6K；8B 时长按 batch/token 预算实测外推）。
- RL-only：B 从 8B-Instruct 直接 GRPO。注意 B 的 MMF 部分全是 pass_rate=0 最难样本，
  起步 reward 稀疏——首轮建议先采样 2-5K 题（或混入 Cauldron 闭式）再全量。
- SFT-then-RL：warmup 取 W ⊂ A（推荐 MMF 长 CoT 20K + Cauldron 16×5K ≈ 100K，1 epoch），
  RL 取 B \ W；**W 与 B 用 split_sft_rl.py 按 image_hash+prompt 去重，保证不相交**，
  train/val 固定 seed 切分。
- 各臂 epoch/步数预算对齐：SFT-only 与 SFT-then-RL 的 SFT 段尽量同配方（同数据同 epoch），
  RL 段统一 n=8。

**训练设置（8B / 8×H200）**

| 项 | SFT warmup | GRPO（MMF RL） |
|---|---|---|
| train_batch_size | 32（动态 bsz 软上限） | 16（长输出，16×8×~3K token/步） |
| max_length / max_response_length | 8192（整序列） | prompt 1024；**response 4096**（MMF 长 CoT；geometry3k 的 1024 不适用） |
| 打包 | use_dynamic_bsz=True + max_token_len_per_gpu=98304 + pad_mode=no_padding | rollout 不打包；actor 更新 dynamic bsz，max_token_len_per_gpu=65536~98304 |
| rollout.n | - | 8 |
| gpu_memory_utilization | - | 0.45-0.5 |
| optim | lr 2e-5, cosine, wd 0.1 | lr 1e-6, kl 0.01 |
| save_freq | 200 | 25 |

**需要和另外两臂确认的接口**：① 数据池范围（SFT-only 是否用全量 A）；② epoch/步数预算
（对齐总计算量）；③ 评估集（同一套：geometry3k test + lmms-eval 5 项）；④ reward 定义
（MMF 用 mmf_reward.py；Cauldron 闭式按 source 判分）。

**pack 概念（verl SFT 的对应物）**

- pack = 序列打包：把多个短样本拼进一条长序列一起算，避免 padding 浪费、提高吞吐。
- verl SFT 没有经典 pack 开关，等价物是 `use_dynamic_bsz=True` + `max_token_len_per_gpu`：
  每个 step 往 GPU 的 token 预算里塞样本（batch 只是软上限）。本次 warmup 用的就是它
  （4B 实测每卡 40.6GB alloc；8B 65536 预算 57GB alloc）。
- 多模态注意：图像 token 边界/attention mask 让“单序列拼接”很麻烦，verl 用
  pad_mode=no_padding + dynamic bsz 实现类 pack；RL rollout 逐样本生成不打包，
  actor 更新时用 dynamic bsz。

### 7.10 RL-GRPO / SFT-RL 配置定稿 + 今晚执行（2026-08-20 晚，飞书文档对齐）

**飞书文档已确认**：benchmark = MMMU-Pro / MathVerse / ViewSpatial-Bench / MMBench /
DynaMath / GQA（SFT 段另有 geometry3k test 601 已就绪）；SFT-only 段（32K packing、
4:3:3、617 step×32 packs、lr 5e-6、warmup 3%、seed 1234、2×GPU ZeRO-3）归 SFT 臂。
RL-GRPO / SFT-RL 两段空白 → 本节给出建议配置，供三臂对齐。

**8B warmup 已完成（可作 SFT-RL 起点）**：`qwen3vl_sft_warmup_20260820_0640`，
global_step_1086（1 epoch / 34,764 行，val/loss 0.3722，HF 权重 35GB 就绪）。
与文档 SFT-only 同数据池（A）但训练实现不同（verl sft_trainer vs 组员 pack trainer）：
三臂对比以“同一数据池 + 同一评测集 + 同一 reward”为准，实现差异如实记录。

**数据切分（三臂）**

| 池 | 内容 | 用途 |
|---|---|---|
| A = SFT 池 | MMF 122,603（长 CoT）+ Cauldron 16 子集 | SFT-only 全量 1 epoch；SFT-RL 的 warmup 采样 |
| B = RL 池 | MMF answer 非空 117,688 + Cauldron 14 闭式 | RL-only / SFT-RL 的 RL 段 |
| W（本次 warmup） | MMF 4,000 分层 + Cauldron 16×~2,000 = 34,764 | SFT-RL warmup（1 epoch，已完成） |
| R₀（今晚 RL 首跑） | `mmf_rl_3k/`：train 3,000 / val 150 | GRPO 首跑（sample_mmf_rl.py 产物） |

**MMF RL 池 GT 形态与奖励覆盖**（117,688 行）：letter 21,115 / yesno 922 /
pure_number 33,261 / has_number 39,830 / other 22,560。`mmf_reward.py` 覆盖前四类
（字母、是或否、数值、LaTeX+mathruler）；“other”（自由文本）暂不进 RL，
待 judge 型奖励再扩。`sample_mmf_rl.py` 还按 image_hash+prompt 剔除了与 W 重叠的
3,174 行，保证 warmup 样本不进 RL。

**长度证据**（Qwen3-VL-8B tokenizer 实测）

- 教师 CoT token 分位（20,436 条）：p50 3,233 / p75 5,980 / p90 9,739 / p95 11,903。
- → `max_response_length=8192` 覆盖 ~85% 教师长度；预算允许可升 12288（~96%）。
- 采样集 prompt token（3,150 条）：p95 402 / max 2,172 → `max_prompt_length=2048`，
  仅 3 行被过滤（1024 会滤 12 行，不要用）。

**RL-GRPO / SFT-RL 建议配置（8×H200，Qwen3-VL-8B-Instruct）**

| 项 | 值 |
|---|---|
| 算法 | GRPO（`algorithm.adv_estimator=grpo`，V0） |
| rollout.n | 8 |
| train_batch / ppo_mini_batch | 16 / 8 |
| max_prompt_length | 2048 |
| max_response_length | 8192（MMF 长 CoT；geometry3k 的 1024 不适用） |
| actor 动态 bsz token budget/GPU | 24576（首步后看 `max_memory_allocated_gb`，可升 32768） |
| lr / kl | 1e-6 / 0.01（low_var_kl），entropy_coeff=0 |
| reward | `mmf_reward.py`（file:// 加载；已修同目录 import：脚本内加 sys.path） |
| save_freq / test_freq | 25 / 10 |
| 总步数 | 3000/16 ≈ 188（1 epoch） |
| 启动 | `scripts/sft_rl/run_grpo_mmf.sh` |

**启动命令（GPU 节点，8 卡）**

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
SFT_RL_MODEL=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_warmup_20260820_0640/global_step_1086/huggingface \
  bash scripts/sft_rl/run_grpo_mmf.sh
```

可覆盖：`SFT_RL_GPUS`、`TRAIN_BATCH_SIZE`、`PPO_MINI_BATCH_SIZE`、`ROLLOUT_N`、
`MAX_RESPONSE_LENGTH`、`SFT_RL_ACTOR_TOKEN_BUDGET`、`SFT_RL_TRAIN/SFT_RL_VAL`、
`SFT_RL_REWARD`。首次启动建议先等 step:1 落盘（~2-5 分钟），确认
`critic/rewards/mean` 合理（MMF 稀疏奖励预期 0.2-0.5 起步）、无
`RayTaskError/OOM` 再离开。

**监控要点（日志 `fc-opd-storage/logs/qwen3vl_grpo_mmf_overnight_*.log`）**

1. `critic/rewards/mean`：GRPO 组内均值，随训练上升；`val-core/.../reward/mean@1`
   每 10 步跑 val 150 题。
2. `response_length/mean` + `clip_ratio`：clip_ratio > 0.4 说明 8192 上限偏低，
   下一轮升 12288。
3. `response/aborted_ratio` 应 ≈0；>0 说明 max_prompt_length/processor 问题。
4. `actor/perf/max_memory_allocated_gb`：>110GB 说明 token budget 24576 要降。
5. 每 25 步 ckpt（HF 35GB/份），5 份 ~180GB，磁盘 1.2T 可用，安全。

**Benchmark（文档口径）**：MMMU-Pro / MathVerse / ViewSpatial-Bench / MMBench /
DynaMath / GQA；SFT warmup 质量 gate 用 Level0 几何 test pass@1/pass@8 + 格式合规率，
最终以 RL 学习曲线选停点（见 7.8）。

### 7.11 事故：reward 返回 key 不一致导致 val 崩（2026-08-20 14:52，已修）

现象：`qwen3vl_grpo_mmf_overnight_20260820_1439` 启动后 ~12 分钟在 **首次 val
（_validate）** 崩 `KeyError: 'extracted'`，链路：
`fit → _validate → AgentLoopWorker._postprocess`：
`reward_extra_keys = list(reward_extra_infos[0].keys())`，随后对每个 key
`np.array([info[key] for info in ...])`。

根因：mmf_reward.py 的字母/是或否/数值分支返回 `{"score": ...}`（无 `extracted`），
符号/mathruler 分支返回带 `extracted`；同一 batch 内第一条样本 key 集合不同 →
后续样本缺 key 直接崩。geo3k_reward 的空 GT 分支同理（当时没触发）。

修复：所有返回路径统一 `{"score": ..., "extracted": ...}`（无抽取结果时
`extracted=""`）。已用 10/10 单测 + 混合 GT 批量 key 一致性检查验证。
**教训：verl 0.9.0 的 reward 函数所有 return 必须返回完全相同的 key 集合。**

### 7.12 事故：warmup ckpt 的 lm_head=F32 与 verl fused PPO 反向 dtype 冲突
（2026-08-20 15:06，已修）

现象：修掉 7.11 后重跑，val 正常（step:0 出分），首个训练步
`actor_rollout_ref_update_actor → loss.backward()` 崩：
`RuntimeError: expected mat1 and mat2 to have the same dtype, but got:
c10::BFloat16 != float`，位置 `verl/utils/experimental/torch_functional.py:60
_fused_linear_for_ppo_bwd`（`hidden_states @ vocab_weights.t()`）。

根因：SFT warmup 导出的 HF ckpt 里 `lm_head.weight` 是 **F32**
（`global_step_1086/huggingface/model.safetensors`，751 个 key；基座
`models/Qwen3-VL-8B-Instruct` 的 lm_head 是 BF16，config 都无 torch_dtype）。
verl `use_fused_kernels=True` 的 PPO 反向不做 dtype 转换，bf16 hidden @ fp32
lm_head 直接崩。非 fused 路径走 autocast，可正确处理。

修复：`run_grpo_mmf.sh` 默认 `actor_rollout_ref.model.use_fused_kernels=False`
（`SFT_RL_FUSED_KERNELS=1` 可开回）。**影响面**：任何从该 warmup ckpt 起跑的
8B GRPO（含 run_grpo_overnight.sh 复用的旧脚本）都必须关 fused kernels；
RL-only 从基座（lm_head=BF16）不受影响。后续可选：把 ckpt 的 lm_head 转成
BF16 再开回 fused（写 35GB，非今晚优先级）。

### 7.13 MMF GRPO 首跑实测（实验 qwen3vl_grpo_mmf_overnight_20260820_1511）

配置：8×H200、batch16/mini8、rollout.n=8、prompt 2048 / response 8192、
actor token budget 24576、lr 1e-6、kl 0.01、use_fused_kernels=False。

step:1 实测：

| 指标 | 值 |
|---|---|
| 每步耗时 | 160.6s（≈188 步/epoch → 全量约 8.4h；7h 窗口约 150 步） |
| 每步 token | 692,215（response mean 5,190 token） |
| throughput | 539 tok/s |
| 显存（actor） | 64.8GB allocated / 75.3GB reserved 每卡（H200 141GB，余量充足） |
| 训练 reward | mean 0.203 / max 1.0 / min 0.0（稀疏基线，正常） |
| response_length | mean 5,190 / max 8,192，**clip_ratio 0.42** |
| aborted_ratio | 0.0 |

结论与后续调整：

1. **clip_ratio 0.42 偏高**：42% rollout 顶到 8192 被截断，最终答案大概率被截掉
   → 这些样本 reward=0，信号变稀疏。下一轮可升 `MAX_RESPONSE_LENGTH=12288`
   （更慢，约 +30-50% 耗时）或对 RL 池按教师 CoT 长度做上限过滤；本轮先跑完。
2. 显存余量充足（~66GB 空闲），`SFT_RL_ACTOR_TOKEN_BUDGET` 可从 24576 升到
   32768 提速。
3. 1 epoch 约 8.4h：若实例 7h 回收，先跑到回收点，ckpt 每 25 步落盘，
   重挂后 resume（verl V0 resume 需要同 config + default_local_dir）。
4. reward 基线 0.203 合理（val 150 题中 MMR1 0.19、GameQA 0.39、BMMR 0.33、
   raven/VisualSphinx 1.0——小样本波动）。

### 7.14 目标 benchmark 评测准备：离线数据集缓存 + 评测脚本（2026-08-21）

**结论**：8B GRPO ckpt 184 已导出 HF 权重
（`fc-opd-storage/outputs/fc_opd/sft_rl/hf/qwen3vl_grpo_mmf_184/`，bf16 17.5GB），
评测直接用它。GPU 节点无外网，数据集已在有网节点预下载并缓存到
`~/.cache/huggingface`（共享盘）：hub 快照 18G + datasets arrow 缓存 18G。

**数据集（对应 configs/eval/project_vision_opd.yaml）**：

| 任务 | HF repo / config | 行数（离线实测） |
|---|---|---:|
| gqa | lmms-lab/GQA testdev_balanced_{instructions,images} | 12,578 |
| dynamath | kcz358/DynaMath (test) | 5,010 |
| viewspatial | oscarqjh/ViewSpatial_lmmseval (test) | 5,712 |
| mmmu_pro | MMMU/MMMU_Pro "standard (10 options)" (test) | 1,730 |
| mathverse | CaraJ/MathVerse-lmmseval testmini_version_split (vision_intensive) | 788 |
| mmbench | lmms-lab/MMBench en (dev) | 4,329 |
| remi | 本地 replay（qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl） | 2,600 |

**坑**：这些 repo 大多没有 dataset_infos.json，datasets 需在线解析一次把元数据
写进 `$HF_HOME/datasets` 才能离线 load；`hf download --include` 在 hub 1.20.1 会
忽略 include（只下到部分文件），统一改用 `snapshot_download(allow_patterns=...)`
（`scripts/sft_rl/download_bench_datasets.sh`）。

**评测命令（GPU 节点，无需网络）**：

```bash
# 冒烟（每项 8 条）
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
SFT_RL_SMOKE=1 setsid bash scripts/sft_rl/run_sftrl_benchmarks.sh \
  > fc-opd-storage/logs/sftrl_bench_smoke.out 2>&1 < /dev/null &
# 全量
setsid bash scripts/sft_rl/run_sftrl_benchmarks.sh \
  > fc-opd-storage/logs/sftrl_bench_full.out 2>&1 < /dev/null &
```

脚本内部：GPU0 vLLM serve 评测模型 :8000；GPU1 serve Qwen3-VL-32B-Instruct-FP8
judge :8001；`HF_HOME` 指向共享缓存 + `HF_HUB_OFFLINE=1`；先跑规则判分
（gqa/dynamath/viewspatial/mmmu_pro/remi），再跑 judge 任务（mmbench/mathverse），
最后自动出 summary。输出 `eval_runs/vision_opd_project_baseline/sftrl_grpo184_*/`。

## 6. 2026-08-22 状态与决策（会话中断后续）

### 中断点

- 08:37 `prep_sft_full.py` 完成：505,535 → 487,353 行（教师 CoT≤8192、整序列≤12288，
  超长丢弃不截断），train 485,367/324 shards、val 1,986/2 shards 落盘
  `fc-opd-storage/outputs/fc_opd/sft_rl/sft_full/`；图片外置 store 465,288 文件（47.4 GB）。
- 08:18 `run_sft_full.sh` 写好但从未启动；无任何 SFT/GRPO 训练进程残留。

### 冲突点结论：SFT/RL 池重叠不是问题（arXiv 2604.23747 §3）

用户指定参考论文 *SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning*
（arXiv 2604.23747，§3 Experimental Setup，PDF 第 4 页）：

> “We use the full dataset for both SFT (prompts paired with demonstrations) and
> RL (prompts paired with ground-truth answers for reward verification).”

即**规范用法 = 同一份数据集同时用于 SFT（配教师 CoT）与 RL（配 GT 答案做奖励验证），
完全不要求 SFT/RL 样本不相交**。论文全部对比方法共享同一 base model、同一 dataset、
同一评测协议；SFT→RL 就是“同一 prompt 集上先 SFT 后 GRPO（500 steps，rollout n=8）”。
因此本组 runbook §5 里 “W 与 B 必须不相交”（DeReason 式 split）是本组自定义设计，
不是通用规范。

本组实测重叠：`mmf_rl_3k`（3,150 行）与 `sft_full`（train+val）按 (source, question)
文本比对重叠 2,716 行 ≈ 86%；按 image bytes sha256 比对为 0%（RL 池字节与 sft_full
图片 store 编码不同，不代表内容不同）。

**决策：不做去重切分，跳过 `split_sft_rl.py`。** 保留 `sft_full` 全量（SFT 段输入），
RL 段继续用现成 `mmf_rl_3k`（3,000 train + 150 val，GT 规则可验证，`mmf_reward.py`
已通过单测，GRPO 链路 8/20 已实际跑通过）。如需更贴近论文“全数据集进 RL”，可后续把
RL 池从 3k 扩到 `mmfinereason/mmfinereason_rl.parquet` 过滤后的 91,954 行可验证池
（GRPO 预算按池大小重新评估，不可用 `train_max_samples` 直接 take 大文件）。

### 论文相关验证（SFT 质量保障）

- 论文两处框架 bug 对本组无影响或已修复：
  1. DeepSpeed CPU-offload 梯度 bug（§2.1）：本组 SFT 用 verl FSDP，不走 DeepSpeed，天然不受影响。
  2. loss 聚合 mean-of-means bug（§2.2，verl PR#3994）：verl 0.9.0 `sft_loss`
     （`verl/workers/utils/losses.py`）已按全局 token 归一化——`-masked_sum / batch_num_tokens * dp_size`，
     其中 `batch_num_tokens` 经 DP all-reduce SUM 聚合、FSDP 梯度平均后即真实 per-token mean，修复已包含。
- 评测协议按论文：temperature 0.6、max_response_length 8192、Math-Verify 验证。

### 2026-08-22 修复：sft_full 图片字段（曾为启动阻塞）

08:37 落盘的 sft_full shards 的 `images` 是 `[{"path": <相对路径>}]`：

- verl `MultiTurnSFTDataset → process_image → qwen_vl_utils.fetch_image` 只认
  `bytes` / `image_url` 键，`{"path": ...}` 直接 `KeyError: 'image_url'`；
- 且路径是相对 `$LZY_ROOT` 的，而 `run_sft_full.sh` 在 verl backend 的 example 目录
  启动 torchrun，相对路径解析不到图片。

已用 `scripts/sft_rl/fix_sft_full_image_urls.py` 就地重写全部 326 个 shard
（487,353 行不变）为 `images=[{"image_url": <绝对路径>}]`；并给
`prep_sft_full.py` 加 `Path(args.out_dir).resolve()` 兜底。验证：

- 326 shard / 487,353 行，schema 全部 `struct<image_url>`，无 .tmp 残留；
- `process_image` 抽样（train 5 + val 2 shards）全部 RGB 加载成功；
- CPU 端到端冒烟：`MultiTurnSFTDataset`（max_length=12288、pad_mode=no_padding、
  truncation=right）构建 + 取数 OK（input_ids/loss_mask/multi_modal_inputs/position_ids）。

### 下一步（GPU 节点）

```bash
# 1) 全量 SFT（8B，2 epoch，max_length 12288，lr 2e-5 cosine，save_freq 400）
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
bash scripts/sft_rl/run_sft_full.sh
# 日志 fc-opd-storage/logs/qwen3vl_sft_full_*.log；ckpt fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_full_*/

# 2) RL 段接 SFT ckpt（找到含 huggingface 的 global_step 目录）
SFT_RL_MODEL=<sft-ckpt>/global_step_*/huggingface \
  bash scripts/sft_rl/run_grpo_mmf.sh
```

### 7.8 2026-08-22 15:03 事故：sft_full 高分辨率文档图再次触发 image-token mismatch（已修）

#### 现象

- 首启 `run_sft_full.sh` 训练正常跑到 step 14/947（loss ~0.47），step 15 前向崩溃：
  `ValueError: Image features and image tokens do not match: tokens: 24561, features 28518`。
- 与 warmup 7.6 同一根因：verl no_padding 模式右截断 input_ids 但不截
  multi_modal_inputs，一旦截断点落在 `<|image_pad|>` 块内即前向断言失败。

#### 根因（sft_full 特有）

- 图片 token 数由**两级 smart_resize** 决定：
  1. verl `process_image → qwen_vl_utils.fetch_image`：factor=patch×merge=32，
     min=4×32²=4096、max=16384×32²=16777216；
  2. `Qwen2VLImageProcessor.preprocess`：min=65536、max=16777216（模型
     preprocessor_config：shortest_edge/longest_edge）。
- 文档子集（docvqa/textvqa/textcaps）高分辨率扫描图经两级展开后单图可达
  16K+ image token，整序列 >12288，截断落在 pad 块内 → mismatch。
- `prep_sft_full.py` 的长度过滤按单级估算，低估了这些图（真实网格以二级
  resize 为准），所以没有提前滤掉。

#### 处置（精确扫描 + 过滤，图片零解码）

- 新脚本 `scripts/sft_rl/scan_sft_full_exact.py`：按两级 smart_resize 公式算
  每图 pads（与真实 MultiTurnSFTDataset 输出在 8 个样本上逐一核对一致），再
  用字面 pad 串 + 真实 tokenizer 复现模板 token 数，mismatch 判据与生产前向
  完全一致。全程不碰 47GB 图片 store。
- 扫描结果：**81 行 bad**（docvqa 71 / textvqa 5 / textcaps 5），金标准抽查
  3 行（shard 0030）真实管线 n_tok=12284 vs n_feat=16280 确认必崩。
- `filter_oversized_samples.py --mismatch-only` 剔除 81 行 → train 485,286
  （324 shards，-81），val 1,986 不变；受影响 19 个 shard 复扫 **0 bad**。
- 数据 schema 仍为 `images=[{"image_url": <绝对路径>}]`，可直接重训。

#### 重启

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
bash scripts/sft_rl/run_sft_full.sh   # 新实验名自动时间戳，数据已干净
```

注意：08:33 之前那轮 PTD 的 `sft_full_oversized.json`（旧慢扫产物）不要用作
过滤依据，一律以 `sft_full_oversized_exact.json`（81 行）为准。

## 7.15 SFT-then-RL 定稿：RL 20K 无去重 + warmup 停止三判据（2026-08-25）

> 本章是对用户四问实现的最终落盘，也是对 §7.9「三臂对齐」的三处修正：
> (a) RL 数据规模 3K→20K；(b) 取消 warmup 去重；(c) 补 warmup 停止判据。

### (a) RL 数据规模 3K → 20K（多样性不够的修正）

- 原 `mmf_rl_3k` 只有 3,000 train / 150 val，用户判定多样性不足、巩固不充分。
  改为 **20,000 train / 500 val**，从规则可判池 `pool_after_filter=95,128` 按
  `source` 分层抽样（largest-remainder，seed 42），覆盖**全部 22 个 source**，
  分布与池一致（MMR1 11,019 / GameQA-140K 3,019 / visualwebinstruct 1,803 /
  BMMR 1,438 / 其余长尾），不丢任何一个领域。
- 产物：`fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_20k/`（train 14 shards +
  val 1 shard，≤1500 行/片）。`sample_mmf_rl.py` 默认 `--n-train 20000 --n-val 500`。

### (b) 无去重方案落实（"RL 无需考虑重叠"）

- **决策**：RL 段不再从 SFT warmup 池中剔除重叠样本——RL 复用同域任务是为了在
  warmup 已锚定的领域上**继续提升能力**，不是泄漏。论文 §3 也只把"重叠"当
  SFT 质量审计点，从未要求 RL 数据与 SFT 不相交；§4 的 RL 直接跑在 SFT 同源的
  OpenR1-Math 训练分布上。
- **代码落实**：`sample_mmf_rl.py` 的 `--warmup-dir` 从 `required` 改为默认关闭
  （`default=None`），不传即**不去重**。重生成的 20K 集 `dropped.warmup_overlap=0`
  （旧 3K 集曾因此丢弃 77,862 行，现在全部拿回）。仅保留 `gt_other=22,560` 的
  有意剔除（自由文本 GT 规则奖励不可判，只会在首批 RL 里注入 0 分噪音）。
- `run_grpo_mmf.sh` 的 `DEFAULT_TRAIN/VAL` 已指向 `mmf_rl_20k`，`PPO_MINI_BATCH_SIZE`
  默认 32（论文 §A 64 与我机 8 之间折中）。

> **2026-08-28 修正**：`PPO_MINI_BATCH_SIZE=32` 是错的，首跑即崩
> `AssertionError: 32 % 64 != 0`（`tensordict_utils.py:588 make_iterator`）。根因：
> verl 内部把 `ppo_mini_batch_size × rollout_n` 当「序列口径 mini-batch」传给
> `make_iterator`（`ray_trainer.py` `_update_actor`），32×8=64 序列；随后按
> `mini_batch_size % data_parallel_size` 均分到 4 卡，每卡 16 序列，但 actor 更新
> 走 `use_dynamic_bsz` 的 per-epoch 后 batch 被切成 32 序列——32 % 64 != 0。若真按
> 论文口径贴（rollout batch 128 / mini-batch 64），`PPO_MINI_BATCH_SIZE` 应填
> **8**（8×8=64 序列 = 论文 mini-batch 64，128 总批 = 2 个 mini-batch）。下文
> 「RL-GRPO 建议配置」的 train_batch/ppo_mini_batch 16/8 吻合。已改回默认 8。

### (c) warmup 停止三判据（非论文，本项项目自定质量 gate）

> 说明：论文**没有**显式的 SFT warmup 停止判据——它只给「3 epochs over 46k @ 4200
> tokens」（Appendix B）和最终 eval（avg@32 / pass@1，见 §3 评测段）。pass@1 一词
> 仅出现在评测那一处，指**评测口径**，不是 warmup 早停信号。所以下面三判据是
> 本项项目自己定的 gate，不是从论文摘录。

SFT train/val loss 只反映拟合，不等于 RL 收益（§7.8 坐下结论的延伸）。warmup 的
目标是**格式/长 CoT 锚定**，因此停点判据不 watch loss：

1. **格式合格率（首要）**：每个 save_freq ckpt 抽 MMF 长 CoT 子集，验证
   `thinking` 起止结构 + `answer` 可提取率。目标 **≥ 95%**；不达标向前取 ckpt，
   不继续跑到 3 epoch 结束。
2. **长 CoT 结构保持**：SFT 后启动输出长度不能塌缩（对齐教师长 CoT 分布，
   p95 ≈ 11,903 token，max_length 12288 覆盖 ~96%）。若响应明显变短说明被
   Cauldron 短 GT 带偏，立即回滚上一 ckpt 并重调 mix。
3. **pass@1/K 不退化**：MMF 规则可判子集上 pass@1 不应低于 base（warmup 是锚定，
   不应掉能力）。最终以 RL 学习曲线（reward 上升 + val reward 不塌）反向确认
   warmup 停点。

三判据任一不满足都**向前取中间 ckpt**（`max_ckpt_to_keep=3` 已留档），而不是盲目
跑满 3 epoch。

### 旧 ckpt 配置不符（务必不要复用）

- 抽查旧 ckpt `qwen3vl_sft_warmup_20260820_0640` 的 hydra 记录，确认是用**旧配置**
  训的：`lr=2e-5 / scheduler=constant / warmup_ratio=0.01 / wd=0.1 /
  betas=[0.9,0.95] / max_length=8192 / total_epochs=1 / seed=1`，数据是旧
  `warmup/`（34,764 行）。
- 这与本定稿（lr 5e-5 / cosine / 10% warmup / min_ratio 0.1 / wd 0.01 /
  betas[0.9,0.999] / max_length 12288 / 3 epoch，见 `run_sft_warmup.sh`）**全部不符**。
- **结论**：必须重跑 `run_sft_warmup.sh` 得到新 ckpt，再用其 huggingface 目录作
  RL 的 `SFT_RL_MODEL`。GPU 端可直接照 `/inspire/hdd/global_user/mengweicheng-
  240108120092/lzy/manuscript-sft-rl-gpu` 这份命令清单执行。

### 数据路径汇总表（本定稿，报告引用用）

| 用途 | 路径 | 规模 |
|---|---|---|
| SFT warmup 数据 | `fc-opd-storage/outputs/fc_opd/sft_rl/warmup_t2/` | train 153,073（MMF 长 CoT 98,003 + Cauldron 短 GT 55,070）/ val 3,123 |
| RL 数据（无去重） | `fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_20k/` | train 20,000 / val 500，pool 95,128，22 source |
| 旧 RL（废弃） | `.../mmf_rl_3k_t2/`、`.../mmf_rl_3k_t3/` | 3K，warmup_overlap 已删 77,862（t2）/未删（t3），均被 20K 取代 |

### 四问数据口径（报告素材，实测）

- **Q1 长/短 CoT 占比**（warmup_t2）：行数 MMF 98,003（64.0%）vs Cauldron 55,070
  （36.0%）；输出 token 口径 MMF 1,404M（99.9%）vs Cauldron 1.5M（0.1%）；平均
  输出字符/行 MMF 14,329 vs Cauldron 27。上一版 warmup 长 CoT 仅 ~13%，已修复。
- **Q4 max_response/lr/cosine**：SFT lr 5e-5 cosine（10% warmup, min_ratio 0.1）=
  论文 §A；RL lr 5e-6 **恒定**（无 cosine）= 论文 §4.3 短程方案（非 §A 的 1e-6）；
  两段独立 optimizer/scheduler。两段 max_length/response **均为 12288**（论文 §A
  8192），是有意偏差——MMF 长 CoT p95≈11,903，8192 会裁掉顶部 ~15%，破坏长 CoT
  锚定；RL 侧同用 12288 以压 clip_ratio。报告里需显式写明此偏差理由。

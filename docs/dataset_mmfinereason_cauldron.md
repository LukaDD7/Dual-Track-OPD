# 统一训练集获取说明：MMFineReason-123K + the_cauldron

日期：2026-08-19
目的：为 SFT / RL-only / SFT-then-RL / OPD-FKL 四路对比准备统一训练集。

## 当前状态（2026-08-19 已全部完成）

- MMFineReason-SFT-123K：✅ 完成。18/18 个 parquet 分片、122,603 行、7.19 GB（`data/` 目录，文件名与 HF 完全一致），parquet 可读校验通过。
- the_cauldron：✅ 完成。50 个子集、938 个 parquet、约 158 GB，无残留 `.incomplete` 文件，抽查 ai2d/chartqa/clevr_math/okvqa/scienceqa/tabmwp 均可读。
- 两个任务均从 ModelScope 下载（setsid 后台），进程已正常退出；日志见 `$DTOPD_ROOT/logs/download_*.log`。

## 数据集信息

| 数据集 | 链接 | 规模 | 下载体量 | 格式/许可 | 关键列 |
|---|---|---|---|---|---|
| MMFineReason-SFT-123K | https://huggingface.co/datasets/OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking | 122,603 条（18 个 parquet） | ~7.2 GB | parquet / apache-2.0，无 gating | `question`, `answer`, `ori_question`, `original_answer`, `image`, `caption`, `qwen3vl_235b_thinking_response`, `pass_rate`, `is_consistent`, `consistency_analysis`, `source`, `id` |
| the_cauldron | https://huggingface.co/datasets/HuggingFaceM4/the_cauldron | 50 子集，1,880,992 条 | ~169 GB（下载）；API 显示 dataset_size 456 GB 系 okvqa/clevr_math 元数据虚高 | parquet / 各子集原始许可 | `images`(序列), `texts`(每轮 `user`/`assistant`/`source`) 多轮聊天格式 |

## 存储位置

- MMFineReason：`$DTOPD_ROOT/dataset/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking/`（parquet 在 `data/` 下，与 HF 布局一致）
- the_cauldron：`$DTOPD_ROOT/dataset/the_cauldron/`（每子集一个目录，ModelScope 文件名带 hash 后缀）

选择 `lzy/dataset` 而非仓库内：仓库只放 manifest 与代码（AGENTS git 卫生要求），数据、权重、原始输出一律不入库；同一文件系统下两个位置无 IO 差异。

## 磁盘

- `/inspire/hdd/...` 为多用户共享 GPFS（10 T）：下载前可用约 730 G（93% 满），下载完成时约 462 G（96% 满）。期间约 140 G 额外占用来自其他用户/进程，非本任务造成。
- 本任务新增：the_cauldron ~158 G + MMFineReason ~7.2 G ≈ 165 G。
- 不要下载到 `/tmp`：overlay 临时盘，重启即丢。

## 下载方式与加速（实测）

本机实测速度：

| 源 | 单流 | 并行 |
|---|---|---|
| huggingface.co 直连 | ~0.65 MB/s | 8 路 ~3.7 MB/s |
| hf-mirror.com | ~0.29 MB/s | 不推荐 |
| ModelScope | ~9.2 MB/s（单流已最快） | 16 并发更快 |

已知坑：

1. `huggingface_hub` >= 1.26 已弃用 `hf_transfer`，默认走 Xet CDN；本集群到 Xet 的 `s3::get_range` 反复重试不可用。因此 MMFineReason 使用仓库脚本并行 Range 下载：
   （备用方案；实测 hf.co 直连约 0.65 MB/s、8 路并行约 3.7 MB/s，但长连接频繁挂起，不推荐主用）
2. 后台任务必须用 `setsid` 启动：本环境 `nohup` 起的子进程会随 exec 会话结束被清理（首次启动两个任务都因此中断过）。
3. **两个数据集的首选下载源都是 ModelScope**（文件与 HF 完全一致，已按大小核对）：
   - Cauldron：`modelscope download --repo-type dataset AI-ModelScope/the_cauldron --local-dir $DTOPD_ROOT/dataset/the_cauldron --max-workers 16`
   - MMFineReason-123K：`modelscope download --repo-type dataset OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking --local-dir $DTOPD_ROOT/dataset/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking --include 'data/*.parquet' --max-workers 16`
4. 实测速度：ModelScope 单流 ~9.2 MB/s、16 并发可达 ~90 MB/s；hf-mirror 反而最慢（~0.29 MB/s）。MMFineReason 在 ModelScope 上有官方镜像（之前误判只有 1.8M 原始组织镜像，实际 123K SFT 子集也在）。

下载环境：`$DTOPD_ROOT/envs/va-opd-qwen35-cu128`（已装 `hf_transfer`（弃用）与 `modelscope 1.39.1`）；`HF_HOME=$DTOPD_ROOT/.conda_cache/huggingface`。

## 对四种训练的适配判断

仓库历史训练设置（约束）：verl PPO/GRPO + parquet（`images` 内嵌 bytes）+ `image_key=images` + `max_response_length` 默认 512–1024 + 自定义 `compute_score`（Geometry3K 式选择题字母/数字抽取）+ FC-OPD 条件 `[full, degraded]` + GKD 纯蒸馏（forward_kl）；仓库内无现成 SFT 训练器配置。

| 训练方式 | 直接可用？ | 要点 |
|---|---|---|
| SFT | 转换后可用 | MMFineReason 几乎开箱（image+question+235B 长 CoT 为 target；建议 `is_consistent` 过滤；`max_response_length` 需提至 4096+）。Cauldron 结构现成但需转 Qwen3-VL chat 模板、处理多图/多轮 |
| RL-only | 不能直接用 | prompt 必须剥离 answer/CoT；verifier 需按 source 重写（现有 compute_score 仅覆盖选择题）；Cauldron 大量开放式任务规则奖励不可行，需 LLM judge/RM；MMFineReason-123K 全为 4B pass_rate=0 的最难样本，RL-only 起步 reward 稀疏，建议混入 586K 或易例 |
| SFT-then-RL | 转换后可用 | 两阶段复用同一数据，固定同一切分，RL prompt 不含 gold/CoT，val 不泄漏 |
| OPD/FKL（本 repo 管线） | 不能直接进 | 需适配 `FCOPDDataset` 的 `extra_info` 字段（question/choices/answer/condition_inputs）、每 source verifier、VA-OPD 需物化 degraded 图、prompt 契约禁字面 `<think>`；新数据集必须先跑 dataset signal audit（见 docs/fc_opd_dataset_selection.md），且 `task_evidence` 不得含 gold |

方向建议：MMFineReason 更契合 OPD/FKL（视觉推理+MCQ 密集+verified answer+teacher 输出），优先选其中 MCQ 密集 source（Geometry3K、MMR1、BMMR 等）做子集；Cauldron 只适合挑 chartqa/tabmwp/scienceqa/ai2d 等可验证子集。FKL 注意：仓库 GKD 是 online teacher logits 的 forward_kl，MMFineReason 的 235B 响应只能做 teacher-forced token CE / offline 蒸馏（对应 `support_transition_loss.prefix_fkl_ce`）。

## 后续步骤

1. 生成 manifest（`data/manifests/`）并记录 SHA256 清单（尚未计算）
2. 写 MMFineReason → 仓库 parquet schema 的 adapter（question/choices/answer/condition_inputs + 图片 bytes）
3. 按 source 实现 verifier，先跑 dataset signal audit 再进真训练
4. Cauldron 按需保留子集（可删除未选中的 config 释放 ~158 GB 中大部分空间）

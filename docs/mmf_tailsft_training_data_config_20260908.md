# MMF TailSFT 复现 — 训练配置与数据（7-A / 7-B 全景）

> 2026-09-08 整理。配套状态记录见
> `mmf_eval_status_openpyxl_bug_20260908.md`。所有数据/ckpt 落盘在
> `fc-opd-storage`（不进 git），本文档只记录路径、口径与来源脚本。

## 阶段总览

| 阶段 | 实验 | 脚本 | 数据 | ckpt |
|---|---|---|---|---|
| SFT | `qwen3vl_sft_tailsft_mmf122k_1ep` | `run_sft_tailsft_mmf.sh` | `mmf_only_sft_tailsft` | `ckpt/.../global_step_1774/huggingface` |
| RL (7-B) | `qwen3vl_gspo_mmf_tailsft_lr1e6_steps300` | `run_gspo_mmf_tailsft.sh` | `mmf_rl_20k` | 训练中 (save_freq=50) |
| 评测 (7-A) | base / tailsft 双臂 14-bench | `launch_mmf_eval.sh` + cfg JSON | 14 个 VLMEvalKit bench | 预测 xlsx |

## SFT 段（TailSFT 算法臂）

- **底座**: Qwen3-VL-8B-Instruct（同 mmf122k_1ep 基线臂）
- **数据**: `mmf_only_sft` 122,603 行 MFR 单源池
  （`prep_sft_single_dataset.py --dataset mmf`，过 length>12288 丢弃、
  确定性 seed 洗牌、val 2317 行）→ 实际训练 113,537 行 × 77/76 shards
- **init_ce 预标注**: `annotate_tailsft_init_ce.py`（base 模型逐样本
  ℓ0，off-line 一次性算，每 shard 追加 `init_ce` 列）→
  `mmf_only_sft_tailsft/`
- **核心超参**（对齐 mmf122k_1ep，唯一差异 = 在线序列级过滤）：
  - epochs=1, lr=5e-5 cosine（warmup 0.10）, wd=0.01
  - batch=64, max_len=12288, token budget=98304
  - **TailSFT 特有**：`data.tailsft.filter_fraction=0.5`（γ 目标过滤比）
    schedule=`ramp`, ramp_steps=800（≈总步数一半，γ 0→0.5）
  - verl 后端：`data.tailsft.enabled` 仅此臂为 True，默认 False 走原
    标准 SFT 路径（不动 run_sft_warmup.sh / 原标准 SFT 产物）
- **结果**: global_step_1774 = 1 epoch 完整收敛点；7-A tailsft 评测臂
  用的就是 `global_step_1774/huggingface` 导出的 HF 格式

## RL 段（GSPO，对齐 MMFineReason 论文 §B.1）

- **起点**: SFT 臂 ckpt `global_step_1774/huggingface`
- **数据**: `mmf_rl_20k`（seed 42 分层抽样 20K，val 500）
  （`sample_mmf_rl.py`；源池同 MFR 122,603，GT 类型过滤 letter/yesno/
  number，22,560 行 other 型被弃）
- **超参**（`run_gspo_mmf_tailsft.sh`）: loss_mode=gspo,
  loss_agg_mode=seq-mean-token-mean, lr=1e-6 constant, wd=0.1,
  warmup 10, 300 步, batch 16×G16=256 序列, prompt 8192 /
  response 16384, clip 3e-4/4e-4 + c=10.0, 无 KL, T=1.0
- **已知偏差**（相对论文）: RL 数据 20K 分层抽样（论文 40K 难度
  过滤，pass-rate 不可复现）；SFT 起点 1ep/122K（论文 3ep/1.8M）
- 事故与修复记录: 二次挂死（单进程 filter 8h 零进度 → 16 workers,
  commit 210fae8）；判分 openpyxl/et_xmlfile 缺包（手动补 et_xmlfile,
  commit 9ed62c7 文档）

## 评测段（7-A，14 bench 双臂）

- **bench 清单**（两臂同 cfg，仅 model_path 不同）:
  MMMU_DEV_VAL / MathVista_MINI / MathVision / MathVerse_MINI /
  DynaMath / LogicVista / VisuLogic / ScienceQA_VAL / RealWorldQA /
  MMBench_DEV_EN / MMStar / AI2D_TEST / CharXiv_descriptive_val /
  CharXiv_reasoning_val
- **base 臂 model**: `models/Qwen3-VL-8B-Instruct` —— **不是**任何 V1
  ckpt，是 Qwen3-VL-8B 原始 Instruct 权重（"base" 指 SFT 前原始底座，
  用于对照 tailsft SFT 增益）
- **tailsft 臂 model**: `ckpt/qwen3vl_sft_tailsft_mmf122k_1ep/
  global_step_1774/huggingface`
- 推理设置（两臂同）: vLLM, temperature=0, max_new_tokens=32768,
  max_pixels=4194304, repetition_penalty=1.05
- **judge**: Qwen3-VL-32B-Instruct vLLM :8801 GPU 3（Math 系 bench
  判分走它；Math-bench 打分阶段间隔期 judge log idle 属正常）
- cfg 文件: `fc-opd-storage/.../vlmeval_cfg/{qwen3vl_8b_base_mfr.json,
  qwen3vl_8b_tailsft_mfr.json}`（生成脚本 `launch_mmf_eval.sh`）
- 预测/判分产物: `vlmeval_runs/{arm}/T20260907-152150/`，per-bench
  `status.json` 记录 done/infer + error_message（MMMU/MathVista
  openpyxl 标记即在此）

## 跑速参考（H200 单卡, 2026-09-08 实测）

| bench | 题数 | infer 用时 | 备注 |
|---|---|---|---|
| MMMU_DEV_VAL | 1050 | 8.5h | 长 CoT, ~29s/it 均值 |
| MathVista_MINI | 1000 | 5.8min | 大量缓存命中（prefetch） |
| MathVision | 3040 | 52min | 10.2 it/s 高吞吐 |
| MathVerse_MINI | 3940 | 8min | 判分 +14min |
| DynaMath | 5010 | 7.5min | 0.41 avg / 0.16 worst |
| LogicVista | 447 | 40s | 40.04 acc |
| VisuLogic |  | ~70-120s/it 进行中 | 视觉逻辑题慢 |

两臂合计 14 bench × 2，judge 分时复用。15:21 起跑，预计 base 臂
1.5-2 天、tailsft 臂 2-3 天（MMMU 慢 bench 已过 75%）。

# Qwen3.5 Track A：120 步 instrumented overnight run 结果

> 日期：2026-08-04 ｜ 依据：`docs/qwen35_overnight_120_plan_20260804.md`（`ad55cf4`）
> 执行：`run_qwen35_v1_n4_overnight_sequence.sh`（canary → gate → 120 步 → 自动 group 汇总）

## 1. 结论

**Go（条件性）**：120 步自然完成（rc=0，墙钟 170 分钟），manifest/checkpoints/
rollout dumps 完整；validation accuracy 0→120 步从 10.5% 升到 12.0%（无退化、后半程
单调向上）；训练长度呈**有界振荡而非持续增长**（后 20 步所有 clip 指标都好于前 20 步）；
loss/grad/entropy 全程有限。按 plan 的 go 判据，下一步应为**相同配置的第二 seed**，
不立刻改超参；建议同时做 ckpt 0/60/120 的固定离线 eval 以确认增益。

## 2. 运行与产物

- launcher：`scripts/hpc/run_qwen35_v1_n4_overnight_sequence.sh`
- canary：5 步 rollout dump + 结构 gate **PASS**（5 steps × 24 seq × 6 groups × 4）
- 长跑 metadata：`fc-opd-storage/logs/qwen35_runs/k1_boxedonly_nonthinking_sampled_r4096_n4_overnight120_r1/`
  （`run_manifest.json` completed rc=0；`resolved_launch_config.json`；
  `tokenizer_alignment.json` PASS 248,070 IDs/0 mismatch；`train.log`；
  `train_rollouts/{1..120}.jsonl`；`group_clip_summary.json`）
- val dump：`fc-opd-storage/logs/val_dump_k1_boxedonly_nonthinking_sampled_r4096_n4_overnight120_r1/{0,20,40,60,80,100,120}.jsonl`
- checkpoints：`global_step_{30,60,90,120}` 全部存在
- 墙钟：**170.2 分钟**（≈85s/步）；repo `4436a94`（tracked_dirty=False）；
  后端 verl-cu130-vllm @ `334d9f8b`

## 3. 训练指标（120 步，n=4：6 prompts × 4 = 24 序列/步）

| 指标 | 值 |
|---|---|
| `critic/score/mean > 0` 的步数 | **80/120（67%）**，score mean 0–0.45 |
| `actor/pg_loss` 非零步数 | 55/120 |
| `actor/entropy` | 0.242–0.468（有限，无坍缩） |
| `actor/grad_norm` | 1.80–36.1（有限） |
| `distillation/loss` | 0.088–0.283（有限） |

## 4. Validation 轨迹（同 200 prompts，n=1，cap 4096）

| step | len_mean | len_p50 | clip | boxed | acc |
|---:|---:|---:|---:|---:|---:|
| 0 | 1549 | 355 | 0.270 | 0.745 | 0.105（21） |
| 20 | 1947 | 1468 | 0.280 | 0.715 | 0.095（19） |
| 40 | 1852 | 1073 | 0.260 | 0.715 | 0.105（21） |
| 60 | 2063 | 1515 | 0.295 | 0.710 | 0.100（20） |
| 80 | 1860 | 977 | 0.285 | 0.695 | 0.110（22） |
| 100 | 2023 | 1604 | 0.305 | 0.680 | 0.110（22） |
| 120 | 1805 | 957 | 0.280 | 0.715 | **0.120（24）** |

解读：clip 全程 26%–30.5% 无增长趋势；boxed 68%–74.5% 稳定；acc 后半程
（80-120）连续三次 ≥11%，step 120 为 12.0%（24/200），相对 step 0 的 10.5%
小幅上行——lr=1e-6 下属于温和但一致的学习信号，需第二 seed 确认。

## 5. 训练 rollout group-level clip（120 步，6 groups × 4）

| 指标 | 全程 mean | 前 20 步 mean | 后 20 步 mean |
|---|---:|---:|---:|
| sequence clip | 0.289 | 0.385 | **0.254** |
| group any-clip（≥1 截断的组） | 0.615 | 0.733 | **0.575** |
| group all-EOS | 0.385 | 0.267 | **0.425** |
| group all-clip | 0.053（max 0.50） | 0.053 | **0.042** |
| group any-correct | 0.175 | 0.158 | **0.233**（80/120 步非零） |

解读：**长度是有界振荡、后程改善**——group any-clip 从 73% 降到 58%，
all-EOS 从 27% 升到 43%，all-clip 组占比全程极低（均值 5.3%、后 20 步 4.2%）。
没有出现 plan 里担心的"最后 20 步 group all-clip 持续上升"。

## 6. 与 go/hold/no-go 对照

- run/manifest/checkpoints（30/60/90/120）完整：**PASS**
- validation accuracy 可信上升或至少不退化：**PASS（10.5%→12.0%，无退化）**
- 最后 20 步 group all-clip 未持续上升：**PASS（4.2%，且 any-clip 下降）**
- loss/grad/entropy 有限：**PASS**
- format reward 结构性为 0（无 think 标签），任务信号来自 accuracy：已知限制，
  报告中 acc reward 与 format reward 分开陈述。

**判定：Go** —— 下一步相同配置第二 seed；并行做 ckpt 0/60/120 固定离线 eval
（比较准确率与长度），不立刻改超参。若第二 seed 复现 acc 上行且长度稳定，
再冻结较长 seeded run。

## 7. 备注

- 全程未用 `CLEAN_START`；GPU 0-3（Track A），4-7 未受影响；Track B 未并行
  8-GPU full job。
- 主 checkout 因过夜 gate 需要保持 tracked 干净：诊断线 8 个未提交文件已快照为
  本地 commit `4436a94`（未改动内容），三个子模块内部改动 stash 保留
  （`wip-before-overnight`），均可恢复/由对方 rebase 处理。

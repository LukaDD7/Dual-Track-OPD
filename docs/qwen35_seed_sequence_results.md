# Qwen3.5 Track A：三 seed 120 步结果（长度与学习一致性）

> 日期：2026-08-06 ｜ 依据：`docs/qwen35_overnight_120_plan_20260804.md`（`ad55cf4`）
> 执行：seed1 = overnight120_r1（`ad55cf4` 序列）；seed2/seed3 =
> `run_qwen35_v1_n4_seed_sequence.sh`（`SEEDS="2 3"`，`data.seed`/`actor.data_loader_seed` 变化）

## 1. 结论

**Go 确认**：三 seed 的 120 步 run 全部 rc=0 自然完成（墙钟 150–170 分钟），
checkpoint/rollout dump/val dump/group 汇总完整；validation acc 三 seed 均无退化
（终值 10.0%–12.0%，相对各自 step 0 持平或小幅上行）；训练长度轮廓在三 seed 间
高度可复现（seq clip mean 0.275–0.289、group any-clip 0.593–0.615），且全部呈现
"前 20 步膨胀、后 20 步回落"的有界模式，无持续增长。按 plan go 判据，可以冻结
本配置（boxed_only + enable_thinking=False + cap 4096 + 显式 sampler + 6×4/8
workers）进入更长的 seeded run；建议同时做三 seed ckpt 0/60/120 的固定离线 eval
以量化学习增益。

## 2. 运行信息

| seed | manifest | rc | 墙钟 | 说明 |
|---|---:|---:|---:|---|
| seed1 | `…overnight120_r1` | 0 | 170.2 min | 首个 120 步（canary gate PASS 后） |
| seed2 | `…overnight120_seed2` | 0 | 154.6 min | `data.seed=2`/`actor.data_loader_seed=2` |
| seed3 | `…overnight120_seed3` | 0 | 150.3 min | `data.seed=3`/`actor.data_loader_seed=3` |

每 seed：120 步、val 0/20/…/120、ckpt 30/60/90/120、逐步 rollout dump
（`train_rollouts/{1..120}.jsonl`）、`group_clip_summary.json`。

## 3. Validation 轨迹（200 prompts，n=1，cap 4096）

**acc（正确数/200）**

| step | seed1 | seed2 | seed3 |
|---:|---:|---:|---:|
| 0 | 10.5%（21） | 9.5%（19） | 9.0%（18） |
| 20 | 9.5%（19） | 10.0%（20） | 10.5%（21） |
| 40 | 10.5%（21） | 12.0%（24） | 9.0%（18） |
| 60 | 10.0%（20） | 9.0%（18） | 9.0%（18） |
| 80 | 11.0%（22） | 12.5%（25） | 10.0%（20） |
| 100 | 11.0%（22） | 8.5%（17） | 10.0%（20） |
| 120 | **12.0%（24）** | **11.0%（22）** | **10.0%（20）** |

**clip / boxed（step 0 → 120）**

| seed | clip 0→120 | boxed 0→120 | len_mean 0→120 |
|---|---:|---:|---:|
| seed1 | 0.270→0.280 | 0.745→0.715 | 1549→1805 |
| seed2 | 0.215→0.225 | 0.800→0.770 | 1407→1535 |
| seed3 | 0.200→0.240 | 0.805→0.750 | 1332→1680 |

解读：三 seed 终值 acc 均 ≥ 各自 step 0（+0.5–1.5pp），无退化；clip 全程
20%–38% 有界，无持续增长；boxed 62%–80% 稳定。lr=1e-6 下增益温和，需固定
离线 eval 定量。

## 4. 训练 rollout group-level clip（120 步，6 groups × 4）

| 指标 | seed1 | seed2 | seed3 |
|---|---:|---:|---:|
| seq clip mean（first20→last20） | 0.289（0.385→0.254） | 0.275（0.375→0.254） | 0.278（0.381→0.256） |
| group any-clip mean（last20） | 0.615（0.575） | 0.606（0.567） | 0.593（0.583） |
| group all-EOS mean（last20） | 0.385（0.425） | 0.394（0.433） | 0.407（0.417） |
| group any-correct mean（非零步数） | 0.175（80/120） | 0.179（91/120） | 0.158（77/120） |

解读：**长度轮廓三 seed 高度可复现**，且都满足"后 20 步好于前 20 步"（seq clip
回落、all-EOS 上升、all-clip 组占比全程极低）；任务信号稳定到达（≥64% 的步有
≥1 正确 rollout 组）。没有出现长度失控。

## 5. 判定与下一步（供 codex）

- **Go**：三 seed 完整、无 val 退化、长度可复现且有界。冻结配置进入更长 seeded
  run；建议第一步做 ckpt 0/60/120 固定离线 eval（三 seed × 3 checkpoint）量化
  acc/boxed/clip 的变化，再定长跑步数与 seed 数。
- 不改超参；不试 8192；format reward 结构性为 0 的已知限制继续沿用（acc reward
  单独报告）。
- 本轮使用 GPU 0-3（Track A）；保活程序在 GPU 4+ 运行，无冲突。

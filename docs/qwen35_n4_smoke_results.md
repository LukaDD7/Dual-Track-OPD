# Qwen3.5 首个 n=4 trainability smoke 结果

> 日期：2026-08-04 ｜ 分支：`codex/va-opd` ｜ launcher：
> `scripts/hpc/run_qwen35_v1_n4_nonthinking_sampled_r4096_smoke.sh`（`f6f07d`）

## 1. 结论

**n=4 trainability smoke 通过**：20 步完整跑完（rc=0），无 OOM/Ray/teacher 错误，
任务奖励在 12/20 步非零、policy gradient 在 9 步非零，蒸馏/梯度/熵全部有限；
tokenizer alignment preflight PASS。观察到训练中段出现**可恢复的长度膨胀**
（val clip 26%→48%→21%），提示长跑必须持续监测 group-level clip。

## 2. 冻结配置（本次运行）

```text
TRAIN_BATCH_SIZE=6  PPO_MINI_BATCH_SIZE=6  ROLLOUT_N=4  ROLLOUT_NUM_WORKERS=8
NGPUS_PER_NODE=3  FORMAL_GPUS=0,1,2,3   USE_TASK_REWARDS=True
PROMPT_VERSION=boxed_only  enable_thinking=False  MAX_RESPONSE_LENGTH=4096
sampler: temperature=1.0 top_p=0.95 top_k=-1（训练与 validation 都显式覆盖）
TOTAL_TRAINING_STEPS=20  VAL_BEFORE_TRAIN=True  TEST_FREQ=5
```

有效 batch = 6 prompts × 4 = 24 序列；8 workers 整除 24。

## 3. 运行与产物

- metadata：`fc-opd-storage/logs/qwen35_runs/k1_boxedonly_nonthinking_sampled_r4096_n4_smoke/`
  （`run_manifest.json` completed rc=0；`resolved_launch_config.json`；
  `tokenizer_alignment.json` PASS 248,070 IDs/0 mismatch；
  `hydra/.hydra/config.yaml`；`train.log`）
- val dump：`fc-opd-storage/logs/val_dump_k1_boxedonly_nonthinking_sampled_r4096_n4_smoke/{0,5,10,15,20}.jsonl`
- 外层日志：`artifacts/fc_opd/nohup_v1_n4_nonthinking_sampled_r4096_smoke_20260804_121053.log`
- 墙钟：**34m56s**（2094.6s，≈105s/步）；repo `f6f07de`（tracked_dirty）；
  后端 verl-cu130-vllm @ `334d9f8b`

## 4. Trainability 证据

1. 任务奖励可达：`critic/score/mean > 0` 出现在 **12/20 步**
   （0.037–0.338，正确样本来自 n=4 采样）；`critic/advantages/max > 0` 9 步
   （最大 1.5）。
2. policy gradient 非零 9 步（step1 −0.0062，其余 +0.0013…+0.089）；
   `actor/entropy` 0.31–0.42、`actor/grad_norm` 1.64–55.7、
   `distillation/loss` 0.092–0.395，全部有限。
3. 训练序列级 clip：`response_length/clip_ratio` 20 步均值 **0.392**
   （0.042–0.708）。注意这是 sequence 口径；**prompt-group-level clip 本轮未落盘**
   （训练 rollout 未 dump），是下一轮必须补的指标。
4. format reward 结构性为 0（无 think 标签），acc reward 单独上报：
   val acc 全程 0.085–0.100，未见明显退化或过拟合。

## 5. Validation 轨迹（同 200 prompts，n=1，cap 4096）

| step | len mean | len p50 | clip | boxed | acc |
|---:|---:|---:|---:|---:|---:|
| 0 | 1593 | 529 | 0.260（52） | 0.760 | 0.095（19） |
| 5 | 2517 | 4095 | 0.480（96） | 0.505 | 0.085（17） |
| 10 | 2493 | 4093 | 0.490（98） | 0.510 | 0.090（18） |
| 15 | 2058 | 1655 | 0.305（61） | 0.700 | 0.100（20） |
| 20 | 1674 | 949 | 0.210（42） | 0.785 | 0.095（19） |

解读：

- **训练中段出现长度膨胀并恢复**：step 5/10 时 clip 升到 ~48%、boxed 掉到 ~50%，
  step 15 开始回落，step 20 恢复至 clip 21%（接近 D3 val 的 19%）、boxed 78.5%。
  这印证 codex 的提醒：D2/D3 只解决初始 rollout contract，训练中长度仍会波动，
  长跑必须逐 step 监测 clip/EOS。
- acc 20 步内基本持平（8.5%–10%），符合 lr=1e-6 的 trainability smoke 预期，
  不构成学习效果结论。

## 6. 通过条件核对（codex f6f07d 要求）

- tokenizer alignment preflight PASS，teacher 消费学生 token ID 长度/内容一致：**PASS**
- OPD loss / gradient / entropy/KL 有限，无 OOM/Ray/teacher 错误：**PASS**
- 每步 sequence 级 clip/EOS 上报：**PASS**（group 级待补）
- 正确/错误、EOS/clip 保留原始计数（val dump 五步全量 200 条）：**PASS**
- metadata、resolved Hydra config、git/backend/dataset hash 与 raw dump 路径完整：**PASS**
- 至少一条正确 rollout 的 group 比例：**本轮无训练 rollout dump，未直接测到**；
  由 val acc（~9-10%）与 sequence clip（~39%）间接支持，下一轮需显式落盘。

## 7. 下一步建议（供 codex）

1. `ecab13c` 已补 group-level rollout dump 与分析工具。先跑5-step instrumentation
   canary，自动确认每步6 groups × 4 rollouts，再开始独立冷启动的120-step run。
2. 今晚冻结本轮配置（4096 非思考 + 显式 sampler），每步落盘 rollout；validation
   每20步、checkpoint 每30步。完整执行见
   `docs/qwen35_overnight_120_plan_20260804.md`。
3. Track B 只可在剩余 GPU 做小 smoke，不在 Track A 结束前启动8-GPU full job。

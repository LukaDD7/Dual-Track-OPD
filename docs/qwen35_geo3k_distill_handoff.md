# qwen3.5/3.6 蒸馏：cu129 栈 + 正式实验交接（2026-08-02）

> ⚠️ 2026-08-03 更新：执行前先读
> `docs/qwen35_promptfix_review_for_claude.md`（commit 87225df）。本文 §4/§5 已按该
> review 修正：实验 #2 改用唯一实验名 + `RESUME_MODE=disable`（防续跑 #1 checkpoint）、
> 奖励分量按 0.9*acc + 0.1*format 口径、通过标准不要求 `pg_clipfrac` 非 0。

## 1. 现状一句话

**qwen3.6-27B → Qwen3.5-4B 蒸馏全链路已跑通**：cu129 栈（torch 2.11.0+cu129 / vllm
0.23.0+cu129）在 driver 570.124.06（最大 CUDA 12.8）上经 MVC 验证可用；smoke（4/4 步）与
正式实验 #1（79/79 步，1h24m）均完成。

## 2. 第一步已完成（正式实验 #1）

### 2.1 配置

| 项 | 值 |
|---|---|
| 学生 | Qwen3.5-4B（本地 `models/Qwen3.5-4B`）|
| 老师 | qwen3.6-27B（本地 `models/qwen3.6-27B`，64 层 GDN 混合 dense + 视觉编码器）|
| loss | k1 + use_policy_gradient=True + **use_task_rewards=False** |
| 资源 | 4 卡（3 actor + 1 teacher；另一实验当时占 GPU 4-7）|
| batch | 24（被 3 和 8 整除）；response 2048；79 步/epoch（1901 行，drop_last）|
| 时长 | 63.5s/it，共 1h24m |

### 2.2 结果（step 79）

| 指标 | 值 | 解读 |
|---|---|---|
| actor/entropy | 0.44 | 健康，无 collapse |
| distillation/abs_loss | 0.175 | 约等于自蒸馏 smoke（0.015）的 10 倍 → 容量差信号真实 |
| distillation/loss | 0.0805 | 正常量级，无 NaN |
| grad_norm | 1.65 | 正常 |
| rollout_probs_diff_mean | 0.0028（pearson 0.9997）| 学生紧跟 27B teacher |
| rollout_corr/kl | 0.00032 | 分布对齐良好 |
| reward / val acc | 0.0 | 见 §3：**任务奖励结构性为 0**（非仅因 use_task_rewards=False） |

### 2.3 产出

- checkpoint：`repos/verl-cu130-vllm/examples/on_policy_distillation_trainer/checkpoints/
  verl_distill_qwen35/qwen3_6_27b_to_qwen3_5_4b_k1_false/`（step 10–70, 79）
- ⚠️ 每个 checkpoint ~51GB（FSDP+optimizer 全量），8 个共 407GB。**正式实验前先规划磁盘**：
  建议 `SAVE_FREQ=20` 或定期清理旧 step。

## 3. 关键诊断：为什么任务奖励一直是 0（2026-08-03）

### 3.1 证据

- 正式实验 #1 的 79 步里 `critic/rewards/mean/max/min` 全部恒 0，val reward/acc 也恒 0；
  但 `timing_s/agent_loop/compute_score` 每步都有真实耗时（0.03–4.1s），日志里
  `RewardLoopWorker` 也在运行 → **奖励管线在算，只是结果恒 0**。
- verl 实际加载的数据（`datasets.load_dataset("parquet", ...)`）里
  `reward_model={'ground_truth': '3', 'style': 'rule'}`、`data_source='hiyouga/geometry3k'`
  都正确；geo3k 奖励函数本地验证：`<think>…</think>…\boxed{3}` → 1.0，无 boxed → 0。
- **根因**：rollout 用的是 `RLHFDataset`（启动日志 `Using dataset class: RLHFDataset`），
  它读到的 prompt 是 parquet 里的原始题目（`<image>\nFind x.`），**没有
  `Put the final answer in \boxed{}` 指令**。4B base 学生从不输出 `\boxed{}` → 每步奖励恒 0。
  项目里写好的 `FCOPDDataset`（会注入 boxed 指令，见
  `src/dual_track_opd/fc_opd/verl_dataset.py` + `prompt_contracts.py`）**没有被接线**——
  verl 需要 `+data.custom_cls.path/name` 才会用它。

### 3.2 结论

原计划的「直接开 `USE_TASK_REWARDS=True`」在奖励恒 0 时是**无效实验**：GRPO 全 0 优势 →
PPO 项恒 0，与 #1 数值上无差别。必须先让任务奖励**可达**。

## 4. 下一步（实验 #2：修 prompt 接线，只改一个变量）

### 4.1 调整的变量

`USE_FCOP_DATASET=1` —— 换用 `FCOPDDataset`，prompt 注入
`Put the final answer in \boxed{}`（学生/老师/loss 模式/batch/卡数/use_task_rewards 全不变）。
同时为诊断隔离加：唯一 `EXPERIMENT_NAME` + `RESUME_MODE=disable`（默认实验名
`..._k1_false` 与实验 #1 的 checkpoint 目录相同，`trainer.resume_mode` 默认 `auto`
会续跑 #1 的 step-79 checkpoint，导致对比无效）、`VAL_BEFORE_TRAIN=True`（step 0 就
dump）、`TOTAL_TRAINING_STEPS=20`（20 步 smoke，不是正式结果）、`SAVE_FREQ=-1`
（不落 ~51GB checkpoint）。

### 4.2 命令

```bash
FORMAL_GPUS=0,1,2,3 \
NGPUS_PER_NODE=3 \
TRAIN_BATCH_SIZE=24 \
PPO_MINI_BATCH_SIZE=24 \
USE_FCOP_DATASET=1 \
USE_TASK_REWARDS=False \
ROLLOUT_N=1 \
RESUME_MODE=disable \
VAL_BEFORE_TRAIN=True \
TOTAL_TRAINING_STEPS=20 \
SAVE_FREQ=-1 \
TEST_FREQ=5 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_taskfalse_fcop_n1_promptfix_smoke \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_promptfix_smoke \
bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/run_qwen35_formal.sh
```

（`VALIDATION_DATA_DIR` 会在 step 0/5/10/15/20 把 val 的 input/output/gts/score dump 下来，
直接看学生到底有没有输出 `\boxed{}`、奖励管线打没打对分。）

### 4.3 预期结果（对比 #1）

先厘清奖励口径（见 review §4）：`total = 0.9 * acc + 0.1 * format`；`format` 用
`re.fullmatch` 要求字面 `<think>...</think>...\boxed{...}`，所以「任意 boxed 输出至少
0.1」是**错的**：

| 输出 | acc | format | total |
|---|---:|---:|---:|
| 错误 + boxed + 无 think 标签 | 0 | 0 | 0.0 |
| 正确 + boxed + 无 think 标签 | 1 | 0 | 0.9 |
| 错误 + boxed + 有 think 标签 | 0 | 1 | 0.1 |
| 正确 + boxed + 有 think 标签 | 1 | 1 | 1.0 |

预期与判据：

1. step 0 起 dump 的 `input` 含 boxed 指令；`gts` 有值且与 parquet 一致。
2. `boxed_rate > 0`，且至少一个样本 total reward 非 0 —— 弱条件：prompt/reward 路径已通。
3. **强条件（跑实验 #3 的前提）**：`accuracy_reward_rate > 0` 且 reward 有非零方差。
   只出现常数 0.1 格式分只证明格式可达，不构成正确性判别。
4. 分别上报 `boxed_rate` / `think_boxed_format_rate` / `accuracy_reward_rate` / total 的
   mean·std·min·max / 0.0·0.1·0.9·1.0 计数 / `response_length/clip_ratio`（尽量按截断与否分层）。
5. `actor/distillation/loss` 量级与 #1 相近（任务奖励仍未参与训练，仅观测）；20 步 smoke
   **不要求** reward 单调上升。

若 20 步后 reward 仍恒 0：**不要直接放宽 `geo3k.format_reward`**。按顺序诊断：
运行时 dataset 类是否为 `FCOPDDataset`（非 `RLHFDataset`）→ dump 的 `input` 是否含注入指令
→ 是否大量触发 2048 token 截断（#1 的 clip_ratio ≈ 0.958）→ 输出语法分类（无 box / box 畸形 /
正确 box 无 think / think+box 完整）→ `gts` 是否为空 → 用后端 `geo3k.compute_score` 复算。
一次只改一个变量；真要改格式奖励须新命名并标注 ablation，共享 scorer 保持原样。

## 5. 再下一步（实验 #3：任务奖励生效）

在实验 #2 配置基础上只改一个变量：

```bash
FORMAL_GPUS=0,1,2,3 \
NGPUS_PER_NODE=3 \
TRAIN_BATCH_SIZE=24 \
PPO_MINI_BATCH_SIZE=24 \
USE_FCOP_DATASET=1 \
USE_TASK_REWARDS=True \
ROLLOUT_N=1 \
RESUME_MODE=disable \
VAL_BEFORE_TRAIN=True \
TOTAL_TRAINING_STEPS=20 \
SAVE_FREQ=-1 \
TEST_FREQ=5 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_tasktrue_fcop_n1_reward_smoke \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_reward_smoke \
bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/run_qwen35_formal.sh
```

前置条件（见 review §7.1）：实验 #2 已确认 custom prompt 路由正确、boxed_rate > 0、
存在 accuracy reward、reward 非常数、截断可接受。

预期（见 review §7.3）：`critic/score/*`、`critic/rewards/*` 反映非 0 任务分；
`critic/advantages/max` 非 0；`actor/pg_loss` 非 0；蒸馏指标有限；grad_norm 有限。
**不要**以 `actor/pg_clipfrac` 非 0 为通过标准——n=1 单样本组在 verl GRPO 里特判
mean=0/std=1，有效任务梯度存在时 clipfrac 仍可为 0。20 步 smoke 不要求 reward 单调上升，
学习结论需更长 seeded run。`ROLLOUT_N=1` 语义上是「无基线 REINFORCE 式」接线验证，
研究级任务-RL 对比需另跑 `ROLLOUT_N=4/8` 并重算显存。

若 reward 升但 KL/entropy 恶化 → 先看任务 pg_loss 与蒸馏 pg_loss 的量级对比、奖励的
acc/format 分量，再考虑调 `distillation_loss_coef` 或缩放任务奖励；`loss_max_clamp`
只裁剪逐 token 蒸馏估计器，不是通用稳定器，别盲降。

## 6. 后续实验线（每步一个变量）

- `LOSS_MODE=k3` / `LOSS_MODE=forward_kl_topk`（同 use_task_rewards 配置下对比 loss 模式）
- `STUDENT=/inspire/.../models/Qwen3.5-9B`（容量差 27B→9B）
- `TEACHER=/inspire/.../models/qwen3.6-35B-A3B`（MoE 老师，需 6+2 卡 + TP=2，
  batch 改 48/72/96/144）
- 卡空闲时恢复 8 卡：去掉 `FORMAL_GPUS`，`NGPUS_PER_NODE=7 TRAIN_BATCH_SIZE=56`

## 7. 环境与资源约束（重要）

- GPU 实例**同时跑多个实验**时，各实验必须用 `FORMAL_GPUS`/`CUDA_VISIBLE_DEVICES` 显式限卡，
  否则 Ray 会把 worker 排到被占的卡上（实测导致 vLLM `request_memory` 报
  `free < desired` 失败，并可能干扰对方实验）。
- 本次 GPU 0-3 空闲、4-7 被诊断实验占用；**下一步等 4 张卡空出**（或对方释放后直接用 8 卡）。
- batch 硬约束：`TRAIN_BATCH_SIZE` 必须同时被 `NGPUS_PER_NODE` 和
  `ROLLOUT_NUM_WORKERS`(默认 8) 整除。

## 8. 运行手册

- 一键正式实验：`bash scripts/run_qwen35_formal.sh`（含 teacher 预检 + GPU 占用检查 +
  batch 校验；`CLEAN_START=1` 可选清旧 Ray 集群）
- smoke 回归：`bash scripts/run_qwen35_gpu_smoke.sh`（8 卡自蒸馏 4 步）
- 环境重放（离线）：`bash scripts/setup_qwen35_cu129.sh`（wheelhouse 在
  `fc-opd-storage/wheelhouse/va-opd-qwen35-cu129/`，7.3G）
- 环境激活必设：`CUDA_HOME=cuda128-toolchain` + PATH/LIBRARY/LD_LIBRARY_PATH +
  `FLASHINFER_WORKSPACE_BASE` + `HF_HUB_OFFLINE=1`（见 environment_registry.md §10.3）

## 9. 诊断经验（踩坑记录）

1. **vLLM 日志默认写 stdout**，Ray 把它分流到 `worker-*.out`；调试 engine init 失败时
   `export VLLM_LOGGING_STREAM=ext://sys.stderr`，或同时看 `.err` 与 `.out`。
2. verl 的 vLLMHttpServer 已内置诊断钩子（`_dump_engine_diag`）：engine init 异常自动把
   traceback + Ray worker 日志尾部 + nvidia-smi 写到
   `fc-opd-storage/logs/vllm_engine_diag_*.log`（共享存储，可离线检查）。
3. "Engine core initialization failed. See root cause above" 的根因在 EngineCore 子进程
   输出里，用户终端看不到；用上面钩子或 Ray worker 日志定位。
4. 27B teacher 单卡需 `TEACHER_GPU_MEM_UTIL=0.55`（权重 ~54GB）；TP=1 可跑，显存不足时
   先确认卡上无残留占用，再考虑 TP=2（6+2 卡）。
5. `LLM.generate()` 在 vllm 0.23 不再接受 `max_tokens` 关键字，需
   `SamplingParams(max_tokens=...)`。
6. verl unpad 路径必须装 flash_attn（`flash_attn-2.8.3` cu128 wheel 已验证与
   torch 2.11.0+cu129 兼容）。
7. 日志尾部 `DataLoader worker killed by signal` 是训练完成后 teardown 的无害噪音。

## 10. 本次提交包含的改动

- `scripts/run_qwen35_formal.sh`（新增）：正式实验入口（预检/校验/参数开关）
  - `USE_FCOP_DATASET=1`：换 FCOPDDataset 注入 boxed 指令（新增）
  - `VALIDATION_DATA_DIR=<dir>`：dump val 生成/得分（新增）
- `scripts/run_qwen35_gpu_smoke.sh`（新增）：8 卡 smoke
- `scripts/setup_qwen35_cu129.sh`（新增）：cu129 栈离线切换
- `scripts/qwen35_vllm_preflight.py`（新增）：GPU 预检（支持 PREFLIGHT_* 覆盖）
- `scripts/build_vllm_022_cu128.sh`（新增）：回退方案 cu128 源码构建
- `patches/vllm-022-skip-deepgemm.patch`（新增）：cu128 源码构建跳过 DeepGEMM
- `patches/verl/qwen35_cu129_stack.patch`（新增）：verl-cu130-vllm 本地修改
  （`run_qwen3_5_4b_fsdp.sh` 的 num_workers/use_task_rewards 参数化 +
  `vllm_async_server.py` 诊断钩子）
- `docs/environment_registry.md`（更新）：§10 环境与实验记录
- 参考：verl-cu130-vllm @ `334d9f8b`（editable），vllm 0.23.0+cu129，
  torch 2.11.0+cu129，flash_attn 2.8.3

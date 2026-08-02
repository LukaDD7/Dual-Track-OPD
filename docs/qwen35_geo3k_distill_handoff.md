# qwen3.5/3.6 蒸馏：cu129 栈 + 正式实验交接（2026-08-02）

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
| reward / val acc | 0.0 | **预期内**：use_task_rewards=False |

### 2.3 产出

- checkpoint：`repos/verl-cu130-vllm/examples/on_policy_distillation_trainer/checkpoints/
  verl_distill_qwen35/qwen3_6_27b_to_qwen3_5_4b_k1_false/`（step 10–70, 79）
- ⚠️ 每个 checkpoint ~51GB（FSDP+optimizer 全量），8 个共 407GB。**正式实验前先规划磁盘**：
  建议 `SAVE_FREQ=20` 或定期清理旧 step。

## 3. 下一步（待卡空后跑）

### 3.1 调整的变量（一次只改这一个）

`USE_TASK_REWARDS=True` —— 打开蒸馏 loss 中的任务奖励项。其余全部不变（学生/老师/loss
模式/batch/卡数）。

### 3.2 命令

```bash
FORMAL_GPUS=0,1,2,3 \
NGPUS_PER_NODE=3 \
TRAIN_BATCH_SIZE=24 \
PPO_MINI_BATCH_SIZE=24 \
USE_TASK_REWARDS=True \
bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/run_qwen35_formal.sh
```

### 3.3 预期结果（对比 #1 判断依据）

1. `critic/rewards/mean`、`critic/rewards/max` **从 0 变为有信号**；val 的
   `hiyouga/geometry3k/reward/mean@1` 与 `acc/mean@1` 不再恒 0。
2. `actor/distillation/loss` 量级/走势变化：任务奖励会与蒸馏项耦合，loss 可能升高或波动更大，
   重点看 **grad_norm 是否稳定**、**reward 是否随训练上升**（期望：训练中 reward mean 上升）。
3. `response_length` 行为可能改变：模型在奖励信号下倾向停止在更优长度（clip_ratio 可能从
   0.958 下降）。
4. 判断标准：reward 上升 + 学生仍紧跟 teacher（rollout_probs_diff 不大幅恶化）＝耦合成功；
   reward 上升但 KL 爆炸/entropy 骤降＝需调 loss_max_clamp 或奖励权重。

### 3.4 后续实验线（每步一个变量）

- `LOSS_MODE=k3` / `LOSS_MODE=forward_kl_topk`（同 use_task_rewards 配置下对比 loss 模式）
- `STUDENT=/inspire/.../models/Qwen3.5-9B`（容量差 27B→9B）
- `TEACHER=/inspire/.../models/qwen3.6-35B-A3B`（MoE 老师，需 6+2 卡 + TP=2，
  batch 改 48/72/96/144）
- 卡空闲时恢复 8 卡：去掉 `FORMAL_GPUS`，`NGPUS_PER_NODE=7 TRAIN_BATCH_SIZE=56`

## 4. 环境与资源约束（重要）

- GPU 实例**同时跑多个实验**时，各实验必须用 `FORMAL_GPUS`/`CUDA_VISIBLE_DEVICES` 显式限卡，
  否则 Ray 会把 worker 排到被占的卡上（实测导致 vLLM `request_memory` 报
  `free < desired` 失败，并可能干扰对方实验）。
- 本次 GPU 0-3 空闲、4-7 被诊断实验占用；**下一步等 4 张卡空出**（或对方释放后直接用 8 卡）。
- batch 硬约束：`TRAIN_BATCH_SIZE` 必须同时被 `NGPUS_PER_NODE` 和
  `ROLLOUT_NUM_WORKERS`(默认 8) 整除。

## 5. 运行手册

- 一键正式实验：`bash scripts/run_qwen35_formal.sh`（含 teacher 预检 + GPU 占用检查 +
  batch 校验；`CLEAN_START=1` 可选清旧 Ray 集群）
- smoke 回归：`bash scripts/run_qwen35_gpu_smoke.sh`（8 卡自蒸馏 4 步）
- 环境重放（离线）：`bash scripts/setup_qwen35_cu129.sh`（wheelhouse 在
  `fc-opd-storage/wheelhouse/va-opd-qwen35-cu129/`，7.3G）
- 环境激活必设：`CUDA_HOME=cuda128-toolchain` + PATH/LIBRARY/LD_LIBRARY_PATH +
  `FLASHINFER_WORKSPACE_BASE` + `HF_HUB_OFFLINE=1`（见 environment_registry.md §10.3）

## 6. 诊断经验（踩坑记录）

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

## 7. 本次提交包含的改动

- `scripts/run_qwen35_formal.sh`（新增）：正式实验入口（预检/校验/参数开关）
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

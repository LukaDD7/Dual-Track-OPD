# VA-OPD Training Bug Tracker

## Environment

- **Hardware**: 4× NVIDIA H200 (141 GB), 8× H200 (queued)
- **Model**: Teacher Qwen3-VL-32B-Instruct, Student Qwen3-VL-4B-Instruct
- **Framework**: verl (PyTorch 2.8.0+cu128, FSDP, Ray, vLLM)
- **Dataset**: Geometry3K (2,101 prompts, 5 epochs → 1,751 steps)
- **Loss mode**: VA-OPD (reverse KL with rollout-level VA reweighting)

---

## Bug Summary

| # | 问题 | 原因 | 修复方案 | 可行性 | 未来风险 |
|---|---|---|---|---|---|
| 1 | **VA=0 everywhere** — 所有 token 的 Visual Advantage 恒为 0，训练完全无效 | `online_batch.py` 中三个函数 `_slice_teacher_topk`、`_pad_teacher_topk`、`_topk_to_device` 重建 `TeacherTopK` 时漏传了 `sampled_log_probs` 字段。该字段默认值为 `None`，丢失后下游 `verl_integration.py` 检测到 `None` 就填满 -30.0 → VA 恒为 0。触发条件：Qwen3-VL 32B teacher 和 4B student 的 tokenizer 不一致，每个样本都需要对齐，每次都丢。 | 三个函数各加一行 `sampled_log_probs=...`（含 None 保护）。验证：25 步测试 VA 值 0.076-0.296 恢复正常。 | ✅ 已修 | 低。`TeacherTopK` dataclass 新字段需在 `signal_decomposer.py` 显式声明默认值 `None`，以后新增字段需同步更新所有构造函数。 |
| 2 | **Background resume 失败** — `--background` + `--resume` 同时使用时，后台 relaunch 丢失 `--resume` 参数，从头训练 | Shell 脚本的 `while` 循环消费完 `$@` 后为空，`RELAUNCH_ARGS` 只迭代空的 `$@`，未显式检查 `RESUME_CKPT` 并添加 `--resume`。 | 在 `RELAUNCH_ARGS` 构建后显式追加：`if [[ -n "${RESUME_CKPT}" ]]; then RELAUNCH_ARGS+=(--resume "${RESUME_CKPT}"); fi`。同样处理 `--keepalive`。 | ✅ 已修 | 低。逻辑简单，回归测试覆盖。 |
| 3 | **NUM_STEPS 计算错误** — 硬编码 1313 步只有 3.75 epochs，论文要求 5 epochs | 公式假设 `TRAIN_BATCH_SIZE=8`，但实际被 GPU 数量限制为 6（`floor(8/3)×3=6`）。 | 自动计算：`NUM_STEPS = ceil(NUM_EPOCHS × DATASET_SIZE / TRAIN_BATCH_SIZE)` = `ceil(5 × 2101 / 6)` = 1751。当 `--steps` 未指定或为 0 时生效。 | ✅ 已修 | 低。如果改 GPU 数量或 ROLLOUT_N，TRAIN_BATCH_SIZE 自动变化，步数自动重新计算。 |
| 4 | **CPU 内存持续增长 → NCCL 超时** — 训练在 step 5-18 或 step 400+ 随机崩溃，NCCL allgather 超时 600s，SIGABRT | 三层根因：(a) `param_offload=true` — FSDP 把 ~16 GB 参数放 CPU，每步 CPU↔GPU 搬运产生 tensor 垃圾；(b) `optimizer_offload=true` — Adam 状态 ~64 GB 在 CPU，optimizer step 更新产生更多 CPU tensor 垃圾；(c) Python GC 惰性回收 — 训练循环中没有 `gc.collect()`，CPU 内存从 262 GB 涨到 352 GB（+90 GB / 400 步）→ OS swap → page fault 卡住 CUDA stream → NCCL watchdog 超时。 | 两步修复：(1) `param_offload=false` — 4B 模型只 ~8 GB，GPU 141 GB 足够，参数留 GPU，CPU 基线降 47 GB；(2) `optimizer_offload=false` — Adam 状态也留 GPU，CPU 基线再降 ~63 GB。最终 CPU 稳定在 ~227 GB 不再增长。 | ✅ 已修 | 中。如果换更大模型（>8B），GPU 可能放不下两套 offload 全关的优化器状态，需恢复 optimizer_offload。此时可配合 `gc.collect()` 每 N 步强制回收。 |
| 5 | **Checkpoint resume 计数器归零** — resume 后 global_step 从 1 开始而非 401，训练 1751 步而非 1351 步 | `default_local_dir` 设为了具体 step 目录（如 `.../global_step_400`），verl 的 `find_latest_ckpt_path` 在此目录内找 `global_step_*` 子目录 → 找不到 → 返回 None → "Training from scratch"。 | 检测 `RESUME_CKPT` 的 basename，如果以 `global_step_` 开头，取 `dirname` 作为 `default_local_dir`（父目录），verl 在其中正确找到 `global_step_400`。 | ✅ 已修 | 低。逻辑健壮，同时兼容传入父目录或具体 step 目录。 |
| 6 | **NCCL allgather 死锁** — `_ALLGATHER_BASE NumelIn=4198742` 在所有训练中重复出现，600s 超时 SIGABRT。与 offload 配置、硬件节点、GPU 数量均无关。 | 该 allgather 操作每次崩溃 NumelIn 完全相同（4198742），是 FSDP backward pass 中特定参数组的梯度同步。死锁特征：所有 rank 同时超时，表示所有 rank 都在等待某个 never-happen 的事件。可能原因：(a) 动态 bsz 导致某个 rank 的 CUDA kernel 执行时间异常长；(b) NCCL 2.27.3 与 CUDA 12.9 的版本兼容问题；(c) FSDP 的 allgather 通信模式与 NVLink 拓扑不匹配。 | 多层防御：(1) verl 原生 `nccl_timeout=1800` 替代 env var；(2) `forward_prefetch=true` — allgather 与计算重叠，减少同步点；(3) `NCCL_NVLS_ENABLE=1` — H200 NVLink Sharp 加速；(4) `NCCL_IBEXT_DISABLE=1` — 禁用 IB 扩展避免兼容问题；(5) `NCCL_BUFFSIZE=4194304` — 4MB 通信缓冲区减少碎片。 | ✅ 初步验证通过（step 478+ 无 crash） | 中。如果仍不稳定，备选方案：切换到 FSDP2 (`strategy=fsdp2`)，重新实现通信层，据论文更稳定且省 7% 显存。代价是不能加载 FSDP1 checkpoint，需从零训练。 |
| 7 | **Checkpoint 磁盘爆炸** — 每个 checkpoint 52 GB，无自动轮转，17 个旧 checkpoint + 子目录占用 1.3 TB | verl 每次保存完整模型权重 (bf16, ~16 GB) + Adam 动量 (fp32, ~32 GB) + Adam 方差 (fp32, ~32 GB) = ~80 GB 数据。FSDP 分片后 ~52 GB/checkpoint。`max_actor_ckpt_to_keep` 未设置（=null=无限）。异常：resume 时子 checkpoint 写入 step 目录内部（因为 default_local_dir 指向 step 目录），产生了额外 208 GB 嵌套文件。 | (1) 设置 `max_actor_ckpt_to_keep=5`（最多 5 个 checkpoint ≈ 260 GB）；(2) 手动删除旧 checkpoint 和嵌套 resume 文件，从 1.3 TB → 307 GB，释放 ~1 TB。 | ✅ 已修 | 低。自动轮转后最大 260 GB，磁盘 10 TB 有充分余量。 |
| 8 | **entropy 异常偏高** — resume 后 entropy=2.35（接近随机），前期成功 resume 时 entropy=0.002 | 尚不清楚。可能原因：FSDP checkpoint 跨节点加载时 optimizer 状态部分丢失；或不同节点 FSDP world_size/rank 映射差异导致部分参数未正确恢复。表现为模型权重已加载（VA 非零）但输出分布接近均匀。 | 待排查。目前训练仍在继续，观察 entropy 是否随时间下降。 | ⚠️ 待确认 | 高。如果 entropy 不下降，说明 400 步训练效果部分丢失。备选方案：不 resume，用 FSDP2 从零训练，1751 步完整跑完。 |

---

## Timeline

```
2026-07-06  发现 VA=0 → 定位 3 处 sampled_log_probs 丢弃 → 修复验证
            修复 background resume、NUM_STEPS 计算
            param_offload=true 在 3 GPU 下 CPU 262→352 GB → NCCL SIGABRT

2026-07-07  param_offload=false → CPU 215 GB 基线改善，但仍涨到 287 GB
            optimizer_offload=false → CPU 227 GB 稳定
            修复 default_local_dir → step 计数器正确恢复
            多次 NCCL deadlock (NumelIn=4198742) → forward_prefetch + NCCL env vars
            当前：step 478/1751 稳定运行中
```

---

## Current Config (Final)

```bash
# FSDP (match verl defaults)
actor_rollout_ref.actor.fsdp_config.param_offload=false
actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
actor_rollout_ref.actor.fsdp_config.forward_prefetch=true

# GPU memory
actor_rollout_ref.rollout.gpu_memory_utilization=0.45
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240

# NCCL stability
actor_rollout_ref.nccl_timeout=1800
export NCCL_NVLS_ENABLE=1
export NCCL_IBEXT_DISABLE=1
export NCCL_BUFFSIZE=4194304
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200

# Checkpoint
trainer.save_freq=25
trainer.max_actor_ckpt_to_keep=5
```

## Key Decisions

1. **param_offload 和 optimizer_offload 都关**：verl 框架默认值是 `false`（正确），Vision-OPD 用 8 GPU 才 override 到 `true`。我们 3 GPU 场景 offload 到 CPU 导致内存压力远超框架设计假设。4B 模型放 GPU 完全够。

2. **resume 时 default_local_dir 必须用父目录**：verl 的 `find_latest_ckpt_path` 在 `default_local_dir` 内扫描 `global_step_*` 子目录，不能把 step 目录本身设为目标。

3. **不追求 30min NCCL 超时作为合理解释**：死锁是真正的问题，不是偶发延迟。forward_prefetch + NCCL_NVLS 才是修复，超时只是安全网。

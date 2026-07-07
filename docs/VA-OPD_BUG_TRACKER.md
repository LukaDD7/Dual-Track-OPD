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
| 6 | **NCCL allgather 死锁（FSDP1 特有）** — `_ALLGATHER_BASE NumelIn=4198742` 在所有 FSDP1 训练中重复出现。1800s 超时仍 SIGABRT，与 offload 配置、硬件节点、GPU 数量均无关。 | **FSDP1 的 flat-parameter 机制**将整个 module group 的参数拼接成一个一维大张量，allgather 时一次传输 12,596,226 个元素（~25 MB bf16）。这个特定大小的 allgather 在 NCCL 2.27.3 + CUDA 12.9 上（可能结合 H200 NVLink 拓扑）触发死锁——某个 rank 的 NCCL 操作永远不完成，其他 rank 无限等待。注意 `reshard_after_forward=false` 并未修复，而是换了不同规模的 allgather 后产生 **rank 间操作类型不同步**的新死锁（rank 0 做 ALLGATHER，rank 1/2 做 ALLREDUCE），进一步证实 flat-parameter 的通信时序问题。所有 FSDP1 缓解措施（forward_prefetch、NCCL_NVLS、NCCL_BUFFSIZE）只是延长了崩溃前的存活步数（10→145→135 步），没有根治。 | **根本修复：切换到 FSDP2** (`strategy=fsdp2`)。FSDP2 使用 DTensor + per-parameter sharding，不 flatten 参数，allgather 粒度是单个参数而非巨型 buffer，通信模式完全不同。同时保持 `param_offload=false`、`optimizer_offload=false`、`forward_prefetch=true`。 | ✅ FSDP2 根治（8 GPU 1751 步无 NCCL 死锁） | 低。FSDP2 是 PyTorch 推荐的现代分布式策略，verl 也推荐使用。但 FSDP2 checkpoint 不兼容 FSDP1。 |
| 7 | **Checkpoint 磁盘爆炸** — 每个 checkpoint 52 GB，无自动轮转，17 个旧 checkpoint + 子目录占用 1.3 TB | verl 每次保存完整模型权重 (bf16, ~16 GB) + Adam 动量 (fp32, ~32 GB) + Adam 方差 (fp32, ~32 GB) = ~80 GB 数据。FSDP 分片后 ~52 GB/checkpoint。`max_actor_ckpt_to_keep` 未设置（=null=无限）。异常：resume 时子 checkpoint 写入 step 目录内部（因为 default_local_dir 指向 step 目录），产生了额外 208 GB 嵌套文件。 | (1) 设置 `max_actor_ckpt_to_keep=5`（最多 5 个 checkpoint ≈ 260 GB）；(2) 手动删除旧 checkpoint 和嵌套 resume 文件，从 1.3 TB → 307 GB，释放 ~1 TB。 | ✅ 已修 | 低。自动轮转后最大 260 GB，磁盘 10 TB 有充分余量。 |
| 8 | **entropy 异常偏高（跨节点 resume）** — FSDP1 resume 后 entropy=2.35，远高于正常值 0.002-0.02。模型权重已加载（VA 非零）但输出分布接近均匀。 | 跨节点 resume 时 FSDP checkpoint 的 optimizer 状态或 RNG 状态未完整恢复。表现为第 1 步 entropy 正常（0.39），随后持续上升到 2.39，说明 optimizer momentum 没有正确引导参数更新方向。后续 FSDP2 从零训练时 entropy 从 0.45 正常收敛到 0.005，确认是 checkpoint 恢复问题而非训练问题。 | FSDP2 从零训练，不依赖跨节点 resume。 | ✅ 已解（FSDP2 从零训练，entropy 正常收敛） | 低。以后如需跨节点 resume，应验证 optimizer state 与 RNG state 完整性。 |
| 9 | **FSDP2 跨 world_size resume 失败** — 4 GPU (3 训练卡) 保存的 FSDP2 checkpoint 无法在 8 GPU (6 训练卡) 上加载，FileNotFound: `model_world_size_6_rank_*.pt`。 | FSDP2 使用 DTensor，checkpoint 文件名编码了 world_size（如 `model_world_size_3_rank_0.pt`）。加载时 verl 期望 world_size 匹配，当前实现不支持自动 reshard。理论上 PyTorch DTensor 支持跨 world_size 的 `distributed_state_dict` 加载，但 verl 的 checkpoint 加载路径未实现此功能。 | 同 world_size 启动训练或从零开始。当前 8 GPU 从零训练。 | ⚠️ 绕过（从零训练） | 低。1751 步完整训练不需要 resume。未来可修改 verl checkpoint 加载代码支持 `full_state_dict` 模式。 |
| 10 | **Hydra 逗号解析冲突** — `teacher_urls=http://127.0.0.1:18080,http://127.0.0.1:18081` 被 Hydra 解释为列表而非字符串。 | Hydra override 语法中逗号是列表分隔符。双 Teacher 的 URL 列表需要用引号保护。 | 加单引号：`'+algorithm.fc_opd.teacher_urls=\\'${TEACHER_URLS}\\''` | ✅ 已修 | 低。Shell 引号转义需仔细。 |
| 11 | **Checkpoint 保存过频** — save_freq=25 时每 ~17 分钟写一次 52 GB checkpoint，1751 步共 ~70 个 checkpoint，IO 压力大、磁盘碎片多。 | 原始设置未考虑 52 GB/checkpoint 的存储成本。 | `save_freq=100`（~70 min/checkpoint，全训练 ~17 个），配合 `max_actor_ckpt_to_keep=5`，峰值磁盘占用 ~260 GB。 | ✅ 已修 | 低。 |
| 12 | **FSDP2 首次验证收敛异常** — 4 GPU FSDP2 从零训练 68 步时 entropy=0.008，但切换到 8 GPU 后从零训练的早期步（step 9-15）entropy=0.45-0.79，之后快速降至 0.005。 | 正常现象：训练初期模型输出分布波动大，entropy 在步间有噪声。8 GPU 的 TRAIN_BATCH_SIZE=6 与 4 GPU 一致，但 GPU 更多导致 FSDP shard 更小、allgather 更碎，可能影响早期梯度同步精度。但 30 步后 entropy 正常收敛，不影响最终结果。 | 无需处理，已自愈。 | ✅ 自愈 | 低。 |

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
            FSDP1 多轮修补治标不治本（step 410→464→545 陆续崩）
            
            🔑 根本修复：切换到 FSDP2 (strategy=fsdp2)
            4 GPU FSDP2 验证通过（68 步无 crash）
            修复 Hydra 逗号解析、teacher_urls 引号
            8 GPU 双 Teacher + FSDP2 1751 步训练中
```

---

## Current Config (Final — FSDP2)

```bash
# Strategy: FSDP2 (per-parameter sharding, no flat-parameter deadlock)
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.ref.strategy=fsdp2

# FSDP (match verl defaults — no CPU offload for 4B model on H200)
actor_rollout_ref.actor.fsdp_config.param_offload=false
actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
actor_rollout_ref.actor.fsdp_config.forward_prefetch=true

# GPU memory
actor_rollout_ref.rollout.gpu_memory_utilization=0.45
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240

# NCCL stability (defense in depth, though FSDP2 avoids the root cause)
actor_rollout_ref.nccl_timeout=1800
export NCCL_NVLS_ENABLE=1
export NCCL_IBEXT_DISABLE=1
export NCCL_BUFFSIZE=4194304

# Checkpoint
trainer.save_freq=100
trainer.max_actor_ckpt_to_keep=5

# Multi-teacher (when >=5 GPUs)
+algorithm.fc_opd.teacher_urls='http://127.0.0.1:18080,http://127.0.0.1:18081'
```

## Key Decisions

1. **param_offload 和 optimizer_offload 都关**：verl 框架默认值是 `false`（正确），Vision-OPD 用 8 GPU 才 override 到 `true`。我们 3 GPU 场景 offload 到 CPU 导致内存压力远超框架设计假设。4B 模型放 GPU 完全够。

2. **resume 时 default_local_dir 必须用父目录**：verl 的 `find_latest_ckpt_path` 在 `default_local_dir` 内扫描 `global_step_*` 子目录，不能把 step 目录本身设为目标。

3. **不追求 30min NCCL 超时作为合理解释**：死锁是真正的问题，不是偶发延迟。forward_prefetch + NCCL_NVLS 才是修复，超时只是安全网。

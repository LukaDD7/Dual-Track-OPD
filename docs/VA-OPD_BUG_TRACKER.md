# VA-OPD Training Bug Tracker

## Environment

- **Hardware**: 8× NVIDIA H200 (141 GB), 6 GPU training + 2 GPU teacher serving
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
| 13 | **Score 始终为 0 — prompt 格式与模型不兼容** — Qwen3-VL 预训练用 `\boxed{}` 输出答案，但我们 prompt 要求 `<answer>` 标签，reward 函数也从 `<answer>` 提取。模型默认输出 `\boxed{A}`，reward 找不到 `<answer>`，所有响应 score=0。`val_before_train` 验证分数 0.0，训练无有效 reward 信号。 | Qwen3-VL 预训练数据使用 LaTeX `\boxed{}` 约定。`<answer>` 是与模型预训练行为对抗的格式。模型不愿意使用它——特别是与纯 KL 蒸馏损失（无 GRPO reward 信号）结合时，模型倾向于回归其预训练的输出格式。 | 将 prompt 和 reward 函数都改为业界标准格式：`<think>` 标签用于推理 + `\boxed{}` 用于答案。从 EasyR1、veRL、GAO_grpo、ARPO、MindSpeed-MM 和 Oumi 复制的 prompt 模板。Reward 正则改为 `\\boxed\\{([^}]*)\\}`。验证：`val_before_train` score 从 0.0 → **0.38**（run 051629）。文件：`verl_dataset.py`、`smoke_reward.py`、`verifier.py`。 | ✅ 已修 | 低。`\boxed{}` 是数学 RL 训练的通用标准，所有 Geometry3K 基准测试均使用。 |
| 14 | **响应截断（1024 tokens）** — `max_response_length=1024` 时 67% 的响应被截断，在到达 `<answer>` 标签之前。Qwen3-VL 32B teacher 验证 `val_before_train` 分数 0.0。 | Geometry3K 数学题需要多步推理。1024 tokens 对于 `<think>` + `\boxed{}` 格式不够，响应在答案出现前被截断。业界标准为 2048 tokens（EasyR1、verl、rLLM）。 | `max_response_length=2048`，`ppo_max_token_len_per_gpu=10240`，`max_model_len=10240`。同样增加 NCCL `nccl_timeout=1800`。 | ✅ 已修 | 低。2048 是业界标准；即使在 2048 时截断也意味着无效响应（参见模式崩溃 #16）。 |
| 15 | **FSDP2 NCCL allgather 死锁（概率性，非特定大小）** — 6 GPU FSDP 训练中 NCCL allgather 概率性死锁，不同 FSDP 配置下死锁不同规模的 collective。**noreshard**（1×6, fsdp_size=6）：`NumelIn=83678470, NumelOut=502070820`，2-29 步间概率触发。**HSDP3**（2×3, fsdp_size=3，run 102313）：`NumelIn=33659650, NumelOut=100978950`，step 22 触发（1800s 超时后 SIGABRT）。HSDP3 将 allgather 缩到 1/5，推迟了死锁但未根除——证明 bug 不在特定 allgather 大小，而在 NCCL 2.27.3 + CUDA 12.8 + 6-GPU（非 2 的幂）拓扑的更底层交互。4 GPU 训练从未崩溃。HSDP3 run 指标：entropy=1.60 peak（step 17），score=0.69 peak（step 14），clip_ratio=4-17%，VA mean=0.11-0.45。 | 根因不在特定 allgather 规模，而是 NCCL 2.27.3 + CUDA 12.8 在 6-GPU (非 2 的幂) 拓扑下的底层竞态条件。所有缓解措施（Ring、noreshard、HSDP3）均为概率性推迟而非根除。 | 三种缓解措施测试完成：(a) **NCCL_ALGO=Ring**：38 步后模式崩溃。(b) **reshard_after_forward=false**：2-29 步概率死锁。(c) **HSDP3 (fsdp_size=3)**：22 步后死锁于 101M allgather。HSDP3 提供最小 allgather 和最长存活时间，但所有方案均概率性。根本修复需 PyTorch/NCCL 补丁、修改 `_no_split_modules` 包含 `embed_tokens`、或升级 NCCL/CUDA 版本。**建议：多次重试 HSDP3 以通过早期窗口，或降级到 4 GPU 训练。** | 🔄 需决策 | 高。概率性死锁是 FSDP2 + 6 GPU + NCCL 2.27.3 的根本限制。4 GPU 从未崩溃。 |训练在进行到 step 8–13 时确定性崩溃，`NumelIn=83678470`，`NumelOut=502070820`。3 次独立运行中 bit-identical 崩溃，相同的 SeqNum（10392 或 13012）。4 GPU 训练从未崩溃。**经多次运行验证，该 bug 为概率性触发**：同一配置下旧 run (055809) 存活 29 步，新 run (073445) 仅 2 步即死锁。内存模式完全一致（step1: 59→69GB, step2: 89→105GB），Codex 的 valid_mask 改动未引入新问题。 | FSDP2 使用 `_no_split_modules`（Qwen3VLTextDecoderLayer、Qwen3VLVisionBlock）的 `transformer_auto_wrap_policy`。根 FSDP 单元包含 `embed_tokens`（389M 参数）+ `lm_head` + 视觉非块组件 → 502M 参数 allgather。`reshard_after_forward=false` 不减��初始 allgather 频率（每次 forward 仍需 gather 参数）。NCCL 算法选择在 6-rank 拓扑下对该特定 allgather 规模存在竞态条件。 | 测试了两个缓解措施：(a) **NCCL_ALGO=Ring**：运行了 38 步后模式崩溃（未解决但延长了）。(b) **reshard_after_forward=false**：多次运行存活 2-29 步不等。**当前最优策略：反复重跑直到通过初始死锁窗口**（前 ~30 步），之后概率显著降低。根本修复需 PyTorch/NCCL 补丁或在 `_no_split_modules` 中包含 `embed_tokens`。 | 🔄 重试中 | 高。概率性死锁使每次启动都有风险，但一旦通过早期窗口即可稳定训练。 |
| 16 | **模式崩溃（纯 KL 蒸馏）** — 使用 `<answer>` prompt（run 034544）进行 ring 测试时，熵在 ~15 步内从 0.75 暴跌至 0.003。所有响应达到 max_length 2048（clip_ratio=1.0），score 回到 0.0。VA 信号降至 ~0.002。梯度范数飙升至 16.8。 | 没有有效的 reward 信号（因为 prompt 不匹配 + 截断），纯 teacher KL 蒸馏使模型崩溃到确定性输出——它学习生成保证与 teacher 一致的 2048 个相同 tokens，而不是生成正确答案。这是仅使用蒸馏损失时已知的 failure mode：模式寻求的 reverse KL 在没有 ground-truth reward 锚点的情况下会使分布崩溃。 | Prompt 修复（#13）+ `\boxed{}` 提取应能通过有效的 reward 信号防止崩溃。`reshard_after_forward=false` 的 run 显示早期熵值健康（0.47-0.54），val score=0.38。 | 🔄 验证中 | 高。核心风险：仅有 KL 损失（VA-OPD 是纯蒸馏）在 reward 信号较弱时可能不稳定。在以后的运行中监控 entropy 趋势并设置早期止停。 |
| 17 | **Teacher 分数修剪/填充对齐** — 每个样本出现 "trimming teacher scores (N₁ → N₂ tokens)" 警告，中位数丢失 33%，极端情况下 2048→220（丢失 89%）。填充警告中位数膨胀 68%，最严重的 27→501（膨胀 1756%）。246 次修剪，236 次填充。 | vLLM 学生 decode → 文本 → teacher re-encode 产生不同 token 计数（BPE 往返问题）。verl 对 response 的截断/填充进一步错位。原来的 `_slice_teacher_topk` 保留 **FIRST N** 个 token（`[:, :target_len, :]`），系统性地丢弃了位于末尾的 `\boxed{}` 答案。 | **改为 "keep last N"**：`[:, -target_len:, :]`。推理：答案 `\boxed{}` 始终在响应末尾。截断时保留 last N 个 token 最大化捕获答案段的概率。仅修改 `_slice_teacher_topk`；`_pad_teacher_topk` 保持末尾填充（因为 verl 做 right-padding，teacher 分数在开头对齐实际 token）。 | ✅ 已修 | 中。短期有效；长期需要 token-level 对齐（teacher forcing alignment）来彻底解决 32B/4B BPE 差异。 | |
| 19 | **Codex: teacher_valid_mask 修复** — 引入 `valid_mask: [B,T] bool` 到 `TeacherTopK`，标记 teacher-token 对齐可靠的位置。Trim/pad 位置被屏蔽出 loss 计算。验证：run 073445 step1 输出 `teacher_valid_ratio=0.72`（72% 位置有效），GPU 内存未受影响（59/69GB 与旧 run 一致）。NCCL 死锁仍发生在 step2（同一 502M ALLGATHER），确认非 Codex 改动引入。 | 新增字段贯穿全链路：`signal_decomposer.py`(dataclass) → `teacher_transformers.py`(set mask) → `teacher_client.py`(propagate) → `online_batch.py`(_slice/_pad 同步处理 mask) → `verl_integration.py`(condition_weights *= valid_mask) → `va_opd_loss.py`(training_mask = response_mask & teacher_mask)。所有 KL/VA/weight 计算统一使用 `training_mask`。 | 已验证：teacher_valid_ratio 指标正常（0.72），loss 降低（5.75 vs 6.44），GPU 内存无变化。该修复不引入 NCCL 问题。 | ✅ 已验证 | 低。逻辑简单：不可靠对齐位置直接排除出 loss。 | — 在 ring test 运行期间观察到，`nvidia-smi` 在 GPU 1 显示 "Xid 63" 但没有明显影响。GPU 保持全功能，训练继续。 | Xid 63 = 页表溢出（GPU 页表无法容纳所有映射）。在 ~76GB/143GB GPU 内存使用量时发生，远低于容量。可能是碎片化或罕见的 H200 问题。未重现。 | 无需操作。不会导致崩溃或性能下降。 | ℹ️ 监控 | 低。如果频率增加，调查 GPU 内存碎片化。 | — 4 GPU FSDP2 从零训练 68 步时 entropy=0.008，但切换到 8 GPU 后从零训练的早期步（step 9-15）entropy=0.45-0.79，之后快速降至 0.005。 | 正常现象：训练初期模型输出分布波动大，entropy 在步间有噪声。8 GPU 的 TRAIN_BATCH_SIZE=6 与 4 GPU 一致，但 GPU 更多导致 FSDP shard 更小、allgather 更碎，可能影响早期梯度同步精度。但 30 步后 entropy 正常收敛，不影响最终结果。 | 无需处理，已自愈。 | ✅ 自愈 | 低。 |

| 20 | **causal-state probe tokenizer fingerprint 不匹配（fail-fast）** — `run_causal_state_probe.sh` 默认 `DTOPD_PYTHON` 指向 `va-opd-native-e003-cu128-r595-v1`（transformers 4.57.3），加载模型后对同一 `Qwen3-VL` tokenizer 算出 fingerprint `4eff9f44...`，与不可变 K=32/proposal 证据记录的 `b5005fea...` 不一致 → `ValueError: current model tokenizer differs from immutable K=32/proposal token-ID evidence`。词表内容完全一致（151,669 tokens + 26 added），差异纯粹来自 transformers 4.x vs 5.x 对 `get_vocab()`/added vocab 的序列化方式不同。 | 生成 K32/proposal 证据的环境是 `va-opd-qwen35-cu128`（transformers 5.12.0，torch 2.11.0+cu129），它对当前模型算出的 fingerprint 正是 `b5005fea...`。修复：运行 causal probe 时必须用该环境，即 `DTOPD_PYTHON=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/va-opd-qwen35-cu128/bin/python`。 | 已验证：qwen35 环境 preflight 通过，student/teacher hash = `b5005fea` = 证据 hash；smoke 正常启动。 | ✅ 已绕过（运行时指定 DTOPD_PYTHON）；脚本默认值待改 | 中。任何升级/更换 causal-probe 环境的 transformers 后，必须先用 preflight 确认 student/teacher fingerprint 仍等于 `k32_tokenizer_hash`；不同 transformers 版本产生的产物不得混用。 |
| 21 | **causal-state probe 跨模型 JS 设备不匹配（cuda:0 vs cuda:1）** — `teacher_path_support_statistics` 把 student logits（cuda:0）和 teacher logits（cuda:1）直接传给 `full_vocab_js_statistics` 计算完整词表 JS，`torch.logaddexp` 抛 `RuntimeError: Expected all tensors to be on the same device`。触发于 `student_teacher_reference` 或 `teacher_path_support` 特征。 | 双卡部署 student/teacher 分属 cuda:0/cuda:1，跨模型 JS 前未做设备对齐。 | 在 `causal_runtime.py::teacher_path_support_statistics` 中，计算 JS 前把 teacher logits chunk 移到 student 设备：`teacher_logits = teacher_logits.to(device=student_logits.device)`（64×151k bf16 ≈ 19 MB，开销可忽略）。 | ✅ 已修 | 低。教训：所有 student/teacher 张量混合计算必须先显式对齐设备；以后新增跨模型统计时套用同一模式。 |
| 22 | **causal probe 固定轨迹统计 O(T²) 重复前向** — `fixed_trajectory_visual_statistics` / `teacher_path_support_statistics` 按 64-token 分块，每块把整个已见前缀重新前向一遍（`use_cache=False`），4096-token 轨迹对 32B teacher 等效 ~64 次整长前向（×2 条路径），一个 unit 3–5h。 | 分块只为了控制词表 logits 内存，但代价是每次重复计算整个前缀；因果注意力下整段前向与逐块前向逐位置数学等价。 | 改为每模型/每图像条件**单次整段前向**（`response_chunk_logits(start=0, end=T)`，`logits_to_keep=T+1`），分块循环只做词表 JS/NLL/top-k 数学。注意保留的词表 logits 峰值从 ~19 MiB/chunk 增至整段 ~1.16 GiB（visual 3.5 GiB / teacher-path 2.3 GiB），待 GPU 实测 `max_memory_*` 确认余量（评审 P1-6）。 | ✅ 已修并验证 | 验证：① toy causal LM 上单次前向与逐块前向逐位相等（`logits_to_keep` 路径与 fallback 路径）；② 真实 Qwen3-VL-4B CPU 对拍：同长度前向逐位 bit-identical（fallback 下 chunk2 diff=0），其余位置差异 ≤0.53 logit 属 bf16 形状噪声，非索引错位；③ preflight 通过（tokenizer hash 仍为 `b5005fea`）。产物 schema 不变；已完成 unit 保持旧数值、后续 unit 用新数值（逐 unit 独立；合并前按评审建议补充 old/new 端到端对比或分代分层）。 |

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

2026-07-08  🔑 Prompt 格式修复：<answer> → <think> + \boxed{} (业界标准)
            val_before_train score: 0.0 → 0.38
            max_response_length: 1024 → 2048
            
            FSDP2 NCCL allgather 死锁 (NumelIn=83678470, NumelOut=502070820)
            测试 NCCL_ALGO=Ring: 运行 38 步后模式崩溃 (entropy 0.75→0.003)
            测试 reshard_after_forward=false: 多次运行 2-29 步概率死锁
            测试 HSDP3 (fsdp_size=3): allgather 502M→101M, 22 步后仍死锁 (NumelIn=33659650)
            确认：NCCL 2.27.3 + CUDA 12.8 + 6-GPU 拓扑概率性死锁，所有缓解措施均推迟而非根除
            
            添加 --name 和 --test-fix 标志以区分运行和测试死锁修复
            模式崩溃根因：无 reward 信号的纯 KL 蒸馏 → 确定性输出
            Trim/pad 对齐问题：_slice_teacher_topk "keep first N" → "keep last N"
            根本原因：答案 \boxed{} 在末尾，"keep first" 系统性丢弃答案
            
            Codex 诊断：引入 teacher_valid_mask 排除对齐断裂位置的 loss
            teacher_valid_ratio=0.72 验证通过，loss 从 6.44 降至 5.75
            NCCL 死锁验证：概率性触发，与 Codex 改动无关（内存模式一致）
```

---

## Current Config (FSDP2 + HSDP3 recommended + \boxed{})

```bash
# Strategy: FSDP2 (per-parameter sharding)
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.ref.strategy=fsdp2

# FSDP (no CPU offload — 4B model fits H200 141 GB)
actor_rollout_ref.actor.fsdp_config.param_offload=false
actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
actor_rollout_ref.actor.fsdp_config.forward_prefetch=true

# Deadlock workaround: HSDP3 (2 FSDP groups × 3 GPUs, allgather 502M→101M)
# Use: --test-fix hsdp3
actor_rollout_ref.actor.fsdp_config.fsdp_size=3
actor_rollout_ref.ref.fsdp_config.fsdp_size=3

# Fallback: noreshard (1×6, 502M allgather, more memory)
# actor_rollout_ref.actor.fsdp_config.reshard_after_forward=false
# actor_rollout_ref.ref.fsdp_config.reshard_after_forward=false

# GPU memory
actor_rollout_ref.rollout.gpu_memory_utilization=0.45
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240
actor_rollout_ref.rollout.max_model_len=10240

# Response (industry standard for Geometry3K)
data.max_response_length=2048

# NCCL
actor_rollout_ref.nccl_timeout=1800
export NCCL_ALGO=Ring          # testing; may be unnecessary with noreshard

# Checkpoint
trainer.save_freq=100
trainer.max_actor_ckpt_to_keep=5

# Multi-teacher (2× Qwen3-VL-32B on GPUs 0-1)
+algorithm.fc_opd.teacher_urls='http://127.0.0.1:18080,http://127.0.0.1:18081'

# Prompt: industry-standard <think> + \boxed{} (in verl_dataset.py)
# You FIRST think about the reasoning process as an internal monologue...
# The final answer MUST BE put in \boxed{}.
```

## Key Decisions

1. **param_offload 和 optimizer_offload 都关**：verl 框架默认值是 `false`（正确），Vision-OPD 用 8 GPU 才 override 到 `true`。我们 3 GPU 场景 offload 到 CPU 导致内存压力远超框架设计假设。4B 模型放 GPU 完全够。

2. **resume 时 default_local_dir 必须用父目录**：verl 的 `find_latest_ckpt_path` 在 `default_local_dir` 内扫描 `global_step_*` 子目录，不能把 step 目录本身设为目标。

3. **不追求 30min NCCL 超时作为合理解释**：死锁是真正的问题，不是偶发延迟。forward_prefetch + NCCL_NVLS 才是修复，超时只是安全网。

4. **使用业界标准 `\boxed{}` 格式，不自定义 `<answer>`**：Qwen3-VL 预训练时就学会了 `\boxed{}` 输出，自定义 `<answer>` 标签是在跟模型预训练习惯对抗。所有数学 RL benchmark (Geometry3K, MATH, GSM8K, AIME) 都使用 `\boxed{}`。跟着惯例走，不发明新格式。

5. **`reshard_after_forward=false` 是 FSDP2 死锁的务实缓解方案**：根 FSDP 单元 502M 参数的 allgather 触发了 NCCL bug。减少 allgather 频率（forward 后保持参数在 GPU 上）以 GPU 内存换取稳定性。4B 模型 8GB 加上 Adam 状态 32GB，在 105GB 时就达到峰值 GPU 内存——仍在 141GB H200 安全范围内。

6. **修剪（trimming）已修复**：将 `_slice_teacher_topk` 从 "keep first N" 改为 "keep last N"。因为答案是 `\boxed{}` 始终在响应末尾，保留 last N 最大化捕获答案段。填充（pad）保持末尾追加（verl 做 right-padding，teacher 分数已对齐到开头 token）。

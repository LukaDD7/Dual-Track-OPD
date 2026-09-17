# qwen35 蒸馏 cu132 vs cu128 AB 对比方案（2026-08-16）

## 1. 目标

在 **同一 GPU 实例**（8×H200，Driver 595.58.03）上，用完全一致的训练配置对比两套栈
的 27B→4B on-policy 蒸馏性能，量化 vLLM 0.27 + torch 2.13 + verl 0.9.0 的提升。

| 侧 | 环境 | 栈 | 状态 |
|---|---|---|---|
| A（新） | `va-opd-qwen35-v090-cu132-r595-v1` | torch 2.13.0+cu132 / vLLM 0.27.1 / flashinfer 0.6.16.post3 / verl v0.9.0（**V0**） | `active`（smoke 3/3 步通过） |
| B（baseline） | `va-opd-qwen35-cu128` | torch 2.11.0+cu129 / vLLM 0.23.0+cu129 / flashinfer 0.6.12 / verl `334d9f8b`（V0） | `active`（旧生产栈） |

**AB 都用 V0 trainer**：旧 baseline 本来就是 V0；vLLM 0.27 与 vLLM 0.23 的 rollout 差异
才是本次对照的主体。V1 + delta_sharded 另立 candidate（见 §5）。

## 2. 配置对齐矩阵（一次一个变量）

两边必须逐项一致（除环境外）：

| 项 | 值 | 备注 |
|---|---|---|
| 数据 | `train_text_only.parquet` / `val_text_only.parquet` | 同文件同 hash |
| 蒸馏 loss | `k1`（正式对照；后续可切 `k3`/`forward_kl_topk` 做 loss 侧消融） | |
| use_task_rewards | `False` | 先跑稳 |
| use_policy_gradient | `True` | |
| STUDENT / TEACHER | Qwen3.5-4B / qwen3.6-27B | 本地路径 |
| batch / response | 56 / 2048（可调） | 每侧相同 |
| GPU 布局 | 同卡数（如 4 卡 3+1 或 8 卡 7+1） | 见 §3 顺序 |
| 注意力实现 | 两侧相同：`flash_attention_2`（推荐，两边都装了 FA 2.8.3）或 `sdpa`（对齐旧正式记录） | 旧正式脚本历史用 sdpa；新 formal 脚本 `ATTENTION_IMPL` 可切 |
| 随机性 | 两边各跑相同 seed 口径（verl 默认 seed=1） | |

**唯一允许的差异**：无。两侧**都设置 `NCCL_NVLS_ENABLE=0`**——本实例 NVLS
multicast 不可用（CUDA error 401，疑似 Fabric Manager/NVSwitch 配置）；旧栈
NCCL 2.28.9 在 Hopper 上同样默认开 NVLS，搬到本实例大概率同样报 401，所以
两侧统一关闭，保证对照只差环境栈。

NVLS 关闭的影响（记录在案）：只禁用 NVSwitch 在网归约，NVLink P2P 带宽不变；
对 FSDP2 all-gather 影响可忽略，对 reduce-scatter 大消息与 teacher 权重广播
（`update_weights`，实测 ~3-4s/step）影响有限（该段即使慢 50% 也仅 ~2s/step）。
smoke 实测 MFU 0.011-0.017，collectives 非瓶颈。可选量化：本机构建 nccl-tests，
`all_reduce_perf` 在 `NCCL_NVLS_ENABLE=0/1` 下各跑一次记录带宽差（1 需 FM 修复后）。

## 3. 执行协议

1. **同节点顺序执行**，避免并行实验互相抢资源（本实例有其它租户负载，串行更干净）。
2. 每侧正式运行前先跑同配置 4-step smoke 预热 JIT/编译缓存（flashinfer GDN、
   vLLM torch.compile），正式计时以第 2 个 run 为准。
3. 记录指标（每侧）：
   - `perf/time_per_step` 与 `perf/throughput`（tok/s）每步值；
   - `timing_s/agent_loop/generate_sequences`（rollout 吞吐）、`timing_s/update_actor`、
     `timing_s/update_weights`（27B teacher 权重同步段）、`timing_s/old_log_prob`；
   - loss 曲线：`distillation/abs_loss`、`distillation/loss`、`actor/entropy`、
     `training/rollout_probs_diff_mean`、`rollout_corr/kl`；
   - `actor/perf/max_memory_allocated_gb`（显存）、checkpoint 耗时 `save_checkpoint`；
   - 实验名/commit/manifest/数据 hash 写入 `RUN_METADATA_DIR` 与 registry。
4. 结果落 `fc-opd-storage/logs/qwen35_runs/`，并回写 `docs/environment_registry.md`。

## 4. 预期与判读

- 新栈预期收益点：vLLM 0.27 的 Qwen3.5 优化（RMSNorm+all-reduce 融合 #46998、
  MoE reduce-scatter #47006、KV cache 布局）、更快的 GDN prefill（FlashInfer 0.6.16
  + vLLM 0.27 集成）、flash-attn 2.8.3 直连 FA3。
- 风险：transformers 5.12.0 × vLLM 0.27 在长上下文下行为差异；`NCCL_NVLS_ENABLE=0`
  对 update_weights 段的潜在影响——若该段明显变慢，再单独跑 NVLS candidate。
- 结论门槛：time_per_step 提升 ≥10% 视为显著；不足则检查是否有单侧配置偏差。

## 5. 后续独立 candidate（不混入 AB）

1. **V1 + transfer_queue**：verl 0.9.0 V1 在本共享节点 transfer_queue 存储单元卡死
   （见 proposal §5.3），需先在平台侧排查（Ray actor 创建慢 + worker 非 OOM 死亡），
   或等 verl 上游修复后重试；跑通后单独测 `delta_sharded` 权重同步收益。
2. **NCCL 2.30**：作为单独 candidate，验证 NVLS 修复或 all-gather 带宽提升。
3. **FP8 teacher**：仅作明确 ablation（遵循 AGENTS.md，主线不删 online student scorer）。

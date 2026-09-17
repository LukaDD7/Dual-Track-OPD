# OPD Training Framework Migration Research

> 调研日期：2026-07-31
>
> 背景：当前自研 FSDP2 + verl v0.7.1 的 OPD 方案维护成本高（NCCL deadlock、ServerAdapter 兼容、RKL 实现 bug 等），需评估升级到官方 verl OPD 或切换到替代框架。

---

## 一、现状分析

### 1.1 当前 verl fork 状态

| 项目 | 详情 |
|------|------|
| 基础版本 | verl v0.7.1 (commit `bec9ef74`) |
| 上游落后 | **538 commits** behind `origin/main` |
| 自定义分支 | `codex/va-opd`，仅 **1 个 commit** 在 v0.7.1 之上 |
| 自定义变更 | `verl/workers/actor/dp_actor.py` (+415 lines)，`verl/trainer/ppo/ray_trainer.py` (+17 lines) |
| GKD recipe | `third_party/verl/recipe/gkd/`（实验性，Megatron-only，已因 ServerAdapter 架构问题放弃） |
| 环境 | cu128 自建 conda env (`vaopd-gkd-cu128`) |

自定义代码职责：
- **`dp_actor.py`**：在 actor forward 中检测 FC-OPD tensors → 调用 `compute_verl_sparse_*` 系列函数计算稀疏 KD loss → 注入 policy_loss
- **`ray_trainer.py`**：post-rollout hook 机制，在 batch 中注入 teacher logits / condition weights

### 1.2 已知问题清单

| 问题 | 状态 | 说明 |
|------|------|------|
| NCCL 2.27.3 NVLS deadlock | ✅ 已修 | `NCCL_NVLS_ENABLE=0` |
| Reverse-KL grad_norm=0 | ✅ 已修 | `pure_fc_opd` 集合漏了 `"reverse"` |
| RKL student 重归一化 bug | ❌ 未修 | `student_total = 1.0` → 归一化恒等映射 |
| Megatron 路径 ServerAdapter | ❌ 放弃 | vLLM 0.11+ 架构级不兼容 |
| 纯 distill 模式 patch 脆弱 | ⚠️ | 每次 verl 升级需手动维护 |

---

## 二、首选方案：升级 verl 到 v0.8+ 使用官方 OPD

### 2.1 verl 官方 OPD 概览

verl 在 **v0.8.0**（PR [#5041](https://github.com/verl-project/verl/commit/455e44c6)）中引入了**原生 On-Policy Distillation** 支持，覆盖：

| 维度 | 支持范围 |
|------|---------|
| 训练后端 | **FSDP1, FSDP2**, Megatron, VeOmni |
| 推理引擎 | **vLLM**, SGLang |
| 模态 | Text, **Vision-Language (Qwen3-VL)** |
| 教师数量 | 单教师 / **多教师**（按 data_source 路由） |
| 训练模式 | Sync, Fully-Async |
| 算法 | **GRPO** + distillation loss |

### 2.2 官方 OPD 支持的 Loss Mode

#### A. Forward KL（GKD 风格）— `loss_mode=forward_kl_topk`

```
use_policy_gradient=False  ← 直接 backprop（监督学习）
```

- 用 teacher 的 **top-K log-prob** 计算 $D_{KL}(P_{teacher} \| P_{student})$
- **直接 backprop** 通过 student，等价于 GKD
- 支持 `distillation_only` 模式（纯 distill，跳过 GRPO policy loss）
- `topk` 默认 64，可配 128
- 有 fused top-K kernel 优化（`use_chunked_topk`），支持长上下文
- 内置 overlap diagnostics（teacher/student top-K 交集比例）

#### B. Reverse KL（PG 风格）— `loss_mode=k1, k3, kl, abs, mse, k2, low_var_kl`

```
use_policy_gradient=True   ← RKL 估计值作为 dense reward
```

- 用**单样本蒙特卡洛估计** $D_{KL}(Q_{student} \| P_{teacher})$
- 负 RKL 估计值作为 **dense reward**，经 PPO clipping 更新
- `k1`：仅用采样 token 的 log-prob 差
- `k3`：用 top-3 student tokens 加权，方差更低
- 必须在 `use_policy_gradient=True` 下使用

#### C. 混合模式

```
use_task_rewards=True   ← distillation loss + task reward 混合
```

### 2.3 与我们最相关的官方示例脚本

```
examples/on_policy_distillation_trainer/
├── run_qwen3_8b_fsdp.sh            # 单教师 text (GSM8K+Math), FSDP
├── run_qwen3_5_4b_fsdp.sh          # 4B student + 35B teacher, FSDP2, Geometry3K ⭐
├── run_qwen3_vl_8b_fsdp.sh         # 单教师 VLM, FSDP
├── run_qwen3_8b_mopd_fsdp.sh       # 多教师 (text+VL), FSDP  ← 最接近 FC-OPD
├── run_qwen3_8b_megatron.sh        # Megatron 后端
└── run_qwen3_0.6b_opd_veomni.sh    # VeOmni 后端
```

**`run_qwen3_5_4b_fsdp.sh` 几乎就是我们的用例**：
- 4B 学生 + 35B 教师
- Geometry3K 数据
- FSDP2 配置：`actor_rollout_ref.actor.strategy=fsdp2`
- vLLM rollout with TP
- 开箱即用，只需改 `STUDENT_MODEL` / `TEACHER_MODEL`

### 2.4 官方 OPD 与我们的 FC-OPD 的差异

| 功能 | 官方 OPD | 我们的 FC-OPD |
|------|---------|--------------|
| Forward KL (GKD) | ✅ `forward_kl_topk` | ✅ `compute_verl_sparse_topk_kd` |
| Reverse KL (direct) | ❌ 只支持 PG 路径 | ✅ `compute_verl_sparse_reverse_kl` |
| Reverse KL (PG) | ✅ `k1, k3` 等 | ❌ |
| JSD | ❌ | ✅ `compute_verl_sparse_jsd` |
| 多条件路由 (full/degraded) | ❌ 只支持按 data_source | ✅ condition_weights + condition_ids |
| 条件敏感性分析 | ❌ | ✅ teacher-vs-student condition 对比 |
| Pure OPD (skip GRPO) | ✅ `distillation_only` | ✅ `pure_fc_opd` |
| VGPO (visual grounding) | ❌ | ✅ advantage modulation |
| Post-rollout hook | ❌ 用 teacher_loop | ✅ 自定义 hook |
| Top-K overlap 诊断 | ✅ | ❌ |

### 2.5 升级路径分析

#### 选项 A：Rebase 到 v0.8.0 + 适配 FC-OPD（推荐）

```
当前 codex/va-opd (1 commit on v0.7.1)
    ↓ rebase
verl v0.8.0 (or origin/main)
    ↓ 适配
FC-OPD 作为 custom loss_mode 注册到 DISTILLATION_LOSS_REGISTRY
```

**优势**：
- 自动获得官方 OPD 的所有基础设施（teacher_loop, vLLM 兼容, NCCL 修复, FSDP2 优化）
- FC-OPD 差异化为自定义 `loss_mode`，代码量小
- 可以利用官方的 `forward_kl_topk` 替代我们自己的实现
- 可以利用官方的 multi-teacher 路由替代 condition_weights 的部分功能

**挑战**：
- v0.7.1 → v0.8.0 的 `dp_actor.py` 有较大重构（引入 v1 trainers, distillation module）
- 我们的 `dp_actor.py` patch 需要重写以适配新的 loss 注册机制
- `ray_trainer.py` 的 post-rollout hook 需要与官方 teacher_loop 对接
- 需要验证 cu128 环境与新版 verl + vLLM 的兼容性

**适配策略**：

1. **`compute_verl_sparse_reverse_kl`（直接 backprop RKL）** → 注册为 `loss_mode=reverse_kl_direct`，在 distillation loss 中直接 backprop（类似 `forward_kl_topk` 但不限于 top-K 交集）
2. **`compute_verl_sparse_jsd`** → 注册为 `loss_mode=jsd_topk`
3. **`fc_condition_weights` + `fc_condition_ids`** → 最干净的方式是扩展官方的 multi-teacher 机制：每个 condition 视为一个"虚拟教师"，condition_weights 控制其在 loss 中的贡献
4. **Post-rollout hook** → 改为在 teacher_loop 的 teacher logprob 回调中注入 condition metadata

**预估工作量**：3-5 天（含测试）

#### 选项 B：重新构建（clean start from v0.8.0）

```
verl v0.8.0 (fresh clone)
    ↓ 只迁移 FC-OPD 核心逻辑
重新实现为官方 OPD 的插件
```

**优势**：
- 干净，没有历史包袱
- 可以充分利用官方 OPD 架构
- GKD recipe 已不需要（官方有 forward_kl_topk）

**挑战**：
- 需要重新配置 cu128 环境（但官方有 Docker 镜像）
- FC-OPD 所有功能需要从头适配
- 风险：可能发现官方 OPD 缺少我们需要的 hook point

**预估工作量**：5-8 天（含环境配置和测试）

#### 选项 C：保持当前版本，只修 bug（不推荐）

- 短期成本低，但长期维护成本持续增长
- 每次 verl 上游更新都要手动 merge
- 无法受益于官方优化（fused top-K kernel, FSDP2 改进, NCCL 修复等）

### 2.6 升级的最小可行步骤

1. **验证环境**：在 cu128 环境下安装 verl v0.8.0 的依赖（主要是 vLLM 版本要求）
2. **跑通官方示例**：用 `run_qwen3_5_4b_fsdp.sh` 跑一个 forward_kl_topk 的 Geometry3K 训练
3. **对比验证**：确认官方 `forward_kl_topk` 与我们 `compute_verl_sparse_topk_kd` 的数值等价性
4. **注册 FC-OPD loss**：把 `reverse_kl_direct` 和 `jsd_topk` 注册到官方 registry
5. **适配 condition routing**：扩展 multi-teacher 机制支持 FC-OPD 的条件路由
6. **全流程测试**：Qwen3-VL-4B + Qwen3-VL-32B，Geometry3K，FC-OPD 多条件

---

## 三、备选方案：ChatLearn（阿里云 PAI）

### 3.1 基本信息

| 项目 | 详情 |
|------|------|
| 仓库 | [github.com/alibaba/ChatLearn](https://github.com/alibaba/ChatLearn) |
| 最新版本 | v1.2.0 (2025-08-25) |
| 训练后端 | FSDP2, Megatron (Mcore) |
| 推理引擎 | vLLM, SGLang |
| 支持算法 | GRPO, GSPO, RLHF, DPO, OnlineDPO |
| 环境 | Docker 镜像：CUDA 12.6, PyTorch 2.6.0, vLLM 0.8.5, Python 3.12 |
| 规模 | 已验证 7B-671B，支持 600B+ |

### 3.2 关键发现：ChatLearn 没有原生 OPD 支持

**ChatLearn 专注于 RLHF/GRPO/GSPO，不包含 on-policy distillation 模块**。其文档中：
- ❌ 无 teacher-student KL distillation loss
- ❌ 无 teacher model server 管理
- ❌ 无 top-K forward/reverse KL 实现
- ❌ 无多教师路由

要在 ChatLearn 上实现 OPD，你需要：
1. 自己搭建 teacher model server（类似 verl 的 teacher_loop）
2. 自己实现 KL loss 计算并注入 GRPO 的 reward 或 loss
3. 自己管理 student rollout ↔ teacher logprob 的数据流

**这个工作量和从头写一个 OPD 框架差不多。**

### 3.3 ChatLearn vs verl 对比

| 维度 | verl v0.8+ | ChatLearn v1.2 |
|------|-----------|---------------|
| OPD 原生支持 | ✅ 完整 | ❌ 需自定义 |
| FSDP2 | ✅ | ✅ |
| VLM (Qwen3-VL) | ✅ | ✅ (GRPO only) |
| Multi-teacher | ✅ | ❌ |
| Forward KL distill | ✅ `forward_kl_topk` | ❌ |
| Reverse KL distill | ✅ `k1/k3` (PG) | ❌ |
| 环境灵活性 | conda env / Docker | 强依赖 Docker 镜像 |
| 社区生态 (OPD) | SOD, OPSD, Draft-OPD, Direct-OPD | 无 OPD 相关项目 |
| 模型下载 | HuggingFace | modelscope（国内镜像） |
| 性能基准 | 无直接对比 | vs DS-Chat/OpenRLHF: +52-208% |

### 3.4 结论：ChatLearn 不适合作为 OPD 方案

ChatLearn 是一个优秀的 RLHF 框架，但**不是 on-policy distillation 框架**。如果你的主要需求是 GRPO/GSPO 训练（不做蒸馏），ChatLearn 值得评估。但对于"学生模型通过 teacher KL 进行 on-policy 蒸馏"这个核心需求，ChatLearn 缺少关键基础设施，迁移成本远高于 verl 升级。

---

## 四、社区生态全景

做 on-policy distillation 的团队几乎**全部基于 verl** 构建：

| 项目 | 基于 | 专注点 |
|------|------|--------|
| **SOD** (字节跳动) | verl + vLLM/SGLang/Megatron | Tool-integrated reasoning agent |
| **OPSD** (Thinking Machines) | verl | Self-distillation with token importance |
| **Draft-OPD** | verl | Speculative decoding draft models |
| **Direct-OPD** (清华+字节) | verl | Weak-to-strong generalization |
| **TrOPD** (三星) | 独立实现 | Trust-region constrained OPD |
| **Spider** (Collinear AI) | 自研 | 轻量级 on/off-policy 蒸馏 |
| **EasyDistill** (阿里) | vLLM+DeepSpeed | 全流程 KD 工具包 |

**verl 已经是 on-policy distillation 的事实标准。**

---

## 五、最终推荐

### 推荐路径：升级 verl 到 v0.8+ 并适配 FC-OPD（选项 A）

**理由**：
1. **最小迁移成本**：我们只改了 ~430 行代码，且大半已在官方 OPD 中有对应物
2. **最大收益**：自动获得官方 OPD 的 bug 修复、性能优化、新功能
3. **生态一致性**：verl 是 OPD 的事实标准，跟随上游是最可持续的策略
4. **FC-OPD 差异化**：条件路由是我们的核心创新，作为插件贡献回 verl 上游也更有价值

### 不推荐 ChatLearn

ChatLearn 缺少 OPD 基础设施，迁移成本远超 verl 升级。

### 下一步行动

1. **立即**：用 cu128 环境试跑 `run_qwen3_5_4b_fsdp.sh`（forward_kl_topk on Geometry3K）
2. **短期**：将 FC-OPD 的 reverse_kl_direct / JSD 注册为官方 loss_mode
3. **中期**：适配 condition routing 到官方 multi-teacher 机制
4. **长期**：考虑将 FC-OPD 贡献回 verl 上游

---

## 参考资料

- [verl OPD PR #5041](https://github.com/verl-project/verl/commit/455e44c6feeabe378ad9dcc790fe9313926de12c)
- [verl OPD 文档](https://verl.readthedocs.io/en/latest/algo/opd.html)
- [verl OPD 示例脚本](https://github.com/verl-project/verl/blob/main/examples/on_policy_distillation_trainer/README.md)
- [ChatLearn GitHub](https://github.com/alibaba/ChatLearn)
- [ChatLearn GRPO FSDP 教程](https://chatlearn.readthedocs.io/en/latest/tutorial/tutorial_grpo_fsdp.html)
- [SOD: Step-wise On-policy Distillation](https://github.com/YoungZ365/SOD)
- [Spider: On/Off-policy Distillation](https://github.com/collinear-ai/spider)
- [EasyDistill: KD Toolkit](https://github.com/modelscope/easydistill)
- [TrOPD: Trust Region OPD](https://github.com/Xingrun-Xing2/TrOPD)

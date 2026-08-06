# 两条 Track 下一阶段实施方案（给 Claude Code）— 2026-08-06

> 命名说明：本文件服从用户最新口径，**Track A = support-aware 研究主线**，
> **Track B = Qwen3.5 训练环境线**。旧文档曾把 Qwen3.5 称为 Track A、把
> support-aware 称为 Track B；读旧日志时必须按内容和 commit 区分，不能按旧标签混用。

## 0. 执行结论

1. Track A 不再尝试用 teacher/student gap 或离线 TOPD proxy 定位前缀位置；Gate C
   已经证伪这一 locator。保留真正成立的两条事实：
   - rare-success prompt 上，32B teacher 能给出 verified correct proposal；
   - answer-free teacher prefix 能因果提高 frozen 4B student 的续写成功率。
2. Track A 锁定的第一条可测技术路线是 **Support-Transition Prefix OPD（STP-OPD）**：
   先用 frozen-policy rescue 实验选择 rare-success、rescue-positive prompt 和最短可救援
   teacher prefix；训练时对 prefix 做 FKL，对 student 自己采样的 suffix 做在线 teacher
   RKL/K1，并混入无 prefix 的 OPD/任务奖励、逐步撤掉脚手架；最终只在无 prefix 条件下
   测学生是否从 rare support 转为稳定 support。
3. suffix 不采用“当前 student 教当前 student”的自蒸馏作为主方案。相同模型、相同
   context 的即时自蒸馏没有新增校正信息；如果用 EMA/student-old 只是在做稳定化，不回答
   “teacher prefix 后怎样把正确延续转移给 student”。主方案必须保留在线 teacher scorer。
4. 更大 teacher 的目标不是直接训练，而是先修复 no-correct proposal supply。优先检查/
   下载 `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`，用 8 卡 TP 做 teacher-only proposal
   generation；teacher 退出后再由 4B student exact-token rescore。不能沿用当前“一卡
   teacher + 一卡 student 同进程”的 32B runner。
5. Track B 的三次 120-step 只改变 `data.seed` 和 `actor.data_loader_seed`，证明的是冻结
   配置的工程稳定性和长度轮廓稳定性，不是严格的全 RNG seed 学习复现。下一步先做固定
   离线评测、完整 seed contract、resume 和真实图片 canary；在此之前不直接加长训练。

---

## 1. Commit 与实验线映射

| commit | 内容 | 本文件口径 |
|---|---|---|
| `72e0841` | 12 prompt prefix intervention；Gate B pass、Gate C fail | Track A 证据 |
| `9d1d417` | offline TOPD proxy；sanity/Gate C' NO-GO | Track A 负结果 |
| `556d60f` | PNPO 论文复盘；缓存复用建议 | Track A 训练实现参考 |
| `1258002` | Qwen3.5 三 seed × 120-step | Track B 稳定性结果 |

实施前使用干净 worktree，记录主仓 commit、dirty 状态和 backend commit。不要把现有用户
worktree 的未提交改动、模型、数据、checkpoint 或 raw JSONL 带进 commit。

---

## 2. 现有 Gate 到底在统计什么

### 2.1 Gate A：teacher proposal supply，统计单位是 prompt

总体 28 个 prompt 来自 student 的独立 K=32 支持确认：

- `no_correct_observed`：11 题，每题 student 正确数 `c=0/32`；
- `rare_success`：17 题，每题 `1<=c<=8/32`。

每题让 32B teacher 独立生成 `M=4` 个 proposal，所以共有 `28×4=112` 条 teacher
rollout。先用 Geometry3K verifier 判正确，再让 4B student 对**正确 teacher proposal 的
原始 token IDs**做 teacher-forced exact-token NLL；每题最多保留 NLL 最低的 2 条。

这里有两种统计单位，不能混写：

- proposal availability 是 **prompt 级**：一题 4 条里至少一条正确即 available；
- proposal correctness/NLL 是 **rollout 级**：112 条里有 30 条正确，最后 22 条 retained。

结果：no-correct available `1/11`，rare-success available `11/17`，总体 `12/28`；
retained 是 12 个 prompt 上的 22 条 proposal。

**没有绝对 likelihood/NLL 阈值。** 当前 acceptance 只要求 verifier correct、NLL finite，
再按每题相对 NLL 排序取 top-2。`trimmed NLL median=0.358` 只是 rare retained 的描述性
统计，不是“合理 likelihood”的门槛；因此不能说“有多少 rollout 通过某个 likelihood
threshold”。若后续需要绝对门，必须先在 held-out proposal 上校准，不能事后用 0.358。

### 2.2 Gate B：prefix 是否因果救援，单位是 prompt × horizon × arm

输入是 Gate A retained 的 12 题（11 rare + 1 no-correct）。对每题、每个 horizon
`h∈{64,128,256,512}`，固定三种 context：

- teacher：verified correct、answer-free teacher prefix；
- wrong：同长度的高概率错误 student prefix；
- unaided：无 prefix。

每个条件采样 `K=8` 个 student continuation。总体 teacher `98/368=26.6%`，wrong
`12/384=3.1%`，unaided `6/96=6.3%`；7/12 prompt 通过预注册 rescue rule，且全部是
rare-success。

Jeffreys posterior 是小样本二项率的贝叶斯估计：观察 `c/K` 后，
`p|c,K ~ Beta(c+0.5, K-c+0.5)`。比较两臂时从两个 Beta 后验采样，估计
`E[p_T-p_W]` 和 `P(p_T>p_W)`。当前严格规则同时要求 teacher 对 wrong、unaided 的
posterior mean lift 都至少 0.20，胜出概率都至少 0.90。

### 2.3 Gate C：旧 locator 是否预测 rescue；已经失败

Spearman 是两个变量**排序**的一致性相关，不假定线性。这里比较“某 prompt 的 gap
locator 分数排名”和“该 prompt 的 rescue lift 排名”。三个预注册统计量的 Spearman
均为负，离线 TOPD proxy 的 sanity 也失败。因此结论只是“这些 locator 不可用”，
不是“teacher prefix rescue 不存在”，也不是对真实 same-prefix TOPD 的否定。

后续 STP-OPD 不再让旧 Gate C 阻塞。prefix 位置直接由 frozen-policy 因果 rescue curve
选择；新的训练 Gate 测的是无 prefix support transition。

---

## 3. 相关工作与可区分空间

### 3.1 已经有人回答过哪些部分

| 工作 | 已经做了什么 | 对我们的约束 |
|---|---|---|
| [FastOPD, arXiv:2602.15260](https://arxiv.org/abs/2602.15260) | 观察 OPD 信号偏早；短 reasoning prefix 就可能显著帮助解题，并通过 prefix-only OPD 降低计算 | “短 prefix 有用”本身不能作为新意 |
| [PrefixRL, arXiv:2601.18795](https://arxiv.org/abs/2601.18795) | 用正确 off-policy prefix 作 context、mask prefix 梯度，只对 suffix 做 on-policy RL；还观察无 prefix back-generalization | 必须对比 masked-prefix + suffix RL，且必须测撤掉 prefix 后的能力 |
| [Prefix Utility Model, arXiv:2606.07190](https://arxiv.org/abs/2606.07190) | 把 prefix utility 定义成 conditioned solve-rate gain，并学习排序/预测 utility | Gate B 本质上是受控的 empirical prefix-gain estimator；不能 claim 首次定义 |
| [Relay-OPD, arXiv:2607.26057](https://arxiv.org/abs/2607.26057) | 在线检测 reflection-token mismatch，让 teacher 短暂接管若干段，再交还 student；只用很少 teacher token 就能改善准确率 | 最接近“teacher prefix → student continuation”；我们必须强调 support-state、verifier、视觉和 support transition |
| [TOPD, arXiv:2606.00305](https://arxiv.org/abs/2606.00305) | 同一 student prefix 上用近未来 teacher/student 轨迹和 OT 找发散点，再构造 guidance | 我们的离线不同前缀 proxy 失败不否定 TOPD；但不再把它当 locator 主线 |
| [TREK, arXiv:2607.05339](https://arxiv.org/abs/2607.05339) | verified teacher proposal、student NLL selection、FKL + GRPO | full-trajectory verified FKL 是必须比较的基线 |
| [TRD, arXiv:2606.08432](https://arxiv.org/abs/2606.08432) | teacher 参考答案后重写/修正完整 student trajectory 再蒸馏 | trajectory-level rewrite 是另一类方案，成本和监督更重 |
| [SG-OPD, arXiv:2606.09304](https://arxiv.org/abs/2606.09304) | verified teacher warm start + token sign-consistency gate + RKL | 说明 verified proposal 与 RKL 可组合，但没有我们的支持状态/视觉 rescue 路由 |
| [PNPO, arXiv:2608.01418](https://arxiv.org/abs/2608.01418) | 多 epoch 复用 rollout 时用 prefix-normalized importance ratio 和 hard rejection | 适合作为缓存复用权重消融，不能替代 causal rescue routing |

### 3.2 我们还能做、且需要用实验守住的差异点

主张不应是“teacher prefix 可以救”，而应是：

> 在真实 VLM 题上，student 的 stochastic support state 能预测哪类题可由 verified、
> answer-free teacher prefix 救援；对 rescue-positive rare-support 题，用 prefix FKL
> 创建可达支持、在 student-generated suffix state 上做 teacher RKL，再撤除 scaffold，
> 能把条件性 prefix gain 转成无 prefix 的 support transition。

四个不可删的差异维度：

1. `no_correct` 与 `rare_success` 分层，不把所有 hard prompt 混在一起；
2. teacher prefix、same-length wrong prefix、unaided 三臂的因果 rescue gate；
3. prefix FKL 与 suffix teacher RKL 的区域化目标，而不是只 mask prefix 或整轨 FKL；
4. 最终评估是 fresh、无 prefix rollouts 的支持状态迁移，并加入 original/degraded/
   shuffled/no-image 视觉反事实。

---

## 4. Track A 锁定路线：STP-OPD

### 4.1 数据与分组

先在现有 7 个 rescue-positive rare prompt 上做 mechanics pilot，只验实现和方向，不做论文
效应量。随后从未参与现有 Gate 的 Geometry3K prompt 扩展：

1. fresh student `K=8` 粗筛；
2. 边界候选补到 `K=32`，按同一 Jeffreys/state 口径确认 rare/no-correct；
3. 在 teacher proposal 前固定 prompt-level `route-train / route-val / held-out-test` split；
4. 目标至少得到 64 个 train rare、32 个 val rare、32 个 held-out rare；no-correct 保留为
   proposal-source 对照，不强行塞进训练主集。

任何 teacher proposal、horizon 或阈值选择只能看 route-train/val；held-out-test 只在方案冻结
后打开。

### 4.2 快速探测 rare-success 是否可救

现有 7 题沿用已经确认的最短 horizon。新 prompt 使用两阶段自适应 probe，减少全量
`4 horizons × 3 arms × K8` 的开销：

1. 先取得至少一个 verified correct teacher proposal，并准备 unaided 和高概率 wrong
   student prefix；所有 prefix 必须 answer-free、tokenizer exact-ID 对齐。
2. Stage 1：只测 `h=128,256`，teacher/wrong/unaided 各 `K=4`，三臂使用成对 seed grid。
3. provisional promote：teacher 同时对 wrong 和 unaided 满足 posterior mean lift ≥0.15、
   `P(teacher>control)≥0.80`。若 h128 promote，再向下测 h64；若 h128/h256 均不 promote
   但 teacher 至少有一次成功，再向上测 h512。
4. Stage 2：只把候选最短 horizon 增采到 `K=8`，套用现有严格规则 lift ≥0.20、
   probability ≥0.90；失败则标记 rescue-negative，不反复调阈值。
5. 输出每题完整 posterior、最短通过 h、teacher/wrong/unaided token 数和计算量。当前样本量
   不足以先训 PUM；累计至少 100 个有标签 prompt 后，才考虑学习 utility predictor。

### 4.3 训练样本构造

对每个 rescue-positive prompt 选 verified teacher trajectory `z` 和最短通过的 answer-free
prefix `z_1:h*`：

1. scaffolded 分支：固定 `z_1:h*` 作 context，由当前 student 采样 suffix
   `y ~ pi_S(.|x,image,z_1:h*)`；
2. teacher 对**同一条 hybrid prefix + student suffix token IDs**在线 forced-score；不允许
   teacher 重新渲染 prompt，不允许 teacher-only router 绕过 student scorer；
3. unscaffolded 分支：同一 prompt 不给 prefix，由 student 正常采样，执行现有 OPD + task
   reward；
4. 一个 batch 内按 prompt 成对放 scaffolded/unscaffolded，避免数据顺序差异冒充方法效果。

### 4.4 区域化目标

主目标：

```text
L = lambda_P * L_prefix_FKL
  + lambda_D * L_suffix_teacher_RKL_K1
  + lambda_R * L_suffix_task_GRPO
```

- `L_prefix_FKL`：teacher-forced correct prefix token CE，按 prefix 有效 token 单独归一化；
- `L_suffix_teacher_RKL_K1`：suffix 由当前 student 采样，teacher 在相同 state 上打分，沿用
  已验证的 sampled-token K1 reverse-KL estimator；
- `L_suffix_task_GRPO`：只在 student suffix 上计算任务优势；固定 teacher prefix 不进 PG
  likelihood ratio；
- 三个区域分别 mask/normalize，不能让 prefix 长度隐式改变 loss 权重；
- 第一个 pilot 固定 `lambda_P=lambda_D=lambda_R=1` 作为显式起点，只在主效应通过后调权重。

scaffold 比例采用预注册 schedule：steps 1–20 为 75%，21–40 为 50%，41–60 为 25%；
其余样本无 prefix。pilot 内 `h*` 固定，不同时调 horizon；后续才消融 horizon annealing。

为什么不是 suffix self-distillation：若 target 就是即时 student 自身分布，KL 期望为零；若
target 是 old/EMA student，只提供稳定化且不携带 teacher 的纠错信息。可将 EMA suffix
distill 作为效率消融，但主线用 teacher RKL。

### 4.5 第一轮 mechanics pilot 的匹配臂

在现有 7 个 rescue-positive prompt 上先跑 4 臂，每臂相同 prompt、生成 token budget、
optimizer steps、LR、batch、seed grid 和任务奖励：

| arm | prefix 区 | suffix 区 | 用意 |
|---|---|---|---|
| A0 no-prefix OPD | 无 | teacher RKL/K1 + GRPO | 现有 OPD 基线 |
| A1 PrefixRL-style | prefix 仅作 context、全 mask | GRPO | 排除“只给正确 context 做 RL 就够了” |
| A2 prefix-FKL | FKL | GRPO | 检验创建 prefix support、但不做 teacher suffix OPD |
| A3 STP-OPD（主） | FKL | teacher RKL/K1 + GRPO | 锁定路线 |

若 A3 通过 mechanics gate，扩展实验再加入：

- A4 TREK-style：完整 verified trajectory FKL，再 fresh no-prefix GRPO；
- A5 PNPO cache reuse：只改变 cached-prefix 多 epoch 权重，作为算力/复用消融；
- A6 EMA self-distill suffix：只作效率/稳定性消融，不作为主方法。

### 4.6 评测与新 Gates

所有最终评测都**不给 teacher prefix**，使用 fresh seed、每 prompt `K=32`；正式确认升到
`K=64`。报告 prompt-macro 指标，不把所有 rollout 池化成一个比例。

主要指标：

- 每题 Jeffreys posterior mean 的 `p_after-p_before`；
- `rare -> stable`、`rare -> no-correct` 的 prompt 数；
- Pass@K、accuracy、boxed/clip/EOS；
- 主方法相对最强基线的 prompt-paired bootstrap CI；
- frozen rescue lift 是否预测无 prefix training gain；
- original/degraded/shuffled/no-image 下的 support transition 和 teacher/student sensitivity。

预注册 gate：

1. **P0 implementation**：exact token hash、prefix/suffix mask、loss normalization、teacher/
   student tokenizer mapping、online student scorer、resume manifest 全部通过；任一失败不跑。
2. **P1 mechanics**：现有 7 题中，A3 至少 4/7 的无 prefix posterior mean 提升，且 A3
   对 A0 的 prompt-macro mean lift ≥0.10；只用于决定是否扩样，不作研究 claim。
3. **P2 held-out**：扩样后的 held-out prompt 上，A3 相对最强基线的 paired bootstrap
   95% CI 下界 >0；否则不 claim support transfer。
4. **P3 visual**：original-image 增益不能在 shuffled/no-image 条件下等量保留；若保留，
   只能表述为文本/答案模式迁移，不能表述为 VLM visual reasoning 改善。

---

## 5. Track A 给 CC 的实现清单

### 5.1 代码边界

研究逻辑放在：

```text
src/dual_track_opd/support_aware/rescue_screen.py
src/dual_track_opd/support_aware/prefix_scaffold.py
src/dual_track_opd/support_aware/support_transition_eval.py
```

配置放在：

```text
configs/experiment/support_aware_rescue_screen.yaml
configs/experiment/support_transition_prefix_opd_pilot.yaml
configs/experiment/support_transition_eval.yaml
```

HPC runner 只做环境变量、分片、manifest、启动/合并；不得把 loss/router 逻辑写进 shell。
需要改 verl 时只产出 `patches/verl/` 下的最小 patch 和说明，不 vendor backend。

至少新增测试：

- adaptive horizon 状态机与 Jeffreys 阈值边界；
- answer leakage、prefix 短于 horizon、三臂 seed pairing；
- exact token hash/tokenizer mismatch fail-fast；
- prefix FKL / suffix RKL / suffix PG mask 不重叠；
- 各区域独立 normalization；
- scaffold schedule 与无 prefix paired batching；
- online student scorer 未被绕过；
- strict merge、resume、manifest/hash。

### 5.2 更大 teacher：立即动作与必要重构

首选模型：`Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`，固定 revision
`7fbcd8c922b966db084258a3d89073cb8f5cac39`。先做 11 个 no-correct，17 个 rare 作为
paired control；保持 `M=4,temp=.7,top_p=.95,max_new_tokens=4096,seed=20260805`，不要
覆盖 32B 产物。

远端节点动作：

```bash
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export MODEL_DIR="$DTOPD_ROOT/models/Qwen3-VL-235B-A22B-Instruct-FP8"
test -f "$MODEL_DIR/config.json" && echo MODEL_PRESENT || true
df -BG "$DTOPD_ROOT/models"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
```

若模型不存在，只有在 models 卷可用空间 ≥300 GiB、没有同名活跃下载时启动：

```bash
export DTOPD_PYTHON="$DTOPD_ROOT/envs/vision-opd-cu128/bin/python"
nohup "$DTOPD_PYTHON" - <<'PY' \
  > "$DTOPD_ROOT/fc-opd-storage/logs/qwen3vl235b_fp8_download.log" 2>&1 &
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    revision="7fbcd8c922b966db084258a3d89073cb8f5cac39",
    local_dir="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-235B-A22B-Instruct-FP8",
    max_workers=8,
)
PY
```

下载完成后校验 snapshot 文件列表/hash/revision。若 `huggingface_hub` 不在该 Python，使用
已有 Qwen3.5 env 的 Python 执行下载；不要为下载就原地修改 legacy inference env。

实现必须把现有 `proposal_feasibility.py` 拆成两阶段：

1. `teacher_proposal_generate`：8 GPU TP teacher-only，保存 raw/display text、原始 token IDs、
   content mask、finish reason、seed、verifier verdict、model revision；
2. teacher 进程彻底退出并确认显存释放；
3. `student_proposal_rescore`：4B student 对正确 proposal exact-token rescore，执行现有
   finite + per-prompt top-2 retention；
4. strict merge 复用现有 schema，并保留 `source_teacher=32B/235B` 字段。

先做 1 prompt × 1 proposal TP8 smoke。当前 `vision-opd-cu128` 只证明了 32B Transformers
路径，未证明 235B MoE FP8 TP；若 vLLM registry/kernel smoke 失败，不要升级污染原 env，
新建有 manifest 的 inference prefix。通过后立即跑 28 prompt paired proposal experiment。

更大 teacher 的决策规则保持 state-separated：no-correct availability ≥25% 才继续该状态；
rare availability ≥50% 且 retained prompts ≥8 才作为新的训练 proposal source。只比较
proposal supply，不因为模型更大就默认进入训练。

### 5.3 Track A 环境口径

| 阶段 | 当前/目标环境 | 状态 |
|---|---|---|
| K32、Gate A/B/C、32B frozen diagnostic | `$DTOPD_ROOT/envs/vision-opd-cu128`，Python 3.12、torch 2.10+cu128 | 历史结果已验证；registry 标为 legacy inference |
| 235B proposal generation | 先在上述环境做 TP/MoE smoke；不通过则建独立 inference env | 尚未验证 |
| STP-OPD 真训练（Qwen3-VL real image） | 目标 `$DTOPD_ROOT/envs/va-opd-native-e003-cu128-r595-v1` + pinned verl/patch | candidate；必须先补 real-image end-to-end smoke |
| Qwen3.5 text OPD | `$DTOPD_ROOT/envs/va-opd-qwen35-cu128`，实际 cu129 | Track B 专用，不能因名字像 cu128 就与研究训练混用 |

因此“Track A 现在用哪个环境”的准确回答是：**现有研究证据用的是
`vision-opd-cu128` 的 frozen-policy 推理；STP-OPD 训练环境尚未通过最终 gate，不能声称已经
可训练。** CC 在写训练代码前先对 native cu128 env 做真实 Geometry3K image tensor、student
rollout、teacher forced-score、task reward、反向更新 4-step canary；通过后更新 registry 为
active，失败则记录 blocker，不静默切到 text-only 数据。

---

## 6. Track B：Qwen3.5 当前到底跑了什么

### 6.1 数据

- train：`geometry3k_gkd/train_text_only.parquet`，1901 rows；
- val：`geometry3k_gkd/val_text_only.parquet`，200 rows；
- `data_source=hiyouga/geometry3k`；
- 为绕过 processor/image assertion，当前 parquet 已移除 `images/condition_inputs`。

`FCOPDDataset` 会把 question 改写成包含字面 `<image>` 的 `boxed_only` prompt，但当前文件
没有真实 image tensor。因此这轮验证的是 **Geometry3K text-only OPD 工程路径**，不能当成
Qwen3.5-VL 图像训练已验证，也不能据此说与全部 Qwen3/Qwen3-VL 系列天然兼容。

### 6.2 方法与冻结配置

| 项 | 当前值 |
|---|---|
| student / teacher | Qwen3.5-4B / qwen3.6-27B |
| objective | sampled-token K1 reverse-KL policy-gradient estimator + accuracy task reward |
| batch | 6 prompt × 4 rollout = 24 sequence/step；PPO mini prompt batch=6；8 workers |
| decoding | `enable_thinking=False,temp=1.0,top_p=.95,top_k=-1,cap=4096` |
| optimization | LR `1e-6`，120 steps，3 actor GPU + 1 teacher GPU |
| eval/checkpoint | val 200 prompt、n=1、每20步；checkpoint 每30步 |
| prompt/reward | `boxed_only`；accuracy reward 可达；legacy format reward 结构性为 0 |

`use_policy_gradient=True` 时，K1 的 detached dense signal 与 student current logprob 形成
reverse-KL estimator；它不是普通 teacher-forced FKL。

### 6.3 三 seed 改了什么、没改什么

`1258002` 的 seed2/3 只显式覆盖：

```text
data.seed = 2/3
actor_rollout_ref.actor.data_loader_seed = 2/3
```

其余训练/模型/数据/解码超参冻结。它主要改变数据顺序，未证明 vLLM rollout、validation、
Ray worker、torch/CUDA 全部 RNG 都随 seed 正确绑定。step-0 val 已在 9.0%–10.5% 间不同，
说明当前 validation 不是 paired fixed-seed eval。

三 run 都 rc=0、150–170 分钟、长度/clip 轮廓高度相似，说明**精确这一条运行路径工程上
稳定可用**；但 n=1 的 200 题 val 只多 1–3 题正确，尚不能证明学习增益。

### 6.4 环境是否可用

环境路径虽然叫 `va-opd-qwen35-cu128`，实际安装是：torch 2.11.0+cu129、vLLM
0.23.0+cu129、Transformers 5.12.0、verl 0.9.dev @ `334d9f8b`、Ray 2.55.1。
H200/driver 570 上已通过 GPU model load、FlashInfer、3+1 卡训练、checkpoint 和三次完整
run。因此：

- 对当前 Qwen3.5-4B ← qwen3.6-27B、text-only、冷启动 120-step 配置：**可用**；
- 对 clean-node rebuild、resume、Qwen3.5-9B、Qwen3 text、Qwen3-VL real image：**未验证**；
- 不能把“Qwen3.5 可用”外推成“我们的所有 Qwen3 系列都兼容”。

### 6.5 Track B 下一步实施顺序

1. **固定离线评测优先**：三 seed 的 base/step60/step120，共 9 个 checkpoint 条件；使用
   同一 200 prompt、同一 `N=8` decoding seed grid，prompt-paired 统计 accuracy/pass@8、
   boxed、clip、EOS、length，并给 bootstrap CI。base 模型只算一次，但在三个 paired 表中
   复用同一原始结果。
2. **补完整 RNG contract**：manifest 写入并显式传播 Python/NumPy/torch/CUDA、data
   sampler、rollout worker、vLLM sampling、validation seed；同 prompt/rollout_id 使用确定性
   seed derivation。验证同 checkpoint 同 seed 可重放响应 hash，允许后端非确定性时必须量化。
3. **resume canary**：同一冻结配置跑 uninterrupted 60 steps，与 30-step checkpoint resume
   到 60 比较 step、optimizer/scheduler、data cursor、manifest、指标分布和产物完整性；不要求
   GPU bitwise identical，但不能重放/跳过 batch。
4. **clean-node replay**：用 wheelhouse 和 setup script 在新 prefix 重建，跑 import、vLLM
   engine、tokenizer alignment、4-step train 和 checkpoint smoke；成功后修正环境名/manifest，
   避免 cu128 路径名掩盖 cu129 实际栈。
5. **兼容矩阵 canary**：
   - Qwen3.5-9B student：tokenizer alignment + 1 rollout + 1 backward step；
   - Qwen3 text model：text-only 4-step；
   - Qwen3-VL/Qwen3.5-VL：必须使用含真实 image column 的 2 prompt 数据，验证 processor、
     multimodal tensors、student rollout、teacher score 和 reward；
   每格单独记录模型 revision/显存/版本，不能只做 `from_pretrained` import。
6. 只有固定离线评测证明 step120 相对 base 有稳定 paired gain 后，才做更长 run。第一轮变量
   仍只改一个：优先 `rollout_n 4→8` 或 task/distill 权重；不先上 8192，不同时改 LR、batch、
   teacher 和 prompt。

---

## 7. 推荐调度顺序与 CC 回传要求

按下面顺序执行，避免两个 track 抢同一批 GPU/结论混淆：

1. 立即远端 inventory：GPU、磁盘、235B model；缺模型且空间足够就启动下载。
2. 下载期间完成 Track B 固定离线 eval runner 和 RNG manifest；eval 可用空闲 GPU 分片。
3. 实现 235B teacher-only generation / student rescore 两阶段，先 TP8 smoke，再跑 28 prompt。
4. 同时在现有 7 个 prompt 上实现 STP-OPD loss/mask 单测和 CPU synthetic test。
5. native Qwen3-VL real-image 4-step canary 通过后，跑四臂 mechanics pilot。
6. mechanics P1 通过才扩 rare cohort；没通过先看 prefix FKL、suffix RKL 和 scaffold removal
   三个机制消融，不用更大规模掩盖失败。

每次 CC 回传必须包含：

- repo/backend commit、dirty 状态、resolved config；
- dataset/model manifest 和 SHA/revision；
- 环境 prefix 与实际 torch/CUDA/vLLM/Transformers/verl 版本；
- GPU mapping、wall time、teacher/student token 数和峰值显存；
- raw output 的 Git 外路径、summary 路径和 hash；
- gate 逐条 pass/fail、失败原因和下一步；
- 不提交权重、数据、checkpoint、raw rollout 或下载日志。

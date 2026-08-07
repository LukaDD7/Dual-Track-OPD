# Causal State Probe（Phase I）

本模块把现有 K=32 support diagnostic、verified teacher proposal、prefix rescue 和
FC-OPD 视觉反事实基础设施接成一个**只诊断、不更新权重**的统一流水线。目标不是把某个
token 人工命名为 perception/reasoning boundary，而是找到 teacher supervision 在 student
当前状态上是否能真正改变 downstream solvability。

## 1. 当前代码库中的位置

研究逻辑在：

```text
src/dual_track_opd/support_aware/
  causal_schema.py       # 统一记录
  causal_dataset.py      # K32/cohort/proposal join
  causal_image.py        # deterministic full/degraded/null
  visual_js.py           # full-vocab JS + multi-scale selector
  causal_probes.py       # posterior lift、taxonomy、barrier
  causal_runtime.py      # Qwen3-VL exact-token forward/generation
  causal_report.py       # JSONL/CSV/SVG/summary
  causal_state_probe.py  # CLI、resume、manifest
```

配置和入口：

```text
configs/diagnostics/causal_state_probe.yaml
scripts/hpc/run_causal_state_probe.sh
```

现有 `diagnostic.py`、`proposal_feasibility.py`、`prefix_intervention.py` 和
`rescue_screen.py` 不被替换；它们继续负责冻结 cohort、产生 verified proposal 和已有
prefix-rescue 对照。本模块消费这些不可变产物。

## 2. 变量与估计量

固定一条 student rollout `y^S`，所有视觉反事实保持 response token IDs 完全相同。

```text
P_t^full = pi_S(. | I_full, q, y^S_<t)
P_t^deg  = pi_S(. | I_degraded, q, y^S_<t)
P_t^null = pi_S(. | I_null, q, y^S_<t)
```

逐 token 计算完整词表 Jensen–Shannon divergence：

```text
J_t^fine = JS(P_t^full, P_t^deg)
J_t^all  = JS(P_t^full, P_t^null)
```

实现使用 fp32 `log_softmax` 与 `logaddexp`，按 response positions 分 chunk。每个 chunk
完成后只保留：

```text
position, token_id,
js_full_degraded, js_full_null,
entropy_full, entropy_degraded, entropy_null
```

完整 `[T,V]` logits 不写磁盘。支持 `logits_to_keep` 的 Qwen forward 只返回当前 chunk
所需位置；不支持时仍逐 chunk 前向并立即释放 logits，同时在 manifest 中标记 fallback。

## 3. Image counterfactual

三种 condition 始终保留相同 image input schema、图像宽高、modality tokens 和
`image_grid_thw`：

- `full`：原图；
- `degraded`：默认 `blur_sigma_2`，复用 `fc_opd.degradation` 的配置语义；
- `null`：同尺寸常数 RGB 图像，默认 255（白色）。

运行时逐样本 fail-fast 检查：

- 三个 condition 的 prompt token IDs 完全一致；
- `image_grid_thw` 完全一致；
- pixel tensor shape 完全一致；
- intervention metadata 与三个 PNG-content SHA256 被记录。

所有变换 deterministic，不使用随机 crop/blur。

## 4. Candidate transition zone

单 token peak 不作为 boundary。对 `w in {8,16,32,64}` 计算 trailing local mean 和：

```text
B_t(w) = mean(J_[t-w:t]) - mean(J_[t:t+w])
```

候选来自：

1. high visual-dependence window；
2. visual-dependence drop；
3. high-before / sustained-low-after；
4. 20% / 50% / 80% fixed-position controls。

随后做 NMS，并在可用时吸附到最近的句号/换行边界。输出 `[start,end)` transition
zone 和 anchor，不声明精确的“视觉边界 token”。每条 trajectory 默认保留 3–6 个候选。

## 5. Outcome probes

### 5.1 Causal visual dependence

对同一个 exact student prefix `p_t`：

```text
q_full(t), q_degraded(t), q_null(t)
```

每个 condition 默认 `K=8` continuation，并由现有 Geometry3K verifier 打分：

```text
D_V^fine(t) = q_full(t) - q_degraded(t)
D_V^all(t)  = q_full(t) - q_null(t)
```

### 5.2 Local teacher relay

```text
baseline:  S(prefix t) -> S
treatment: S(prefix t) -> T_L -> S, L in {32,64,128}
```

teacher 得到相同图像、问题和 student token prefix，不增加“请纠错” instruction。relay
gain 相对相同位置的 `S -> S` 计算，并保存 Jeffreys posterior lift probability。
若 student prefix + teacher relay segment 已暴露可抽取 final answer，该 relay sample 不交给
student 续写，按 answer-leakage/malformed 记录并保守计入非成功。

### 5.3 Teacher-prefix transport

从 `proposal_feasibility_20260805_merged/retained_proposals.jsonl` 取得 verified-correct、
exact-token teacher path。student candidate 的 relative position 映射到 teacher path，prefix
先通过 answer-free gate，再做：

```text
T_prefix -> S
```

transport 同时比较两个 control：fresh unaided student continuation，以及 same-prompt、
same-token-length、answer-free 的高 student-likelihood wrong prefix。记录两组 Jeffreys
posterior；只有 teacher prefix 同时超过二者时才允许进入
`transportable_low_reachability`，避免把 generic context priming 当成正确状态 transport。

### 5.4 Answer leakage

answer-free teacher prefix 被作为 partial assistant reasoning，随后增加一个新的 answer-only
user turn，要求只输出 `\boxed{}`。这是 text-level leakage estimand，不被冒充为 exact-token
continuation。若 prefix 已包含 final-answer marker 或 verifier 已能抽取答案，transport 与
leakage 都直接 skip 并记录原因。

### 5.5 Teacher path support / reachability barrier

沿 verified teacher token path 计算：

- student token NLL；
- full-vocab student/teacher JS；
- top-k overlap。

对 student NLL 做 window aggregation 和 NMS，输出局部 high-NLL barrier。它是后续
Localized FKL Bridge 的候选上游区间，当前 Phase I 不连接任何 loss。

## 6. 状态分类

首版使用保守三分类：

```text
on_policy_repairable
  relay gain high

transportable_low_reachability
  relay gain low, transport gain high

unresolved_under_current_intervention
  relay gain low, transport gain low
```

其余为 `insufficient_evidence`。第三类的名字刻意不写成“student 无能力”；thinking-pattern
mismatch、teacher-path incompatibility、perception、representation 和 execution 都尚未拆开。

## 7. 实际模型、环境和路径

当前 canonical pair：

```text
student: $DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct
teacher: $DTOPD_MODEL_ROOT/Qwen3-VL-32B-Instruct
```

HPC 根目录：

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy
```

默认 Python 使用已经通过 real-image canary 的 active cu128 环境：

```text
$DTOPD_ROOT/envs/va-opd-native-e003-cu128-r595-v1/bin/python
```

输入默认对应现有已完成实验：

```text
$OUT/support_aware_opd/diag_full_k32_20260804_merged
$OUT/support_aware_opd/k32_cohort_20260804
$OUT/support_aware_opd/proposal_feasibility_20260805_merged
```

run 时 `CUDA_VISIBLE_DEVICES` 暴露两张卡，逻辑 `cuda:0` 放 4B student，逻辑 `cuda:1`
放 32B teacher。不要用同一张卡同时加载两个模型。

## 8. Tokenizer / prompt contract

加载后必须满足 student/teacher tokenizer fingerprint 完全一致。fingerprint 包含 token→ID、
special tokens 和 added vocab。另逐 condition 检查 rendered prompt IDs。

当前 K32/proposal 产物使用 `legacy_answer` prompt，Phase I 默认继续使用它，以保证固定
trajectory 的原始上下文语义一致。不要在同一 run 中切换到 `boxed_only` 或训练侧 prompt。

## 9. 命令

CPU/登录节点只做路径、包版本和 tokenizer preflight：

```bash
MODE=preflight bash scripts/hpc/run_causal_state_probe.sh
```

两卡 smoke：

```bash
CUDA_VISIBLE_DEVICES=3,4 SMOKE=1 \
  bash scripts/hpc/run_causal_state_probe.sh
```

smoke 固定为 1 prompt、各 probe `K=2`、relay `L=32`、continuation cap 256。smoke
必须先确认：

- tokenizer aligned；
- full/degraded/null prompt IDs 与 grid 一致；
- JS finite/non-negative；
- 每个 accepted prefix hash 非空；
- verifier 对每条 continuation 有 verdict；
- summary `complete=true`。

正式单 shard：

```bash
CUDA_VISIBLE_DEVICES=3,4 \
  bash scripts/hpc/run_causal_state_probe.sh
```

多 GPU pair 分片时为每个进程设置不同 `CUDA_VISIBLE_DEVICES`、`SHARD_INDEX`、
`NUM_SHARDS` 和独立 `CAUSAL_PROBE_OUTPUT_DIR`。合并：

```bash
python -m dual_track_opd.support_aware.causal_state_probe summarize \
  --input-dir /path/to/shard0 \
  --input-dir /path/to/shard1 \
  --output-dir /path/to/merged
```

## 10. 输出

```text
run_manifest.json                 # git/config/input/model/runtime provenance
trajectory_results/*.json         # atomic resume unit
causal_state_records.jsonl         # 统一 schema
candidate_windows.csv              # flat analysis table
summary.json                       # aggregate + completion state
trajectory_overview.svg            # JS curves + candidate anchors
```

manifest 记录 repo SHA/dirty、三个输入 hash、resolved config、模型配置 hash、tokenizer hash、
CUDA/Python/Torch/Transformers 版本和输出路径。原始 rollout/proposal、模型、数据、logits 和
checkpoint 都不进入 Git。

## 11. Feature-off 与训练边界

新增逻辑没有接入现有 OPD/VA-OPD actor loss，因此关闭/不运行本 CLI 时 baseline 行为不变。
配置中的昂贵子 probe 独立开关；`run` 至少要求 `visual_js=true`。

Phase I 完成前不实现或启动 Outcome-Calibrated RKL / Localized FKL。进入训练开发至少需要：

1. adaptive candidates 对 high relay/visual dependence 的 recall 优于 fixed/random；
2. relay gain 不等价于 raw JS/KL/entropy；
3. 稳定存在 `transport high / relay low` 的 Case 2；
4. 所有 gain 不是 answer leakage 或 generic-length priming；
5. held-out prompt 上仍成立。

之后才把 calibrated relay predictor 接入 native RKL，并最后实现 barrier-localized FKL。

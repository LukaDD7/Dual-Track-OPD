# VA-OPD 主线根因调查与恢复方案（2026-07-19）

## 结论先行

之前的 VA-OPD 不是卡在一个 bug 上，而是四层问题叠加：目标函数实现不完整、旧 GKD recipe 与当前 verl API 漂移、H200 节点上的 offload/NCCL 拓扑问题，以及 prompt/长度/评测协议失配。继续给旧的 `recipe/gkd` 打补丁，只会把下一处兼容性错误向后推。

本轮采用的主线是：

1. 固定新的、已原生支持 VLM teacher force-scoring 的 verl commit `e003163181731412595257a72ec173071efb125f`；
2. 仅用 3 个上游文件、约 60 行 transport/registration patch 增加第二次 degraded-image teacher pass；
3. 将 VA 公式、分组、校验和 loss adapter 全部留在 `src/dual_track_opd/va_opd/`；
4. 主目标保持论文的 sampled-token reverse KL，JSD 只保留为显式 ablation；
5. actor 固定为 4-rank FSDP2，32B teacher 使用另外 2 张卡、TP=2。可见 6 张卡不等于 6-rank FSDP；
6. 禁用 parameter/optimizer CPU offload，K 固定为 4，response 上限恢复为 2048；
7. CPU 实例构建共享环境和 CUDA 12.8 编译工具链，GPU 实例只做 preflight、smoke、pilot、full run。

该方案优先解决“科学目标正确”和“失败可定位”，而不是追求复用最多旧代码。

## 调查范围与 Git 状态

本轮读取了当前代码、全部 VA/GKD 相关历史提交、事故文档、旧 launcher、verl 子模块历史、上游最新 native OPD 实现，以及其它 on-policy distillation 主仓库。

开始工作时安全拉取了 `origin/codex/va-opd`：

- 本地从 `4778d23` fast-forward 到 `621c73c374f19101f90556f6e99954622b67ec72`；
- 新提交为 `docs: Vision-OPD baseline evaluation — full 15-benchmark results`；
- pull 前的 4 个用户修改先 stash、pull 后重放，仍保持 unstaged；
- 防护 stash 保留为 `stash@{0}: codex-preserve-local-vision-opd-before-621c73c`；
- `third_party/Vision-OPD` 初始化在 gitlink `c2e345fcab10c806ba83e2ec6e1e246d73e7aba2`；
- 现有 `third_party/verl` gitlink 仍为 v0.7.1 commit `bec9ef74768dd201881cd4e54cd0385e87caae27`，没有改写；
- 新 native backend 被设计为 Git 外的独立 checkout，不改变第三方子模块。

Vision-OPD baseline 推送因此已经纳入当前工作树，但不作为 VA-OPD 的训练 backend。

## 我们真正要复现的目标

目标论文为 [Visual-Advantage On-Policy Distillation for Vision-Language Models](https://arxiv.org/abs/2605.21924)。从论文正文和公式核对出的关键协议如下：

- divergence 是 `KL(p_student || p_teacher)`，即 reverse KL；
- student 每个 prompt 在线采样 `K=4` 条 response；
- teacher 对同一条 student response、同一 token ID 序列做 force-scoring；
- full condition 使用原图；
- degraded condition 先把空间分辨率降到原来的 10%（bilinear），再 nearest-neighbor 放回原尺寸；
- token VA 为 `max(log pT(y_t | full) - log pT(y_t | degraded), 0)`；
- rollout VA 是有效 response token 的平均值；
- 同一 prompt 的 K 条 rollout 内先 z-score，再以 `tau=1` softmax；
- 每条 response 内 VA 最高的 20% token 为 high group；
- high/low 两组分别取均值，再以 `lambda=0.5` 混合；
- 论文设置为 batch 16、5 epochs。

论文图示给出的 rollout VA `[.045, .03, .02, .01]` 对应权重约 `[.656, .206, .095, .044]`。只有 population standard deviation（`unbiased=False`）能复现这个例子；PyTorch 默认 sample standard deviation 不能。这是旧实现中一个不易察觉但会改变实验的细节。

## 历史根因矩阵

| 层级 | 观察到的现象 | 根因 | 本轮处理 |
|---|---|---|---|
| 信号传输 | `VA=0` | `sampled_log_probs` 在 3 个 `TeacherTopK` 重建函数中被丢弃，后续用固定低值填充 | 新路径直接使用 native teacher sampled-token logprob tensor，并做 exact response-ID equality gate |
| loss 组合 | `fc_opd_loss > 0` 但 `grad_norm=0` | pure distillation 分支在 actor patch 重写时被错误清零 | 使用 verl native distillation PG 组合；finalizer 强制要求 finite loss、finite grad、completed step |
| rollout 分组 | rollout 权重没有实际作用 | launcher 默认 `rollout.n=1` | OPD baseline 与 VA-OPD 都固定 `K=4`；preflight 拒绝其它 K |
| rollout 分组 | sibling 被当成不同 prompt | group ID fallback 到 row index，未读取 `extra_info.sample_uid` | 旧 hook 已补 group-ID 优先级；native 路径使用 verl 在 repeat 前生成并保留的 `uid` |
| teacher 对齐 | 同组 sibling 可能拿到第一条 rollout 的 teacher 分数 | scorer result 用相同 group UID 查找，而不是按对象/rollout 身份 | 旧 hook 改为对象 identity；native teacher 每个 agent-loop output 独立 force-score |
| 公式 | rollout softmax 与论文图示不一致 | `torch.std()` 默认 sample std | 新 objective 显式 `unbiased=False`，并有论文数值回归测试 |
| 旧 GKD recipe | B6–B24 连续兼容错误 | recipe pin、verl API、Ray async、vLLM sync rollout、TensorDict 版本相互漂移 | 停止把旧 recipe 当主线，迁移到 upstream native OPD |
| 旧 GKD recipe | `No current event loop`、sync weights 卡死 | 同步 Ray worker 调 async server adapter；对应 sync rollout 已被上游移除 | native async VLM teacher manager，不恢复已删除的 sync rollout |
| 旧 GKD recipe | locked TensorDict 写入失败、200-step 在 32 step clean exit | TensorDict 行为变化、epoch/step 终止语义漂移 | native current TensorDict path；显式 `total_training_steps`，finalizer 比对 requested/completed |
| CPU RAM | 内存约增长 90 GB，随后 swap/NCCL timeout | param offload + optimizer offload + 惰性 GC 导致 CPU 搬运和残留 | 两种 offload 都固定为 false；H200 足够容纳 4B actor |
| 分布式 | 3/5/6 actor rank 概率性 allgather hang，旧栈后期连 4-rank Trees path 也出现概率问题 | NCCL 2.27.3 + CUDA 12.8 的 collective 竞态；非 2 的幂 topology 最易触发，Ring 等缓解只延后 | actor 只使用证据上风险最低的 4-rank FSDP2；另 2 张卡属于 teacher pool；同时升级到锁定的新 runtime，并把历史 gather size 做成启动 gate，不宣称无条件根治 |
| 分布式 | FSDP1 特定 flat buffer allgather hang | flat-parameter 大 collective 和通信时序 | 显式 `actor.strategy=fsdp2` |
| prompt | 学到错误输出契约、验证 0 分 | legacy `<answer>`/人工 think prompt 与 Qwen3-VL native `\boxed{}` 行为不一致 | `FCOPDDataset` 统一到 native image+question prompt；不注入 literal `<think>` |
| 长度 | 约 67% response 在答案前被截断 | `max_response_length=1024` | 设为 2048；actor token budget 10240；记录 clip ratio |
| 优化 | reverse-KL 长跑模式坍缩 | reverse KL mode-seeking，加上错误 prompt、过长训练和无早停 | main 仍保持论文 RKL；采用 3-step smoke、50-step pilot、频繁 eval/ckpt 和早期 checkpoint 选择；JSD 仅作为命名 ablation |
| baseline | forward GKD 后期 reward 归零 | forward KL baseline 在错误 prompt 下快速退化；改正 prompt 后只有早期增益 | 不把 forward GKD 当 VA 主实现；公平 baseline 使用同一 native sampled-RKL、K=4，只关闭 VA weighting |

## 为什么旧 `recipe/gkd` 不再适合作为主线

仓库里的 `scripts/hpc/run_gkd_geometry3k_qwen3vl.sh` 名字容易让人误解。它实际使用主 PPO/FSDP trainer、项目 HTTP Transformers teacher 和两个本地 patch，并不是原始 `recipe/gkd` 的稳定复用。

旧 community GKD recipe 的问题不是“少改一个 import”：

1. 它最初的 teacher 协议是 text/token-ID oriented，没有将 full/degraded 两个图像条件作为同一 response 的两次严格对齐 scoring；
2. 它依赖已经被上游移除或重构的 sync rollout/weight sync；
3. B6–B24 记录显示每修一层就进入下一层 API drift；
4. 继续维护会迫使研究逻辑进入第三方 backend，违反本仓库 separation-of-concerns 约束；
5. 最新 verl 已经原生提供 Qwen3-VL async teacher、`image_data` 和 exact sequence force-scoring，重复造 teacher protocol 的收益为负。

旧 launcher 和 patch 仍保留用于历史复现/ablation，但文档必须将其标为 legacy。

## 新架构及目标函数映射

### 数据与两次 teacher scoring

数据准备阶段生成：

- `full_image.path`：原图；
- `degraded_image.path`：10% bilinear downsample 后 nearest upsample、与原图同尺寸；
- `sample_uid`：数据样本 ID；
- `extra_info.condition_inputs`：给 agent loop 的可靠 transport。

student 只看 full image 并采样 response。native teacher manager 随后对完全相同的 `prompt_ids + response_ids` 执行两次：

1. full-image pass：它的 sampled-token logprob 是 reverse-KL target；
2. degraded-image pass：只用于计算 VA，不参与 KL target。

两次 teacher 返回的 response IDs 必须逐 token 等于 student response；否则立即退出。degraded tensor 在 VA 权重计算后被删除，不进入 actor microbatch，避免无意义内存复制。

### reverse-KL estimator

verl 的 `k1` 返回 sampled token 上的：

```text
log p_student(y_t) - log p_teacher(y_t)
```

当 `use_policy_gradient=true` 时，verl 把该值 detach 后作为负 reward，再用当前 student logprob 做 policy-gradient。对 student 自身的 on-policy sample 求期望时，这是 reverse-KL 梯度 estimator；它避免了直接把同一个 `k1` expression 当普通可微 loss 时一阶项抵消的问题。

VA adapter 注册 `va_opd_k1`，先把每 token 的 k1 乘以固定 VA multiplier，再进入同一个 native PG path。

### 为什么 VA 使用 `seq-mean-token-sum`

对于每个 prompt 的 K 条 rollout，项目代码令每条 rollout 的 token multiplier 总和为 `K * rollout_weight`。verl 再对展开后的 `G*K` 条 sequence 做 sequence mean：

```text
1/(G*K) * sum_g sum_k [K * w_gk * grouped_token_loss_gk]
= 1/G * sum_g sum_k [w_gk * grouped_token_loss_gk]
```

这正好是先在每个 prompt 内按 rollout weight 加权、再跨 prompt 平均。普通 OPD baseline 不需要该 multiplier，因此使用 `seq-mean-token-mean`。

### 一 token 退化情况

论文 high/low 两组公式在只有一个有效 token 时 low group 为空。实现不会静默丢掉一半 loss mass，而是把该 rollout 的全部质量放到唯一 high token，并记录 `va_opd/degenerate_sequences`。正常数学 response 不应频繁触发；若触发，数据/生成已异常。

## 版本选择

### backend

- verl native OPD commit：`e003163181731412595257a72ec173071efb125f`；
- patch：`patches/verl/va_opd_native_e0031631.patch`；
- patch scope：仅 `agent_loop.py`、`losses.py`、`ray_trainer.py`；
- setup 会执行正向或反向 `git apply --check`；
- preflight 还校验 3 个 patched file 的 SHA-256，拒绝同文件内的隐式手改。

### runtime

不能把 Vision-OPD baseline 机器上的 `pip freeze` 当成可重建的依赖锁。该快照同时出现 vLLM 0.18.0 和 Transformers 5.5.0，但 vLLM 的发布 metadata 明确要求 Transformers `<5`；它只能说明那台机器经过增量/`--no-deps` 安装后曾运行 eval，不能证明 clean resolver 能建立相同环境。

native VA-OPD 改为服从 exact verl backend 自己的版本边界。该 commit 声明 `vllm>=0.8.5,<=0.12.0`，因此选最新允许版本 0.12.0；vLLM 0.12.0 又精确要求 torch 2.9.0、torchvision 0.24.0、torchaudio 2.9.0、FlashInfer 0.5.3 和 Transformers `>=4.56,<5`。Transformers 固定为 4.57.3，与项目 teacher extra 和 Qwen3-VL 支持一致。NumPy 仍不照抄 Vision-OPD 的 1.26.4，因为 exact verl commit 要求 `numpy>=2.0.0`：

- Python 3.12；
- torch 2.9.0 + CUDA 12.8；
- torchvision 0.24.0；
- torchaudio 2.9.0；
- vLLM upstream 0.12.0（commit `4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e`，在 CPU 实例从源码构建，安装版本记录为 `0.12.0+cu128`）；
- transformers 4.57.3；
- Ray 2.53.0；
- TensorDict 0.10.0；
- FlashInfer 0.5.3；
- NumPy 2.2.6；
- PyArrow 22.0.0；
- flash-attn 2.8.3（使用独立 conda CUDA 12.8 toolchain 构建）。

目标 driver 570 只能安全覆盖 CUDA 12.8，因此不采用可能含 CUDA 12.9/13 扩展的 vLLM 发布 wheel。setup 使用独立 conda `cuda-toolkit=12.8` 与 GCC/G++ 12，固定 H200 `TORCH_CUDA_ARCH_LIST=9.0`，构建 vLLM 和 flash-attn；禁止借用 `/usr/bin/nvcc`。这是 backend metadata、模型支持和目标 driver 三者交集中的最小可审计组合。

## 公平实验顺序

不能直接只跑 VA full run。顺序应为：

1. native OPD 3-step smoke：验证 student rollout、full teacher force-score、RKL gradient；
2. VA-OPD 3-step smoke：在 1 的基础上验证 degraded pass、ID alignment、K=4 分组与权重；
3. OPD 50-step pilot；
4. VA-OPD 50-step pilot；
5. 比较 reward、entropy、response clip ratio、VA positive ratio、loss 和 grad；
6. 只有两个 pilot 都稳定，才各跑 5 epochs；
7. 用相同 checkpoint cadence 和 validation set 选择 early checkpoint，而不是默认最后一步；
8. 最终报告同时列 baseline 与 VA 的最佳验证 checkpoint、最后 checkpoint，以及是否出现 collapse。

OPD baseline 也保持 `K=4`、同 batch、同 sampling、同 teacher、同 response length。它只关闭第二次 degraded pass 和 VA multiplier，从而隔离 VA 的贡献。

## 运行必须满足的 gates

### 启动前

- backend commit、patch scope、patched file hashes 完全匹配；
- 环境版本完全匹配；
- `nvcc` 不来自 `/usr`；
- student/teacher 都是 Qwen3-VL，vocab size 相同；
- train/val parquet hash 被记录；
- 每个被审计样本的 full/degraded 文件存在且像素尺寸相同；
- degradation metadata 只能是 `lowres_10pct_nearest`；
- `rollout.n == 4`；
- prompt batch 能被 4 actor GPUs 整除；
- visible GPU 数严格等于 actor pool + teacher pool。

### 运行时

- full/degraded teacher IDs 与 response IDs 完全相等；
- teacher logprobs finite；
- 每组正好 4 个 sibling；
- 每个 prompt 的 rollout weights sum-to-one error ≤ `1e-5`；
- distillation loss、grad norm、entropy finite；
- requested steps 全部完成；
- 没有 GPU compute PID 冲突（除非人工检查后显式 override）。

### pilot/full 决策

- `response_length/clip_ratio` 若持续接近 1，先停止，不要把最大长度 nonsense 当正常训练；
- entropy 快速趋近 0 或异常升高并伴随 reward 下降，判为 collapse；
- VA mean 全为 0 时停止，检查 degraded image 和 teacher transport；
- VA positive ratio 长期接近 0 或 1 都需要抽样检查；
- finite 不是充分条件：必须看 validation reward/accuracy 的 trajectory；
- 50-step pilot 最佳 checkpoint 优于/不显著差于 step 0 才进入 full run。

## 尚未在本机完成的部分

本机没有目标 H200 和共享模型权重，因此无法声称 GPU 训练已经通过。已完成的是代码、静态验证、CPU 可执行 preflight 逻辑、environment/backend setup、launcher、结果 finalizer 和操作手册。目标集群仍必须依次执行 GPU smoke/pilot gates。

最新 runtime 是否完全消除旧 NCCL 竞态也只能由目标节点证明。架构已经避免已知最危险的 6-rank actor topology，但不能把“4-rank + 新版本”写成无条件保证。

## 主要代码入口

- `src/dual_track_opd/va_opd/objective.py`：论文 VA/rollout/high-low weighting；
- `src/dual_track_opd/va_opd/native_verl.py`：degraded VLM transport、batch gate、native loss registration；
- `patches/verl/va_opd_native_e0031631.patch`：最小 backend overlay；
- `configs/experiment/qwen3vl_geometry3k_va_opd_native.yaml`：实验语义参考；
- `configs/environment/verl_va_opd_e003_cu128.constraints.txt`：版本锁；
- `scripts/setup/prepare_va_opd_native_verl.sh`：固定 backend；
- `scripts/hpc/setup_va_opd_native_env.sh`：CPU 实例环境构建；
- `scripts/hpc/preflight_va_opd_native.py`：fail-fast + manifest；
- `scripts/hpc/collect_va_opd_gpu_facts.sh`：CPU CC 无法直接查看 GPU 时的只读事实采集；
- `docs/va_opd_server_readiness_template.md`：CPU CC 根据 GPU 回传维护的 Git-safe readiness 摘要；
- `scripts/hpc/run_va_opd_native.sh`：GPU pipeline；
- `scripts/hpc/finalize_va_opd_native_run.py`：terminal gate + result JSON；
- `docs/va_opd_hpc_runbook.md`：CPU/GPU 逐步手册。

## 外部一手资料

- [VA-OPD paper](https://arxiv.org/abs/2605.21924)
- [verl native OPD documentation](https://verl.readthedocs.io/en/latest/algo/opd.html)
- [verl repository and native Qwen3-VL example](https://github.com/verl-project/verl)
- [verl-recipe](https://github.com/verl-project/verl-recipe)
- [OPSD On-Policy Distillation](https://github.com/HJSang/OPSD_OnPolicyDistillation)
- [THUNLP OPD](https://github.com/thunlp/OPD)
- [Vision-OPD](https://github.com/VisionOPD/Vision-OPD)

其它仓库带来的共识是：student rollout token 必须原样 force-score、prompt/template 必须对齐、teacher/student vocabulary contract 必须明确、长跑前必须用短 pilot 选择 checkpoint。VA-OPD 在此基础上额外要求同一 response 的 full/degraded 图像对齐和 K-sibling weighting。

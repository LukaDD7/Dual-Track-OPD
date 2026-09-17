# Track B(Qwen3.5)固定离线评测结果 — 2026-08-06

## 0. 一句话结论

**训练增益真实但很小,行为变化很大。** base → step60/120 的 prompt-paired
accuracy 提升约 **+2.0–2.4pp**(200 题 × N=8,10k bootstrap CI 下界全部 >0),
三个 seed 一致;真正的大变化是行为轮廓:boxed 率 16%→30–40%、clip 率
94%→65–75%、EOS 率 6%→25–35%、平均长度 −500~−760 token。60→120 步
accuracy 基本平台,无持续增长。

## 1. 方法与条件

| 项 | 值 |
|---|---|
| 评测集 | Geometry3K val text-only,200 prompts(boxed_only prompt contract) |
| 采样 | N=8/prompt,固定 seed grid 0–7,temp 1.0,top-p 0.95,cap 4096 |
| 条件 | base(Qwen3.5-4B)+ 3 seed(r1/seed2/seed3)× step60/step120,共 7 条件 |
| 评分 | `extract_final_answer_candidate` 保守抽取 vs gold(与训练 reward 同一抽取器) |
| 统计 | 每题 8 条 → 每题 mean acc / pass@8;base 与各条件 prompt-paired bootstrap(10k,seed 42) |
| 合并 | verl FSDP checkpoint → HF(`verl.model_merger`),CPU 合并,GPU 0 评测 |
| 产物 | `$OUT/qwen35_fixed_eval_20260806/{eval_dumps,summary/summary.json}` |

## 2. 结果(200 prompts)

| 条件 | acc | pass@8 | boxed | clip | EOS | mean_len |
|---|---:|---:|---:|---:|---:|---:|
| base | 2.5% | 4.5% | 16.4% | 93.6% | 6.4% | 3970 |
| seedr1_step60 | 4.7% | 11.0% | 29.7% | 73.7% | 26.3% | 3489 |
| seedr1_step120 | 4.5% | 9.0% | 38.6% | 65.3% | 34.7% | 3242 |
| seedseed2_step60 | 4.9% | 9.5% | 38.8% | 65.4% | 34.6% | 3206 |
| seedseed2_step120 | 4.8% | 11.0% | 39.6% | 65.7% | 34.3% | 3240 |
| seedseed3_step60 | 4.4% | 9.0% | 30.5% | 74.6% | 25.4% | 3510 |
| seedseed3_step120 | 4.9% | 11.5% | 34.6% | 69.4% | 30.6% | 3354 |

### acc paired delta(base → 训练后,bootstrap CI)

| 条件 | Δ acc | CI95 |
|---|---:|---:|
| seedr1_step60 | +2.19pp | 1.13 – 3.38 |
| seedr1_step120 | +2.00pp | 0.88 – 3.31 |
| seedseed2_step60 | +2.38pp | 1.13 – 3.88 |
| seedseed2_step120 | +2.31pp | 1.19 – 3.63 |
| seedseed3_step60 | +1.94pp | 0.88 – 3.25 |
| seedseed3_step120 | +2.44pp | 1.19 – 3.94 |

### 跨 seed 汇总

| step | Δ acc 均值(范围) | Δ pass@8 均值 |
|---|---:|---:|
| step60 | +2.2pp(1.94–2.38) | +5.3pp |
| step120 | +2.3pp(2.00–2.44) | +6.0pp |

## 3. 解读

1. **增益真实但小**:所有 6 个条件的 paired CI 下界都 >0(0.88–1.19pp),
   三个 seed 方向一致;但绝对 acc 仍只有 4.4–4.9%,且 60→120 步混合方向,
   提示当前 boxed_only + acc-reward + lr=1e-6 的配置在 120 步内已到平台。
2. **行为轮廓变化显著**:模型明显学到"写 boxed 答案并停止"(boxed ×2、EOS ×4–5、
   clip 94%→65–75%、长度 −500~−760),这与训练目标一致;准确率只小幅跟涨,
   说明长度/格式行为先收敛,解题能力提升滞后。
3. 与 handoff §6.5.1 对齐:这是固定离线评测的第一步,提供了下一步决策依据
   (step120 相对 step60 无额外增益 → 更长 run 前先考虑 rollout_n 或权重变量,
   不要只加步数);RNG contract、resume canary、clean-node replay 仍未做。

## 4. 复现与已知事项

- 复现命令:`SKIP_MERGE=1 bash scripts/hpc/run_qwen35_fixed_offline_eval.sh`
  (需 qwen35 env + CUDA 工具链 env,见脚本内注释)。
- 本次运行产物沿用旧 tag 命名(`seedr1_step60`、`seedseed2_step60`…);
  launcher 已修正为 `r1_step60`/`seed2_step60` 命名,只影响后续运行。
- 评分口径为本仓库的保守 boxed 抽取器;若后续要与训练 reward 完全一致,
  以 verl reward 函数为准做一次对照,不改变本表相对量级。

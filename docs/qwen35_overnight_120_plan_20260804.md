# Qwen3.5 Track A：今晚 6 小时实例的执行决策

> 输入：`178ac5e` 首个 n=4 smoke、`ecab13c` group-level clip instrumentation。
> 目标：在实例回收前自然结束，留下可判断学习与长度稳定性的中长 run；不追求占满6小时。

## 决策

今晚按一个无人值守序列执行：

1. 5-step instrumentation canary，真实落盘每个训练 rollout；
2. 自动验证每步恰为 24 sequences、6 prompt groups、每组4条；
3. canary 和结构 gate 通过后，从原始学生冷启动独立的120-step run；
4. 每20步 validation、每30步 checkpoint、每步 training rollout dump；
5. 训练自然结束后自动生成 group-level clip/EOS/correct summary。

启动命令：

```bash
nohup bash scripts/hpc/run_qwen35_v1_n4_overnight_sequence.sh \
  > artifacts/fc_opd/nohup_qwen35_n4_overnight120_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

脚本要求 project tracked worktree clean；不清理、不覆盖旧输出。如果已有同名 canary
或 long-run 目录，它会 fail fast，保留原产物供人工判断。

## 为什么是 120 步，不是把6小时吃满

20-step smoke 墙钟约35分钟，其中含 step 0/5/10/15/20 五轮 validation。120-step
将 validation 降到每20步一次，保守按 smoke 的 105秒/步上界约3.5小时，再加两次
preflight、5-step canary、四次 checkpoint 和最终分析，预期仍有明显回收余量。

160–180步虽然可能塞进6小时，但实例启动、checkpoint、validation 和 NFS 抖动会使
尾部 checkpoint/manifest 来不及完成。自然完成并留下完整120步证据，比被回收打断的
更长曲线更有价值。

## 冻结配置

```text
student: Qwen3.5-4B
teacher: qwen3.6-27B online scorer
loss: K1 + policy gradient + task accuracy reward
prompt: boxed_only
thinking: disabled
sampler: temperature=1.0, top_p=.95, top_k=-1
response cap: 4096
batch: 6 prompts x n=4 = 24 sequences
PPO mini-batch: 6 prompts = 24 effective sequences
workers: 8
actor/teacher GPUs: 3 + 1
learning rate: 1e-6
steps: 120
validation: step 0,20,40,60,80,100,120
checkpoint: step 30,60,90,120
```

训练与 validation sampler 均显式覆盖，不依赖 pinned backend 的 `top_p=1.0` 默认值。
tokenizer-ID alignment preflight 每个 phase 都重新运行并写 manifest。

## 明早必须回答的问题

### 训练是否真的在学习

- validation accuracy/boxed 是否相对 step0 有持续改善，而非单点波动；
- task reward 非零 step 比例、any-correct group 比例是否提升；
- OPD loss、task PG、entropy/KL、grad 是否有限且无单向漂移；
- step30/60/90/120 checkpoint 是否完整。

### 长度是否稳定

- 每步 sequence clip；
- group any-clip、all-clip、all-EOS 和 mean clipped/group；
- validation clip/EOS/长度分位轨迹；
- 是否重现20-step smoke 的中段膨胀后恢复，或演变成持续增长。

### go/no-go

- **Go：** run/manifest/checkpoints 完整，validation accuracy 有可信上升或至少不退化，
  最后20步 group all-clip 没有持续上升，loss/grad/entropy 有限。下一步做相同配置的
  第二 seed，而不是立刻改超参。
- **Hold：** accuracy 持平但长度稳定。先比较 checkpoint 0/60/120 的固定离线 eval，
  再决定延长或换 seed。
- **No-go：** clip/长度持续上升、boxed/accuracy持续下降、all-clip group 扩大，或数值/
  系统失败。回到2048或引入单独命名的长度稳定化 arm；不把失败 run 续跑成长实验。

本轮 format reward 仍结构性为0，任务信号实际来自 accuracy；报告时必须拆分，不能把
总 reward 描述成格式学习。

## 两条任务线的 GPU 调度

今晚 Track A 先占 GPU 0–3。Track B 可以在确认无资源/Ray 冲突后，用独立环境和一对
剩余 GPU 做 proposal/prefix intervention smoke；不要在 Track A 运行期间启动其8-GPU
full job。Track A 自然结束后若实例仍有时间，也不自动启动第二个大任务，优先保证
checkpoint、manifest、group summary 和日志完整落盘。

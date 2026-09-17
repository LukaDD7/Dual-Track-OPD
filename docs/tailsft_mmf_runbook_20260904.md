# TailSFT MMF 复现 GPU Runbook（2026-09-04）

复现 arXiv:2608.25756（TailSFT: Filtered Fine-Tuning Improves Post-Training
Performance）于本项目的 MMF-only 单源 SFT 臂。**不动标准 SFT**：所有产物使用
新的 `*_tailsft` 命名，verl 后端 `sft_loss` 原样保留，TailSFT 仅在
`data.tailsft.enabled=True` 时启用。

- 对比基线：`qwen3vl_sft_mmf122k_1ep/global_step_1774`（113,537 行 1 epoch，B 段六项 macro avg **0.3714**）
- 唯一实验差异：TailSFT 在线序列级过滤（γt ramp 0→0.5 @ 800 步），其余超参与
  `run_sft_warmup.sh` 完全一致
- 实现细节与测试见 `patches/verl/README.md` 的 TailSFT 一节 +
  `tests/sft_rl/test_tailsft_filtering.py`（20/20 green，训练 env 已复核）

## GPU 实例命令序列

以下命令全部在 **GPU 实例**上执行（CPU 实例无卡，无法本地验证 GPU 路径）。
`DTOPD=/inspire/hdd/global_user/mengweicheng-240108120092/lzy`。

### 0. 环境（每步都先做）

```bash
export DTOPD=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
source $DTOPD/miniconda3/etc/profile.d/conda.sh
conda activate $DTOPD/envs/va-opd-qwen35-v090-cu132-r595-v1
cd $DTOPD/projects/Dual-Track-OPD
```

annotate 脚本自身不做任何 env 设置（无 conda activate / CUDA_VISIBLE_DEVICES），
所以必须先执行上面四行。`qwen_vl_utils`（图片解码依赖）只装在训练 env
（CPU 测试 env 没装，dataset[0] 会报 ModuleNotFoundError —— 这是预期差异，
不是 bug）。

### 1. Annotate 冒烟（1 卡，~2 分钟，验证 ℓ0 管线）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sft_rl/annotate_tailsft_init_ce.py \
  --pool-dir $DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft \
  --out-dir  $DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft_tailsft_smoke \
  --model    $DTOPD/models/Qwen3-VL-8B-Instruct \
  --max-shards 1 --limit-rows 4
```

注意 `--limit-rows` 会截断写出的 shard（只写前 4 行），所以冒烟必须用
**抛弃型 out-dir**（上面的 `_smoke` 后缀），不要直接写正式目录。

**通过标准**：输出出现 `[tailsft-init-ce] wrote sft_warmup_train__part_0000.parquet:
4 rows, shard mean ℓ0 = <数值>`。ℓ0 应是 O(0.5–3) 的正数（base 模型在 MMF
数据上的平均 CE；合理的数值范围，参考训练日志里 mmf_only SFT 的初期
val loss 量级）。然后验证列存在：

```bash
python -c "import pyarrow.parquet as pq; \
  print(pq.read_schema('$DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft_tailsft_smoke/sft_warmup_train__part_0000.parquet').names)"
# 期望: ['messages', 'images', 'source', 'image_hash', 'init_ce']
```

冒烟通过后删掉冒烟目录：

```bash
rm -rf $DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft_tailsft_smoke
```

### 2. 全量 annotate（1 卡，113,537 行 × 1 forward，可断点续跑）

```bash
nohup env CUDA_VISIBLE_DEVICES=0 python scripts/sft_rl/annotate_tailsft_init_ce.py \
  --pool-dir $DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft \
  --out-dir  $DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft_tailsft \
  --model    $DTOPD/models/Qwen3-VL-8B-Instruct \
  > $DTOPD/fc-opd-storage/logs/annotate_tailsft_init_ce.log 2>&1 &
tail -f $DTOPD/fc-opd-storage/logs/annotate_tailsft_init_ce.log
```

- 76 个 train shard + 2 个 val shard（val 原样拷贝，无 init_ce —— 训练 val
  路径自动退回纯 CE，val 曲线与标准 SFT 同口径）
- 每 50 行打印 running mean ℓ0；每写完一个 shard 打一行 `wrote ...`
- **可恢复**：重跑同命令会跳过已存在的输出 shard（`shard ... already
  annotated, skip`），中断后直接重新 nohup 即可
- 速率参考：H200 上 bf16 单 forward 约 0.3–1 s/行 → 113,537 行 ≈ **10–32
  小时**。若嫌慢可换 `--batch 4`（4 序列/forward，显存允许时），但先确认
  batch=1 冒烟数值正确再换
- 结束标志：`[tailsft-init-ce] DONE. rows annotated=113537 mean ℓ0=<数值>`

跑完后快速核对行数对齐：

```bash
python - <<'PY'
import pyarrow.parquet as pq, glob
src = sorted(glob.glob("$DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft/sft_warmup_train__part_*.parquet"))
dst = sorted(glob.glob("$DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft_tailsft/sft_warmup_train__part_*.parquet"))
assert len(src) == len(dst) == 76, (len(src), len(dst))
for s, d in zip(src, dst):
    assert pq.read_metadata(s).num_rows == pq.read_metadata(d).num_rows, (s, d)
print("row alignment OK: 76 shards, rows match")
PY
```

### 3. 训练（4 卡，脚本自带 env + init_ce 列守卫）

```bash
bash scripts/sft_rl/run_sft_tailsft_mmf.sh
```

- 脚本内部：conda activate、ray stop、CUDA toolchain env、init_ce 列校验
  （缺列直接 FATAL 提示先跑 annotate）、val shard 拷贝检查
- 名字/目录（全部新建，不碰标准 SFT 产物）：
  - experiment `qwen3vl_sft_tailsft_mmf122k_1ep`
  - ckpt `$DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_tailsft_mmf122k_1ep`
  - log `$DTOPD/fc-opd-storage/logs/qwen3vl_sft_tailsft_mmf122k_1ep.log`
- 关键超参：1 epoch（对齐 mmf122k_1ep）、batch 64、lr 5e-5 cosine、
  max_len 12288、γt ramp 0→0.5 @ 800 步（`SFT_RL_TAILSFT_*` 可覆盖）
- 训练日志确认（TailSFT 生效的证据）：
  - 启动横幅 `== tailsft: f=0.5 schedule=ramp ramp=800 ==`
  - metric `tailsft/dropped`（随 ramp 从 0 涨到 ~batch 的一半）、
    `tailsft/batch`、`tailsft/margin_mean`（应为负且随训练变负——策略在
    改善）、`tailsft/retained_tokens`
  - 若这些 metric 缺失而训练在跑 → loss 走了 fallback 路径（init_ce 未
    读到），停下来查
- 总步数应与基线一致（同 batch/epoch 下 ≈ 1774 步量级）
- val/loss 口径与标准 SFT 相同（val 无 init_ce → 纯 CE），可直接对表

### 4. 评测（训练完成后，4 卡）

```bash
EVAL_CKPT=$DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_tailsft_mmf122k_1ep/global_step_1774/huggingface \
EVAL_TAG=tailsft_mmf122k_1ep \
  bash scripts/sft_rl/run_sft_eval_ladder_4gpu.sh
```

- `global_step_*` 以实际目录为准（`ls $DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_tailsft_mmf122k_1ep/`）
- A 段 pass@1/pass@8（geo3k 601 + MMF val 500）+ B 段六项 benchmark
  （judge Qwen3-VL-32B-FP8）；产物与既有四臂完全同名同目录结构
- 先 `SMOKE=1` 冒烟（各 8 行）再全量，与既有流程一致
- 对表基准（mmf_only_1ep，docs/sft_rl_full_results_ptdpo_20260831.md）：
  GQA 0.3759 / DynaMath 0.6253 / ViewSpatial 0.0940 / MMMU-Pro 0.2312 /
  ReMI 0.3662 / MMBench 53.608（/100）→ **Avg 0.3714**（MMBench /100 后
  macro 平均）

## 已知注意点

1. **annotate 无自带 env**：必须先 conda activate 训练 env（qwen_vl_utils
   只在该 env）。CUDA_VISIBLE_DEVICES 也需显式指定。
2. **冒烟 out-dir 必须抛弃型**：`--limit-rows` 截断写出 shard，混入正式
   目录会污染训练数据。
3. **ℓ0 数值范围**：shard mean ℓ0 合理区间 O(0.5–3)。若出现 0.0 或 NaN，
   停下检查（0.0 = loss_mask 全零或 chunked_target_logprobs 路径异常）。
4. **训练 fallback 风险**：日志里看不到 `tailsft/dropped` 等 metric =
   loss 走了 plain SFT fallback（init_ce 未达 loss 层）。训练前可 grep
   日志确认 `data.tailsft.enabled=True` 生效 + TailsFTDataset 被加载。
5. **中断恢复**：annotate 按输出 shard 存在性跳过；训练
   `trainer.resume_mode=auto`。

## 产物清单（全部新建，不覆盖标准 SFT）

| 产物 | 路径 |
|---|---|
| ℓ0-annotated pool | `$DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft_tailsft/` |
| annotate log | `$DTOPD/fc-opd-storage/logs/annotate_tailsft_init_ce.log` |
| train ckpt | `.../sft_rl/ckpt/qwen3vl_sft_tailsft_mmf122k_1ep/` |
| train log | `$DTOPD/fc-opd-storage/logs/qwen3vl_sft_tailsft_mmf122k_1ep.log` |
| eval（A 段） | `.../sft_rl/eval/tailsft_mmf122k_1ep/` |
| eval（B 段） | `$DTOPD/eval_runs/vision_opd_project_baseline/tailsft_mmf122k_1ep_*` |
| backend patch | `patches/verl/tailsft_online_filtering.patch`（+ `patches/verl/README.md` TailSFT 节） |

# manuscript-tailsft-gpu 精简版（2026-09-08）

## 当前状态

- Judge 已启动：GPU 3，`Qwen3-VL-32B-Instruct`，端口 `8801`
- base 评测已启动：GPU 0，正在跑 `MMMU_DEV_VAL`
- tailsft 评测已启动：GPU 1，正在跑 `MMMU_DEV_VAL`
- 你执行的 359/360 行命令**没有带 `--reuse`**，所以这是**新跑**，不是续跑

如果你是想接着之前的结果继续跑，应该停掉这两条评测后改用：

```bash
bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/eval/launch_mmf_eval.sh base 0 --reuse
bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/eval/launch_mmf_eval.sh tailsft 1 --reuse
```

## 命令用途

| 命令 | 作用 |
|---|---|
| `launch_judge.sh` | 启动 Qwen3-VL-32B judge，GPU 3，端口 8801 |
| `launch_mmf_eval.sh base` | 启动 base 模型 14-bench 评测，默认 GPU 0 |
| `launch_mmf_eval.sh tailsft` | 启动 tailSFT 模型 14-bench 评测，默认 GPU 1 |
| `launch_mmf_eval.sh ... --reuse` | 断点续跑，复用已有预测，不从头开始 |
| `tail -f vlmeval_base_mfr.log` | 实时看 base 评测进度 |
| `tail -f vlmeval_tailsft_mfr.log` | 实时看 tailSFT 评测进度 |

## 进度查看

```bash
tail -f /inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/vlmeval_base_mfr.log
tail -f /inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/vlmeval_tailsft_mfr.log
```

结束标志：日志末尾出现 `Run Summary Report`。

## 结果目录

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/eval/vlmeval_runs/
```

里面按模型名和 eval_id 分目录；每个 benchmark 的预测文件是 `*.xlsx`，评分文件是 `*_acc.csv` 或 `*_score.csv`。

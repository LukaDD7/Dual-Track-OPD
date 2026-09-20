# VA-OPD ViRL39K K=4 and Performance Plan

Date: 2026-09-20

## Purpose

The active ViRL39K run is a project-domain scaling run:

```text
Qwen3-VL-32B-Instruct teacher -> Qwen3-VL-8B-Instruct student, K=8
```

The paper training contract instead uses K=4.  These are different experiments:

- **K=4**: paper-reproduction priority and the main comparison arm.
- **K=8**: project-scaling exploration.  It may improve the base model, but its
  rollout grouping and training dynamics are not directly comparable to K=4.

No VA loss, visual-advantage computation, rollout reweighting, degradation,
dataset, scorer, or model identity is changed by the performance plan.

## Paper K=4 entrypoint

The isolated K=4 wrapper fixes the config and rollout count:

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD

bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k_paper_k4.sh \
  --full \
  --run-id qwen3vl_32b_teacher_8b_student_virl39k_va_opd_paper_k4_full_v1
```

Resume with the same run ID:

```bash
bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k_paper_k4.sh \
  --resume \
  --run-id qwen3vl_32b_teacher_8b_student_virl39k_va_opd_paper_k4_full_v1
```

The K=4 config is:

```text
configs/experiment/qwen3vl_32b_8b_virl39k_va_opd_paper_k4.yaml
```

It keeps the paper contract where the arXiv v1 text specifies it: 5 epochs,
batch 16, K=4, ViRL39K, AdamW.  The learning rate remains the documented
operational default `1e-6` because the paper text does not publish that value.

## Performance A/B

Wait until the active K=8 run reaches a healthy step-25 checkpoint and
validation.  Then run one arm at a time; never run two Ray VA-OPD launchers on
one host.

| Arm | Dynamic token limit | Gradient checkpointing |
|---|---:|---:|
| A | 10240 | true |
| B | 16384 | true |
| C | 20480 | true |
| D | 16384 | false |
| E | 20480 | false |
| F | 32768 | false (advanced OOM screen) |

Example 4-step OOM screen:

```bash
bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k_perf_arm.sh \
  --arm B \
  --steps 4 \
  --run-id va_opd_virl39k_perf_b_oom4_v1
```

If stable, run 8 steps with a separate run ID:

```bash
bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k_perf_arm.sh \
  --arm B \
  --steps 8 \
  --run-id va_opd_virl39k_perf_b_timing8_v1
```

Compare, in order:

1. `timing_s/step`;
2. `timing_s/old_log_prob`;
3. `timing_s/update_actor`;
4. `actor/perf/max_memory_allocated_gb`;
5. `actor/perf/max_memory_reserved_gb`;
6. VA, loss, gradient, entropy, and response-length health gates.

Dynamic microbatching and gradient checkpointing do not change the objective
formula, but can change microbatch splitting and floating-point accumulation
order.  A continuation using a chosen arm must record that as a new resolved
configuration.

## Best-validation checkpoint protection

The trainer writes checkpoints before validation and may prune old checkpoints.
The external protector does not modify the trainer.  It watches the shared
training log and hard-links the best-validation checkpoint under `best_val/`:

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD

setsid nohup /inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/va-opd-native-e003-cu128-r595-v1/bin/python \
  scripts/hpc/protect_va_opd_best_checkpoint.py \
  --checkpoint-root /inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/checkpoints/va_opd_native/qwen3vl_32b_teacher_8b_student_virl39k_va_opd_full_until_reclaim_v2 \
  --train-log /inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/runs/va_opd_native/qwen3vl_32b_teacher_8b_student_virl39k_va_opd_full_until_reclaim_v2/train.log \
  --watch \
  --apply \
  > /inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/runs/va_opd_native/qwen3vl_32b_teacher_8b_student_virl39k_va_opd_full_until_reclaim_v2/best_val_protector.log \
  2>&1 &
```

Check the dry-run result first by omitting `--apply`.  The default retention in
the K=4 and performance wrappers is small enough to avoid tens of terabytes of
checkpoint growth, while the protected best remains available for export and
evaluation.

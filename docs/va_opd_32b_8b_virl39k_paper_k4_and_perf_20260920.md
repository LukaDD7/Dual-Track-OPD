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

## Performance results

All performance arms used the same K=8, batch-16, ViRL39K, 32B teacher ->
8B student contract.  They changed only execution parameters.  Raw evidence is
outside Git under `fc-opd-storage/runs/va_opd_native`.

| Arm | Token limit | Checkpointing | 4-step result | Peak actor memory | Notes |
|---|---:|---:|---|---|---|
| B | 16384 | true | PASS, 4/4, exit 0 | 48.93 GB alloc / 58.04 GB reserved | 54m17s total; validation 0.590 -> 0.604 |
| C | 20480 | true | PASS, 4/4, exit 0 | 48.60 GB alloc / 58.66 GB reserved | 48m51s total; validation 0.582 -> 0.586 |
| D | 16384 | false | FAIL, 0/4, exit 1 | not reached | Colocated vLLM wake-up OOM after validation |
| E | 20480 | false | FAIL, 0/4, exit 1 | 134.29 GB PyTorch allocation | Actor forward OOM |

Run records:

```text
B: va_opd_virl39k_perf_b_oom4_v1
C: va_opd_virl39k_perf_c_oom4_v1
D: va_opd_virl39k_perf_d_oom4_v1
E: va_opd_virl39k_perf_e_oom4_v1
```

The C arm was about 10% faster than B over four steps while using effectively
the same peak memory.  Both B and C kept VA grouping, loss, gradients, entropy,
and response-length health gates finite and non-degenerate.  Closing gradient
checkpointing was rejected: D failed in vLLM wake-up and E failed in actor
forward.  F was not run because both no-checkpointing screens failed at lower
token limits.

The selected execution configuration is therefore:

```text
ppo_max_token_len_per_gpu = 20480
enable_gradient_checkpointing = true
max_actor_ckpt_to_keep = 2
```

This is recorded as **perfC**.  It is an execution setting, not an algorithm
change; it can alter dynamic microbatch boundaries and floating-point
accumulation order.

## Paper-K4 full run using perfC

The paper-contract full run uses the K=4 entrypoint with perfC:

```text
run ID: qwen3vl_32b_teacher_8b_student_virl39k_va_opd_paper_k4_full_perfC_v1
run dir: /inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/runs/va_opd_native/qwen3vl_32b_teacher_8b_student_virl39k_va_opd_paper_k4_full_perfC_v1
checkpoint root: /inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/checkpoints/va_opd_native/qwen3vl_32b_teacher_8b_student_virl39k_va_opd_paper_k4_full_perfC_v1
```

Launch:

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD

RUN_ID=qwen3vl_32b_teacher_8b_student_virl39k_va_opd_paper_k4_full_perfC_v1

setsid nohup env \
  VA_OPD_RUN_ID="$RUN_ID" \
  VA_OPD_PPO_MAX_TOKEN_LEN_PER_GPU=20480 \
  VA_OPD_GRADIENT_CHECKPOINTING=true \
  VA_OPD_MAX_ACTOR_CKPT_TO_KEEP=2 \
  bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k_paper_k4.sh \
    --full \
    --run-id "$RUN_ID" \
  > "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/runs/va_opd_native/${RUN_ID}/launcher.log" \
  2>&1 &
```

Resume after GPU-instance reclamation, always with the same run ID:

```bash
setsid nohup env \
  VA_OPD_RUN_ID="$RUN_ID" \
  VA_OPD_PPO_MAX_TOKEN_LEN_PER_GPU=20480 \
  VA_OPD_GRADIENT_CHECKPOINTING=true \
  VA_OPD_MAX_ACTOR_CKPT_TO_KEEP=2 \
  bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k_paper_k4.sh \
    --full \
    --resume \
    --run-id "$RUN_ID" \
  >> "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/runs/va_opd_native/${RUN_ID}/launcher.log" \
  2>&1 &
```

Historical status on 2026-09-21:

- First launch completed through step 51; the GPU instance was then reclaimed.
- The latest complete checkpoint was `global_step_50`.
- Resume mode found `global_step_50` and loaded model, optimizer, RNG, and LR
  scheduler state on all four actor ranks.
- Post-resume validation at step 50 was 0.610.
- Best protected validation checkpoint was 0.624 at step 25.
- The resumed run completed step 51 and continued training normally.

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

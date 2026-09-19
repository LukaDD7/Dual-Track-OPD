# VA-OPD 32B→8B Scale Runbook

Date: 2026-09-17

## Purpose

This experiment scales the native VA-OPD reproduction from the paper's 8B teacher
and 2B student to a 32B teacher and 8B student. It is a controlled method-scaling
run, not the final project-domain training run.

The experiment keeps Geometry3K fixed so the model pair is the main changed
variable. The later MMFineReason/Cauldron project-domain runs should build on
the same algorithm, gates, and checkpoint policy, but should not mix their
results with this controlled scaling run.

## Canonical configuration

Configuration file:

```text
configs/experiment/qwen3vl_32b_8b_geometry3k_va_opd_paper.yaml
```

Models:

| Role | Local path | Constraint |
|---|---|---|
| Student | `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct` | Native training model |
| Teacher | `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct` | Native teacher, TP=2 |

The teacher and student tokenizers must remain identical. The current local
`tokenizer.json` SHA-256 is:

```text
a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7
```

This identity is required because VA-OPD compares teacher and student token
log-probabilities on the exact student response IDs.

## Data strategy

This scale run uses the existing controlled Geometry3K VA-OPD preparation:

```text
Train parquet:
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/geometry3k_va_paper/train.parquet

Validation parquet:
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/geometry3k_va_paper/val_monitor.parquet
```

Policy:

- `--profile paper` uses the 2,101-row `geometry3k_va_paper/train.parquet`
  preparation.
- `--profile smoke` uses the script's smaller smoke preparation:
  1,901-row `geometry3k_gkd/train.parquet`. `--config-reference` is a
  preflight/manifest reference and does not override the training parquet path.
- 2,101 training rows, 5 epochs, prompt batch size 16; expected 655 optimizer steps.
- 200 validation rows are used only as an overlapping monitoring/validation set.
- Each visual example has a full image and a degraded `lowres_10pct_nearest` image.
- Degraded and full images must have identical dimensions to preserve Qwen-VL
  visual-token alignment.
- `sample_uid` and dataset hashes remain part of the run manifest.
- Geometry3K is a method-validation dataset. It is not the intended final project
  domain for MMFineReason/Cauldron work.

For the first project-domain pilot, use a reasoning/grounding-balanced mixture
with MMFineReason weighted around 40% and Cauldron around 60% for balanced
transfer. A reasoning-primary alternative is approximately 60% MMFineReason and
40% Cauldron, but that choice is a separate experiment and must not be folded
into this scaling result.

## Algorithm and training configuration

Native VA-OPD settings:

- Rollouts per prompt: 4.
- Rollout temperature: 1.0, `top_p=0.99`.
- Visual Advantage rectification: positive.
- Group std: population standard deviation.
- Top token fraction: 0.20.
- High group weight: 0.50.
Full-image teacher log-probabilities are the KL target. Degraded-image
teacher log-probabilities are used in the visual-advantage calculation.

Distillation:

- Backend: native verl OPD with backend commit `e003163181731412595257a72ec173071efb125f`.
- Loss: `va_opd_k1`.
- Divergence: reverse KL with policy-gradient form.
- Loss aggregation: `seq-mean-token-sum`.
- Task rewards disabled in the distillation objective.
- Loss clamp: 10.0.

Training:

- Learning rate: `1e-6`. This is an operational default, not a value specified
  by the current arXiv text.
- Optimizer: AdamW.
- Epochs: 5.
- `ppo_epochs=1`.
- `train_batch_size=16`.
- Max prompt length: 6,144.
- Max response length: 2,048.
- FSDP2 actor, gradient checkpointing enabled.
- Actor and optimizer CPU offload disabled.

Hardware layout:

- Minimum 6 visible H200-class GPUs.
- Actor: 4 GPUs.
- Teacher: 2 GPUs with tensor parallel size 2.
- CUDA 12.8 runtime; the shared CUDA toolchain is used rather than system nvcc.

## Required gates

The run is valid only when all of the following hold:

- Every rollout group has exactly 4 siblings.
- Teacher and student response IDs match exactly.
- Full/degraded image dimensions match exactly.
- Rollout group weights sum to 1 within `1e-5`.
- Teacher log-probabilities and gradients are finite.
- At least one update completes.
- VA weights are not identically zero.

Checkpoint selection must use the validation curve, not the final step by default.

## Smoke command

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD

bash scripts/hpc/run_va_opd_native.sh \
  --objective va_opd \
  --profile smoke \
  --visible-gpus 0,1,2,3,4,5 \
  --actor-gpus 4 \
  --teacher-gpus 2 \
  --teacher-tp 2 \
  --student-model /inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct \
  --teacher-model /inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct \
  --config-reference /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/configs/experiment/qwen3vl_32b_8b_geometry3k_va_opd_paper.yaml \
  --steps 4 \
  --batch-size 4 \
  --run-id qwen3vl_32b_8b_geometry3k_va_opd_smoke_v1
```

Current smoke run:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/runs/va_opd_native/qwen3vl_32b_8b_geometry3k_va_opd_smoke_v1
```

As of 2026-09-17 01:58 UTC, the run was active. Teacher and student engines had
loaded, initial validation completed with score 0.395, and no fatal traceback,
OOM, or SIGKILL had appeared. The final four-step smoke result must be read from
`result.json` after the run exits.

Final smoke result, 2026-09-17 02:00 UTC:

```text
passed: true
completed_steps: 4/4
exit_code: 0
finite_distillation_loss_count: 8
finite_gradient_count: 4
finite_entropy_count: 4
last_distillation_loss: 0.19950655847787857
last_gradient_norm: 9.730185508728027
last_va_mean: 0.10620339214801788
max_group_weight_sum_error: 1.1920928955078125e-07
initial/final validation score: 0.395
```

The run used the smoke profile's 1,901-row `geometry3k_gkd` training parquet and
the 200-row validation parquet. This is intentional for smoke but is not the
full paper-profile data preparation.

## Full paper-scaling command

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD

export VA_OPD_MAX_ACTOR_CKPT_TO_KEEP=null

bash scripts/hpc/run_va_opd_native.sh \
  --objective va_opd \
  --profile paper \
  --visible-gpus 0,1,2,3,4,5 \
  --actor-gpus 4 \
  --teacher-gpus 2 \
  --teacher-tp 2 \
  --student-model /inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct \
  --teacher-model /inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct \
  --config-reference /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/configs/experiment/qwen3vl_32b_8b_geometry3k_va_opd_paper.yaml \
  --run-id qwen3vl_32b_8b_geometry3k_va_opd_paper_full5e_v1 \
  --save-freq 25 \
  --test-freq 25
```

Keep `VA_OPD_MAX_ACTOR_CKPT_TO_KEEP=null` for the full run. Save and validate
every 25 steps so checkpoint selection can use the entire validation curve.

## Implementation map

- Entry point: `scripts/hpc/run_va_opd_native.sh`.
- Experiment config: `configs/experiment/qwen3vl_32b_8b_geometry3k_va_opd_paper.yaml`.
- VA-OPD native research hooks:
  `src/dual_track_opd/va_opd/native_verl.py`.
- Full/degraded teacher inference and response-ID alignment hooks are in the
  pinned verl backend patch:
  `patches/verl/va_opd_native_e0031631.patch`.
- Run manifests, raw logs, checkpoints, and model weights remain outside Git.

For the ViRL39K project-domain scaling run, use
`docs/va_opd_32b_8b_virl39k_runbook_20260919.md` and
`scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k.sh`; do not mix its
results with this controlled Geometry3K run.
